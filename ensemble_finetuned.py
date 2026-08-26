'''
Ensemble LoRA fine-tuned models, the way the DeepPeptide paper's headline is produced.

The fine-tuned counterpart to ensemble_predict.py, which handles the frozen
cached-embedding path. Same averaging protocol, so the two ensemble gains are
directly comparable if the frozen members are ever retrained.

WHY THIS IS THE RIGHT ARM TO ENSEMBLE

Ensemble gain scales with member diversity at fixed member quality. Measured on
this task:

    frozen members   sd 0.0186  (n=15)
    LoRA members     sd 0.0595  (n=8)     3.2x more diverse

The instability that made any single fine-tuned run unquotable -- two of eight
replicates landed BELOW the frozen baseline -- is the property that should make
an ensemble of them gain more than an ensemble of frozen models. That is the
standard variance/diversity decomposition, and it is why upstream's headline is
a 20-model ensemble rather than a best single model.

WHAT IS AVERAGED

Identical to ensemble_predict.py:

  * the 2-logit emissions from head.features_to_emissions, combined under
    --combine:

      product (default)  mean of the log-softmax. A product of experts, the
                         standard combination for log-linear models like a CRF.
                         Averaging raw logits instead would let one
                         over-confident member dominate.
      mixture            logsumexp over members minus log n, i.e. the mean of
                         the PROBABILITIES. A mixture of experts.

    The two differ in who holds the veto. Under a product every member must
    agree before a peptide state survives, so one confidently wrong member
    suppresses a call the others make; under a mixture one confident member
    carries it. That is a precision/recall dial, and the first run of this
    script showed it pointing the wrong way for this task: the 4-member product
    scored P 0.7073 / R 0.5190 against members at P 0.632-0.655 / R 0.540-0.586,
    so precision came in ABOVE every member and recall BELOW every member. F1
    here is recall-limited, so that trade loses -- the ensemble landed at 0.5987,
    under its own best member's 0.6080.

    Which rule to use is an open question per task, so pick it on VALIDATION
    (--partition 3) and report the winner on test. Choosing it by test F1 is
    selection on the test set and is not quotable.
  * the CRF's transitions / start_transitions / end_transitions, averaged
    arithmetically under BOTH rules. Low-dimensional, structurally identical
    across members (same allowed-transition mask), and log-potentials, so an
    arithmetic mean is a product of experts. --combine names a rule for the
    emissions only; see the note at the averaging site for why a mixture over
    transitions is not a thing this script can compute.

The encoder is NOT averaged. Each member runs its own forward pass with its own
adapters, and only the outputs are combined. Averaging LoRA weights across
independently trained members would be a model soup, which is a different thing.

Decoding happens ONCE, on the averaged emissions with the averaged transitions,
so the CRF state grammar is enforced exactly as for a single model.

THE COMPARISON THAT MAKES A WIN COUNT

Teufel's published propeptide figure is an ENSEMBLE. A single ESM-2 run is not
the right comparator for an ESM3 ensemble, and a reviewer will say so. Before
quoting any ensemble result against ESM-2, ensemble ESM-2 too -- its embeddings
are cached on the HPC, so replicates there are cheap.
'''
import argparse
import glob
import json
import math
import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models import LSTMCNNCRF
from src.models.plm_backbone import PLMBackbone
from src.utils.finetune_glue import SequenceCRFDataset, FineTunedCRF
from src.utils.manuscript_metrics import compute_all_metrics

# Members must agree on these or the averaged emissions do not correspond to the
# same function. lora_lr and seed are deliberately NOT here: differing on them is
# the point, since that is where the diversity comes from.
MUST_MATCH = ('model_name', 'max_len', 'lora_blocks', 'lora_r', 'lora_alpha',
              'num_filters', 'hidden_size', 'kernel_size', 'data_file',
              'partitioning_file', 'no_input_norm')


def load_configs(run_dirs):
    cfgs = []
    for d in run_dirs:
        c, m = os.path.join(d, 'config.json'), os.path.join(d, 'model.pt')
        if not (os.path.isfile(c) and os.path.isfile(m)):
            raise SystemExit(f'{d}: needs both config.json and model.pt')
        cfgs.append(json.load(open(c)))
    ref = cfgs[0]
    for d, c in zip(run_dirs[1:], cfgs[1:]):
        diff = [k for k in MUST_MATCH if c.get(k) != ref.get(k)]
        if diff:
            raise SystemExit(
                f'{d} differs from {run_dirs[0]} on {diff}. Ensemble members must '
                'share architecture and adapter shape; refusing to average them.')
    return cfgs


def build(ref, device) -> FineTunedCRF:
    backbone = PLMBackbone(
        model_name=ref['model_name'], n_lora_blocks=ref['lora_blocks'],
        lora_r=ref['lora_r'], lora_alpha=ref['lora_alpha'], lora_dropout=0.0,
        grad_checkpoint=False, max_len=ref['max_len'], device=device,
    )
    head = LSTMCNNCRF(
        input_size=backbone.d_model, num_labels=2, num_states=51,
        dropout_input=ref['dropout'], n_filters=ref['num_filters'],
        hidden_size=ref['hidden_size'], filter_size=ref['kernel_size'],
        dropout_conv1=ref['conv_dropout'],
    )
    return FineTunedCRF(backbone, head,
                        normalize_input=not ref.get('no_input_norm', False),
                        autocast_backbone=False).to(device).eval()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', required=True,
                   help='Glob for member directories, e.g. "results/ft_b12_lr1e-4*". '
                        'Quote it so the shell does not expand it.')
    p.add_argument('--out_json', default='')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--num_cpu_threads', type=int, default=0)
    p.add_argument('--tolerance', type=int, default=3)
    p.add_argument('--combine', choices=('product', 'mixture'), default='product',
                   help='product: mean of log-probs (every member holds a veto). '
                        'mixture: mean of probs (one confident member carries a call, '
                        'so recall is higher). Pick on validation, report on test.')
    p.add_argument('--partition', type=int, default=4,
                   help='4 = test (default), 3 = validation. Use 3 to choose --combine '
                        'and --min_val_f1 without touching the test set.')
    p.add_argument('--min_val_f1', type=float, default=0.0,
                   help="Drop members whose own valid_metrics.json F1 is below this. "
                        'Screening on validation is legitimate model selection; screening '
                        'on test_metrics.json would not be, which is why this reads the '
                        'validation file only.')
    args = p.parse_args()

    if args.num_cpu_threads:
        torch.set_num_threads(args.num_cpu_threads)

    run_dirs = sorted(glob.glob(args.runs))
    if len(run_dirs) < 2:
        raise SystemExit(f'{args.runs} matched {len(run_dirs)} run(s); need at least 2.')

    if args.min_val_f1 > 0:
        kept = []
        for d in run_dirs:
            v = os.path.join(d, 'valid_metrics.json')
            if not os.path.isfile(v):
                raise SystemExit(f'--min_val_f1 needs {v}, which does not exist.')
            f1 = json.load(open(v))['f1 propeptides']
            if f1 >= args.min_val_f1:
                kept.append(d)
            else:
                print(f'  dropped {os.path.basename(d):24} val F1 {f1:.4f} '
                      f'< {args.min_val_f1}')
        if len(kept) < 2:
            raise SystemExit(f'--min_val_f1 {args.min_val_f1} left {len(kept)} member(s); '
                             'need at least 2.')
        run_dirs = kept

    cfgs = load_configs(run_dirs)
    ref = cfgs[0]
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'{len(run_dirs)} members | lora_blocks={ref["lora_blocks"]} r={ref["lora_r"]} '
          f'max_len={ref["max_len"]}')

    # shuffle=False: compute_all_metrics pairs predictions in LOADER order with
    # names in DATASET order, positionally. Any permutation here silently
    # mismatches every protein.
    ds = SequenceCRFDataset(ref['data_file'], ref['partitioning_file'],
                            (args.partition,), ref['max_len'])
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=SequenceCRFDataset.collate_fn,
                        num_workers=args.num_workers)

    model = build(ref, device)
    summed, crf_sum, n_batches = None, {}, 0

    for i, d in enumerate(run_dirs, 1):
        sd = torch.load(os.path.join(d, 'model.pt'), map_location=device)
        # Accepts both the 5.5 GB full state_dict and the ~12 MB adapter-only
        # one, since strict=False tolerates a checkpoint that omits the frozen
        # encoder already in memory.
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not unexpected, f'{d}: unexpected keys {unexpected[:3]}'
        assert all(k.startswith('backbone.plm.') and 'lora_' not in k for k in missing), \
            f'{d}: checkpoint missing trainable parameters {missing[:3]}'

        for key in ('transitions', 'start_transitions', 'end_transitions'):
            t = getattr(model.head.crf, key, None)
            if t is not None:
                crf_sum[key] = t.detach().clone() if key not in crf_sum else crf_sum[key] + t.detach()

        batch_em = []
        with torch.no_grad():
            for sequences, _lab, _prop in loader:
                reps, mask = model.backbone(sequences)
                reps = model.input_norm(reps.float())
                feats = model.head.feature_extractor(reps.permute(0, 2, 1), mask)
                raw = model.head.features_to_emissions(feats)      # (B, L, 2)
                batch_em.append(F.log_softmax(raw.float(), dim=-1).cpu())

        if summed is None:
            summed, n_batches = batch_em, len(batch_em)
        else:
            if len(batch_em) != n_batches:
                raise SystemExit(f'{d} produced {len(batch_em)} batches, expected {n_batches}.')
            # Both rules accumulate in log space and stream, so peak memory is one
            # member's emissions regardless of how many members are combined.
            # logaddexp is the numerically stable running logsumexp.
            summed = [s + e for s, e in zip(summed, batch_em)] if args.combine == 'product' \
                else [torch.logaddexp(s, e) for s, e in zip(summed, batch_em)]
        print(f'  [{i}/{len(run_dirs)}] {os.path.basename(d)}')

    n = len(run_dirs)
    # product: mean of log-probs. mixture: log of the mean of probs.
    # Both are shifted by a constant per position, which Viterbi is invariant to;
    # the shift is kept anyway so the emissions stay interpretable as log-probs.
    averaged = [s / n for s in summed] if args.combine == 'product' \
        else [s - math.log(n) for s in summed]
    with torch.no_grad():
        # Transitions are averaged arithmetically under BOTH rules. --combine
        # names a rule for the emissions only: a mixture over full label paths
        # is not the mixture of per-position transition matrices, and computing
        # it properly means decoding each member separately and combining paths,
        # which is a different algorithm. The transition matrices are also near
        # identical across members (same allowed-transition mask, all members
        # from the same init), so this choice moves almost nothing.
        for key, tot in crf_sum.items():
            getattr(model.head.crf, key).copy_(tot / n)

    preds = []
    with torch.no_grad():
        for emis, (sequences, _lab, _prop) in zip(averaged, loader):
            _, mask = model.backbone(sequences)     # mask only; cheap relative to the rest
            full = model.head._repeat_emissions(emis.to(device))
            paths, _ = model.head.crf.decode(emissions=full, mask=mask.to(device).byte(),
                                             top_k=1, return_path_scores=False)
            preds.extend(paths)

    assert len(preds) == len(ds.names), \
        f'{len(preds)} predictions for {len(ds.names)} proteins -- loader order diverged.'

    metrics = compute_all_metrics(None, preds, None, ds.names, ds.data,
                                  windows=[args.tolerance])[0]

    # Compare against the members' scores on the SAME partition, so a validation
    # run is not silently benchmarked against test numbers.
    member_file = 'test_metrics.json' if args.partition == 4 else 'valid_metrics.json'
    singles = []
    for d in run_dirs:
        t = os.path.join(d, member_file)
        if os.path.isfile(t):
            singles.append(json.load(open(t))['f1 propeptides'])

    split = {4: 'test', 3: 'validation'}.get(args.partition, f'partition {args.partition}')
    print(f'\n=== {n}-member {args.combine} ensemble on {split}, '
          f'+/-{args.tolerance} tolerance ===')
    for k, v in metrics.items():
        print(f'  {k:26} {v:.4f}')
    if singles:
        mean = sum(singles) / len(singles)
        print(f'\n  member mean  {mean:.4f}   best member {max(singles):.4f}')
        print(f'  ensemble gain over the mean: {metrics["f1 propeptides"] - mean:+.4f}')
        print(f'  ensemble gain over the best: {metrics["f1 propeptides"] - max(singles):+.4f}')

    if args.out_json:
        json.dump({**metrics, 'n_members': n, 'member_f1': singles,
                   'combine': args.combine, 'partition': args.partition,
                   'members': [os.path.basename(d) for d in run_dirs]},
                  open(args.out_json, 'w'), indent=2)
        print(f'\nwritten to {args.out_json}')


if __name__ == '__main__':
    main()
