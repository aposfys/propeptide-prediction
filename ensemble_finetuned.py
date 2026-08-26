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

  * the 2-logit emissions from head.features_to_emissions, averaged as
    log-softmax -- a product of experts, the standard combination for
    log-linear models like a CRF. Averaging raw logits would let one
    over-confident member dominate.
  * the CRF's transitions / start_transitions / end_transitions, averaged
    directly. Low-dimensional, structurally identical across members (same
    allowed-transition mask), and log-potentials, so an arithmetic mean is again
    a product of experts.

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
    args = p.parse_args()

    if args.num_cpu_threads:
        torch.set_num_threads(args.num_cpu_threads)

    run_dirs = sorted(glob.glob(args.runs))
    if len(run_dirs) < 2:
        raise SystemExit(f'{args.runs} matched {len(run_dirs)} run(s); need at least 2.')
    cfgs = load_configs(run_dirs)
    ref = cfgs[0]
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'{len(run_dirs)} members | lora_blocks={ref["lora_blocks"]} r={ref["lora_r"]} '
          f'max_len={ref["max_len"]}')

    # shuffle=False: compute_all_metrics pairs predictions in LOADER order with
    # names in DATASET order, positionally. Any permutation here silently
    # mismatches every protein.
    ds = SequenceCRFDataset(ref['data_file'], ref['partitioning_file'], (4,), ref['max_len'])
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
            summed = [s + e for s, e in zip(summed, batch_em)]
        print(f'  [{i}/{len(run_dirs)}] {os.path.basename(d)}')

    n = len(run_dirs)
    averaged = [s / n for s in summed]
    with torch.no_grad():
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

    singles = []
    for d in run_dirs:
        t = os.path.join(d, 'test_metrics.json')
        if os.path.isfile(t):
            singles.append(json.load(open(t))['f1 propeptides'])

    print(f'\n=== {n}-member ensemble, +/-{args.tolerance} tolerance ===')
    for k, v in metrics.items():
        print(f'  {k:26} {v:.4f}')
    if singles:
        mean = sum(singles) / len(singles)
        print(f'\n  member mean  {mean:.4f}   best member {max(singles):.4f}')
        print(f'  ensemble gain over the mean: {metrics["f1 propeptides"] - mean:+.4f}')
        print(f'  ensemble gain over the best: {metrics["f1 propeptides"] - max(singles):+.4f}')

    if args.out_json:
        json.dump({**metrics, 'n_members': n, 'member_f1': singles},
                  open(args.out_json, 'w'), indent=2)
        print(f'\nwritten to {args.out_json}')


if __name__ == '__main__':
    main()
