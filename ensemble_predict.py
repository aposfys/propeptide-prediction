'''
Score an ensemble of trained models, the way the DeepPeptide paper's headline
numbers are produced.

Upstream reports a 20-model ensemble (btad616 Fig. S4); this repo trains and
reports single models, so a single-model result here is being compared against
an ensembled result there. That is not like-for-like, and it costs us -- the
paper's propeptide-only figure is 0.535 (Fig. S7) with ensembling, against
0.5547 for one ESM3 model without it.

WHAT IS AVERAGED, AND WHY

Two things are learned per model and both have to be combined:

  * the 2-logit emissions from features_to_emissions, averaged as log-softmax.
    Averaging in log space is a product of experts, which is the standard
    combination for log-linear models like a CRF: the ensemble's emission score
    is the geometric mean of the members' per-position distributions. Averaging
    raw logits instead would let one over-confident member dominate.

  * the CRF's transitions / start_transitions / end_transitions, averaged
    directly. These are low-dimensional, structurally identical across models
    (every member is built with the same allowed-transition mask, so the same
    entries are live in all of them), and they are log-potentials -- so an
    arithmetic mean in log space is again a product of experts.

The feature extractor's weights are NOT averaged. Independently initialised
networks are not in a shared basis and averaging them produces nonsense; that
would be a "model soup", which is a different thing from an ensemble. Each
member runs its own forward pass and only the outputs are combined.

Decoding happens ONCE, on the averaged emissions with the averaged transitions,
so the CRF's state grammar is enforced exactly as in a single model. Averaging
decoded paths instead would discard all the confidence information that makes
ensembling work.

REQUIREMENTS

Every run directory must hold a model.pt and a config.json, and all configs must
agree on the architecture and the embeddings -- otherwise the members are not
ensembleable and the script refuses rather than silently averaging incompatible
models.
'''
import argparse
import glob
import json
import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from src.train_loop_crf import get_dataloaders, get_model
from src.utils.manuscript_metrics import compute_all_metrics

# Fields that must match across members. embeddings_dir and embedding_dim decide
# what the model reads; the rest decide its shape. A mismatch means the averaged
# emissions would not correspond to the same function.
MUST_MATCH = ('embeddings_dir', 'embedding_dim', 'model', 'num_filters',
              'hidden_size', 'kernel_size', 'data_file', 'partitioning_file')


def _load_configs(run_dirs):
    cfgs = []
    for d in run_dirs:
        cpath, mpath = os.path.join(d, 'config.json'), os.path.join(d, 'model.pt')
        if not (os.path.isfile(cpath) and os.path.isfile(mpath)):
            raise SystemExit(f'{d}: needs both config.json and model.pt')
        cfgs.append(json.load(open(cpath)))

    ref = cfgs[0]
    for d, c in zip(run_dirs[1:], cfgs[1:]):
        diff = [k for k in MUST_MATCH if c.get(k) != ref.get(k)]
        if diff:
            raise SystemExit(
                f'{d} differs from {run_dirs[0]} on {diff}. Members of an ensemble '
                'must share architecture and embeddings; refusing to average them.'
            )
    return cfgs


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', required=True,
                   help='Glob for the member run directories, e.g. '
                        '"results/esm3_ens/model_*". Quote it so the shell does '
                        'not expand it.')
    p.add_argument('--out_json', default='',
                   help='Optional path to write the ensemble metrics to.')
    p.add_argument('--batch_size', type=int, default=20)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--num_cpu_threads', type=int, default=0)
    p.add_argument('--tolerance', type=int, default=3,
                   help='Boundary tolerance in residues. 3 matches every other '
                        'number in RESULTS.md.')
    args = p.parse_args()

    if args.num_cpu_threads:
        torch.set_num_threads(args.num_cpu_threads)

    run_dirs = sorted(glob.glob(args.runs))
    if len(run_dirs) < 2:
        raise SystemExit(f'{args.runs} matched {len(run_dirs)} run(s); need at least 2.')
    cfgs = _load_configs(run_dirs)
    ref = cfgs[0]
    print(f'{len(run_dirs)} members, {ref["embedding_dim"]}-dim '
          f'{os.path.basename(str(ref["embeddings_dir"]))} embeddings')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    margs = SimpleNamespace(**ref)
    margs.batch_size = args.batch_size
    margs.num_workers = args.num_workers
    margs.embedding = ref.get('embedding', 'precomputed')

    # shuffle=False on the test loader is an invariant of get_dataloaders, which
    # is what makes per-batch emissions comparable across members.
    _, _, test_loader = get_dataloaders(margs, [0, 1, 2], [3], [4])

    summed = None          # list of (B, L, 2) log-softmax emissions, one per batch
    crf_sum = {}
    n_batches = 0

    for i, (d, cfg) in enumerate(zip(run_dirs, cfgs), 1):
        margs_i = SimpleNamespace(**cfg)
        margs_i.batch_size, margs_i.num_workers = args.batch_size, args.num_workers
        model = get_model(margs_i).to(device).eval()
        model.load_state_dict(torch.load(os.path.join(d, 'model.pt'), map_location=device))

        for key in ('transitions', 'start_transitions', 'end_transitions'):
            t = getattr(model.crf, key, None)
            if t is not None:
                crf_sum[key] = t.detach().clone() if key not in crf_sum else crf_sum[key] + t.detach()

        batch_emissions = []
        with torch.no_grad():
            for b, batch in enumerate(test_loader):
                embeddings, mask, _, _ = batch
                feats = model.feature_extractor(embeddings.to(device), mask.to(device))
                raw = model.features_to_emissions(feats)        # (B, L, 2)
                batch_emissions.append(F.log_softmax(raw.float(), dim=-1).cpu())

        if summed is None:
            summed, n_batches = batch_emissions, len(batch_emissions)
        else:
            if len(batch_emissions) != n_batches:
                raise SystemExit(f'{d} produced {len(batch_emissions)} batches, '
                                 f'expected {n_batches}. Loader order differs.')
            summed = [s + e for s, e in zip(summed, batch_emissions)]
        print(f'  [{i}/{len(run_dirs)}] {os.path.basename(d)}')

    n = len(run_dirs)
    averaged = [s / n for s in summed]

    # One decoder, carrying the averaged transitions.
    decoder = get_model(SimpleNamespace(**ref)).to(device).eval()
    decoder.load_state_dict(torch.load(os.path.join(run_dirs[0], 'model.pt'),
                                       map_location=device))
    with torch.no_grad():
        for key, tot in crf_sum.items():
            getattr(decoder.crf, key).copy_(tot / n)

    preds = []
    with torch.no_grad():
        for emis, batch in zip(averaged, test_loader):
            _, mask, _, _ = batch
            full = decoder._repeat_emissions(emis.to(device))
            paths, _ = decoder.crf.decode(emissions=full, mask=mask.to(device).byte(),
                                          top_k=1, return_path_scores=False)
            preds.extend(paths)

    # compute_all_metrics reads only preds, names and true_df -- probs and labels
    # are accepted and ignored, so they are not collected here.
    metrics = compute_all_metrics(
        None, preds, None,
        test_loader.dataset.names, test_loader.dataset.data,
        windows=[args.tolerance],
    )[0]

    print(f'\n=== {n}-model ensemble, +/-{args.tolerance} tolerance ===')
    for k, v in metrics.items():
        print(f'  {k:26} {v:.4f}')

    if args.out_json:
        json.dump(metrics, open(args.out_json, 'w'), indent=2)
        print(f'\nwritten to {args.out_json}')


if __name__ == '__main__':
    main()
