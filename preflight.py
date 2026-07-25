'''
Validate the setup before starting a long run. This branch is GPU-only, so a
machine without a CUDA GPU fails this check.

Written for the hand-off case: the code is staged on one machine and the search
is run on another, by someone who did not set it up. Everything checked here
otherwise fails minutes-to-hours into a run, with an error that points at the
wrong thing (a missing hash, or a shape mismatch deep inside conv1).

Usage:
    python preflight.py --embeddings_dir /path/to/embeddings/esm3
    python preflight.py --embeddings_dir ... --embedding_dim 1536

Exits 0 if the run can start, 1 otherwise. run_optuna_gpu.sh calls this first.

Run it on the GPU machine — running it on a CPU login node will (correctly)
report the missing GPU as a failure, so use it there only to check the data and
embeddings, reading past that one line.
'''
import argparse
import os
import sys
from hashlib import md5

problems = []
warnings = []


def ok(msg):
    print(f'  ok    {msg}')


def bad(msg):
    print(f'  FAIL  {msg}')
    problems.append(msg)


def warn(msg):
    print(f'  warn  {msg}')
    warnings.append(msg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--embeddings_dir', required=True)
    p.add_argument('--data_file', default='data/labeled_sequences.csv')
    p.add_argument('--partitioning_file', default='data/graphpart_assignments.csv')
    p.add_argument('--embedding_dim', type=int, default=1536)
    args = p.parse_args()

    print('=== Environment ===')
    try:
        import torch
        ok(f'torch {torch.__version__}')
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            ok(f'CUDA GPU: {props.name} ({props.total_memory / 1e9:.0f} GB), '
               f'cuda {torch.version.cuda}')
        else:
            # This branch is GPU-only, so no GPU is a failure, not a warning.
            bad(f'no CUDA GPU visible (torch.version.cuda={torch.version.cuda}). '
                'This branch is GPU-only — a full search is hundreds of trainings '
                'and is not viable on CPU. Run on a GPU node; if one should be '
                'visible here, check `nvidia-smi` and whether this is a CPU-only '
                'torch build (cuda=None).')
    except ImportError as e:
        bad(f'torch not importable: {e}')
        print()
        print('Install the dependencies first:  pip install -r requirements.txt')
        sys.exit(1)

    for mod in ('optuna', 'pandas', 'numpy', 'tensorboard'):
        try:
            __import__(mod)
            ok(mod)
        except ImportError:
            bad(f'{mod} not installed  (pip install -r requirements.txt)')

    import pandas as pd

    print()
    print('=== Data ===')
    for f in (args.data_file, args.partitioning_file):
        if os.path.isfile(f):
            ok(f'{f} ({os.path.getsize(f) / 1e6:.1f} MB)')
        else:
            bad(f'{f} not found — run from the repository root')

    if problems:
        _report()

    data = pd.read_csv(args.data_file, index_col='protein_id')
    parts = pd.read_csv(args.partitioning_file, index_col='AC')
    joined = data.join(parts)
    joined = joined.loc[joined['cluster'].isin([0, 1, 2, 3, 4])]
    if 'propeptide_coordinates' not in joined.columns:
        bad("data file has no 'propeptide_coordinates' column — wrong data file "
            'for a propeptide-only model')
    ok(f'{len(joined)} sequences across clusters '
       f'{sorted(int(c) for c in joined["cluster"].unique())}')

    print()
    print('=== Embeddings ===')
    d = args.embeddings_dir
    if not os.path.isdir(d):
        bad(f'{d} does not exist. The embeddings do not travel with the git repo — '
            'copy them to this machine or regenerate them (see OPTUNA_GPU.md).')
        _report()

    # The dataset addresses embeddings by md5 of the sequence, so check the exact
    # files this run will ask for rather than just counting the directory.
    needed = {md5(s.encode()).digest().hex() for s in joined['sequence']}
    present = {f[:-3] for f in os.listdir(d) if f.endswith('.pt')}
    missing = needed - present

    ok(f'{len(present)} .pt files in {d}')
    if missing:
        bad(f'{len(missing)} of {len(needed)} required embeddings are MISSING. '
            f'Example absent hash: {sorted(missing)[0]}. '
            'Regenerate with src/utils/make_embeddings.py, or copy the full set.')
    else:
        ok(f'all {len(needed)} required embeddings present')

    # Dimension check — the highest-value test here. Several embedding sets
    # commonly sit side by side (ESM-2 = 1280, ESM3 = 1536, ProstT5 = 1024), and
    # pointing at the wrong one otherwise surfaces as a shape error inside conv1.
    sample = sorted(present & needed) or sorted(present)
    if sample:
        import torch
        t = torch.load(os.path.join(d, f'{sample[0]}.pt'), map_location='cpu')
        dim = t.shape[-1]
        if dim == args.embedding_dim:
            ok(f'embedding dim {dim} matches --embedding_dim {args.embedding_dim} '
               f'(sample shape {tuple(t.shape)}, {t.dtype})')
        else:
            bad(f'embedding dim is {dim} but --embedding_dim is {args.embedding_dim}. '
                'These are the wrong embeddings for this run '
                '(ESM-2 = 1280, ESM3 = 1536, ProstT5 = 1024).')
        if t.dim() != 2:
            bad(f'expected a 2-D (length, dim) tensor, got {tuple(t.shape)}')

        # RAM footprint: embeddings are cached in-process for the whole run.
        total_res = sum(len(s) for s in joined['sequence'])
        gb = total_res * args.embedding_dim * 4 / 1e9
        print()
        print('=== Resources ===')
        ok(f'embedding cache will hold ~{gb:.1f} GB in host RAM at full dataset '
           f'({gb * 0.8:.1f} GB for the 4/5 partitions one outer fold uses)')
        warn('one process per GPU only — do not use run_parallel.sh here, it starts '
             '5 processes and each keeps its own copy of that cache')

    _report()


def _report():
    print()
    if problems:
        print(f'PREFLIGHT FAILED — {len(problems)} problem(s) above must be fixed.')
        sys.exit(1)
    if warnings:
        print(f'Preflight passed with {len(warnings)} warning(s). The run can start.')
    else:
        print('Preflight passed. The run can start.')
    sys.exit(0)


if __name__ == '__main__':
    main()
