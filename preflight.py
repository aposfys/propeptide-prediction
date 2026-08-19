'''
Validate the setup before starting a long run. This branch runs the joint
(3-label / 101-state) model on CPU, so a
missing GPU is reported for information only, not as a failure.

Written for the hand-off case: the code is staged on one machine and the run
happens on another. Everything checked here otherwise fails minutes-to-hours
into a run, with an error that points at the wrong thing (a missing hash, or a
shape mismatch deep inside conv1) — or, worse, does not fail at all and quietly
produces a bad result.

Usage:
    python preflight.py --embeddings_dir /path/to/embeddings/esm3
    python preflight.py --embeddings_dir ... --embedding_dim 1536

Exits 0 if the run can start, 1 otherwise.
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

        # This branch is CPU-capable, so no GPU is not an error. Report what is
        # there so the log records which device actually produced the result.
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            ok(f'CUDA GPU available: {props.name} '
               f'({props.total_memory / 1e9:.0f} GB), cuda {torch.version.cuda}')
        else:
            ok(f'no CUDA GPU — running on CPU (torch.version.cuda='
               f'{torch.version.cuda}). Expected on this branch.')

        n_cpu = os.cpu_count() or 1
        ok(f'{n_cpu} CPU cores visible; torch default threads '
           f'{torch.get_num_threads()}')
        if torch.get_num_threads() > n_cpu:
            warn(f'torch is set to {torch.get_num_threads()} threads on {n_cpu} '
                 'cores — oversubscription slows things down. Set --num_cpu_threads.')
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
    data_missing = False
    for f in (args.data_file, args.partitioning_file):
        if os.path.isfile(f):
            ok(f'{f} ({os.path.getsize(f) / 1e6:.1f} MB)')
        else:
            bad(f'{f} not found — run from the repository root')
            data_missing = True

    if data_missing:
        _report()

    data = pd.read_csv(args.data_file, index_col='protein_id')
    parts = pd.read_csv(args.partitioning_file, index_col='AC')
    joined = data.join(parts)
    joined = joined.loc[joined['cluster'].isin([0, 1, 2, 3, 4])]
    # The joint model predicts peptides AND propeptides, so it needs both
    # coordinate columns; the propeptide-only branches need only the second.
    for col in ('coordinates', 'propeptide_coordinates'):
        if col not in joined.columns:
            bad(f"data file has no '{col}' column — wrong data file for the "
                'joint (3-label / 101-state) model')
    ok(f'{len(joined)} sequences across clusters '
       f'{sorted(int(c) for c in joined["cluster"].unique())}')

    print()
    print('=== Embeddings ===')
    d = args.embeddings_dir
    if not os.path.isdir(d):
        bad(f'{d} does not exist. The embeddings do not travel with the git repo — '
            'copy them to this machine or regenerate them.')
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

        # Scale check. LSTMCNN has no input normalisation, so it assumes
        # LayerNorm-scale features. Every correctly-extracted set lands near
        # 0.3 * sqrt(dim): ESM-2 L33 = 0.283, ESM3 after transformer.norm = 0.297,
        # ProstT5 last_hidden_state is post-final_layer_norm too. ESM3's
        # ESMOutput.embeddings is the PRE-norm residual stream and sits at ~250 --
        # ~840x too large, which saturates ~91% of the biLSTM gates at init and
        # looks like "this embedder is just worse". Pre- and post-norm embeddings
        # share shape, dtype and filename, so the dimension check above passes on
        # both; only the magnitude of the values gives it away.
        t32 = t.to(torch.float32)
        if not torch.isfinite(t32).all():
            bad(f'embeddings contain NaN/Inf ({sample[0]}.pt) — the extraction run '
                'did not finish cleanly; regenerate them')
        elif t32.abs().max() == 0:
            bad(f'embeddings are all zeros ({sample[0]}.pt) — likely a truncated or '
                'half-written file; regenerate them')
        else:
            ratio = (t32.norm(dim=-1).median() / dim ** 0.5).item()
            if ratio > 10:
                bad(f'embedding scale is {ratio:.1f} x sqrt(dim); LayerNorm-ed sets '
                    'sit near 0.3. These look like PRE-LayerNorm activations. For '
                    'ESM3 use esm_model.transformer.norm(out.embeddings), not '
                    'out.embeddings — see src/utils/make_embeddings.py. Repair an '
                    'existing set with src/utils/renorm_esm3_embeddings.py.')
            elif not 0.03 < ratio < 3:
                warn(f'embedding scale is {ratio:.2f} x sqrt(dim); LayerNorm-ed sets '
                     'sit near 0.3. Check which layer/tensor the extraction used.')
            else:
                ok(f'embedding scale {ratio:.2f} x sqrt(dim) (LayerNorm-like)')

        # RAM footprint: embeddings are cached in-process for the whole run.
        total_res = sum(len(s) for s in joined['sequence'])
        gb = total_res * args.embedding_dim * 4 / 1e9
        print()
        print('=== Resources ===')
        ok(f'embedding cache will hold ~{gb:.1f} GB in host RAM at full dataset '
           f'({gb * 0.8:.1f} GB for the 4/5 partitions one outer fold uses)')
        warn(f'each concurrent process keeps its own copy of that cache. Running N '
             f'folds in parallel needs ~{gb * 0.8:.0f} GB x N — check `free -g` '
             'before using run_parallel.sh.')

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
