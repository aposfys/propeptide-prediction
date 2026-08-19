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
    p.add_argument('--out_dir', default=None,
                   help='Intended output directory. Checked for a stale Optuna '
                        'study and for free disk space. run_optuna_gpu.sh passes '
                        'this automatically.')
    p.add_argument('--n_trials', type=int, default=30,
                   help='Trial budget the run will use; only used to judge whether '
                        'a pre-existing study in --out_dir is already exhausted.')
    p.add_argument('--n_sample', type=int, default=25,
                   help='How many embedding files to spot-check (default 25). '
                        'Checking one file cannot catch a partially written set.')
    args = p.parse_args()

    print('=== Environment ===')
    try:
        import torch
        ok(f'torch {torch.__version__}')

        # expandable_segments is what keeps this run out of the allocator's
        # flush-and-retry path, which costs ~30 s each time it fires. It is a
        # silent no-op on both counts below — an old torch ignores the key, and
        # an unset variable is simply the default allocator — so neither failure
        # announces itself in the log. Warn here or not at all.
        try:
            major, minor = (int(x) for x in torch.__version__.split('.')[:2])
            if (major, minor) < (2, 1):
                warn(f'torch {torch.__version__} predates expandable_segments '
                     '(added in 2.1). PYTORCH_CUDA_ALLOC_CONF will be ignored and '
                     'fragmentation stalls will return. Upgrade if the log shows '
                     'repeated allocator warnings.')
        except ValueError:
            pass

        alloc_conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')
        if 'expandable_segments' in alloc_conf:
            ok(f'PYTORCH_CUDA_ALLOC_CONF={alloc_conf}')
        else:
            warn('PYTORCH_CUDA_ALLOC_CONF does not set expandable_segments. The '
                 'run works either way, but padded batches of varying length '
                 'fragment the caching allocator, and each recovery stalls the '
                 'GPU for ~30 s. Set it before starting:  '
                 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True')

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
    data_missing = False
    for f in (args.data_file, args.partitioning_file):
        if os.path.isfile(f):
            ok(f'{f} ({os.path.getsize(f) / 1e6:.1f} MB)')
        else:
            bad(f'{f} not found — run from the repository root')
            data_missing = True

    # Bail only if the data files themselves are unreadable, since the pandas
    # read below would crash. Any *other* problem recorded so far (a missing GPU,
    # most likely) must NOT stop us here — checking the embeddings on a CPU login
    # node before handing the job to a GPU machine is a main use of this script.
    if data_missing:
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

        # Scale check. LSTMCNN has no input normalisation, so it assumes
        # LayerNorm-scale features. Every correctly-extracted set lands near
        # 0.3 * sqrt(dim): ESM-2 L33 = 0.283, ESM3 after transformer.norm = 0.297,
        # ProstT5 last_hidden_state is post-final_layer_norm too. ESM3's
        # ESMOutput.embeddings is the PRE-norm residual stream and sits at ~250 --
        # ~840x too large. Same shape, same dtype, same filenames, no crash: only
        # the magnitude of the values gives it away, which is why the dimension
        # check above passes on broken embeddings.
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
                    'out.embeddings — see src/utils/make_embeddings.py.')
            elif not 0.03 < ratio < 3:
                warn(f'embedding scale is {ratio:.2f} x sqrt(dim); LayerNorm-ed sets '
                     'sit near 0.3. Check which layer/tensor the extraction used.')
            else:
                ok(f'embedding scale {ratio:.2f} x sqrt(dim) (LayerNorm-like)')

        if t.dtype in (torch.bfloat16, torch.float16):
            warn(f'embeddings are stored as {t.dtype}. Training upcasts them, so the '
                 'run works, but a half-precision set carries ~0.4% relative error. '
                 'Re-extract in float32 for anything thesis-facing.')

        # Length agreement. An embedding must have exactly one row per residue.
        # Off-by-one or off-by-two here means the BOS/EOS handling in the
        # extractor is wrong, which silently shifts every label by a position --
        # the model still trains, and every metric is quietly wrong.
        import hashlib
        by_hash = {}
        for s in joined['sequence']:
            by_hash[hashlib.md5(s.encode()).digest().hex()] = len(s)
        checked = sorted(present & needed)[:args.n_sample]
        bad_len, bad_scale, bad_finite = [], [], []
        for h in checked:
            tt = torch.load(os.path.join(d, f'{h}.pt'), map_location='cpu')
            if tt.shape[0] != by_hash[h]:
                bad_len.append((h, tt.shape[0], by_hash[h]))
            tt32 = tt.to(torch.float32)
            if not torch.isfinite(tt32).all():
                bad_finite.append(h)
            elif tt32.abs().max() > 0:
                r = (tt32.norm(dim=-1).median() / tt.shape[-1] ** 0.5).item()
                if not 0.03 < r < 3:
                    bad_scale.append((h, r))
        if bad_len:
            h, got, want = bad_len[0]
            bad(f'{len(bad_len)}/{len(checked)} embeddings do not match their '
                f'sequence length (e.g. {h}: {got} rows for a {want}-residue '
                'protein). The extractor is mishandling BOS/EOS. This does NOT '
                'crash training -- it shifts every label, and every metric is '
                'silently wrong. Regenerate.')
        elif checked:
            ok(f'{len(checked)} spot-checked embeddings all match their sequence length')
        if bad_finite:
            bad(f'{len(bad_finite)}/{len(checked)} spot-checked embeddings contain '
                f'NaN/Inf (e.g. {bad_finite[0]}) — partially written set, regenerate')
        if bad_scale:
            h, r = bad_scale[0]
            bad(f'{len(bad_scale)}/{len(checked)} spot-checked embeddings are off-scale '
                f'(e.g. {h}: {r:.2f} x sqrt(dim)). A mixed set means an interrupted '
                'run wrote some files before a fix and some after — regenerate all.')
        elif checked and not bad_len:
            ok(f'{len(checked)} spot-checked embeddings consistent in scale')

        # RAM footprint: embeddings are cached in-process for the whole run.
        total_res = sum(len(s) for s in joined['sequence'])
        gb = total_res * args.embedding_dim * 4 / 1e9
        print()
        print('=== Resources ===')
        ok(f'embedding cache will hold ~{gb:.1f} GB in host RAM at full dataset '
           f'({gb * 0.8:.1f} GB for the 4/5 partitions one outer fold uses)')
        warn('one training process per GPU — each process keeps its own copy of that '
             'cache, so do not launch several folds concurrently on one device')

        _check_out_dir(args)

    _report()


def _check_out_dir(args):
    '''Stale-study and disk-space checks on the intended output directory.'''
    if not args.out_dir:
        warn('no --out_dir given, so the stale-study and disk checks were skipped. '
             'Pass it (run_optuna_gpu.sh does this automatically).')
        return

    import shutil

    # Free space. Each retrained model pickles its test outputs, and those files
    # are ~440 MB apiece: 4 per outer fold, 20 for a full nested CV.
    if os.path.isdir(args.out_dir) or os.path.isdir(os.path.dirname(args.out_dir) or '.'):
        target = args.out_dir if os.path.isdir(args.out_dir) else (os.path.dirname(args.out_dir) or '.')
        free_gb = shutil.disk_usage(target).free / 1e9
        if free_gb < 5:
            bad(f'only {free_gb:.1f} GB free at {target}. Each fold writes 4 test-output '
                'pickles of ~440 MB (~1.8 GB/fold, ~8.8 GB for five).')
        elif free_gb < 15:
            warn(f'{free_gb:.1f} GB free at {target} — enough for one fold '
                 '(~1.8 GB) but not a full nested CV (~8.8 GB).')
        else:
            ok(f'{free_gb:.0f} GB free at {target}')

    # THE expensive mistake: Optuna persists each study to SQLite and
    # train_nested_cv() resumes with load_if_exists=True, running only
    # (n_trials - already_done) more. Re-using an out_dir that already holds a
    # completed study therefore runs ZERO new trials, silently returns the old
    # best_params, and retrains from them -- producing fresh-looking output files
    # from a stale search. On a multi-day run that is a very costly no-op.
    dbs = []
    if os.path.isdir(args.out_dir):
        dbs = [f for f in os.listdir(args.out_dir) if f.endswith('.db')]
    if not dbs:
        ok(f'{args.out_dir}: no pre-existing Optuna study — this will be a fresh search')
        return

    try:
        import optuna
        for db in dbs:
            path = os.path.abspath(os.path.join(args.out_dir, db))
            names = optuna.study.get_all_study_names(f'sqlite:///{path}')
            for n in names:
                st = optuna.load_study(study_name=n, storage=f'sqlite:///{path}')
                done = len([t for t in st.trials
                            if t.state in (optuna.trial.TrialState.COMPLETE,
                                           optuna.trial.TrialState.PRUNED)])
                if done >= args.n_trials:
                    bad(f'{db} already holds {done} finished trials for study "{n}", '
                        f'and the budget is {args.n_trials}. The run would execute '
                        'ZERO new trials and retrain from the OLD best_params, '
                        'writing files that look fresh. Use a new --out_dir (or '
                        'delete this .db) unless you are deliberately resuming.')
                else:
                    warn(f'{db} holds {done} finished trials for study "{n}"; the run '
                         f'will resume and add {args.n_trials - done} more. Intended?')
    except ImportError:
        warn(f'optuna not importable, cannot inspect {dbs} — check by hand that '
             'you are not resuming a finished study.')
    except Exception as e:                                   # noqa: BLE001
        warn(f'could not read the Optuna study in {args.out_dir} ({e}). Check by '
             'hand that you are not resuming a finished study.')


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
