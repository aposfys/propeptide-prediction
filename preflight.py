'''
Validate the setup before starting a long run. These branches run on either CPU
or GPU, so a missing GPU is reported for information only, not as a failure.

Written for the hand-off case: the code is staged on one machine and the run
happens on another. Everything checked here otherwise fails minutes-to-hours
into a run, with an error that points at the wrong thing (a missing hash, or a
shape mismatch deep inside conv1) — or, worse, does not fail at all and quietly
produces a bad result.

Usage:
    python preflight.py --embeddings_dir /path/to/embeddings/prost5
    python preflight.py --embeddings_dir ... --embedding_dim 1024

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
    p.add_argument('--embedding_dim', type=int, default=1024)
    p.add_argument('--batch_size', type=int, default=100,
                   help='the --batch_size the training run will use; the '
                        'accelerator-memory probe is sized from it')
    p.add_argument('--skip_memory_probe', action='store_true',
                   help='skip the worst-case forward/backward on the accelerator')
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
               f'{torch.version.cuda}). Fine for training; note that ProstT5 '
               '--half needs a GPU.')

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

    for mod in ('pandas', 'numpy', 'tensorboard', 'transformers'):
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
    if 'propeptide_coordinates' not in joined.columns:
        bad("data file has no 'propeptide_coordinates' column — wrong data file")
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

        # Row count must equal the sequence length. The ProstT5 tokenisation is
        # [<AA2fold>, r1..rL, </s>], so the extractor slices [1:L+1]. An off-by-one
        # here leaks the prefix or the EOS into the features, and nothing
        # downstream would complain -- the CRF just trains on a shifted sequence.
        by_hash = {md5(s.encode()).digest().hex(): s for s in joined['sequence']}
        bad_len = []
        nonfinite = []
        for h in sample[:200]:
            seq = by_hash.get(h)
            if seq is None:
                continue
            tt = torch.load(os.path.join(d, f'{h}.pt'), map_location='cpu')
            rows = tt.shape[0]
            if rows != len(seq):
                bad_len.append((h, rows, len(seq)))
            # Checked on every sampled file, not just one: a --half extraction can
            # overflow on a handful of long sequences and leave the rest perfect.
            if not torch.isfinite(tt.to(torch.float32)).all():
                nonfinite.append(h)
        if bad_len:
            h, rows, n = bad_len[0]
            bad(f'{len(bad_len)} of the {min(len(sample), 200)} sampled embeddings have the '
                f'wrong number of rows (e.g. {h}: {rows} rows for a {n}-residue sequence). '
                'The prefix/EOS slice is off — regenerate with src/utils/make_embeddings.py.')
        else:
            ok(f'row counts match sequence lengths on {min(len(sample), 200)} sampled files')

        if nonfinite:
            bad(f'{len(nonfinite)} of the {min(len(sample), 200)} sampled embeddings contain '
                f'NaN or Inf (e.g. {nonfinite[0]}.pt). This is what an fp16 (--half) '
                'extraction looks like when T5 overflows. Regenerate in full precision.')
        else:
            ok(f'no NaN/Inf in {min(len(sample), 200)} sampled files')

        # Storage-bloat check, specific to this branch's history. A deleted
        # duplicate extractor (make_embeddings_prost5.py) saved the output slice
        # without .clone(), so torch.save serialised the whole padded batch
        # storage behind the view: ~400 MB per file instead of ~0.2 MB. The
        # tensor loads back correctly, so only the file size gives it away.
        sizes = [(h, os.path.getsize(os.path.join(d, f'{h}.pt'))) for h in sample[:200]]
        bloated = []
        for h, nbytes in sizes:
            seq = by_hash.get(h)
            if seq is None:
                continue
            expected = len(seq) * args.embedding_dim * 4
            if nbytes > max(4 * expected, 10 * 1024 ** 2):
                bloated.append((h, nbytes, expected))
        if bloated:
            h, nbytes, expected = bloated[0]
            bad(f'{len(bloated)} sampled file(s) are far larger than their contents '
                f'(e.g. {h}: {nbytes / 1e6:.1f} MB on disk for {expected / 1e6:.2f} MB of '
                'values). These came from the old extractor that omitted .clone() on the '
                'output slice, so each file carries the whole padded batch. The values are '
                'correct but the set will exhaust the disk — regenerate with '
                'src/utils/make_embeddings.py into a fresh directory.')
        else:
            total_mb = sum(n for _, n in sizes) / 1e6
            ok(f'file sizes consistent with contents ({total_mb:.1f} MB across '
               f'{len(sizes)} sampled files)')

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
                    'sit near 0.3. ProstT5 should never land here: HuggingFace '
                    'T5Stack.forward applies final_layer_norm before returning, so '
                    'last_hidden_state is already normalised. A large value means '
                    'the extraction took something other than last_hidden_state. '
                    'Do NOT "fix" this by normalising afterwards — regenerate with '
                    'src/utils/make_embeddings.py.')
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
             'first.')

        if not args.skip_memory_probe:
            memory_probe(joined, d, args.embedding_dim, args.batch_size)

    _report()


def _probe_device():
    import torch
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return None


def memory_probe(joined, embeddings_dir, embedding_dim, batch_size):
    '''Run the single worst batch of the epoch and report what it costs.

    Activation memory here is dominated by the CRF forward algorithm, which is
    O(batch x length x num_states^2) — at batch 100 it measures 1.88 GB per 1000
    residues of padded length. Batches are padded to their longest member, so
    the peak of a whole run is set by one batch: the one holding the longest
    sequence (this dataset goes up to 3971 residues). Training shuffles, so that
    batch can turn up at any epoch — an OOM found here costs a minute, the same
    OOM found by the run can cost a night.
    '''
    import torch
    from src.models import LSTMCNNCRF

    dev = _probe_device()
    if dev is None:
        ok('no accelerator — CPU training is limited by host RAM, see above')
        return

    lengths = joined['sequence'].str.len().sort_values(ascending=False)
    worst = lengths.index[:batch_size]
    pad_len = int(lengths.iloc[0])
    embs = []
    for name in worst:
        h = md5(joined.loc[name, 'sequence'].encode()).digest().hex()
        f = os.path.join(embeddings_dir, f'{h}.pt')
        if not os.path.isfile(f):
            warn('memory probe skipped — some of the longest sequences have no '
                 'embedding yet')
            return
        embs.append(torch.load(f, map_location='cpu').to(torch.float32))

    x = torch.nn.utils.rnn.pad_sequence(embs, batch_first=True).permute(0, 2, 1)
    mask = torch.nn.utils.rnn.pad_sequence(
        [torch.ones(e.shape[0]) for e in embs], batch_first=True)
    tgt = torch.zeros(mask.shape, dtype=torch.long)
    del embs

    model = LSTMCNNCRF(
        input_size=embedding_dim, num_labels=2, dropout_input=0.1,
        num_states=51, n_filters=32, hidden_size=64, filter_size=3,
        dropout_conv1=0.1,
    ).to(dev)

    if dev.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
    else:
        total = None

    try:
        x, mask, tgt = x.to(dev), mask.to(dev), tgt.to(dev)
        _, _, loss = model(x, mask, tgt, skip_marginals=True, skip_decode=True)
        # The forward activations are the peak; read them before backward frees them.
        mps_peak = (torch.mps.current_allocated_memory() / 1e9
                    if dev.type == 'mps' else None)
        loss.backward()
    except RuntimeError as e:   # torch.cuda.OutOfMemoryError subclasses this
        if 'out of memory' in str(e).lower():
            bad(f'OUT OF MEMORY on the worst batch ({batch_size} sequences padded to '
                f'{pad_len} residues) — this run would die mid-epoch. Lower '
                '--batch_size (and say so in the run config, it is a training '
                'hyperparameter, not a detail).')
            return
        raise

    if dev.type == 'cuda':
        peak = torch.cuda.max_memory_allocated() / 1e9
        msg = (f'worst batch ({batch_size} x {pad_len} residues) peaks at '
               f'{peak:.1f} GB of {total:.0f} GB VRAM')
        if peak > 0.85 * total:
            warn(msg + ' — under 15% headroom; consider a smaller --batch_size')
        else:
            ok(msg)
    else:
        peak = mps_peak
        ok(f'worst batch ({batch_size} x {pad_len} residues) allocates '
           f'{peak:.1f} GB on {dev.type} — CUDA will be in the same range')


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
