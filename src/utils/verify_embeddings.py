'''
Acceptance test for a directory of precomputed per-residue embeddings.

Run this after make_embeddings.py and before spending GPU-days training on the
result. Every failure mode below has actually happened in this project:

  * a full disk truncated .pt files mid-write, and the error only surfaced
    epochs later as `unexpected pos 832 vs 726`
  * 1190 embeddings were written in bfloat16 when the rest were fp32
  * the ESM3 extractor returned the RAW pre-LayerNorm residual stream, whose
    per-token norm is ~9793 against ESM-2's ~10; nothing crashed, the model
    just trained badly for weeks

None of those raise an exception at extraction time. They are all visible in
thirty seconds of arithmetic over the finished directory.

  python src/utils/verify_embeddings.py \
      --embeddings_dir /mnt/storage/fysekidis/embeddings/esm2 \
      --data_file data/labeled_sequences.csv \
      --expect_dim 1280 --expect_norm 10.12

--expect_norm is the single most informative check, and it is embedder-specific.
Known good values for this project (mean L2 norm of a per-residue vector):

    ESM-2 650M, layer 33, normed     10.12
    ESM3 1.4B, post-transformer.norm  ~10
    ESM3 1.4B, RAW pre-norm          ~9793   <- the bug, do not accept

Exit status is 1 if any check fails, so it can gate a shell pipeline:

    python src/utils/verify_embeddings.py ... && python run.py ...
'''
import argparse
import os
import sys
from hashlib import md5

import numpy as np
import pandas as pd
import torch

# make_embeddings.py names files by md5 of the sequence, and the training
# datasets look them up the same way (src/utils/dataset.py:make_hashes). The
# two must agree byte for byte, which is exactly what the coverage check below
# is testing -- a FASTA that differs from the CSV by so much as a trailing
# character produces a directory full of files nothing will ever read.
def hash_aa_string(s: str) -> str:
    return md5(s.encode()).digest().hex()


# Upstream's extractor slices long inputs into disjoint windows of this many
# tokens (make_embeddings.py:52-63). The windows carry no overlap and no BOS
# after the first, so the representation is discontinuous at each boundary.
# That is upstream's behaviour and is reproduced deliberately; this script
# measures the size of the discontinuity rather than removing it.
CHUNK_TOKENS = 722


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--embeddings_dir', required=True)
    p.add_argument('--data_file', required=True,
                   help='labeled_sequences.csv -- the sequences that must be covered.')
    p.add_argument('--expect_dim', type=int, default=0,
                   help='Feature dimension every embedding must have (1280 for ESM-2 '
                        '650M, 1536 for ESM3). 0 to infer from the first file.')
    p.add_argument('--expect_norm', type=float, default=0.0,
                   help='Expected mean per-residue L2 norm. Flags an order-of-magnitude '
                        'miss, which is what an unapplied final LayerNorm looks like.')
    p.add_argument('--norm_tol', type=float, default=0.35,
                   help='Relative tolerance on --expect_norm. Generous by design: this '
                        'is checking for 1000x, not 1%%.')
    p.add_argument('--sample', type=int, default=0,
                   help='Check only N sequences (0 = all). Use for a quick look while '
                        'extraction is still running.')
    args = p.parse_args()

    data = pd.read_csv(args.data_file)
    seqs = data['sequence'].tolist()
    names = data['protein_id'].tolist()
    # Duplicate sequences share one file, so dedup before counting: the expected
    # file count is the number of DISTINCT sequences, not the number of rows.
    uniq = {}
    for n, s in zip(names, seqs):
        uniq.setdefault(hash_aa_string(s), (n, s))
    print(f'{len(seqs)} rows -> {len(uniq)} distinct sequences expected on disk')

    items = list(uniq.items())
    if args.sample:
        rng = np.random.default_rng(0)
        items = [items[i] for i in rng.choice(len(items), min(args.sample, len(items)),
                                              replace=False)]
        print(f'sampling {len(items)} of them')

    missing, bad_shape, bad_dtype, nonfinite, zero_rows, unreadable = [], [], [], [], [], []
    norms, dims = [], set()
    # Cosine similarity between neighbouring residues, collected separately at a
    # window boundary and at a control position, to size the discontinuity.
    cos_boundary, cos_control = [], []

    for h, (name, seq) in items:
        path = os.path.join(args.embeddings_dir, f'{h}.pt')
        if not os.path.isfile(path):
            missing.append(name)
            continue
        try:
            # weights_only=True refuses to execute pickled code. These files are
            # ours, but a verifier that can be compromised by the thing it is
            # verifying is not a verifier.
            e = torch.load(path, map_location='cpu', weights_only=True).float()
        except Exception as exc:                       # truncated by a full disk
            unreadable.append(f'{name}: {type(exc).__name__}: {exc}')
            continue

        if e.ndim != 2 or e.shape[0] != len(seq):
            bad_shape.append(f'{name}: got {tuple(e.shape)}, expected ({len(seq)}, D)')
            continue
        dims.add(e.shape[1])
        if torch.load(path, map_location='cpu', weights_only=True).dtype != torch.float32:
            bad_dtype.append(name)
        if not torch.isfinite(e).all():
            nonfinite.append(name)
        # make_embeddings.py:68 silently rewrites NaN to 0.0, so an all-zero row
        # is the fingerprint of a forward pass that failed without saying so.
        rz = int((e.abs().sum(dim=1) == 0).sum())
        if rz:
            zero_rows.append(f'{name}: {rz}/{len(seq)} rows all zero')

        # Only pool a finite mean. One NaN otherwise poisons the aggregate, and
        # every comparison against NaN is False -- so `rel > tol` below would
        # silently PASS a directory whose norms are three orders of magnitude
        # wrong. That is the exact shape of the bug this script exists to catch.
        n = float(e.norm(dim=1).mean())
        if np.isfinite(n):
            norms.append(n)

        # Token index b is the first token of the second window; subtract 1 for
        # BOS to get the residue index.
        b = CHUNK_TOKENS - 1
        if len(seq) > b + 8:
            en = torch.nn.functional.normalize(e, dim=1)
            cos_boundary.append(float((en[b - 1] * en[b]).sum()))
            c = len(seq) // 2 if abs(len(seq) // 2 - b) > 8 else b // 2
            cos_control.append(float((en[c - 1] * en[c]).sum()))

    fail = []
    print()
    if missing:
        fail.append(f'{len(missing)} sequences have no .pt file')
        print(f'MISSING      {len(missing)}   e.g. {missing[:4]}')
    if unreadable:
        fail.append(f'{len(unreadable)} files could not be loaded')
        print(f'UNREADABLE   {len(unreadable)}   e.g. {unreadable[:2]}')
    if bad_shape:
        fail.append(f'{len(bad_shape)} wrong-shaped embeddings')
        print(f'BAD SHAPE    {len(bad_shape)}   e.g. {bad_shape[:2]}')
    if bad_dtype:
        fail.append(f'{len(bad_dtype)} embeddings are not float32')
        print(f'BAD DTYPE    {len(bad_dtype)}   e.g. {bad_dtype[:4]}')
    if nonfinite:
        fail.append(f'{len(nonfinite)} contain NaN or Inf')
        print(f'NON-FINITE   {len(nonfinite)}   e.g. {nonfinite[:4]}')
    if zero_rows:
        fail.append(f'{len(zero_rows)} contain all-zero rows')
        print(f'ZERO ROWS    {len(zero_rows)}   e.g. {zero_rows[:2]}')

    if len(dims) > 1:
        fail.append(f'mixed feature dimensions {sorted(dims)}')
        print(f'MIXED DIMS   {sorted(dims)}')
    elif dims:
        d = dims.pop()
        print(f'dimension    {d}')
        if args.expect_dim and d != args.expect_dim:
            fail.append(f'dimension {d}, expected {args.expect_dim}')
            print(f'  EXPECTED   {args.expect_dim}')

    if norms:
        a = np.array(norms)
        print(f'per-residue L2 norm   mean {a.mean():.3f}   '
              f'p1 {np.percentile(a,1):.3f}   p99 {np.percentile(a,99):.3f}')
        # The mean alone cannot see a small number of badly scaled files: at 8061
        # sequences, one written at 900x moves the mean by 11%, inside the
        # tolerance. Compare each sequence against the median instead, which is
        # how the 1190 bfloat16 files would have been caught on the day.
        med = np.median(a)
        outliers = int(((a > 3 * med) | (a < med / 3)).sum())
        if outliers:
            fail.append(f'{outliers} sequences have a norm >3x off the median {med:.3f}')
            print(f'  OUTLIERS   {outliers} sequences more than 3x from the median '
                  f'{med:.3f} -- a subset was written differently from the rest')
        if args.expect_norm:
            rel = abs(a.mean() - args.expect_norm) / args.expect_norm
            if rel > args.norm_tol:
                fail.append(f'mean norm {a.mean():.3f} vs expected {args.expect_norm} '
                            f'({rel:.1%} off)')
                print(f'  EXPECTED   {args.expect_norm} '
                      f'-- {rel:.1%} off, check the final LayerNorm')
            else:
                print(f'  matches expected {args.expect_norm} ({rel:.1%} off)')

    if cos_boundary:
        b, c = np.mean(cos_boundary), np.mean(cos_control)
        print(f'\nwindow-boundary discontinuity ({len(cos_boundary)} long sequences)')
        print(f'  adjacent-residue cosine at token {CHUNK_TOKENS}: {b:.4f}')
        print(f'  adjacent-residue cosine mid-sequence:      {c:.4f}')
        # Not a failure. Upstream's extractor is expected to be discontinuous
        # here; this quantifies it so the thesis can state the size rather than
        # hand-wave, and so a future change to the windowing is visible.
        print(f'  -> drop of {c - b:+.4f}. Expected and reproduced from upstream; '
              f'informational only.')

    print()
    if fail:
        print('FAILED:')
        for f in fail:
            print(f'  - {f}')
        sys.exit(1)
    print(f'OK -- {len(items) - len(missing)} embeddings verified.')


if __name__ == '__main__':
    main()
