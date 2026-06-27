#!/usr/bin/env python3
'''
Write a copy of a labeled_sequences CSV keeping only rows whose embedding .pt file
exists in --embeddings_dir (files are named by the MD5 hash of the sequence).

Needed for the ESM3 branch: its embedder skips sequences > 1022 residues, so a few
sequences have no embedding and would otherwise crash training with FileNotFoundError.
(ESM-2 windows long sequences, so it has full coverage and does not need this.)

Usage:
  python filter_data_by_embeddings.py \
      --data_file data/labeled_sequences.csv \
      --embeddings_dir ~/embeddings/esm3 \
      --out data/labeled_sequences_esm3.csv
'''
import argparse
import glob
import os
from hashlib import md5

import pandas as pd


def hash_aa_string(s):
    return md5(str(s).encode()).digest().hex()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_file', required=True)
    p.add_argument('--embeddings_dir', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    df = pd.read_csv(args.data_file)
    present = {os.path.basename(x)[:-3] for x in glob.glob(os.path.join(args.embeddings_dir, '*.pt'))}
    keep = df['sequence'].astype(str).map(hash_aa_string).isin(present)
    df[keep].to_csv(args.out, index=False)
    print(f'kept {int(keep.sum())}/{len(df)} rows '
          f'({len(df) - int(keep.sum())} dropped: no embedding) -> {args.out}')


if __name__ == '__main__':
    main()
