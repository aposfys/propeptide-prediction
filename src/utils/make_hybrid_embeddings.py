'''
Add a structural channel to an embedding set that already exists.

Written for one specific finding: if the bar is "beat the best arm on this
benchmark", then adding structure to a representation that is LOSING by 0.09 is
the wrong move. ESM-2 at T4 scores 0.6153. A resolvable win at 8 replicates
needs roughly 0.650, which is +0.127 from ESM3 and +0.131 from ProstT5 -- larger
than any effect measured anywhere in this study -- but only **+0.035** from
ESM-2 itself. So the structural channel should be bolted onto the arm that is
already winning.

This concatenates a per-residue structural channel onto any existing embedding
directory. Both sides are keyed by the md5 of the sequence, as every extractor
here keys them, so no alignment step is needed and none is invented.

TWO STRUCTURAL CHANNELS, and the small one is the better experiment
    --mode onehot   20 dimensions, the 3Di letter as a one-hot vector. Adds
                    1.6% to a 1280-dim input, so the head barely changes size
                    and a gain cannot be dismissed as extra capacity. This is
                    the structural analogue of the one-hot amino-acid control
                    already in this study, and it is the cleanest available test
                    of whether the structural ALPHABET carries anything.
    --mode prostt5  1024 dimensions, ProstT5's own encoding of the 3Di string,
                    from make_embeddings_prost5_struct.py --tracks 3di. More
                    information, but it widens conv1 by 78% on a 1280-dim input,
                    so a gain is confounded with capacity and needs the shuffled
                    control to interpret.

Run both. If onehot moves the number, the finding is clean and cheap. If only
prostt5 moves it, the extra capacity is a live alternative explanation.

WHY NOT ELEMENT-WISE FUSION HERE
FUSION.md offers sum/mean/renorm for the ProstT5 arms because both channels come
out of the SAME encoder and share a vector space. ESM-2 and ProstT5 do not.
Adding vectors from two different models is not meaningful, so concatenation is
the only option offered, and its cost in parameters is reported rather than
hidden.

MISSING STRUCTURE
Proteins with no usable AlphaFold model get a zero structural block and keep
their sequence channel, the same policy as every other structure arm here, so
the GraphPart partitions are untouched. --shuffle is the control: it pairs each
protein with another protein's 3Di, keeping the dimensionality and the zero-mask
pattern and destroying only the correspondence.

Usage
-----
    python -m src.utils.make_hybrid_embeddings \
        --base_dir /path/to/embeddings/esm2 \
        --three_di three_di/three_di.json \
        --out_dir embeddings/esm2_3di --mode onehot
'''
import argparse
import json
import os
import random
from hashlib import md5

import pandas as pd
import torch
from tqdm.auto import tqdm

# Foldseek's 3Di alphabet. Fixed order so a rerun produces identical columns.
THREE_DI_ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
INDEX = {letter: i for i, letter in enumerate(THREE_DI_ALPHABET)}


def hash_aa_string(string: str) -> str:
    return md5(string.encode()).digest().hex()


def onehot_3di(three_di: str) -> torch.Tensor:
    '''(L, 20) one-hot. An unrecognised letter yields an all-zero row rather
    than an arbitrary column, which keeps it distinguishable from a real state.'''
    out = torch.zeros(len(three_di), len(THREE_DI_ALPHABET), dtype=torch.float32)
    for position, letter in enumerate(three_di.upper()):
        column = INDEX.get(letter)
        if column is not None:
            out[position, column] = 1.0
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base_dir', required=True,
                        help='Existing per-residue embeddings, e.g. the ESM-2 set.')
    parser.add_argument('--out_dir', required=True, help='Fresh output directory.')
    parser.add_argument('--three_di', required=True,
                        help='three_di.json from make_3di.py.')
    parser.add_argument('--mode', choices=['onehot', 'prostt5'], default='onehot')
    parser.add_argument('--prostt5_dir', default='',
                        help='Required for --mode prostt5: a 1024-dim directory '
                             'from make_embeddings_prost5_struct.py --tracks 3di.')
    parser.add_argument('--data_file', default='data/labeled_sequences.csv')
    parser.add_argument('--shuffle', action='store_true',
                        help='CONTROL: pair each protein with another protein\'s '
                             '3Di. Same dimensionality, same mask pattern, no real '
                             'correspondence.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.mode == 'prostt5' and not args.prostt5_dir:
        raise SystemExit('--mode prostt5 needs --prostt5_dir')

    three_di_by_hash = json.load(open(args.three_di))
    if args.shuffle:
        rng = random.Random(args.seed)
        keys = [k for k, v in three_di_by_hash.items() if v]
        values = [three_di_by_hash[k] for k in keys]
        rng.shuffle(values)
        three_di_by_hash = dict(three_di_by_hash)
        three_di_by_hash.update(dict(zip(keys, values)))
        print(f'--shuffle: {len(keys)} 3Di strings permuted with seed {args.seed}. '
              'CONTROL arm.')

    frame = pd.read_csv(args.data_file)
    sequences = {hash_aa_string(s): s for s in frame['sequence'].astype(str)}
    os.makedirs(args.out_dir, exist_ok=True)

    n_written = n_masked = 0
    base_dim = struct_dim = None

    for digest, sequence in tqdm(sorted(sequences.items())):
        out_path = os.path.join(args.out_dir, f'{digest}.pt')
        if os.path.isfile(out_path):
            n_written += 1
            continue
        base_path = os.path.join(args.base_dir, f'{digest}.pt')
        if not os.path.isfile(base_path):
            continue
        base = torch.load(base_path).to(torch.float32)
        if base.shape[0] != len(sequence):
            raise RuntimeError(
                f'{digest}: base embedding is {base.shape[0]} rows for a '
                f'{len(sequence)}-residue sequence. Refusing to concatenate onto '
                'a misaligned tensor.')
        base_dim = base.shape[1]

        three_di = three_di_by_hash.get(digest)
        if three_di and len(three_di) != len(sequence):
            three_di = None          # stale or shuffled to the wrong length

        if args.mode == 'onehot':
            struct_dim = len(THREE_DI_ALPHABET)
            block = (onehot_3di(three_di) if three_di
                     else torch.zeros(len(sequence), struct_dim))
        else:
            struct_dim = 1024
            path = os.path.join(args.prostt5_dir, f'{digest}.pt')
            if three_di and os.path.isfile(path):
                block = torch.load(path).to(torch.float32)
                if block.shape[0] != len(sequence):
                    block = torch.zeros(len(sequence), struct_dim)
                    three_di = None
            else:
                block = torch.zeros(len(sequence), struct_dim)
                three_di = None

        if not three_di:
            n_masked += 1
        torch.save(torch.cat([base, block], dim=-1), out_path)
        n_written += 1

    total = (base_dim or 0) + (struct_dim or 0)
    json.dump({'base_dir': args.base_dir, 'mode': args.mode,
               'prostt5_dir': args.prostt5_dir or None, 'shuffle': args.shuffle,
               'seed': args.seed if args.shuffle else None,
               'base_dim': base_dim, 'struct_dim': struct_dim,
               'embedding_dim': total},
              open(os.path.join(args.out_dir, 'extraction_config.json'), 'w'), indent=2)

    print(f'\n{n_written} embeddings in {args.out_dir}')
    print(f'{base_dim} + {struct_dim} = {total} dims. Pass --embedding_dim {total}.')
    print(f'{n_masked} have a zero structural block (no usable structure).')
    if struct_dim and base_dim:
        print(f'conv1 grows by {100*struct_dim/base_dim:.1f}%; everything downstream '
              'is unchanged.')


if __name__ == '__main__':
    main()
