#!/usr/bin/env python3
'''Generate one-hot amino-acid "embeddings", one file per sequence.

The no-pretraining control arm. Same output contract as make_embeddings.py --
a [L, D] float32 tensor per sequence, torch.save'd to <md5(sequence)>.pt -- so
the dataset loader, the training loop and verify_embeddings.py all consume it
unchanged. The only difference the model sees is D = 21 instead of 1280/1536/1024.

Why this arm exists. The DeepPeptide head is not a linear probe: LSTMCNN's
convolution, bidirectional LSTM and CRF together carry enough capacity to learn a
substantial amount of the task from residue identity alone. Hewitt and Liang
(EMNLP 2019) show that a high-capacity probe confounds "the representation
encodes this property" with "the probe can learn this property", and that the
interpretable quantity is performance measured against a control that carries no
pretrained information. Without this arm the experiment can rank pretrained
representations against each other but cannot say what pretraining bought.

The hashing and FASTA parsing are copied verbatim from make_embeddings.py rather
than imported, because that module imports the ESM3 SDK at call time and this
script must run on a machine with no GPU and no fair-esm/esm install.
'''

from hashlib import md5
import argparse
import os
import sys

import torch

# 20 canonical residues plus one catch-all column. The catch-all matters: with a
# bare 20-letter alphabet every B/Z/J/U/O/X residue would encode as an all-zero
# row, which is both a silent loss of the position and something
# verify_embeddings.py reports as corruption.
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
OTHER_INDEX = len(AMINO_ACIDS)
DIM = len(AMINO_ACIDS) + 1

INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


def hash_aa_string(string):
    return md5(string.encode()).digest().hex()


def _read_fasta(fasta_file):
    '''Parse FASTA, deduplicate by sequence hash, return list of (label, seq).'''
    sequences = {}
    label, seq_parts = None, []
    with open(fasta_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if label is not None and seq_parts:
                    seq = ''.join(seq_parts)
                    h = hash_aa_string(seq)
                    if h not in sequences:
                        sequences[h] = (label, seq)
                label = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
    if label is not None and seq_parts:
        seq = ''.join(seq_parts)
        h = hash_aa_string(seq)
        if h not in sequences:
            sequences[h] = (label, seq)
    return list(sequences.values())


def encode(seq):
    '''One-hot encode a sequence as [len(seq), DIM] float32.'''
    out = torch.zeros(len(seq), DIM, dtype=torch.float32)
    unknown = 0
    for pos, aa in enumerate(seq):
        idx = INDEX.get(aa)
        if idx is None:
            idx = OTHER_INDEX
            unknown += 1
        out[pos, idx] = 1.0
    return out, unknown


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('fasta_file', type=str,
                   help='FASTA to encode -- use the same deduplicated file the '
                        'pLM extractors were given, so the hashes line up')
    p.add_argument('output_dir', type=str,
                   help='output directory for the one-hot tensors')
    p.add_argument('--overwrite', action='store_true',
                   help='rewrite files that already exist (default: skip)')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dataset = _read_fasta(args.fasta_file)
    print('{} unique sequences to encode'.format(len(dataset)))

    written = skipped = 0
    total_residues = total_unknown = 0
    unknown_chars = {}

    for _label, seq in dataset:
        out_path = os.path.join(args.output_dir,
                                '{}.pt'.format(hash_aa_string(seq)))
        if os.path.isfile(out_path) and not args.overwrite:
            skipped += 1
            continue
        tensor, unknown = encode(seq)
        torch.save(tensor, out_path)
        written += 1
        total_residues += len(seq)
        total_unknown += unknown
        if unknown:
            for aa in seq:
                if aa not in INDEX:
                    unknown_chars[aa] = unknown_chars.get(aa, 0) + 1

    print('wrote {}, skipped {} already present'.format(written, skipped))
    print('dimension {} ({} canonical + 1 catch-all)'.format(DIM, len(AMINO_ACIDS)))
    if total_residues:
        pct = 100.0 * total_unknown / total_residues
        print('non-canonical residues: {}/{} ({:.4f}%)'.format(
            total_unknown, total_residues, pct))
        if unknown_chars:
            print('  ' + ', '.join(
                '{}:{}'.format(c, n) for c, n in sorted(unknown_chars.items())))
        # A large catch-all fraction would mean the control arm is encoding
        # something other than amino-acid identity, which would make it a bad
        # floor. Anything under a fraction of a percent is the usual UniProt
        # B/Z/X residue noise.
        if pct > 1.0:
            print('WARNING: over 1% of residues fell into the catch-all column',
                  file=sys.stderr)


if __name__ == '__main__':
    main()
