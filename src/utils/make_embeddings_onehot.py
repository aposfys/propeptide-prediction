'''
Generate one-hot token embeddings using the ESM3 sequence tokenizer vocabulary
and save as one file per sequence (MD5 hash as filename).

This produces a (L, vocab_size) float32 tensor per sequence — the same shape
convention as the ESM3 embeddings from make_embeddings.py — so the downstream
dataset/model code can be reused without modification by setting embedding_dim
to the ESM3 tokenizer vocab size.

NOTE: This script is not used in the standard ESM3 propeptide pipeline.
It is kept for experimental comparison with the full ESM3 embeddings.
'''
from hashlib import md5
import torch
import os
import argparse
import pathlib
from tqdm.auto import tqdm


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


def generate_onehot_embeddings(fasta_file, output_dir):
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein

    esm_model = ESM3.from_pretrained('esm3_sm_open_v1').eval()
    vocab_size = esm_model.tokenizers.sequence.vocab_size

    dataset = _read_fasta(fasta_file)
    print(f'  {len(dataset)} unique sequences to embed (one-hot, vocab_size={vocab_size})')

    with torch.no_grad():
        if torch.cuda.is_available():
            esm_model = esm_model.cuda()

        for label, seq in tqdm(dataset):
            out_path = os.path.join(output_dir, f'{hash_aa_string(seq)}.pt')
            if os.path.isfile(out_path):
                continue

            protein = ESMProtein(sequence=seq)
            encoded = esm_model.encode(protein)

            # encoded.sequence: 1D LongTensor of shape [L+2] (includes CLS and EOS)
            toks = encoded.sequence.cpu()
            seq_embedding = torch.nn.functional.one_hot(
                toks.long(), num_classes=vocab_size
            ).float()[1:-1]  # strip CLS/EOS → (L, vocab_size)

            torch.save(seq_embedding, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('fasta_file', type=pathlib.Path,
                        help='FASTA file to embed')
    parser.add_argument('output_dir', type=pathlib.Path,
                        help='Output directory for one-hot .pt files')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    generate_onehot_embeddings(args.fasta_file, args.output_dir)


if __name__ == '__main__':
    main()
