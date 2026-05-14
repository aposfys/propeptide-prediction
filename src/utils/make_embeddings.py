'''
Generate ESM3 embeddings (per position) and save as one
file per sequence. Use md5 hash of sequence as file name.
Adapted from DeepTMHMM.
'''
from hashlib import md5
import torch
import os
import argparse
import pathlib

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


from tqdm.auto import tqdm
def generate_esm_embeddings(fasta_file, esm_embeddings_dir, repr_layers=33):
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein

    esm_model = ESM3.from_pretrained('esm3_sm_open_v1').eval()

    dataset = _read_fasta(fasta_file)
    print(f'  {len(dataset)} unique sequences to embed')

    with torch.no_grad():
        if torch.cuda.is_available():
            esm_model = esm_model.cuda()

        print('Starting to generate embeddings')

        for label, seq in tqdm(dataset):
            out_path = os.path.join(esm_embeddings_dir, f'{hash_aa_string(seq)}.pt')
            if os.path.isfile(out_path):
                continue

            protein = ESMProtein(sequence=seq)
            encoded = esm_model.encode(protein)

            out = esm_model(
                sequence_tokens=encoded.sequence.unsqueeze(0),
            )

            seq_embedding = out.embeddings[0, 1:-1].cpu()  # strip CLS/EOS
            torch.save(seq_embedding, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fasta_file",
        type=pathlib.Path,
        help="FASTA file on which to extract representations",
    )
    parser.add_argument(
        "output_dir",
        type=pathlib.Path,
        help="output directory for extracted representations",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)


    generate_esm_embeddings(args.fasta_file, args.output_dir, repr_layers=33)

if __name__ == '__main__':
    main()