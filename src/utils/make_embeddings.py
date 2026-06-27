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


MAX_LEN = 1022  # ESM3 sequence positional limit (excl. BOS/EOS)


def _select_device(name='auto'):
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


from tqdm.auto import tqdm
def generate_esm_embeddings(fasta_file, esm_embeddings_dir, device='auto'):
    '''Embed sequences with ESM3 (esm3_sm_open_v1, HuggingFace open weights).
    Saves one (L, 1536) float32 tensor per sequence, named by MD5 hash of the sequence.
    Sequences longer than MAX_LEN are skipped (handle separately if full coverage is needed).'''
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein

    device = _select_device(device)
    print(f'Loading ESM3 (esm3_sm_open_v1) on {device} ...')
    esm_model = ESM3.from_pretrained('esm3_sm_open_v1').to(device).eval()

    dataset = _read_fasta(fasta_file)
    print(f'  {len(dataset)} unique sequences to embed')

    n_done, n_skip_long, n_mismatch = 0, 0, 0
    with torch.no_grad():
        print('Starting to generate embeddings')

        for label, seq in tqdm(dataset):
            out_path = os.path.join(esm_embeddings_dir, f'{hash_aa_string(seq)}.pt')
            if os.path.isfile(out_path):
                continue
            if len(seq) > MAX_LEN:
                n_skip_long += 1
                continue

            protein = ESMProtein(sequence=seq)
            encoded = esm_model.encode(protein)
            seq_tokens = encoded.sequence.unsqueeze(0).to(device)

            out = esm_model(sequence_tokens=seq_tokens)

            seq_embedding = out.embeddings[0, 1:-1].float().cpu()  # strip BOS/EOS -> (L, 1536)
            if seq_embedding.shape[0] != len(seq):
                n_mismatch += 1
                print(f'  length mismatch {hash_aa_string(seq)}: emb {seq_embedding.shape[0]} vs seq {len(seq)} — skipping')
                continue

            torch.save(seq_embedding.contiguous(), out_path)  # .contiguous() avoids saving full batch storage
            n_done += 1

    print(f'Done. wrote {n_done} new embeddings; skipped {n_skip_long} seqs > {MAX_LEN} residues; {n_mismatch} length mismatches.')


def main():
    parser = argparse.ArgumentParser(
        description='Generate ESM3 per-residue embeddings (1536-dim). '
                    'Output files use MD5-hash naming. '
                    'Pass --embedding_dim 1536 to the training script.'
    )
    parser.add_argument(
        'fasta_file',
        type=pathlib.Path,
        help='FASTA file on which to extract representations',
    )
    parser.add_argument(
        'output_dir',
        type=pathlib.Path,
        help='output directory for extracted representations',
    )
    parser.add_argument(
        '--device', default='auto', choices=['auto', 'cpu', 'mps', 'cuda'],
        help='compute device for embedding generation (default: auto -> cuda/mps/cpu)',
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    generate_esm_embeddings(args.fasta_file, args.output_dir, device=args.device)


if __name__ == '__main__':
    main()
