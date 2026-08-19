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
def generate_esm_embeddings(fasta_file, esm_embeddings_dir):
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein

    esm_model = ESM3.from_pretrained('esm3_sm_open_v1').eval()

    dataset = _read_fasta(fasta_file)
    print(f'  {len(dataset)} unique sequences to embed')

    with torch.no_grad():
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        esm_model = esm_model.to(device)
        print(f'Embedding on {device}')

        print('Starting to generate embeddings')

        for label, seq in tqdm(dataset):
            out_path = os.path.join(esm_embeddings_dir, f'{hash_aa_string(seq)}.pt')
            if os.path.isfile(out_path):
                continue

            protein = ESMProtein(sequence=seq)
            encoded = esm_model.encode(protein)

            # .to(device) is required: the tokens come back on CPU, and feeding
            # them to a CUDA model raises a device-mismatch RuntimeError.
            out = esm_model(
                sequence_tokens=encoded.sequence.unsqueeze(0).to(device),
            )

            # Apply ESM3's own final LayerNorm. ESMOutput.embeddings is NOT the
            # tensor the model's heads consume: TransformerStack.forward returns
            # `self.norm(x), x, hiddens` and ESM3.forward unpacks it as
            # `x, embedding, _`, so `.embeddings` is the raw pre-norm residual
            # stream. Measured per-token L2 norm is ~9800 for the raw stream vs
            # ~11.6 after the norm — and ~10.1 for the ESM-2 L33 representations
            # this pipeline was built around (fair-esm applies emb_layer_norm_after
            # and overwrites representations[33] with the normalised tensor).
            # Feeding the raw stream to LSTMCNN, which has no input normalisation,
            # saturates ~91% of the biLSTM gates at init.
            # The norm is per-token over the feature dim, so it commutes with the
            # BOS/EOS slice below.
            seq_embedding = esm_model.transformer.norm(out.embeddings)[0, 1:-1].cpu()
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


    generate_esm_embeddings(args.fasta_file, args.output_dir)

if __name__ == '__main__':
    main()