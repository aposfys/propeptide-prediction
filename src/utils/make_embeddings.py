'''
Generate ESM-2 (esm2_t33_650M_UR50D) per-residue embeddings via fair-esm and save
one file per sequence, named by the MD5 hash of the sequence.

Faithful to the original DeepPeptide embedder (same model, layer 33, and 1022-residue
sliding window for long sequences). The only change vs the original is device
auto-selection (cuda -> mps -> cpu) so it runs off-GPU, plus a `.contiguous()` before
saving to avoid writing the whole batch storage to disk.
Adapted from DeepTMHMM.
'''
from hashlib import md5
from esm import pretrained
import torch
import os
import argparse
import pathlib


def hash_aa_string(string):
    return md5(string.encode()).digest().hex()


def _select_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


from tqdm.auto import tqdm
def generate_esm_embeddings(fasta_file, esm_embeddings_dir, repr_layers=33):
    from esm import FastaBatchedDataset

    device = _select_device()
    print(f'Loading ESM-2 (esm2_t33_650M_UR50D) on {device} ...')
    esm_model, esm_alphabet = pretrained.load_model_and_alphabet('esm2_t33_650M_UR50D')
    esm_model = esm_model.to(device).eval()
    batch_converter = esm_alphabet.get_batch_converter()

    dataset = FastaBatchedDataset.from_file(fasta_file)

    with torch.no_grad():
        print('Starting to generate embeddings')
        for label, seq in tqdm(dataset):
            out_path = os.path.join(esm_embeddings_dir, f'{hash_aa_string(seq)}.pt')
            if os.path.isfile(out_path):
                continue

            _, _, toks = batch_converter([('seq', seq)])
            toks = toks.to(device)

            # Window sequences longer than 1022 residues (esm2 positional limit), with
            # a 300-residue overlap between windows — identical to the original.
            minibatch_max_length = toks.size(1)
            tokens_list = []
            end = 0
            while end <= minibatch_max_length:
                start = end
                end = start + 1022
                if end <= minibatch_max_length:
                    end = end - 300
                rep = esm_model(toks[:, start:end], repr_layers=[repr_layers], return_contacts=False)
                tokens_list.append(rep['representations'][repr_layers])

            out = torch.cat(tokens_list, dim=1).cpu()
            out[out != out] = 0.0                 # nan -> 0
            seq_embedding = out[0, 1:-1]          # strip BOS/EOS -> (L, 1280)
            torch.save(seq_embedding.contiguous(), out_path)


def main():
    parser = argparse.ArgumentParser(
        description='Generate ESM-2 per-residue embeddings (1280-dim) with fair-esm. '
                    'Output files use MD5-hash naming; pass --embedding_dim 1280 to training.'
    )
    parser.add_argument('fasta_file', type=pathlib.Path,
                        help='FASTA file on which to extract representations')
    parser.add_argument('output_dir', type=pathlib.Path,
                        help='output directory for extracted representations')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    generate_esm_embeddings(args.fasta_file, args.output_dir, repr_layers=33)


if __name__ == '__main__':
    main()
