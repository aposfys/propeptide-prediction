'''
Generate ESM-2 embeddings (per position), one file per sequence, keyed by the
md5 of the sequence. Adapted from DeepTMHMM.

Copied from `main` and RENAMED. On this branch `make_embeddings.py` is the ESM3
extractor, so the two cannot share a name -- that is precisely how two arms end
up trained on embeddings nobody can identify afterwards.

The docstring upstream said ESM-1b; the code loads `esm2_t33_650M_UR50D`, which
is ESM-2 650M, and that is what every ESM-2 number in this study was produced
with. The comment was stale, not the code.
'''
from hashlib import md5
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, FastaBatchedDataset
import torch
import os
import argparse
import pathlib

def hash_aa_string(string):
    return md5(string.encode()).digest().hex()

from tqdm.auto import tqdm
def generate_esm_embeddings(fasta_file, esm_embeddings_dir, repr_layers=33):
    esm_model, esm_alphabet = pretrained.load_model_and_alphabet('esm2_t33_650M_UR50D') # esm1b_t33_650M_UR50S

    dataset = FastaBatchedDataset.from_file(fasta_file)
    
    with torch.no_grad():
        if torch.cuda.is_available():
            esm_model = esm_model.cuda()

        batch_converter = esm_alphabet.get_batch_converter()
        
        print("Starting to generate embeddings")

            
        for idx, item in enumerate(tqdm(dataset)):
            
            label, seq = item
            
            if os.path.isfile(f'{esm_embeddings_dir}/{hash_aa_string(seq)}.pt'):
                print("Already processed sequence")
                continue
                                
            
            seqs = list([("seq", s) for s in [seq]])
            labels, strs, toks = batch_converter(seqs)

            repr_layers_list = [
                (i + esm_model.num_layers + 1) % (esm_model.num_layers + 1) for i in range(repr_layers+1)
            ]

            out = None

            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)

            minibatch_max_length = toks.size(1)

            tokens_list = []
            end = 0
            while end <= minibatch_max_length:
                start = end
                end = start + 1022
                if end <= minibatch_max_length:
                    # we are not on the last one, so make this shorter
                    end = end - 300
                tokens = esm_model(toks[:, start:end], repr_layers=repr_layers_list, return_contacts=False)["representations"][33]
                tokens_list.append(tokens)

            out = torch.cat(tokens_list, dim=1).cpu()

            # set nan to zeros
            out[out!=out] = 0.0

            res = out.transpose(0,1)[1:-1] 
            seq_embedding = res[:,0]
            #print(seq_embedding.size())

            output_file = open(f'{esm_embeddings_dir}/{hash_aa_string(seq)}.pt', 'wb')
            torch.save(seq_embedding, output_file)
            output_file.close()

            #print(f"Saved embedding to {esm_embeddings_dir}")


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