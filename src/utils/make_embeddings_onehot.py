'''
Generate one-hot embeddings and save as one 
file per sequence. Use md5 hash of sequence as file name.
We do it this way so that we can just reuse the whole ESM-based setup without
any changes aside from the input dimension.

'''
from hashlib import md5
from esm.data import FastaBatchedDataset
from esm import pretrained
import torch
import os
import argparse
import pathlib

def hash_aa_string(string):
    return md5(string.encode()).digest().hex()

from tqdm.auto import tqdm
def generate_esm_embeddings(fasta_file, esm_embeddings_dir, repr_layers=33):
    from esm.models.esm3 import ESM3
    esm_model = ESM3.from_pretrained('esm3_sm_open_v1').eval()

    dataset = FastaBatchedDataset.from_file(fasta_file)
    
    with torch.no_grad():
        if torch.cuda.is_available():
            #torch.cuda.set_device(1)
            esm_model = esm_model.cuda()

        batch_converter = esm_alphabet.get_batch_converter()
        
        print("Starting to generate embeddings")

            
        for idx, item in enumerate(tqdm(dataset)):
            
            label, seq = item
            
            # if os.path.isfile(f'{esm_embeddings_dir}/{hash_aa_string(seq)}.pt'):
            #     print("Already processed sequence")
            #     continue
                                
            #print(f"Sequence length: {len(original_aa_string)}")
            
            
            # ESM3 encoding
            from esm.sdk.api import ESMProtein
            protein = ESMProtein(sequence=seq)
            encoded = esm_model.encode(protein)
            
            toks = encoded.sequence_tokens

            seq_embedding = torch.nn.functional.one_hot(toks, num_classes=esm_model.tokenizers.sequence.vocab_size) # (1, seq_len, dim)
            seq_embedding = seq_embedding[0][1:-1] 

            output_file = open(f'{esm_embeddings_dir}/{hash_aa_string(seq)}.pt', 'wb')
            torch.save(seq_embedding, output_file)
            output_file.close()


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