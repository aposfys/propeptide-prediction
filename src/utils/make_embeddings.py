'''
Generate ESM-2 embeddings (per position) and save as one
file per sequence. Use md5 hash of sequence as file name.
Adapted from DeepTMHMM.
'''
from hashlib import md5
from esm import pretrained
import torch
import os
import argparse
import pathlib
from tqdm.auto import tqdm


def hash_aa_string(string):
    return md5(string.encode()).digest().hex()


def read_fasta(fasta_file):
    '''Yield (label, sequence) pairs from a FASTA file.'''
    label, seq = None, []
    with open(fasta_file) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('>'):
                if label is not None:
                    yield label, ''.join(seq)
                label = line[1:]
                seq = []
            else:
                seq.append(line)
    if label is not None:
        yield label, ''.join(seq)


def _embed_long_sequence(esm_model, batch_converter, seq, repr_layer, device):
    '''Sliding-window embedding for sequences longer than 1022 tokens.'''
    _, _, toks = batch_converter([('seq', seq)])
    toks = toks.to(device)
    L = toks.size(1)

    tokens_list = []
    end = 0
    while end <= L:
        start = end
        end = start + 1022
        if end <= L:
            end = end - 300
        chunk = esm_model(toks[:, start:end], repr_layers=[repr_layer], return_contacts=False)['representations'][repr_layer]
        tokens_list.append(chunk)

    out = torch.cat(tokens_list, dim=1).cpu()  # (1, L, hidden)
    out[out != out] = 0.0
    # strip BOS/EOS, squeeze batch dim: (seq_len, hidden)
    return out[0, 1:len(seq)+1, :]


def generate_esm_embeddings(fasta_file, esm_embeddings_dir, repr_layer=33, batch_size=32):
    esm_model, esm_alphabet = pretrained.load_model_and_alphabet('esm2_t33_650M_UR50D')

    if torch.cuda.is_available():
        _device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        _device = torch.device('mps')
    else:
        _device = torch.device('cpu')
    print(f'Using device: {_device}')

    sequences = list(read_fasta(fasta_file))
    seen = set()
    unique_sequences = []
    for label, seq in sequences:
        h = hash_aa_string(seq)
        if h not in seen:
            seen.add(h)
            unique_sequences.append((label, seq, h))
    print(f'{len(unique_sequences)} unique sequences (out of {len(sequences)} total)')

    batch_converter = esm_alphabet.get_batch_converter()

    short = [(lab, seq, h) for lab, seq, h in unique_sequences if len(seq) <= 1020]
    long  = [(lab, seq, h) for lab, seq, h in unique_sequences if len(seq) >  1020]
    print(f'Short (batchable): {len(short)}, Long (sliding window): {len(long)}')

    with torch.no_grad():
        esm_model = esm_model.to(_device)

        pending = [(lab, seq, h) for lab, seq, h in short
                   if not os.path.isfile(f'{esm_embeddings_dir}/{h}.pt')]
        # sort by length so sequences of similar length are batched together (minimizes padding waste)
        pending.sort(key=lambda x: len(x[1]))
        print(f'{len(pending)} short sequences to embed')

        for i in tqdm(range(0, len(pending), batch_size)):
            batch = pending[i:i+batch_size]
            pairs = [('seq', seq) for _, seq, _ in batch]
            _, _, toks = batch_converter(pairs)
            toks = toks.to(_device)
            out = esm_model(toks, repr_layers=[repr_layer], return_contacts=False)['representations'][repr_layer]
            out = out.cpu()
            out[out != out] = 0.0
            # out shape: (B, padded_len, hidden) — BOS at 0, EOS at len(seq)+1
            for j, (_, seq, h) in enumerate(batch):
                emb = out[j, 1:len(seq)+1, :].contiguous()  # (seq_len, hidden) — contiguous avoids saving full batch storage
                torch.save(emb, open(f'{esm_embeddings_dir}/{h}.pt', 'wb'))

        pending_long = [(lab, seq, h) for lab, seq, h in long
                        if not os.path.isfile(f'{esm_embeddings_dir}/{h}.pt')]
        print(f'{len(pending_long)} long sequences to embed')
        for _, seq, h in tqdm(pending_long):
            emb = _embed_long_sequence(esm_model, batch_converter, seq, repr_layer, _device)
            torch.save(emb, open(f'{esm_embeddings_dir}/{h}.pt', 'wb'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('fasta_file', type=pathlib.Path)
    parser.add_argument('output_dir', type=pathlib.Path)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    generate_esm_embeddings(args.fasta_file, args.output_dir, batch_size=args.batch_size)

if __name__ == '__main__':
    main()
