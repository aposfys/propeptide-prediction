'''
Generate ESM-2 (esm2_t33_650M_UR50D) per-residue embeddings via HuggingFace
transformers. Saves one (seq_len, 1280) float32 tensor per sequence, named
by MD5 hash of the sequence — identical convention to the original script.
'''
from hashlib import md5
import os
import argparse
import pathlib

import torch
from tqdm.auto import tqdm


MODEL_NAME = 'facebook/esm2_t33_650M_UR50D'
MAX_LEN = 1022   # ESM-2 positional limit (excl. BOS/EOS)
OVERLAP  = 200   # overlap between chunks for long sequences


def hash_aa_string(s):
    return md5(s.encode()).digest().hex()


def read_fasta(fasta_file):
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


def _embed_batch(seqs, model, tokenizer, device):
    '''Embed a list of sequences (all <= MAX_LEN). Returns list of (seq_len, 1280) tensors.'''
    enc = tokenizer(
        seqs,
        return_tensors='pt',
        padding=True,
        truncation=False,
        add_special_tokens=True,
    ).to(device)
    with torch.no_grad():
        out = model(**enc).last_hidden_state  # (B, padded+2, 1280)
    results = []
    for i, seq in enumerate(seqs):
        emb = out[i, 1:len(seq)+1, :].float().cpu()
        emb[emb != emb] = 0.0
        results.append(emb)
    return results


def _embed_long(seq, model, tokenizer, device):
    '''Sliding-window embedding for sequences longer than MAX_LEN.'''
    chunks, starts = [], []
    start = 0
    while start < len(seq):
        end = min(start + MAX_LEN, len(seq))
        chunks.append(seq[start:end])
        starts.append(start)
        if end == len(seq):
            break
        start = end - OVERLAP

    emb_full = torch.zeros(len(seq), 1280)
    counts   = torch.zeros(len(seq), 1)
    for chunk, s in zip(chunks, starts):
        enc = tokenizer(chunk, return_tensors='pt', add_special_tokens=True).to(device)
        with torch.no_grad():
            out = model(**enc).last_hidden_state[0, 1:len(chunk)+1, :].float().cpu()
        out[out != out] = 0.0
        emb_full[s:s+len(chunk)] += out
        counts[s:s+len(chunk)]   += 1
    return emb_full / counts.clamp(min=1)


def generate_esm_embeddings(fasta_file, output_dir, batch_size=16):
    from transformers import EsmModel, EsmTokenizer

    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))
    print(f'Using device: {device}')
    print(f'Loading {MODEL_NAME}...')
    tokenizer = EsmTokenizer.from_pretrained(MODEL_NAME)
    model = EsmModel.from_pretrained(MODEL_NAME).to(device).eval()

    sequences = list(read_fasta(fasta_file))
    seen, unique = set(), []
    for label, seq in sequences:
        h = hash_aa_string(seq)
        if h not in seen:
            seen.add(h)
            unique.append((label, seq, h))
    print(f'{len(unique)} unique sequences ({len(sequences)} total)')

    short = [(lab, seq, h) for lab, seq, h in unique if len(seq) <= MAX_LEN]
    long  = [(lab, seq, h) for lab, seq, h in unique if len(seq) >  MAX_LEN]
    print(f'Short (<={MAX_LEN}): {len(short)}   Long (>{MAX_LEN}): {len(long)}')

    # --- short sequences: batch ---
    pending = [(lab, seq, h) for lab, seq, h in short
               if not os.path.isfile(os.path.join(output_dir, f'{h}.pt'))]
    pending.sort(key=lambda x: len(x[1]))
    print(f'Short to embed: {len(pending)}')

    for i in tqdm(range(0, len(pending), batch_size)):
        batch = pending[i:i+batch_size]
        seqs_b = [seq for _, seq, _ in batch]
        embs   = _embed_batch(seqs_b, model, tokenizer, device)
        for (_, _, h), emb in zip(batch, embs):
            torch.save(emb, os.path.join(output_dir, f'{h}.pt'))

    # --- long sequences: one at a time ---
    pending_long = [(lab, seq, h) for lab, seq, h in long
                    if not os.path.isfile(os.path.join(output_dir, f'{h}.pt'))]
    print(f'Long to embed: {len(pending_long)}')
    for _, seq, h in tqdm(pending_long):
        emb = _embed_long(seq, model, tokenizer, device)
        torch.save(emb, os.path.join(output_dir, f'{h}.pt'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('fasta_file', type=pathlib.Path)
    p.add_argument('output_dir', type=pathlib.Path)
    p.add_argument('--batch_size', type=int, default=16)
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    generate_esm_embeddings(args.fasta_file, args.output_dir, args.batch_size)


if __name__ == '__main__':
    main()
