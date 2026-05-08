#!/usr/bin/env python3
"""
Minimal generator for per-residue ESM3 embeddings.
Reads sequences from a FASTA file and writes .pt files containing a
(seq_len, hidden_size) tensor for each sequence. Uses sliding windows
for sequences longer than ~1024 tokens.
"""
import os
import sys
import argparse
import hashlib
from pathlib import Path

import torch

# Limit CPU threads to 8 according to cluster guidelines
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
torch.set_num_threads(8)
torch.set_num_interop_threads(1)

try:
    from tqdm import tqdm  # progress bar
except ImportError:
    def tqdm(x, **kwargs): return x

try:
    from Bio import SeqIO
except ImportError:
    print("BioPython is required (pip install biopython)", file=sys.stderr)
    sys.exit(1)

try:
    from esm.models.esm3 import ESM3
    from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
except ImportError:
    print("esm package is required (pip install esm)", file=sys.stderr)
    sys.exit(1)


def md5(seq: str) -> str:
    return hashlib.md5(seq.encode('utf-8')).hexdigest()


def embed_windows(model, tokenizer, seq: str, window: int = 1022, overlap: int = 300) -> torch.Tensor:
    """Return per-residue embeddings for a protein sequence using sliding windows."""
    L = len(seq)
    # Infer embedding dimension from a tiny forward pass
    ids0 = tokenizer("A")["input_ids"]
    tok0 = torch.tensor(ids0, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        out0 = model(sequence_tokens=tok0)
    d_model = int(out0.embeddings.shape[-1])

    final = torch.zeros((L, d_model), dtype=torch.float32)
    counts = torch.zeros((L, 1), dtype=torch.float32)
    start = 0
    while start < L:
        end = min(start + window, L)
        sub = seq[start:end]
        try:
            ids = tokenizer(sub)["input_ids"]
            tokens = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
            with torch.no_grad():
                out = model(sequence_tokens=tokens)
            emb = out.embeddings[0, 1:-1, :].cpu().float()
            # adjust length
            if emb.shape[0] > len(sub):
                emb = emb[:len(sub)]
            elif emb.shape[0] < len(sub):
                pad = torch.zeros((len(sub) - emb.shape[0], d_model), dtype=torch.float32)
                emb = torch.cat([emb, pad], dim=0)
            final[start:end] += emb
            counts[start:end] += 1
        except Exception as e:
            print(f"Warning: failed window [{start}:{end}]: {e}", file=sys.stderr)
        start = end
    # average overlaps
    mask = counts.squeeze() > 0
    final[mask] /= counts[mask]
    return final


def main():
    parser = argparse.ArgumentParser(description="Generate ESM3 per-residue embeddings")
    parser.add_argument("fasta_file")
    parser.add_argument("output_dir")
    parser.add_argument("--model", default="esm3-sm-open-v1", dest="model_id")
    parser.add_argument("--skip_check", action="store_true")
    args = parser.parse_args()

    print(f"Loading {args.model_id} on CPU...")
    model = ESM3.from_pretrained(args.model_id).to("cpu").eval()
    tokenizer = EsmSequenceTokenizer()
    print(f"Model loaded. Computing embedding dimension...")
    # optional sanity check
    if not args.skip_check:
        test_seq = "MSRSLLLRFLLFLLLLPPLP"
        emb = embed_windows(model, tokenizer, test_seq)
        if emb.shape != (len(test_seq), emb.shape[1]):
            print("Sanity check failed", file=sys.stderr)
            sys.exit(1)

    records = list(SeqIO.parse(args.fasta_file, "fasta"))
    if not records:
        print("No sequences found", file=sys.stderr)
        sys.exit(1)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    processed = skipped = failed = 0
    for rec in tqdm(records, desc="embedding"):
        seq = str(rec.seq).upper()
        out_file = Path(args.output_dir) / f"{md5(seq)}.pt"
        if out_file.exists():
            skipped += 1
            continue
        try:
            emb = embed_windows(model, tokenizer, seq)
            torch.save(emb.float(), out_file)
            processed += 1
        except Exception as e:
            print(f"Error processing {rec.id}: {e}", file=sys.stderr)
            failed += 1
    print(f"Processed: {processed}, skipped: {skipped}, failed: {failed}")

if __name__ == "__main__":
    main()
