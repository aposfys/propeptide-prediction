# DeepPeptide (ESM3)
Predicting cleaved peptides in protein sequences using ESM3.

https://www.science.org/doi/10.1126/science.ade2574

This branch replaces the ESM-2 embedder with **ESM3** (`esm3_sm_open_v1`, 1536-dim)
while keeping the full multi-label CRF (propeptide + mature peptide coordinates).
Runs on CUDA or CPU (single device; the original FSDP/distributed wrapper was removed).

Embedder: `esm3_sm_open_v1` (ESM3, 1536-dim per residue).

---

## Before you run this

This code accompanies an MSc thesis. **If you intend to run it, please contact me
first** — apostolosfysekidis1@gmail.com. I would like to know who is using it.

The trained model weights are **not published here**. They are available from me on
request. Without them you can read, adapt and retrain the method, but you cannot run
the predictor as reported in the thesis.

This repository is MIT licensed, so the licence does not oblige you to make contact.
The above is a request, not a condition.

## Training

[`TRAINING.md`](TRAINING.md) has the full procedure: install, data preparation,
precomputing embeddings, the training invocation and evaluation.

## Results

See [RESULTS.md](RESULTS.md).

## Predicting

[See the predictor README](predictor/README.md)
