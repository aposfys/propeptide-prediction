# DeepPeptide (ESM-2)
Predicting cleaved peptides and propeptides in protein sequences using ESM-2.

[![DOI](https://zenodo.org/badge/593202385.svg)](https://zenodo.org/badge/latestdoi/593202385)

Embedder: `esm2_t33_650M_UR50D` (ESM-2, 1280-dim per residue). Runs on CUDA, MPS (Apple Silicon), or CPU.

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
