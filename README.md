# DeepPeptide (ProstT5, propeptide-only)
Predicting propeptide cleavage sites in protein sequences using ProstT5.

https://www.biorxiv.org/content/10.1101/2023.07.23.550085v1

This branch restricts training and evaluation to the **propeptide label only**
(states 1–50; state 0 = background). Mature peptide coordinates are ignored.
The CRF head is hardcoded to 51 states and 2 label classes.

Embedder: `Rostlab/ProstT5` (1024-dim per residue).

---

## What this branch is

The **ProstT5** arm of the representation comparison, propeptides only: 2 labels,
51 CRF states, mature-peptide coordinates dropped. Embeddings are 1024 dims per
residue. Everything else — the head, the data, the Graph-Part splits, the metric
and the training budget — is identical to `main`, so a difference in score is a
difference in the embeddings.

Scored on the test split at ±3 residue tolerance, propeptide F1:

| | mean | sd | n |
|---|---|---|---|
| tuned (T4 hyperparameters) | **0.5189** | 0.0306 | 5 |
| default hyperparameters | 0.4091 | 0.1094 | 5 |

For reference, ESM-2 reaches 0.6153 tuned. The per-run metrics for every arm are
collected on the `esm3-propeptide` branch under `results/`. See `main` for the
branch map and the caution about comparing F1 across branches.

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

## Predicting

[See the predictor README](predictor/README.md)
