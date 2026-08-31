# Propeptide prediction

Predicting propeptide cleavage sites in protein sequences from frozen protein language
model embeddings, built on
[DeepPeptide](https://github.com/fteufel/DeepPeptide) (Teufel et al., *Bioinformatics*
2023, [`btad616`](https://doi.org/10.1093/bioinformatics/btad616)). The point is to compare
protein language models on one task: same head, same data, same splits, same metric, same
training budget, and only the embeddings change. If one arm scores better, that should be
the embeddings rather than something else in the pipeline.

This branch is the ESM-2 arm (`esm2_t33_650M_UR50D`, 1280 dims per residue). Runs on CPU.

## Before you run this

This code accompanies an MSc thesis. **If you intend to run it, please contact me
first** — apostolosfysekidis1@gmail.com. I would like to know who is using it.

The trained model weights are **not published here**. They are available from me on
request. Without them you can read, adapt and retrain the method, but you cannot run
the predictor as reported in the thesis.

This repository is MIT licensed, so the licence does not oblige you to make contact.
The above is a request, not a condition.

## Branches

One branch per embedding model. Each is a full working copy, not a patch on top of another.

| branch | embeddings | what it is |
|---|---|---|
| `main` | ESM-2, 1280 | propeptides only, the reference arm |
| `baseline-upstream` | ESM-2, 1280 | faithful to upstream: joint peptides + propeptides, upstream metric |
| `esm3-propeptide` | ESM3 `esm3_sm_open_v1`, 1536 | propeptides only. Also holds the structure channel, LoRA fine-tuning, Optuna/nested CV, and the analysis scripts |
| `esm3-full` | ESM3, 1536 | joint peptides + propeptides |
| `prost5-propeptide` | ProstT5, 1024 | propeptides only |
| `prost5-full` | ProstT5, 1024 | joint peptides + propeptides |
| `archive/eirini-esm1b` | ESM-1b | an older contributed fork, kept for the record. Different code and a different metric version, so its numbers don't belong in a table with the rest |

Don't compare F1 across branches without checking which metric produced it.
`baseline-upstream` keeps the upstream metric on purpose, so that it reproduces the
published figures. Every other branch fixes that metric, which shifts the values.

## Results

One Graph-Part split: clusters 0,1,2 to train, 3 to validate, 4 to test (4,455 / 1,630 /
1,538 sequences). Scored at a ±3 residue boundary tolerance with the paper's T4
hyperparameters (`lr 0.0055, batch 20, dropout 0.6902, conv_dropout 0.2672, kernel 5,
filters 48, hidden 48`).

| model | precision | recall | F1 (propeptide) |
|---|---|---|---|
| propeptides only (this branch), ESM-2 T4 | 0.717 | 0.556 | **0.626** |
| joint peptide+propeptide, ESM-2 T4 (`baseline-upstream`) | 0.685 | 0.527 | 0.596 |
| DeepPeptide paper (propeptides, ±3) | 0.64 | 0.46 | ~0.535 |

Dropping the peptide labels helps propeptide detection, roughly +0.03 F1 and mostly from
precision, and it lands above the published propeptide numbers. On the test split the model
predicts 1,100 propeptides where there are 1,420 true ones across 1,538 proteins.
Validation F1 peaks near 0.76 around epochs 6–20, after which the run can diverge. The saved
checkpoint is from the peak.

## Training and prediction

[`TRAINING.md`](TRAINING.md) has the full procedure: install, the two input CSVs
and their formats, generating embeddings, and the exact `run.py` invocation with
every hyperparameter that reproduces the result above.

To run the predictor on a raw sequence see the
[predictor README](predictor/README.md), or call
`LSTMCNNCRF.predict_from_sequence(seq)`, which embeds with ESM-2 internally and
returns the Viterbi propeptide spans.

## License and credit

BSD 3-Clause, inherited from upstream. `LICENSE` is unchanged from
[fteufel/DeepPeptide](https://github.com/fteufel/DeepPeptide) and keeps the original notice,
`Copyright (c) 2023, F Teufel`, as the license requires. Modifications in this repo are
released under the same terms.

The method, the dataset and the architecture are Teufel et al.'s. This is a derivative repo
for a thesis, not the reference implementation. For that, use the upstream repo. If you use
this work, cite the original paper:

> Teufel, F., Refsgaard, J.C., Kasimova, M.A., Deibler, K., Madsen, C.T., Stahlhut, C.,
> Grønborg, M., Winther, O., Madsen, D. (2023). DeepPeptide predicts cleaved peptides in
> proteins using conditional random fields. *Bioinformatics* 39(6), btad616.
