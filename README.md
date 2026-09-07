# DeepPeptide (ProstT5, structure-conditioned, propeptide-only)
Predicting propeptide cleavage sites using ProstT5 over **amino acids and
Foldseek 3Di**, not amino acids alone.

This branch is `prost5-propeptide` plus a structural input channel. ProstT5 is
bilingual — it was trained to translate between residue sequences and 3Di
structural strings — and the sequence-only arm used only one of those languages.
Here both are encoded and concatenated to 2048 dims per residue, with the 3Di
channel derived from AlphaFold DB models.

**Nothing has been run on this branch yet.** It carries the extraction pipeline,
the ablation design, the controls and the power calculation. The runs and their
order are in [NEXT_STEPS.md](NEXT_STEPS.md). Do not read the table below as a
result for the structure arm — it is the sequence-only baseline this arm has to
beat.

https://www.biorxiv.org/content/10.1101/2023.07.23.550085v1

This branch restricts training and evaluation to the **propeptide label only**
(states 1–50; state 0 = background). Mature peptide coordinates are ignored.
The CRF head has 51 states and 2 label classes by default; the state count now
follows `--max_peptide_len` rather than being hardcoded.

Embedder: `Rostlab/ProstT5`. 1024 dims per residue per language, so 1024 for
`--tracks aa` or `--tracks 3di`, and 2048 for `--tracks aa+3di`.

### Arms on this branch

| arm | flags | dims | what it is |
|---|---|---|---|
| sequence-only | `--tracks aa` | 1024 | reproduces `prost5-propeptide` |
| structure-only | `--tracks 3di` | 1024 | how much of the task is structural |
| both | `--tracks aa+3di --fuse renorm` | 1024 | the treatment |
| **shuffled control** | `… --fuse renorm --shuffle_3di` | 1024 | same architecture, same mask, wrong structures |
| both, concatenated | `--tracks aa+3di --fuse concat` | 2048 | second experiment, widens `conv1` |

**The architecture is unchanged.** At 1024 dims this branch builds a
byte-identical model to `prost5-propeptide` — 51 CRF states, the same constraint
mask, 192,369 trainable parameters — and `test_architecture.py` asserts it
against values measured on the parent branch. Only `--fuse concat` changes
anything, and it changes exactly one layer. See [FUSION.md](FUSION.md).

The shuffled control is not optional either way. It is the only pair that
separates structural information from the mere presence of a second channel.

### Also on this branch

- **Both boundary tolerances are scored.** Every run writes
  `f1 propeptides@1` and `f1 propeptides@3` to `test_metrics.json`. The
  unsuffixed `f1 propeptides` key still holds ±3, so every existing number and
  reader is unaffected. Model selection still uses ±3.
- **The CRF grammar is configurable** via `--min_peptide_len` / `--max_peptide_len`,
  defaulting to the published 5..50 window. [GRAMMAR.md](GRAMMAR.md) measures what
  that window covers in UniProt and what widening it would cost. `test_grammar.py`
  asserts the defaults reproduce the published grammar exactly.

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
