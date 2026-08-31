# Training — prost5-propeptide

How to reproduce this branch's numbers. See [`README.md`](README.md) for what the branch is and what it scored.

## Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## Step 2 — Prepare data

Two CSV files are required (already provided under `data/` for the UniProt 2022 benchmark):

**`labeled_sequences.csv`** (indexed by `protein_id`):

| column | description |
|---|---|
| `sequence` | full precursor amino acid sequence |
| `propeptide_coordinates` | propeptide coordinates, e.g. `(12-45)` |
| `organism` | organism name or taxon |

**`graphpart_assignments.csv`** (indexed by `AC`):

| column | description |
|---|---|
| `cluster` | partition index 0–4 (from [Graph-Part](https://github.com/graph-part/graph-part)) |

---

## Step 3 — Precompute embeddings

Embeddings are computed **once** and cached as `.pt` files (one per sequence, named by
MD5 hash of the sequence). The training script only reads these files.

```bash
python -m src.utils.make_embeddings \
    data/protein_sequences.fasta \
    PATH/TO/EMBEDDINGS/ \
    --half          # GPU only — see below
```

The script skips sequences whose `.pt` file already exists, so it is safe to
interrupt and resume.

> **`--half` is GPU-only.** The ProstT5 README states that "only GPUs support
> half-precision currently; if you want to run on CPU use full-precision". On a
> CPU run the flag is ignored with a warning and the model stays in fp32.
>
> **fp16 and fp32 embeddings are not interchangeable.** The values differ, so a
> model trained on one set is not directly comparable with numbers produced from
> the other. Pick one precision for a whole comparison and record which.
>
> **`--half` needs a fresh output directory.** The extractor skips any sequence
> whose `.pt` file already exists, so pointing `--half` at a directory that
> already holds fp32 files writes nothing and silently trains on the fp32 set.
>
> T5 is known to overflow in fp16. The extractor prints a warning naming any
> sequence with non-finite values (they are zeroed on save), and `preflight.py`
> re-checks the written files — run it before training rather than discovering
> it as a diverged loss forty epochs in.

---

## Step 4 — Train

```bash
python -m src.train_loop_crf \
    --embeddings_dir PATH/TO/EMBEDDINGS \
    --data_file data/labeled_sequences.csv \
    --partitioning_file data/graphpart_assignments.csv \
    --out_dir PATH/TO/OUTPUT
```

**Key arguments:**

| argument | default | description |
|---|---|---|
| `--embedding_dim` | 1024 | ProstT5 output dimension |
| `--epochs` | 30 | training epochs (run in full unless `--patience` > 0) |
| `--batch_size` | 100 | sequences per batch |
| `--patience` | 0 | early stopping: epochs without propeptide F1 improvement. `0` = disabled (upstream behaviour, and the default for reported runs) |
| `--model` | `lstmcnncrf` | model architecture (`lstmcnncrf`, `lstmcnncrf_simple`, `selfattentioncrf`) |
| `--out_dir` | `train_run` | where checkpoints and logs are saved |
| `--lr` | 1e-4 | learning rate (constant — no scheduler, matching the ESM branches) |
| `--dropout` | 0.1 | input and conv dropout |
| `--num_filters` | 32 | number of CNN filters |
| `--hidden_size` | 64 | biLSTM hidden size |
| `--kernel_size` | 3 | CNN kernel size |

Training writes to `--out_dir`:
- `model.pt` — best checkpoint (by validation propeptide F1)
- `valid_metrics.json` — validation metrics at best epoch
- `test_metrics.json` — test metrics
- TensorBoard logs (run `tensorboard --logdir PATH/TO/OUTPUT`)

---

## Running on a GPU

Nothing in the training code is CPU-specific; `src/train_loop_crf.py` picks
`cuda` when it is available. Two things are worth knowing before starting a long
GPU run.

**The Viterbi backtrace is the bottleneck, not the network.** `multi_tag_crf`
reconstructs each path in a Python loop that calls `.item()` once per
(sequence x timestep), and on an accelerator every one of those calls is a
device-to-host sync. Measured at batch 100, length 142: 0.02 s for the whole
forward pass, 2.47 s for the decode that follows it. The training loop never
reads the training-set paths — there are no train metrics — so `forward()` takes
`skip_decode`, and `run_dataloader` passes it while training. Gradients come from
the CRF likelihood and are untouched: training with `skip_decode=True` is
bit-identical, verified epoch by epoch against the unpatched branch. Validation
and test still decode, because their paths are what the metrics are computed
from.

**A GPU run is not numerically comparable to a CPU run of the same code.** cuDNN
picks different reduction orders and its LSTM kernels are not deterministic by
default, so expect small differences in the reported F1. Keep every branch of a
comparison on the same hardware.

A single 50-epoch run (fixed split: partitions 0-2 train, 3 validation, 4 test):

```bash
python preflight.py --embeddings_dir PATH/TO/EMBEDDINGS --embedding_dim 1024

python run.py \
    --embeddings_dir PATH/TO/EMBEDDINGS \
    --data_file data/labeled_sequences.csv \
    --partitioning_file data/graphpart_assignments.csv \
    --embedding_dim 1024 \
    --epochs 50 \
    --out_dir results/prost5_propeptide_50ep
```

`--patience` stays at its default of 0, so all 50 epochs run and the reported
model is the best-on-validation checkpoint.

---

## Step 5 — Evaluate

After training completes, test-set metrics are written automatically to `--out_dir/test_metrics.json`.
These include precision, recall, and F1 for propeptide predictions, computed with a
±3-residue boundary tolerance.

The published results were produced by running 5-fold nested CV (5 outer folds × 4 inner folds,
20 checkpoints total), `evaluation/measure_performance.py` aggregates the saved predictions from those 20 runs — it
requires the original checkpoint directories and cannot be re-run without them.

- PeptideLocator was evaluated as a licensed executable and cannot be provided in this repo.

---
