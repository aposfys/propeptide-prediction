# DeepPeptide (ProstT5, propeptide-only)
Predicting propeptide cleavage sites in protein sequences using ProstT5.

[![DOI](https://zenodo.org/badge/593202385.svg)](https://zenodo.org/badge/latestdoi/593202385)

This branch restricts training and evaluation to the **propeptide label only**
(states 1–50; state 0 = background). Mature peptide coordinates are ignored.
The CRF head is hardcoded to 51 states and 2 label classes.

Embedder: `Rostlab/ProstT5` (1024-dim per residue).

---

## Training

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

---

### Step 2 — Prepare data

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

### Step 3 — Precompute embeddings

Embeddings are computed **once** and cached as `.pt` files (one per sequence, named by
MD5 hash of the sequence). The training script only reads these files.

```bash
python -m src.utils.make_embeddings_prost5 \
    data/protein_sequences.fasta \
    PATH/TO/EMBEDDINGS/ \
    --half          # fp16 halves memory; accuracy impact is negligible
```

The script skips sequences whose `.pt` file already exists, so it is safe to
interrupt and resume.

---

### Step 4 — Train

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
| `--epochs` | 30 | max training epochs (subject to early stopping) |
| `--batch_size` | 100 | sequences per batch |
| `--patience` | 10 | early stopping: epochs without propeptide F1 improvement |
| `--model` | `lstmcnncrf` | model architecture (`lstmcnncrf`, `lstmcnncrf_simple`, `selfattentioncrf`) |
| `--out_dir` | `train_run` | where checkpoints and logs are saved |
| `--lr` | 1e-4 | peak learning rate (warmed up linearly, then cosine-decayed) |
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

### Step 5 — Evaluate

We used 5-fold nested CV to produce 20 model checkpoints (5 outer folds × 4 inner folds).
The selected checkpoints are hardcoded in `evaluation/measure_performance.py`, which
computes precision / recall / F1 from saved predictions.

- PeptideLocator was evaluated as a licensed executable and cannot be provided in this repo.

---

## Predicting

[See the predictor README](predictor/README.md)
