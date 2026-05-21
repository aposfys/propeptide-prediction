# DeepPeptide (ESM3)
Predicting cleaved peptides in protein sequences using ESM3.

[![DOI](https://zenodo.org/badge/593202385.svg)](https://zenodo.org/badge/latestdoi/593202385)

This branch replaces the ESM-2 embedder with **ESM3** (`esm3_sm_open_v1`, 1536-dim)
while keeping the full multi-label CRF (propeptide + mature peptide coordinates).
CPU compatible: the distributed backend falls back to `gloo` when no GPU is available.

Embedder: `esm3_sm_open_v1` (ESM3, 1536-dim per residue).

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
| `coordinates` | peptide coordinates, e.g. `(12-45),(78-102)` |
| `propeptide_coordinates` | propeptide coordinates in same format |
| `organism` | organism name or taxon |

**`graphpart_assignments.csv`** (indexed by `AC`):

| column | description |
|---|---|
| `cluster` | partition index 0–4 (from [Graph-Part](https://github.com/graph-part/graph-part)) |

---

### Step 3 — Precompute embeddings

```bash
python -m src.utils.make_embeddings \
    data/protein_sequences.fasta \
    PATH/TO/EMBEDDINGS/
```

Embeddings are saved as `.pt` files named by MD5 hash of the sequence. The script
skips sequences already processed, so it is safe to interrupt and resume.

---

### Step 4 — Train

```bash
python run.py \
    --embeddings_dir PATH/TO/EMBEDDINGS \
    -df data/labeled_sequences.csv \
    -pf data/graphpart_assignments.csv
```

**Key arguments:**

| argument | default | description |
|---|---|---|
| `--embedding_dim` | 1536 | ESM3 output dimension |
| `--epochs` | 30 | max training epochs (subject to early stopping) |
| `--batch_size` | 100 | sequences per batch |
| `--patience` | 10 | early stopping: epochs without mean peptide+propeptide F1 improvement |
| `--label_type` | `multistate_with_propeptides` | CRF label scheme |
| `--model` | `lstmcnncrf` | model architecture |
| `--out_dir` | `train_run` | where checkpoints and logs are saved |
| `--lr` | 1e-4 | peak learning rate (warmed up linearly, then cosine-decayed) |
| `--dropout` | 0.1 | input dropout |
| `--conv_dropout` | 0.1 | conv layer dropout |
| `--num_filters` | 32 | number of CNN filters |
| `--hidden_size` | 64 | biLSTM hidden size |
| `--kernel_size` | 3 | CNN kernel size |

---

### Step 5 — Evaluate

After training completes, test-set metrics are written automatically to `--out_dir/test_metrics.json`.
These include precision, recall, and F1 for both peptide and propeptide predictions, computed with a
±3-residue boundary tolerance.

To reproduce the metrics from the original publication, which used 5-fold nested CV
(5 outer folds × 4 inner folds, 20 checkpoints total), run:

```bash
python evaluation/measure_performance.py
```

The selected checkpoints are hardcoded in that script.

- PeptideLocator was evaluated as a licensed executable and cannot be provided in this repo.

---

## Predicting

[See the predictor README](predictor/README.md)
