# DeepPeptide (ESM-2)
Predicting cleaved peptides and propeptides in protein sequences using ESM-2.

[![DOI](https://zenodo.org/badge/593202385.svg)](https://zenodo.org/badge/latestdoi/593202385)

Embedder: `esm2_t33_650M_UR50D` (ESM-2, 1280-dim per residue). Runs on CUDA, MPS (Apple Silicon), or CPU.

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

Embeddings are computed **once** and cached as `.pt` files (one per sequence, named by
MD5 hash of the sequence). The training script only reads these files.


```bash
python -m src.utils.make_embeddings \
    data/protein_sequences.fasta \
    PATH/TO/EMBEDDINGS/
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
| `--embedding_dim` | 1280 | ESM-2 output dimension |
| `--epochs` | 30 | max training epochs (subject to early stopping) |
| `--batch_size` | 100 | sequences per batch |
| `--patience` | 10 | unused (kept for CLI compatibility); the loop runs all epochs and keeps the best-validation checkpoint |
| `--label_type` | `multistate_with_propeptides` | CRF label scheme |
| `--model` | `lstmcnncrf` | model architecture |
| `--out_dir` | `train_run` | where checkpoints and logs are saved |
| `--lr` | 1e-4 | learning rate (constant — faithful to the original, no scheduler) |
| `--dropout` | 0.1 | input dropout |
| `--conv_dropout` | 0.1 | conv layer dropout |
| `--num_filters` | 32 | number of CNN filters |
| `--hidden_size` | 64 | biLSTM hidden size |
| `--kernel_size` | 3 | CNN kernel size |

Training writes to `--out_dir`:
- `model.pt` — best checkpoint (by validation F1)
- `valid_metrics.json` — validation metrics at best epoch
- `test_metrics.json` — test metrics
- TensorBoard logs (run `tensorboard --logdir PATH/TO/OUTPUT`)

---

### Step 5 — Evaluate

After training completes, test-set metrics are written automatically to `--out_dir/test_metrics.json`.
These include precision, recall, and F1 for both peptide and propeptide predictions, computed with a
±3-residue boundary tolerance.

The published results were produced by running 5-fold nested CV (5 outer folds × 4 inner folds,
20 checkpoints total) using `esm3-propeptide-only` branch with Optuna hyperparameter search.
`evaluation/measure_performance.py` aggregates the saved predictions from those 20 runs — it
requires the original checkpoint directories and cannot be re-run without them.

- PeptideLocator was evaluated as a licensed executable and cannot be provided in this repo.

---

## Results — ESM-2 single-fold reproduction

All runs below use a single split (`train=[0,1,2]`, `val=[3]`, `test=[4]`), 50 epochs,
best-checkpoint-on-validation (stopping metric = mean of peptide & propeptide F1), evaluated at
±3-residue tolerance. Embeddings: ESM-2 (`esm2_t33_650M_UR50D`, **layer 33**, 1280-dim).

| run | learning rate | other hyperparameters | f1 peptides | f1 propeptides |
|---|---|---|---|---|
| default HPs | 1e-4 | dropout 0.1, conv 0.1, batch 100, kernel 3, filters 32, hidden 64 | 0.399 | 0.462 |
| **tuned (paper Table S2, fold T4)** | **5.5e-3** | dropout 0.69, conv 0.27, batch 20, kernel 5, filters 48, hidden 48 | **0.579** | **0.590** |

The default hyperparameters severely **under-train** the model — the default learning rate is 55×
too low, so in 50 epochs it barely converges. Using the original paper's Optuna-tuned
hyperparameters for the fold-4 outer model (Supplementary **Table S2**, row **T4**) reproduces
published ESM-2 performance:

- **Original DeepPeptide** (Teufel et al., *Bioinformatics* 2023, ESM-2 layer 33, ±3):
  peptide P 0.69 / R 0.43 (F1 ≈ 0.53 on split 0; 20-model mean P 0.68 / R 0.49, F1 ≈ 0.57);
  propeptide P 0.64 / R 0.46 (F1 ≈ 0.535).
- **This run** (tuned, fold 4): peptide P 0.73 / R 0.48 (**F1 0.579**); propeptide P 0.68 / R 0.52 (**F1 0.590**).

The single-fold result lands in the centre of the paper's 20-model distribution — i.e. the gap to
the original was entirely **hyperparameters**, not the model, data, or metric.

Reproduce with:

```bash
python run.py \
    --embeddings_dir PATH/TO/ESM2_EMBEDDINGS \
    -df data/labeled_sequences.csv -pf data/graphpart_assignments.csv \
    --embedding_dim 1280 --epochs 50 \
    --lr 0.0055 --batch_size 20 --dropout 0.6902 --conv_dropout 0.2672 \
    --kernel_size 5 --num_filters 48 --hidden_size 48 \
    --out_dir results/esm2_T4
```

**Note:** this is a *single fold*. The paper's headline numbers are the mean ± std over the 20
nested-CV models (5 outer folds × 4 inner, each outer fold using its own T0–T4 hyperparameters from
Table S2). Reproducing the mean ± std requires the full 5-fold run.

---

## Predicting

[See the predictor README](predictor/README.md)
