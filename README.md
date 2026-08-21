# DeepPeptide (ProstT5)
Predicting cleaved peptides in protein sequences using ProstT5.

https://www.biorxiv.org/content/10.1101/2023.07.23.550085v1

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
    PATH/TO/EMBEDDINGS/ \
    --half          # GPU only — see below
```

The script skips sequences whose `.pt` file already exists, so it is safe to
interrupt and resume.

> **`--half` is GPU-only.** The ProstT5 README states that "only GPUs support
> half-precision currently; if you want to run on CPU use full-precision". On a
> CPU run the flag is ignored with a warning and the model stays in fp32. Since
> these branches train on the CPU-only HPC node, just omit it there.

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
| `--epochs` | 30 | training epochs (run in full unless `--patience` > 0) |
| `--batch_size` | 100 | sequences per batch |
| `--patience` | 0 | early stopping: epochs without mean peptide+propeptide F1 improvement. `0` = disabled (upstream behaviour, and the default for reported runs) |
| `--label_type` | `multistate_with_propeptides` | CRF label scheme |
| `--model` | `lstmcnncrf` | model architecture |
| `--out_dir` | `train_run` | where checkpoints and logs are saved |
| `--lr` | 1e-4 | learning rate (constant — no scheduler, matching the ESM branches) |
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
20 checkpoints total), `evaluation/measure_performance.py` aggregates the saved predictions from those 20 runs — it
requires the original checkpoint directories and cannot be re-run without them.

- PeptideLocator was evaluated as a licensed executable and cannot be provided in this repo.

---

## Predicting

[See the predictor README](predictor/README.md)
