# DeepPeptide (ESM3)
Predicting cleaved peptides in protein sequences using ESM3.

https://www.science.org/doi/10.1126/science.ade2574

This branch replaces the ESM-2 embedder with **ESM3** (`esm3_sm_open_v1`, 1536-dim)
while keeping the full multi-label CRF (propeptide + mature peptide coordinates).
Runs on CUDA or CPU (single device; the original FSDP/distributed wrapper was removed).

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

## Results — ESM3 (single fold, default HPs)

> 🛑 **RETRACTED — every row below predates the training/metric unification and must be
> regenerated.** Two changes invalidate them:
>
> 1. **Metric.** These numbers were scored with the upstream `get_counts_for_protein`, which
>    reused `idx` across the true and pred loops and marked a matched true peptide at the
>    *pred's* index. The resulting phantom row is dropped by `groupby('group')`, turning real
>    true positives into false negatives — so **recall and F1 here are understated**. Fixed now.
> 2. **Gradient clipping.** These models trained with upstream's `clip_grad_norm_` call placed
>    *before* `backward()`, where no gradients exist yet — i.e. **unclipped**. Clipping now
>    happens after `backward()` at 0.25, as on every other branch. This changes training, so
>    re-scoring is not enough: **the ESM3 models must be retrained.**
>
> The ProstT5 row additionally used a warmup+cosine LR schedule and a patience break that no
> branch uses any more. Regenerate all three rows under the current recipe before comparing.

Single split (`train=[0,1,2]`, `val=[3]`, `test=[4]`), 50 epochs, best-checkpoint-on-validation
(stopping metric = mean of peptide & propeptide F1), ±3-residue tolerance. ESM3 embeddings:
`esm3_sm_open_v1`, 1536-dim, layer-final, full 8061/8061 coverage. **All rows below use the code's
_default_ hyperparameters** (lr 1e-4, dropout 0.1, batch 100, kernel 3, filters 32, hidden 64):

| embedder (default HPs) | f1 peptides | f1 propeptides | f1 all |
|---|---|---|---|
| ESM-2 (1280-d) | 0.399 | 0.462 | 0.429 |
| **ESM3 (1536-d)** | **0.395** | **0.496** | **0.448** |
| ProstT5 (1024-d)¹ | 0.200 | 0.242 | 0.220 |

At default HPs, **ESM3 modestly edges ESM-2** (higher propeptide + overall F1; peptide tied), and
both clearly beat ProstT5.

> ⚠️ **These default-HP numbers are *not* at published-performance level.** The default learning
> rate (1e-4) under-trains the model. On the `main` branch the *same* ESM-2 setup with the paper's
> Optuna-tuned hyperparameters (Supplementary Table S2, fold T4) jumps to **peptide F1 0.579 /
> propeptide F1 0.590** — ~0.15 higher. So a fair ESM-2-vs-ESM3 comparison needs **tuned HPs for
> ESM3 too**. The paper only published hyperparameters for ESM-1b/ESM-2 (Table S2), so ESM3 requires
> either a fresh Optuna search or reuse of the ESM-2 T4 HPs. **Pending — do not draw conclusions
> from the default-HP table above.**

Reproduce (ESM3, default HPs):

```bash
python run.py \
    --embeddings_dir PATH/TO/ESM3_EMBEDDINGS \
    -df data/labeled_sequences.csv -pf data/graphpart_assignments.csv \
    --embedding_dim 1536 --epochs 50 --out_dir results/esm3
```

If some sequences >1022 residues are missing from your ESM3 embeddings, filter
`labeled_sequences.csv` to the embedded subset first (match by MD5 hash of the sequence).

¹ ProstT5 was run with the same faithful code from the `deeppeptide-prost5` branch; shown here for comparison.

---

## Predicting

[See the predictor README](predictor/README.md)
