# Propeptide prediction

Predicting **propeptide** cleavage sites in protein sequences from frozen protein
language model embeddings.

MSc Bioinformatics thesis (NKUA). The project adapts
[DeepPeptide](https://github.com/fteufel/DeepPeptide) (Teufel et al., *Bioinformatics*
2023, [`btad616`](https://doi.org/10.1093/bioinformatics/btad616)) into a controlled
comparison: architecture, data, splits, metric and training budget are held fixed while
the frozen input representation is swapped, so that a difference in performance is
attributable to the representation rather than to the model built around it.

The original 3-label / 101-state joint model (peptides **and** propeptides) is reduced to
a 2-label / 51-state model predicting **propeptides only** — states 1–50, state 0 =
background; mature-peptide coordinates are ignored. Everything else stays faithful to the
original: embeddings → CNN–biLSTM–CNN → linear-chain CRF with a duration-encoded state
grammar (length 5–50), Viterbi decoding, constant-LR Adam, no scheduler, best-on-validation
checkpointing.

This branch is the **ESM-2** arm (`esm2_t33_650M_UR50D`, 1280-dim per residue). Runs on CPU.

---

## Branches

One branch per representation. Each is a complete working copy, not a patch series.

| branch | representation | scope |
|---|---|---|
| `main` | ESM-2, 1280 | propeptide-only — the reference arm |
| `baseline-upstream` | ESM-2, 1280 | upstream-faithful: joint peptides + propeptides, **upstream metric** |
| `esm3-propeptide` | ESM3 `esm3_sm_open_v1`, 1536 | propeptide-only |
| `esm3-multimodal-propeptide` | ESM3 + structure channel, 1536 | propeptide-only; also carries the Optuna / nested-CV machinery |
| `esm3-propeptide-optuna-gpu` | ESM3, 1536 | propeptide-only, GPU hyperparameter search |
| `esm3-full` | ESM3, 1536 | joint peptides + propeptides |
| `prost5-propeptide` | ProstT5, 1024 | propeptide-only |
| `prost5-full` | ProstT5, 1024 | joint peptides + propeptides |
| `eirini_branch` | ESM-1b | contributed baseline run (propeptide F1 0.371 at ±3) |

> **Numbers are not interchangeable across branches.** `baseline-upstream` is deliberately
> scored with the *unfixed* upstream metric so that it reproduces the published DeepPeptide
> figures like for like. Every other branch corrects that metric (see Caveats), so its
> values sit on a different footing and must not be tabulated alongside.

---

## Results

Single Graph-Part split (train = clusters 0,1,2 / val = 3 / test = 4 → 4,455 / 1,630 / 1,538
sequences). Propeptide detection at a **±3-residue** boundary tolerance, using the paper's
tuned **T4 hyperparameters** (`lr 0.0055, batch 20, dropout 0.6902, conv_dropout 0.2672,
kernel 5, filters 48, hidden 48`).

| model | precision | recall | **F1 (propeptide)** |
|---|---|---|---|
| **propeptide-only (this branch)** — ESM-2, T4 | **0.717** | **0.556** | **0.626** |
| joint peptide+propeptide — ESM-2, T4 (`baseline-upstream`) | 0.685 | 0.527 | 0.596 |
| DeepPeptide paper (propeptides, ±3) | 0.64 | 0.46 | ~0.535 |

Restricting the model to propeptides improves propeptide detection over the joint model
(+0.03 F1, driven mainly by precision) and matches or exceeds the published DeepPeptide
propeptide numbers. On the test split the model predicts 1,100 propeptides against 1,420
true ones across 1,538 proteins. Validation F1 peaks near 0.76 around epochs 6–20 before a
late-training divergence; best-on-validation checkpointing keeps the peak model.

### Caveats

- **Single split, no error bars.** These are point estimates from one training run. Run-to-run
  variation in this pipeline is not negligible, so a difference of this size between two
  representations should not be read as a ranking without replicates. The replicated,
  variance-aware comparison across representations is the substance of the thesis; these
  numbers are the reference arm, not the result.
- **Metric fix.** Upstream `get_counts_for_protein` reuses a loop variable across the true
  and predicted loops, marking a matched true peptide at the *prediction's* index. The
  resulting phantom row is dropped by `groupby('group')`, converting real true positives
  into false negatives — so **upstream understates recall and F1**, and does so more the more
  false positives a model emits. This branch fixes it, and the joint baseline above is
  re-scored with the same fixed metric for a fair comparison. The published paper numbers
  use the unfixed metric.
- **Gradient clipping.** Upstream calls `clip_grad_norm_` before `backward()`, where no
  gradients exist yet, so the published models trained unclipped. This branch clips.

---

## Training

### 1 — Install
```bash
pip install -r requirements.txt   # torch >= 2.0, fair-esm, pandas, numpy, tensorboard, tqdm
```

### 2 — Data
Two CSVs, provided under `data/` for the UniProt-2022 benchmark:

- **`labeled_sequences.csv`** (indexed by `protein_id`): `sequence`, `propeptide_coordinates`
  (e.g. `(12-45),(98-113)`), `organism`.
- **`graphpart_assignments.csv`** (indexed by `AC`): `cluster`, partition index 0–4 from
  [Graph-Part](https://github.com/graph-part/graph-part).

### 3 — Precompute embeddings
```bash
python -m src.utils.make_embeddings data/protein_sequences.fasta PATH/TO/EMBEDDINGS/
```
Saved as one `.pt` per sequence, named by MD5 hash of the sequence. Safe to interrupt and
resume — existing files are skipped.

### 4 — Train
Reproduces the result above:
```bash
python run.py \
    --embeddings_dir PATH/TO/EMBEDDINGS \
    -df data/labeled_sequences.csv -pf data/graphpart_assignments.csv \
    --embedding_dim 1280 --epochs 50 \
    --lr 0.0055 --batch_size 20 --dropout 0.6902 --conv_dropout 0.2672 \
    --kernel_size 5 --num_filters 48 --hidden_size 48 \
    --out_dir results/esm2_prop
```

| argument | value used | description |
|---|---|---|
| `--embedding_dim` | 1280 | representation dimension — 1536 for ESM3, 1024 for ProstT5 |
| `--epochs` | 50 | training epochs |
| `--lr` | 0.0055 | **constant** learning rate (T4); default `1e-4` |
| `--batch_size` | 20 | sequences per batch |
| `--dropout` / `--conv_dropout` | 0.6902 / 0.2672 | input / conv dropout |
| `--num_filters` / `--hidden_size` / `--kernel_size` | 48 / 48 / 5 | CNN filters / biLSTM hidden / CNN kernel |
| `--patience` | 0 | early-stopping patience; **0 disables it**, which is upstream behaviour and the default |
| `--out_dir` | — | checkpoints, metrics and TensorBoard logs |

> **Stability.** The model reaches its best validation F1 within roughly 20 epochs and can
> diverge to `NaN` later. This is harmless: checkpointing only writes on improvement, and
> once validation F1 collapses the comparison never fires again, so `model.pt` holds the
> pre-divergence peak regardless of what the tail does.

Writes `model.pt`, `valid_metrics.json`, `test_metrics.json`, `test_outputs.pickle` /
`valid_outputs.pickle`, and TensorBoard logs (`tensorboard --logdir <out_dir>`).

Test metrics at ±3 tolerance are written automatically when training finishes.

<details>
<summary>Re-scoring a saved checkpoint without retraining</summary>

Useful if a run was interrupted before its test phase, or to re-score with an updated metric.

```python
import json, torch
from torch.utils.data import DataLoader
from src.models.crf_models import LSTMCNNCRF
from src.utils.dataset import PrecomputedCSVForOverlapCRFDataset
from src.utils.manuscript_metrics import compute_all_metrics, convert_path_to_peptide_borders

cfg = json.load(open('results/esm2_prop/config.json'))
m = LSTMCNNCRF(input_size=cfg['embedding_dim'], num_labels=2, num_states=51,
               n_filters=cfg['num_filters'], hidden_size=cfg['hidden_size'],
               filter_size=cfg['kernel_size'], dropout_input=cfg['dropout'],
               dropout_conv1=cfg['conv_dropout'])
m.load_state_dict(torch.load('results/esm2_prop/model.pt', map_location='cpu')); m.eval()

ds = PrecomputedCSVForOverlapCRFDataset(cfg['embeddings_dir'], cfg['data_file'],
                                        cfg['partitioning_file'], partitions=[4])
loader = DataLoader(ds, batch_size=cfg['batch_size'], shuffle=False, collate_fn=ds.collate_fn)
preds = []
with torch.no_grad():
    for emb, mask, label, _ in loader:
        _, p, _ = m(emb, mask, label, skip_marginals=True); preds.extend(p)

r = compute_all_metrics(None, preds, None, ds.names, ds.data, windows=[3])[0]
print("test f1_prop=%.4f  P=%.3f  R=%.3f" % (r['f1 propeptides'], r['precision propeptides'], r['recall propeptides']))
# predicted propeptide spans per protein: convert_path_to_peptide_borders(pred, 1, 50, offset=1)
```
</details>

---

## Predicting from a raw sequence

See the [predictor README](predictor/README.md), or call
`LSTMCNNCRF.predict_from_sequence(seq)`, which embeds with ESM-2 internally and returns
Viterbi propeptide spans.

---

## Credit

All credit for the original method, dataset and architecture belongs to Teufel et al.
This repository is a derivative work for a thesis; it is not the reference implementation.
For that, use [fteufel/DeepPeptide](https://github.com/fteufel/DeepPeptide).
