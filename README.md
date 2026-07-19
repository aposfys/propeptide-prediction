# DeepPeptide (ESM-2, propeptide-only)

Predicting **propeptide** cleavage sites in protein sequences using ESM-2.

Adapted from the original DeepPeptide (Teufel et al., *Bioinformatics* 2023, `btad616`).
This branch reduces the original **3-label / 101-state** joint model (peptides **and**
propeptides) to a **2-label / 51-state** model that predicts **propeptides only**
(states 1–50; state 0 = background). Mature-peptide coordinates are ignored
(`true_peptides = []`).

Everything else is kept faithful to the original architecture: ESM-2 embeddings →
CNN–biLSTM–CNN feature extractor → linear-chain CRF with a duration-encoded state
grammar (min length 5, max 50), decoded with Viterbi. Constant-LR Adam, no scheduler,
no early stopping, best-on-validation checkpointing — exactly as upstream. Runs on CPU.

Embedder: `esm2_t33_650M_UR50D` (ESM-2, 1280-dim per residue).

---

## Results

Single Graph-Part split (train = clusters 0,1,2 / val = 3 / test = 4 → 4,455 / 1,630 / 1,538
sequences). Propeptide detection at a **±3-residue** boundary tolerance, using the paper's
tuned **T4 hyperparameters** (`lr 0.0055, batch 20, dropout 0.6902, conv_dropout 0.2672,
kernel 5, filters 48, hidden 48`).

| model | precision | recall | **F1 (propeptide)** |
|---|---|---|---|
| **propeptide-only (this branch)** — ESM-2, T4 | **0.717** | **0.556** | **0.626** |
| joint peptide+propeptide — ESM-2, T4 (`main`) | 0.685 | 0.527 | 0.596 |
| DeepPeptide paper (propeptides, ±3) | 0.64 | 0.46 | ~0.535 |

**Restricting the model to propeptides improves propeptide detection** over the joint
model (+0.03 F1, driven mainly by precision) and matches/exceeds the published DeepPeptide
propeptide numbers. On the test split the model predicts 1,100 propeptides against 1,420
true ones (1,538 proteins). Validation F1 peaks at ~0.76 around epochs 6–20 before a
late-training divergence; best-on-validation checkpointing saves the peak model.

**Caveats:**
- **Single split, not nested CV** — no error bars. A full 5-fold run is needed for confidence intervals.
- **Metric fix.** Upstream `get_counts_for_protein` reused a loop variable, marking matched
  true peptides at the *prediction's* index; when the matching prediction sat past the number
  of true peptides, the match was dropped, **undercounting true positives** (recall/F1
  understated). This branch fixes it, and the joint baseline above is **re-scored with the same
  fixed metric** for a fair comparison (the fix shifts values by <0.01). The original paper's
  published numbers use the unfixed metric.

---

## Training

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt   # torch >= 2.0, fair-esm / esm, pandas, numpy, tensorboard, tqdm
```

### Step 2 — Data
Two CSVs are required (provided under `data/` for the UniProt-2022 benchmark, 7,623 sequences):

- **`labeled_sequences.csv`** (indexed by `protein_id`): `sequence`, `propeptide_coordinates`
  (e.g. `(12-45),(98-113)`), `organism`.
- **`graphpart_assignments.csv`** (indexed by `AC`): `cluster` — partition index 0–4 from
  [Graph-Part](https://github.com/graph-part/graph-part).

### Step 3 — Precompute embeddings
```bash
python -m src.utils.make_embeddings data/protein_sequences.fasta PATH/TO/EMBEDDINGS/
```
Embeddings are saved as `.pt` files named by MD5 hash of the sequence. Safe to interrupt/resume.

### Step 4 — Train
Reproduce the result above with the T4 hyperparameters:
```bash
python run.py \
    --embeddings_dir PATH/TO/EMBEDDINGS \
    -df data/labeled_sequences.csv -pf data/graphpart_assignments.csv \
    --embedding_dim 1280 --epochs 50 \
    --lr 0.0055 --batch_size 20 --dropout 0.6902 --conv_dropout 0.2672 \
    --kernel_size 5 --num_filters 48 --hidden_size 48 \
    --out_dir results/esm2_prop
```

**Key arguments:**

| argument | value used | description |
|---|---|---|
| `--embedding_dim` | 1280 | ESM-2 output dimension |
| `--epochs` | 50 | training epochs (all run; **no early stopping**) |
| `--lr` | 0.0055 | **constant** learning rate (T4); default `1e-4` |
| `--batch_size` | 20 | sequences per batch |
| `--dropout` / `--conv_dropout` | 0.6902 / 0.2672 | input / conv dropout |
| `--num_filters` / `--hidden_size` / `--kernel_size` | 48 / 48 / 5 | CNN filters / biLSTM hidden / CNN kernel |
| `--out_dir` | — | where checkpoints, metrics, and TensorBoard logs are written |
| `--patience` | (ignored) | accepted for CLI compatibility; the faithful loop does not early-stop |

> **Note on stability.** At full data scale the model converges to its best validation F1
> within ~20 epochs and can diverge (loss → NaN) later. This is harmless: `model.pt` holds the
> best-on-validation checkpoint, so the reported result comes from the peak epoch regardless.
> Using `--epochs 25` reaches the same checkpoint and avoids the divergent tail.

Training writes to `--out_dir`: `model.pt` (best checkpoint), `valid_metrics.json`,
`test_metrics.json`, `test_outputs.pickle` / `valid_outputs.pickle` (raw predictions), and
TensorBoard logs (`tensorboard --logdir <out_dir>`).

### Step 5 — Evaluate
Test metrics (precision / recall / F1 at ±3 tolerance) are written automatically to
`--out_dir/test_metrics.json` when training finishes.

To (re-)score a saved checkpoint on the test split without retraining — e.g. if a run was
interrupted before its test phase, or to re-score with an updated metric:
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

---

## Predicting from a raw sequence
See the [predictor README](predictor/README.md), or use `LSTMCNNCRF.predict_from_sequence(seq)`
(embeds with ESM-2 internally and returns Viterbi propeptide spans).
