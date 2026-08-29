# Propeptide prediction

Predicting propeptide cleavage sites in protein sequences from frozen protein language
model embeddings.

MSc Bioinformatics thesis (NKUA), built on
[DeepPeptide](https://github.com/fteufel/DeepPeptide) (Teufel et al., *Bioinformatics*
2023, [`btad616`](https://doi.org/10.1093/bioinformatics/btad616)). The point is to compare
protein language models on one task: same head, same data, same splits, same metric, same
training budget, and only the embeddings change. If one arm scores better, that should be
the embeddings rather than something else in the pipeline.

The original model predicts peptides and propeptides together with 3 labels and 101 CRF
states. Here it's cut down to propeptides only: 2 labels, 51 states (1–50 propeptide, 0
background), and mature-peptide coordinates are dropped. Everything else follows the paper.
Embeddings go into a CNN–biLSTM–CNN, then a linear-chain CRF with a duration-encoded state
grammar (length 5 to 50), decoded with Viterbi. Adam at a constant learning rate, no
scheduler, keep the best checkpoint on validation.

This branch is the ESM-2 arm (`esm2_t33_650M_UR50D`, 1280 dims per residue). Runs on CPU.

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

### Things to keep in mind

**One split, one run, no error bars.** These are single point estimates. This pipeline
varies enough between repeats that you can't rank two embedding models from one run each.
The replicated comparison is the actual thesis work; the table above is just the reference
arm.

**The metric here is fixed, and that changes the numbers.** Upstream
`get_counts_for_protein` reuses one loop variable across the true and the predicted loop,
so a matched true peptide gets marked at the prediction's index instead. `groupby('group')`
drops the resulting row, and a real true positive becomes a false negative. Upstream
therefore reports recall and F1 too low, and does so more for models that emit more false
positives. This branch fixes it, and the joint baseline above is re-scored the same way.
The paper's published numbers use the old version.

**Gradient clipping.** Upstream calls `clip_grad_norm_` before `backward()`, when there are
no gradients yet, so the clip does nothing and the published models trained unclipped. This
branch clips.

## Training

### 1. Install

```bash
pip install -r requirements.txt   # torch >= 2.0, fair-esm, pandas, numpy, tensorboard, tqdm
```

### 2. Data

Two CSVs, both under `data/` for the UniProt-2022 benchmark:

- `labeled_sequences.csv`, indexed by `protein_id`: `sequence`, `propeptide_coordinates`
  (like `(12-45),(98-113)`), `organism`.
- `graphpart_assignments.csv`, indexed by `AC`: `cluster`, the partition index 0–4 from
  [Graph-Part](https://github.com/graph-part/graph-part).

### 3. Embeddings

```bash
python -m src.utils.make_embeddings data/protein_sequences.fasta PATH/TO/EMBEDDINGS/
```

One `.pt` per sequence, named by the MD5 of the sequence. Existing files are skipped, so
it's safe to interrupt and restart.

### 4. Train

This reproduces the result above:

```bash
python run.py \
    --embeddings_dir PATH/TO/EMBEDDINGS \
    -df data/labeled_sequences.csv -pf data/graphpart_assignments.csv \
    --embedding_dim 1280 --epochs 50 \
    --lr 0.0055 --batch_size 20 --dropout 0.6902 --conv_dropout 0.2672 \
    --kernel_size 5 --num_filters 48 --hidden_size 48 \
    --out_dir results/esm2_prop
```

| argument | used here | notes |
|---|---|---|
| `--embedding_dim` | 1280 | 1536 for ESM3, 1024 for ProstT5 |
| `--epochs` | 50 | |
| `--lr` | 0.0055 | constant, no scheduler. Default is `1e-4` |
| `--batch_size` | 20 | |
| `--dropout` / `--conv_dropout` | 0.6902 / 0.2672 | input and conv dropout |
| `--num_filters` / `--hidden_size` / `--kernel_size` | 48 / 48 / 5 | CNN filters, biLSTM hidden, CNN kernel |
| `--patience` | 0 | early stopping. 0 turns it off, which is upstream behaviour and the default |
| `--out_dir` | | checkpoints, metrics, TensorBoard logs |

The model usually reaches its best validation F1 within about 20 epochs and can go to NaN
later. That's harmless. The checkpoint is only written when validation improves, and once
validation collapses it never improves again, so `model.pt` holds the pre-divergence best
whatever the tail does.

You get `model.pt`, `valid_metrics.json`, `test_metrics.json`, `test_outputs.pickle` and
`valid_outputs.pickle`, plus TensorBoard logs (`tensorboard --logdir <out_dir>`). Test
metrics at ±3 are written automatically when training ends.

<details>
<summary>Re-scoring a checkpoint without retraining</summary>

Handy if a run was killed before its test phase, or to re-score with an updated metric.

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
# predicted spans per protein: convert_path_to_peptide_borders(pred, 1, 50, offset=1)
```
</details>

## Predicting on a raw sequence

See the [predictor README](predictor/README.md), or call
`LSTMCNNCRF.predict_from_sequence(seq)`, which embeds with ESM-2 internally and returns the
Viterbi propeptide spans.

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
