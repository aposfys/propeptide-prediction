# Results — ESM3 (single fold, default HPs)

Branch `esm3-full`. See [README.md](README.md) for what this branch is and [BRANCHES.md](BRANCHES.md) on `main` before comparing F1 across branches.

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
