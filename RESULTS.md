# Results — ESM-2 single-fold reproduction

Branch `baseline-upstream`. See [README.md](README.md) for what this branch is and `BRANCHES.md` on `main` before comparing F1 across branches.

All runs below use a single split (`train=[0,1,2]`, `val=[3]`, `test=[4]`), 50 epochs,
best-checkpoint-on-validation (stopping metric = mean of peptide & propeptide F1), evaluated at
±3-residue tolerance. Embeddings: ESM-2 (`esm2_t33_650M_UR50D`, **layer 33**, 1280-dim).

| run | learning rate | other hyperparameters | f1 peptides | f1 propeptides |
|---|---|---|---|---|
| default HPs | 1e-4 | dropout 0.1, conv 0.1, batch 100, kernel 3, filters 32, hidden 64 | 0.399 | 0.462 |
| **tuned (paper Table S2, fold T4)** | **5.5e-3** | dropout 0.69, conv 0.27, batch 20, kernel 5, filters 48, hidden 48 | **0.579** | **0.590** |

> **Scored with the upstream metric — do not compare these numbers directly with the other
> branches.** This branch is kept faithful to the original DeepPeptide, which means it retains
> two upstream defects the embedder branches have since fixed:
>
> - `get_counts_for_protein` reuses `idx` across the true and pred loops, so a matched true
>   peptide is marked at the *pred's* index. The resulting phantom row is dropped by
>   `groupby('group')`, converting real true positives into false negatives — **recall and F1
>   are understated**, more so the more false positives a model emits.
> - `clip_grad_norm_` is called before `backward()`, where no gradients exist yet, so
>   clipping is a no-op and these models trained **unclipped** — as the paper's did.
>
> Both are deliberate here. They are precisely why the comparison against Teufel et al. below
> is like-for-like: the published figures were produced by this same code path. The
> `esm2-propeptide`, `esm3-propeptide`, `esm3-full` and `prost5-*` branches correct both, so
> their F1 values sit on a different footing and are **not** interchangeable with these.

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
