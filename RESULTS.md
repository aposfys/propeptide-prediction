# Results

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

Per-run metrics for every arm of the comparison are collected on the
`esm3-propeptide` branch under `results/`. See [BRANCHES.md](BRANCHES.md) before
comparing F1 across branches.
