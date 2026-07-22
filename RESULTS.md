# Propeptide cleavage prediction — results

All numbers at **±3-residue boundary tolerance**, single Graph-Part split on the **full**
7,623-sequence benchmark (train = clusters 0,1,2 = 4,455 seqs / val = 3 / test = 4 = 1,538 seqs).

Metric note: propeptide-only models use the **corrected** boundary-matching metric
(`get_counts_for_protein` fix). The joint-model rows carry the original per-run scoring; the
correction shifts values by <0.01 and does not change any ranking. The joint ESM-2 (T4) row is
re-scored with the fixed metric for a like-for-like comparison to the propeptide-only models.

## Propeptide-only models (2-label, 51-state) — corrected metric

| embedder | hyperparameters | precision | recall | **F1** | best val F1 | notes |
|---|---|---|---|---|---|---|
| **ESM-2** | T4 (tuned) | 0.717 | 0.556 | **0.626** | 0.769 | best propeptide model to date |
| **ESM3** | default | 0.563 | 0.468 | **0.511** | 0.655 | under-tuned (lr 1e-4); early-stopped @ ep46 |
| ESM3 | tuned | — | — | _pending_ | — | run at lr ≈ 5e-4 to close the comparison |

## Joint models (3-label, 101-state) — reference (embedder comparison)

| embedder | hyperparameters | prop P | prop R | prop F1 | metric |
|---|---|---|---|---|---|
| ESM-2 | T4 (tuned) | 0.685 | 0.527 | 0.596 | corrected |
| ESM-2 | default | 0.688 | 0.348 | 0.462 | original |
| ESM3 | default | 0.569 | 0.439 | 0.496 | original |
| ESM3 | ESM-2's T4 | 0.621 | 0.170 | 0.266 | original (T4 breaks ESM3) |
| ProstT5 | default | 0.307 | 0.199 | 0.242 | original |

## Key comparisons

- **Specialisation helps both embedders** (propeptide-only > joint, same HPs):
  - ESM-2 (T4): 0.626 vs 0.596  →  **+0.030**
  - ESM3 (default): 0.511 vs 0.496  →  **+0.015**
- **Best propeptide model:** ESM-2 propeptide-only, T4 → **F1 0.626**.
- **ESM3 is under-tuned:** its 0.511 uses default HPs (conservative lr 1e-4); on the joint model,
  tuning lifted ESM-2 by +0.13, so a tuned ESM3 propeptide model plausibly reaches ~0.6+.
- **Caveat:** single split — no error bars yet. 5-fold CV is the next step.
