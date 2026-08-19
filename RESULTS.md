# Propeptide cleavage prediction — results

> ## ⚠ Every ESM3 row below is INVALIDATED (2026-08-19)
>
> The ESM3 embeddings these runs consumed were extracted from the wrong tensor.
> `src/utils/make_embeddings.py` saved `ESMOutput.embeddings`, which is ESM3's raw
> **pre-LayerNorm residual stream**, not the representation its own output heads use.
> fair-esm does the opposite for ESM-2 (it applies `emb_layer_norm_after` and
> overwrites `representations[33]`), so **the ESM-2 baseline trained on normalised
> features and ESM3 did not.**
>
> Measured per-token L2 norm on six sequences from `data/labeled_sequences.csv`:
>
> | features | per-token ‖x‖ |
> |---|---|
> | ESM-2 L33 (the baseline) | 10.12 |
> | ESM3 after `transformer.norm` | 11.62 |
> | ESM3 `.embeddings`, as saved | **9792.73** |
>
> ~840× too large. `LSTMCNN` has no input normalisation, so this saturates **90.7%**
> of the biLSTM gates at initialisation with 65× inflated conv1 gradients. That, not
> the embedder, explains the collapsed optimal learning rate, "T4 breaks ESM3", the
> precision/recall split, and the flat 0.49–0.545 plateau across the whole search.
>
> Fixed in `80db796`. Repair existing embeddings with
> `src/utils/renorm_esm3_embeddings.py` (no ESM3 re-run needed — the final norm is
> per-token, so it commutes with the BOS/EOS slice). `preflight.py` now fails any
> embedding set whose scale is more than 10× √dim.
>
> **The ESM-2 and ProstT5 rows are unaffected** and remain valid. The ESM3 rows are
> kept rather than deleted because they are the provenance for the numbers currently
> in the thesis.
>
> Corroboration: arXiv 2505.20036v1 evaluates ESM3-SM-Open as a frozen feature
> extractor, reports training instability, and applies L2-normalisation "to mitigate
> training instability caused by large magnitude outputs" — the same symptom, patched
> rather than diagnosed.

All numbers at **±3-residue boundary tolerance**, single Graph-Part split on the **full**
7,623-sequence benchmark (train = clusters 0,1,2 = 4,455 seqs / val = 3 / test = 4 = 1,538 seqs).

Metric note: propeptide-only models use the **corrected** boundary-matching metric
(`get_counts_for_protein` fix). The joint-model rows carry the original per-run scoring; the
correction shifts values by <0.01 and does not change any ranking. The joint ESM-2 (T4) row is
re-scored with the fixed metric for a like-for-like comparison to the propeptide-only models.

## Propeptide-only models (2-label, 51-state) — corrected metric

| embedder | hyperparameters | precision | recall | **F1** | best val F1 | notes |
|---|---|---|---|---|---|---|
| **ESM-2** | T4 (tuned) | 0.717 | 0.556 | **0.626** | 0.769 | valid — best propeptide model to date |
| ~~ESM3~~ | default | 0.563 | 0.468 | ~~0.511~~ | 0.655 | ⚠ INVALIDATED (pre-norm embeddings, `80db796`). Also ran with `use_focal=true` while the ESM-2 row did not — a second confound |
| ~~ESM3~~ | tuned | — | — | — | — | ⚠ WITHDRAWN. The fold-0 search (mean test F1 0.4328) ran on pre-norm embeddings; its conclusions do not hold |

## Joint models (3-label, 101-state) — reference (embedder comparison)

| embedder | hyperparameters | prop P | prop R | prop F1 | metric |
|---|---|---|---|---|---|
| ESM-2 | T4 (tuned) | 0.685 | 0.527 | 0.596 | corrected — valid |
| ESM-2 | default | 0.688 | 0.348 | 0.462 | original — valid |
| ~~ESM3~~ | default | 0.569 | 0.439 | ~~0.496~~ | ⚠ INVALIDATED (`80db796`) |
| ~~ESM3~~ | ESM-2's T4 | 0.621 | 0.170 | ~~0.266~~ | ⚠ INVALIDATED (`80db796`). "T4 breaks ESM3" was the embedding scale, not the hyperparameters |
| ProstT5 | default | 0.307 | 0.199 | 0.242 | original — valid (HF T5 applies `final_layer_norm` before `last_hidden_state`) |

## Key comparisons

These all depended on at least one invalidated ESM3 number and are withdrawn pending
re-runs on normalised embeddings:

- ~~**Specialisation helps both embedders**: ESM-2 (T4) 0.626 vs 0.596 → +0.030;
  ESM3 (default) 0.511 vs 0.496 → +0.015~~ — the ESM-2 half stands; the ESM3 half does not.
- ~~**ESM3 is under-tuned**: a tuned ESM3 propeptide model plausibly reaches ~0.6+~~ —
  this was the wrong diagnosis. ESM3 was not under-tuned; it was being fed features
  ~840× out of scale, which is *why* every learning rate the search tried looked bad.
- **Best propeptide model:** ESM-2 propeptide-only, T4 → **F1 0.626**. Still stands.
- **Caveat:** single split — no error bars yet.

## What has to be re-run

| run | branch | status |
|---|---|---|
| ESM3 propeptide, defaults (`--no-use_focal`) | `esm3-propeptide` (CPU) | replaces the 0.511 row |
| ESM3 propeptide, T4 hyperparameters | `esm3-propeptide` (CPU) | new — direct like-for-like against ESM-2's 0.626 |
| ESM3 joint, defaults and T4 | `esm3-full` (CPU) | replaces 0.496 and 0.266 |
| ESM3 propeptide, fold-0 Optuna search | `esm3-propeptide-optuna-gpu` (GPU) | replaces the withdrawn tuned row — **use a fresh `--out_dir`**, the persisted study will otherwise resume the 30 stale trials and run none |
