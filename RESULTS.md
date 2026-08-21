# Propeptide cleavage prediction — results

> ## ⚠ Every ESM3 row from before 2026-08-19 is INVALIDATED
>
> The ESM3 embeddings those runs consumed were extracted from the wrong tensor.
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
> of the biLSTM gates at initialisation with 65× inflated conv1 gradients.
>
> Fixed in `80db796`. Repair existing embeddings with
> `src/utils/renorm_esm3_embeddings.py` (no ESM3 re-run needed — the final norm is
> per-token, so it commutes with the BOS/EOS slice). `preflight.py` now fails any
> embedding set whose scale is more than 10× √dim.
>
> **The ESM-2 and ProstT5 rows are unaffected** and remain valid. The invalidated ESM3
> rows are kept rather than deleted because they are the provenance for the numbers
> currently in the thesis.
>
> Corroboration: arXiv 2505.20036v1 evaluates ESM3-SM-Open as a frozen feature
> extractor, reports training instability, and applies L2-normalisation "to mitigate
> training instability caused by large magnitude outputs" — the same symptom, patched
> rather than diagnosed.

> ## ⚠ Second disclaimer: the fix does not yet have a clean test-set demonstration
>
> As of 2026-08-20 there is **no matched pair** of runs differing only in the
> embeddings. The one broken-vs-fixed pair on the propeptide-only task,
> `esm3_prop_default` (0.5110) vs `esm3_prop_default_normed` (0.0440), also differs in
> `use_focal`, and the second run collapsed to the all-background solution. The case
> for the fix currently rests on the norm measurements above and on the Optuna search
> (below), **not** on a test-set improvement. `esm3_prop_T4_broken` is queued to close
> this gap.
>
> Do not compare `esm3_T4` (0.2664) against `esm3_prop_T4_normed` (0.5270). Those are
> different tasks — joint vs propeptide-only — and the difference measures the task,
> not the embeddings.

> ## ⚠ Third disclaimer: the ESM-2 propeptide-only baseline is untraceable
>
> Earlier revisions of this file reported ESM-2 propeptide-only (T4) as
> **P 0.717 / R 0.556 / F1 0.626, best val F1 0.769**. That number **cannot be traced
> to any surviving metrics file.** No archived `test_metrics.json` yields 0.626, and no
> archived `valid_metrics.json` yields 0.769 (closest: `esm2_prop_full`, val F1 0.7662).
> Four ESM-2 propeptide-only runs at lr 0.0055 exist — `esm2_propeptide_T4`,
> `esm2_prop_faithful`, `esm2_prop_full`, `esm2_prop_T4_sched` — but **none was ever
> evaluated on the test partition**, and their checkpoints were not retained.
>
> The 0.626 row has therefore been removed from the tables below. `esm2_prop_T4_test`
> is queued to re-establish the baseline. Until it lands, the best *traceable* ESM-2
> propeptide-only result is `esm2_prop_lr5e4` at **F1 0.5711**.

All numbers at **±3-residue boundary tolerance**, single Graph-Part split on the **full**
7,623-sequence benchmark (train = clusters 0,1,2 = 4,455 seqs / val = 3 / test = 4 = 1,538 seqs).

**No run is seeded.** Upstream DeepPeptide seeds nothing (`git grep -ni manual_seed upstream/main`
returns no matches) and this branch matches it, so every row is a single draw from a different
random initialisation and shuffle order. Differences below ~0.03 F1 should not be read as real
until replicates land. Replicates, not seeding, are the upstream-faithful remedy — the paper's
headline uses a 20-model ensemble (Figure S4).

Metric note: propeptide-only models use the **corrected** boundary-matching metric
(`get_counts_for_protein` fix). The joint-model rows carry the original per-run scoring; the
correction shifts values by <0.01 and does not change any ranking.

## Training recipe and deviations from upstream

Upstream DeepPeptide trains for a fixed epoch budget with a **constant Adam LR**, keeps the
**best-on-validation checkpoint**, and has **no scheduler, no early stopping and no focal term**.
As of 2026-08-21 every branch matches that recipe by default.

Before that date the branches had drifted apart, and — worse — a run's `config.json` did not
record which recipe it used. `--patience` was accepted and **silently ignored** on `esm3-full`
and `esm2-propeptide`, so those configs read `"patience": 10` for runs that trained the full
budget. What a run actually did was determined by the branch state on the day it ran.

| branch | early stopping (before) | scheduler (before) | now |
|---|---|---|---|
| `esm3-propeptide` | active, patience 10 | none | `--patience` default 0 |
| `esm3-propeptide-optuna-gpu` | active, patience 10 | none | default 0; `--search_patience 10` inside the search only |
| `esm3-full` | **flag accepted, never implemented** | none | implemented, default 0 |
| `esm2-propeptide` | **flag accepted, never implemented** | none | implemented, default 0 |
| `prost5-propeptide` | active, patience 10 | warmup + cosine (`LambdaLR`) | scheduler removed (`be28af3`); `--patience` default 0 |
| `prost5-full` | active, patience 10 | warmup + cosine (`LambdaLR`) | scheduler removed; patience default pending |

**Why no early stopping for reported models.** Best-checkpoint selection *is* early stopping: both
recipes return the argmax-on-validation model. A patience break can therefore only return an equal
or worse model than the full budget — it stops paying for epochs that might still have improved. It
cost a real result once: `esm3_prop_default_normed` was stopped at patience 10 while its training
loss was still falling monotonically.

**Why the search keeps it.** The Optuna objective trains 4 inner-fold models per trial, and outer
fold 0 alone cost 4d23h *with* the break. `--search_patience` (default 10) applies inside the
objective only and is restored afterwards, so it ranks configurations without touching the
retrained, reported model. Early stopping and pruning inside a hyperparameter search are standard
practice; the deviation is declared, not hidden.

**`--use_focal` now defaults to False** on `esm3-propeptide` (it defaulted to True). Upstream trains
on the CRF negative log-likelihood alone. The old default is why `esm3_prop_default` silently picked
up the focal term — the confound flagged in the second disclaimer above.

**Which existing rows this invalidates: almost none.** Once a run diverges to `nan`,
`score > best_score` is `False` forever, so the checkpoint freezes at the pre-divergence peak and
the remaining epochs cannot change the result. For a diverged run the two recipes are provably
identical. `esm3_prop_T4_normed` (peak ep 12, `nan` ep 13) and `esm3_prop_T4_lr1e3` (peak ep 5,
`nan` ep 6) are therefore unaffected, as are all `esm3-full` and `esm2-propeptide` rows, which
already trained the full budget. The exceptions are `esm3_prop_default_normed` (stopped while still
improving) and every ProstT5 row (trained under a scheduler).

Other standing deviations, unchanged: gradient clipping is applied **after** `backward()` (upstream
calls `clip_grad_norm_` before it, making upstream's clipping a no-op); `shuffle=True` on the train
loader; `Dropout2d` → `Dropout1d`; FSDP dropped; two metric bugs fixed in `manuscript_metrics.py`.
No run is seeded, matching upstream.

## Propeptide-only models (2-label, 51-state)

| run | embedder | embeddings | lr | focal | P | R | **F1** | status |
|---|---|---|---|---|---|---|---|---|
| `esm2_prop_lr5e4` | ESM-2 | `esm2` | 5e-4 | n/a | 0.6747 | 0.4951 | **0.5711** | valid — best traceable propeptide model |
| `esm3_prop_T4_normed` | ESM3 | `esm3_normed` | 5.5e-3 | off | 0.6729 | 0.4331 | **0.5270** | valid — best ESM3 result; diverged to `nan` after ep 12, reported from the ep-12 checkpoint |
| `esm2_prop_lr1e4` | ESM-2 | `esm2` | 1e-4 | n/a | 0.6321 | 0.4465 | 0.5233 | valid |
| ~~`esm3_prop_default`~~ | ESM3 | `esm3` | 1e-4 | **on** | 0.5632 | 0.4676 | ~~0.5110~~ | ⚠ INVALIDATED (pre-norm embeddings). Also the only run with `use_focal=true` — a second confound |
| `esm3_prop_T4_lr1e3` | ESM3 | `esm3_normed` | 1e-3 | off | 0.7655 | 0.3655 | 0.4948 | valid; diverged after ep 5, reported from the ep-5 checkpoint |
| `fold4` | — | — | — | n/a | 0.3865 | 0.4124 | 0.3990 | ⚠ no config — provenance unknown, excluded from all comparisons |
| `esm3_prop_default_normed` | ESM3 | `esm3_normed` | 1e-4 | off | 0.0311 | 0.0754 | 0.0440 | valid but degenerate — collapsed to the all-background solution |

Four further ESM-2 propeptide-only runs at lr 5.5e-3 (`esm2_propeptide_T4`, `esm2_prop_faithful`,
`esm2_prop_full`, `esm2_prop_T4_sched`) and one at lr 1e-3 (`esm2_prop_lr1e3`) have
`valid_metrics.json` only — **never evaluated on the test partition.** Their validation F1 values
are 0.7116, 0.7558, 0.7662, 0.5933 and 0.7296 respectively.

## Joint models (3-label, 101-state) — reference

| run | embedder | embeddings | lr | prop P | prop R | prop F1 | status |
|---|---|---|---|---|---|---|---|
| `esm2_T4` | ESM-2 | `esm2` | 5.5e-3 | 0.6824 | 0.5201 | **0.5903** | valid |
| `esm2` | ESM-2 | `esm2` | 1e-4 | 0.6875 | 0.3483 | 0.4624 | valid |
| `older_runs_main_esm2` | ESM-2 | `esm2` | 1e-4 | 0.4676 | 0.3761 | 0.4169 | valid, superseded |
| ~~`esm3`~~ | ESM3 | `esm3` | 1e-4 | 0.5688 | 0.4391 | ~~0.4956~~ | ⚠ INVALIDATED (pre-norm embeddings) |
| ~~`esm3_T4`~~ | ESM3 | `esm3` | 5.5e-3 | 0.6211 | 0.1696 | ~~0.2664~~ | ⚠ INVALIDATED. The old "T4 breaks ESM3" claim rested on this row |
| `prost5` | ProstT5 | `prost5` | 1e-4 | 0.3069 | 0.1993 | 0.2417 | valid — extraction verified to use the `<AA2fold>` prefix |

## Learning rate behaviour after the fix

The optimum **moved**, which is what a ~840× input-scale error predicts — off-scale inputs
inflate the effective step size, so the broken embeddings tolerated (and needed) a much smaller lr:

| lr | ESM3 broken | ESM3 fixed |
|---|---|---|
| 1e-4 | 0.4956 (joint) / 0.5110 (prop, focal on) | 0.0440 — underfits, collapses to all-background |
| 1e-3 | — | 0.4948 |
| 5.5e-3 | 0.2664 (joint) | 0.5270 |

This retires the earlier finding that "ESM3's optimal lr collapses to 3e-4–1e-3" — that was an
artifact of the unnormalised features.

Both fixed-embedding runs at lr ≥ 1e-3 reach val F1 ≈ 0.67–0.68 and then diverge to `nan`
(A: peak ep 12, diverged ep 13; C: peak ep 5, diverged ep 6). Divergence timing is **not**
monotonic in lr, which is expected given that the runs are unseeded. Gradient clipping is applied
correctly (`clip_grad_norm_(…, 0.25)` after `backward()`, `src/train_loop_crf.py:170`), but the
optimizer is Adam, whose per-parameter step is ≈ `lr` regardless of gradient magnitude, so
clipping does not bound the update. Best-on-validation checkpointing means the divergence costs
no result.

## Optuna search (outer fold 0, GPU)

The search is being re-run on repaired embeddings. TPE is seeded at 42 with
`n_startup_trials=10`, and the space is unchanged, so **trials 0–9 propose identical
hyperparameters** to the invalidated run — a genuine paired comparison.

First six paired trials:

| trial | lr | old (broken) | new (fixed) | Δ |
|---|---|---|---|---|
| 0 | 5.61e-4 | 0.5166 | 0.5556 | +0.039 |
| 1 | 5.40e-3 | 0.3758 | 0.4766 | +0.101 |
| 2 | 2.31e-4 | 0.5009 | 0.4858 | −0.015 |
| 3 | 1.90e-4 | 0.4870 | 0.4687 | −0.018 |
| 4 | 1.53e-3 | 0.2997 | 0.3048 | +0.005 |
| 5 | 4.14e-3 | 0.4365 | 0.3291 | −0.107 |
| **mean** | | **0.4361** | **0.4367** | **+0.0007** |

**The mean is unchanged.** What has moved is the ceiling: the new run's best after 6 trials
(0.5556) already exceeds the old run's best over all 30 (0.5450, trial 21). Search trial values
are mean F1 over the 4 **inner** folds and are not comparable to the single-split test F1 in the
tables above.

The old search's per-trial record is in `evaluation/optuna_trials_outer0_INVALIDATED.csv`.

## Queued runs

| run | purpose |
|---|---|
| `esm3_prop_T4_broken` | broken embeddings at lr 5.5e-3 — the missing control that makes the embedding fix a measured result |
| `esm3_prop_T4_normed_rep2`, `_rep3` | replicates of the headline ESM3 number, for an error bar |
| `esm3_full_T4_normed` | joint model on fixed embeddings — pairs with `esm3_T4` and `esm2_T4` |
| `esm2_prop_T4_test` | ESM-2 propeptide-only at lr 5.5e-3 **with test metrics** — re-establishes the untraceable 0.626 baseline |
| `prost5_prop_T4` | ProstT5 under the unified recipe — the archived 0.2417 used a scheduler and is not comparable |

All queued runs use the unified recipe (`--patience` 0, `--no-use_focal`), so they are directly
comparable to each other and to every row above that trained the full budget.

## How runs are classified

Every row above is classified from its own `config.json`, not from its name:

- `label_type: multistate_with_propeptides` → joint model (101 states). Absent → propeptide-only (51 states).
- `embeddings_dir` ending `esm3` → pre-norm, invalid. Ending `esm3_normed` → repaired.
- `use_focal` → `esm3_prop_default` is the only `true`; treat any comparison against it as confounded.
- `patience` → **only meaningful for runs after 2026-08-21.** Before that it was accepted and
  ignored on `esm3-full` and `esm2-propeptide`, so an older config recording `"patience": 10` may
  describe a run that trained the full budget. For older runs the branch state on the run date
  decides, not the config.

Run names do **not** encode the focal setting or the task, which is how the
`esm3_T4`-vs-`esm3_prop_T4_normed` mistake was originally made.
