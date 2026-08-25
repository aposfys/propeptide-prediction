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

> ## ✅ RESOLVED 2026-08-21: the fix now has a clean test-set demonstration
>
> `esm3_prop_T4_broken` closed this gap. It is a **fully matched pair** with
> `esm3_prop_T4_normed` — same branch, same task, same lr 5.5e-3, same
> `use_focal=false`, same `patience=10`; only `embeddings_dir` differs:
>
> | embeddings | test F1 |
> |---|---|
> | `esm3` (raw pre-LayerNorm) | 0.3274 |
> | `esm3_normed` (repaired) | **0.5270** |
> | | **+0.1996** |
>
> The Optuna search corroborates it under the paper's own ablation protocol:
> 0.4328 → **0.4800** on outer fold 0 (below).
>
> Still true, and still the trap that caught this project twice: do not compare
> `esm3_T4` (0.2664) against `esm3_prop_T4_normed` (0.5270). Those are different
> tasks — joint vs propeptide-only — and the difference measures the task, not the
> embeddings.

> ## ✅ RESOLVED 2026-08-21: the ESM-2 baseline is re-established, and it was real
>
> Earlier revisions reported ESM-2 propeptide-only (T4) as **F1 0.626**, which could
> not be traced to any surviving metrics file — no archived `test_metrics.json`
> yielded it, and none of the four ESM-2 propeptide runs at lr 0.0055 had ever been
> evaluated on the test partition.
>
> `esm2_prop_T4_test` re-ran it and landed at **F1 0.6307**, within 0.005 of the lost
> figure. The original number was evidently genuine; only its provenance was missing.
> It is now reproducible from a committed `test_metrics.json`.

All numbers at **±3-residue boundary tolerance**, single Graph-Part split on the **full**
7,623-sequence benchmark (train = clusters 0,1,2 = 4,455 seqs / val = 3 / test = 4 = 1,538 seqs).

**No run is seeded.** Upstream DeepPeptide seeds nothing (`git grep -ni manual_seed upstream/main`
returns no matches) and this branch matches it, so every row is a single draw from a different
random initialisation and shuffle order. That spread is now **measured, not assumed**: fifteen
replicates of one configuration give **sd 0.0185** (see the headline section), so a single-run
difference below ~0.037 (2σ) should not be read as real. Replicates, not seeding, are the
upstream-faithful remedy — the paper's headline uses a 20-model ensemble (Figure S4).

The fine-tuning arm (`finetune.py`) is the exception and seeds deliberately: adapting the
encoder adds its own initialisation noise on top of the head's, and the seed is recorded in
each run's `config.json`.

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

## Headline result

| embedder | run | P | R | **test F1** |
|---|---|---|---|---|
| ESM-2 (650M), frozen | `esm2_prop_T4_test` | — | — | **0.6307** |
| **ESM3 (1.4B), LoRA fine-tuned** | `ft_b12_lr5e-5` | 0.6550 | **0.5401** | **0.5920** |
| ESM3 (1.4B), frozen — replicate mean (n=15) | `esm3_ens/model_*` | — | — | 0.5203 ± 0.0185 |
| ESM3 (1.4B), frozen — frozen control, FT code path | `ft_frozen_control` | 0.7312 | 0.4176 | 0.5316 |

**Frozen ESM-2 still leads, by 0.039 (2.1σ).** But fine-tuning closed most of the gap:
from 0.110 against the frozen replicate mean to 0.039, and ESM3+LoRA now beats one ESM-2
configuration outright (`esm2_prop_lr5e4`, 0.5711).

### The run-to-run spread is 0.0185, measured

Fifteen replicates of `esm3_T4_b70_wd`'s exact configuration, differing only in random
initialisation: **mean 0.5203, sd 0.0185, min 0.4881, max 0.5445.**

Two consequences, both of which change earlier readings in this file:

1. **The 0.5547 previously headlined here is a fortunate draw.** It sits *above the
   maximum of fifteen replicates of its own configuration*. It was selected as the best of
   ~20 configs, and that selection inflated it by roughly two standard deviations. ESM3's
   honest frozen number is **0.5203 ± 0.0185**.
2. Earlier text assumed a ~0.03 spread. The measured value is smaller, so margins are
   more significant than previously stated — a single-run difference needs ~0.037 (2σ).

### Fine-tuning: +0.060 over its own control, entirely in recall

| | test F1 | P | R |
|---|---|---|---|
| ESM3 + LoRA (12 blocks, r=8, lr 5e-5) | **0.5920** | 0.6550 | 0.5401 |
| ESM3 frozen, same code path | 0.5316 | 0.7312 | 0.4176 |
| | **+0.0604 (3.3σ)** | −0.076 | **+0.123** |

Every frozen ESM3 result in this project lost on recall at high precision. Fine-tuning
traded 0.076 of precision for 0.123 of recall. ESM3+LoRA's recall (0.5401) now **exceeds
ESM-2's** (0.4950 for `esm2_prop_lr5e4`) — the first time that has happened.

**The comparison is LoRA-adapted ESM3 vs frozen ESM-2, and every caption must say so.**
ESM-2 has had neither adapters nor the batch/weight-decay treatment. The experiment asks
whether adaptation closes a measured representation gap, not which encoder wins outright.

⚠ σ = 0.0185 was measured on the *frozen* configuration. The adapted arm's own spread is
unmeasured, and adapters could plausibly add variance. Replicates before any final claim.

### The fine-tuning path is validated

`ft_frozen_control` (`--lora_blocks 0`) runs the identical code path with no adapters:
**0.5316** against the cached-embedding pipeline's **0.5280** at the same configuration —
a difference of 0.0036, 0.19σ, with recall matching to four decimal places. So
tokenisation, padding, the residue mask, the input LayerNorm, `max_len=2048` and bf16
autocast together shifted the result by less than a fifth of the noise floor, and any
difference the adapters produce is attributable to the adapters. Three protocols agree on the direction:

| protocol | ESM-2 | ESM3 |
|---|---|---|
| single split, matched pair (this file) | **0.6307** | 0.5270 |
| paper's ablation protocol, outer fold 0, tuned | 0.503–0.564 (upstream's own `crf_model_all_cv_balancedsplit.csv`) | **0.4800** |
| paper's propeptide-only figure (S7) | 0.535 | 0.4800 |

The nested-CV row is corroboration, not proof: 0.4800 ± 0.0172 is a spread over 4 inner
models, whereas upstream's 0.503–0.564 spans outer folds, so their fold-to-fold variance
(~0.06) exceeds the gap. The single-split matched pair is what carries the conclusion.

**A mechanism, from the ESM3 authors.** Hayes et al. 2025 state that "high masking rates
improve the generative capability, while lower masking rates improve representation
learning", and that ESM3 was trained on "a noise schedule that **balances** generative
capabilities with representation learning". ESM3's representation quality is a deliberate
trade against generative ability; ESM-2 made no such trade. That predicts the direction
measured here without invoking any defect.

## Propeptide-only models (2-label, 51-state)

| run | embedder | embeddings | lr | focal | P | R | **F1** | status |
|---|---|---|---|---|---|---|---|---|
| `esm2_prop_T4_test` | ESM-2 | `esm2` | 5.5e-3 | off | — | — | **0.6307** | valid — **best model**; re-establishes the previously untraceable 0.626 |
| `esm2_prop_lr5e4` | ESM-2 | `esm2` | 5e-4 | n/a | 0.6747 | 0.4951 | 0.5711 | valid |
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

## ESM3 optimisation sweep

Roughly a dozen configurations were tried, on GPU, to establish whether ESM3's deficit was
a tuning artifact. It is not — but two knobs did help, and both are omitted from Table S1.

**Divergence was the binding constraint, and batch size fixes it.** Every ESM3 run at
Table S1's batch 20 explodes to `nan`; the epoch it survives to tracks batch size:

| config | stable epochs | peak val | test F1 |
|---|---|---|---|
| T4, batch 20, wd 0 | 12 | 0.7000 | 0.5280 |
| T4, batch 20, wd 1e-4 | 6 | 0.6814 | — |
| **T4, batch 70, wd 1e-4** | **21** | **0.7045** | **0.5547** ⚠ |
| T4, batch 100, wd 1e-4 | 35 | 0.6979 | 0.5246 |

⚠ 0.5547 is an outlier, not this configuration's expected value. Fifteen replicates of it
give mean 0.5203, sd 0.0185, **max 0.5445** — 0.5547 exceeds all fifteen. Selecting the best
of ~20 configurations inflated it by roughly 2σ. Use 0.5203 ± 0.0185 as the frozen-ESM3
number and treat the rows below as single draws from that distribution.

Weight decay is worth ~+0.026 at matched settings (0.4911 → 0.5167 on the tuned
architecture) and, combined with batch 70, delays divergence from epoch 6 to epoch 22.
Batch 100 is stabler still but trains too slowly per epoch to reach as high an optimum.

**The Optuna-tuned architecture underperforms Table S1's.** The 30-trial search
(`esm3_prop_optuna_normed`, repaired embeddings, `space: table_s1`) selected
`hidden_size` 16, `kernel_size` 1, `lr` 1.77e-3 — which scores 0.5167 on the single split
versus T4's 0.5547. It was selected on the inner-fold CV mean, a different protocol with a
smaller training set, and does not transfer.

**The search space is mis-centred for ESM3**, worth noting even though re-searching was not
affordable: the fold-0 winner pinned three of seven parameters to a boundary —
`hidden_size` 16 (min), `kernel_size` 1 (min), `dropout` 0.696 (max ≈ 0.7). The search
wants less capacity and more regularisation than Table S1 permits.

**Validation tops out at ~0.70 regardless.** Five distinct configurations span just 0.011
on validation (0.6935–0.7045), while ESM-2 reaches 0.7662 without any of this effort. The
gap is not a tuning artifact.

## Layer selection — the last fairness gap, now closed

Upstream **tuned** ESM-2's layer, sweeping it and choosing 33 of 33 (btad616 Fig. S9; their
`crf_model_all_cv_balancedsplit.csv` gives L32 0.494–0.542 against L33 0.503–0.564). ESM3
had only ever been read at layer 48 of 48, chosen by analogy. That was the one respect in
which ESM-2 received treatment ESM3 did not — and unlike batch size or weight decay, it
could have narrowed the gap rather than lifting both models.

Extraction from arbitrary blocks was added in `bc8157a` (`--layer`), via a forward hook on
`TransformerStack`, whose `forward` returns `self.norm(x), x, hiddens` but whose caller
discards the third value. Verified: `--layer 48` reproduces the default path **bit-for-bit**
on 20/20 sampled sequences.

| layer | peak val | test F1 | behaviour |
|---|---|---|---|
| **48** (final) | 0.7045 | **0.5547** | diverges at ep 22 |
| 44 | **0.7064** | 0.5388 | diverges at ep 28 |
| 30 | 0.6558 | 0.5351 | never diverges — 47 stable epochs |

Layers 44 and 48 are indistinguishable, and revealingly the two partitions rank them
**oppositely**: validation favours 44 by 0.0019, test favours 48 by 0.0159. Both margins sit
well inside the measured run-to-run spread (sd 0.0185, n=15). Reported as a negative
control, not a selection procedure: layer 48 was the a-priori choice, fixed before the sweep
as the analogue of ESM-2's layer 33, and the sweep asked only whether another layer beats
it. It does not.

**Layer 48 is ESM3's best, as layer 33 is ESM-2's.** Both models are read at their final
layer, both by measurement rather than assumption, so the comparison is fair in this
respect too.

A mechanistic aside worth reporting: layer 30 trained completely stably (loss monotonically
20.85 → 3.26 over 47 epochs, stopped by a plateau rather than a collapse) while layer 48
diverges. The later blocks carry the task-relevant signal *and* the instability.

## Multimodal ESM3 (branch `esm3-multimodal-propeptide`)

Every ESM3 result above was produced with `sequence_tokens=` alone: `make_embeddings.py`
never passed the structure, SASA, ss8 or function tracks, so **ESM3's multimodality was
never engaged**. That was not a decision — it is what a drop-in replacement for a
sequence-only ESM-2 pipeline produces, and the dataset carries no 3D data.

`make_embeddings_esm3_struct.py` adds every track that cannot leak the label, conditioned
on AlphaFold DB models (98.7% coverage over the 8,449 accessions; 8/8 sampled sequences
matched exactly): `structure_coords` via Geometric Attention, `structure_tokens`, and
`sasa_tokens`. The function and residue-annotation tracks are **excluded** — the paper
states they encode InterPro/GO keywords and "catalytic sites and post-translational
modifications", and propeptide cleavage is a PTM annotated in the very UniProt records
these labels come from.

### The pipeline reproduces, so the comparison is attributable

The `--no_structure` control runs the identical script on the identical GPU with every
structural track masked. Its best result matches the HPC's sequence-only number to
**0.001**:

| | best test F1 |
|---|---|
| `esm3_prop_T4_normed` — HPC, CPU, `make_embeddings.py` | 0.5270 |
| `esm3_seqonly_gpu_lr0.0005` — Rucker, GPU, `make_embeddings_esm3_struct.py --no_structure` | **0.5280** |

Two machines, two scripts, two devices, one number. That retires the CPU-vs-GPU worry for
this pipeline and means any difference in the structure runs is attributable to the
structure tracks rather than to the extraction change.

### Structure conditioning makes it worse

Both variants were swept over lr, so best-of-sweep is the fair comparison:

| ESM3 variant | best test F1 | P | R |
|---|---|---|---|
| sequence-only | **0.5280** | 0.7179 | 0.4176 |
| + structure + SASA | 0.4712 | 0.7269 | 0.3486 |
| | **−0.0568** | +0.009 | **−0.069** |

**The entire loss is in recall** — 0.4176 → 0.3486, a 17% relative drop, with precision
flat. Structural conditioning makes the model more conservative: it predicts fewer
propeptides and misses more real ones.

Full sweep:

| run | embeddings | lr | P | R | **F1** |
|---|---|---|---|---|---|
| `esm3_seqonly_gpu_lr0.0005` | seq-only | 5e-4 | 0.7179 | 0.4176 | **0.5280** |
| `esm3_struct_lr0.0005` | +struct | 5e-4 | 0.7269 | 0.3486 | 0.4712 |
| `esm3_struct_T4` | +struct | 5.5e-3 | 0.6835 | 0.3528 | 0.4654 |
| `esm3_struct_lr0.001` | +struct | 1e-3 | 0.7214 | 0.3282 | 0.4511 |
| `esm3_seqonly_gpu_lr0.0055` | seq-only | 5.5e-3 | 0.6278 | 0.3373 | 0.4388 |

One caveat stated plainly: at the *matched* lr of 5.5e-3 the structure model is slightly
**better** (0.4654 vs 0.4388). But 5.5e-3 diverges by epoch 6 on both sets, so neither is
trained there. At 5e-4, where both train stably for 6–9 epochs, sequence-only wins clearly.

### Notes

**5e-4 is the better learning rate for ESM3**, not the 5.5e-3 inherited from Table S1.
Peak validation F1 was **0.7000** at 5e-4 versus 0.5619 at 5.5e-3 on identical data.

**Divergence is an ESM3 property, not a structure one.** Every run above eventually
explodes to `nan`, on both embedding sets, at every lr tested. Best-on-validation
checkpointing means it costs no result, but it is worth reporting.

**ESM3 loads bfloat16 on CUDA and float32 on CPU.** All embeddings compared here are
float32, forced explicitly on the GPU side (`dde0444`). Left unforced, a GPU extraction
would silently differ in precision from the CPU baseline.

**Geometric Attention is unconditional.** It sits in the first transformer block and
allocates an L×L×heads×3 tensor whether or not coordinates are passed — 37.6 GiB at
L≈3,600, which OOMs a 31 GB card. The 15 sequences over 2,000 residues were embedded on
CPU (`--gpu_max_len 2000`), giving identical float32 output.

## Fine-tuning ESM3 (branch `esm3-multimodal-propeptide`)

Every ESM3 number above measures **frozen-feature quality** through a ~458k-parameter head.
No ESM3 parameter had ever been trained on this task, so "does ESM3 improve when its
parameters are allowed to move" had never been asked. `finetune.py` asks it.

Setup: LoRA r=8 α=16 on the last 12 blocks (2.36M trainable, 0.17% of 1.4B), head LR
5.5e-3 with adapters two orders below, 3-epoch head-only warm start, 5% linear warmup,
AdamW, effective batch 72 by accumulation, length-bucketed, `max_len` 2048, bf16 autocast
on the encoder only (the CRF's log-space DP stays fp32), seeded.

| run | test F1 | P | R | best val | epoch |
|---|---|---|---|---|---|
| `ft_b12_lr5e-5` | **0.5920** | 0.6550 | 0.5401 | 0.7292 | 19 |
| `ft_frozen_control` | 0.5316 | 0.7312 | 0.4176 | 0.6590 | 10 |

Val 0.7292 is the highest any ESM3 model has reached in this project (previous best 0.7064)
and exceeds the archived ESM-2 propeptide validation (0.6930, `esm2_prop_lr5e4`).

Four ESM3-specific facts had to be right, each verified against `esm 3.2.1` rather than
inferred, and two of them break a naive script:

- LoRA targets are `attn.layernorm_qkv.1`, `attn.out_proj`, `ffn.1`, `ffn.3`. In ESM3
  `layernorm_qkv` and `ffn` are `nn.Sequential`, so the ESM-2 idiom
  `target_modules=['layernorm_qkv']` matches a container and attaches nothing.
- Geometric attention runs in block 0 unconditionally and allocates an L×L×heads tensor
  whether or not structure is passed. For sequence-only input it contributes exactly zero,
  so `use_geom_attn = False` is bit-exact and frees 3.0 GiB at L=1024 — the fix that
  retires the `--gpu_max_len` CPU fallback.
- The tokenizer emits `input_ids`, not `sequence_tokens` (that is the name of ESM3's
  *forward argument*), and it **right-pads** — so a residue mask taken as
  `attention_mask[:, 1:-1]` marks each short sequence's EOS as a valid residue. The mask is
  built from true sequence lengths.
- `ESM3.from_pretrained` returns bfloat16 on CUDA and float32 on CPU. Master weights are
  forced to fp32 so the frozen control stays numerically comparable to the cached runs.

## Queued runs

| run | purpose |
|---|---|
| `ft_b12_lr1e-4`, `ft_b12_lr3e-4` | the rest of the LoRA learning-rate sweep |
| replicates of the winning LoRA config | σ=0.0185 is the FROZEN arm's spread; the adapted arm's is unmeasured, and the headline claim rests on it |
| ESM-2 + batch 70 + weight decay | ESM-2 has only ever run at Table S1 defaults; the comparison is asymmetric until it gets the same treatment |
| `esm2_full_T4` | ESM-2 joint model — `esm3_full_T4_normed` is uninterpretable without it, since the archived `esm2_T4` predates the metric fix and real gradient clipping |
| `esm3_prop_T4_normed_rep2`, `_rep3` | replicates of the ESM3 number, for an error bar |
| `esm2_prop_T4_test_rep2` | replicate of the **winning** row — worth more than a second draw of the loser |
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
