# Propeptide cleavage prediction — results

Every number in this file was produced by `src/utils/summarize_results.py` reading
`test_metrics.json` and `valid_metrics.json` under `results/`. None was copied from
a log, a notebook, or an earlier draft. The full grouped output, including the
complete pairwise matrix, is in `results_summary.json`.

Metric throughout: segment-level propeptide F1 at ±3 residue boundary tolerance,
micro-averaged over the test partition (cluster 4, 1,538 sequences). A prediction
counts only if **both** boundaries fall within tolerance. This is the fixed metric,
not upstream's — see the metric section of [EXPERIMENT.md](EXPERIMENT.md).

Groups are formed by the whole recorded configuration minus `out_dir` and `seed`,
so two runs share a group only if every other field is identical.

---

## Headline

**ESM3 does not outperform ESM-2 on propeptide prediction, and is statistically
indistinguishable from ProstT5.**

| representation | configuration | n | mean F1 | sd | min | max |
|---|---|---|---|---|---|---|
| ESM-2 | T4 (lr 0.0055, bs 20) | 10 | **0.6153** | 0.0242 | 0.5772 | 0.6510 |
| ESM3 | lr 5e-4, bs 70 | 15 | 0.5203 | 0.0185 | 0.4881 | 0.5445 |
| ProstT5 | T4 (lr 0.0055, bs 20) | 5 | 0.5189 | 0.0306 | 0.4717 | 0.5436 |

| comparison | difference | t | df | p |
|---|---|---|---|---|
| ESM-2 vs ESM3 | 0.0949 | 10.531 | 15.81 | **1.5e-08** |
| ESM-2 vs ProstT5 | 0.0964 | 6.141 | 6.60 | **0.00059** |
| ESM3 vs ProstT5 | 0.0014 | 0.098 | 5.01 | 0.926 |

Welch's unequal-variance t-test, two-sided.

### What each of these licenses

The arms did not receive equal tuning, so each comparison has to be read with its
bias direction attached.

**ESM-2 vs ESM3 is conservative.** ESM3's configuration was selected by comparing
test F1 across the in-house sweep, so its 0.5203 is optimistically biased. It lost
by 0.0949 anyway. This one can be stated plainly.

**ESM-2 vs ProstT5 is not.** It is a clean controlled swap — the two groups match
on every configuration field, and the summariser groups on the whole config, so
that is checked rather than assumed — but the configuration is *ESM-2's*, taken
from Teufel et al.'s Table S2. It licenses "ProstT5 is worse **at ESM-2's
hyperparameters**", which is an upper bound on its deficit. It does not establish
that ProstT5 is a worse representation.

**ESM3 vs ProstT5 is the expected outcome.** Two models, each near its own best
configuration, separated by 0.0014 with p = 0.926. Lucic et al. found the same
pattern across GAN architectures: once tuning budget is spent, most models reach
similar scores, and apparent differences trace to budget rather than method.

---

## The untuned comparison, and what it actually measured

All four arms at upstream's default (`lr 1e-4, bs 100, dropout 0.1, 50 epochs,
patience 0`), five seeds each. Equal configuration for every arm, including the
no-pretraining control.

| arm | n | mean F1 | sd | min | max |
|---|---|---|---|---|---|
| ESM-2 | 5 | 0.5079 | 0.0714 | 0.3853 | 0.5682 |
| ProstT5 | 5 | 0.4091 | 0.1094 | 0.2210 | 0.4974 |
| ESM3 | 5 | 0.2335 | **0.1930** | 0.0440 | 0.4438 |
| one-hot (control) | 5 | 0.2308 | 0.0901 | 0.0802 | 0.3073 |

| comparison | difference | t | df | p |
|---|---|---|---|---|
| ESM-2 vs ESM3 | 0.2744 | 2.982 | 5.07 | 0.030 |
| ESM-2 vs one-hot | 0.2771 | 5.390 | 7.60 | 0.00078 |
| ProstT5 vs ESM3 | 0.1757 | 1.771 | 6.33 | 0.124 |
| ProstT5 vs one-hot | 0.1784 | 2.815 | 7.72 | 0.024 |
| ESM3 vs one-hot | 0.0027 | 0.028 | 5.66 | 0.978 |

**This table does not rank representations, and must not be presented as if it
did.** Two facts make that clear.

First, nothing converged. Reading the validation curves off the TensorBoard
scalars, 17 of 19 completed runs reached their best validation F1 within the last
ten of 50 epochs. At `lr 1e-4` the budget ends while the models are still on a
steep part of the curve.

Second, the ESM3 arm is bimodal rather than merely noisy:

| seed | best validation F1 |
|---|---|
| 1 | 0.6111 |
| 3 | 0.6159 |
| 2 | 0.1017 |
| 4 | 0.1423 |
| 5 | 0.0451 |

Two seeds learned the task; three never left the floor. An sd of 0.1930 is
describing a coin flip, not a distribution around a mean. So `ESM3 vs one-hot,
p = 0.978` says nothing about whether ESM3 embeddings carry propeptide signal —
it says that at this learning rate, ESM3 training usually fails to start.

What the table *does* support is a robustness claim, and it is specific to ESM3:
**at upstream's default hyperparameters, ESM-2 trained on 5/5 seeds and ProstT5
recovered to 0.4091, while ESM3 failed on 3/5.** ProstT5's two additional seeds
raised its mean from the n=3 value, so the instability is not a general property
of "everything except ESM-2".

This is the design's own failure mode, and it is worth stating as a methodological
result: equalising the *configuration* across arms does not equalise the
*treatment*. A shared default leaves different representations at different points
on the optimisation curve, and for ESM3 it leaves most seeds nowhere.

---

## Fine-tuning ESM3 with LoRA

ESM3 only. `src/models/plm_backbone.py` instantiates `ESM3.from_pretrained`
directly, so this path cannot fine-tune ESM-2 or ProstT5, and nothing here
generalises to them.

| group | n | mean F1 | sd | mean validation F1 |
|---|---|---|---|---|
| LoRA lr 3e-4 | 1 | 0.5972 | — | 0.7209 |
| LoRA lr 1e-4 | 4 | 0.5759 | 0.0577 | 0.7159 |
| LoRA lr 5e-5 | 4 | 0.5634 | 0.0613 | 0.6998 |
| frozen control (same path) | 1 | 0.5316 | — | 0.6590 |

Fine-tuning does not beat frozen ESM-2:

| comparison | difference | t | df | p |
|---|---|---|---|---|
| ESM-2 frozen vs LoRA lr 1e-4 | 0.0394 | 1.319 | 3.43 | 0.268 |
| ESM-2 frozen vs LoRA lr 5e-5 | 0.0519 | 1.644 | 3.38 | 0.188 |
| LoRA lr 1e-4 vs ESM3 frozen | 0.0556 | 1.900 | 3.17 | 0.149 |

Neither LoRA group separates from frozen ESM-2, and neither separates from frozen
ESM3 either. The reason is the spread: at sd 0.0577 and 0.0613, these are the
noisiest replicate groups in the study, roughly 2.5× the frozen ESM-2 arm. Both
groups contain one seed that collapsed (0.4896 and 0.4722 against ~0.60 for the
others), and in both cases validation F1 also dropped for that seed, so
validation-based screening would have caught them.

Schmirler et al. report that task-specific fine-tuning almost always improves
downstream predictions. That is not reproduced here, but the comparison is
underpowered — n=4 against n=10 — and the honest statement is that this study
cannot resolve a difference of 0.04 F1 in the fine-tuned arm, not that fine-tuning
fails.

---

## Structure conditioning

| run | n | F1 | precision | recall |
|---|---|---|---|---|
| ESM3 sequence-only, lr 5e-4 bs 20 | 1 | 0.5280 | 0.7179 | 0.4176 |
| ESM3 + structure, lr 5e-4 bs 20 | 1 | 0.4712 | 0.7269 | 0.3486 |

−0.0568, entirely in recall, at matched configuration. Both are single runs, and
the widest well-estimated replicate band in this study is ±0.0409 (from the n=15
group), so this sits just outside what one run each can resolve. Suggestive, not
established. Two further structure runs at other learning rates land lower still
(0.4654 at T4, 0.4511 at lr 1e-3), which is consistent, but they are also n=1.

Separately, and unrelated to performance: geometric attention contributes
**exactly nothing** to sequence-only ESM3 embeddings. Extracting with the block
enabled and disabled produces bit-identical output, `max |difference| = 0.0`.

---

## The noise floor

95% prediction interval for one further run of the same configuration,
`mean ± t(n−1, .975) · sd · √(1 + 1/n)`:

| group | n | interval |
|---|---|---|
| ESM3 frozen, tuned | 15 | ±0.0409 |
| ESM-2 frozen, T4 | 10 | ±0.0574 |
| ProstT5 frozen, T4 | 5 | ±0.0931 |
| LoRA lr 1e-4 | 4 | ±0.2053 |
| ESM-2, default config | 5 | ±0.2171 |
| ProstT5, default config | 5 | ±0.3326 |
| ESM3, default config | 5 | ±0.5871 |

This is the number against which every single-run result in the sweep tables must
be read. The ESM3 optimisation sweep contains 28 single-run configurations spanning
0.4388 to 0.5547 — a range of 0.116, against a ±0.0409 band from the best-estimated
group. Most of the individual comparisons within that sweep are inside the noise.
Concretely, `esm3_b100_lr0.0008` (0.5245) and `esm3_b100_lr0.0012` (0.5345) differ
by 0.0100, which is a quarter of the band; treating that as evidence that one
learning rate beats the other is not supportable.

**The replicate groups are unseeded.** `torch.manual_seed` is called once in
`src/train_loop_crf.py`, inside `train_nested_cv`; the single-run path `train()`
that `run.py` calls never seeds. The `seed: 42` recorded in those configs is the
argparse default being serialised, never applied. The runs therefore differ in
initialisation, data order and GPU nondeterminism together — which is the
whole-pipeline variation Bouthillier et al. argue a replicate should capture, but
it must be described as unseeded rather than seed-controlled. The `default_*` and
`ft_*` groups were launched with explicit distinct seeds, but on the same
unseeded code path, so the same caveat applies to them.

---

## Not claimed

- **That ESM-2 is a better protein language model than ESM3.** One task, one head,
  one dataset, one split. Representation quality is not a scalar.
- **That ProstT5 is a worse representation.** Only that it is worse at ESM-2's
  hyperparameters, which is where it was measured.
- **That structure conditioning hurts.** n=1 per setting.
- **That fine-tuning does not help.** Underpowered, not negative.
- **Anything from the untuned sweep about representation quality.** Those runs
  measure optimisation robustness.
- **Anything cross-branch.** `baseline-upstream` is scored with the unfixed
  upstream metric so that it reproduces the published figures. Its numbers do not
  belong in any table above.

---

## Retractions

The previous version of this file, preserved as
`RESULTS_superseded_2026-08.md`, stated several things this one does not.

- **"Fine-tuning: +0.060 over its own control, entirely in recall", reported at
  3.3σ.** The σ came from the *frozen* arm (0.0185). Against the fine-tuned arm's
  own spread — 0.0577 and 0.0613 — the ESM-2-versus-LoRA comparison gives
  p = 0.268 and p = 0.188. Withdrawn.
- **Comparisons pooling runs across learning rates.** Pooling across a
  validation-selected hyperparameter treats a tuning choice as replicate noise and
  inflates n. The summariser now groups on the full configuration, which prevents
  this by construction.
- **A claimed +0.2776 F1 effect from a scheduler change.** No such ablation pair
  exists in this repository. The value appears only as two coincidental cells in
  `evaluation/lm_selection/crf_model_all_cv_balancedsplit.csv`, which is upstream's
  ESM-1b sweep. Withdrawn as unsupported.
- **Numbers mixed across the two metric versions.** All numbers above come from
  the fixed metric.

The ESM3-versus-ProstT5 equivalence asserted in the old file happens to be
supported by the current data (p = 0.926), but the figures it was asserted from
were not the ones that support it.

---

## Provenance

- 84 finished runs; 10 more have a `config.json` but no `test_metrics.json`
  (`esm3_ens/model_15` through `model_19`, three `*_long` runs, one struct run).
- 68 of 84 have no `valid_metrics.json`: the single-run path in
  `train_loop_crf.py` writes test metrics but not validation metrics, so those
  runs have no auditable record of which epoch was selected. One line fixes it,
  mirroring `main:src/train_loop_crf.py:125`.
- Embeddings for all four arms were verified with `src/utils/verify_embeddings.py`
  before use: coverage against the dataset hashes, shape against sequence length,
  dtype, finiteness, dimension, and mean per-residue L2 norm. ESM3 came in at
  11.023, the post-LayerNorm range, not the ~9800 raw pre-norm residual stream.
- ESM3 computes in bfloat16 and the other arms in fp32. That is the models' own
  difference — ESM3 ships bf16 weights with no fp32 release — so each is run as
  published; all arms store fp32 on disk.
