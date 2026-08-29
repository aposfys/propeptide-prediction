# Experimental protocol

Pre-registration for the propeptide-prediction representation comparison. Written
before the deciding runs are executed, so that the analysis is fixed in advance
rather than chosen after seeing the numbers.

No performance figures appear in this document. Results belong in `RESULTS.md`,
and every number there must be regenerated from `test_metrics.json` /
`valid_metrics.json` by `src/utils/summarize_results.py`.

---

## 1. The question

**Under DeepPeptide's propeptide-prediction protocol, does substituting the
frozen sequence representation change segment-level F1 by more than the
protocol's own run-to-run variation, when every representation is trained at the
same, untuned configuration?**

That is the claim the available compute can support. Per-representation
hyperparameter search is not affordable here, so the budget is equalised at zero
rather than at some larger number: every arm runs at upstream's default
configuration, which was tuned for none of them. §3 sets out why that is a fair
comparison and what it does and does not license.

Three things this deliberately is not:

- **Not "ESM3 is a better protein language model than ESM-2."** One task, one
  head, one dataset. Representation quality is not a scalar and does not transfer
  across tasks; Vieira et al. find larger models do not consistently beat smaller
  ones, and Schmirler et al. find the ranking of pLMs changes across their eight
  tasks. The scope of the claim is this task.
- **Not "ESM3's structure channel helps/hurts propeptide prediction."** That arm
  exists at n=1 per setting and is reported descriptively only.
- **Not a fine-tuning comparison.** `src/models/plm_backbone.py` instantiates
  `ESM3.from_pretrained` at line 217 and encodes five ESM3-specific assumptions,
  so `finetune.py` can only fine-tune ESM3. The LoRA arm is single-model by
  construction and is reported as an ESM3-internal frozen-vs-adapted contrast,
  never as evidence about ESM-2 or ProstT5.

The original framing — "prove ESM3 is better" — is not a hypothesis, it is a
conclusion looking for support, and it cannot survive review. The defensible
thesis is the one above. If the answer turns out to be "no", that is a result
with published company (Vieira et al.; Xu et al.), not a failed project.

---

## 2. The metric

### Definition

`src/utils/manuscript_metrics.py`. Segment-level, boundary-tolerant, micro-averaged.

1. Viterbi decode → state path per protein. `convert_path_to_peptide_borders`
   converts the path to `(start, end)` spans, opening on `PROPEPTIDE_START_STATE
   = 1` and closing on `PROPEPTIDE_END_STATE = 50`.
2. A predicted span is a **true positive** only if *both* its start and its end
   fall within `tolerance` residues of a true span's start and end.
3. Overlapping true spans are clustered (`groupby('group')`); a cluster
   contributes at most one TP and one FN, because the model cannot emit overlaps
   and recovering any member is "good enough".
4. TP / FN / FP are **summed over all proteins**, then precision, recall and F1
   are computed once from those totals. Micro-averaged over segments, not
   macro-averaged over proteins.

**Primary endpoint:** `f1 propeptides` at `tolerance = 3` on partition 4.

This is a demanding metric. It is not per-residue accuracy: a span with a correct
start and a 4-residue-late end scores as one FP *and* one FN. That is the right
choice biologically — propeptide boundaries are protease cleavage sites, and a
boundary is either right or it is not — but it means F1 here is not comparable to
per-residue F1 reported by other peptide predictors.

### The comparability rule

The fork's metric differs from upstream DeepPeptide's in two ways that change the
numbers (`git diff baseline/original-deeppeptide esm3-multimodal-propeptide --
src/utils/manuscript_metrics.py`):

- Upstream reuses the loop variable `idx` across the true and predicted loops, so
  a matched true span is marked at the *prediction's* index. The phantom row is
  dropped by `groupby('group')`, turning real true positives into false
  negatives. **Upstream understates recall and F1**, and does so more the more
  false positives a model emits — i.e. non-uniformly across models.
- Upstream guards the recall division with `(tp + fp) > 0` instead of
  `(tp + fn) > 0`.

Therefore: **every number in the thesis's comparison tables comes from the fixed
metric.** Numbers scored with the upstream metric live in exactly one table, the
one reproducing Teufel et al., under the caveat already written into the
`baseline/original-deeppeptide` README. The two are never placed in the same
table or the same sentence. The same applies to `clip_grad_norm_`, which upstream
calls before `backward()` — a no-op, so the published models trained unclipped
while the embedder branches clip at 0.25.

### What gets reported

- P, R and F1 separately, not F1 alone. Product-of-experts ensembling and CRF
  decoding both move precision and recall in opposite directions, and an F1-only
  table hides that.
- The full tolerance curve, `windows=[0, 1, 2, 3]`, not just window 3. Window 3
  is the most permissive setting and reporting it alone invites the question.
  `test_outputs.pickle` is saved by every run, so the curve costs no retraining.

---

## 3. Design

A single-factor design over representation, with the head, data, splits, metric
and tuning budget held fixed.

| Arm | Representation | Dim | Role |
|---|---|---|---|
| `onehot` | one-hot amino acid | 21 | **control** — no pretraining |
| `esm2` | ESM-2 650M, layer 33 | 1280 | reference |
| `esm3` | ESM3 sequence-only | 1536 | test |
| `prost5` | ProstT5, `<AA2fold>` | 1024 | test |

Held constant: `LSTMCNN` → `Linear` → linear-chain CRF, 51 propeptide states;
homology-partitioned splits (train 0/1/2, validation 3, test 4); the fixed metric;
the tuning budget; the replicate count.

### The one-hot control is not optional

Without it the experiment can rank representations but cannot say whether
*pretraining* contributes anything at all on this task. Hewitt and Liang's
argument applies directly: this head is not a linear probe. Its `LSTMCNN`
feature extractor plus CRF has enough capacity to solve a good deal of the task
from sequence identity alone, so a difference between two pretrained
representations is only interpretable against the floor set by no pretraining.
The control costs no GPU embedding extraction and one training run per replicate.

### Equal tuning budget

This is the change that decides whether the comparison is publishable.

At present the arms are tuned unequally in a way that maps exactly onto the
failure Musgrave et al. and Dacrema et al. document — with the direction
reversed. ESM-2 runs on upstream's Table S2 configuration, which Teufel et al.
tuned *for ESM-1b/ESM-2*. ESM3 received an ad-hoc in-house sweep. ProstT5
received no tuning at all and is run at ESM-2's learning rate. The usual failure
is an under-tuned baseline flattering a new method; here the new representations
are the under-tuned ones. The methodological error is the same and so is the fix.

That this matters is already visible in the ProstT5 arm, where replicates
diverged to `nan` at ESM-2's learning rate while ESM-2 completed 10/10 at the
same setting. The mechanism is not established — embedding norms differ across
representations (ProstT5 ~7.8, ESM-2 ~10.1), which is one candidate, but a 1.3x
ratio is modest and this has not been isolated. What is measured is that a
learning rate stable for one representation is unstable for another. Gradient
clipping does not rescue it: Adam normalises each coordinate by its own
second-moment estimate, so after `clip_grad_norm_` the step is still of order
`lr`. Clipping bounds the gradient, not the update.

**Protocol.** Per-arm Optuna search is the textbook fix and is not affordable
here — it multiplies the run count by the trial budget. The budget is therefore
equalised at zero. Every arm trains at upstream's argparse default (`lr 1e-4,
batch 100, dropout 0.1`), which was tuned for none of the representations in the
comparison, on identical splits for an identical number of epochs.

This is a weaker but honest design, and the distinction belongs in the write-up.
An all-untuned comparison answers *which representation works best out of the box
under this architecture's defaults*. It does not answer *which representation is
best when each is used well* — that needs the per-arm search. What it does have
is the property that matters for validity: no arm is advantaged, so a difference
between arms cannot be an artefact of one having been tuned.

The runs at upstream's T4 configuration (`lr 0.0055, batch 20, dropout 0.6902`)
are kept, but reported separately and framed as what they actually are: a
**hyperparameter transferability** experiment, asking whether a configuration
tuned for ESM-2 carries to other representations. The ProstT5 divergences are a
result of that experiment, not a defect in it.

### Adaptation (secondary, ESM3 only)

Frozen vs LoRA, ESM3 only, captioned as such. Schmirler et al. is the reference
point for what fine-tuning is expected to buy and for parameter-efficient
fine-tuning reaching full-fine-tuning quality more cheaply. Any cross-model
fine-tuning claim requires backbone classes that do not exist in this repo, and
is out of scope.

---

## 4. Statistical protocol

Two different uncertainties, both needed, answering different questions.
Bouthillier et al. model both and show that reporting either alone overstates
confidence.

### 4a. Training variance — replicate runs

**N = 5 replicates per arm**, at the shared default configuration, with
`--seed 1 … 5`. Ten would be better; five is what the schedule allows at roughly
40 minutes per run across four arms. With the ~0.025–0.03 run-to-run standard
deviation the existing sets show, n=5 per arm resolves a difference of about
0.045 F1 at conventional power, which is below the gaps in play. The sweep loop
skips completed runs, so extending to seeds 6–10 later costs only the new runs.

A correction on seeding, because it changes how the existing sets must be
captioned. `torch.manual_seed(args.seed)` appears exactly once in
`train_loop_crf.py`, at line 609, inside `train_nested_cv`. The single-run path
is `train()` at line 547, which `run.py` calls directly and which never seeds. So
the `seed: 42` recorded in every existing replicate's `config.json` is the
argparse default being serialised, never applied. Those runs are **unseeded**:
they vary in initialisation, data order and GPU nondeterminism together.

That is a better variance estimate than seed-controlled runs, not a worse one —
it is the whole-pipeline variation Bouthillier et al. argue a replicate should
capture, and Summers and Dinneen find "all sources of nondeterminism have similar
effects on measures of model diversity", with "even one-bit changes in initial
parameters" producing models that converge to very different values. But it must
be labelled as unseeded rather than as seed-controlled. Making `--seed` bite on
this path needs one line, `torch.manual_seed(args.seed)`, at the top of `train()`;
without it the seed column in the new sweep is decoration too.

Report per arm: n, mean, sd, min, max, and the 95% prediction interval for one
further run, `mean ± t(n−1, .975) · sd · √(1 + 1/n)`. The prediction interval is
the honest answer to "would a single run have told us this", and it is the number
against which every n=1 result in the sweep tables must be read.

### 4b. Test-set sampling variance — paired bootstrap

The test partition is finite. Two arms can differ on it by chance even with zero
training variance. **Paired bootstrap over test proteins**, 10,000 resamples,
resampling proteins (not segments, since segments within a protein are not
independent), recomputing the micro-averaged F1 for both arms on each resample,
and reporting the distribution of the difference. This is the standard test for
this situation (Dror et al.).

This requires no retraining: `test_outputs.pickle` is already written by every
run.

### 4c. Pre-registered analysis

- **Primary:** Welch's unequal-variance t-test on the 10 replicate F1 values,
  for each of the three pairwise comparisons among `esm2`, `esm3`, `prost5`.
  Two-sided, α = 0.05, **Holm-corrected across the three comparisons.**
- **Secondary:** each pLM arm vs `onehot`, same test, reported separately from
  the primary family.
- **Effect size:** the F1 difference with its confidence interval is reported in
  every case. A p-value without an effect size is not a result.
- Groups are pooled **only** when they share a full configuration.
  `src/utils/summarize_results.py` enforces this by grouping on the whole
  `config.json` minus `out_dir` and `seed`. Pooling runs that differ in a
  validation-selected hyperparameter treats a tuning choice as replicate noise
  and inflates n.

### 4d. Divergence rule, fixed in advance

A run is **diverged** if training loss becomes non-finite, or exceeds ten times
its running minimum, at any epoch.

- Diverged runs are **kept** in the F1 analysis. Checkpoint selection is
  `improved = score > best_score` (`train_loop_crf.py:329`), and once validation
  F1 collapses the comparison is False forever, so the saved weights are the
  pre-divergence best. Early stopping on validation is part of the protocol, and
  discarding runs *after* seeing their test scores is exactly the practice this
  design exists to avoid.
- The **number** of diverged runs per arm is reported as a stability statistic in
  its own right. An arm that diverges under its own tuned configuration is
  reporting something real about the representation.

---

## 5. Threats to validity

Ordered by how much they could change the conclusion.

| # | Threat | Status | Mitigation |
|---|---|---|---|
| 1 | Unequal tuning budget across arms | **resolved** | §3, all arms at upstream's untuned default; T4 runs reported separately as a transferability experiment |
| 2 | No control for "any pretraining at all" | **resolved** | §3, one-hot arm built (8061 sequences, 21 dims, verified: norm exactly 1.000, no all-zero rows) |
| 3 | Embedding provenance differs: ESM-2 and ProstT5 use embeddings checked with `verify_embeddings.py`; every ESM3 set predated that check | **resolved** | ESM3 re-extracted to `/mnt/storage` and verified: 8061/8061, dim 1536, per-residue L2 norm 11.023 (post-LayerNorm range, not the ~9800 raw residual stream) |
| 4 | Single train/val/test split, not cross-validation | **open** | see §6, optional; the largest remaining upgrade |
| 5 | Extraction protocols differ: upstream's ESM-2 extractor chunks >1022-residue inputs into disjoint 722-token windows with no overlap and no BOS after the first chunk; ESM3 and ProstT5 extract at full context | **documented** | report; sensitivity analysis restricted to sequences ≤1022 |
| 6 | Head capacity is a function of embedding dim — `conv1` is the only layer that touches it, so arms differ by `(d₁−d₂)·n_filters·kernel_size` trainable parameters. Measured: 155,665 at dim 21 against ~332,513 at dim 1024 | **documented** | report per-arm parameter counts; optionally project all arms to a common dim |
| 7 | Numerical precision differs: ESM3 computes in bfloat16, ESM-2 and ProstT5 in fp32 | **documented** | this is the models' own difference — ESM3 ships bf16 weights and has no fp32 release — so each is run as published; all arms store fp32 on disk |
| 8 | Two metric versions in the repository | **resolved** | §2, fixed metric only, one segregated legacy table |
| 9 | Replicates are unseeded — `train()` never calls `torch.manual_seed`, so the recorded `seed` was never applied | **documented** | §4a; caption existing sets as unseeded whole-pipeline variance, which is what Bouthillier et al. want anyway |
| 10 | LoRA arm is ESM3-only | **structural** | scoped out in §1; never generalised |
| 11 | `train_loop_crf.py` single-run path writes `test_metrics.json` but not `valid_metrics.json` (lines 552–583), so 15 existing runs have no auditable model-selection record | **open** | mirror `esm2-propeptide:125`; one line |
| 12 | Early stopping interacts with the arm: `best_score` starts at −1.0, so an all-negative epoch 1 counts as an improvement and patience then counts from a score of 0.0. The one-hot control early-stopped at epoch 11 with F1 0.0 while the loss was still falling steeply | **resolved** | `--patience 0` for every arm, so all four get exactly 50 epochs and the stopping rule cannot favour one |

Threat 6 deserves a note. It cannot be removed without either changing the head
(breaking comparability with upstream) or projecting the embeddings (adding an
untrained layer whose initialisation becomes another variance source). Reporting
it is the proportionate response. It runs *against* the smaller representations,
so it cannot manufacture a win for ProstT5 or for the control — but it does mean
that if the one-hot arm underperforms, "no pretraining" and "smaller head" are
not fully separable. The clean fix, if it is ever worth the compute, is a random
fixed embedding per amino acid at 1280 dimensions: same information content as
one-hot, matched head capacity.

Threat 12 is the reason `--patience 0` is not a detail. Under early stopping, an
arm that improves keeps training while an arm stuck at zero is cut off; the
existing default-config ESM-2 run reached its best at epoch 45, so it effectively
used its full budget while the control would have received 11 epochs.

---

## 6. Run manifest

**Done.**

1. One-hot embeddings for the same 8,061 deduplicated sequences, 21 dims,
   verified.
2. ESM3 re-extracted to `/mnt/storage/fysekidis/embeddings/esm3` and verified:
   8061/8061, dim 1536, norm 11.023, 4m17s. Required two fixes to
   `make_embeddings.py` that existed only in the structure extractor — autocast
   instead of an fp32 upcast for the pLDDT dtype mismatch, and disabling
   geometric attention, which contributes bit-identical output for sequence-only
   input (`max |difference| = 0.0`) while allocating ~3 KiB · L² and OOMing on
   long sequences.

**Running — this is the experiment.**

3. Five replicates per arm at the shared default (`lr 1e-4, batch 100,
   dropout 0.1, epochs 50, patience 0`), `--seed 1 … 5`, four arms — one-hot,
   ESM-2, ESM3, ProstT5. Twenty runs, ~40 minutes each.
4. `src/utils/summarize_results.py` over the result tree; `RESULTS.md` written
   from its JSON output and nothing else.

**Cheap and worth it — no retraining, from the saved `test_outputs.pickle`.**

5. Tolerance curve `windows=[0,1,2,3]` rather than window 3 alone.
6. Paired bootstrap (§4b).
7. The one-line `valid_metrics.json` fix (threat 11) and the one-line
   `torch.manual_seed` fix in `train()` (threat 9).

**Optional, in descending value per unit of compute.**

8. **5-fold cross-validation** over the homology partitions instead of the single
   0/1/2–3–4 split. The largest remaining validity upgrade: it removes the
   dependence of the whole result on which partition happened to be held out, and
   `train_nested_cv` already implements it. Costs 5×.
9. Seeds 6–10 to take every arm to n=10. The sweep loop skips completed runs, so
   this costs only the new runs.
10. Sensitivity analysis on sequences ≤1022 residues (threat 5).
11. A random-fixed-embedding control at 1280 dims to separate "no pretraining"
    from "smaller head" (threat 6).

---

## 7. References

Verified; each was checked against the publisher or ACL/arXiv record rather than
reconstructed from memory.

- Teufel, F., Refsgaard, J.C., Kasimova, M.A., Deibler, K., Madsen, C.T.,
  Stahlhut, C., Grønborg, M., Winther, O., Madsen, D. (2023). DeepPeptide
  predicts cleaved peptides in proteins using conditional random fields.
  *Bioinformatics* 39(6), btad616.
- Bouthillier, X., Delaunay, P., Bronzi, M., Trofimov, A., Nichyporuk, B.,
  Szeto, J., Mohammadi Sepahvand, N., Raff, E., Madan, K., Voleti, V.,
  Ebrahimi Kahou, S., Michalski, V., Arbel, T., Pal, C., Varoquaux, G.,
  Vincent, P. (2021). Accounting for Variance in Machine Learning Benchmarks.
  *MLSys 2021*. arXiv:2103.03098. — replicate protocol; both variance sources.
- Summers, C., Dinneen, M.J. (2021). Nondeterminism and Instability in Neural
  Network Optimization. *ICML 2021*. arXiv:2103.04514. — "all sources of
  nondeterminism have similar effects on measures of model diversity"; justifies
  §4a.
- Dror, R., Baumer, G., Shlomov, S., Reichart, R. (2018). The Hitchhiker's Guide
  to Testing Statistical Significance in Natural Language Processing. *ACL 2018*,
  1383–1392. — significance-test selection; the paired bootstrap of §4b.
- Musgrave, K., Belongie, S., Lim, S.-N. (2020). A Metric Learning Reality Check.
  *ECCV 2020*. — equal tuning budget as a precondition for comparison.
- Dacrema, M.F., Cremonesi, P., Jannach, D. (2019). Are we really making much
  progress? A worrying analysis of recent neural recommendation approaches.
  *RecSys 2019*. — under-tuned baselines producing illusory gains.
- Hewitt, J., Liang, P. (2019). Designing and Interpreting Probes with Control
  Tasks. *EMNLP-IJCNLP 2019*, 2733–2743. — probe capacity confounds
  representation quality; motivates the control arm of §3.
- Melis, G., Dyer, C., Blunsom, P. (2018). On the State of the Art of Evaluation
  in Neural Language Models. *ICLR 2018*. arXiv:1707.05589. — a well-tuned older
  architecture matching or beating newer ones under equal search.
- Schmirler, R., Heinzinger, M., Rost, B. (2024). Fine-tuning protein language
  models boosts predictions across diverse tasks. *Nature Communications* 15,
  7407. — frozen vs fine-tuned pLMs across eight tasks; parameter-efficient
  fine-tuning.
- Vieira, L.C., Handojo, M.L., Wilke, C.O. (2025). Medium-sized protein language
  models perform well at transfer learning on realistic datasets. *Scientific
  Reports*; preprint bioRxiv 10.1101/2024.11.22.624936. — "larger models do not
  necessarily outperform smaller ones"; frozen embeddings and LoRA.
- Dallago, C., Mou, J., Johnston, K.E., Wittmann, B.J., Bhattacharya, N.,
  Goldman, S., Madani, A., Yang, K.K. (2021). FLIP: Benchmark tasks in fitness
  landscape inference for proteins. *NeurIPS Datasets and Benchmarks 2021*. —
  frozen-embedding benchmarking with one-hot baselines.
- Xu, J., Shi, Y., Lang, L., Cui, T., Zhang, Z., Chen, G., Qiu, J., Heng, P.-A.
  (2025). InstructPLM-mu: 1-Hour Fine-Tuning of ESM2 Beats ESM3 in Protein
  Mutation Predictions. arXiv:2510.03370. — cite with care: the title claims
  ESM2 "beats" ESM3, but the abstract claims only performance "comparable to
  ESM3". Use for the weaker, supported statement.
