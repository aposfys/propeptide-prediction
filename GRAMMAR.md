# How big should the CRF grammar be?

Short answer: **the 5..50 length window is correct for this benchmark and cannot
be improved on it. It is wrong for UniProt, where it excludes 37.5% of annotated
propeptide proteins. And the cheapest repair is not more states — it is lowering
the minimum length from 5 to 2, which costs nothing.**

Read the next section first if you are comparing against the paper's "101
states": that is two branches of 50, and this branch runs one of them.

Reproduce every number here with:

```
python -m src.utils.propeptide_length_audit --cache propep.tsv
```

## 51 states here, 101 in the paper — these are not in conflict

Teufel et al. state it plainly: "We design our state space with one state for
not being in a peptide, and 50 states each for being in a propeptide or peptide,
corresponding to positions 1–50", giving "101 states in total". `baseline-upstream`
implements exactly that — `num_states = 101 if 'with_propeptides' in
args.label_type else 51`.

**So the original DeepPeptide grammar is 101 states, and it is two branches of 50
plus a shared background state, not a length cap of 101.**

| | background | peptide states | propeptide states | total |
|---|---|---|---|---|
| DeepPeptide, joint (`baseline-upstream`, `*-full`) | 1 | 50 | 50 | **101** |
| this branch, propeptide-only | 1 | — | 50 | **51** |

The propeptide-only fork drops the mature-peptide branch because it does not
predict mature peptides. **The length window is 5..50 in both**, so every
coverage figure in this document applies unchanged to the 101-state model: it is
the same 50 positions, counted once instead of twice.

`--max_peptide_len` controls the length cap *within a branch*. It does not add or
remove branches, and it cannot turn this into the joint model. That is a separate
axis (`label_type`, `num_labels=3`), and it lives on the `*-full` branches.

## What the grammar is

`crf_label_utils.peptide_list_to_label_sequence` encodes a propeptide of length
ℓ as a path through a state-space CRF: the first `min_len - 2` states are
dedicated N-terminal positions, and the remaining states count backwards from
the C-terminus. Background is state 0. With `min_len=5, max_len=50` that is
**51 states on this branch**, and it can represent exactly the lengths 5..50.

So the state count is a hard cap on length. A propeptide of 78 residues has no
representation in this grammar at all — not a bad one, none. That is equally true
of the 101-state joint model, whose propeptide branch is also 50 positions.

## The two populations, and why they disagree

| population | features | median | < 5 aa | 5–50 aa | > 50 aa |
|---|---|---|---|---|---|
| distributed benchmark (`data/labeled_sequences.csv`) | 8,201 | 21 | 0.00% | **100.00%** | 0.00% |
| reviewed SwissProt PROPEP (UniProt 2026_04) | 15,100 | 23 | 13.43% | **65.45%** | 21.12% |

The benchmark is 100% in range because Teufel et al. built it that way. Their
Methods say they "filtered the peptide annotations for a length range of 5–50
AAs and discarded all proteins that have no peptides within this range", and
report 90% coverage for peptides against **63%** for propeptides.

That 63% reproduces here independently: the current grammar covers **62.49%** of
UniProt proteins whose every propeptide fits the window. So the published figure
is not an artefact of the 2022 release, and the audit is measuring the right
thing.

**Measuring the cap against the benchmark is circular.** It will always report
that 51 states suffice, because anything that did not fit was removed before the
file was written.

## Coverage against state count

Protein-level coverage means every propeptide in that protein fits the window,
which is the condition under which the protein is usable. Viterbi decoding is
O(L · S²), so the cost column is the decode-time multiplier against 51 states.

| grammar | states | decode cost | features | proteins |
|---|---|---|---|---|
| 5..50 (current) | 51 | 1.0× | 65.45% | 62.49% |
| 5..60 | 61 | 1.4× | 68.40% | 65.46% |
| 5..105 | 106 | 4.3× | 76.83% | 74.55% |
| **2..50** | **51** | **1.0×** | **78.82%** | **76.11%** |
| 2..60 | 61 | 1.4× | 81.77% | 79.37% |
| **2..105** | **106** | **4.3×** | **90.20%** | **88.84%** |
| 2..150 | 151 | 8.8× | 93.90% | 92.99% |
| 2..262 | 263 | 26.6× | 98.99% | 98.82% |

## The finding that matters

**The floor costs more than the ceiling, and the floor is free.**

Of the 34.55% of annotations the current grammar cannot represent, 13.43 points
are *too short* and 21.12 points are too long. The short tail is real and
concentrated: 642 features of length 2, 1,048 of length 3, 329 of length 4.

Dropping `min_len` from 5 to 2 keeps the state count at 51 and lifts annotation
coverage from 65.45% to 78.82% — a bigger gain than doubling `max_len` to 105
buys (+11.4 points) at 4.3× the decode cost.

It is not literally free in modelling terms. The `min_len - 2` dedicated
N-terminal states are what let the CRF learn an explicit start signature; at
`min_len=2` there are none, and every length is encoded purely as distance from
the C-terminus. `min_len=3` keeps one prefix state and still reaches 78.5%.
Which of those is better is an empirical question this repository has not asked.

## Recommendation

- **For the thesis: change nothing.** The distributed benchmark has 8,201
  propeptide features and every one fits inside 5..50. A larger grammar cannot
  raise F1 by a single prediction, and would only slow decoding. This settles the
  open question in the thesis Methods: the 50-position cap is **not** the recall
  bottleneck.
- **Keep the propeptide-only 51-state form for the representation comparison.**
  Every arm in [RESULTS.md](RESULTS.md) — ESM-2 0.6153, ESM3 0.5203, ProstT5
  0.5189 — was measured with it, and the claim being made is a controlled
  comparison across embeddings. Switching to the 101-state joint model changes
  the task at the same time as the representation and costs all of those numbers.
- **The 101-state joint model is a legitimate separate arm, not a correction.**
  Mature peptides and propeptides sit adjacent in a precursor, so predicting both
  gives the CRF a constraint the propeptide-only model discards. That is a real
  "does more information help" question, it needs no new embeddings, and the code
  is already on `esm3-full` / `prost5-full` — whose results are currently
  retracted and unregenerated. Run it as an addition, scored with the fixed
  metric, and compare propeptide F1 against the 51-state number.
- **For a follow-on study on unfiltered UniProt: `min_len=2, max_len=105`, 106
  states.** It reaches 90.2% of annotations and 88.8% of proteins at 4.3× decode
  cost, which is the knee of the curve. Going to 99% costs 26.6× for 8.8 more
  points.

## Should the dataset be rebuilt?

**Not for the thesis.** Rebuilding from unfiltered UniProt forces a re-run of
GraphPart, which produces different partitions, which invalidates every number in
[RESULTS.md](RESULTS.md) and every published DeepPeptide figure the thesis
compares against. The design's whole claim is a controlled swap — same data, same
splits, same head, same metric, only the embeddings change — and changing the
data removes the control.

It is also not a repair that can be done halfway. Training on the filtered set
and evaluating on the unfiltered one does not work either: the training data
contains no propeptide longer than 50, so a larger grammar would have no examples
of the states it added.

As a separate study it is well motivated, and the order is fixed: rebuild from
UniProt without the length filter → re-run GraphPart → re-baseline all four
representation arms → only then compare. That is a new experiment with its own
baselines, not an update to this one.

`--min_peptide_len` and `--max_peptide_len` exist on this branch so that study
does not have to start by editing four hardcoded constants. At their defaults
(5 and 50) they reproduce the current grammar exactly, which `test_grammar.py`
asserts.
