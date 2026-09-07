# Coverage: the rebuilt dataset against the published one

Every figure below is produced by

```bash
python -m src.utils.build_dataset --out_dir data_v4 --max_len 100
python -m src.utils.validate_dataset --data_dir data_v4 --max_len 100
```

## Checked against the paper first

Every figure the paper states is reproduced from the distributed files:

| paper | value | recomputed |
|---|---|---|
| dataset after GraphPart | 7,623 proteins | 7,623 ✓ |
| Table 1, peptides | 6,169 | 6,169 ✓ |
| Table 1, propeptides | 7,359 | 7,359 ✓ |
| Table 1, taxonomy sum | 531 + 7,092 | 7,623 ✓ |
| "63% of propeptides fall in 5–50" | 63% | 64.4% ✓ |

**Two conventions that matter.** The paper counts annotations **unmerged**, so
overlapping alternative boundaries count separately. Merging gives 7,349 rather
than 7,359, and every span figure below is unmerged to match.

And the paper's 63% **counts CAAX tripeptides as propeptides**. On the same
denominator I get 64.4%, which is the agreement. On the denominator used below —
which excludes them — the same window covers 68.3%. The two figures are not in
conflict; they count different things, and the difference is 781 annotations of a
different reaction.

## The denominator

**Reviewed UniProt, excluding viral proteins and fragments, carrying at least one
usable propeptide: 10,913 proteins and 12,894 spans.**

Those two exclusions are Teufel et al.'s own. *Usable* additionally excludes two
classes of annotation that share the `PROPEP` key without being propeptides in
the sense of the task:

- **781 CAAX tripeptides.** Prenylation of a C-terminal cysteine, then RCE1
  removing the aaX tripeptide, then methylation. Verified in the sequences: of
  twelve 3-residue C-terminal propeptides sampled from prenylated proteins,
  twelve are immediately preceded by cysteine. Fixed length, fixed position, no
  dibasic site.
- **293 PROSITE ProRule sorting signals** (PRU00477, PRU01070), which Teufel et
  al. also remove.

## The comparison

Spans counted unmerged, as the paper does.

| | proteins | of eligible | labelled spans | of eligible |
|---|---|---|---|---|
| published, as distributed | 7,213 | 66.1% | 8,211 | 63.7% |
| **published, after GraphPart — what the model saw** | **6,392** | **58.6%** | **7,359** | **57.1%** |
| **rebuilt, before GraphPart** | **8,516** | **78.0%** | **9,615** | **74.6%** |
| rebuilt, at 90% retention (estimated) | 7,664 | 70.2% | 8,653 | 67.1% |

The like-for-like claim is the second row against the fourth, since GraphPart has
not yet run on the rebuilt file: **58.6% → 70.2% of proteins, +11.7 points, and
57.1% → 67.1% of spans, +10.0 points.** The unpartitioned comparison is +11.9 and
+10.9, and needs no retention assumption.

**The 90% is an estimate, not a measurement.** It is the rate GraphPart achieved
on the published file, 7,623 of 8,449. If the rebuilt file retains 80% instead —
plausible, since the added proteins skew toward protease families, which
cluster — coverage lands near 62% rather than 70% and the gain shrinks to about
+4 points. Quote the unpartitioned row until GraphPart has run.

Quote the post-GraphPart figure. The published 7,213 is not what the model was
trained on; 826 rows are left unpartitioned and the loader drops them.

## The second difference, which is not about size

| | published | rebuilt |
|---|---|---|
| real propeptides inside kept proteins that the labels omit | **680** | **0** |

Teufel et al. filtered per *annotation*: out-of-range spans were deleted and the
protein kept. A protein with one propeptide of 20 residues and another of 90 was
retained with the 90 labelled as background. The rebuild rejects such proteins
outright, which costs 436 of them and leaves no propeptide unlabelled anywhere.
`validate_dataset.py` checks this and the published file fails it.

## Where the gain lands, by mechanism

| mechanism | eligible | published | rebuilt | rebuilt covers |
|---|---|---|---|---|
| convertase (dibasic) | 2,809 | 1,580 | **2,068** | 73.6% |
| zymogen / protease | 2,185 | 641 | **1,467** | 67.1% |
| unassigned | 5,919 | 3,798 | 4,981 | 84.2% |

The gain is concentrated in the **zymogen/protease** class, which more than
doubles. That is expected — protease prodomains are the mechanism that sits
above 50 residues — and it is the class the published benchmark is too thin to
measure, with 97 spans in its test partition against roughly 311 in the rebuilt
one.

## What is still excluded, with numbers

- **2,396 proteins** carrying a propeptide longer than 100 residues. Reaching
  them needs 263 states for 90.1% of the ceiling, at 27× the decode cost.
- **970 proteins**: 664 fragments and 306 viral. Protocol exclusions, not length
  ones, and not recoverable by any grammar.
- **912 proteins** whose only propeptides are shorter than 5 residues. The floor
  is a mechanism filter, not a length one.
- **76,141 unreviewed entries** carry a propeptide annotation, against 13,416
  reviewed. Six times more data, almost all propagated by homology from the
  reviewed set. Using it would mean training on predictions.

So the rebuilt file holds 78.0% of what is curated and usable. The next 5.5
points cost twice the compute and the rest is either exponentially expensive, a
different reaction, or unreviewed.

## One sentence for the paper

> Rebuilding the benchmark under the published protocol with the length window
> widened from 5–50 to 5–100 residues raises coverage of curated propeptides from
> 66.1% to 78.0% of eligible proteins and from 63.7% to 74.6% of annotated spans,
> and removes 680 propeptides that the published labels leave annotated as
> background.
