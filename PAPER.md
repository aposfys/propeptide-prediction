# A better approach: the paper

Not the thesis. The thesis is a controlled comparison of representations on a
fixed benchmark, and it should stay that way. This is what to build instead if
the goal is a paper.

## The idea

**The 5–50 length filter is doing two different jobs, and only one of them is
right.**

Teufel et al. present it as a single length filter. Cut the annotations by
mechanism instead of by length and it splits in two:

| mechanism | n | 2–4 aa | 5–50 | 51–105 | >105 |
|---|---|---|---|---|---|
| CAAX (prenylated, C-terminal) | 781 | **100.0%** | 0.0% | 0.0% | 0.0% |
| convertase (dibasic keyword) | 3,963 | 7.9% | **64.2%** | 15.3% | 12.6% |
| zymogen, no dibasic | 2,215 | 4.3% | 43.5% | 26.8% | 25.4% |
| neither marker | 7,009 | 9.7% | 79.8% | 6.0% | 4.5% |

- **The floor is a mechanism filter and it is correct.** Every CAAX feature is
  below 5 residues. That reaction is RCE1 removing the aaX tripeptide after
  prenylation, followed by ICMT methylation — verified in the sequences, where
  12 of 12 sampled 3-residue C-terminal propeptides in prenylated proteins are
  preceded by cysteine. Fixed length, fixed position, no dibasic site. Excluding
  it is the right call; presenting that as a length filter is not.
- **The ceiling is a truncation and it is wrong.** 27.9% of convertase-processed
  propeptides are longer than 50 residues. The cap does not select a mechanism,
  it cuts one in half. Raising it recovers members of the class the model already
  targets.

No published work makes this distinction. That is the paper.

## What to build

### 1. The benchmark

Reviewed UniProt, non-viral, non-fragment — DeepPeptide's own exclusions,
confirmed in their paper and reproduced here — with **`min_len` kept at 5** and
the ceiling raised. Coverage with the floor fixed:

| grammar | states | decode | all non-CAAX | convertase class | proteins |
|---|---|---|---|---|---|
| 5..50 (DeepPeptide) | 51 | 1.0× | 69.0% | 64.2% | 62.3% |
| 5..105 | 106 | 4.3× | 81.3% | 79.5% | 74.6% |
| **5..150** | **151** | **8.8×** | **85.0%** | **83.9%** | **78.5%** |
| 5..262 | 263 | 26.6× | 90.7% | 90.1% | 84.6% |

**Take 5..150.** It covers 83.9% of the convertase class against 64.2% today, at
a decode cost that is nothing on a small head. 5..262 buys 6 more points for 3×
the cost and can be a reported ablation.

Every propeptide carries a **mechanism label** — CAAX (excluded, but recorded),
convertase, zymogen, unassigned — derived from UniProt keywords and position.
That label is the paper's main dataset contribution.

Re-run GraphPart at 30% identity. At roughly 16k sequences this is ~3.5× the
original pairwise load; use the mmseqs2 backend rather than needleall.

### 2. The model

CRF state-space decoder over frozen PLM embeddings, as in DeepPeptide, SignalP
6.0 and DeepTMHMM. Two changes:

- **151 states** instead of 51.
- **Predict the mechanism class alongside the segment**, the way SignalP 6.0
  predicts the signal-peptide region "together with the SP type". Do not pool.
  Keep the position grammar shared at 151 states and add a segment-level type
  head, so decoding stays O(L·151²) rather than multiplying the state space by
  the number of classes.

Representation arms, all frozen, all with the same head:

| arm | why |
|---|---|
| ESM-2 650M | DeepPeptide's choice, the baseline to beat |
| ESM-C 600M | newer sequence-only, controls for recency |
| **SaProt 650M** | structure-aware by joint AA+3Di vocabulary |
| ProstT5 + 3Di | bilingual, the branch here already builds it |
| ESM3-open + tracks | native multimodal conditioning |

Each structure arm gets its scrambled control, which is how SaProt's own authors
validated their gains.

### 3. The hypothesis that ties it together

**Structure should help differentially by mechanism, in a predictable order.**

- **CAAX**: a fixed C-terminal sequence motif. Structure should add nothing.
- **Convertase**: sites sit in surface-exposed unstructured loops. PiTou already
  scores furin sites with a solvent-accessibility term, so structure should help
  moderately.
- **Zymogen prodomains**: folded inhibitory domains, mostly internal. Structure
  should help most.

If that ordering appears, it is mechanistically interpretable evidence that the
structural signal is real rather than extra head capacity. That is a far stronger
claim than a pooled F1 going up, and it is only measurable because the dataset
carries mechanism labels. **This is the result the paper is built around.**

### 4. The prospective test

The negatives are defined by absence of annotation, which is positive-unlabeled
learning, not binary classification. Rather than only caveating it, test it.

**Train on UniProt 2022_02. Evaluate on what curators added by 2026.** Of the
1,231 proteins the 2022 benchmark calls negative, **420 have gained a propeptide
in 2..105 by 2026 and 773 still have none.** If the model ranks those 420 above
the 773, it predicted four years of curation, and the "false negatives" in the
training set are shown to be real rather than assumed.

Two things make this work:

- It **requires the wider grammar**. Only 4 of the 420 fall inside 5..50, so the
  published model cannot represent what it would need to predict. The extension
  and the validation justify each other.
- The flipped set is **not a curation sweep**. Its largest family is 12.9%
  (spider toxins), followed by cathelicidins, insulin, somatostatin, natriuretic
  peptide and calcitonin families across many organisms, and 56% carry the
  dibasic keyword. It is a broad sample of classical secretory precursors.

`uniprot_sprot-only2022_02.tar.gz` is 1.4 GB from UniProt's `previous_releases`
archive, so the 2022 label set can be rebuilt exactly rather than approximated.

## Contributions, in the order a reviewer will weigh them

1. **The mechanism decomposition of `PROPEP`**, with sequence-level verification.
   Novel, and it reframes an inherited filter as a biological choice.
2. **A prospective evaluation against real curation**, which is a rare thing to
   have for a positive-unlabeled problem.
3. **A benchmark** covering 83.9% of convertase-processed propeptides against
   64.2%, mechanism-labelled and homology-partitioned.
4. **Differential structure benefit by mechanism**, which explains *when*
   structure-aware PLMs help — currently an open question in the literature.
5. A model, which is the least interesting part and should be presented that way.

## What would sink it

- **The mechanism labels are keyword-derived.** Only 28% of 5–50 features carry
  the dibasic keyword, so "unassigned" is the largest class and the decomposition
  is partly an annotation artefact. Report per-class n, and treat "unassigned" as
  unassigned rather than as a fourth mechanism.
- **AlphaFold models the precursor**, and propeptide regions are often
  low-confidence. A structure effect could be a disorder effect. Correlate the
  per-class gain with per-residue pLDDT before claiming geometry.
- **Long propeptides are folded domains**, so a gain at 51–150 may be domain
  recognition rather than cleavage-site recognition. The boundary-tolerance
  metric partly controls this; say so.
- **GraphPart may remove more at the new size**, and the largest families are
  concentrated. Report the retention rate.

## Order of work

1. Rebuild from UniProt 2026 with mechanism labels; re-run GraphPart. Report the
   dataset paper-ready before training anything.
2. Rebuild the 2022 label set from the release archive for the prospective test.
3. ESM-2 baseline at 151 states, to establish the new benchmark's numbers.
4. SaProt and the structure arms with scrambled controls, 8 replicates each.
5. Per-mechanism breakdown. This is the figure.
6. The prospective test.

Steps 1 and 2 are the paper. Steps 3 to 6 are the evidence.
