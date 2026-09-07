# Is this approach valid? What the literature says

Each design choice in this repository, checked against published practice. The
short version: **five of the six choices are standard and well precedented. One
is not, and it is the one that matters most for the proposed dataset rebuild.**

## 1. Frozen embeddings, one fixed head, swap only the representation — STANDARD

This is shallow probing, the established way to compare representations. TAPE
(Rao et al. 2019) and FLIP (Dallago et al. 2021) are the canonical benchmarks
built on it, and the logic is explicit in the probing literature: keeping the
encoder frozen and fitting only a small head measures what the representation
already makes accessible, independent of downstream capacity.

The design in [EXPERIMENT.md](EXPERIMENT.md) — same head, same data, same
splits, same metric, same budget, only the embeddings change — is that method
applied to one task.

**Known caveat, already handled.** Schmirler et al. (*Nat Commun* 2024) report
that fine-tuning almost always beats frozen embeddings. [RESULTS.md](RESULTS.md)
cites this and correctly states that the LoRA arm here is underpowered rather
than negative.

## 2. PLM + CRF state-space decoder for segment labelling — STANDARD, same lineage

Not an idiosyncratic architecture. It is the house design of the group that
produced the benchmark:

| tool | encoder | decoder |
|---|---|---|
| SignalP 6.0 (*Nat Biotechnol* 2022) | ProtBert | CRF over signal-peptide regions |
| DeepTMHMM (2022) | ESM-1b + biLSTM | CRF over topology states |
| DeepPeptide (*Bioinformatics* 2023) | ESM-2 + LSTM-CNN | 101-state CRF |

All three use a state-space model that constrains segment length, and all three
argue the same way: the CRF scores the whole labelling rather than each position
independently. Nothing about the 51- or 101-state grammar is unusual.

## 3. Homology partitioning before splitting — REQUIRED, not optional

GraphPart (Teufel et al., *NAR Genomics Bioinf* 2023) exists because "ignoring
this tends to overestimate the performance of prediction methods". The benchmark
is partitioned at 30% identity. Any rebuild must re-run it, and that is a
requirement rather than a nicety.

## 4. Structure should help cleavage-site prediction — WELL SUPPORTED

The domain literature is consistent, and it predates protein language models:

- Protease cleavage sites are "primarily distributed in loop regions of the
  substrate proteins".
- Furin sites specifically are "located in unstructured loops which are presented
  on the protein surface"; PiTou's score is a core binding term plus a **flanking
  region solvent-accessibility term**.
- PROSPER (*PLOS ONE* 2012), iProt-Sub (*Brief Bioinform* 2019) and Procleave
  (2020) all combine sequence with predicted secondary structure, solvent
  accessibility and disorder, and report that the combination beats sequence
  alone.

So the hypothesis behind the structure branches is not speculative. It is the
standard feature set of the pre-PLM cleavage-site field, re-asked with a
representation that might already contain it.

## 5. But structure-aware PLMs give mixed results — CONTESTED, which is the point

The counter-evidence is real and should be stated:

- "When Does Structure Help? The Information Bonus of AlphaFold2 Representations
  over Protein Language Models" (2026) treats this as an open question.
- Sequence embeddings alone match or exceed structure-based methods on several
  tasks.
- Models trained directly on **predicted** structures have failed to transfer to
  real ones, with training loss falling while downstream representations did not
  improve. Every structure here comes from AlphaFold DB, so this applies.

**And SaProt validates the control design used on these branches.** Its authors
show that "residue-sequence-only SaProt (without 3Di token) performs highly
similar to the official ESM-2 35M model, which helps establish that the
improvements in SaProt's performance come from the structural information
integration, not other factors". That is the same argument as the shuffled-3Di
and scrambled-structure controls in [NEXT_STEPS.md](NEXT_STEPS.md): a structure
arm has to beat a matched arm with the structure destroyed, not merely beat the
sequence-only baseline.

A contested question with a clean control is a good thesis question. A null
result here is publishable.

## 6. Negatives defined by absent annotation — A NAMED PROBLEM, not a detail

This is **positive-unlabeled (PU) learning**, and it has its own review
literature (*Brief Bioinform* 2022) and applications in protein function
prediction (PU-GO, *Bioinformatics* 2024). The framing is that unlabeled
instances are not negatives: "unlabeled samples might hide positive annotations
yet to be discovered", and treating them as negatives "erroneously guide[s]
classifiers to predict false negatives".

That is exactly what [DATASET_V2.md](DATASET_V2.md) measures. Of the 1,231
proteins this benchmark calls negative, 4 are wrong under the 5..50 grammar and
**378 are wrong under 2..100**. The current window is, by luck, a regime where
the PU problem is negligible. Widening it moves the task into the regime the PU
literature exists to handle, and the honest options are to say so, to use a PU
objective, or not to widen.

## 7. Pooling three mechanisms under one label — NO PRECEDENT FOUND

This is the gap. I could find no published work that treats UniProt's `PROPEP`
key as more than one class, and none that justifies the 5–50 window on
mechanistic grounds — Teufel et al. present it purely as a length filter.

The decomposition in [DATASET_V2.md](DATASET_V2.md) — CAAX/aaX removal at 2–4
residues, convertase cleavage at 5–50, zymogen prodomains at 51–100 — appears to
be new. That cuts both ways. It is a contribution, and it also means there is no
precedent to lean on if a reviewer disagrees.

**But there is a direct architectural precedent for the fix, from the same lab.**
SignalP 6.0 handles five signal-peptide types in one model by predicting "the SP
region at each sequence position **together with the SP type**". It does not pool
them under a single label. If the propeptide window is widened, that is the
design to copy: one CRF, one extra output for the mechanism class, per-class
metrics reported.

## Verdict

| choice | status |
|---|---|
| frozen-embedding probing comparison | standard |
| PLM + state-space CRF | standard, same lineage as the benchmark |
| GraphPart at 30% identity | required |
| structure should help cleavage prediction | well supported |
| structure-aware PLMs actually helping | contested — needs the controls, which exist here |
| widening to 2..100 with pooled labels | **no precedent, and PU problems** |

The thesis as it stands is methodologically conventional in the parts that should
be conventional. The structure arms ask a live question with the field's own
control design. The dataset extension is the only piece that would need defending
from first principles, and the defensible version of it predicts the mechanism
class alongside the segment rather than pooling.
