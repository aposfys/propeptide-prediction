# Is the benchmark out of date?

Short answer: **no. There is no newer published dataset, the labels you have are
99.1% identical to what UniProt says today, and a rebuild would add under 10%
usable positives while invalidating every existing number. Don't rebuild.**

The one thing worth knowing is in section 4: the benchmark does not label
everything UniProt annotates, and that is a limitation to state rather than a
defect to fix.

Reproduce all of it with:

```bash
python -m src.utils.dataset_currency_audit --cache propep.tsv
```

Measured 2026-09-07 against UniProt 2026_04. The benchmark was built from
UniProt 2022_02 (upstream's own default path is `uniprot_12052022_cv_5_50`).

## 0. Is there a newer published dataset?

No. `fteufel/DeepPeptide` has shipped one dataset and no v2. Nothing since 2023
has published a replacement propeptide benchmark. The neighbouring tools are
older and narrower — ProP 1.0 (2004) predicts proprotein convertase sites from
sequence motifs and is not a segment-level dataset — and recent peptide language
model work such as PepBERT (2026) is about representation, not about propeptide
annotation. So the choice is this file or one you build yourself.

## 1. What the benchmark actually is

| | |
|---|---|
| rows in `data/labeled_sequences.csv` | 8,449 |
| rows in `data/graphpart_assignments.csv` | 7,623 |
| **dropped by GraphPart, never used** | **826** |
| effective dataset | 7,623 |
| carrying a propeptide | 6,392 |
| negatives (peptide annotation, no propeptide) | 1,231 |
| distinct organisms | 1,660 |
| **viral / phage proteins** | **0** |

Two things are worth noticing here.

**826 proteins in the CSV are never used.** They have no cluster assignment, so
`dataset.py`'s `isin(partitions)` filter drops them. That is correct behaviour,
but it means the file is 11% larger than the dataset. Quoting 8,449 as the
dataset size is wrong; it is 7,623, which is the figure the paper reports.

**Zero viral proteins across 1,660 organisms is a deliberate exclusion.** It is
not chance, and it is a sensible choice: viral polyproteins are cut by viral
proteases with unrelated specificity and would dominate the length distribution.
It is not stated in the paper, and it bounds what the benchmark can claim.

## 2. Currency — are the labels still correct?

Comparing the 2022 labels against current UniProt, with current spans restricted
to 5..50 so the length filter is not miscounted as an annotation change:

| | n | share |
|---|---|---|
| identical | 6,336 | 99.1% |
| propeptides added | 4 | 0.1% |
| propeptides removed | 11 | 0.2% |
| coordinates changed | 16 | 0.3% |
| no longer annotated | 25 | 0.4% |

**0.9% of labels would move.** The data is not stale. Four years of UniProt
curation has barely touched these entries, which is what you would expect for
well-studied precursor proteins.

## 3. Growth — how much bigger would a rebuild be?

12,949 reviewed proteins carry a PROPEP feature today. 5,299 of them are absent
from the benchmark, but only **1,513** have at least one propeptide inside 5..50,
which is all the current grammar could use. Of those 1,513:

| reason | n | share |
|---|---|---|
| fragment | 452 | 29.9% |
| viral / phage | 250 | 16.5% |
| created after 2022-05-12 | 325 | 21.5% |
| explained by those three | 978 | 64.6% |
| **unexplained** | **535** | **35.4%** |

Only **434** reviewed PROPEP entries in all of UniProt were created after the
benchmark date, so this is not a story about the database growing. It is a story
about filters: fragments and viruses were excluded on purpose, and correctly.

The 535 unexplained are **+8.4%** on the 6,392 the benchmark uses. They are
dominated by bacterial strain variants (125 *Staphylococcus aureus* alone) and
287 of them are annotated only by homology, so after deduplication and GraphPart
the usable gain is smaller still.

Under other filters a rebuild reaches 8,702 proteins (DeepPeptide's own rule),
11,504 (a 2..105 grammar) or 12,949 (no length filter) — but the last two are a
different task with different labels, not a bigger version of this one.

## 4. Fidelity — the finding that matters

**The 5..50 filter was applied to annotations, not to proteins.** The paper says
they "filtered the peptide annotations for a length range of 5–50 AAs and
discarded all proteins that have no peptides within this range". A protein with
one propeptide in range and one outside is therefore *kept*, with the
out-of-range one silently absent from its labels.

| | |
|---|---|
| benchmark proteins carrying an unlabelled real propeptide | **548** (8.6%) |
| unlabelled real propeptide spans | **715** |

In the test partition (cluster 4, 1,538 proteins) there are 1,420 labelled spans
and 225 unlabelled ones — 67 shorter than 5 residues, 158 longer than 50.

**This does not inflate the reported false-positive count.** Every one of those
225 is outside the 5..50 the grammar can emit, and a hit requires both boundaries
within tolerance, so the model cannot score a true positive on them whatever it
predicts. The reported precision is not contaminated by them.

What they cost is **external validity**. The benchmark measures performance on a
length-filtered slice of the propeptide population, and 8.6% of its own proteins
contain evidence that the slice is not the whole picture. A model scoring 0.62 F1
here has not been shown to score 0.62 on UniProt propeptides.

That belongs in the thesis as a stated limitation. It does not need a rebuild to
fix, and a rebuild would not fix it anyway — it is a property of the length
window, which is the same thing [GRAMMAR.md](GRAMMAR.md) measures from the other
direction.

## Recommendation

**Keep the dataset.** Rebuilding costs every number in
[RESULTS.md](RESULTS.md), forces a GraphPart re-run that changes the partitions,
and breaks comparability with the published DeepPeptide figures — in exchange for
under 10% more positives, most of them homology-inferred strain variants.

State three things in the thesis instead:

1. The effective dataset is 7,623 proteins, not 8,449.
2. Viral proteins are excluded, so nothing here generalises to viral polyprotein
   processing.
3. 8.6% of the propeptide proteins carry a real propeptide the labels omit,
   because the length filter was applied per annotation.
