# Why the dataset changed, and what it covers

Every claim here is produced by `src/utils/validate_dataset.py`, which checks a
built dataset against live UniProt. Run it on either file:

```bash
python -m src.utils.validate_dataset --data_dir data_v2 --max_len 100
python -m src.utils.validate_dataset --data_dir data     --max_len 50
```

## Why it changed

**Because the 101-state grammar has nothing to represent otherwise.** Every
propeptide in the distributed benchmark is 5–50 residues, so states 51–100 would
never be visited by a label. The grammar and the data have to move together or
neither move is worth making.

**And because the distributed file fails five of seven integrity checks.** Not
all of those are design faults — two are just age — but three are.

| check | distributed | rebuilt |
|---|---|---|
| every kept protein carries all its non-CAAX UniProt propeptides | **FAIL, 564 proteins** | PASS |
| no label absent from UniProt | FAIL, 28 (annotation withdrawn since 2022) | PASS |
| sequences match UniProt | FAIL, 24 differ (sequence updates since 2022) | PASS |
| all spans representable by the grammar | PASS | PASS |
| no fragments | **FAIL, 1** | PASS |
| no viral proteins | PASS | PASS |
| no negative carries a UniProt propeptide | **FAIL, 462** | PASS |

The three real ones:

- **564 proteins carry a propeptide their labels omit.** Teufel et al. applied
  the 5–50 filter to *annotations*, not proteins, so a protein with one span in
  range and one outside was kept with the out-of-range span silently unlabelled.
  680 real propeptides sit inside kept proteins and are not labelled.
- **462 of the 1,236 negatives carry a propeptide in current UniProt.** Under a
  5–50 grammar only 4 of those are in range, so the old file gets away with it.
  Under 5..100 it would be 30.7% of the negative set mislabelled.
- **One fragment**, against their own stated exclusion.

The rebuilt file rejects a protein *entirely* if any of its spans falls outside
the window, rather than keeping it with that span unlabelled. That costs 2,400
proteins and is why check 1 passes.

## What it covers

The eligible population is reviewed UniProt, non-viral, non-fragment, with at
least one non-CAAX propeptide: **11,200 proteins and 13,187 spans**.

| | distributed | rebuilt |
|---|---|---|
| proteins | 7,213 — 64.4% of eligible | **8,800 — 78.6%** |
| spans actually labelled | 8,211 — 62.3% | **9,900 — 75.1%** |
| unlabelled propeptides inside kept proteins | **680** | **0** |
| negatives | 1,236 | 4,781 (`--max_negatives 1700` restores the old ratio) |

So the rebuild is +14.2 points of protein coverage, +12.8 points of span
coverage, and it removes the unlabelled-positive defect entirely.

## Why the window is 5..100 and not something else

The floor and the ceiling are doing different jobs, and only one of them was
right in the original.

**The floor stays at 5 because it is a mechanism filter.** All 781 CAAX features
in reviewed UniProt are shorter than 5 residues — 100% of them. That is a
different reaction: prenylation of a C-terminal cysteine, then RCE1 removing the
aaX tripeptide, then methylation. Checked in the sequences, not inferred: of
twelve 3-residue C-terminal propeptides sampled from prenylated proteins, twelve
are immediately preceded by cysteine. No dibasic site, fixed length, fixed
position. `build_dataset.py` drops those spans and keeps their proteins.

**The ceiling rises to 100 because it was a truncation.** 27.9% of
convertase-processed propeptides are longer than 50 residues. The old cap does
not select a mechanism, it cuts one in half. With the floor held at 5, the
convertase class is covered 64.2% at max 50 and **77.9% at max 100**.

101 states is also DeepPeptide's own budget — 1 background + 50 peptide + 50
propeptide — reallocated so every non-background state models propeptides. Going
further to 151 states would add only 5.9 points of convertase coverage for 2.3×
the decode cost.

## What it still does not cover, stated with numbers

- **11.5% of non-CAAX propeptides are longer than 100 residues** and are out of
  scope. Mostly TGF-beta LAPs, NGF-beta, venom metalloproteinase and large
  peptidase S8/C1 prodomains.
- **The 781 CAAX features**, deliberately.
- **2,400 proteins** rejected because one of their spans falls outside the
  window. Keeping them would reintroduce the defect the rebuild removes.
- **Viral proteins and fragments**, following Teufel et al.
- **Negatives are unlabelled, not negative.** This is positive-unlabeled data:
  37% of the 2022 negatives had gained a propeptide annotation by 2026. No filter
  fixes that; the write-up has to say it.

## One open warning

587 sequences appear more than once under different accessions (1,438 redundant
rows). They share one embedding file, since embeddings are keyed by sequence
hash, and they must not straddle a partition boundary. GraphPart at 30% identity
should place identical sequences together — verify it did, after partitioning,
before trusting any number.
