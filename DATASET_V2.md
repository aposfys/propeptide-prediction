# Can I use a bigger dataset with a 101-state, 2..100 CRF?

**Yes, and it is valid — but it is a different task, not a bigger version of this
one, and there is one step you must do first that costs almost nothing.**

Reproduce every number here with:

```bash
python -m src.utils.propeptide_class_audit --cache propep_rich.tsv
python -m src.utils.dataset_currency_audit --cache propep.tsv
```

## The grammar itself is fine

`--max_peptide_len 100 --min_peptide_len 2` gives 101 states: 1 background plus
100 propeptide positions. Verified working, not assumed:

- the CRF loss is finite on a 78-residue propeptide label
- every length from 2 to 100 is a legal path under the constraint mask
- Viterbi returns full-length paths; marginals come back at (batch, L, 101)
- gradients reach states 51–100, because the partition function covers every
  path whether or not the data uses it

Cost is 200,069 trainable parameters against 192,369, so +4%. Emissions are
shared across states, so the extra states cost transitions, not width. Decoding
is 3.9× slower, since Viterbi is O(L·S²).

**One trap.** A 101-state run scored with `stop_state=50` does not report zero.
It reports a *truncated* span: a true propeptide at 11–88 comes back as 11–38,
because the decoder closes the segment the first time it sees state 50, which
now sits mid-path. `compute_all_metrics` takes `end_state` for exactly this
reason and the training loop passes `--max_peptide_len` into it. Never score
across grammars without checking that argument.

## Do this first, before building anything

**Run the 101-state grammar on the existing benchmark.** Same data, same labels,
same embeddings, same hyperparameters — only the state space changes. Every
propeptide in the file is 5–50, so the extra 50 states are never visited by a
label, and predictions should be identical.

If F1 holds, the wider grammar is free and you can spend on the dataset. If F1
drops, you have learned that the extra states cost accuracy through the
transition matrix alone, before spending a week on a rebuild. There is reason to
expect some cost: emissions are shared across all propeptide states, so the CRF
separates lengths by transitions alone, and 2..100 asks it to distinguish 99
length hypotheses instead of 46 from the same training signal.

This is 8 runs. Do it before anything else in this document.

## The real problem is not the grammar, it is what the window lets in

`PROPEP` is one UniProt feature key covering at least three unrelated processes.
The 5–50 window happens to select one of them; 2..100 merges all three.

| marker | 2–4 aa | 5–50 aa | 51–100 aa |
|---|---|---|---|
| Prenylation | **42.4%** | 0.2% | 0.2% |
| Methylation | **50.8%** | 1.5% | 0.3% |
| Cleavage on pair of basic residues | 16.8% | **28.0%** | 36.9% |
| Zymogen | 5.7% | 12.1% | **43.8%** |
| Protease | 5.9% | 11.8% | **44.2%** |
| C-terminal position | **58.2%** | 27.9% | 14.6% |
| internal position | 22.8% | 59.0% | **75.7%** |

- **2–4 aa is CAAX processing.** Prenylation on a C-terminal cysteine, then RCE1
  removes the "aaX" tripeptide, then ICMT methylates the cysteine. UniProt
  annotates the removed tripeptide as `PROPEP`. Checked directly in the
  sequences: of twelve 3-residue C-terminal propeptides sampled from prenylated
  proteins, **twelve out of twelve are immediately preceded by cysteine**. There
  are 807 such features. The recognition signal is the CaaX box. There is no
  dibasic site and the length is always 3.
- **5–50 aa is proprotein convertase processing.** Secretory precursors cut at
  mono- and dibasic sites. Insulin family, conotoxins, frog skin peptides,
  peptidase S1. This is what the published numbers describe.
- **51–100 aa is zymogen prodomains.** Peptidase S8, C1, M10A. Folded inhibitory
  domains acting as intramolecular chaperones, usually removed autocatalytically,
  and overwhelmingly internal.

A single CRF with one propeptide label and one shared emission has to learn all
three from one transition matrix. A 2..100 positive set is:

| | features | share |
|---|---|---|
| 5–50, mechanism not keyworded | 6,554 | 52.7% |
| convertase, dibasic keyword | 3,301 | 26.5% |
| CAAX / aaX removal | 781 | 6.3% |
| other short (2–4) | 774 | 6.2% |
| zymogen prodomain (51–100) | 680 | 5.5% |
| other long (51–100) | 355 | 2.9% |

**26.9% of the positive class would sit outside the window the current model was
built around**, with a different recognition signal. A change in F1 after
widening cannot be attributed to the model, the embeddings or the grammar,
because the task changed at the same time.

## The blocking problem: the negatives

This is the part that stops a naive rebuild.

The benchmark's 1,231 negatives are proteins with a mature-peptide annotation and
no propeptide. Checking them against UniProt 2026_04:

| | n | share |
|---|---|---|
| still no propeptide | 773 | 62.8% |
| now has one in 2..100 but not 5..50 | **374** | **30.4%** |
| now has one, outside 2..100 | 80 | 6.5% |
| now has one inside 5..50 | 4 | 0.3% |

**Under the current 5..50 grammar, 4 of 1,231 negatives are wrong. Under 2..100,
378 are — 30.7% of the negative set.**

So widening does not only add positives from outside the file. It relabels a
third of the proteins the model is trained to call negative. Any rebuild must
redo the negative set, and "negative" here means "no propeptide was annotated",
not "this protein has no propeptide" — UniProt never asserts absence. The
available negative pool under the current scheme is 5,329 reviewed proteins with
a peptide annotation and no propeptide.

That 0.3% is worth dwelling on. The 5–50 window is not just convenient; it is a
window in which UniProt's negatives happen to be **trustworthy**. That property
does not survive widening, and it is not recoverable by curation you can do.

## What DeepPeptide actually filtered, confirmed

From the paper: UniProt Release 2022_01, "excluding viral proteins and protein
fragments", annotations filtered to 5–50 AAs and proteins with none in range
discarded. All three are visible in the file — zero viral proteins across 1,660
organisms, no fragments — and none of them is stated in the repository. The
length filter was applied per *annotation*, which is why 548 benchmark proteins
carry 715 real propeptides their labels omit. See [DATASET.md](DATASET.md).

## Verdict, and what I would actually do

**Valid, under three conditions, and only as a second study.**

1. **Run the 101-state grammar on the current data first.** Eight runs. It
   isolates the grammar's cost from the dataset's, and it is the only cheap
   experiment here.
2. **Do not pool the mechanisms silently.** Either exclude the CAAX class
   (`Prenylation` keyword and length ≤ 4 at the C-terminus removes 781 features
   and most of the incoherence), or keep them and report per-bin metrics so a
   gain on easy CAAX motifs cannot hide a loss on convertase sites. Reporting one
   pooled F1 over three mechanisms is the failure mode to avoid.
3. **Rebuild the negatives, not just the positives**, and say in the write-up
   that negatives mean "unannotated". Then re-run GraphPart, which at ~15,800
   sequences is roughly 3.5× the pairwise alignment load of the original.

And keep it separate from the thesis. Every number in [RESULTS.md](RESULTS.md)
comes from the 5–50 propeptide-only setup, and the thesis claim is a controlled
comparison across embeddings. This changes the task, the labels and the negative
set at once. It is a good second paper. It is not an upgrade to the first one.

If you want the middle path: **2..100 minus the CAAX class** is 9,697 proteins
and 11,309 features against the current 7,446 and 8,207 — most of the size gain,
without the mechanism that is provably a different reaction.
