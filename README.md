# ESM3, sequence + structure — arm 2

One of three arm branches for the propeptide comparison. **51 CRF states,
propeptide-only labels, the published dataset.**

| | |
|---|---|
| representation | ESM3 `esm3_sm_open_v1`, sequence + coordinates + structure tokens + SASA |
| embedding dims | `--embedding_dim 1536` |
| grammar | 51 states: background + propeptide positions 1–50 |
| labels | two: none, propeptide |
| data | `data/labeled_sequences.csv`, unchanged |
| hyperparameters | T4, replayed from `results/esm2_rep1/config.json` |

## Why 51 states and not 101

The published model has 101 states, but they are **two branches of 50**: states
1–50 are peptide positions 1–50, states 51–100 are *propeptide* positions 1–50.
The propeptide half is 50 states and caps propeptide length at 50.

So a propeptide-only model with 51 states already has **exactly** the original's
propeptide capacity. Nothing was lost by dropping the peptide branch. Using 101
states here would not be parity with the published model — it would double the
propeptide branch beyond anything DeepPeptide had.

On this dataset that extra half is unusable: every propeptide in the benchmark is
5–50 residues, so states 51–100 would never be visited by a label, at 3.9× the
decode cost. `MAX_LEN=100 bash run_arm.sh` runs the 101-state variant if you want
it; the grammar ablation on `paper-3arm` measures what it costs.

Coverage, for the record: the 5..50 window reaches 69.0% of non-CAAX propeptides
in current UniProt and 64.2% of the convertase class. Widening to 100 would reach
80.2% and 77.9% — but only with data that contains them, which this benchmark
does not.

## The other two arms

| branch | representation |
|---|---|
| [`esm2-101`](../../tree/esm2-101) | ESM-2, sequence only — **the baseline** |
| [`esm3-101`](../../tree/esm3-101) | ESM3, sequence + structure tracks |
| [`prost5-101`](../../tree/prost5-101) | ProstT5, sequence + 3Di |

**The training core is byte-identical across all three.** `train_loop_crf.py`,
`crf_models.py`, `dataset.py`, `manuscript_metrics.py`, `crf_label_utils.py`
and `run.py` are the same blobs on every branch, so a difference in F1 is a
difference in the representation and nothing else. Check it:

```bash
python test_core_identical.py
```

That check exists because it has already gone wrong: before the arms were split,
`esm3-multimodal` and `prost5-multimodal` differed in four of those files, and
`RESULTS.md` warns against comparing F1 across branches for that reason.

## Running this arm

First fetch structures and extract, then train:

```bash
python -m src.utils.fetch_afdb_structures --out_dir structures/
python -m src.utils.make_embeddings_esm3_struct \\
    --structures_dir structures/ --out_dir embeddings/esm3_struct \\
    --gpu_max_len 2000 --max_struct_len 1024
```

Its own sequence-only control at 101 states is required, or this arm reads only
as a loss to the baseline:

```bash
python -m src.utils.make_embeddings_esm3_struct \\
    --structures_dir structures/ --out_dir embeddings/esm3_seqonly --no_structure
```

`--scramble_structure` is the negative control: tracks present, residue axis
permuted, no true fold.

```bash
ARM=esm3_struct EMB=embeddings/esm3_struct DIM=1536 bash run_arm.sh 8
```

Prefix with `DRY_RUN=1` to print the commands without running them. Existing
output directories are skipped, so a partial batch can be resumed.

Then:

```bash
python -m src.utils.summarize_results results/
```

## Before trusting anything from here

```bash
python test_architecture.py     # the head is the published one
python test_grammar.py          # the grammar does what it claims
python test_training_smoke.py   # the loop runs at 51 and 101 states
python test_core_identical.py   # the three arms share a training core
```

The plan, its gates and what each result would mean are in [PLAN.md](PLAN.md).
