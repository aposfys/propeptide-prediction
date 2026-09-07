# ESM-2, sequence only — the baseline arm

One of three arm branches for the propeptide comparison. **101 CRF states,
propeptide-only labels, the published dataset.**

| | |
|---|---|
| representation | ESM-2 `esm2_t33_650M_UR50D`, sequence only |
| embedding dims | `--embedding_dim 1280` |
| grammar | 101 states: background + propeptide positions 1–100 |
| labels | two: none, propeptide |
| data | `data/labeled_sequences.csv`, unchanged |
| hyperparameters | T4, replayed from `results/esm2_rep1/config.json` |

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

Embeddings already exist; no extraction needed.

```bash
ARM=esm2 EMB=/mnt/storage/fysekidis/embeddings/esm2 DIM=1280 bash run_arm.sh 8
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
