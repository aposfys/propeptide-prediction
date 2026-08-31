# Options, search space and output — esm3-propeptide

Reference for running the hyperparameter search on this branch. For what the
branch is and how to start a run, see [`README.md`](README.md). For the reasoning
behind the search space and its costs, see [`OPTUNA_GPU.md`](OPTUNA_GPU.md).

## Common options

| flag | default | meaning |
|---|---|---|
| `--fold N` | all 5 | run one outer fold only (`--fold 0` = the ablation protocol) |
| `--n_trials` | 30 | Optuna trials per outer fold |
| `--epochs` | 50 | epochs for the final retraining |
| `--optuna_epochs` | 35 | epochs during the search (shorter, to save time) |
| `--patience` | 10 | early-stopping patience; the best checkpoint is kept regardless |
| `--seed` | 42 | seeds the sampler, so a search is reproducible |
| `--prune` | off | abandon clearly-losing trials early; saves time, makes the search non-exhaustive |
| `--space` | `table_s1` | search space; see [OPTUNA_GPU.md](OPTUNA_GPU.md) |

Running on one GPU, the folds go sequentially — that is what `run_optuna_gpu.sh`
does, and it is the only launcher on this branch. (The CPU branches carry
`run_parallel.sh` / `run_simple_cv.sh`, which start all 5 folds at once. Those are
deliberately absent here: on a single GPU that means five copies of the model
competing for VRAM plus ~50 GB of host RAM, because the embedding cache is
per-process.) With several GPUs, run one fold per GPU:

```bash
for f in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=$f nohup bash run_optuna_gpu.sh --fold $f \
      --out_dir results/esm3_prop_optuna > logs/fold$f.log 2>&1 &
done
```

## What Optuna optimises

These are searched in the inner CV loop, so their CLI defaults are **not** what
gets trained:

`--lr`, `--batch_size`, `--dropout`, `--conv_dropout`, `--kernel_size`,
`--num_filters`, `--hidden_size` — seven parameters. (`--weight_decay` is searched
only under `--space wide`; `table_s1` leaves it at 0, as upstream does.)

Setting any of them on the command line has no effect on a search — it only
affects a plain single-split `run.py` / `run_fold.py` training run.

Because it is a seven-dimensional space, **`--n_trials` needs to be ≥ ~30.** Very
low trial counts sample it too thinly to find anything; the default is 30.

## Output

Two scripts read the results, and they answer different questions:

```bash
# winning hyperparameters + F1 vs the reference numbers, at ±3 tolerance
python summarize_optuna.py --out_dir results/esm3_prop_optuna

# precision/recall/F1 swept across tolerances 0–3, as in the paper's Fig. 2a/b
python evaluation/measure_performance.py \
    --out_dir results/esm3_prop_optuna \
    --data_file data/labeled_sequences.csv
```

`measure_performance.py` discovers `test_outputs_outer*_inner*.pickle`, which only
the nested-CV path writes — it does not work on a plain `run_fold.py` run. The two
scripts agree at ±3; use the sweep when you need the tolerance curve, and
`summarize_optuna.py` when you need the hyperparameters.

Files in `--out_dir`:

| file | contents |
|---|---|
| `best_params_outer{N}.json` | the winning hyperparameters, complete |
| `effective_config_outer{N}.json` | full config + best inner-CV F1; reproduces the run on its own |
| `optuna_trials_outer{N}.csv` | every trial, for inspecting the search itself |
| `fold_summary_outer{N}.json` | test metrics for that fold's models |
| `test_outputs_outer{N}_inner{i}.pickle` | raw predictions, read by `measure_performance.py` |
| `metrics_per_model.csv`, `metrics_aggregated.csv` | written by `measure_performance.py` |
| `model_outer{N}_inner{i}.pt` | trained models (git-ignored) |

## What preflight checks

| check | why it matters |
|---|---|
| dependencies, data files, CUDA device | fails in minutes instead of hours |
| every required embedding hash present | a missing file kills a run mid-epoch |
| embedding dimension | ESM-2 (1280) / ESM3 (1536) / ProstT5 (1024) are easy to mix up |
| **embedding scale ≈ 0.3 × √dim** | catches the pre-LayerNorm bug above. Same shape, same dtype, same filenames — only the values differ |
| **rows == sequence length** | catches BOS/EOS mishandling, which shifts every label without crashing |
| NaN/Inf, all-zero, mixed scale across files | catches a partially written extraction |
| **stale Optuna study in `--out_dir`** | catches the zero-trial resume described above |
| free disk, host RAM for the embedding cache | each fold writes ~1.8 GB of prediction pickles |

## What you need

| | | |
|---|---|---|
| code | ships with the repo | — |
| data | `data/labeled_sequences.csv`, `data/graphpart_assignments.csv` | ships with the repo |
| **embeddings** | `{md5-of-sequence}.pt`, each `(length, 1536)` float32 | **~12 GB, does NOT ship — see below** |
| **GPU** | CUDA device | **required** — the run aborts without one |

**The embeddings must be on the machine that trains.** They are read from disk on
the first epoch of every fold. ESM3 itself is never loaded during training — its
weights are frozen and the embeddings already capture its output — so you need the
embedding directory, not the model.

Either copy the directory across:

```bash
rsync -av --progress user@source:/path/to/embeddings/esm3/ /local/embeddings/esm3/
```

or regenerate them (this step *does* load ESM3 and wants a GPU; note that
`esm3_sm_open_v1` is gated on Hugging Face, so you must be logged in):

```bash
huggingface-cli login
python src/utils/make_embeddings.py data/protein_sequences.fasta /local/embeddings/esm3_normed
```

`data/protein_sequences.fasta` holds 14,583 records that deduplicate to **8,061
unique sequences**, so that is how many `.pt` files should appear. The extractor
**skips any hash it already finds**, so regenerating into a directory that already
has files writes nothing and looks like success — always use a fresh directory.

If you already have a set made before the LayerNorm fix, repair it instead of
re-running ESM3 — it takes minutes on CPU rather than 8,061 GPU forward passes:

```bash
python -m src.utils.renorm_esm3_embeddings /path/to/esm3 /path/to/esm3_normed
```

It refuses to run on embeddings that already look normalised, so it cannot be
applied twice by accident.

## Choosing a protocol

The paper uses two, and for comparing encoders it uses the cheaper one — *"these
model ablation experiments were done in standard cross-validation using partition 0
as the test set"*. Full nested CV was reserved for its two final models.

| | what it is | models | cost | command |
|---|---|---|---|---|
| **Ablation** | standard CV, partition 0 held out | 4 | 1 search | `--fold 0` |
| **Full nested CV** | 5 outer folds, mean ± std | 20 | 5 searches | *(omit `--fold`)* |

**Start with the ablation.** It answers "is ESM3 competitive once tuned?" at a
fifth of the cost, using the protocol the paper itself used for that question. Run
the full nested CV once the ablation says it is worth it.

Either way, each Optuna trial is scored by 4-fold inner CV — the winner is the set
with the best mean validation F1 across the 4 inner models — and the resulting
models are scored on the held-out test partition at ±3 residue tolerance.
