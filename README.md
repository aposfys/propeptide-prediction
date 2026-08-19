# DeepPeptide (ESM3) — propeptide cleavage prediction

Adapted from the original DeepPeptide (Teufel et al., Bioinformatics 2023) to use
ESM3 (`esm3_sm_open_v1`, 1536-dim) as the sequence encoder and a propeptide-only
CRF (51 states: background + 50 propeptide positions). The training logic is
upstream's — constant Adam LR, CRF negative log-likelihood, best-on-validation
checkpointing — with only the changes those two differences require.

This branch runs the **hyperparameter search** for that model.

> ### ⚠ Read this before reusing any ESM3 embeddings made before 2026-08-19
>
> `src/utils/make_embeddings.py` used to save `ESMOutput.embeddings`. That field is
> ESM3's **raw pre-LayerNorm residual stream**, not the representation its own
> output heads consume — `TransformerStack.forward` returns `self.norm(x), x, …`
> and `ESM3.forward` unpacks it as `x, embedding, _`. fair-esm does the opposite for
> ESM-2 (it applies `emb_layer_norm_after` and overwrites `representations[33]`), so
> **the ESM-2 baseline trained on normalised features and ESM3 did not.**
>
> | features | per-token ‖x‖ |
> |---|---|
> | ESM-2 L33 (the baseline) | 10.12 |
> | ESM3 after `transformer.norm` | 11.62 |
> | ESM3 `.embeddings`, as previously saved | **9792.73** |
>
> ~840× too large. `LSTMCNN` has no input normalisation, so this saturates **90.7%**
> of the biLSTM gates at initialisation. It is why the optimal learning rate
> collapsed, why ESM-2's T4 settings "broke" ESM3, and why the first 30-trial search
> sat on a flat plateau. Every ESM3 result predating the fix is invalid — see
> [RESULTS.md](RESULTS.md).
>
> **Old embeddings can be repaired without re-running ESM3.** The final norm is
> per-token, so it commutes with the BOS/EOS slice:
>
> ```bash
> python -m src.utils.renorm_esm3_embeddings /path/to/esm3 /path/to/esm3_normed
> ```
>
> `preflight.py` now refuses to start a run on mis-scaled embeddings.

## Quick start

Four steps. Run everything from the repository root.

```bash
# 1. environment  (needs a CUDA torch build — see requirements.txt)
conda create -n deeppeptide python=3.10 -y && conda activate deeppeptide
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda)"  # cuda must NOT be None

# 2. check the setup before spending hours on it
python preflight.py --embeddings_dir /path/to/embeddings/esm3_normed \
                    --out_dir results/esm3_prop_optuna_normed

# 3. run the search  (--out_dir MUST be new — see below)
bash run_optuna_gpu.sh --fold 0 \
    --embeddings_dir /path/to/embeddings/esm3_normed \
    --out_dir results/esm3_prop_optuna_normed

# 4. read the results
python summarize_optuna.py --out_dir results/esm3_prop_optuna_normed
```

`run_optuna_gpu.sh` runs preflight itself and aborts if it fails, so step 2 is only
needed when checking a machine ahead of time.

**`--out_dir` must be new for a new search.** Optuna persists each study to SQLite
and `train_nested_cv()` resumes with `load_if_exists=True`, running only
`n_trials - already_done` more. Pointing at a directory that already holds a
finished study runs **zero** new trials, silently retrains from the old
`best_params`, and writes fresh-looking output. Preflight now detects this and
fails.

### What preflight checks

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

## Where to read more

- **[OPTUNA_GPU.md](OPTUNA_GPU.md)** — the search space and where it comes from,
  cost estimates, and the reasoning behind the defaults.
- **[RESULTS.md](RESULTS.md)** — results so far, and the ESM-2 comparison this
  search is meant to settle.
- **[CHANGELOG.md](CHANGELOG.md)** — what differs from the original DeepPeptide.
- **[predictor/README.md](predictor/README.md)** — inference with the original
  pretrained model. Not used by the search; safe to exclude from a clone (it is
  ~300 MB of the repository).
