# ESM3 propeptide-only — hyperparameter search on GPU

Branch: `esm3-propeptide-optuna-gpu` (branched from `esm3-propeptide`)

## What this is for

The ESM3 propeptide-only model currently sits at **F1 0.511**, trained on the
untuned defaults (`lr 1e-4`). The equivalent ESM-2 model, using the paper's tuned
"T4" hyperparameters (`lr 0.0055`), reaches **F1 0.626**. The open question is
whether ESM3's deficit is real or just a consequence of never having been tuned.

This branch runs the hyperparameter search that answers it.

| model | hyperparameters | F1 (propeptide, ±3) |
|---|---|---|
| ESM-2 propeptide-only | T4 (tuned) | **0.626** |
| ESM3 propeptide-only | default, `lr 1e-4` | 0.511 |
| ESM3 propeptide-only | **tuned — this run** | _to be measured_ |

## Relationship to the original DeepPeptide

The training logic is unchanged from upstream `fteufel/DeepPeptide`: constant Adam
learning rate (no scheduler), CRF negative log-likelihood, best-on-validation
checkpointing. The only deviations are the two the project requires — ESM3
embeddings (1536-dim instead of 1280) and propeptide-only labelling (2 labels /
51 CRF states instead of 3 / 101) — plus early stopping on validation propeptide
F1, which cannot change a reported number because the best checkpoint is saved
regardless of when training stops.

Upstream ships **no hyperparameter-search code**, so the search space is not
inherited from the repo. It is taken from the paper's **Table S1** (btad616
supplementary), reproduced parameter-for-parameter:

| Parameter | Lower | Upper | Distribution |
|---|---|---|---|
| Learning rate | 0.0001 | 0.01 | log-uniform |
| Batch size | 10 | 100 | step 10 |
| Embedding dropout | 0 | 0.7 | uniform |
| Convolution dropout | 0 | 0.7 | uniform |
| Kernel size | 1 | 5 | step 2 |
| CNN channels | 40 | 128 | step 8 |
| LSTM hidden size | 16 | 192 | step 16 |

`weight_decay` is deliberately **not** searched: Table S1 does not list it and
upstream never exposes it (Adam stays at its default 0).

Table S1's space was written for the ESM-1b/ESM-2 models, and this is an ESM3
model — reusing it is a deliberate choice. It keeps the ESM3-vs-ESM-2 comparison
clean (a different space would confound "better embedder" with "better search"),
and the embedder change only alters conv1's *input* width, which is set by
`--embedding_dim` and never searched. If ESM3's optimum genuinely lies outside
this box, the search reports it by piling every fold's winner against a bound;
`summarize_optuna.py` checks for that on the learning rate and says so.

### Which protocol — run the cheap one first

The paper uses two, and for an embedder comparison it uses the cheap one. From the
methods: *"To save resources, these model ablation experiments were done in
standard cross-validation using partition 0 as the test set."* Full nested CV was
reserved for the two final models (ESM-1b L32, ESM-2 L33).

| protocol | what it is | models | cost | command |
|---|---|---|---|---|
| **Ablation** | standard CV, partition 0 held out — the paper's setup for comparing embedders, which is exactly what ESM3-vs-ESM-2 is | 4 | 1 search | `--fold 0` |
| **Full nested CV** | 5 outer folds, mean ± std — the headline setup | 20 | 5 searches | *(no `--fold`)* |

**Start with the ablation.** It answers "is ESM3 competitive once tuned?" at a
fifth of the cost, and it is the protocol the paper itself used for this question.
Only spend the full nested-CV budget once the ablation says ESM3 is worth it.

Within either protocol, each Optuna trial is scored by 4-fold inner CV: the winner
is the set with the best **mean validation F1 across the 4 inner models**, exactly
as Table S1's caption describes. Those 4 models are then retrained and scored on
the held-out test partition, at ±3 tolerance, reported as mean ± std — matching
"the average and standard deviations over 20 models on their held-out test fold".

## Prerequisites

**The precomputed ESM3 embeddings must be on the machine that runs the search.**
The model itself is small and trains on cached embeddings — ESM3 is never loaded
during training, its weights are frozen and it has already done its job. So you
need the embedding directory, not the ESM3 model:

- `/data/apostolos/embeddings/esm3` — one `{md5-of-sequence}.pt` per sequence,
  each `(L, 1536)` float32, ~12 GB total for the 7,623-sequence benchmark.
- They are read via `--embeddings_dir` and cached in RAM on first epoch.
- If any are missing, regenerate with
  `python src/utils/make_embeddings.py data/protein_sequences.fasta /data/apostolos/embeddings/esm3`
  (this step *does* load ESM3 and wants a GPU).

## Where this runs

**This branch is GPU-only.** A search is hundreds of trainings; on CPU the same
job takes orders of magnitude longer, so rather than degrade quietly it aborts
when no CUDA device is visible — in `preflight.py`, in `run_optuna_gpu.sh`, and in
`train_loop_crf.py` itself. (`--allow_cpu` overrides the last of these, for
smoke-testing plumbing only. Results from such a run are not usable.)

HPC7 (`apostolos@192.168.20.107`) has **no GPU**, so nothing trains there. Its
role is to hold the code and the embeddings for collection:

```bash
ssh apostolos@192.168.20.107
cd /data/apostolos/deeppeptide-esm3
git fetch origin && git checkout esm3-propeptide-optuna-gpu && git pull
```

Running preflight there will report the missing GPU as a failure, which is
correct. It is still useful for checking the data and embeddings — read past that
one line.

## Running it

On the GPU machine:

```bash
conda activate deeppeptide

# confirm a GPU is visible before committing hours to it
python -c "import torch; print('cuda:', torch.cuda.is_available())"

# ABLATION — the recommended first run (partition 0 held out, 4 models)
nohup bash run_optuna_gpu.sh --fold 0 \
    --embeddings_dir /data/apostolos/embeddings/esm3 \
    --out_dir results/esm3_prop_optuna_ablation \
    > logs/optuna_ablation.log 2>&1 &

tail -f logs/optuna_ablation.log
```

Drop `--fold 0` for the full 5-fold nested CV. Defaults in `run_optuna_gpu.sh`:
`--n_trials 30 --epochs 50 --optuna_epochs 35 --patience 10 --seed 42
--space table_s1`. Override any by appending the flag.

### If the search runs on different hardware than HPC7

The embeddings must sit on whichever machine runs the training — they are read
from disk every first epoch. Either copy them across (~12 GB):

```bash
rsync -av --progress apostolos@192.168.20.107:/data/apostolos/embeddings/esm3/ \
    /path/on/gpu/machine/embeddings/esm3/
```

or regenerate them there (this step needs ESM3 and a GPU):

```bash
python src/utils/make_embeddings.py data/protein_sequences.fasta <emb_dir>
```

## Letting the search range wider

`--space` picks the search space:

- `table_s1` (default) — the paper's own space. **Use this for anything compared
  against the ESM-2 numbers.** Tuning both embedders over the same space is what
  makes the comparison a statement about the embedder rather than about who got
  the bigger search.
- `wide` — every bound loosened, plus `weight_decay` searched. Exploratory only:
  it is **not** comparable to the ESM-2 T4 result, and a bigger space needs a
  bigger `--n_trials`; reusing the default budget samples it thinly and usually
  scores *worse*. `summarize_optuna.py` prints a warning when results came from
  this space, so a `wide` number cannot be mistaken for a comparable one.

Both are recorded in `effective_config_outer{N}.json`, so every result carries the
space that produced it.

### Folds run sequentially, on purpose

`run_optuna_gpu.sh` runs the 5 outer folds one after another, and it is the only
launcher on this branch. The CPU branches carry `run_parallel.sh` and
`run_simple_cv.sh`, which start all 5 folds at once; both are deliberately absent
here. On a 48-core CPU node that is the right thing to do, but on a single GPU it
means five copies of the model and its padded activations competing for VRAM, plus
~50 GB of host RAM because the embedding cache is per-process — and time-slicing
one GPU buys no wall-clock benefit anyway.

If the node has several GPUs, run one fold per GPU instead:

```bash
for f in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=$f nohup bash run_optuna_gpu.sh --fold $f \
      --out_dir results/esm3_prop_optuna > logs/fold$f.log 2>&1 &
done
```

### Cost

One trial = 4 inner-fold trainings of ≤35 epochs.

| protocol | trainings |
|---|---|
| ablation (`--fold 0`, 30 trials) | 120 + 4 retrains at 50 epochs |
| full nested CV (30 trials × 5 folds) | 600 + 20 retrains at 50 epochs |

If it is too slow: lower `--n_trials`, or add `--prune`, which abandons a trial
once its running inner-fold mean is clearly behind. Pruning is off by default
because it makes the search non-exhaustive.

## Reading the results

```bash
python summarize_optuna.py --out_dir results/esm3_prop_optuna
```

Prints per-fold and overall F1 (mean ± std over the 20 models), compares against
the 0.511 and 0.626 reference points, and tabulates each fold's winning
hyperparameters. Works on a partial run and tells you how many folds are missing.

Per-fold outputs in `--out_dir`:

| file | contents |
|---|---|
| `best_params_outer{N}.json` | the winning hyperparameters — complete, every searched value |
| `effective_config_outer{N}.json` | full config + best inner-CV F1 + trial counts; reproduces the run on its own |
| `optuna_trials_outer{N}.csv` | every trial, for inspecting the search itself |
| `fold_summary_outer{N}.json` | test metrics for that fold's 4 models |
| `model_outer{N}_inner{i}.pt` | the 4 trained models (git-ignored) |

## Caveat

The 0.511 and 0.626 reference numbers are from a **single** Graph-Part split,
whereas this run is full nested CV over 5 folds. The nested-CV mean carries error
bars and is the more trustworthy figure, but it is not strictly the same quantity
as the two single-split numbers — expect a modest shift from the change in
protocol alone, independent of tuning. Compare like with like when writing this
up.
