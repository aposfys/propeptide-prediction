# DeepPeptide (ESM3) — propeptide cleavage prediction

Adapted from the original DeepPeptide (Teufel et al., Bioinformatics 2023) to use
ESM3 (`esm3_sm_open_v1`, 1536-dim) as the sequence encoder and a propeptide-only
CRF (51 states: background + 50 propeptide positions). The training logic is
upstream's — constant Adam LR, CRF negative log-likelihood, best-on-validation
checkpointing — with only the changes those two differences require.

This is the consolidated ESM3 branch. It holds the sequence-only and
structure-conditioned extractors, LoRA fine-tuning, the Optuna / nested-CV
hyperparameter search, the ensembling scripts, and the analysis tooling
(`verify_embeddings.py`, `summarize_results.py`, `progress.py`). It was formed by
merging the former `esm3-multimodal-propeptide` and `esm3-propeptide-optuna-gpu`
branches, so paths and commands from either still apply.

The experimental protocol governing every arm of the comparison is in
[EXPERIMENT.md](EXPERIMENT.md); measured results are in [RESULTS.md](RESULTS.md).

> **⚠ ESM3 embeddings made before 2026-08-19 are mis-scaled by ~840× and every
> result from them is invalid.** They can be repaired without re-running ESM3, and
> `preflight.py` refuses to start on them. See [EMBEDDINGS.md](EMBEDDINGS.md).

## Before you run this

This code accompanies an MSc thesis. **If you intend to run it, please contact me
first** — apostolosfysekidis1@gmail.com. I would like to know who is using it.

The trained model weights and the search outputs are **not published here**. They
are available from me on request. Without them you can read, adapt and rerun the
method, but you cannot reproduce the reported numbers directly.

This repository is MIT licensed, so the licence does not oblige you to make contact.
The above is a request, not a condition.

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

## Setup, options and output

[`TRAINING.md`](TRAINING.md) documents what you need before a run, how to choose a
protocol, the command-line options, what the search optimises over, and every file
a run writes.

## Where to read more

- **[EMBEDDINGS.md](EMBEDDINGS.md)** — the ESM3 embedding scaling bug and how to repair
  affected embeddings.
- **[OPTUNA_GPU.md](OPTUNA_GPU.md)** — the search space and where it comes from,
  cost estimates, and the reasoning behind the defaults.
- **[RESULTS.md](RESULTS.md)** — results so far, and the ESM-2 comparison this
  search is meant to settle.
- **[CHANGELOG.md](CHANGELOG.md)** — what differs from the original DeepPeptide.
- **[predictor/README.md](predictor/README.md)** — inference with the original
  pretrained model. Not used by the search.
