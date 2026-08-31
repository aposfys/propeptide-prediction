#!/usr/bin/env bash
# Optuna hyperparameter search for the ESM3 propeptide-only model, on GPU.
#
# Goal: close the gap between the untuned ESM3 propeptide model (F1 0.511, lr 1e-4)
# and the tuned ESM-2 propeptide model (F1 0.626, lr 5.5e-3). The search space in
# src/train_loop_crf.py:objective() spans both regimes.
#
# ---------------------------------------------------------------------------
# Which protocol to run
# ---------------------------------------------------------------------------
# The paper uses TWO protocols, and for an embedder comparison it uses the cheap
# one. From the methods: "To save resources, these model ablation experiments were
# done in standard cross-validation using partition 0 as the test set." Full nested
# CV was reserved for the two final models (ESM-1b L32, ESM-2 L33).
#
#   ABLATION (recommended first run) — standard CV, partition 0 held out.
#   This is the paper's protocol for comparing embedders, which is exactly what
#   ESM3-vs-ESM-2 is. One Optuna search, 4 models, ~1/5 the cost.
#
#       bash run_optuna_gpu.sh --fold 0 \
#           --embeddings_dir /data/apostolos/embeddings/esm3 \
#           --out_dir results/esm3_prop_optuna_ablation
#
#   FULL NESTED CV — all 5 outer folds, 20 models, mean +/- std. This is the
#   headline protocol, and what you run once the ablation says ESM3 is worth it.
#
#       bash run_optuna_gpu.sh \
#           --embeddings_dir /data/apostolos/embeddings/esm3 \
#           --out_dir results/esm3_prop_optuna
#
#   Several GPUs available? Spread the 5 folds across them:
#       CUDA_VISIBLE_DEVICES=0 bash run_optuna_gpu.sh --fold 0 --out_dir results/... &
#       CUDA_VISIBLE_DEVICES=1 bash run_optuna_gpu.sh --fold 1 --out_dir results/... &
#
# Anything after the recognised flags is forwarded to src.train_loop_crf, so
# --n_trials / --epochs / --optuna_epochs / --seed / --space can all be overridden.
#
# GPU REQUIRED. A search is hundreds of trainings and is not viable on CPU, so the
# run aborts if no CUDA device is visible rather than starting a job that appears
# to be alive for days without finishing.
#
# ---------------------------------------------------------------------------
# Why folds run SEQUENTIALLY and not in parallel (unlike the former run_parallel.sh)
# ---------------------------------------------------------------------------
# That script was written for a 48-core CPU node and started 5 processes at
# once. Do not use it on a single GPU:
#   * VRAM   — each process holds its own model + padded activations. A padded
#              batch is 1536 x L_max x batch floats (~1.5 GB at batch 64,
#              L_max 3971); five of those at once will OOM most cards.
#   * RAM    — the embedding cache (src/utils/dataset.py:_EMBEDDING_CACHE) is
#              per-process and holds ~10 GB for 4/5 partitions. Five processes
#              is ~50 GB of duplicated host RAM.
#   * Speed  — the folds would time-slice one GPU, so wall-clock is no better
#              than running them one after another.
# On a GPU each fold is fast enough that sequential is the right default. If the
# node has N GPUs, launch N single-fold jobs with CUDA_VISIBLE_DEVICES instead.

set -euo pipefail

FOLD=""
OUT_DIR="results/esm3_prop_optuna"
EMB_DIR="/data/apostolos/embeddings/esm3"
PASSTHRU=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --fold)            FOLD="$2";    shift 2 ;;
        --out_dir|-od)     OUT_DIR="$2"; shift 2 ;;
        --embeddings_dir)  EMB_DIR="$2"; shift 2 ;;
        *)                 PASSTHRU+=("$1"); shift ;;
    esac
done

mkdir -p "$OUT_DIR" logs

# Validate the setup before committing hours to it. Catches missing or wrong
# embeddings, a missing dependency, and a CPU-only environment — each of which
# otherwise fails much later with a misleading error.
if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
    echo "=== Preflight ==="
    if ! python preflight.py --embeddings_dir "$EMB_DIR" --embedding_dim 1536 \
                             --out_dir "$OUT_DIR" --n_trials 30; then
        echo ""
        echo "Aborting: fix the problems above, or set SKIP_PREFLIGHT=1 to override."
        exit 1
    fi
    echo ""
fi

# Defaults chosen for a GPU budget. --optuna_epochs 35 covers the phase
# transition (~22-30 epochs) while saving ~30% versus the full 50; retraining
# always uses the full --epochs regardless.
DEFAULTS=(
    --embeddings_dir   "$EMB_DIR"
    --data_file        data/labeled_sequences.csv
    --partitioning_file data/graphpart_assignments.csv
    --embedding_dim    1536
    --model            lstmcnncrf
    --out_dir          "$OUT_DIR"
    --epochs           50
    --patience         10
    --optuna_epochs    35
    --n_trials         30
    --num_workers      0     # keeps the embedding cache in the main process
    --seed             42
)

echo "=== ESM3 propeptide Optuna search (GPU required) ==="
python - <<'PY' || exit 1
import sys, torch
if not torch.cuda.is_available():
    print(f'  ERROR: no CUDA GPU visible (torch {torch.__version__}, '
          f'cuda={torch.version.cuda}).')
    print('  This branch is GPU-only. Run on a GPU node; check nvidia-smi, and')
    print('  whether this env has a CPU-only torch build (cuda=None).')
    sys.exit(1)
print(f'  GPU        : {torch.cuda.get_device_name(0)} '
      f'({torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB)')
print(f'  torch/cuda : {torch.__version__} / {torch.version.cuda}')
PY
echo "  out_dir    : ${OUT_DIR}"
echo "  embeddings : ${EMB_DIR}"
echo ""

run_fold () {
    local fold=$1
    echo "--- outer fold ${fold} -> logs/optuna_gpu_fold${fold}.log ---"
    python -m src.train_loop_crf \
        --outer_fold "$fold" \
        "${DEFAULTS[@]}" \
        ${PASSTHRU[@]+"${PASSTHRU[@]}"} \
        2>&1 | tee "logs/optuna_gpu_fold${fold}.log"
}

if [[ -n "$FOLD" ]]; then
    if [[ "$FOLD" == "0" ]]; then
        echo "Protocol: ABLATION — standard CV, partition 0 held out (the paper's"
        echo "          protocol for embedder comparison). 4 models, 1 search."
        echo ""
    fi
    run_fold "$FOLD"
else
    echo "Protocol: FULL NESTED CV — 5 outer folds, 20 models, mean +/- std."
    echo "          For an embedder comparison the paper used --fold 0 instead,"
    echo "          which costs 1/5 as much. See the header of this script."
    echo ""
    for fold in 0 1 2 3 4; do
        run_fold "$fold"
    done
fi

echo ""
echo "=== Search complete ==="
echo "Per-fold winning hyperparameters : ${OUT_DIR}/best_params_outer*.json"
echo "Full reproducible configs        : ${OUT_DIR}/effective_config_outer*.json"
echo "Per-trial search history         : ${OUT_DIR}/optuna_trials_outer*.csv"
echo ""
# The closing message has to match the protocol that actually ran. It used to say
# "aggregate the 5 folds" unconditionally, which on the single-fold ablation path
# reads as though four more folds are still coming -- i.e. as though the run had
# stopped early. It had not; --fold runs exactly one outer fold by design.
if [[ -n "$FOLD" ]]; then
    echo "ABLATION protocol: outer fold ${FOLD} only. This is the complete run --"
    echo "there are no further folds, and nothing was stopped early. It is the"
    echo "paper's own protocol for comparing language models (Supplementary Fig. S8:"
    echo "\"cross-validation with partition 0 as the test set\")."
    echo ""
    echo "Summarise it:"
    echo "  python summarize_optuna.py --out_dir ${OUT_DIR}"
else
    echo "FULL NESTED CV: all 5 outer folds ran."
    echo ""
    echo "Aggregate the 5 folds:"
    echo "  python summarize_optuna.py --out_dir ${OUT_DIR}"
fi
