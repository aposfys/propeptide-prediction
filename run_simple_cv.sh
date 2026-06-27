#!/usr/bin/env bash
# Run all 5 folds of simple CV — adapts automatically to the hardware it finds:
#   - >=1 GPU : folds run pinned one-per-GPU (CUDA_VISIBLE_DEVICES), in batches of N_GPUS.
#   - 0 GPU   : folds run in parallel on CPU, THREADS each.
#
# This is SIMPLE 5-fold CV (3 train / 1 val / 1 test per fold), matching the original
# DeepPeptide per-model training recipe (no nested CV, no Optuna). One trained model per fold.
#
# Usage:
#   bash run_simple_cv.sh \
#       --embeddings_dir ~/embeddings/esm2 \
#       --data_file data/labeled_sequences.csv \
#       --partitioning_file data/graphpart_assignments.csv \
#       --embedding_dim 1280 \
#       --label_type multistate_with_propeptides \
#       --out_dir results/main_esm2 \
#       --epochs 30
#
# CPU thread budget (only used when no GPU is found): THREADS env var (default 8).
# On a SLURM cluster, prefer the array template instead:  sbatch --array=0-4 train_fold.sbatch
#
# Per-fold logs: logs/fold_0.log … logs/fold_4.log

set -euo pipefail

THREADS=${THREADS:-8}
mkdir -p logs

# ── detect GPUs ────────────────────────────────────────────────────────────────
if command -v nvidia-smi >/dev/null 2>&1; then
    N_GPUS=$(( $( (nvidia-smi -L 2>/dev/null || true) | wc -l) ))
else
    N_GPUS=0
fi
echo "Detected ${N_GPUS} GPU(s)."

FAILED=()

if [ "$N_GPUS" -ge 1 ]; then
    # GPU mode: launch in batches of N_GPUS, each fold pinned to a distinct GPU.
    fold=0
    while [ "$fold" -le 4 ]; do
        PIDS=(); FOLDS=()
        for ((g=0; g<N_GPUS && fold<=4; g++, fold++)); do
            CUDA_VISIBLE_DEVICES="$g" python run_fold.py \
                --test_fold "$fold" --num_workers 0 \
                "$@" > "logs/fold_${fold}.log" 2>&1 &
            PIDS+=($!); FOLDS+=("$fold")
            echo "Launched fold ${fold} on GPU ${g} (PID $!)"
        done
        for i in "${!PIDS[@]}"; do
            wait "${PIDS[$i]}" || FAILED+=("fold ${FOLDS[$i]}")
        done
    done
else
    # CPU mode: all 5 folds in parallel, THREADS each.
    export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
    PIDS=(); FOLDS=()
    for fold in 0 1 2 3 4; do
        python run_fold.py \
            --test_fold "$fold" --num_cpu_threads "$THREADS" --num_workers 0 \
            "$@" > "logs/fold_${fold}.log" 2>&1 &
        PIDS+=($!); FOLDS+=("$fold")
        echo "Launched fold ${fold} (PID $!)"
    done
    for i in "${!PIDS[@]}"; do
        wait "${PIDS[$i]}" || FAILED+=("fold ${FOLDS[$i]}")
    done
fi

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "ERROR: failed folds: ${FAILED[*]}  (see logs/fold_*.log)"
    exit 1
fi

echo ""
echo "All 5 folds complete. Aggregate test metrics:"
python -c "
import json, glob, numpy as np, sys
args = sys.argv[1:]
out_dir = 'train_run_cv'
for i, a in enumerate(args):
    if a in ('--out_dir', '-od') and i+1 < len(args):
        out_dir = args[i+1]
files = sorted(glob.glob(f'{out_dir}/summary_fold*.json'))
if not files:
    print(f'  No summary files found in {out_dir}/'); sys.exit(0)
pep, pro, all_ = [], [], []
for f in files:
    t = json.load(open(f))['test']
    pep.append(t.get('f1 peptides', 0)); pro.append(t.get('f1 propeptides', 0)); all_.append(t.get('f1 all', 0))
print(f'  peptides F1   : {np.mean(pep):.3f} ± {np.std(pep):.3f}')
print(f'  propeptides F1: {np.mean(pro):.3f} ± {np.std(pro):.3f}')
print(f'  all F1        : {np.mean(all_):.3f} ± {np.std(all_):.3f}')
" -- "$@"
