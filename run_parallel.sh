#!/usr/bin/env bash
# Run all 5 outer CV folds in parallel on a multi-core CPU node.
#
# Usage:
#   THREADS=2 bash run_parallel.sh \
#       --embeddings_dir /data/apostolos/esm3_embeddings \
#       --data_file data/labeled_sequences.csv \
#       --partitioning_file data/graphpart_assignments.csv \
#       --out_dir /data/apostolos/train_run \
#       --epochs 50 --patience 10 --n_trials 5 \
#       --optuna_epochs 35
#
# --optuna_epochs 35: gives Optuna enough room to see past the warmup
# (3 epochs) and through the typical phase-transition zone (~22-30 epochs)
# before patience kicks in, while saving ~30% time vs full 50 epochs.
#
# Core budget: taskset -c START-END gives (END - START + 1) cores total.
#   e.g. taskset -c 37-47  →  11 cores  →  THREADS=2  (5×2=10, 1 left for OS)
#        taskset -c 0-44   →  45 cores  →  THREADS=8  (5×8=40, 5 left for OS)
# Rule of thumb: THREADS = floor((total_cores - 1) / 5)
# Set THREADS before calling, or export it in your job script.
#
# IMPORTANT: num_workers=0 is required so embeddings stay cached in the main
# process and are reused across all epochs/trials without extra disk reads.
# NOTE: The first epoch of each fold still loads embeddings from disk (cache
# is empty at startup). Expect the first epoch to be slow; subsequent epochs
# run from RAM and will be much faster.

set -euo pipefail

THREADS=${THREADS:-2}   # safe default for 11-core allocation (taskset -c 37-47)

# Propagate OMP/MKL thread counts so the underlying BLAS respects the limit.
export OMP_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS

mkdir -p logs

PIDS=()
for fold in 0 1 2 3 4; do
    python -m src.train_loop_crf \
        --outer_fold "$fold" \
        --num_cpu_threads "$THREADS" \
        --num_workers 0 \
        "$@" \
        > "logs/outer_fold_${fold}.log" 2>&1 &
    PIDS+=($!)
    echo "Launched outer fold ${fold} (PID ${PIDS[-1]})"
done

# Wait for all folds and report which ones failed.
FAILED=()
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        FAILED+=("fold $i (PID ${PIDS[$i]})")
    fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "ERROR: the following folds failed: ${FAILED[*]}"
    echo "Check logs/outer_fold_*.log for details."
    exit 1
fi

echo ""
echo "All 5 outer folds complete."
echo "Run: python evaluation/measure_performance.py --out_dir <out_dir> --data_file <data_file>"
