#!/usr/bin/env bash
# Run all 5 folds of simple CV in parallel — one process per fold.
# Matches the original DeepPeptide paper training protocol (no nested CV, no Optuna).
#
# Usage:
#   THREADS=8 bash run_simple_cv.sh \
#       --embeddings_dir embeddings_esm2/ \
#       --data_file data/labeled_sequences.csv \
#       --partitioning_file data/graphpart_assignments.csv \
#       --embedding_dim 1280 \
#       --label_type multistate_with_propeptides \
#       --out_dir train_run_cv \
#       --epochs 50 --patience 10
#
# Thread budget (THREADS env var):
#   THREADS = floor((total_cores - 2) / 5)
#   e.g.  40 cores → THREADS=7   (5×7=35, 5 left for OS)
#         47 cores → THREADS=9   (5×9=45, 2 left for OS)
#
# Logs go to logs/fold_0.log … logs/fold_4.log

set -euo pipefail

THREADS=${THREADS:-8}

export OMP_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS

mkdir -p logs

PIDS=()
for fold in 0 1 2 3 4; do
    python run_fold.py \
        --test_fold "$fold" \
        --num_cpu_threads "$THREADS" \
        --num_workers 0 \
        "$@" \
        > "logs/fold_${fold}.log" 2>&1 &
    PIDS+=($!)
    echo "Launched fold ${fold} (PID ${PIDS[-1]})"
done

FAILED=()
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        FAILED+=("fold $i (PID ${PIDS[$i]})")
    fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "ERROR: failed folds: ${FAILED[*]}"
    echo "Check logs/fold_*.log"
    exit 1
fi

echo ""
echo "All 5 folds complete."
echo ""
echo "Aggregate results:"
python -c "
import json, glob, numpy as np, sys
# Find --out_dir from args passed to this script
args = sys.argv[1:]
out_dir = 'train_run_cv'
for i, a in enumerate(args):
    if a in ('--out_dir', '-od') and i+1 < len(args):
        out_dir = args[i+1]
files = sorted(glob.glob(f'{out_dir}/summary_fold*.json'))
if not files:
    print(f'No summary files found in {out_dir}/')
    sys.exit(0)
pep, pro, all_ = [], [], []
for f in files:
    t = json.load(open(f))['test']
    pep.append(t.get('f1 peptides', 0))
    pro.append(t.get('f1 propeptides', 0))
    all_.append(t.get('f1 all', 0))
print(f'  peptides F1 : {np.mean(pep):.3f} ± {np.std(pep):.3f}')
print(f'  propeptides F1: {np.mean(pro):.3f} ± {np.std(pro):.3f}')
print(f'  all F1      : {np.mean(all_):.3f} ± {np.std(all_):.3f}')
" -- "$@"
