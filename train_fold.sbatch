#!/usr/bin/env bash
#SBATCH --job-name=deeppeptide_cv
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/fold_%a.log
#
# SLURM array job: runs the 5 CV folds as 5 independent tasks, ONE GPU each.
# This is the recommended way to run on a shared SLURM cluster (run_simple_cv.sh is
# for a single interactive node instead).
#
# HOW TO USE:
#   1. Create the conda env once:        bash hpc_setup.sh
#   2. Edit the per-run variables below for the branch/embedder you're training.
#   3. Submit:                           sbatch train_fold.sbatch
#      Array task k trains fold k on its own GPU. Logs -> logs/fold_<k>.log
#
# Tune the #SBATCH lines (partition, account, --time, --mem) to your cluster's policy.

set -euo pipefail

# ── EDIT THESE per run ──────────────────────────────────────────────────────────
CONDA_ENV=deeppeptide
EMB_DIR="$HOME/embeddings/esm2"        # esm2 | esm3 | prost5
EMBEDDING_DIM=1280                     # 1280 (ESM-2) | 1536 (ESM3) | 1024 (ProstT5)
DATA_FILE=data/labeled_sequences.csv
PART_FILE=data/graphpart_assignments.csv
OUT_DIR=results/main_esm2
EPOCHS=30
# ────────────────────────────────────────────────────────────────────────────────

mkdir -p logs "$OUT_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

python run_fold.py \
    --test_fold "${SLURM_ARRAY_TASK_ID}" \
    --embeddings_dir "$EMB_DIR" \
    --data_file "$DATA_FILE" \
    --partitioning_file "$PART_FILE" \
    --embedding_dim "$EMBEDDING_DIM" \
    --label_type multistate_with_propeptides \
    --epochs "$EPOCHS" \
    --out_dir "$OUT_DIR"
