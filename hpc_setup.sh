#!/usr/bin/env bash
# HPC setup: sync all branches from GitHub + generate missing embeddings.
#
# Run once on the HPC login node (or a compute node with internet access).
# Idempotent — safe to re-run; skips embeddings that already exist.
#
# Usage:
#   bash hpc_setup.sh [--repo-dir DIR] [--emb-dir DIR] [--fasta FILE] [--skip-esm2] [--skip-esm3] [--skip-prost5]
#
# Defaults:
#   --repo-dir  ~/DeepPeptide_esm3
#   --emb-dir   ~/embeddings          (shared across branches)
#   --fasta     <repo>/data/protein_sequences.fasta
#
# After this script completes, run training with run_simple_cv.sh on each branch.

set -euo pipefail

# ── defaults ──────────────────────────────────────────────────────────────────
REPO_URL="https://github.com/AfysFys/DeepPeptide_esm3.git"   # update if needed
REPO_DIR="${HOME}/DeepPeptide_esm3"
EMB_DIR="${HOME}/embeddings"
FASTA=""          # resolved from repo after clone/pull
SKIP_ESM2=false
SKIP_ESM3=false
SKIP_PROST5=false
CONDA_ENV="deeppeptide"

# ── argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --repo-dir)   REPO_DIR="$2";  shift 2 ;;
        --emb-dir)    EMB_DIR="$2";   shift 2 ;;
        --fasta)      FASTA="$2";     shift 2 ;;
        --conda-env)  CONDA_ENV="$2"; shift 2 ;;
        --skip-esm2)  SKIP_ESM2=true; shift ;;
        --skip-esm3)  SKIP_ESM3=true; shift ;;
        --skip-prost5) SKIP_PROST5=true; shift ;;
        --repo-url)   REPO_URL="$2";  shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

EMB_ESM2="${EMB_DIR}/esm2"
EMB_ESM3="${EMB_DIR}/esm3"
EMB_PROST5="${EMB_DIR}/prost5"

# ── helpers ───────────────────────────────────────────────────────────────────
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }

count_files() { ls "$1" 2>/dev/null | wc -l; }

run_in_env() {
    conda run -n "$CONDA_ENV" --no-capture-output "$@"
}

# ── 1. clone or pull repo ─────────────────────────────────────────────────────
log "=== Step 1: Sync repo ==="
if [ -d "${REPO_DIR}/.git" ]; then
    log "Repo exists at ${REPO_DIR} — pulling all branches..."
    cd "$REPO_DIR"
    git fetch --all
    # Update each local branch to match origin
    for branch in main deeppeptide-esm3 esm2-propeptide-only esm3-propeptide-only \
                  deeppeptide-prost5 prost5-propeptide-only; do
        if git show-ref --verify --quiet "refs/remotes/origin/${branch}"; then
            git checkout "$branch" --quiet 2>/dev/null || git checkout -b "$branch" "origin/${branch}" --quiet
            git pull origin "$branch" --ff-only --quiet
            log "  ${branch}: up to date ($(git log -1 --format='%h %s'))"
        fi
    done
    git checkout main --quiet
else
    log "Cloning ${REPO_URL} into ${REPO_DIR}..."
    git clone "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
    # Create local tracking branches
    for branch in deeppeptide-esm3 esm2-propeptide-only esm3-propeptide-only \
                  deeppeptide-prost5 prost5-propeptide-only; do
        git checkout -b "$branch" "origin/${branch}" --quiet 2>/dev/null || true
    done
    git checkout main --quiet
    log "Clone complete."
fi

[ -z "$FASTA" ] && FASTA="${REPO_DIR}/data/protein_sequences.fasta"
[ -f "$FASTA" ] || die "FASTA not found: ${FASTA}"
TOTAL_SEQS=$(grep -c "^>" "$FASTA")
log "FASTA: ${FASTA} (${TOTAL_SEQS} sequences)"

# ── 2. conda environment ──────────────────────────────────────────────────────
log "=== Step 2: Conda environment ==="
if conda env list | grep -q "^${CONDA_ENV} "; then
    log "Environment '${CONDA_ENV}' already exists."
else
    log "Creating conda environment '${CONDA_ENV}'..."
    conda create -n "$CONDA_ENV" python=3.10 -y
    run_in_env pip install -r "${REPO_DIR}/requirements.txt"
    log "Environment ready."
fi

# ── 3. embedding directories ──────────────────────────────────────────────────
log "=== Step 3: Embedding status ==="
mkdir -p "$EMB_ESM2" "$EMB_ESM3" "$EMB_PROST5"

N_ESM2=$(count_files "$EMB_ESM2")
N_ESM3=$(count_files "$EMB_ESM3")
N_PROST5=$(count_files "$EMB_PROST5")

log "  ESM-2   : ${N_ESM2} / ${TOTAL_SEQS} files  (${EMB_ESM2})"
log "  ESM-3   : ${N_ESM3} / ${TOTAL_SEQS} files  (${EMB_ESM3})"
log "  ProstT5 : ${N_PROST5} / ${TOTAL_SEQS} files  (${EMB_PROST5})"

# ── 4. generate missing embeddings ───────────────────────────────────────────
log "=== Step 4: Generate embeddings (skips existing files) ==="

# --- ESM-2 ---
if [ "$SKIP_ESM2" = false ] && [ "$N_ESM2" -lt "$TOTAL_SEQS" ]; then
    log "Generating ESM-2 embeddings (${N_ESM2} done, need ${TOTAL_SEQS})..."
    git checkout main --quiet
    run_in_env python src/utils/make_embeddings.py "$FASTA" "$EMB_ESM2"
    log "ESM-2 embeddings complete: $(count_files "$EMB_ESM2") files."
else
    log "ESM-2: skipped (${N_ESM2}/${TOTAL_SEQS} already done)."
fi

# --- ESM-3 ---
if [ "$SKIP_ESM3" = false ] && [ "$N_ESM3" -lt "$TOTAL_SEQS" ]; then
    log "Generating ESM-3 embeddings (${N_ESM3} done, need ${TOTAL_SEQS})..."
    git checkout deeppeptide-esm3 --quiet
    run_in_env python src/utils/make_embeddings.py "$FASTA" "$EMB_ESM3"
    log "ESM-3 embeddings complete: $(count_files "$EMB_ESM3") files."
else
    log "ESM-3: skipped (${N_ESM3}/${TOTAL_SEQS} already done)."
fi

# --- ProstT5 ---
if [ "$SKIP_PROST5" = false ] && [ "$N_PROST5" -lt "$TOTAL_SEQS" ]; then
    log "Generating ProstT5 embeddings (${N_PROST5} done, need ${TOTAL_SEQS})..."
    git checkout deeppeptide-prost5 --quiet
    run_in_env python src/utils/make_embeddings.py "$FASTA" "$EMB_PROST5"
    log "ProstT5 embeddings complete: $(count_files "$EMB_PROST5") files."
else
    log "ProstT5: skipped (${N_PROST5}/${TOTAL_SEQS} already done)."
fi

git checkout main --quiet

# ── 5. summary ────────────────────────────────────────────────────────────────
log ""
log "=== Setup complete ==="
log ""
log "Embedding directories:"
log "  ESM-2   → ${EMB_ESM2}   ($(count_files "$EMB_ESM2") files)"
log "  ESM-3   → ${EMB_ESM3}   ($(count_files "$EMB_ESM3") files)"
log "  ProstT5 → ${EMB_PROST5}  ($(count_files "$EMB_PROST5") files)"
log ""
log "To train — example (main branch, ESM-2, 40 cores):"
log ""
log "  cd ${REPO_DIR}"
log "  git checkout main"
log "  conda activate ${CONDA_ENV}"
log "  THREADS=8 bash run_simple_cv.sh \\"
log "      --embeddings_dir ${EMB_ESM2} \\"
log "      --data_file data/labeled_sequences.csv \\"
log "      --partitioning_file data/graphpart_assignments.csv \\"
log "      --embedding_dim 1280 \\"
log "      --label_type multistate_with_propeptides \\"
log "      --out_dir results/main_esm2 \\"
log "      --epochs 50 --patience 10"
log ""
log "For other branches, swap --embeddings_dir and --embedding_dim accordingly:"
log "  esm2-propeptide-only  → ${EMB_ESM2}, dim=1280"
log "  deeppeptide-esm3      → ${EMB_ESM3}, dim=1536"
log "  esm3-propeptide-only  → ${EMB_ESM3}, dim=1536"
log "  deeppeptide-prost5    → ${EMB_PROST5}, dim=1024"
log "  prost5-propeptide-only → ${EMB_PROST5}, dim=1024"
