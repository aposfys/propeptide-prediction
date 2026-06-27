#!/usr/bin/env bash
# HPC setup: clone the repo, create a SEPARATE conda env per branch, and generate
# that branch's embeddings.
#
# Why per-branch envs: fair-esm (ESM-2) and the EvolutionaryScale `esm` SDK (ESM3) BOTH
# import as the top-level `esm` package, so they cannot coexist in one environment.
# Each branch therefore gets its own env built from its own requirements.txt.
#
# Run once on an HPC node WITH internet (login node, or a compute node that can reach
# pypi.org and huggingface.co). Idempotent: skips embeddings that already exist.
#
# Usage:
#   bash hpc_setup.sh [--repo-dir DIR] [--emb-dir DIR] [--fasta FILE]
#                     [--skip-esm2] [--skip-esm3] [--with-prost5]
#
# Defaults: --repo-dir ~/DeepPeptide_esm3   --emb-dir ~/embeddings
# After this, train with run_simple_cv.sh (interactive node) or train_fold.sbatch (SLURM).

set -euo pipefail

REPO_URL="https://github.com/aposfys/DeepPeptide-ESM3.git"
REPO_DIR="${HOME}/DeepPeptide_esm3"
EMB_DIR="${HOME}/embeddings"
FASTA=""
SKIP_ESM2=false
SKIP_ESM3=false
SKIP_PROST5=true   # ProstT5 branch is optional / not vetted here; enable with --with-prost5

while [[ $# -gt 0 ]]; do
    case $1 in
        --repo-dir)    REPO_DIR="$2"; shift 2 ;;
        --emb-dir)     EMB_DIR="$2";  shift 2 ;;
        --fasta)       FASTA="$2";    shift 2 ;;
        --repo-url)    REPO_URL="$2"; shift 2 ;;
        --skip-esm2)   SKIP_ESM2=true;  shift ;;
        --skip-esm3)   SKIP_ESM3=true;  shift ;;
        --with-prost5) SKIP_PROST5=false; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }
count() { ls "$1" 2>/dev/null | wc -l | tr -d ' '; }

# ── 1. clone or pull repo ─────────────────────────────────────────────────────
log "=== Step 1: sync repo ==="
if [ -d "${REPO_DIR}/.git" ]; then
    cd "$REPO_DIR"; git fetch --all --quiet
else
    git clone "$REPO_URL" "$REPO_DIR"; cd "$REPO_DIR"
fi
for b in main deeppeptide-esm3 deeppeptide-prost5; do
    if git show-ref --verify --quiet "refs/remotes/origin/${b}"; then
        git checkout "$b" --quiet 2>/dev/null || git checkout -b "$b" "origin/${b}" --quiet
        git pull --ff-only --quiet origin "$b" 2>/dev/null || true
    fi
done
git checkout main --quiet

[ -z "$FASTA" ] && FASTA="${REPO_DIR}/data/protein_sequences.fasta"
[ -f "$FASTA" ] || die "FASTA not found: ${FASTA}"
log "FASTA: ${FASTA} ($(grep -c '^>' "$FASTA") sequences)"

# ── helper: ensure a per-branch env, then generate that branch's embeddings ───
generate() {   # $1=env_name  $2=branch  $3=emb_subdir
    local env="$1" branch="$2" dir="${EMB_DIR}/$3"
    mkdir -p "$dir"
    log "--- ${branch}: env '${env}' -> ${dir} ($(count "$dir") files already present) ---"
    if ! conda env list | grep -q "^${env} "; then
        log "creating conda env '${env}' (python 3.10)"
        conda create -n "$env" python=3.10 -y
    fi
    git checkout "$branch" --quiet
    log "installing ${branch} requirements into '${env}' ..."
    conda run -n "$env" --no-capture-output pip install -q -r "${REPO_DIR}/requirements.txt"
    log "generating embeddings (skips existing files) ..."
    conda run -n "$env" --no-capture-output python src/utils/make_embeddings.py "$FASTA" "$dir"
    log "${branch}: now $(count "$dir") embedding files."
    git checkout main --quiet
}

# ── 2. per-branch envs + embeddings ───────────────────────────────────────────
log "=== Step 2: per-branch envs + embeddings ==="
if [ "$SKIP_ESM2" = false ];   then generate deeppeptide_esm2   main               esm2;   else log "ESM-2 skipped"; fi
if [ "$SKIP_ESM3" = false ];   then generate deeppeptide_esm3   deeppeptide-esm3   esm3;   else log "ESM-3 skipped"; fi
if [ "$SKIP_PROST5" = false ]; then generate deeppeptide_prost5 deeppeptide-prost5 prost5; else log "ProstT5 skipped (use --with-prost5 to enable)"; fi

# ── 3. summary + training hint ────────────────────────────────────────────────
log ""
log "=== Setup complete ==="
log "  ESM-2 -> ${EMB_DIR}/esm2 ($(count "${EMB_DIR}/esm2") files)   [env deeppeptide_esm2, dim 1280]"
log "  ESM-3 -> ${EMB_DIR}/esm3 ($(count "${EMB_DIR}/esm3") files)   [env deeppeptide_esm3, dim 1536]"
log ""
log "Train main (ESM-2), example:"
log "  git checkout main && conda activate deeppeptide_esm2"
log "  bash run_simple_cv.sh \\"
log "      --embeddings_dir ${EMB_DIR}/esm2 \\"
log "      --data_file data/labeled_sequences.csv \\"
log "      --partitioning_file data/graphpart_assignments.csv \\"
log "      --embedding_dim 1280 --label_type multistate_with_propeptides \\"
log "      --out_dir results/main_esm2 --epochs 30"
log ""
log "Train deeppeptide-esm3 (ESM3): checkout deeppeptide-esm3, conda activate deeppeptide_esm3,"
log "  swap --embeddings_dir ${EMB_DIR}/esm3 and --embedding_dim 1536."
