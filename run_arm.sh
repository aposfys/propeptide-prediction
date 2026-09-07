#!/usr/bin/env bash
# Run this branch's arm: N replicates at 101 CRF states, propeptide-only labels.
#
# The hyperparameters are REPLAYED from an existing run's config.json rather than
# retyped. The published T4 group used epochs 100 and patience 15, not the 50 and
# 0 in the README, and summarize_results groups on the whole config -- so one
# retyped field puts an arm in its own group and the comparison is lost.
#
# Only three things change per arm: --embeddings_dir, --embedding_dim, --out_dir.
# Everything else is identical across all three branches by construction.
#
#   ARM=esm2 EMB=/path/to/esm2 DIM=1280 bash run_arm.sh 8
#   DRY_RUN=1 ... bash run_arm.sh 8      # print, run nothing
set -euo pipefail

N_REPS="${1:-8}"
MAX_LEN="${MAX_LEN:-100}"
REFERENCE="${REFERENCE:-results/esm2_rep1/config.json}"
: "${ARM:?set ARM, e.g. ARM=esm3_struct}"
: "${EMB:?set EMB to the embeddings directory}"
: "${DIM:?set DIM, e.g. DIM=1536}"

[ -f "$REFERENCE" ] || { echo "No reference config at $REFERENCE" >&2; exit 1; }
[ -d "$EMB" ] || { echo "FAIL: embeddings dir '$EMB' does not exist." >&2; exit 1; }

ARGS=$(python - "$REFERENCE" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
skip = {'out_dir', 'seed', 'outer_fold', 'max_peptide_len', 'min_peptide_len',
        'embeddings_dir', 'embedding_dim'}
parts = []
for key, value in cfg.items():
    if key in skip or value is None:
        continue
    if isinstance(value, bool):
        if value:
            parts.append(f'--{key}')
        continue
    parts.append(f'--{key} {value}')
print(' '.join(parts))
PY
)

echo "arm        : $ARM"
echo "embeddings : $EMB  ($DIM dims)"
echo "grammar    : 5..$MAX_LEN  ($((MAX_LEN + 1)) states), labels: none/propeptide"
echo "replicates : $N_REPS"
echo

for i in $(seq 1 "$N_REPS"); do
  OUT="results/${ARM}_g$((MAX_LEN + 1))_rep${i}"
  if [ -d "$OUT" ]; then echo "skip $OUT (exists)"; continue; fi
  CMD="python run.py $ARGS --embeddings_dir $EMB --embedding_dim $DIM \
--max_peptide_len $MAX_LEN --out_dir $OUT"
  echo "[$i/$N_REPS] $CMD"
  [ "${DRY_RUN:-0}" = "1" ] && continue
  $CMD
done

echo
echo "Compare with:  python -m src.utils.summarize_results results/"
