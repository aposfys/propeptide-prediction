#!/usr/bin/env bash
# Grammar ablation: does widening the CRF state space cost anything on data that
# cannot use the extra states?
#
# Every propeptide in the benchmark is 5-50 residues, so states 51-100 are never
# visited by a label. If F1 holds, the wider grammar is free and the dataset
# extension in PAPER.md can proceed. If it drops, the transition matrix alone
# costs accuracy, and that is worth knowing for 10 runs rather than after a
# month of rebuilding.
#
# The treatment arm REPLAYS AN EXISTING RUN'S config.json and changes exactly one
# field. Retyping the hyperparameters by hand is how two arms silently end up
# differing in three places -- and summarize_results.py groups on the whole
# config, so any stray difference splits the groups and the comparison is lost.
#
# Usage:
#   bash run_grammar_ablation.sh                          # 10 reps at 101 states
#   bash run_grammar_ablation.sh 10 100 results/esm2_rep1/config.json
#   DRY_RUN=1 bash run_grammar_ablation.sh                # print, run nothing
set -euo pipefail

N_REPS="${1:-10}"
MAX_LEN="${2:-100}"
REFERENCE="${3:-results/esm2_rep1/config.json}"
PREFIX="${PREFIX:-esm2_g$((MAX_LEN + 1))_rep}"

[ -f "$REFERENCE" ] || { echo "No reference config at $REFERENCE" >&2; exit 1; }

# Rebuild the argument list from the reference, dropping bookkeeping fields and
# nulls. --seed is deliberately dropped: the single-run path never applies it
# (RESULTS.md, "The replicate groups are unseeded"), so passing it would imply a
# control that does not exist.
ARGS=$(python - "$REFERENCE" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
skip = {'out_dir', 'seed', 'outer_fold', 'max_peptide_len', 'min_peptide_len'}
parts = []
for key, value in cfg.items():
    if key in skip or value is None:
        continue
    if isinstance(value, bool):
        if value:
            parts.append(f'--{key}')          # store_true flags
        continue
    parts.append(f'--{key} {value}')
print(' '.join(parts))
PY
)

echo "reference : $REFERENCE"
echo "grammar   : 5..$MAX_LEN  ($((MAX_LEN + 1)) states)"
echo "replicates: $N_REPS"
echo "args      : $ARGS"
echo

EMB=$(python -c "import json,sys; print(json.load(open('$REFERENCE'))['embeddings_dir'])")
if [ ! -d "$EMB" ]; then
  echo "FAIL: embeddings_dir '$EMB' from the reference config does not exist." >&2
  echo "      Check it before launching; a stale path has wasted a run here twice." >&2
  exit 1
fi
echo "embeddings: $EMB  (exists)"
echo

for i in $(seq 1 "$N_REPS"); do
  OUT="results/${PREFIX}${i}"
  if [ -d "$OUT" ]; then echo "skip $OUT (exists)"; continue; fi
  CMD="python run.py $ARGS --max_peptide_len $MAX_LEN --out_dir $OUT"
  echo "[$i/$N_REPS] $CMD"
  if [ "${DRY_RUN:-0}" = "1" ]; then continue; fi
  $CMD
done

echo
echo "Done. Compare the two grammars with:"
echo "  python -m src.utils.summarize_results results/"
echo "summarize_results groups on the whole config, so the 51- and 101-state runs"
echo "form separate groups automatically -- max_peptide_len is the only field"
echo "that differs."
