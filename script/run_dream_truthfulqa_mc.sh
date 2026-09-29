#!/bin/bash
# Dream generalisation on TruthfulQA MC1 (817 items, --source truthfulqa_mc) with
# the deployed defense (dream_common.sh; DEFS="$TAG none" adds the undefended
# reference), per seed, scored by eval_utility.py (letter accuracy).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/dream_common.sh"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
DEFS=${DEFS:-$TAG}

for SEED in $SEEDS; do
  for DEF in $DEFS; do
    OUT=$DREAM_OUT/TQAmc-none-$DEF-$SEED.json
    if complete_json "$OUT" 817; then echo "skip $OUT (complete)"; else
      if [ "$DEF" = none ]; then EXTRA=(--defense none); else EXTRA=("${DREAM_OURS[@]}"); fi
      say "truthfulqa_mc def=$DEF seed=$SEED ($(date '+%F %T'))"
      "$PY" exp.py "${DREAM_BASE[@]}" "${EXTRA[@]}" --attack none --source truthfulqa_mc --n 817 \
          --seed "$SEED" --out "$OUT" > "log/dream_tqamc_${DEF}_${SEED}.log" 2>&1 \
          || { echo "FAILED: def=$DEF seed=$SEED"; continue; }
    fi
    "$PY" eval_utility.py --in "$OUT" --out "${OUT%.json}_acc.json" > /dev/null 2>&1 && \
      "$PY" -c 'import json, sys; s = json.load(open(sys.argv[1])); s = s.get("summary", s)
print("ACC tqamc", sys.argv[2], sys.argv[3], {k: s.get(k) for k in ("total", "correct", "accuracy", "unparsed")})' \
        "${OUT%.json}_acc.json" "$DEF" "$SEED"
  done
done
echo "ALL TQAMC DONE $(date '+%F %T')"
