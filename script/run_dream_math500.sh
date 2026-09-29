#!/bin/bash
# Dream utility on MATH-500 (500 items, --source math500): the deployed defense
# (dream_common.sh) vs no defense, per seed, gen_length $GEN (128, the
# over-refusal setting). Scored by eval_utility.py (last \boxed{} vs reference).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/dream_common.sh"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
DEFS=${DEFS:-"$TAG none"}

for SEED in $SEEDS; do
  for DEF in $DEFS; do
    OUT=$DREAM_OUT/MATH500-none-$DEF-$SEED.json
    if complete_json "$OUT" 500; then echo "skip $OUT (complete)"; else
      if [ "$DEF" = none ]; then EXTRA=(--defense none); else EXTRA=("${DREAM_OURS[@]}"); fi
      say "math500 def=$DEF seed=$SEED ($(date '+%F %T'))"
      "$PY" exp.py "${DREAM_BASE[@]}" "${EXTRA[@]}" --attack none --source math500 --n 500 \
          --seed "$SEED" --out "$OUT" > "log/dream_math500_${DEF}_${SEED}.log" 2>&1 \
          || { echo "FAILED: def=$DEF seed=$SEED"; continue; }
    fi
    "$PY" eval_utility.py --in "$OUT" --out "${OUT%.json}_acc.json" > /dev/null 2>&1 && \
      "$PY" -c 'import json, sys; s = json.load(open(sys.argv[1])); s = s.get("summary", s)
print("ACC", sys.argv[2], sys.argv[3], {k: s.get(k) for k in ("total", "correct", "accuracy", "unparsed")})' \
        "${OUT%.json}_acc.json" "$DEF" "$SEED"
  done
done
echo "ALL MATH500 DONE $(date '+%F %T')"
