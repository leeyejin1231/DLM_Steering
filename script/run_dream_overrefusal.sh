#!/bin/bash
# Dream over-refusal: ours (v3 remask + adaptive steering, see dream_common.sh),
# no attack, on XSTest-safe (250) and TruthfulQA (817, the full set so the same
# generations feed the truthful/informative metric). One run per seed; exp.py
# shards the prompt set over $GPUS. Then:
#   script/judge_dream_overrefusal.sh   XSTest 3-way refusal judge + seed summary
#   script/judge_dream_truthfulqa.sh    TruthfulQA truthful/informative judge
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/dream_common.sh"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
SOURCES=${SOURCES:-"xstest_safe truthfulqa"}

for SEED in $SEEDS; do
  for SRC in $SOURCES; do
    case $SRC in
      xstest_safe) N=250; OUT=$DREAM_OUT/XSTest-safe-none-$TAG-$SEED.json ;;
      truthfulqa)  N=817; OUT=$DREAM_OUT/TQA-none-$TAG-$SEED.json ;;
      *) echo "unknown source $SRC"; exit 1 ;;
    esac
    if complete_json "$OUT" "$N"; then echo "skip $OUT (complete)"; continue; fi
    say "$SRC seed=$SEED -> $OUT ($(date '+%F %T'))"
    "$PY" exp.py "${DREAM_BASE[@]}" "${DREAM_OURS[@]}" --attack none \
        --source "$SRC" --n "$N" --seed "$SEED" --out "$OUT" \
        > "log/dream_or_${SRC}_${TAG}_${SEED}.log" 2>&1 \
        || echo "FAILED: $SRC seed=$SEED (log/dream_or_${SRC}_${TAG}_${SEED}.log)"
  done
done
echo "ALL GENERATION DONE $(date '+%F %T')"
