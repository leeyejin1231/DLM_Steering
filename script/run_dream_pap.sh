#!/bin/bash
# Dream under the cache-based PAP attack (one PAP_Better paraphrase per row from
# data/attacks/pap_better/<source>/seed<seed>.json) with the deployed defense
# (dream_common.sh). gen 128, temperature 0.2. Default: JBB only; set
# SOURCES="jbb_harmful harmbench strongreject" for the full set. Grade with
# script/eval_dream_pap.sh afterwards.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/dream_common.sh"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
SOURCES=${SOURCES:-jbb_harmful}
TEMPERATURE=${TEMPERATURE:-0.2}

for SRC in $SOURCES; do
  case $SRC in
    jbb_harmful) N=100; P=JBB ;; harmbench) N=393; P=HB ;; strongreject) N=313; P=SR ;;
    *) echo "unknown source $SRC"; exit 1 ;;
  esac
  for SEED in $SEEDS; do
    OUT=$DREAM_OUT/$P-pap-$TAG-$SEED.json
    if complete_json "$OUT" "$N"; then echo "skip $OUT (complete)"; continue; fi
    say "$SRC pap seed=$SEED -> $OUT ($(date '+%F %T'))"
    "$PY" exp.py "${DREAM_BASE[@]}" "${DREAM_OURS[@]}" --attack pap \
        --source "$SRC" --n "$N" --seed "$SEED" --temperature "$TEMPERATURE" --out "$OUT" \
        > "log/dream_pap_${P}_${TAG}_${SEED}.log" 2>&1 \
        || echo "FAILED: $SRC seed=$SEED (log/dream_pap_${P}_${TAG}_${SEED}.log)"
  done
done
echo "ALL PAP GENERATION DONE $(date '+%F %T')"
