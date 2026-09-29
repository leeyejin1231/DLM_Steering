#!/bin/bash
# TruthfulQA truthful/informative judge (gpt-oss:20b over ollama) on the Dream
# over-refusal generations: $DREAM_OUT/TQA-none-$TAG-<seed>.json -> *_tqa.json.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/dream_common.sh"

for SEED in $SEEDS; do
  F=$DREAM_OUT/TQA-none-$TAG-$SEED.json; OUT=${F%.json}_tqa.json
  [ -s "$F" ] || { echo "missing $F"; continue; }
  [ -s "$OUT" ] && { echo "skip $OUT"; continue; }
  say "judge $F ($(date '+%F %T'))"
  "$PY" -m dlm_steering.fitting.judge_truthfulqa --in "$F" --out "$OUT" --gpus "$GPUS" \
      > "log/tqajudge_${TAG}_${SEED}.log" 2>&1 || echo "JUDGE FAILED: $F"
  grep -E '"(truthful|informative|truthful_informative|n_parse_error)"' "$OUT" 2>/dev/null | head -4
done
podman stop $(for g in ${GPUS//,/ }; do echo -n "ollama-$((50001 + g)) "; done) >/dev/null 2>&1 || true   # containers may already be down; common.sh sets -e
echo "ALL TQA JUDGING DONE $(date '+%F %T')"
