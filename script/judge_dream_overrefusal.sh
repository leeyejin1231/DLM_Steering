#!/bin/bash
# Judge the Dream over-refusal generations with the XSTest 3-way rubric
# (gpt-oss:20b, one ollama container per GPU in $GPUS), then aggregate the
# seeds into $DREAM_OUT/OR-summary-$TAG.{json,md}. Skips finished files.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/dream_common.sh"

for SEED in $SEEDS; do
  for F in "$DREAM_OUT/XSTest-safe-none-$TAG-$SEED.json" "$DREAM_OUT/TQA-none-$TAG-$SEED.json"; do
    OUT=${F%.json}_judged.json
    [ -s "$F" ] || { echo "missing $F"; continue; }
    if [ -s "$OUT" ] && "$PY" -c 'import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:])
sys.exit(0 if len(a["results"]) == len(b["results"]) else 1)' "$F" "$OUT"; then
      echo "skip $OUT"; continue
    fi
    say "judge $F ($(date '+%F %T'))"
    "$PY" -m dlm_steering.fitting.judge_refusal --in "$F" --out "$OUT" --gpus "$GPUS" \
        > "log/judge_$(basename "${F%.json}").log" 2>&1 || echo "JUDGE FAILED: $F"
    grep -A3 '"refusal_rate"' "$OUT" 2>/dev/null | head -4
  done
done
podman stop $(for g in ${GPUS//,/ }; do echo -n "ollama-$((50001 + g)) "; done) >/dev/null 2>&1 || true   # containers may already be down; common.sh sets -e

say "summary"
"$PY" -m dlm_steering.fitting.aggregate_dream_overrefusal --out-dir "$DREAM_OUT" --tag "$TAG" --seeds "${SEEDS// /,}"
echo "ALL JUDGING DONE $(date '+%F %T')"
