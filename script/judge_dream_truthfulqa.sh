#!/bin/bash
# TruthfulQA truthful/informative judge over the Dream over-refusal generations (seeds 42/43/44).
set -u
cd /home/yejin/contents/DLM_Steering
export LD_PRELOAD=/home/yejin/anaconda3/lib/libstdc++.so.6
PY=/home/yejin/anaconda3/bin/python
for SEED in 42 43 44; do
  F=outputs/dream/TQA-none-v3rp80-${SEED}.json; OUT=${F%.json}_tqa.json
  echo "=== $(date '+%F %T') judge $F"
  $PY steering/judge_truthfulqa.py --in "$F" --out "$OUT" --gpus 0,1 > "log/tqajudge_${SEED}.log" 2>&1
  echo "=== $(date '+%F %T') judged rc=$? -> $OUT"; grep -E '"(truthful|informative|truthful_informative|n_parse_error)"' "$OUT" | head -4
done
echo "ALL TQA JUDGING DONE $(date '+%F %T')"
