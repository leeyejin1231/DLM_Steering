#!/bin/bash
# Judge every finished generation JSON with the XSTest 3-way rubric
# (gpt-oss:20b via two ollama containers, one per GPU). Skips already-judged files.
set -u
cd /home/yejin/contents/DLM_Steering
export LD_PRELOAD=/home/yejin/anaconda3/lib/libstdc++.so.6
PY=/home/yejin/anaconda3/bin/python
TAG=v3rp80
for SEED in 42 43 44; do
  for F in outputs/dream/XSTest-safe-none-${TAG}-${SEED}.json outputs/dream/TQA-none-${TAG}-${SEED}.json; do
    OUT=${F%.json}_judged.json
    [ -s "$F" ] || { echo "missing $F"; continue; }
    if [ -s "$OUT" ] && $PY -c "import json,sys; a=json.load(open('$F')); b=json.load(open('$OUT')); sys.exit(0 if len(a['results'])==len(b['results']) else 1)"; then
      echo "skip $OUT"; continue; fi
    echo "=== $(date '+%F %T') judge $F"
    $PY steering/judge_refusal.py --in "$F" --out "$OUT" --gpus 0,1 > "log/judge_$(basename ${F%.json}).log" 2>&1
    echo "=== $(date '+%F %T') judged rc=$? -> $OUT"; grep -A3 '"refusal_rate"' "$OUT" | head -4
  done
done
echo "ALL JUDGING DONE $(date '+%F %T')"
