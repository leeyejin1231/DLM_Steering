#!/bin/bash
# Dream over-refusal: ours (v3 remask + adaptive steering L20, prompt remask 80%,
# response_detector3_committed_L20 thr 0.387), no attack, seeds 42/43/44,
# XSTest-safe (250) + TruthfulQA (817, full set so it can be reused for the
# generalisation analysis). Both A6000s via --gpus 0,1 sharding.
set -u
cd /home/yejin/contents/DLM_Steering
export LD_PRELOAD=/home/yejin/anaconda3/lib/libstdc++.so.6
export HF_HOME=/mnt/shared/huggingface-cache/hub HUGGINGFACE_HUB_CACHE=/mnt/shared/huggingface-cache/hub HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
PY=/home/yejin/anaconda3/bin/python
TAG=v3rp80
COMMON=(--model dream --attack none --defense ours --remask v3 --steer adaptive --layer 20
        --response-detector outputs/dream/response_detector3_committed_L20.pt
        --remask-prompt --remask-prompt-frac 0.8
        --gen-length 128 --steps 128 --block-length 32 --gpus 0,1)
for SEED in 42 43 44; do
  for SRC in xstest_safe truthfulqa; do
    case $SRC in
      xstest_safe) N=250; OUT=outputs/dream/XSTest-safe-none-${TAG}-${SEED}.json ;;
      truthfulqa)  N=817; OUT=outputs/dream/TQA-none-${TAG}-${SEED}.json ;;
    esac
    if [ -s "$OUT" ] && $PY -c "import json,sys; d=json.load(open('$OUT')); sys.exit(0 if len(d['results'])==$N else 1)"; then
      echo "skip $OUT (complete)"; continue
    fi
    echo "=== $(date '+%F %T') start $SRC seed=$SEED -> $OUT"
    $PY exp.py "${COMMON[@]}" --source $SRC --n $N --seed $SEED --out "$OUT" \
        > "log/dream_or_${SRC}_${TAG}_${SEED}.log" 2>&1
    echo "=== $(date '+%F %T') done  $SRC seed=$SEED rc=$?"
  done
done
echo "ALL GENERATION DONE $(date '+%F %T')"
