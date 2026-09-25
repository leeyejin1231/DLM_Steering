#!/bin/bash
# Dream utility on MATH-500 (500 items): ours (v3 remask + adaptive steering L20,
# prompt remask 80%, response_detector3_committed_L20) vs no defense, seeds 42/43/44,
# gen_length 128 (same setting as the over-refusal runs). Scored by eval_utility.py.
set -u
cd /home/yejin/contents/DLM_Steering
export LD_PRELOAD=/home/yejin/anaconda3/lib/libstdc++.so.6
export HF_HOME=/mnt/shared/huggingface-cache/hub HUGGINGFACE_HUB_CACHE=/mnt/shared/huggingface-cache/hub HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
PY=/home/yejin/anaconda3/bin/python
GEN=${GEN:-128}
BASE=(--model dream --attack none --source math500 --n 500 --gen-length $GEN --steps 128 --block-length 32 --gpus 0,1)
OURS=(--defense ours --remask v3 --steer adaptive --layer 20
      --response-detector outputs/dream/response_detector3_committed_L20.pt
      --remask-prompt --remask-prompt-frac 0.8)
for SEED in 42 43 44; do
  for DEF in v3rp80 none; do
    OUT=outputs/dream/MATH500-none-${DEF}-${SEED}.json
    if [ -s "$OUT" ] && $PY -c "import json,sys; sys.exit(0 if len(json.load(open('$OUT'))['results'])==500 else 1)"; then
      echo "skip $OUT"; else
      echo "=== $(date '+%F %T') start math500 def=$DEF seed=$SEED"
      if [ $DEF = none ]; then EXTRA=(--defense none); else EXTRA=("${OURS[@]}"); fi
      $PY exp.py "${BASE[@]}" "${EXTRA[@]}" --seed $SEED --out "$OUT" > "log/dream_math500_${DEF}_${SEED}.log" 2>&1
      echo "=== $(date '+%F %T') done  math500 def=$DEF seed=$SEED rc=$?"
    fi
    $PY eval_utility.py --in "$OUT" --out "${OUT%.json}_acc.json" > /dev/null 2>&1 && \
      $PY -c "import json; s=json.load(open('${OUT%.json}_acc.json')); s=s.get('summary',s); print('ACC $DEF $SEED', {k:s.get(k) for k in ('total','correct','accuracy','unparsed')})"
  done
done
echo "ALL MATH500 DONE $(date '+%F %T')"
