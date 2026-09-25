#!/bin/bash
# Dream generalisation on TruthfulQA MC1 (817 items, --source truthfulqa_mc): ours
# (v3 remask + adaptive steering L20, prompt remask 80%, response_detector3_committed_L20),
# seeds 42/43/44, gen_length 128, scored by eval_utility.py (letter accuracy).
set -u
cd /home/yejin/contents/DLM_Steering
export LD_PRELOAD=/home/yejin/anaconda3/lib/libstdc++.so.6
export HF_HOME=/mnt/shared/huggingface-cache/hub HUGGINGFACE_HUB_CACHE=/mnt/shared/huggingface-cache/hub HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
PY=/home/yejin/anaconda3/bin/python
# let the running ollama judge finish before taking the GPUs
until grep -q "ALL TQA JUDGING DONE" log/tqajudge_driver.log 2>/dev/null; do sleep 30; done
podman stop ollama-50001 ollama-50002 >/dev/null 2>&1
ARGS=(--model dream --attack none --defense ours --remask v3 --steer adaptive --layer 20
      --response-detector outputs/dream/response_detector3_committed_L20.pt
      --remask-prompt --remask-prompt-frac 0.8
      --source truthfulqa_mc --n 817 --gen-length 128 --steps 128 --block-length 32 --gpus 0,1)
for SEED in 42 43 44; do
  OUT=outputs/dream/TQAmc-none-v3rp80-${SEED}.json
  if [ -s "$OUT" ] && $PY -c "import json,sys; sys.exit(0 if len(json.load(open('$OUT'))['results'])==817 else 1)"; then
    echo "skip $OUT"; else
    echo "=== $(date '+%F %T') start truthfulqa_mc seed=$SEED"
    $PY exp.py "${ARGS[@]}" --seed $SEED --out "$OUT" > "log/dream_tqamc_v3rp80_${SEED}.log" 2>&1
    echo "=== $(date '+%F %T') done  truthfulqa_mc seed=$SEED rc=$?"
  fi
  $PY eval_utility.py --in "$OUT" --out "${OUT%.json}_acc.json" > /dev/null 2>&1 && \
    $PY -c "import json; s=json.load(open('${OUT%.json}_acc.json'))['summary']; print('ACC tqamc $SEED', {k:s.get(k) for k in ('total','correct','accuracy','unparsed')})"
done
echo "ALL TQAMC DONE $(date '+%F %T')"
