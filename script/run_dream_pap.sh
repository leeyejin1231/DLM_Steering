#!/bin/bash
# Dream under the cache-based PAP attack (code-share-dream recipe: one PAP_Better
# paraphrase per row, Qwen3-14B caches in data/attacks/pap_better/) with the
# current defense (v3 remask + adaptive steering L20, prompt remask 80%,
# response_detector3_committed_L20). seeds 42/43/44; JBB (100), HarmBench (393),
# StrongREJECT (313); gen 128, temperature 0.2 as in that branch's README.
set -u
cd /home/yejin/contents/DLM_Steering
export LD_PRELOAD=/home/yejin/anaconda3/lib/libstdc++.so.6
export HF_HOME=/mnt/shared/huggingface-cache/hub HUGGINGFACE_HUB_CACHE=/mnt/shared/huggingface-cache/hub HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
PY=/home/yejin/anaconda3/bin/python
TAG=v3rp80
ARGS=(--model dream --attack pap --defense ours --remask v3 --steer adaptive --layer 20
      --response-detector outputs/dream/response_detector3_committed_L20.pt
      --remask-prompt --remask-prompt-frac 0.8
      --gen-length 128 --steps 128 --block-length 32 --temperature 0.2 --gpus 0,1)
for SRC in jbb_harmful harmbench strongreject; do
  case $SRC in jbb_harmful) N=100; P=JBB;; harmbench) N=393; P=HB;; strongreject) N=313; P=SR;; esac
  for SEED in 42 43 44; do
    OUT=outputs/dream/${P}-pap-${TAG}-${SEED}.json
    if [ -s "$OUT" ] && $PY -c "import json,sys; sys.exit(0 if len(json.load(open('$OUT'))['results'])==$N else 1)"; then
      echo "skip $OUT"; continue; fi
    echo "=== $(date '+%F %T') start $SRC seed=$SEED -> $OUT"
    $PY exp.py "${ARGS[@]}" --source $SRC --n $N --seed $SEED --out "$OUT" > "log/dream_pap_${P}_${TAG}_${SEED}.log" 2>&1
    echo "=== $(date '+%F %T') done  $SRC seed=$SEED rc=$?"
  done
done
echo "ALL PAP GENERATION DONE $(date '+%F %T')"
