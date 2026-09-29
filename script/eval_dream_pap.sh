#!/bin/bash
# Evaluate the Dream PAP runs: Llama Guard 4 (kotox env, transformers 5.x) and
# StrongREJECT (gpt-oss:20b over ollama), both sharded over GPUs 0,1.
set -u
cd /home/yejin/contents/DLM_Steering
export HF_HOME=/mnt/shared/huggingface-cache/hub HUGGINGFACE_HUB_CACHE=/mnt/shared/huggingface-cache/hub HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=/home/yejin/anaconda3/bin/python
PYLG=/home/yejin/anaconda3/envs/kotox/bin/python
until grep -q "ALL PAP GENERATION DONE" log/dream_pap_driver.log 2>/dev/null; do sleep 30; done
for SEED in 42 43 44; do
  F=outputs/dream/JBB-pap-v3rp80-${SEED}.json
  [ -s "$F" ] || { echo "missing $F"; continue; }
  LG=${F%.json}_lg4.json
  if [ ! -s "$LG" ]; then
    echo "=== $(date '+%F %T') LG4 $F"
    $PYLG eval_llamaguard.py --in "$F" --out "$LG" --gpus 0,1 > "log/lg4_JBB-pap-v3rp80-${SEED}.log" 2>&1
    echo "=== $(date '+%F %T') LG4 rc=$?"; grep -E '"asr"' "$LG" | head -1
  fi
done
for SEED in 42 43 44; do
  F=outputs/dream/JBB-pap-v3rp80-${SEED}.json
  SR=${F%.json}_sr.json
  if [ -s "$F" ] && [ ! -s "$SR" ]; then
    echo "=== $(date '+%F %T') SR $F"
    LD_PRELOAD=/home/yejin/anaconda3/lib/libstdc++.so.6 $PY run_sr_eval.py --in "$F" --out "$SR" --gpus 0,1 > "log/sr_JBB-pap-v3rp80-${SEED}.log" 2>&1
    echo "=== $(date '+%F %T') SR rc=$?"; grep -E '"asr"' "$SR" | head -1
  fi
done
podman stop ollama-50001 ollama-50002 >/dev/null 2>&1
echo "ALL PAP EVAL DONE $(date '+%F %T')"
