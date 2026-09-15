#!/usr/bin/env bash
# Score outputs/newenv_dija0_jbb_{off,steer,repair}.json with Llama-Guard-4-12B (GPU 0)
# and StrongREJECT via ollama gpt-oss:20b on :50001, concurrently. Runs in dlm_remask env.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PY=/home/yejin/anaconda3/envs/dlm_remask/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1
export HF_HOME=/mnt/shared/huggingface-cache/hub HUGGINGFACE_HUB_CACHE=/mnt/shared/huggingface-cache/hub HF_HUB_OFFLINE=1
P=outputs/newenv_dija0_jbb
curl -s --max-time 5 http://localhost:50001/api/tags >/dev/null || { echo "ollama not reachable" >&2; exit 1; }
(
  for c in off steer repair; do
    CUDA_VISIBLE_DEVICES=0 $PY eval_llamaguard.py --in ${P}_$c.json --out ${P}_${c}_lg4.json > log/newenv_dija0_jbb_${c}_lg4.log 2>&1 || echo "LG4 FAILED: $c" >> log/newenv_dija0_jbb_eval.done
  done
) &
A=$!
(
  for c in off steer repair; do
    $PY run_sr_eval.py --in ${P}_$c.json --out ${P}_${c}_sr.json > log/newenv_dija0_jbb_${c}_sr.log 2>&1 || echo "SR FAILED: $c" >> log/newenv_dija0_jbb_eval.done
  done
) &
B=$!
wait $A; wait $B
echo "EVAL_FINISHED" >> log/newenv_dija0_jbb_eval.done
