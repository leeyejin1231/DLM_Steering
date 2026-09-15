#!/usr/bin/env bash
# DIJA attack (interleaved mask template), gen_length 0, JBB-harmful 100, greedy,
# run in the fresh `dlm_remask` conda env (python 3.10.12, torch 2.3.0+cu121,
# transformers 4.57.1). off / steer on GPU 0, repair on GPU 1.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PY=/home/yejin/anaconda3/envs/dlm_remask/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1
export HF_HOME=/mnt/shared/huggingface-cache/hub HUGGINGFACE_HUB_CACHE=/mnt/shared/huggingface-cache/hub HF_HUB_OFFLINE=1
COMMON=(--source jbb_harmful --attack dija --gen-length 0 --steps 64 --temperature 0.0 --n 100)
P=outputs/newenv_dija0_jbb
mkdir -p outputs log
(
  CUDA_VISIBLE_DEVICES=0 $PY llada_steering_remasking_v2.py "${COMMON[@]}" --mode off    --out ${P}_off.json   > log/newenv_dija0_jbb_off.log 2>&1
  CUDA_VISIBLE_DEVICES=0 $PY llada_steering_remasking_v2.py "${COMMON[@]}" --mode steer  --out ${P}_steer.json > log/newenv_dija0_jbb_steer.log 2>&1
) &
A=$!
CUDA_VISIBLE_DEVICES=1 $PY llada_steering_remasking_v2.py "${COMMON[@]}" --mode repair --out ${P}_repair.json > log/newenv_dija0_jbb_repair.log 2>&1 &
B=$!
fail=0; wait $A || fail=1; wait $B || fail=1
echo "GENERATION_EXIT=$fail" > log/newenv_dija0_jbb_generation.done
