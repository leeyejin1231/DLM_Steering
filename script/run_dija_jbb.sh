#!/bin/bash
# DIJA (paper refined prompts) on JBB under one defense, then Llama Guard 4 and
# StrongREJECT (gpt-oss via ollama) grading. DEFENSE_ARGS picks the defense,
# TAG names the outputs: outputs/JBB-dija-${TAG}-42{,_lg4,_sr}.json
set -e
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
GPU=${GPU:-1}
TAG=${TAG:-v2}
DEFENSE_ARGS=${DEFENSE_ARGS:---defense ours --remask v2 --steer adaptive}
OUT=outputs/JBB-dija-${TAG}-42.json

CUDA_VISIBLE_DEVICES=$GPU python exp.py --attack dija $DEFENSE_ARGS \
    --source jbb_harmful --n 100 --out "$OUT"
CUDA_VISIBLE_DEVICES=$GPU python eval_llamaguard.py --in "$OUT" --out "${OUT%.json}_lg4.json"
python run_sr_eval.py --in "$OUT" --out "${OUT%.json}_sr.json" --port 50001 --gpu "$GPU"
