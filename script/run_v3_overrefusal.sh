#!/bin/bash
# v3 over-refusal on TruthfulQA (STEER=adaptive|triggered, TAG names the output),
# matched to outputs/or_full_truthfulqa_v2.json
# (817 prompts, temperature 0, steps 128, gen_length 128, block 32), then judged
# with the XSTest 3-way refusal rubric on JUDGE_GPU.
#
# exp.py --gpus shards the prompt set across $GPUS itself and merges the parts;
# per-row seeding keeps the result identical to a single-GPU run.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
JUDGE_GPU=${JUDGE_GPU:-1}
N=${N:-817}
STEER=${STEER:-adaptive}          # adaptive | triggered | none
TAG=${TAG:-v3}                    # output name: outputs/TQA-none-${TAG}-42.json
OUT=outputs/TQA-none-${TAG}-42.json

$PY exp.py --attack none --defense ours --remask v3 --steer "$STEER" \
    --source truthfulqa --temperature 0.0 --gen-length 128 --steps 128 \
    --block-length 32 --n "$N" --gpus "$GPUS" --procs-per-gpu "$PROCS_PER_GPU" --out "$OUT"

$PY -m steering.judge_refusal --in "$OUT" --out "${OUT%.json}_judged.json" --gpu "$JUDGE_GPU"
