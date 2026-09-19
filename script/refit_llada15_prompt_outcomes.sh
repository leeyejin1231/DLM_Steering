#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

deadline=$((SECONDS + 21600))
until [ -s outputs/llada1.5/steer_vector.pt ]; do
    if (( SECONDS >= deadline )); then
        echo "LLaDA 1.5 target-generated vector did not finish in six hours" >&2
        exit 1
    fi
    sleep 30
done
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m steering.fit_prompt_detector_outcomes \
    --model llada1.5 --device cuda:0
