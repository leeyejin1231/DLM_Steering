#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# The response fits own GPUs 1 and 4 while they finish generation and grading.
deadline=$((SECONDS + 7200))
until [ -s outputs/response_detector.pt ] &&
      [ -s outputs/llada1.5/response_detector.pt ]; do
    if (( SECONDS >= deadline )); then
        echo "Response detector fits did not finish within two hours" >&2
        exit 1
    fi
    sleep 30
done
sleep 30

.venv/bin/python -u -m steering.build_model_pairs \
    --model llada1.5 --device cuda:1 --guard-device cuda:4 \
    --out data/llada1.5_model_pairs.jsonl
.venv/bin/python -m steering.assemble_model_pairs \
    --in data/llada1.5_model_pairs.jsonl \
    --out data/llada1.5_steer_pairs.json
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u -m steering.fit_vector \
    --model llada1.5 --pairs data/llada1.5_steer_pairs.json
