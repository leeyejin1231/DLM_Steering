#!/bin/bash
# v3 over-refusal on TruthfulQA, matched to outputs/or_full_truthfulqa_v2.json
# (817 prompts, temperature 0, steps 128, gen_length 128, block 32). The prompt
# set is split across two GPUs, merged, then judged with the XSTest 3-way
# refusal rubric on JUDGE_GPU.
set -e
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
GPUS=(${GPUS:-1 0})
JUDGE_GPU=${JUDGE_GPU:-1}
N=${N:-817}
OUT=outputs/TQA-none-v3-42.json
COMMON=(--attack none --defense ours --remask v3 --source truthfulqa
        --temperature 0.0 --gen-length 128 --steps 128 --block-length 32)

k=${#GPUS[@]}
per=$(( (N + k - 1) / k ))
parts=()
for i in "${!GPUS[@]}"; do
    start=$(( i * per ))
    n=$(( N - start < per ? N - start : per ))
    part="${OUT%.json}.part$i.json"
    parts+=("$part")
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} python exp.py "${COMMON[@]}" \
        --start "$start" --n "$n" --out "$part" > "log/v3_tqa_or_gpu${GPUS[$i]}.log" 2>&1 &
done
wait

python - "$OUT" "${parts[@]}" <<'PY'
import json, sys
out, parts = sys.argv[1], sys.argv[2:]
payloads = [json.load(open(p)) for p in parts]
merged = {**payloads[0], "results": sorted(
    (r for p in payloads for r in p["results"]), key=lambda r: r["index"])}
json.dump(merged, open(out, "w"), ensure_ascii=False, indent=2)
print(f"merged {len(merged['results'])} results from {len(parts)} parts -> {out}")
PY

python steering/judge_refusal.py --in "$OUT" --out "${OUT%.json}_judged.json" --gpu "$JUDGE_GPU"
