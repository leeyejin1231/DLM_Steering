#!/usr/bin/env bash
# Score the gated generations. ~12 min.
#
#   harmful  Llama-Guard-4-12B (GPU) and StrongREJECT (ollama gpt-oss:20b)
#   benign   XSTest three-way rubric via the same ollama server
#
# The two graders run concurrently because only Llama Guard needs the GPU.
# run_sr_eval.py starts the podman ollama container itself when it is down.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

require outputs/gated_len128.json outputs/gated_len256.json \
        outputs/gated_or30_xstest.json outputs/gated_or30_jbb.json outputs/gated_or30_tqa.json

# Llama Guard keeps GPU_A to itself; the ollama containers take every other
# card in $GPUS (one server per card, ports 50001+i, reused across the five
# files below because a container outlives the process that started it).
OLLAMA_GPUS=""
for g in ${GPUS//,/ }; do
    [ "$g" = "$GPU_A" ] || OLLAMA_GPUS="${OLLAMA_GPUS:+$OLLAMA_GPUS,}$g"
done
[ -n "$OLLAMA_GPUS" ] || OLLAMA_GPUS="$GPU_A"

say "scoring (Llama Guard on GPU $GPU_A, ollama on $OLLAMA_GPUS)"
(
    for L in 128 256; do
        CUDA_VISIBLE_DEVICES=$GPU_A $PY eval_llamaguard.py \
            --in "outputs/gated_len$L.json" --out "outputs/gated_lg4_len$L.json" \
            > "log/gated_lg4_$L.log" 2>&1
    done
) &
A=$!
(
    for L in 128 256; do
        $PY run_sr_eval.py --gpus "$OLLAMA_GPUS" \
            --in "outputs/gated_len$L.json" \
            --out "outputs/gated_sr_len$L.json" > "log/gated_sr_$L.log" 2>&1
    done
    for s in xstest jbb tqa; do
        $PY -m steering.judge_refusal --gpus "$OLLAMA_GPUS" \
            --in "outputs/gated_or30_$s.json" \
            --out "outputs/gated_or30_${s}_judged.json" > "log/gated_judge_$s.log" 2>&1
    done
) &
B=$!

fail=0
wait $A || fail=1
wait $B || fail=1
[ $fail -eq 0 ] || { echo "a grader failed; see log/gated_*.log" >&2; exit 1; }

say "results"
$PY script/report.py
