#!/usr/bin/env bash
# Score the gated generations. ~12 min.
#
#   harmful  Llama-Guard-4-12B (GPU) and StrongREJECT (ollama gpt-oss:20b)
#   benign   XSTest three-way rubric via the same ollama server
#
# The two graders run concurrently because only Llama Guard needs the GPU.
# StrongREJECT needs the ollama container from ollama_setting/podman up on :50001.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

require outputs/gated_len128.json outputs/gated_len256.json \
        outputs/gated_or30_xstest.json outputs/gated_or30_jbb.json outputs/gated_or30_tqa.json

curl -s --max-time 5 http://localhost:50001/api/tags >/dev/null \
    || { echo "ollama not reachable on :50001 -- start ollama_setting/podman" >&2; exit 1; }

say "scoring"
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
        $PY run_sr_eval.py --in "outputs/gated_len$L.json" \
            --out "outputs/gated_sr_len$L.json" > "log/gated_sr_$L.log" 2>&1
    done
    for s in xstest jbb tqa; do
        $PY steering/judge_refusal.py --in "outputs/gated_or30_$s.json" \
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
