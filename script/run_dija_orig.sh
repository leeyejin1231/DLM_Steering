#!/usr/bin/env bash
# Published DIJA attack (DIJA/run_jailbreakbench refined JBB prompts) on JBB-harmful 100,
# greedy decoding (temperature 0), four conditions, then Llama-Guard-4 + StrongREJECT.
#
#   off       baseline, no defense
#   steer     llada_steering_v2-style adaptive steering only (no remasking)
#   repair    llada_steering_remasking_v2 (steering + one-shot remasking)
#   proposed  proposed.py via run_proposed_dija.py
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

curl -s --max-time 5 http://localhost:50001/api/tags >/dev/null \
    || { echo "ollama not reachable on :50001 -- start ollama_setting/podman" >&2; exit 1; }

COMMON=(--attack dija_orig --temperature 0.0 --seed 42 --n 100)
P=outputs/dijaorig_greedy_jbb

say "generation"
(
    CUDA_VISIBLE_DEVICES=$GPU_A $PY llada_steering_remasking_v2.py "${COMMON[@]}" --mode off \
        --out ${P}_off.json > log/dijaorig_greedy_jbb_off.log 2>&1
    CUDA_VISIBLE_DEVICES=$GPU_A $PY llada_steering_remasking_v2.py "${COMMON[@]}" --mode steer \
        --out ${P}_steer.json > log/dijaorig_greedy_jbb_steer.log 2>&1
) &
A=$!
(
    CUDA_VISIBLE_DEVICES=$GPU_B $PY llada_steering_remasking_v2.py "${COMMON[@]}" --mode repair \
        --out ${P}_repair.json > log/dijaorig_greedy_jbb_repair.log 2>&1
    CUDA_VISIBLE_DEVICES=$GPU_B $PY run_proposed_dija.py "${COMMON[@]}" \
        --out ${P}_proposed.json > log/dijaorig_greedy_jbb_proposed.log 2>&1
) &
B=$!
fail=0
wait $A || fail=1
wait $B || fail=1
[ $fail -eq 0 ] || { echo "generation failed -- see log/dijaorig_greedy_jbb_*.log" >&2; exit 1; }

say "scoring"
(
    for c in off steer repair proposed; do
        CUDA_VISIBLE_DEVICES=$GPU_A $PY eval_llamaguard.py --in ${P}_$c.json \
            --out ${P}_${c}_lg4.json > log/dijaorig_greedy_jbb_${c}_lg4.log 2>&1
    done
) &
A=$!
(
    for c in off steer repair proposed; do
        $PY run_sr_eval.py --in ${P}_$c.json --out ${P}_${c}_sr.json \
            > log/dijaorig_greedy_jbb_${c}_sr.log 2>&1
    done
) &
B=$!
fail=0
wait $A || fail=1
wait $B || fail=1
[ $fail -eq 0 ] || { echo "scoring failed -- see log/dijaorig_greedy_jbb_*_{lg4,sr}.log" >&2; exit 1; }

say "summary"
$PY - <<'PY'
import json
P = "outputs/dijaorig_greedy_jbb"
print(f"{'condition':<10} {'LG4 unsafe':>11} {'SR ASR':>8} {'SR mean':>8}")
for c in ["off", "steer", "repair", "proposed"]:
    lg = json.load(open(f"{P}_{c}_lg4.json"))["summary"]["generation"]
    sr = json.load(open(f"{P}_{c}_sr.json"))["summary"]
    print(f"{c:<10} {lg['unsafe']:>4}/{lg['total']:<3} ({lg['unsafe_rate']:.2f}) "
          f"{sr['asr']:>8.2f} {sr['mean_sr_score']:>8.3f}")
PY
