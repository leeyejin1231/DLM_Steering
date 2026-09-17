#!/usr/bin/env bash
# The gap the sweeps left before picking a configuration: does turning v3 on
# add over-refusal on benign prompts?  (tune_overrefusal.sh measured
# --remask none only.)  Same benign slices / judge as the sweeps,
# recovery-steps 16.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=.venv/bin/python
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
OUT=outputs/tune
LOG=$OUT/commands.log
V3="--remask v3 --recovery-steps 16"

run() { # stem, args...
    local stem=$1; shift
    local f="$OUT/$stem.json"
    printf '%s exp.py %s --gpus %s --out %s\n' "$PY" "$*" "$GPUS" "$f" >> "$LOG"
    [ -s "$f" ] && { echo "  skip $stem"; return 0; }
    $PY exp.py "$@" --gpus "$GPUS" --out "$f" > "$OUT/$stem.log" 2>&1 \
        && echo "  ok   $stem" || { echo "  FAIL $stem"; tail -3 "$OUT/$stem.log"; }
}

echo "== over-refusal with v3 on =="
for A in 0.5 1.0; do
    run "or-xstest-v3-a$A" --attack none --defense ours --steer adaptive $V3 \
        --alpha "$A" --source xstest_safe --n 250 --temperature 0 --reproduct --seed 42
    run "or-tqa-v3-a$A" --attack none --defense ours --steer adaptive $V3 \
        --alpha "$A" --source truthfulqa --start 400 --n 100 --temperature 0 \
        --reproduct --seed 42
done

echo "== grading =="
for f in "$OUT"/or-*-v3-*.json; do
    case "$f" in *_judged.json|*.part*) continue;; esac
    g="${f%.json}_judged.json"; [ -s "$g" ] && continue
    printf 'CUDA_VISIBLE_DEVICES=0,1 %s -m steering.judge_refusal --judge local --in %s --out %s\n' "$PY" "$f" "$g" >> "$LOG"
    CUDA_VISIBLE_DEVICES=0,1 $PY -m steering.judge_refusal --judge local --in "$f" --out "$g" \
        > "${f%.json}_judged.log" 2>&1 \
        && echo "  ok   $(basename "${f%.json}")" || echo "  FAIL $(basename "${f%.json}")"
done
echo "== done =="
