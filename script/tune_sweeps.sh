#!/usr/bin/env bash
# Hyper-parameter sweeps on the FIT split, never on an eval benchmark.
#
#   data/llada8b_wild_unsafe_only.csv rows 0..19 are the held-out set
#   script/run_gated.sh reports, so every sweep here starts at row 20.
#   --reproduct pins seeds, deterministic kernels and the math SDPA backend.
#
# Every command is echoed to outputs/tune/commands.log as it runs; RESULTS.md
# quotes that file, so the log is the record of how the numbers were made.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=.venv/bin/python
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
FIT="--source wj_unsafe --start 20 --n 30"
OUT=outputs/tune
LOG=$OUT/commands.log
mkdir -p "$OUT"; : > "$LOG"

run() { # out-stem, args...
    local stem=$1; shift
    local f="$OUT/$stem.json"
    printf '%s exp.py %s --gpus %s --out %s\n' "$PY" "$*" "$GPUS" "$f" >> "$LOG"
    if [ -s "$f" ]; then echo "  skip $stem"; return 0; fi
    $PY exp.py "$@" --gpus "$GPUS" --out "$f" > "$OUT/$stem.log" 2>&1 \
        && echo "  ok   $stem" || { echo "  FAIL $stem"; tail -3 "$OUT/$stem.log"; }
}

echo "== A. alpha (steer-only, no attack, T=0) =="
for A in 0 0.25 0.5 1.0 2.0; do
    run "alpha-$A" --attack none --defense ours --steer adaptive --remask none \
        --alpha "$A" $FIT --temperature 0 --reproduct --seed 42
done

echo "== B. temperature robustness (dija_template + v3, 3 seeds) =="
for T in 0 0.2 0.5 1.0; do
    for S in 42 43 44; do
        run "temp-$T-seed$S" --attack dija_template --defense ours \
            --steer adaptive --remask v3 $FIT --temperature "$T" \
            --reproduct --seed "$S"
    done
done

echo "== C. v3 recovery counts (dija_template, T=0.2) =="
for R in 16 32 64; do
    for N in 1 2 3; do
        run "rec-s$R-n$N" --attack dija_template --defense ours \
            --steer adaptive --remask v3 --recovery-steps "$R" \
            --recovery-rounds "$N" $FIT --temperature 0.2 --reproduct --seed 42
    done
done

echo "== D. v2 remask budget (dija_template, T=0.2) =="
for K in 8 16 32; do
    run "v2tok-$K" --attack dija_template --defense ours --steer adaptive \
        --remask v2 --max-remask-tokens "$K" $FIT --temperature 0.2 \
        --reproduct --seed 42
done
echo "== done =="
