#!/usr/bin/env bash
# Over-refusal across the alpha sweep -- the cost side of the ASR numbers.
#
# Two benign sets on purpose: xstest_safe is the standard over-refusal
# benchmark (harmless prompts that merely SOUND harmful), truthfulqa is plain
# questions, so a defense that refuses there is damaging ordinary use. The
# truthfulqa slice starts at 400 so rows 0..399 stay clean for reporting.
#
# Judged with --judge local (the XSTest rubric on a local HF model): this box
# has no podman, so the ollama judge cannot run. Do not mix its labels with
# gpt-oss:20b labels inside one comparison.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=.venv/bin/python
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
OUT=outputs/tune
LOG=$OUT/commands.log
mkdir -p "$OUT"

gen() { # stem, source-args...
    local stem=$1; shift
    local f="$OUT/$stem.json"
    printf '%s exp.py --attack none --defense ours --steer adaptive --remask none %s --gpus %s --out %s\n' \
        "$PY" "$*" "$GPUS" "$f" >> "$LOG"
    [ -s "$f" ] && { echo "  skip $stem"; return 0; }
    $PY exp.py --attack none --defense ours --steer adaptive --remask none \
        "$@" --gpus "$GPUS" --out "$f" > "$OUT/$stem.log" 2>&1 \
        && echo "  ok   $stem" || { echo "  FAIL $stem"; tail -3 "$OUT/$stem.log"; }
}

for A in 0 0.25 0.5 1.0 2.0; do
    gen "or-xstest-a$A" --alpha "$A" --source xstest_safe --n 250 \
        --temperature 0 --reproduct --seed 42
    gen "or-tqa-a$A" --alpha "$A" --source truthfulqa --start 400 --n 100 \
        --temperature 0 --reproduct --seed 42
done

echo "== judging =="
for f in "$OUT"/or-*.json; do
    case "$f" in *_judged.json|*.part*) continue;; esac
    g="${f%.json}_judged.json"
    [ -s "$g" ] && continue
    printf 'CUDA_VISIBLE_DEVICES=0,1 %s -m steering.judge_refusal --judge local --in %s --out %s\n' \
        "$PY" "$f" "$g" >> "$LOG"
    CUDA_VISIBLE_DEVICES=0,1 $PY -m steering.judge_refusal --judge local \
        --in "$f" --out "$g" > "${f%.json}_judged.log" 2>&1 \
        && echo "  ok   $(basename "${f%.json}")" || echo "  FAIL $(basename "${f%.json}")"
done
echo "== done =="
