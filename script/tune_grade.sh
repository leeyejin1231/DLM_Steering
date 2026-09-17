#!/usr/bin/env bash
# Llama Guard ASR for every sweep run that does not have a verdict file yet.
# Appended to outputs/tune/commands.log so RESULTS.md can quote it.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=.venv/bin/python
GPU=${GPU:-0}
OUT=outputs/tune
for f in "$OUT"/*.json; do
    case "$f" in *_lg4.json|*.part*) continue;; esac
    g="${f%.json}_lg4.json"
    [ -s "$g" ] && continue
    printf 'CUDA_VISIBLE_DEVICES=%s %s eval_llamaguard.py --in %s --out %s\n' \
        "$GPU" "$PY" "$f" "$g" >> "$OUT/commands.log"
    CUDA_VISIBLE_DEVICES=$GPU $PY eval_llamaguard.py --in "$f" --out "$g" \
        > "${f%.json}_lg4.log" 2>&1 \
        && echo "  ok   $(basename "${f%.json}")" || echo "  FAIL $(basename "${f%.json}")"
done
