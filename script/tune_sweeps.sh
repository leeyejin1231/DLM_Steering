#!/usr/bin/env bash
# Hyper-parameter sweeps on the FIT split, never on an eval benchmark.
#
#   data/llada8b_wild_unsafe_only.csv rows 0..19 are the held-out set
#   script/run_gated.sh reports, so every sweep here starts at row 20.
#   --reproduct pins seeds, deterministic kernels and the math SDPA backend.
#
# Sweeps run attack-free on purpose: the only in-prompt attack that could run
# here (dija_template) was deleted, and --attack dija's refined prompts exist
# only for the eval benchmarks, so attack-side tuning would leak the test set.
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
JOBS="$OUT/jobs-alpha.json"
printf '[]\n' > "$JOBS"

run() { # out-stem, args...
    local stem=$1; shift
    local f="$OUT/$stem.json"
    printf '%s exp.py %s --gpus %s --out %s\n' "$PY" "$*" "$GPUS" "$f" >> "$LOG"
    if [ -s "$f" ]; then echo "  skip $stem"; return 0; fi
    "$PY" - "$JOBS" "$@" --out "$f" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
jobs = json.loads(path.read_text())
jobs.append(sys.argv[2:])
path.write_text(json.dumps(jobs))
PY
}

echo "== A. alpha (steer-only, no attack, T=0) =="
for A in 0 0.25 0.5 1.0 2.0; do
    run "alpha-$A" --attack none --defense ours --steer adaptive --remask none \
        --alpha "$A" $FIT --temperature 0 --reproduct --seed 42
done
printf '%s exp.py --jobs %s --gpus %s\n' "$PY" "$JOBS" "$GPUS" >> "$LOG"
"$PY" exp.py --jobs "$JOBS" --gpus "$GPUS" \
    || { echo "generation failed; see $OUT/.parts/ logs"; exit 1; }
echo "== done =="
