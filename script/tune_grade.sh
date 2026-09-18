#!/usr/bin/env bash
# Llama Guard ASR for every sweep run that does not have a verdict file yet.
# Appended to outputs/tune/commands.log so RESULTS.md can quote it.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=.venv/bin/python
GPU=${GPU:-0}
OUT=outputs/tune
JOBS="$OUT/jobs-grade.json"
"$PY" - "$OUT" "$JOBS" <<'PY'
import json, sys
from pathlib import Path
jobs = []
for source in sorted(Path(sys.argv[1]).glob("*.json")):
    if source.name.endswith(("_lg4.json", "_judged.json")) or ".part" in source.name:
        continue
    data = json.loads(source.read_text())
    # Job manifests, timing files and run status are not generations.
    if not isinstance(data, dict) or not {"model", "config", "results"} <= data.keys():
        continue
    output = source.with_name(source.stem + "_lg4.json")
    if output.exists() and output.stat().st_size:
        continue
    jobs.append(["--in", str(source), "--out", str(output)])
Path(sys.argv[2]).write_text(json.dumps(jobs))
PY
GRADE_GPUS=${GRADE_GPUS:-$GPU}
printf '%s eval_llamaguard.py --jobs %s --gpus %s\n' \
    "$PY" "$JOBS" "$GRADE_GPUS" >> "$OUT/commands.log"
"$PY" eval_llamaguard.py --jobs "$JOBS" --gpus "$GRADE_GPUS"
