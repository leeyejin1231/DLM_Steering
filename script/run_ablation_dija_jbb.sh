#!/bin/bash
# LLaDA-8B-Instruct, DIJA (paper refined prompts, gen_length 128) on JBB (100
# rows), our defense and its three ablations, then Llama Guard 4 and
# StrongREJECT (gpt-oss:20b via ollama) grading.
#
#   full        steer adaptive + v3 remask + 80% prompt remask   (the method)
#   noremask    steer adaptive, no remasking                     (ablation 1)
#   promptrm0   steer adaptive + v3 remask, 0% prompt remask     (ablation 2)
#   nosteer     no steering, v3 remask + 80% prompt remask       (ablation 3)
#
# Method settings: steering layer 25, gate detector layer 18 / threshold 7.0,
# response detector layer 18 / threshold 0.12, alpha 1.
# Outputs: outputs/ablation/JBB-dija-<cond>-<seed>{,_lg4,_sr}.json
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/mnt/shared/huggingface-cache}"
SEED=${SEED:-42}
PYLG=${PYLG:-/home/yejin/anaconda3/envs/kotox/bin/python}   # transformers 5.x for Llama Guard 4
OUT=outputs/ablation
mkdir -p "$OUT"

COMMON=(--attack dija --defense ours --source jbb_harmful --n 100
        --seed "$SEED" --reproduct
        --gen-length 128 --steps 128 --block-length 32 --temperature 0.2
        --vector "$VECTOR" --detector "$DETECTOR"
        --layer 25 --detector-layer 18 --gate-threshold 7.0 --alpha "$ALPHA"
        --response-detector outputs/response_detector.pt --response-threshold 0.12)
declare -A COND=(
  [full]="--steer adaptive --remask v3 --remask-prompt --remask-prompt-frac 0.8"
  [noremask]="--steer adaptive --remask none"
  [promptrm0]="--steer adaptive --remask v3"
  [nosteer]="--steer none --remask v3 --remask-prompt --remask-prompt-frac 0.8"
  # analysis: where the response detector may trigger (full method otherwise)
  [allbnd]="--steer adaptive --remask v3 --remask-prompt --remask-prompt-frac 0.8 --audit-all-boundaries"
  [bnd1]="--steer adaptive --remask v3 --remask-prompt --remask-prompt-frac 0.8 --audit-boundary 1"
  [bnd2]="--steer adaptive --remask v3 --remask-prompt --remask-prompt-frac 0.8 --audit-boundary 2"
  [bnd3]="--steer adaptive --remask v3 --remask-prompt --remask-prompt-frac 0.8 --audit-boundary 3"
)
# CONDS="allbnd bnd1 bnd2 bnd3" selects a subset; default = the four ablation conditions.
ORDER=(${CONDS:-full noremask promptrm0 nosteer})

JOBS=$OUT/jobs-dija-${CONDS:+analysis-}$SEED.json
"$PY" - "$JOBS" "${ORDER[@]}" <<PY
import json, sys, shlex
from pathlib import Path
common = shlex.split("""${COMMON[@]}""")
cond = {"full": shlex.split("""${COND[full]}"""),
        "noremask": shlex.split("""${COND[noremask]}"""),
        "promptrm0": shlex.split("""${COND[promptrm0]}"""),
        "nosteer": shlex.split("""${COND[nosteer]}"""),
        "allbnd": shlex.split("""${COND[allbnd]}"""),
        "bnd1": shlex.split("""${COND[bnd1]}"""),
        "bnd2": shlex.split("""${COND[bnd2]}"""),
        "bnd3": shlex.split("""${COND[bnd3]}""")}
jobs = []
for name in sys.argv[2:]:
    out = Path("$OUT") / f"JBB-dija-{name}-$SEED.json"
    if out.exists() and out.stat().st_size:
        print("skip", out); continue
    jobs.append([*common, *cond[name], "--out", str(out)])
Path(sys.argv[1]).write_text(json.dumps(jobs, indent=1))
print(len(jobs), "generation jobs ->", sys.argv[1])
PY

say "generation ($(date '+%F %T'))"
"$PY" exp.py --jobs "$JOBS" --gpus "$GPUS" --procs-per-gpu "$PROCS_PER_GPU" \
    || { echo "generation failed"; exit 1; }

say "Llama Guard 4 ($(date '+%F %T'))"
for c in "${ORDER[@]}"; do
  F=$OUT/JBB-dija-$c-$SEED.json; LG=${F%.json}_lg4.json
  [ -s "$F" ] || { echo "missing $F"; continue; }
  [ -s "$LG" ] && continue
  HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" \
    "$PYLG" eval_llamaguard.py --in "$F" --out "$LG" --gpus "$GPUS" > "log/lg4_JBB-dija-$c-$SEED.log" 2>&1 \
    || echo "LG4 FAILED: $c"
done

say "StrongREJECT / gpt-oss ($(date '+%F %T'))"
for c in "${ORDER[@]}"; do
  F=$OUT/JBB-dija-$c-$SEED.json; SR=${F%.json}_sr.json
  [ -s "$F" ] || continue
  [ -s "$SR" ] && continue
  "$PY" run_sr_eval.py --in "$F" --out "$SR" --gpus "$GPUS" > "log/sr_JBB-dija-$c-$SEED.log" 2>&1 \
    || echo "SR FAILED: $c"
done
podman stop ollama-50001 ollama-50002 >/dev/null 2>&1

say "report"
"$PY" script/report.py $(ls $OUT/JBB-dija-*-${SEED}_lg4.json $OUT/JBB-dija-*-${SEED}_sr.json 2>/dev/null)
echo "ALL ABLATION DONE $(date '+%F %T')"
