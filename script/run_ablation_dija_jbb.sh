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
# Checkpoints, layers and thresholds come from common.sh (ours_args llada).
# Outputs: outputs/ablation/JBB-dija-<cond>-<seed>{,_lg4,_sr}.json
#   SEED=42   CONDS="allbnd bnd1 bnd2 bnd3" selects the boundary analysis instead.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
MODEL=llada; model_setup
SEED=${SEED:-42}
OUT=outputs/ablation
mkdir -p "$OUT"

COMMON=(--model llada --attack dija --defense ours --source jbb_harmful --n 100
        --seed "$SEED" --reproduct
        --gen-length 128 --steps 128 --block-length 32 --temperature 0.2
        $(ours_args llada))
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
ORDER=(${CONDS:-full noremask promptrm0 nosteer})

FILES=()
say "generation ($(date '+%F %T'))"
for c in "${ORDER[@]}"; do
  F=$OUT/JBB-dija-$c-$SEED.json
  FILES+=("$F")
  if complete_json "$F" 100; then echo "skip $F (complete)"; continue; fi
  echo "== $c -> $F"
  "$PY" exp.py "${COMMON[@]}" ${COND[$c]} --gpus "$GPUS" --procs-per-gpu "$PROCS_PER_GPU" --out "$F" \
      > "log/gen_JBB-dija-$c-$SEED.log" 2>&1 \
      || { echo "generation failed: $c (log/gen_JBB-dija-$c-$SEED.log)"; exit 1; }
done
grade_files "${FILES[@]}"
echo "ALL ABLATION DONE $(date '+%F %T')"
