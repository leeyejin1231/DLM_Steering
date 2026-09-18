#!/usr/bin/env bash
# DIJA infilling through exp.py and the included DiffuGuard generator.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
GPU=${GPU:-$GPU_A}
SOURCE=${SOURCE:-jbb_harmful}
SEED=${SEED:-42}
CONFIG=${CONFIG:-full}
case "$SOURCE" in
    jbb_harmful) PREFIX=JBB ;;
    harmbench) PREFIX=HarmBench ;;
    strongreject) PREFIX=SR ;;
    *) echo "unknown SOURCE=$SOURCE" >&2; exit 1 ;;
esac
case "$CONFIG" in
    full) REMASKING=adaptive_step ;;
    hidden) REMASKING=low_confidence ;;
    *) echo "unknown CONFIG=$CONFIG (hidden|full)" >&2; exit 1 ;;
esac
OUT=outputs/${PREFIX}-dija-dgm-${CONFIG}-${SEED}.json
mkdir -p outputs log
CUDA_VISIBLE_DEVICES=$GPU "$PY" exp.py --attack dija --defense diffuguard \
    --source "$SOURCE" --seed "$SEED" --reproduct \
    --gen-length 0 --steps 200 --block-length 200 --temperature 0.2 \
    --remasking "$REMASKING" --repair-scope all --sp-threshold 0.2 \
    --refinement-steps 8 --remask-ratio 0.9 --out "$OUT" \
    > "log/diffuguard_${PREFIX}_${CONFIG}_${SEED}.log" 2>&1
CUDA_VISIBLE_DEVICES=$GPU "$PY" eval_llamaguard.py --in "$OUT" --out "${OUT%.json}_lg4.json"
"$PY" run_sr_eval.py --in "$OUT" --out "${OUT%.json}_sr.json" --port 50001 --gpu "$GPU"
echo "DONE $OUT"
