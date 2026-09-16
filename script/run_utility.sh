#!/bin/bash
# Utility (accuracy) run: SOURCE in mmlu|gsm8k|truthfulqa_mc, DEFENSE_ARGS picks the
# defense, TAG names the output. Greedy decoding (temperature 0) so the answer
# letter/number is deterministic; eval_utility.py scores it on CPU afterwards.
#   outputs/<PREFIX>-none-<TAG>-42.json and ..._acc.json
set -e
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
GPU=${GPU:-1}
SOURCE=${SOURCE:-truthfulqa_mc}
N=${N:-817}
TAG=${TAG:-v3}
DEFENSE_ARGS=${DEFENSE_ARGS:---defense ours --remask v3 --steer adaptive}
GEN_LENGTH=${GEN_LENGTH:-128}
case "$SOURCE" in
    truthfulqa_mc) PREFIX=TQAmc ;; mmlu) PREFIX=MMLU ;; gsm8k) PREFIX=GSM8K ;;
    *) echo "unknown SOURCE=$SOURCE"; exit 1 ;;
esac
MODEL=${MODEL:-llada}                 # llada | dream (artifacts under outputs/<model>/)
MODEL_ARGS=""; OUT_DIR=outputs
if [ "$MODEL" != llada ]; then MODEL_ARGS="--model $MODEL"; OUT_DIR=outputs/$MODEL; mkdir -p "$OUT_DIR"; fi
OUT=$OUT_DIR/${PREFIX}-none-${TAG}-42.json
CUDA_VISIBLE_DEVICES=$GPU python exp.py $MODEL_ARGS --attack none $DEFENSE_ARGS --source "$SOURCE" --n "$N" \
    --temperature 0.0 --gen-length "$GEN_LENGTH" --out "$OUT"
python eval_utility.py --in "$OUT" --out "${OUT%.json}_acc.json"
echo "DONE $OUT"
