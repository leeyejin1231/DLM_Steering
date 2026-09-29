#!/bin/bash
# Utility / generalisation: graded sets under the deployed defense and without
# it, scored by eval_utility.py (accuracy).
#
#   MODEL=llada|dream   SOURCES="math500 truthfulqa_mc"  (also mmlu, gsm8k; N=<rows> to subsample)
#   DEFS="ours none"    each entry is a --defense; ours uses the deployed flags from common.sh
#   SEEDS="42 43 44"    TAG=ours   output: $MODEL_OUT/<MATH500|TQAmc|...>-none-<TAG|none>-<seed>.json, *_acc.json
#   GEN=128 STEPS=128 BLOCK=32   TEMPERATURE=<exp.py default unless set>   REPRODUCT=0
#
#   MODEL=dream TAG=v3rp80 script/run_utility.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
model_setup
SOURCES=${SOURCES:-"math500 truthfulqa_mc"}
DEFS=${DEFS:-"ours none"}
SEEDS=${SEEDS:-"42 43 44"}
TAG=${TAG:-ours}
GEN=${GEN:-128}; STEPS=${STEPS:-128}; BLOCK=${BLOCK:-32}
OPT_ARGS=(); [ -n "${TEMPERATURE:-}" ] && OPT_ARGS+=(--temperature "$TEMPERATURE")
[ "${REPRODUCT:-0}" = 1 ] && OPT_ARGS+=(--reproduct)

for SRC in $SOURCES; do
    P=$(source_prefix "$SRC"); ROWS=$(source_rows "$SRC")
    for DEF in $DEFS; do
        if [ "$DEF" = ours ]; then DTAG=$TAG; else DTAG=$DEF; fi
        DEF_ARGS=$(defense_args "$MODEL" "$DEF")
        for SEED in $SEEDS; do
            OUT=$MODEL_OUT/$P-none-$DTAG-$SEED.json
            if complete_json "$OUT" "$ROWS"; then echo "skip $OUT (complete)"; else
                say "$MODEL $DEF $SRC seed=$SEED -> $OUT ($(date '+%F %T'))"
                "$PY" exp.py --model "$MODEL" --attack none $DEF_ARGS \
                    --source "$SRC" --n "$ROWS" --seed "$SEED" "${OPT_ARGS[@]}" \
                    --gen-length "$GEN" --steps "$STEPS" --block-length "$BLOCK" \
                    --gpus "$GPUS" --procs-per-gpu "$PROCS_PER_GPU" ${EXTRA_ARGS:-} --out "$OUT" \
                    > "log/gen_$(basename "${OUT%.json}").log" 2>&1 \
                    || { echo "generation failed: $OUT"; continue; }
            fi
            "$PY" eval_utility.py --in "$OUT" --out "${OUT%.json}_acc.json" > /dev/null 2>&1 \
                && "$PY" -c 'import json, sys; s = json.load(open(sys.argv[1])); s = s.get("summary", s)
print("ACC", sys.argv[2], sys.argv[3], sys.argv[4], {k: s.get(k) for k in ("total", "correct", "accuracy", "unparsed")})' \
                    "${OUT%.json}_acc.json" "$SRC" "$DEF" "$SEED" \
                || echo "eval_utility failed: $OUT"
        done
    done
done
echo "ALL DONE $(date '+%F %T')"
