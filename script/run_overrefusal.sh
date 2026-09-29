#!/bin/bash
# Over-refusal: benign prompts under the deployed defense (no attack), judged
# with the XSTest three-way refusal rubric (gpt-oss:20b via ollama), then a
# per-seed summary. TruthfulQA generations can also be judged for
# truthful/informative (TQA_JUDGE=1).
#
#   MODEL=llada|dream   DEFENSE=ours|none   SOURCES="xstest_safe truthfulqa"   SEEDS="42 43 44"
#   TAG=$DEFENSE        output: $MODEL_OUT/<XSTest-safe|TQA>-none-$TAG-<seed>.json, *_judged.json,
#                       $MODEL_OUT/OR-summary-$TAG.{json,md}
#   GEN=128 STEPS=128 BLOCK=32   TEMPERATURE=<exp.py default unless set>   REPRODUCT=0
#   TQA_JUDGE=0         1 also writes *_tqa.json for the truthfulqa runs
#
#   MODEL=dream TAG=v3rp80 script/run_overrefusal.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
model_setup
DEFENSE=${DEFENSE:-ours}
SOURCES=${SOURCES:-"xstest_safe truthfulqa"}
SEEDS=${SEEDS:-"42 43 44"}
TAG=${TAG:-$DEFENSE}
GEN=${GEN:-128}; STEPS=${STEPS:-128}; BLOCK=${BLOCK:-32}
DEF_ARGS=${DEFENSE_ARGS:-$(defense_args "$MODEL" "$DEFENSE")}
OPT_ARGS=(); [ -n "${TEMPERATURE:-}" ] && OPT_ARGS+=(--temperature "$TEMPERATURE")
[ "${REPRODUCT:-0}" = 1 ] && OPT_ARGS+=(--reproduct)

FILES=()
for SRC in $SOURCES; do
    P=$(source_prefix "$SRC"); ROWS=$(source_rows "$SRC")
    for SEED in $SEEDS; do
        OUT=$MODEL_OUT/$P-none-$TAG-$SEED.json
        FILES+=("$OUT")
        if complete_json "$OUT" "$ROWS"; then echo "skip $OUT (complete)"; continue; fi
        say "$MODEL $DEFENSE $SRC seed=$SEED -> $OUT ($(date '+%F %T'))"
        "$PY" exp.py --model "$MODEL" --attack none $DEF_ARGS \
            --source "$SRC" --n "$ROWS" --seed "$SEED" "${OPT_ARGS[@]}" \
            --gen-length "$GEN" --steps "$STEPS" --block-length "$BLOCK" \
            --gpus "$GPUS" --procs-per-gpu "$PROCS_PER_GPU" ${EXTRA_ARGS:-} --out "$OUT" \
            > "log/gen_$(basename "${OUT%.json}").log" 2>&1 \
            || { echo "generation failed: $OUT"; exit 1; }
    done
done

say "refusal judge ($(date '+%F %T'))"
for F in "${FILES[@]}"; do
    J=${F%.json}_judged.json
    [ -s "$F" ] || { echo "missing $F"; continue; }
    [ -s "$J" ] && continue
    "$PY" -m dlm_steering.fitting.judge_refusal --in "$F" --out "$J" --gpus "$GPUS" \
        > "log/judge_$(basename "${F%.json}").log" 2>&1 || echo "JUDGE FAILED: $F"
done
if [ "${TQA_JUDGE:-0}" = 1 ]; then
    say "TruthfulQA truthful/informative judge ($(date '+%F %T'))"
    for F in "${FILES[@]}"; do
        case "$(basename "$F")" in TQA-none-*) ;; *) continue ;; esac
        T=${F%.json}_tqa.json
        [ -s "$F" ] && [ ! -s "$T" ] || continue
        "$PY" -m dlm_steering.fitting.judge_truthfulqa --in "$F" --out "$T" --gpus "$GPUS" \
            > "log/tqajudge_$(basename "${F%.json}").log" 2>&1 || echo "TQA JUDGE FAILED: $F"
    done
fi
stop_ollama
say "summary"
"$PY" -m dlm_steering.fitting.aggregate_overrefusal --out-dir "$MODEL_OUT" --tag "$TAG" --seeds "${SEEDS// /,}"
echo "ALL DONE $(date '+%F %T')"
