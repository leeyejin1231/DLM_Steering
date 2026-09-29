#!/bin/bash
# Benchmark runs: one attack under one defense over the harmful sets and seeds,
# then Llama Guard 4 and StrongREJECT grading and a report. The main-table
# setting: gen_length 128, 128 steps, block 32, temperature 0.2, --reproduct.
#
#   MODEL=llada|dream                    (default llada)
#   ATTACK=pap|dija|pair|prefix|none     (default pap; pap needs the seed's cache, README 2-1)
#   DEFENSE=ours|diffuguard|none|selfreminder   (default ours; deployed flags from common.sh)
#   SOURCES="jbb_harmful harmbench strongreject"   SEEDS="42 43 44"
#   TAG=$DEFENSE        output name: $MODEL_OUT/<JBB|HB|SR>-$ATTACK-$TAG-<seed>.json
#   DEFENSE_ARGS=...    replaces the deployed defense flags entirely
#   EXTRA_ARGS=...      appended to every exp.py call
#   GEN=128 STEPS=128 BLOCK=32 TEMPERATURE=0.2 REPRODUCT=1 (dream: 0)  N=<rows per source>
#   GRADE=1             0 skips grading
#
#   MODEL=dream ATTACK=pap SOURCES=jbb_harmful TAG=v3rp80 REPRODUCT=0 script/run_benchmark.sh
#   ATTACK=dija DEFENSE=diffuguard GEN=0 STEPS=200 BLOCK=200 script/run_benchmark.sh   # DiffuGuard infilling setting
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
model_setup
ATTACK=${ATTACK:-pap}
DEFENSE=${DEFENSE:-ours}
SOURCES=${SOURCES:-"jbb_harmful harmbench strongreject"}
SEEDS=${SEEDS:-"42 43 44"}
TAG=${TAG:-$DEFENSE}
GEN=${GEN:-128}; STEPS=${STEPS:-128}; BLOCK=${BLOCK:-32}; TEMPERATURE=${TEMPERATURE:-0.2}
if [ "$MODEL" = dream ]; then REPRODUCT=${REPRODUCT:-0}; else REPRODUCT=${REPRODUCT:-1}; fi
DEF_ARGS=${DEFENSE_ARGS:-$(defense_args "$MODEL" "$DEFENSE")}
REPRO_ARGS=(); [ "$REPRODUCT" = 1 ] && REPRO_ARGS=(--reproduct)

FILES=()
for SRC in $SOURCES; do
    P=$(source_prefix "$SRC"); ROWS=$(source_rows "$SRC")
    for SEED in $SEEDS; do
        OUT=$MODEL_OUT/$P-$ATTACK-$TAG-$SEED.json
        FILES+=("$OUT")
        if complete_json "$OUT" "$ROWS"; then echo "skip $OUT (complete)"; continue; fi
        say "$MODEL $ATTACK/$DEFENSE $SRC seed=$SEED -> $OUT ($(date '+%F %T'))"
        "$PY" exp.py --model "$MODEL" --attack "$ATTACK" $DEF_ARGS \
            --source "$SRC" --n "$ROWS" --seed "$SEED" "${REPRO_ARGS[@]}" \
            --gen-length "$GEN" --steps "$STEPS" --block-length "$BLOCK" --temperature "$TEMPERATURE" \
            --gpus "$GPUS" --procs-per-gpu "$PROCS_PER_GPU" ${EXTRA_ARGS:-} --out "$OUT" \
            > "log/gen_$(basename "${OUT%.json}").log" 2>&1 \
            || { echo "generation failed: $OUT (log/gen_$(basename "${OUT%.json}").log)"; exit 1; }
    done
done
[ "${GRADE:-1}" = 1 ] && grade_files "${FILES[@]}"
echo "ALL DONE $(date '+%F %T')"
