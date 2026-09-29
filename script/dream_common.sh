#!/bin/bash
# Shared settings for the Dream-v0-Instruct-7B scripts. Source it after
# common.sh; do not run it.
#
# The deployed Dream defense (README "Dream-v0-Instruct-7B"): v3 remask +
# adaptive steering at layer 20, an 80% random prompt remask on trigger, and
# the first-boundary response detector response_detector3_committed_L20.pt,
# read at its own layer 20 with its own cutoff (0.387). The gate stays at the
# detector bundle's best layer (14) with outputs/dream/gate_threshold.json.
# Every knob can be overridden from the environment, e.g.
#   TAG=v3rp80 SEEDS="42 43 44" STEER_LAYER=20 PROMPT_FRAC=0.8 script/run_dream_overrefusal.sh
DREAM_OUT=${DREAM_OUT:-outputs/dream}
TAG=${TAG:-v3rp80}                       # output name: $DREAM_OUT/<SET>-<attack>-$TAG-<seed>.json
SEEDS=${SEEDS:-"42 43 44"}
STEER_LAYER=${STEER_LAYER:-20}
PROMPT_FRAC=${PROMPT_FRAC:-0.8}
RESPONSE_DETECTOR=${RESPONSE_DETECTOR:-$DREAM_OUT/response_detector3_committed_L20.pt}
GEN=${GEN:-128}
STEPS=${STEPS:-128}
BLOCK=${BLOCK:-32}

DREAM_BASE=(--model dream --gen-length "$GEN" --steps "$STEPS" --block-length "$BLOCK"
            --gpus "$GPUS" --procs-per-gpu "$PROCS_PER_GPU")
DREAM_OURS=(--defense ours --remask v3 --steer adaptive --layer "$STEER_LAYER"
            --vector "$DREAM_OUT/steer_vector.pt" --detector "$DREAM_OUT/steer_detector.pt"
            --response-detector "$RESPONSE_DETECTOR"
            --remask-prompt --remask-prompt-frac "$PROMPT_FRAC")

for f in "$DREAM_OUT/steer_vector.pt" "$DREAM_OUT/steer_detector.pt" \
         "$DREAM_OUT/gate_threshold.json" "$RESPONSE_DETECTOR"; do
    [ -e "$f" ] || { echo "missing Dream checkpoint: $f (README 'Dream-v0-Instruct-7B')" >&2; exit 1; }
done
mkdir -p "$DREAM_OUT"

# complete_json FILE N: succeeds when FILE already holds N result rows.
complete_json() {
    [ -s "$1" ] && "$PY" -c 'import json, sys
sys.exit(0 if len(json.load(open(sys.argv[1]))["results"]) == int(sys.argv[2]) else 1)' "$1" "$2"
}
