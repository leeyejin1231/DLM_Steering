#!/usr/bin/env bash
# One-time construction of everything the gated method needs. ~40 min on one A6000
# for steps 1-4; step 5 generates 384 responses and judges them (~1 h, 2 GPUs).
#
# Steps 1-4 never train: forward passes under torch.no_grad() plus a mean
# difference. Step 5 fits a logistic regression on pooled features. Model
# weights are never touched.
#
#   1 build_pairs            the 4-arm contrast set, with the 20 benchmark prompts excluded
#   2 fit_vector             actuator direction (LLaDA: layer 25) -- causes refusal
#   3 fit_detector           detector direction (LLaDA: layer 18) -- reads harmfulness
#   4 pick_threshold         gate threshold, chosen on fit-split prompts only
#   5 fit_response_detector  boundary response detector for --remask v3 (Llama-Guard labels)
#
#   MODEL=dream script/build_vectors.sh   -> outputs/dream/ with Dream-v0-Instruct-7B;
#   layers default to each fitter's best_layer (override with DETECTOR_LAYER=..).
#
# Re-running is safe but wasteful; pass --force to rebuild existing artifacts.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

skip() { [ $FORCE -eq 0 ] && [ -e "$1" ] && { echo "exists, skipping: $1"; return 0; }; return 1; }

say "1/4 build_pairs -- contrast dataset (eval prompts held out)"
skip data/steer_pairs.json || $PY steering/build_pairs.py

LAYER_ARG=${DETECTOR_LAYER:+--layer $DETECTOR_LAYER}   # empty -> fitter's best_layer
LOG_TAG=${MODEL/llada/}                                 # llada keeps the old log names
LOG_TAG=${LOG_TAG:+_$LOG_TAG}

say "2/5 fit_vector -- actuator direction ($MODEL)"
skip "$OUT_DIR/steer_vector.pt" || \
    CUDA_VISIBLE_DEVICES=$GPU_A $PY steering/fit_vector.py $MODEL_ARGS \
        2>&1 | tee "log/fit_vector$LOG_TAG.log"

say "3/5 fit_detector -- prompt-side harmfulness detector"
skip "$OUT_DIR/steer_detector.pt" || \
    CUDA_VISIBLE_DEVICES=$GPU_A $PY steering/fit_detector.py $MODEL_ARGS \
        2>&1 | tee "log/fit_detector$LOG_TAG.log"

say "4/5 pick_threshold -- gate threshold from fit split"
skip "$OUT_DIR/gate_threshold.json" || \
    CUDA_VISIBLE_DEVICES=$GPU_A $PY steering/pick_threshold.py $MODEL_ARGS $LAYER_ARG \
        2>&1 | tee "log/pick_threshold$LOG_TAG.log"

say "5/5 fit_response_detector -- boundary response detector for --remask v3"
if [ "$GPU_A" = "$GPU_B" ]; then VIS=$GPU_A; GUARD=cuda:0; else VIS=$GPU_A,$GPU_B; GUARD=cuda:1; fi
skip "$OUT_DIR/response_detector.pt" || \
    CUDA_VISIBLE_DEVICES=$VIS $PY steering/fit_response_detector.py $MODEL_ARGS $LAYER_ARG \
        --device cuda:0 --guard-device "$GUARD" 2>&1 | tee "log/fit_response_detector$LOG_TAG.log"

say "done"
OUT_DIR="$OUT_DIR" $PY - <<'PY'
import json, os, torch
o = os.environ["OUT_DIR"]
a = torch.load(f"{o}/steer_vector.pt", map_location="cpu")
d = torch.load(f"{o}/steer_detector.pt", map_location="cpu")
th = json.load(open(f"{o}/gate_threshold.json"))
r = torch.load(f"{o}/response_detector.pt", map_location="cpu", weights_only=False)
li = a["layers"].index(a["best_layer"])
print(f"model    : {a.get('model')}  ({o}/)")
print(f"actuator : best layer {a['best_layer']}  pairs {len(a['kept_pair_ids'])}  "
      f"mean|h| {a['mean_act_norm'][li]:.1f}")
print(f"detector : best layer {d['best_layer']}  gen_length {d['gen_length']}")
print(f"threshold: {th['threshold']} at layer {th['layer']}  (opens for "
      f"{th['open_rate_harmful']:.2f} of fit harmful, {th['open_rate_benign']:.2f} of fit benign)")
print(f"response : layer {r['layer']}  cutoff {r['threshold']}")
if a["best_layer"] <= th["layer"]:
    print("WARNING: actuator best_layer is not after the gate layer; pass --layer "
          "explicitly to exp.py (--defense ours needs steering layers > detector layer)")
PY
