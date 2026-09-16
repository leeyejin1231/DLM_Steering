#!/usr/bin/env bash
# One-time construction of everything the gated method needs. ~40 min on one A6000.
#
# Nothing here trains: every step is forward passes under torch.no_grad() plus a
# mean difference. Model weights are never touched.
#
#   1 build_pairs     the 4-arm contrast set, with the 20 benchmark prompts excluded
#   2 fit_vector      actuator direction (layer 25) -- causes refusal
#   3 fit_detector    detector direction (layer 18) -- reads harmfulness
#   4 pick_threshold  gate threshold, chosen on fit-split prompts only
#
# Re-running is safe but wasteful; pass --force to rebuild existing artifacts.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

skip() { [ $FORCE -eq 0 ] && [ -e "$1" ] && { echo "exists, skipping: $1"; return 0; }; return 1; }

say "1/4 build_pairs -- contrast dataset (eval prompts held out)"
skip data/steer_pairs.json || $PY -m steering.build_pairs

say "2/4 fit_vector -- actuator direction"
skip outputs/steer_vector.pt || \
    CUDA_VISIBLE_DEVICES=$GPU_A $PY -m steering.fit_vector 2>&1 | tee log/fit_vector.log

say "3/4 fit_detector -- prompt-side harmfulness detector"
skip outputs/steer_detector.pt || \
    CUDA_VISIBLE_DEVICES=$GPU_A $PY -m steering.fit_detector 2>&1 | tee log/fit_detector.log

say "4/4 pick_threshold -- gate threshold from fit split"
skip outputs/gate_threshold.json || \
    CUDA_VISIBLE_DEVICES=$GPU_A $PY -m steering.pick_threshold --layer "$DETECTOR_LAYER" \
        2>&1 | tee log/pick_threshold.log

say "done"
$PY - <<'PY'
import json, torch
a = torch.load("outputs/steer_vector.pt", map_location="cpu")
d = torch.load("outputs/steer_detector.pt", map_location="cpu")
th = json.load(open("outputs/gate_threshold.json"))
li = a["layers"].index(a["best_layer"])
print(f"actuator : layer {a['best_layer']}  pairs {len(a['kept_pair_ids'])}  "
      f"mean|h| {a['mean_act_norm'][li]:.1f}")
print(f"detector : layer {th['layer']}  gen_length {d['gen_length']}")
print(f"threshold: {th['threshold']}  (opens for {th['open_rate_harmful']:.2f} of fit harmful, "
      f"{th['open_rate_benign']:.2f} of fit benign)")
PY
