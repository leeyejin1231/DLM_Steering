#!/usr/bin/env bash
# Generate with gated steering: 20 held-out harmful prompts at two lengths, plus
# three benign sets that measure over-refusal. ~17 min across two GPUs.
#
# Per prompt the detector is read once with the answer region fully masked -- the
# state of the first denoising step -- and the actuator hook is enabled only if
# the projection clears the threshold. A closed gate means zero intervention.
#
# KNOWN DEFECT: gate_projection reads the detector at the *generation* length,
# but the detector was extracted at gen_length=128. At 256 the projections drift
# down (mean 6.41 -> 5.42), the gate opens for 13/20 instead of 16/20, and ASR
# degrades accordingly. Fix is to pin detection to 128 regardless of output
# length; not applied yet, so len256 numbers here are pessimistic.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

require "$VECTOR" "$DETECTOR" outputs/gate_threshold.json data/llada8b_wild_unsafe_only.csv

D="--detector $DETECTOR --detector-layer $DETECTOR_LAYER --vector $VECTOR --alpha $ALPHA"

say "generating on GPU $GPU_A and GPU $GPU_B"

(
    CUDA_VISIBLE_DEVICES=$GPU_A $PY llada_steering.py $D \
        --steps 128 --gen-length 128 --n "$N_HARMFUL" \
        --out outputs/gated_len128.json > log/gated_len128.log 2>&1
    CUDA_VISIBLE_DEVICES=$GPU_A $PY steering/run_overrefusal.py $D \
        --source xstest_safe --limit "$N_BENIGN" \
        --out outputs/gated_or30_xstest.json > log/gated_xstest.log 2>&1
    CUDA_VISIBLE_DEVICES=$GPU_A $PY steering/run_overrefusal.py $D \
        --source jbb_benign --limit "$N_BENIGN" \
        --out outputs/gated_or30_jbb.json > log/gated_jbb.log 2>&1
) &
A=$!
(
    CUDA_VISIBLE_DEVICES=$GPU_B $PY llada_steering.py $D \
        --steps 256 --gen-length 256 --n "$N_HARMFUL" \
        --out outputs/gated_len256.json > log/gated_len256.log 2>&1
    CUDA_VISIBLE_DEVICES=$GPU_B $PY steering/run_overrefusal.py $D \
        --source truthfulqa --limit "$N_BENIGN" \
        --out outputs/gated_or30_tqa.json > log/gated_tqa.log 2>&1
) &
B=$!

fail=0
wait $A || fail=1
wait $B || fail=1

say "gate open rates"
for f in gated_len128 gated_len256 gated_xstest gated_jbb gated_tqa; do
    printf '  %-16s ' "$f"
    grep -oE "gate opened on [0-9]+/[0-9]+" "log/$f.log" \
        || { grep -oE "IndexError|Traceback|Error" "log/$f.log" | head -1 || echo "-"; }
done

[ $fail -eq 0 ] || { echo "a generation job failed; see log/gated_*.log" >&2; exit 1; }
