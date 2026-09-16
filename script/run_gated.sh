#!/usr/bin/env bash
# Generate with gated steering: 20 held-out harmful prompts at two lengths, plus
# three benign sets that measure over-refusal. ~17 min across two GPUs.
#
# Per prompt the detector is read once with the answer region fully masked -- the
# state of the first denoising step -- and the actuator hook is enabled only if
# the projection clears the threshold. A closed gate means zero intervention.
# That is `--defense ours --steer fixed --remask none`: gate read at step 0,
# steering latched from it, nothing ever remasked.
#
# KNOWN DEFECT: gate_projection reads the detector at the *generation* length,
# but the detector was extracted at gen_length=128. At 256 the projections drift
# down (mean 6.41 -> 5.42), the gate opens for 13/20 instead of 16/20, and ASR
# degrades accordingly. Fix is to pin detection to 128 regardless of output
# length; not applied yet, so len256 numbers here are pessimistic.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

require "$VECTOR" "$DETECTOR" outputs/gate_threshold.json data/llada8b_wild_unsafe_only.csv

GATE=(--attack none --defense ours --steer fixed --remask none
      --detector "$DETECTOR" --detector-layer "$DETECTOR_LAYER"
      --vector "$VECTOR" --alpha "$ALPHA" --temperature 0 --block-length 32
      --gpus "$GPUS")

# Each job shards its own prompt set across every card in $GPUS, so they run
# one after another rather than pinned to a card each. Per-row seeding makes
# the output identical to a single-GPU run.
# wj_unsafe rows 0..N_HARMFUL-1 are the held-out eval split (the fits use 20+).
harmful() { # length, output stem
    $PY exp.py "${GATE[@]}" --source wj_unsafe --n "$N_HARMFUL" \
        --steps "$1" --gen-length "$1" --out "outputs/$2.json"
}
benign() { # source, output stem
    $PY exp.py "${GATE[@]}" --source "$1" --n "$N_BENIGN" \
        --steps 128 --gen-length 128 --out "outputs/$2.json"
}

say "generating across GPUs $GPUS"

fail=0
harmful 128 gated_len128             > log/gated_len128.log 2>&1 || fail=1
harmful 256 gated_len256             > log/gated_len256.log 2>&1 || fail=1
benign xstest_safe gated_or30_xstest > log/gated_xstest.log 2>&1 || fail=1
benign jbb_benign  gated_or30_jbb    > log/gated_jbb.log 2>&1 || fail=1
benign truthfulqa  gated_or30_tqa    > log/gated_tqa.log 2>&1 || fail=1
say "gate open rates"
for f in gated_len128 gated_len256 gated_xstest gated_jbb gated_tqa; do
    printf '  %-16s ' "$f"
    grep -oE "gate opened on [0-9]+/[0-9]+" "log/$f.log" \
        || { grep -oE "IndexError|Traceback|Error" "log/$f.log" | head -1 || echo "-"; }
done

[ $fail -eq 0 ] || { echo "a generation job failed; see log/gated_*.log" >&2; exit 1; }
