#!/usr/bin/env bash
# Shared environment for every script in this folder. Source it, do not run it.
#
# Two settings here are not optional and are the usual cause of a failed run:
#
#   PY   Depending on the shell session, `python3` may resolve to the kotox2
#        conda env (transformers 5.9), where LLaDA's trust_remote_code class
#        dies with "LLaDAModelLM object has no attribute all_tied_weights_keys".
#        The base env (transformers 4.57.1 / torch 2.8.0) is also the one every
#        baseline and vector was produced under, so mixing them invalidates the
#        comparisons as well as crashing.
#
#   LD_PRELOAD  Without it, `import transformers` fails with
#        "GLIBCXX_3.4.29 not found" -- the system libstdc++ stops at 3.4.25 but
#        scipy's _ckdtree extension needs 3.4.29.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PY="${PY:-/home/yejin/anaconda3/bin/python}"
export LD_PRELOAD="${LD_PRELOAD:-/home/yejin/anaconda3/lib/libstdc++.so.6}"
export HF_HOME="${HF_HOME:-/mnt/shared/huggingface-cache/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/mnt/shared/huggingface-cache/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# Two A6000s by default; set GPU_A=GPU_B=0 to run everything on one card.
export GPU_A="${GPU_A:-0}"
export GPU_B="${GPU_B:-1}"

# Method hyper-parameters. The actuator layer and alpha come from the fit-split
# sweep; the detector layer from the length-controlled AUROC check; the
# threshold is read from outputs/gate_threshold.json unless overridden.
export ALPHA="${ALPHA:-1.0}"
export DETECTOR_LAYER="${DETECTOR_LAYER:-18}"
export N_HARMFUL="${N_HARMFUL:-20}"
export N_BENIGN="${N_BENIGN:-30}"

export VECTOR="${VECTOR:-outputs/steer_vector.pt}"
export DETECTOR="${DETECTOR:-outputs/steer_detector.pt}"

mkdir -p outputs log

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

require() {
    for f in "$@"; do
        [ -e "$f" ] || { echo "missing: $f -- run script/build_vectors.sh first" >&2; exit 1; }
    done
}
