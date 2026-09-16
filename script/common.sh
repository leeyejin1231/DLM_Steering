#!/usr/bin/env bash
# Shared environment for every script in this folder. Source it, do not run it.
#
#   PY   The project venv: Python 3.12 + `uv pip sync requirements.lock`
#        (torch 2.3.1 / transformers 4.55.4), as set up in README "0. 환경 세팅".
#        Other interpreters on a given machine ship different transformers
#        versions, where LLaDA's trust_remote_code class dies with
#        "LLaDAModelLM object has no attribute all_tied_weights_keys" -- and
#        mixing environments invalidates comparisons against results produced
#        under this one. Override with PY=... only if that env matches the lock.
#
#   LD_PRELOAD  Unset by default. Some conda interpreters need a newer
#        libstdc++ preloaded ("GLIBCXX_3.4.29 not found" on `import
#        transformers`); the .venv interpreter does not.
#
#   HF_HOME  Left alone unless the legacy shared cache mount is present.
#        common.hf_glob searches $HF_HOME, ~/.cache/huggingface and that mount
#        in turn, so an unset HF_HOME resolves fine on its own.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PY="${PY:-$ROOT/.venv/bin/python}"
if [ ! -x "$PY" ]; then
    echo "no interpreter at $PY -- create the venv (README '0. 환경 세팅':" >&2
    echo "  uv venv --python 3.12 .venv && uv pip sync -p .venv/bin/python requirements.lock" >&2
    echo "or point PY= at an env matching requirements.lock" >&2
    exit 1
fi

LEGACY_HF_CACHE=/mnt/shared/huggingface-cache
if [ -z "${HF_HOME:-}" ] && [ -d "$LEGACY_HF_CACHE" ]; then
    export HF_HOME="$LEGACY_HF_CACHE"
fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# Cards `--gpus` may shard a prompt set across: one subprocess per id over a
# contiguous slice, merged on exit. Generation scales close to linearly
# (measured 3.8x on 4 cards) and per-row seeding makes the result independent
# of how the set is partitioned, so the default is every card this shell may
# use. A single id costs nothing extra -- exp.py runs that case inline.
#
# CUDA_VISIBLE_DEVICES wins when set: on a shared box `CUDA_VISIBLE_DEVICES=2
# script/run_gated.sh` must stay on card 2, and nvidia-smi does not honour it.
# Override either way with `GPUS=0,1 script/run_gated.sh`.
if [ -z "${GPUS:-}" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        GPUS="$CUDA_VISIBLE_DEVICES"
    else
        GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null \
                | paste -sd, - || true)"
        [ -n "$GPUS" ] || GPUS="0"
    fi
fi
export GPUS

# Cards for the jobs that want exactly one. Taken from $GPUS rather than fixed
# at 0 and 1, so `CUDA_VISIBLE_DEVICES=4,5 script/...` keeps them on 4 and 5
# instead of reaching for cards this shell was told not to touch.
_gpu_ids=(${GPUS//,/ })
export GPU_A="${GPU_A:-${_gpu_ids[0]}}"
export GPU_B="${GPU_B:-${_gpu_ids[1]:-${_gpu_ids[0]}}}"

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
