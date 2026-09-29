#!/usr/bin/env bash
# Shared environment for every script in this folder. Source it, do not run it.
#
#   PY   The project venv: Python 3.12 + `uv pip install -r requirements.txt`
#        (torch 2.3.1 / transformers 4.55.4), as set up in README "1. Environment".
#        Other interpreters ship different transformers versions, where LLaDA's
#        trust_remote_code class dies with "LLaDAModelLM object has no attribute
#        all_tied_weights_keys" -- and mixing environments invalidates
#        comparisons against results produced under this one. Override with
#        PY=... only if that env matches requirements.txt.
#   PYLG Interpreter for Llama Guard 4 grading (default: $PY).
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
    echo "no interpreter at $PY -- create the venv (README '1. Environment':" >&2
    echo "  uv venv --python 3.12 .venv && uv pip install -p .venv/bin/python -r requirements.txt" >&2
    echo "or point PY= at an env matching requirements.txt" >&2
    exit 1
fi
export PYLG="${PYLG:-$PY}"

LEGACY_HF_CACHE=/mnt/shared/huggingface-cache
if [ -z "${HF_HOME:-}" ] && [ -d "$LEGACY_HF_CACHE" ]; then
    export HF_HOME="$LEGACY_HF_CACHE"
fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false

# Cards `--gpus` may shard a prompt set across: one subprocess per id over a
# contiguous slice, merged on exit. Generation scales close to linearly
# (measured 3.8x on 4 cards) and per-row seeding makes the result independent
# of how the set is partitioned, so the default is every card this shell may
# use. A single id costs nothing extra -- exp.py runs that case inline.
#
# CUDA_VISIBLE_DEVICES wins when set: on a shared box `CUDA_VISIBLE_DEVICES=2
# script/run_benchmark.sh` must stay on card 2, and nvidia-smi does not honour
# it. Override either way with `GPUS=0,1 script/run_benchmark.sh`.
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

# Generation workers per card (1, 2 or auto = by free memory). Generations are
# identical either way; one worker already saturates the GPU on ~330-token
# rows and a second measured ~10% slower, so 2 only pays on short prompts.
export PROCS_PER_GPU="${PROCS_PER_GPU:-1}"

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
export VECTOR="${VECTOR:-outputs/steer_vector.pt}"
export DETECTOR="${DETECTOR:-outputs/steer_detector.pt}"

mkdir -p outputs log

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

require() {
    for f in "$@"; do
        [ -e "$f" ] || { echo "missing: $f -- run script/build_vectors.sh first" >&2; exit 1; }
    done
}

# ------------------------------------------------ shared by the run_*.sh drivers
# MODEL selects the target (llada, dream, llada1.5) and its output folder.
model_setup() {
    export MODEL="${MODEL:-llada}"
    case "$MODEL" in
        llada)    MODEL_OUT=outputs ;;
        dream)    MODEL_OUT=outputs/dream ;;
        llada1.5) MODEL_OUT=outputs/llada1.5 ;;
        *) echo "unknown MODEL=$MODEL (llada|dream|llada1.5)" >&2; exit 1 ;;
    esac
    export MODEL_OUT
    mkdir -p "$MODEL_OUT"
}

# The deployed `ours` settings per model. Every driver builds its defense
# flags from here, so a change to the method is made once.
#   LLaDA-8B  steering layer 25, gate detector layer 18 / threshold 7.0,
#             response detector outputs/response_detector.pt (layer 18) / cutoff 0.12
#   Dream     steering layer 20, gate at the detector bundle's best layer (14,
#             gate_threshold.json), response detector
#             outputs/dream/response_detector3_committed_L20.pt (its own layer 20 / cutoff 0.387)
# Both: alpha 1, adaptive steering, V3 remask with an 80% random prompt remask.
OURS_POLICY="--steer adaptive --remask v3 --remask-prompt --remask-prompt-frac 0.8"
ours_args() { # MODEL -> checkpoint/layer flags
    case "$1" in
        llada) echo "--layer 25 --detector-layer 18 --gate-threshold 7.0 --alpha $ALPHA" \
                    "--response-detector outputs/response_detector.pt --response-threshold 0.12" ;;
        dream) echo "--layer 20 --alpha $ALPHA" \
                    "--response-detector outputs/dream/response_detector3_committed_L20.pt" ;;
        *)     echo "--alpha $ALPHA" ;;
    esac
}
defense_args() { # MODEL DEFENSE -> exp.py flags
    case "$2" in
        ours) echo "--defense ours $OURS_POLICY $(ours_args "$1")" ;;
        diffuguard|none|selfreminder) echo "--defense $2" ;;
        *) echo "unknown DEFENSE=$2 (ours|diffuguard|none|selfreminder)" >&2; return 1 ;;
    esac
}

# Output-name prefix and row count of a --source.
source_prefix() {
    case "$1" in
        jbb_harmful) echo JBB ;;         harmbench) echo HB ;;          strongreject) echo SR ;;
        advbench) echo AdvBench ;;       xstest_safe) echo XSTest-safe ;; xstest_unsafe) echo XSTest-unsafe ;;
        truthfulqa) echo TQA ;;          truthfulqa_mc) echo TQAmc ;;   math500) echo MATH500 ;;
        gsm8k) echo GSM8K ;;             mmlu) echo MMLU ;;             jbb_benign) echo JBB-benign ;;
        wj_benign) echo WJ-benign ;;     wj_unsafe) echo WJ-unsafe ;;
        *) echo "unknown source $1" >&2; return 1 ;;
    esac
}
source_rows() { # N overrides the full row count
    if [ -n "${N:-}" ]; then echo "$N"; else
        "$PY" -c "from common import load_prompts; print(len(load_prompts('$1')))"
    fi
}

# complete_json FILE N: succeeds when FILE already holds N result rows.
complete_json() {
    [ -s "$1" ] && "$PY" -c 'import json, sys
sys.exit(0 if len(json.load(open(sys.argv[1]))["results"]) == int(sys.argv[2]) else 1)' "$1" "$2"
}

# The ollama containers the sharded graders start: one per shard, ollama-50001+i.
stop_ollama() {
    local i names=""
    for ((i = 0; i < ${#_gpu_ids[@]}; i++)); do names="$names ollama-$((50001 + i))"; done
    podman stop $names >/dev/null 2>&1 || true   # may already be down; -e is set
}

# grade_files FILE...: Llama Guard 4 (ASR) then StrongREJECT (gpt-oss:20b via
# ollama) for every generation file that lacks the grade, then one report.
grade_files() {
    local F LG SR
    say "Llama Guard 4 ($(date '+%F %T'))"
    for F in "$@"; do
        LG=${F%.json}_lg4.json
        [ -s "$F" ] || { echo "missing $F"; continue; }
        [ -s "$LG" ] && continue
        "$PYLG" eval_llamaguard.py --in "$F" --out "$LG" --gpus "$GPUS" \
            > "log/lg4_$(basename "${F%.json}").log" 2>&1 || echo "LG4 FAILED: $F"
    done
    say "StrongREJECT / gpt-oss ($(date '+%F %T'))"
    for F in "$@"; do
        SR=${F%.json}_sr.json
        [ -s "$F" ] || continue
        [ -s "$SR" ] && continue
        "$PY" run_sr_eval.py --in "$F" --out "$SR" --gpus "$GPUS" \
            > "log/sr_$(basename "${F%.json}").log" 2>&1 || echo "SR FAILED: $F"
    done
    stop_ollama
    say "report"
    "$PY" script/report.py $(for F in "$@"; do ls "${F%.json}_lg4.json" "${F%.json}_sr.json" 2>/dev/null; done)
}
