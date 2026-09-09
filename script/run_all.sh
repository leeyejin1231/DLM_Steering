#!/usr/bin/env bash
# Full pipeline: build the vectors if missing, generate with the gate, score.
# ~40 min for the vectors (skipped when they exist) + ~17 min generate + ~12 min score.
#
#   script/run_all.sh              reuse existing vectors
#   script/run_all.sh --force      rebuild vectors from scratch
#   GPU_A=0 GPU_B=0 script/run_all.sh    single GPU (slower)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

"$ROOT/script/build_vectors.sh" "$@"
"$ROOT/script/run_gated.sh"
"$ROOT/script/evaluate.sh"
