#!/usr/bin/env bash
# v3 tuning with the real DIJA attack (the paper's refined prompts, gen_length 0)
# on each dataset DIJA ships prompts for. v3 only -- v2 is not part of the
# experiments. Every command is appended to outputs/tune/commands.log.
#
#   PHASE=boundary  where v3 may trigger: checkpoints every 32 committed mask
#                   slots (--infill-checkpoint 32), boundary k = after 32*(k+1)
#                   slots, plus the old end-of-infilling audit as reference.
#   PHASE=ref       steer-only and no-defense references for the same runs.
#   PHASE=rest      temperature (3 seeds) and recovery steps x rounds, at the
#                   boundary chosen from the first phase: BOUNDARY=<k|end>.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=.venv/bin/python
PHASE=${PHASE:-boundary}

# Batch-1 decoding leaves the GPU idle ~60% of each step waiting on Python to
# launch kernels, so two processes per card fill that gap: measured on JBB
# (100 rows, 8x A6000 40GB) one per card 172s, two per card 117s, identical
# generations (per-row seeding makes the shard layout irrelevant). LLaDA-8B
# takes ~16GB, so PROCS_PER_GPU=2 needs 40GB cards -- use 1 on 24GB ones.
# Without an OMP cap each process spun 450 threads at ~830% CPU for no gain.
PROCS_PER_GPU=${PROCS_PER_GPU:-2}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} \
       OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
if [ -z "${GPUS:-}" ]; then
    CARDS=${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)}
    GPUS=""
    for g in ${CARDS//,/ }; do
        for _ in $(seq "$PROCS_PER_GPU"); do GPUS="${GPUS:+$GPUS,}$g"; done
    done
fi
OUT=outputs/tune
LOG=$OUT/commands.log
mkdir -p "$OUT"
declare -A PFX=( [jbb_harmful]=JBB [harmbench]=HarmBench [strongreject]=SR )

run() { # stem, args...
    local stem=$1; shift
    local f="$OUT/$stem.json"
    printf '%s exp.py %s --gpus %s --out %s\n' "$PY" "$*" "$GPUS" "$f" >> "$LOG"
    [ -s "$f" ] && { echo "  skip $stem"; return 0; }
    $PY exp.py "$@" --gpus "$GPUS" --out "$f" > "$OUT/$stem.log" 2>&1 \
        && echo "  ok   $stem" || { echo "  FAIL $stem"; tail -3 "$OUT/$stem.log"; }
}

audit_args() { # k|end -> v3 audit placement flags
    if [ "$1" = end ]; then echo "--infill-checkpoint 0"
    else echo "--infill-checkpoint 32 --audit-boundary $1"; fi
}

for S in jbb_harmful harmbench strongreject; do
    P=${PFX[$S]}
    COMMON=(--attack dija --defense ours --steer adaptive --remask v3
            --source "$S" --n 600 --reproduct)

    if [ "$PHASE" = ref ]; then
        # References for reading the boundary sweep: the same attack and
        # sampling with no recovery at all (steer-only) and with no defense.
        echo "== $P: references (T=0.2) =="
        run "dija-$P-steeronly" --attack dija --defense ours --steer adaptive \
            --remask none --source "$S" --n 600 --reproduct --temperature 0.2 --seed 42
        run "dija-$P-nodef" --attack dija --defense none \
            --source "$S" --n 600 --reproduct --temperature 0.2 --seed 42
    elif [ "$PHASE" = boundary ]; then
        echo "== $P: audit boundary (T=0.2) =="
        for B in 0 1 2 3 end; do
            run "dija-$P-bnd$B" "${COMMON[@]}" $(audit_args "$B") \
                --temperature 0.2 --seed 42
        done
    else
        B=${BOUNDARY:?set BOUNDARY=<k|end> for PHASE=rest}
        echo "== $P: temperature @ boundary $B (seeds 42/43/44) =="
        for T in 0 0.2 0.5 1.0; do
            for SEED in 42 43 44; do
                run "dija-$P-bnd$B-temp$T-seed$SEED" "${COMMON[@]}" $(audit_args "$B") \
                    --temperature "$T" --seed "$SEED"
            done
        done
        echo "== $P: recovery steps x rounds @ boundary $B (T=0.2) =="
        for R in 16 32 64; do
            for N in 1 2 3; do
                run "dija-$P-bnd$B-rec-s$R-n$N" "${COMMON[@]}" $(audit_args "$B") \
                    --recovery-steps "$R" --recovery-rounds "$N" --temperature 0.2 --seed 42
            done
        done
    fi
done

echo "== grading =="
for f in "$OUT"/dija-*.json; do
    case "$f" in *_lg4.json|*.part*) continue;; esac
    g="${f%.json}_lg4.json"; [ -s "$g" ] && continue
    printf 'CUDA_VISIBLE_DEVICES=0 %s eval_llamaguard.py --in %s --out %s\n' "$PY" "$f" "$g" >> "$LOG"
    CUDA_VISIBLE_DEVICES=0 $PY eval_llamaguard.py --in "$f" --out "$g" > "${f%.json}_lg4.log" 2>&1 \
        && echo "  ok   $(basename "${f%.json}")" || echo "  FAIL $(basename "${f%.json}")"
done
echo "== done =="
