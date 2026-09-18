#!/usr/bin/env bash
# v3 tuning with the real DIJA attack (the paper's refined prompts, gen_length 0)
# on each dataset DIJA ships prompts for, using v3 recovery. Every command is appended to outputs/tune/commands.log.
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
GEN_JOBS="$OUT/jobs-dija-$PHASE.json"
printf '[]\n' > "$GEN_JOBS"
declare -A PFX=( [jbb_harmful]=JBB [harmbench]=HarmBench [strongreject]=SR )

run() { # stem, args...
    local stem=$1; shift
    local f="$OUT/$stem.json"
    printf '%s exp.py %s --gpus %s --out %s\n' "$PY" "$*" "$GPUS" "$f" >> "$LOG"
    [ -s "$f" ] && { echo "  skip $stem"; return 0; }
    "$PY" - "$GEN_JOBS" "$@" --out "$f" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
jobs = json.loads(path.read_text())
jobs.append(sys.argv[2:])
path.write_text(json.dumps(jobs))
PY
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

printf '%s exp.py --jobs %s --gpus %s\n' "$PY" "$GEN_JOBS" "$GPUS" >> "$LOG"
"$PY" exp.py --jobs "$GEN_JOBS" --gpus "$GPUS" \
    || { echo "generation failed; see $OUT/.parts/ logs"; exit 1; }

echo "== grading =="
GRADE_JOBS="$OUT/jobs-dija-grade.json"
"$PY" - "$OUT" "$GRADE_JOBS" <<'PY'
import json, sys
from pathlib import Path
jobs = []
for source in sorted(Path(sys.argv[1]).glob("dija-*.json")):
    if source.name.endswith("_lg4.json") or ".part" in source.name:
        continue
    output = source.with_name(source.stem + "_lg4.json")
    if output.exists() and output.stat().st_size:
        continue
    jobs.append(["--in", str(source), "--out", str(output)])
Path(sys.argv[2]).write_text(json.dumps(jobs))
PY
# Reuse the grader across files. One GPU was faster for the small-file
# benchmark; GRADE_GPUS can assign whole files to additional distinct cards.
GRADE_GPUS=${GRADE_GPUS:-${GPUS%%,*}}
printf '%s eval_llamaguard.py --jobs %s --gpus %s\n' \
    "$PY" "$GRADE_JOBS" "$GRADE_GPUS" >> "$LOG"
"$PY" eval_llamaguard.py --jobs "$GRADE_JOBS" --gpus "$GRADE_GPUS" \
    || { echo "grading failed; see $OUT/*_lg4.log"; exit 1; }
echo "== done =="
