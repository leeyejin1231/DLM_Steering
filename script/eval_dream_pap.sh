#!/bin/bash
# Grade the Dream PAP runs from script/run_dream_pap.sh: Llama Guard 4 (ASR) and
# StrongREJECT (gpt-oss:20b over ollama), both sharded over $GPUS. Llama Guard 4
# runs under $PYLG (transformers 5.x), as in script/run_ablation_dija_jbb.sh.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/dream_common.sh"
export HF_HOME="${HF_HOME:-/mnt/shared/huggingface-cache}" TOKENIZERS_PARALLELISM=false
PYLG=${PYLG:-/home/yejin/anaconda3/envs/kotox/bin/python}
SOURCES=${SOURCES:-jbb_harmful}

FILES=()
for SRC in $SOURCES; do
  case $SRC in jbb_harmful) P=JBB ;; harmbench) P=HB ;; strongreject) P=SR ;; *) echo "unknown source $SRC"; exit 1 ;; esac
  for SEED in $SEEDS; do FILES+=("$DREAM_OUT/$P-pap-$TAG-$SEED.json"); done
done

say "Llama Guard 4 ($(date '+%F %T'))"
for F in "${FILES[@]}"; do
  LG=${F%.json}_lg4.json
  [ -s "$F" ] || { echo "missing $F"; continue; }
  [ -s "$LG" ] && continue
  HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" \
    "$PYLG" eval_llamaguard.py --in "$F" --out "$LG" --gpus "$GPUS" > "log/lg4_$(basename "${F%.json}").log" 2>&1 \
    || echo "LG4 FAILED: $F"
  grep -E '"asr"' "$LG" 2>/dev/null | head -1
done

say "StrongREJECT / gpt-oss ($(date '+%F %T'))"
for F in "${FILES[@]}"; do
  SR=${F%.json}_sr.json
  [ -s "$F" ] || continue
  [ -s "$SR" ] && continue
  "$PY" run_sr_eval.py --in "$F" --out "$SR" --gpus "$GPUS" > "log/sr_$(basename "${F%.json}").log" 2>&1 \
    || echo "SR FAILED: $F"
  grep -E '"asr"' "$SR" 2>/dev/null | head -1
done
podman stop $(for g in ${GPUS//,/ }; do echo -n "ollama-$((50001 + g)) "; done) >/dev/null 2>&1 || true   # containers may already be down; common.sh sets -e

say "report"
"$PY" script/report.py $(for F in "${FILES[@]}"; do ls "${F%.json}_lg4.json" "${F%.json}_sr.json" 2>/dev/null; done)
echo "ALL PAP EVAL DONE $(date '+%F %T')"
