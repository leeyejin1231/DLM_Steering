#!/bin/bash
# DiffuGuard baseline: the official runner (DiffuGuard/models/jailbreakbench_llada.py,
# unmodified) against DIJA under the SAME sampling conditions as our
# `exp.py --attack dija` runs: no assistant tail (gen_length 0), temperature 0.2,
# no CFG, low-confidence remasking, one mask per step (steps 200 >= the largest
# mask count). Defense settings are the LLaDA-8B DIJA line of DiffuGuard/test.sh:
# hidden self-check 0.2, remask 90%, 8 refinement steps, fill_all_masks.
#
#   SOURCE=jbb_harmful|harmbench|strongreject   which DIJA refined prompt set
#   CONFIG=hidden|full                          full = hidden + --remasking adaptive_step
#   GPU=<index>
#
# Output: outputs/<JBB|HarmBench|SR>-dija-dgm-<CONFIG>-42{,_lg4,_sr}.json
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/script/diffuguard_stubs:${PYTHONPATH}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
GPU=${GPU:-1}
SOURCE=${SOURCE:-jbb_harmful}      # jbb_harmful | harmbench | strongreject
case "$SOURCE" in
    jbb_harmful)  PROMPTS=DIJA/run_jailbreakbench/refine_prompt/jailbreakbench_data_refined_Qwen.json; PREFIX=JBB ;;
    harmbench)    PROMPTS=DIJA/run_harmbench/refine_prompt/harmbench_behaviors_text_all_refined_Qwen.json; PREFIX=HarmBench ;;
    strongreject) PROMPTS=DIJA/run_strongreject/refine_prompt/strongreject_data_refined_Qwen.json; PREFIX=SR ;;
    *) echo "unknown SOURCE=$SOURCE"; exit 1 ;;
esac
BASE=(--model_path GSAI-ML/LLaDA-8B-Instruct --attack_method DIJA --attack_prompt "$PROMPTS"
      --gen_length 0 --steps 200 --block_length 200 --temperature 0.2 --cfg_scale 0.0
      --sp_mode hidden --sp_threshold 0.2 --refinement_steps 8 --remask_ratio 0.9
      --fill_all_masks --no_auto_pick_gpu --debug_print)
CFG='{"steps": "200 (<=1 mask/step)", "gen_length": 0, "temperature": 0.2, "cfg_scale": 0.0, "fill_all_masks": true}'

CONFIG=${CONFIG:-hidden}
case "$CONFIG" in
    hidden) EXTRA=() ;;
    full)   EXTRA=(--remasking adaptive_step) ;;
    *) echo "unknown CONFIG=$CONFIG (hidden|full)"; exit 1 ;;
esac
TAG=dgm-$CONFIG
RAW=outputs/diffuguard_${PREFIX}_${TAG}_raw.json
OUT=outputs/${PREFIX}-dija-${TAG}-42.json

CUDA_VISIBLE_DEVICES=$GPU python DiffuGuard/models/jailbreakbench_llada.py "${BASE[@]}" "${EXTRA[@]}" \
    --output_json "$RAW" > "log/diffuguard_${PREFIX}_${TAG}.log" 2>&1
python script/convert_diffuguard.py --in "$RAW" --out "$OUT" --source "$SOURCE" --config "$CFG" \
    --defense "{\"defense\": \"diffuguard\", \"config\": \"$CONFIG\", \"sp_mode\": \"hidden\", \"sp_threshold\": 0.2, \"refinement_steps\": 8, \"remask_ratio\": 0.9, \"remasking\": \"$([ "$CONFIG" = full ] && echo adaptive_step || echo low_confidence)\"}"
CUDA_VISIBLE_DEVICES=$GPU python eval_llamaguard.py --in "$OUT" --out "${OUT%.json}_lg4.json"
python run_sr_eval.py --in "$OUT" --out "${OUT%.json}_sr.json" --port 50001 --gpu "$GPU"
echo "DONE $OUT"
