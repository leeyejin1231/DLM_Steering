# 대화형 실행 결과

상태: running

전체 설정과 실행 이력: [plan.json](plan.json)

## 1. run_sr_eval.py

상태: pending

```bash
env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/work/DLM_Steering/.venv/bin/python run_sr_eval.py --in /home/work/DLM_Steering/outputs/interactive/20260919T212054Z_experiment_7a5aea40/llada1.5_harmbench_prefix_ours_seed43.json --out /home/work/DLM_Steering/outputs/interactive/20260920T105717Z_evaluation_9daa7800/gptoss.json --auto-server --gpus 4,5 --gpu 4 --model gpt-oss:20b --workers 4 --reasoning-effort low --num-predict 4096
```
