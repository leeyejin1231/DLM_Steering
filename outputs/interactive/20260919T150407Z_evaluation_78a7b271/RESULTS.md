# 대화형 실행 결과

상태: interrupted

전체 설정과 실행 이력: [plan.json](plan.json)

## 1. eval_llamaguard.py

상태: interrupted

```bash
env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/work/DLM_Steering/.venv/bin/python eval_llamaguard.py --in /home/work/DLM_Steering/outputs/interactive/20260919T145838Z_experiment_6ce806f7/jbb_harmful_prefix_none_seed42.json --out /home/work/DLM_Steering/outputs/interactive/20260919T150407Z_evaluation_78a7b271/lg4.json --batch-size 16 --gpus 0,1,2,3,4,5,6,7
```

## 2. run_sr_eval.py

상태: pending

```bash
env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/work/DLM_Steering/.venv/bin/python run_sr_eval.py --in /home/work/DLM_Steering/outputs/interactive/20260919T145838Z_experiment_6ce806f7/jbb_harmful_prefix_none_seed42.json --out /home/work/DLM_Steering/outputs/interactive/20260919T150407Z_evaluation_78a7b271/gptoss.json --auto-server --gpus 0,1,2,3,4,5,6,7 --gpu 0 --model gpt-oss:20b --workers 4 --reasoning-effort low --num-predict 4096
```
