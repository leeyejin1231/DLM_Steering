# 대화형 실행 결과

상태: complete

전체 설정과 실행 이력: [plan.json](plan.json)

## 1. exp.py

상태: complete

```bash
env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/work/DLM_Steering/.venv/bin/python exp.py --model llada1.5 --attack dija --defense ours --source harmbench --start 0 --n 393 --gen-length 128 --steps 128 --block-length 32 --temperature 0.2 --gpus 7 --procs-per-gpu 1 --reproduct --alpha 1.0 --remask v3 --remask-prompt --seed 42 --out /home/work/DLM_Steering/outputs/interactive/20260919T212123Z_experiment_104dfe4f/llada1.5_harmbench_dija_ours_seed42.json
```

## 2. exp.py

상태: complete

```bash
env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/work/DLM_Steering/.venv/bin/python exp.py --model llada1.5 --attack dija --defense ours --source harmbench --start 0 --n 393 --gen-length 128 --steps 128 --block-length 32 --temperature 0.2 --gpus 7 --procs-per-gpu 1 --reproduct --alpha 1.0 --remask v3 --remask-prompt --seed 43 --out /home/work/DLM_Steering/outputs/interactive/20260919T212123Z_experiment_104dfe4f/llada1.5_harmbench_dija_ours_seed43.json
```

## 3. exp.py

상태: complete

```bash
env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/work/DLM_Steering/.venv/bin/python exp.py --model llada1.5 --attack dija --defense ours --source harmbench --start 0 --n 393 --gen-length 128 --steps 128 --block-length 32 --temperature 0.2 --gpus 7 --procs-per-gpu 1 --reproduct --alpha 1.0 --remask v3 --remask-prompt --seed 44 --out /home/work/DLM_Steering/outputs/interactive/20260919T212123Z_experiment_104dfe4f/llada1.5_harmbench_dija_ours_seed44.json
```
