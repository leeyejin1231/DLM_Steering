# 대화형 실행 결과

상태: running

전체 설정과 실행 이력: [plan.json](plan.json)

## 1. exp.py

상태: pending

```bash
env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/work/DLM_Steering/.venv/bin/python exp.py --model llada1.5 --attack dija --defense diffuguard --source jbb_harmful --start 0 --n 100 --gen-length 128 --steps 128 --block-length 32 --temperature 0.2 --gpus 1 --reproduct --remasking adaptive_step --repair-scope all --seed 43 --out /home/work/DLM_Steering/outputs/interactive/matrix_llada1.5_jbb_harmful_dija_diffuguard_seed43/llada1.5_jbb_harmful_dija_diffuguard_seed43.json
```
