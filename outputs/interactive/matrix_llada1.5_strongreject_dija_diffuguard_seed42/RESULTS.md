# 대화형 실행 결과

상태: complete

전체 설정과 실행 이력: [plan.json](plan.json)

## 1. exp.py

상태: complete

```bash
env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/work/DLM_Steering/.venv/bin/python exp.py --model llada1.5 --attack dija --defense diffuguard --source strongreject --start 0 --n 313 --gen-length 128 --steps 128 --block-length 32 --temperature 0.2 --gpus 1 --reproduct --remasking adaptive_step --repair-scope all --seed 42 --out /home/work/DLM_Steering/outputs/interactive/matrix_llada1.5_strongreject_dija_diffuguard_seed42/llada1.5_strongreject_dija_diffuguard_seed42.json
```
