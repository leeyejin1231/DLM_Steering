# 대화형 실행 결과

상태: complete

전체 설정과 실행 이력: [plan.json](plan.json)

## 1. interface.py

상태: complete

```bash
env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/work/DLM_Steering/.venv/bin/python interface.py --pap-generate --source strongreject --seed 42 --pap-model Qwen/Qwen3-14B --gpus 0,1,2,3,4,5,6,7 --out /home/work/DLM_Steering/data/attacks/pap_better/strongreject/seed42.json --reproduct
```

## 2. interface.py

상태: complete

```bash
env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/work/DLM_Steering/.venv/bin/python interface.py --pap-generate --source strongreject --seed 43 --pap-model Qwen/Qwen3-14B --gpus 0,1,2,3,4,5,6,7 --out /home/work/DLM_Steering/data/attacks/pap_better/strongreject/seed43.json --reproduct
```

## 3. interface.py

상태: complete

```bash
env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/work/DLM_Steering/.venv/bin/python interface.py --pap-generate --source strongreject --seed 44 --pap-model Qwen/Qwen3-14B --gpus 0,1,2,3,4,5,6,7 --out /home/work/DLM_Steering/data/attacks/pap_better/strongreject/seed44.json --reproduct
```
