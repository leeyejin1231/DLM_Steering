# 대화형 실행 결과

상태: running

전체 설정과 실행 이력: [plan.json](plan.json)

## 1. run_sr_eval.py

상태: pending

```bash
env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/work/DLM_Steering/.venv/bin/python run_sr_eval.py --in /home/work/DLM_Steering/outputs/interactive/20260920T105451Z_evaluation_aaf0989e/dija_template_input.json --out /home/work/DLM_Steering/outputs/interactive/20260920T105451Z_evaluation_aaf0989e/gptoss_dija_template.json --auto-server --gpus 0,1 --gpu 0 --model gpt-oss:20b --workers 4 --reasoning-effort low --num-predict 4096
```

DIJA 평가 범위: assistant 제외 (템플릿만)
