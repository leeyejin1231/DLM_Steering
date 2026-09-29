"""Over-refusal grading entry point for interface.py (XSTest 3-way rubric).

Same CLI as `python -m dlm_steering.fitting.judge_refusal`; a top-level script so the
interactive launcher records it like eval_llamaguard.py and run_sr_eval.py.

Usage:
    python eval_refusal.py --in <xstest_safe or truthfulqa generations> --out <judged.json> \
                           --auto-server --gpus 0,1
"""
from dlm_steering.fitting.judge_refusal import main

if __name__ == "__main__":
    main()
