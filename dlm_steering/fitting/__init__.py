"""Vector, detector and threshold fitting for the steering defense, plus the
refusal / TruthfulQA judges and the Dream over-refusal aggregation.

Members are entry points, run from the repo root as modules so their
absolute imports of common.py / Evaluator.py resolve:

    python -m dlm_steering.fitting.fit_vector
    python -m dlm_steering.fitting.judge_refusal --in ... --out ...
"""
