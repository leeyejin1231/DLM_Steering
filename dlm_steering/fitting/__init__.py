"""Vector, detector and threshold fitting for the steering defense, plus the
refusal / TruthfulQA judges and the Dream over-refusal aggregation.

Members are entry points, run from the repo root as modules so that the
repository root is on sys.path for `dlm_steering` and the top-level modules
(sampler.py, models.py, ollama_runtime.py):

    python -m dlm_steering.fitting.fit_vector
    python -m dlm_steering.fitting.judge_refusal --in ... --out ...
"""
