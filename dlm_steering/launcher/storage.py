"""Atomic launch metadata and shared subprocess environment."""
from dlm_steering.runtime.utils import atomic_write_json as save


WORKER_ENV = {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}
