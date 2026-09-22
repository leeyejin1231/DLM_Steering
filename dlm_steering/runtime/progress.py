"""One progress display per command tree; child processes keep their logs."""
import os

CHILD_PROGRESS_ENV = {"DLM_PROGRESS_OWNER": "parent", "TQDM_DISABLE": "1"}


def task_progress(*args, **kwargs):
    # Keep the launcher importable before it re-execs into the project venv.
    from tqdm import tqdm

    kwargs.setdefault("disable", bool(os.environ.get("DLM_PROGRESS_OWNER")))
    kwargs.setdefault("dynamic_ncols", True)
    return tqdm(*args, **kwargs)
