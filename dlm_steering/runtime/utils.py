"""Serialization and numerical helpers shared by experiments."""
import json
import math
from pathlib import Path


def atomic_write_json(path, value):
    """Replace a JSON artifact only after its full contents have been written."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def write_json(path, payload, *, compact=False):
    with open(path, "w", encoding="utf-8") as f:
        if compact:
            # dumps uses the C encoder for compact JSON. A single write also
            # avoids streaming thousands of tiny fragments for gate traces.
            f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        else:
            json.dump(payload, f, ensure_ascii=False, indent=2)


def step_scale(schedule, i, n_steps):
    """alpha multiplier at denoising step i of n_steps."""
    if schedule == "const":
        return 1.0
    frac = i / max(1, n_steps - 1)
    if schedule == "linear":
        return 1.0 - frac
    if schedule == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * frac))
    raise ValueError(schedule)


def auroc(pos, neg):
    """Rank-based AUROC of pos scoring above neg."""
    import numpy as np

    x = np.concatenate([pos, neg])
    order = x.argsort()
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    # Average ranks over ties so exact duplicates score 0.5, not 0 or 1.
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    n_p, n_n = len(pos), len(neg)
    return (ranks[:n_p].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n)
