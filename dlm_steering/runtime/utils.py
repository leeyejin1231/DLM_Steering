import json
import math
import numpy as np
import torch
from pathlib import Path


def atomic_write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def write_json(path, payload, *, compact=False):
    with open(path, "w", encoding="utf-8") as f:
        if compact:
            f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        else:
            json.dump(payload, f, ensure_ascii=False, indent=2)


def step_scale(schedule, i, n_steps):
    if schedule == "const":
        return 1.0
    frac = i / max(1, n_steps - 1)
    if schedule == "linear":
        return 1.0 - frac
    if schedule == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * frac))
    raise ValueError(schedule)


def auroc(pos, neg):
    x = np.concatenate([pos, neg])
    order = x.argsort()
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    n_p, n_n = len(pos), len(neg)
    return (ranks[:n_p].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n)


def fit_logistic(X, y, C, steps=400, balanced=False):
    X = X.to(torch.float64)
    y = y.to(torch.float64)
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")
    lam = 1.0 / (C * len(X))
    pos_weight = torch.tensor([(len(y) - y.sum()) / y.sum()], dtype=torch.float64) if balanced and y.sum() > 0 else None

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y, pos_weight=pos_weight) + 0.5 * lam * (w @ w)
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().to(torch.float32), float(b.detach())
