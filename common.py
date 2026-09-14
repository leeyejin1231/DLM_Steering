"""Shared constants, model loading and IO helpers for the steering experiments.

MODEL_NAME and MASK_ID stay canonically in llada.py (the reference sampler);
everything else shared lives here.
"""

import glob
import json
import math
from pathlib import Path

import numpy as np
import torch

from llada import MODEL_NAME, MASK_ID  # noqa: F401  (re-exported)

EOT_ID = 126348   # <|eot_id|>, closes the user turn in LLaDA's chat template
NEWLINE_ID = 198  # '\n'; used to locate the DIJA template inside the prompt

JBB_HARMFUL_GLOB = ("/mnt/shared/huggingface-cache/hub/datasets--JailbreakBench--JBB-Behaviors"
                    "/snapshots/*/data/harmful-behaviors.csv")


def load_llada(device=None):
    """LLaDA tokenizer and eval-mode bf16 model on `device`."""
    from transformers import AutoModel, AutoTokenizer
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True,
                                      dtype=torch.bfloat16).to(device).eval()
    return tokenizer, model


def encode_prompt(tokenizer, user_message, device):
    """Chat-template a user message; return (1, L) token ids on `device`."""
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        add_generation_prompt=True, tokenize=False)
    return torch.tensor(tokenizer(formatted)["input_ids"], device=device).unsqueeze(0)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
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


def load_prompts(source, csv_path):
    """Return a list of {"index", "prompt", "target"} rows."""
    import pandas as pd
    if source == "csv":
        df = pd.read_csv(csv_path)
        return [{"index": int(i), "prompt": str(r["prompt"]), "target": None}
                for i, r in df.iterrows()]
    if source == "jbb_harmful":
        df = pd.read_csv(glob.glob(JBB_HARMFUL_GLOB)[0])
        return [{"index": int(r["Index"]), "prompt": str(r["Goal"]), "target": str(r["Target"])}
                for _, r in df.iterrows()]
    raise ValueError(source)


def load_detector(detector_path, layer, device, threshold=None):
    """Detector bundle vector on `device` + gate threshold.

    threshold=None reads the gate_threshold.json sitting next to the bundle.
    """
    db = torch.load(detector_path, map_location="cpu")
    det_vec = db["vector"][db["layers"].index(layer)].to(device)
    if threshold is None:
        threshold = json.loads(
            (Path(detector_path).parent / "gate_threshold.json").read_text())["threshold"]
    return det_vec, threshold
