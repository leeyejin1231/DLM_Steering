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
JBB_BENIGN_GLOB = JBB_HARMFUL_GLOB.replace("harmful-", "benign-")
XSTEST_GLOB = ("/mnt/shared/huggingface-cache/hub/datasets--walledai--XSTest"
               "/snapshots/*/**/*.parquet")
WJ_EVAL_GLOB = ("/mnt/shared/huggingface-cache/datasets/allenai___wildjailbreak"
                "/eval-*/0.0.0/*/*.arrow")
TRUTHFULQA_GLOB = ("/mnt/shared/huggingface-cache/hub/datasets--domenicrosati--TruthfulQA"
                   "/snapshots/*/**/*.csv")
WALLEDAI_GLOB = ("/mnt/shared/huggingface-cache/hub/datasets--walledai--{}"
                 "/snapshots/*/**/*.parquet")


def load_llada(device=None):
    """LLaDA tokenizer and eval-mode bf16 model on `device`."""
    from transformers import AutoModel, AutoTokenizer
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True,
                                      dtype=torch.bfloat16).to(device).eval()
    return tokenizer, model


def prompt_token_ids(tokenizer, user_message):
    """Chat-template a user message; return the token id list."""
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        add_generation_prompt=True, tokenize=False)
    return tokenizer(formatted)["input_ids"]


def encode_prompt(tokenizer, user_message, device):
    """Chat-template a user message; return (1, L) token ids on `device`."""
    return torch.tensor(prompt_token_ids(tokenizer, user_message),
                        device=device).unsqueeze(0)


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


def load_eval_prompts(source, limit):
    """Benchmark prompt lists as plain strings.

    Sources: xstest_safe/xstest_unsafe, jbb_benign/jbb_harmful, advbench,
    harmbench, strongreject, truthfulqa, wj_benign.
    """
    import pandas as pd
    if source.startswith("xstest"):
        df = pd.read_parquet(glob.glob(XSTEST_GLOB, recursive=True)[0])
        # "xstest_unsafe".endswith("safe") is True, so match the suffix explicitly.
        want = "unsafe" if source.endswith("_unsafe") else "safe"
        df = df[df["label"] == want]
        prompts = df["prompt"].tolist()
    elif source == "jbb_benign":
        # Each benign behaviour is the index-matched counterpart of a harmful one
        # ("fictional story about heroin use" vs "defamatory article claiming a
        # president is addicted to heroin"), so topic is held roughly constant
        # and only permissibility differs.
        df = pd.read_csv(glob.glob(JBB_BENIGN_GLOB)[0])
        prompts = df["Goal"].tolist()
    elif source == "jbb_harmful":
        df = pd.read_csv(glob.glob(JBB_HARMFUL_GLOB)[0])
        prompts = df["Goal"].tolist()
    elif source in ("advbench", "harmbench", "strongreject"):
        # Out-of-distribution harmful benchmarks: bare requests in attack styles
        # the steering and detector directions never saw. Shards are concatenated
        # and de-duplicated because some of these ship several splits.
        name = {"advbench": "AdvBench", "harmbench": "HarmBench",
                "strongreject": "StrongREJECT"}[source]
        files = sorted(glob.glob(WALLEDAI_GLOB.format(name), recursive=True))
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        prompts = list(dict.fromkeys(df["prompt"].astype(str).tolist()))
    elif source == "truthfulqa":
        # Plain harmless questions: nothing here should ever be refused, so the
        # refusal rate is a direct read on utility damage rather than on
        # sensitivity to harmful-sounding surface words.
        df = pd.read_csv(glob.glob(TRUTHFULQA_GLOB, recursive=True)[0])
        prompts = df["Question"].tolist()
    elif source == "wj_benign":
        import pyarrow as pa
        f = glob.glob(WJ_EVAL_GLOB)[0]
        with pa.memory_map(f) as src:
            ev = pa.ipc.open_stream(src).read_all().to_pandas()
        prompts = ev[ev["data_type"] == "adversarial_benign"]["adversarial"].tolist()
    else:
        raise ValueError(source)
    return prompts[:limit] if limit else prompts


def load_detector_bundle(detector_path):
    """The raw detector checkpoint dict (vector, layers, gen_length, ...)."""
    return torch.load(detector_path, map_location="cpu")


def load_detector(detector_path, layer=None, device=None, threshold=None):
    """Detector vector + resolved layer + gate threshold.

    layer=None uses the bundle's best_layer; threshold=None reads the
    gate_threshold.json sitting next to the bundle.
    """
    db = load_detector_bundle(detector_path)
    layer = layer or db["best_layer"]
    det_vec = db["vector"][db["layers"].index(layer)]
    if device is not None:
        det_vec = det_vec.to(device)
    if threshold is None:
        threshold = json.loads(
            (Path(detector_path).parent / "gate_threshold.json").read_text())["threshold"]
    return det_vec, layer, threshold


def steer_vector_at(bundle, layer=None, device=None):
    """(v, layer, act_norm) from a loaded steering bundle; None = best_layer."""
    layer = layer or bundle["best_layer"]
    li = bundle["layers"].index(layer)
    v = bundle["vector"][li]
    if device is not None:
        v = v.to(device)
    return v, layer, bundle["mean_act_norm"][li]
