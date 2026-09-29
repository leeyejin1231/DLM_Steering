import json
import os
import random
from pathlib import Path
import numpy as np
import torch
from model_loading import load_pretrained
from transformers import AutoModel, AutoTokenizer
from .constants import MODEL, MODEL_NAME


class ShiftedLogits(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner = model

    def forward(self, input_ids, **kwargs):
        kwargs.setdefault("use_cache", False)
        out = self.inner(input_ids, **kwargs)
        logits = out.logits
        out.logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        return out

    @property
    def device(self):
        return self.inner.device

    @property
    def config(self):
        return self.inner.config

    @property
    def blocks(self):
        return model_blocks(self.inner)


def model_blocks(model):
    """The transformer block list hooks attach to (LLaDA, Dream, or a wrapper)."""
    if hasattr(model, "blocks"):
        return model.blocks
    if hasattr(model, "model") and hasattr(model.model, "transformer"):
        return model.model.transformer.blocks      # LLaDA
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers                  # Dream (Qwen2 layout)
    raise AttributeError("cannot locate transformer blocks on the model")


def load_model(device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if MODEL["chat_control"]:
        tokenizer.add_special_tokens({"additional_special_tokens": list(MODEL["chat_control"])}, replace_additional_special_tokens=False)
    model = load_pretrained(AutoModel, MODEL_NAME, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={"": device}).eval()
    if MODEL["shift_logits"]:
        model = ShiftedLogits(model).eval()
    return tokenizer, model


load_llada = load_model  # Backward-compatible name for the selected target.


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def enable_reproducibility(seed=42):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.utils.deterministic.fill_uninitialized_memory = False


def release_cublas_env(device):
    if not torch.cuda.is_available() or "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        return
    probe = torch.ones(2, 2, device=device)
    (probe @ probe).sum().item()
    del os.environ["CUBLAS_WORKSPACE_CONFIG"]


def force_math_attention():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def prompt_token_ids(tokenizer, user_message):
    formatted = tokenizer.apply_chat_template([{"role": "user", "content": user_message}], add_generation_prompt=True, tokenize=False)
    return tokenizer(formatted)["input_ids"]


def encode_prompt(tokenizer, user_message, device):
    return torch.tensor(prompt_token_ids(tokenizer, user_message), device=device).unsqueeze(0)


def load_detector_bundle(detector_path):
    return torch.load(detector_path, map_location="cpu")


def load_detector(detector_path, layer=None, device=None, threshold=None):
    db = load_detector_bundle(detector_path)
    layer = layer or db["best_layer"]
    det_vec = db["vector"][db["layers"].index(layer)]
    if device is not None:
        det_vec = det_vec.to(device)
    if threshold is None:
        threshold = json.loads((Path(detector_path).parent / "gate_threshold.json").read_text())["threshold"]
    return det_vec, layer, threshold


def steer_vector_at(bundle, layer=None, device=None):
    layer = layer or bundle["best_layer"]
    li = bundle["layers"].index(layer)
    v = bundle["vector"][li]
    if device is not None:
        v = v.to(device)
    return v, layer, bundle["mean_act_norm"][li]
