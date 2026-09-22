"""Model loading, reproducibility, encoding, and detector checkpoints."""
import json
import os
from pathlib import Path
import numpy as np
import torch
from model_loading import load_pretrained
from .constants import MODEL, MODEL_NAME


class ShiftedLogits(torch.nn.Module):
    """Dream's lm_head is trained with an autoregressive shift: logits[:, p]
    scores the token at position p+1. Dream's own sampler realigns them with
    cat([logits[:, :1], logits[:, :-1]], 1) before reading masked slots, and
    this wrapper does the same so model(x).logits[0, p] scores slot p exactly
    as LLaDA's does. Hidden states are untouched (they are per-position
    residual streams, which is what the detectors and steering hooks read),
    so hooks are registered on .blocks of the wrapped model as usual.
    """

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
    """Selected target tokenizer and eval-mode bf16 model on `device`.

    device_map puts each checkpoint shard straight on the target device.
    `.to(device)` instead materialises the whole 16 GB state dict in CPU RAM
    first, which measures ~3x slower (15.6s -> 5.0s) for bit-identical weights.
    """
    from transformers import AutoModel, AutoTokenizer
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if MODEL["chat_control"]:
        tokenizer.add_special_tokens({"additional_special_tokens": list(MODEL["chat_control"])},
                                     replace_additional_special_tokens=False)
    model = load_pretrained(AutoModel, MODEL_NAME, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16,
                                      device_map={"": device}).eval()
    if MODEL["shift_logits"]:
        model = ShiftedLogits(model).eval()
    return tokenizer, model


load_llada = load_model  # Backward-compatible name for the selected target.


def seed_all(seed):
    """Seed python/np/torch RNGs. Applied always so runs are repeatable."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def enable_reproducibility(seed=42):
    """Bitwise-deterministic generation: fixed seeds + deterministic kernels.

    Must run before the first CUDA op so CUBLAS_WORKSPACE_CONFIG is already
    set when the cublas handle is created."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Deterministic mode NaN-fills every torch.empty() to expose reads of
    # uninitialised memory. Nothing here reads any, so the fill is one wasted
    # kernel per allocation (~3% of a forward) for bit-identical outputs.
    torch.utils.deterministic.fill_uninitialized_memory = False


def release_cublas_env(device):
    """Drop CUBLAS_WORKSPACE_CONFIG once `device`'s cuBLAS handle exists.

    torch 2.3 re-parses that variable with a freshly compiled std::regex on
    EVERY matmul while it is set, which makes an 8B forward launch-bound: 73 ms
    regardless of length against 37 ms without it, for bit-identical logits.
    The variable only matters up to the first cuBLAS call -- torch checks it
    once (a static) and from then on hands cuBLAS an explicit workspace on
    every call, which is what makes the results deterministic. So run that
    first call here, then unset it. Children spawned later set it again for
    themselves in enable_reproducibility.

    Call after enable_reproducibility and model loading, and only when every
    model in the process lives on `device`: a handle created for another
    device afterwards would size its workspace from the default instead.
    """
    if not torch.cuda.is_available() or "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        return
    probe = torch.ones(2, 2, device=device)
    (probe @ probe).sum().item()
    del os.environ["CUBLAS_WORKSPACE_CONFIG"]


def force_math_attention():
    """Restrict SDPA to the math backend -- the only backend whose numerics are
    stable across GPU architectures. Call AFTER load_llada: the remote model
    code re-enables flash_sdp in __init__."""
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


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
