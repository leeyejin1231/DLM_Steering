"""Run the unmodified DiffuGuard Dream runner (DiffuGuard/models/jailbreakbench_dream.py) on Dream.

Three patches, nothing else (the runner's own argparse/CLI is used as-is):
  1. torch.no_grad__ shim -- the author file decorates two functions with this typo, killing import.
  2. AutoModel.from_pretrained returns the model wrapped in common.ShiftedLogits, so the runner's
     direct model(x).logits reads are aligned like Dream's own sampler (AR shift). Attribute access
     (diffusion_generate, config, ...) is delegated to the inner model, so the native path is untouched.
  3. tokenizer.batch_decode replaces the "<|im_end|>\\n<|im_start|>assistant\\n" turn boundary with the
     marker "\\n<<<ASSISTANT>>>\\n", so the runner's DIJA cut response.split("assistant\\n")[0] no longer
     drops the assistant tail; the graded text is "filled template + assistant answer" like exp.py.
     Replace the marker with "\\n\\n" before script/convert_diffuguard.py.
Seed: DGM_SEED (the author runner has no seed flag; use the same value as exp.py --seed).

Usage (from the DLM_Steering root):
  PYTHONPATH=$PWD:$PWD/script/diffuguard_stubs DGM_SEED=43 \
  python script/diffuguard_dream/run_dgm_dream_native_gen128.py <jailbreakbench_dream.py args>
"""
import os
import random
import runpy
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
MARKER = "\n<<<ASSISTANT>>>\n"

seed = int(os.environ.get("DGM_SEED", "42"))
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# 1. typo shim
torch.no_grad__ = torch.no_grad

# 2. ShiftedLogits wrap
os.environ.setdefault("DLM_MODEL", "dream")
sys.path.insert(0, str(ROOT))
from common import MODEL, ShiftedLogits  # noqa: E402
import transformers  # noqa: E402


class _Shifted(ShiftedLogits):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.inner, name)


_orig_model = transformers.AutoModel.from_pretrained.__func__
transformers.AutoModel.from_pretrained = classmethod(
    lambda cls, *a, **k: _Shifted(_orig_model(cls, *a, **k)))

# 3. turn-boundary marker in batch_decode
_orig_tok = transformers.AutoTokenizer.from_pretrained.__func__
_BOUNDARY = "<|im_end|>\n<|im_start|>assistant\n"


def _patched_tokenizer(cls, *a, **k):
    tok = _orig_tok(cls, *a, **k)
    orig_batch_decode = tok.batch_decode

    def batch_decode(*args, **kwargs):
        return [t.replace(_BOUNDARY, MARKER) for t in orig_batch_decode(*args, **kwargs)]
    tok.batch_decode = batch_decode
    return tok


transformers.AutoTokenizer.from_pretrained = classmethod(_patched_tokenizer)

for p in (ROOT / "script/diffuguard_stubs", ROOT / "DiffuGuard", ROOT / "DiffuGuard/models"):
    sys.path.insert(0, str(p))
os.chdir(ROOT / "DiffuGuard")
runpy.run_path(str(ROOT / "DiffuGuard/models/jailbreakbench_dream.py"), run_name="__main__")
