from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import torch
from dlm_steering.runtime.constants import MODEL, MODEL_LOCK


def _prompt_text_mask(tokenizer, prompt_ids):
    marker = "__PROMPT_CONTENT_BOUNDARY__"
    template = tokenizer.apply_chat_template([{"role": "user", "content": marker}], tokenize=False, add_generation_prompt=True)
    prefix, suffix = template.split(marker)
    prefix_len = len(tokenizer(prefix, add_special_tokens=False).input_ids)
    suffix_len = len(tokenizer(suffix, add_special_tokens=False).input_ids)
    special = torch.tensor(tokenizer.all_special_ids, device=prompt_ids.device)
    editable = ~torch.isin(prompt_ids[0], special)
    editable[:prefix_len] = False
    if suffix_len:
        editable[-suffix_len:] = False
    return editable


def _hidden(output):
    return output[0] if isinstance(output, tuple) else output


def _replace(output, hidden):
    return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden


class _GateReached(Exception):
    ...


@dataclass
class _GatePass:
    pool: Any = None        # positions the gate projection is averaged over
    fired: bool = False     # the gate hook ran (it is skipped when idle)
    projection: Any = None  # 0-d tensor; stays unsynced until _read_scalars
    alpha: Any = 0.0        # effective steering alpha the steer hook applied


@dataclass
class _ForwardPlan:
    source: str             # "generated" once any answer token is committed
    n_committed: int
    read_gate: bool
    steer: bool
    gate_pool: Any = None        # index tensor for the gate, or None
    steer_mask: Any = None       # bool [seq]: masked slots the steering hook pushes
    audit: Any = None            # _PendingAudit riding this forward, or None
    audit_pools: Any = None      # index tensors the audit hook pools over


@dataclass
class _StepScalars:
    projection: float
    gate_strength: float
    effective_alpha: float
    audit_values: list = field(default_factory=list)


@dataclass
class _PendingAudit:
    block_number: int       # which block just finished (0-based)
    block_row: Any          # bool [seq]: that block's answer slots
    prompt_length: int
    temperature: float
    remasking: str
    rng: Any = None
    sampling: Any = None
    chunks: list = field(default_factory=list)


@dataclass
class _BoundaryReading:
    projection: float        # gate direction over every committed token
    block_projections: list  # ... and over each chunk of the finished block
    response_logit: float
    response_probability: float
    strength: float          # gate strength implied by `projection`


class Defender(ABC):
    name: str
    infill_checkpoint = 0

    @classmethod
    def add_args(cls, parser):
        ...

    @classmethod
    @abstractmethod
    def from_args(cls, args, model):
        ...

    def prepare(self, tokenizer, vanilla_ids):
        self.tokenizer = tokenizer
        self.vanilla_ids = vanilla_ids

    def transform_prompt(self, user_message):
        return user_message

    @abstractmethod
    def reset(self):
        ...

    @abstractmethod
    def forward(self, x, region, *, schedule_scale, logit_positions=None,
                n_masks=None):
        ...

    def after_block(self, x, region, *, block_number, block_positions, prompt_length, temperature, remasking, last_block=False, rng=None, sampling=None):
        ...

    @abstractmethod
    def result_fields(self):
        ...

    @staticmethod
    def summarize(results):
        return None

    def describe(self):
        return {"defense": self.name}

    def defend(self, model, prompt_ids, rng=None, **gen_config):
        from sampler import generate
        return generate(model, prompt_ids, self, rng=rng, **gen_config)

    def defend_batch(self, model, prompt_ids_list, rng=None, **gen_config):
        return [self.defend(model, ids, rng=rng, **gen_config)
                for ids in prompt_ids_list]


class NullDefender(Defender):
    name = "none"

    def __init__(self, model):
        self.model = model

    @classmethod
    def from_args(cls, args, model):
        return cls(model)

    def reset(self):
        pass

    def forward(self, x, region, *, schedule_scale, logit_positions=None, n_masks=None):
        if logit_positions is None or MODEL["shift_logits"]:
            return self.model(x)
        ln_f = self.model.model.transformer.ln_f
        with MODEL_LOCK:
            handle = ln_f.register_forward_hook(lambda m, i, o: o[:, logit_positions])
            try:
                return self.model(x)
            finally:
                handle.remove()

    def defend_batch(self, model, prompt_ids_list, rng=None, **gen_config):
        from sampler import generate_batch
        return generate_batch(model, prompt_ids_list, rng=rng, **gen_config)

    def result_fields(self):
        return {}
