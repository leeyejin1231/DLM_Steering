"""Defense protocol, plain forwarding, and shared per-response state."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import torch
from dlm_steering.runtime.constants import MODEL, MODEL_LOCK


def _prompt_text_mask(tokenizer, prompt_ids):
    """Editable user text; retain the chat headers even when HF omits their
    IDs from all_special_ids (as LLaDA's tokenizer does)."""
    marker = "__PROMPT_CONTENT_BOUNDARY__"
    template = tokenizer.apply_chat_template(
        [{"role": "user", "content": marker}], tokenize=False,
        add_generation_prompt=True)
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
    """Stops a detector-only forward once the gate block has been captured.

    Nothing after blocks[gate_layer - 1] can change what the detector reads, so
    on a probe or audit pass the remaining blocks and the vocab projection are
    pure waste (~1.8x on this model, where the gate sits at layer 18 of 32).
    The captured features are bit-identical to a full forward's.
    """


@dataclass
class _GatePass:
    """Scratch shared by one forward() and its gate/steering hooks."""
    pool: Any = None        # positions the gate projection is averaged over
    fired: bool = False     # the gate hook ran (it is skipped when idle)
    projection: Any = None  # 0-d tensor; stays unsynced until _read_scalars
    alpha: Any = 0.0        # effective steering alpha the steer hook applied


@dataclass
class _ForwardPlan:
    """What one forward() reads, steers and audits -- decided before the call.

    Every index tensor is resolved here rather than inside a hook: a nonzero()
    in the middle of the model call would sync the GPU mid-forward.
    """
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
    """The scalars one forward needs, after a single batched GPU sync."""
    projection: float
    gate_strength: float
    effective_alpha: float
    audit_values: list = field(default_factory=list)


@dataclass
class _PendingAudit:
    """A finished block whose boundary audit rides the next forward.

    V3 defers non-final boundaries so the gate-layer features come from a
    forward the sampler needs anyway, saving one full pass per clean boundary.
    """
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
    """What the response detector saw at one block boundary."""
    projection: float        # gate direction over every committed token
    block_projections: list  # ... and over each chunk of the finished block
    response_logit: float
    response_probability: float
    strength: float          # gate strength implied by `projection`


class Defender(ABC):
    """Per-run defense policy; the sampler owns token sampling."""

    name: str
    # gen_length 0 only: ask the sampler for an after_block audit every this
    # many committed mask slots (0 = only at the end). See sampler.generate.
    infill_checkpoint = 0

    @classmethod
    def add_args(cls, parser):
        """Register defense-specific CLI arguments (optional)."""

    @classmethod
    @abstractmethod
    def from_args(cls, args, model):
        """Build the policy: load bundles, construct, return the instance."""

    def prepare(self, tokenizer, vanilla_ids):
        """Provide token metadata and the clean reference for this row."""
        self.tokenizer = tokenizer
        self.vanilla_ids = vanilla_ids

    def transform_prompt(self, user_message):
        """Prompt-side hook; identity for activation-level defenses."""
        return user_message

    @abstractmethod
    def reset(self):
        """Reset per-response state."""

    @abstractmethod
    def forward(self, x, region, *, schedule_scale, logit_positions=None,
                n_masks=None):
        """One model forward, optionally with in-forward detection + steering.
        Must return the model output (with .logits). logit_positions, when
        given, restricts the vocab projection to those rows -- .logits is then
        aligned to them. Implementations may ignore it (full logits).
        n_masks, when given, is the number of still-masked region slots, which
        the sampler knows without asking the GPU."""

    def after_block(self, x, region, *, block_number, block_positions,
                    prompt_length, temperature, remasking, last_block=False,
                    rng=None, sampling=None):
        """Called once after each block's denoising loop completes.

        block_number: which block just finished, 0-based (not to be confused
        with common.block_index, which maps a layer to a transformer block).
        block_positions: answer slots of the just-finished block. prompt_length
        marks the prompt/generation boundary; committed answer slots before it
        (e.g. filled DIJA spans) are also remaskable. last_block marks the
        final block (no later forward exists to piggyback an audit on). May
        audit the block or rewrite committed tokens back to MASK_ID. rng is
        the row's torch.Generator, needed by implementations that resample
        (V3 recovery).
        """

    @abstractmethod
    def result_fields(self):
        """Fields merged into this response's result record."""

    @staticmethod
    def summarize(results):
        """One-line run summary of a finished run, or None.

        Static like Evaluator.summarize so the --gpus parent can summarise the
        merged shards without building a policy of its own.
        """
        return None

    def describe(self):
        """Config dict stored in the result payload."""
        return {"defense": self.name}

    def defend(self, model, prompt_ids, rng=None, **gen_config):
        """Run one defended generation through the unified sampler."""
        from sampler import generate
        return generate(model, prompt_ids, self, rng=rng, **gen_config)

    def defend_batch(self, model, prompt_ids_list, rng=None, **gen_config):
        """A batch of defended generations; sequential by default.

        Stateful defenses (remask/recovery branch per sequence) cannot share
        a denoising loop, so only NullDefender overrides this with the truly
        batched sampler.
        """
        return [self.defend(model, ids, rng=rng, **gen_config)
                for ids in prompt_ids_list]


class NullDefender(Defender):
    """Undefended reference: plain model forwards."""

    name = "none"

    def __init__(self, model):
        self.model = model

    @classmethod
    def from_args(cls, args, model):
        return cls(model)

    def reset(self):
        pass

    def forward(self, x, region, *, schedule_scale, logit_positions=None,
                n_masks=None):
        if logit_positions is None or MODEL["shift_logits"]:
            return self.model(x)
        ln_f = self.model.model.transformer.ln_f
        with MODEL_LOCK:
            handle = ln_f.register_forward_hook(
                lambda m, i, o: o[:, logit_positions])
            try:
                return self.model(x)
            finally:
                handle.remove()

    def defend_batch(self, model, prompt_ids_list, rng=None, **gen_config):
        """No hooks -> rows denoise uniformly; safe to share one loop."""
        from sampler import generate_batch
        return generate_batch(model, prompt_ids_list, rng=rng, **gen_config)

    def result_fields(self):
        return {}
