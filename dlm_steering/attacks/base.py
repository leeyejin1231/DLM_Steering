from abc import ABC, abstractmethod
from pathlib import Path
from typing import NamedTuple


class AttackResult(NamedTuple):
    attack_prompt: str     # user message as seen by the target (post-transform)
    generation: str        # text passed to the evaluators
    extra: dict            # attack-specific result fields
    cfg: dict              # sampler config of the recorded generation
    prompt_ids: object     # token ids of the recorded prompt (for num_* fields)


class Attacker(ABC):
    name: str
    needs_vanilla = False        # decode() uses the un-attacked prompt ids
    needs_second_device = False  # attack drives a second local LLM
    records_attempts = False

    @classmethod
    def add_args(cls, parser):
        ...

    @classmethod
    def validate_inputs(cls, args):
        ...

    @classmethod
    def from_args(cls, args):
        return cls()

    def prepare_rows(self, rows):
        ...

    @abstractmethod
    def build_prompt(self, row):
        ...

    @abstractmethod
    def decode(self, tokenizer, x, prompt_ids, vanilla_ids=None):
        ...

    def run(self, row, respond, tokenizer, vanilla_ids=None, respond_batch=None):
        """Drive the per-row attack; default is the one-shot path.

        respond(user_message) sends one user turn through the defense and
        returns (x, prompt_ids, cfg, shown) where shown is the text the target
        actually saw (after defender.transform_prompt). respond_batch does the
        same for a list of messages -- a real batched sampler under --defense
        none, a sequential fallback otherwise."""
        out, ids, cfg, shown = respond(self.build_prompt(row))
        generation, extra = self.decode(tokenizer, out, ids, vanilla_ids)
        return AttackResult(shown, generation, extra, cfg, ids)

    def gen_overrides(self, prompt_ids):
        """Per-prompt overrides of the sampler config (e.g. steps)."""
        return {}

    def describe(self):
        """Config dict stored in the result payload."""
        return {"attack": self.name}


class NoAttack(Attacker):
    name = "none"

    def build_prompt(self, row):
        return row["prompt"]

    def decode(self, tokenizer, x, prompt_ids, vanilla_ids=None):
        assistant = tokenizer.batch_decode(
            x[:, prompt_ids.shape[1]:], skip_special_tokens=True)[0]
        return assistant, {"assistant_text": assistant}


class Prefix(NoAttack):
    """Prepends a fixed jailbreak prefix (default attacks/prefix.txt)."""

    name = "prefix"

    def __init__(self, prefix_file="attacks/prefix.txt"):
        self.prefix_file = prefix_file
        self.prefix = Path(prefix_file).read_text().rstrip()

    @classmethod
    def from_args(cls, args):
        return cls()

    def build_prompt(self, row):
        return f"{self.prefix}\n\n{row['prompt']}"

    def describe(self):
        return {"attack": self.name, "prefix_file": self.prefix_file}


def _assistant_text(tokenizer, out, ids):
    return tokenizer.batch_decode(
        out[:, ids.shape[1]:], skip_special_tokens=True)[0]


def _default_attack_device():
    import torch
    if torch.cuda.device_count() > 1:
        return "cuda:1"
    print("warning: one GPU visible -- the attack/judge LLM shares the card with the target model and may OOM")
    return "cuda:0"
