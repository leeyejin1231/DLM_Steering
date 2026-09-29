import json
import re
from pathlib import Path
from dlm_steering.paths import DATA_DIR
from dlm_steering.runtime.constants import MASK_TOKEN, MASK_ID
from .base import NoAttack


DIJA_MASK_PATTERN = re.compile(r"<mask:(\d+)>")


DIJA_REFINED = {
    "jbb_harmful": ("run_jailbreakbench/refine_prompt/jailbreakbench_data_refined_{v}.json", "goal", "refined_goal"),
    "harmbench": ("run_harmbench/refine_prompt/harmbench_behaviors_text_all_refined_{v}.json", "Behavior", "Refined_behavior"),
    "strongreject": ("run_strongreject/refine_prompt/strongreject_data_refined_{v}.json", "vanilla prompt", "refined prompt"),
}


class DIJA(NoAttack):
    name = "dija"
    needs_vanilla = True

    def __init__(self, source, dija_dir=None, version="Qwen", steps="auto"):
        if source not in DIJA_REFINED:
            raise ValueError(f"--attack dija has refined prompts only for {sorted(DIJA_REFINED)}, not {source!r}")
        rel, vanilla_key, refined_key = DIJA_REFINED[source]
        self.source, self.version, self.steps = source, version, steps
        self.refined_file = Path(dija_dir or DATA_DIR / "dija") / rel.format(v=version)
        items = json.loads(self.refined_file.read_text())
        self.refined = {it[vanilla_key].strip(): it[refined_key] for it in items}

    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--dija-steps", default="auto", help="'auto': one prompt mask per step for gen-length 0; use --steps when appending an answer. Or an integer.")

    @classmethod
    def from_args(cls, args):
        steps = args.dija_steps if args.dija_steps == "auto" else int(args.dija_steps)
        if steps == "auto" and args.gen_length > 0:
            steps = args.steps
        return cls(args.source, steps=steps)

    def build_prompt(self, row):
        key = row["prompt"].strip()
        if key not in self.refined:
            raise KeyError(f"no refined DIJA prompt for {self.source} row {row['index']}: {key[:80]!r}")
        self._current_refined = self.refined[key]
        return DIJA_MASK_PATTERN.sub(lambda m: MASK_TOKEN * int(m.group(1)), self._current_refined)

    def gen_overrides(self, prompt_ids):
        if self.steps == "auto":
            return {"steps": int((prompt_ids == MASK_ID).sum())}
        return {"steps": self.steps}

    def decode(self, tokenizer, x, prompt_ids, vanilla_ids=None):
        if vanilla_ids is None:
            raise ValueError("DIJA.decode needs the vanilla prompt ids")
        a, b = prompt_ids[0].tolist(), vanilla_ids[0].tolist()
        matching = min(sum(u == v for u, v in zip(a, b)), len(b))
        text = tokenizer.decode(x[0, matching:].tolist(), skip_special_tokens=True)
        response = text.split("assistant\n")[0].strip()
        assistant = tokenizer.decode(x[0, prompt_ids.shape[1]:].tolist(), skip_special_tokens=True)
        return response, {"filled_template": response, "assistant_text": assistant, "refined_prompt": self._current_refined}

    def describe(self):
        return {"attack": self.name, "refined_file": str(self.refined_file),
                "version": self.version, "steps": self.steps}
