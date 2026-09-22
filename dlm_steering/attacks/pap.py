"""PAP target calls using validated, pre-generated attack prompts."""
from pap_common import TOP5, SAMPLING, assign_techniques, load_cache, require_cache
from .base import NoAttack, AttackResult, _assistant_text


class PAP(NoAttack):
    """PAP: assign one of five techniques per row and apply it exactly once.

    Reads prepared better in-context attack prompts from a validated cache.
    Each row makes one target call; final grading is a separate evaluation.
    """

    name = "pap"
    records_attempts = True
    def __init__(self, cache, seed=0):
        self.cache = cache
        self.seed = seed
        self.techniques = list(TOP5)
        self.technique_by_index = None

    def prepare_rows(self, rows):
        self.technique_by_index = assign_techniques(rows, self.seed)

    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--pap-cache", help="Override the default data/attacks/pap_better/<source>/seed<seed>.json file.")
        parser.set_defaults(temperature=0.0)

    @classmethod
    def validate_inputs(cls, args):
        args.pap_cache = str(require_cache(args.pap_cache, args.source, args.seed, args.reproduct))

    @classmethod
    def from_args(cls, args):
        cls.validate_inputs(args)
        cache = load_cache(args.pap_cache, args.source, args.seed, args.reproduct)
        return cls(cache, seed=args.seed)

    def run(self, row, respond, tokenizer, vanilla_ids=None, respond_batch=None):
        goal, idx = row["prompt"], int(row.get("index", 0))
        if self.technique_by_index is None or idx not in self.technique_by_index:
            raise ValueError("Call PAP.prepare_rows with the full dataset before run")
        tech = self.technique_by_index[idx]
        if self.cache is None:
            raise ValueError("PAP.run requires a prepared cache; inline generation is disabled")
        entry = self.cache["by_index"].get(idx)
        if entry is None or entry["prompt"] != goal or entry["technique"] != tech:
            raise ValueError(f"PAP cache input/technique mismatch at index {idx}")
        pap = entry["attack_prompt"]
        out, ids, cfg, shown = respond(pap)
        resp = _assistant_text(tokenizer, out, ids)
        history = [{"trial": 1, "technique": tech, "attempt_index": 0,
                    "pap": pap, "response": resp}]
        print(f"    [pap] single attempt {tech}", flush=True)
        return AttackResult(shown, resp, {
            "assistant_text": resp,
            "pap": {"technique": tech,
                    "trials_run": 1, "target_queries": 1,
                    "history": history}}, cfg, ids)

    def describe(self):
        return {"attack": self.name, "techniques": self.techniques,
                "technique_assignment": "seeded full-dataset shuffle; balanced top5",
                "assignment_seed": self.seed,
                "dataset_technique_counts": {
                    technique: sum(t == technique for t in (self.technique_by_index or {}).values())
                    for technique in self.techniques},
                "trial_definition": "one paraphrase of the assigned technique",
                "trials": 1,
                "max_target_queries": 1,
                "paraphraser_prompt": "author PAP_Better_Incontext_Sample",
                "template_file": "attacks/pap_better_templates.json",
                "sampling": dict(SAMPLING),
                "backend": self.cache["backend"],
                "llm": self.cache["model"],
                "prompt_cache_sha256": self.cache.get("sha256"),
                "diagnostic_judge": "removed"}
