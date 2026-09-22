"""Execute one experiment row and preserve candidate results on failure."""
from copy import deepcopy
import time
import traceback

import torch

from Attacker import AttackResult
from common import ERROR_SENTINEL, MASK_ID, encode_prompt


def graded_fields(row):
    """Carry dataset answer keys through to the utility evaluators."""
    return {key: row[key] for key in ("task", "answer", "subject", "category", "level")
            if key in row}


class RowExecution:
    """Per-row RNG, target calls, and result recording for one worker lane."""

    def __init__(self, row, attacker, defender, model, tokenizer, config, seed):
        self.row = row
        self.attacker = attacker
        self.defender = defender
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device
        # Row identity, rather than worker order, determines sampler randomness.
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(seed + int(row["index"]))
        self.attempts = []
        self.candidate_ids = []

    def _save_attempt(self, shown, ids, config, generation, fields, error=None):
        attempt = {"attempt_index": len(self.attempts), "attack_prompt": shown,
                   "generation": generation, "config": config,
                   "defense_fields": fields}
        if error is not None:
            attempt["error"] = error
        self.attempts.append(attempt)
        self.candidate_ids.append(ids)

    def respond(self, user_message):
        """Transform, encode, defend, then snapshot this candidate's state."""
        shown = self.defender.transform_prompt(user_message)
        ids = encode_prompt(self.tokenizer, shown, self.device)
        config = {**self.config, **self.attacker.gen_overrides(ids)}
        try:
            out = self.defender.defend(self.model, ids, rng=self.rng, **config)
        except Exception as exc:
            if self.attacker.records_attempts:
                self._save_attempt(shown, ids, config, ERROR_SENTINEL, {},
                                   error=f"{type(exc).__name__}: {exc}")
            raise
        if self.attacker.records_attempts:
            generation = self.tokenizer.batch_decode(
                out[:, ids.shape[1]:], skip_special_tokens=True)[0]
            self._save_attempt(shown, ids, config, generation,
                               deepcopy(self.defender.result_fields()))
        return out, ids, config, shown

    def respond_batch(self, user_messages):
        """Keep candidate recording sequential; otherwise use defender batching."""
        if self.attacker.records_attempts:
            return [self.respond(message) for message in user_messages]
        shown = [self.defender.transform_prompt(message) for message in user_messages]
        all_ids = [encode_prompt(self.tokenizer, text, self.device) for text in shown]
        configs = [{**self.config, **self.attacker.gen_overrides(ids)} for ids in all_ids]
        if all(config == configs[0] for config in configs):
            outs = self.defender.defend_batch(self.model, all_ids, rng=self.rng, **configs[0])
        else:
            outs = [self.defender.defend(self.model, ids, rng=self.rng, **config)
                    for ids, config in zip(all_ids, configs)]
        return list(zip(outs, all_ids, configs, shown))

    def record(self, result, elapsed):
        extra = result.extra
        if result.cfg != self.config:
            extra = {**extra, "gen_overrides": {
                key: value for key, value in result.cfg.items()
                if self.config.get(key) != value}}
        fields = self.defender.result_fields()
        if self.attempts:
            selected = next(i for i, ids in enumerate(self.candidate_ids)
                            if ids is result.prompt_ids)
            fields = self.attempts[selected]["defense_fields"]
            extra = {**extra, "attempts": self.attempts,
                     "selected_attempt_index": selected,
                     "target_queries": len(self.attempts), "search_complete": True}
        return {"index": self.row["index"], "prompt": self.row["prompt"],
                **graded_fields(self.row), "attack_prompt": result.attack_prompt,
                "generation": result.generation, **extra,
                "num_prompt_tokens": int(result.prompt_ids.shape[1]),
                "num_prompt_masks": int((result.prompt_ids == MASK_ID).sum()),
                "seconds": round(elapsed, 2), **fields}

    def record_failure(self, elapsed, error):
        """Retain completed candidates even if a later attack step fails."""
        record = {"index": self.row["index"], "prompt": self.row["prompt"],
                  **graded_fields(self.row), "attack_prompt": None,
                  "generation": ERROR_SENTINEL, "error": error}
        if self.attacker.records_attempts:
            valid = [i for i, attempt in enumerate(self.attempts)
                     if attempt["generation"] != ERROR_SENTINEL]
            if valid:
                index = valid[-1]
                attempt = self.attempts[index]
                fallback = AttackResult(attempt["attack_prompt"], attempt["generation"],
                                        {}, attempt["config"], self.candidate_ids[index])
                record = self.record(fallback, elapsed)
            record.update(attempts=self.attempts, target_queries=len(self.attempts),
                          search_complete=False, attack_error=error)
        return record

    def run(self):
        vanilla_ids = None
        if self.attacker.needs_vanilla or getattr(self.defender, "needs_vanilla", False):
            vanilla_ids = encode_prompt(
                self.tokenizer, self.defender.transform_prompt(self.row["prompt"]), self.device)
        self.defender.prepare(self.tokenizer, vanilla_ids)
        start = time.time()
        try:
            result = self.attacker.run(self.row, self.respond, self.tokenizer, vanilla_ids,
                                       respond_batch=self.respond_batch)
            return self.record(result, time.time() - start)
        except Exception:
            traceback.print_exc()
            return self.record_failure(time.time() - start, traceback.format_exc(limit=5))
