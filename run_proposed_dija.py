"""Drive proposed.Proposed with the same sampler and DIJA prompts as
llada_steering_remasking_v2.py, so its ASR is directly comparable.

proposed.py is used unmodified: latched binary gate read by a separate
detector forward each step until done, projection-removal steering on layers
12/16/20/24 at strength 0.4 over ALL positions, one remasking action on the
first detection with committed tokens, and initial_only=True by default (a
closed step-0 gate ends monitoring for the response).

The gate checkpoint format proposed.py expects (center/scale) is built here from
outputs/steer_detector.pt with center=0, scale=1 so the threshold from
outputs/gate_threshold.json keeps its original h.v units.

Usage:
    CUDA_VISIBLE_DEVICES=1 python run_proposed_dija.py --source jbb_harmful --attack dija \
        --n 100 --out outputs/dija_jbb_proposed.json
"""

import argparse
import json
import time
from pathlib import Path

import torch

from llada import MODEL_NAME, MASK_ID
from proposed import Proposed
from llada_steering_remasking_v2 import (
    load_prompts, build_user_message, decode_response, generate_defended)


class ProposedAdapter:
    """Expose proposed.Proposed through the policy interface generate_defended uses."""

    def __init__(self, model, gate, csd, *, layers, total_steps, **options):
        self.model, self.gate, self.csd = model, gate, csd
        self.layers, self.options, self.total_steps = layers, options, total_steps
        self.policy = None

    def reset(self):
        # proposed.py wants one instance per response.
        self.policy = Proposed.from_llada(self.model, self.gate, self.csd, layers=self.layers,
                                          **self.options)
        self.step, self.trace, self.remask_event = 0, [], None

    def before_step(self, x, region, *, scope, steps_remaining):
        # proposed.py is block-agnostic: it budgets against the whole generation.
        before = int(((x == MASK_ID) & region).sum())
        masks, _, _ = self.policy.before_step(
            x, region, steps_remaining=self.total_steps - self.step, commit_count=1)
        reopened = int(masks.sum()) - before
        commit_count = None
        if reopened > 0:
            self.remask_event = {"step": self.step, "reopened": reopened}
            commit_count = 1  # any value: the sampler rebalances the block schedule
        self.trace.append({"step": self.step, "armed": self.policy.armed, "done": self.policy.done})
        return masks, int((masks & scope).sum()), commit_count

    def forward(self, x, region, *, schedule_scale=1.0):
        self.step += 1
        return self.policy.forward(x, region)

    def result_fields(self):
        return {"gate_open": self.policy.armed,
                "armed_at_step": next((t["step"] for t in self.trace if t["armed"]), None),
                "monitoring_steps": sum(1 for t in self.trace if not t["done"]),
                "remasked": self.remask_event is not None,
                "remask_event": self.remask_event}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["csv", "jbb_harmful"], default="jbb_harmful")
    p.add_argument("--csv", default="data/llada8b_wild_unsafe_only.csv")
    p.add_argument("--attack", choices=["none", "dija"], default="dija")
    p.add_argument("--dija-steps", type=int, default=4)
    p.add_argument("--dija-span", type=int, default=16)
    p.add_argument("--vector", default="outputs/steer_vector.pt")
    p.add_argument("--detector", default="outputs/steer_detector.pt")
    p.add_argument("--detector-layer", type=int, default=18)
    p.add_argument("--gate-threshold", type=float, default=None)
    p.add_argument("--layers", default="12,16,20,24")
    p.add_argument("--strength", type=float, default=0.4)
    p.add_argument("--mode", choices=["baseline", "steer", "repair"], default="repair")
    p.add_argument("--max-remask-tokens", type=int, default=16)
    p.add_argument("--max-parallel-commit", type=int, default=2)
    p.add_argument("--no-initial-only", action="store_true")
    p.add_argument("--out", default="outputs/dija_jbb_proposed.json")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--gen-length", type=int, default=128)
    p.add_argument("--block-length", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    return p.parse_args()


def main():
    from transformers import AutoModel, AutoTokenizer

    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    csd = torch.load(args.vector, map_location="cpu")
    db = torch.load(args.detector, map_location="cpu")
    det_vec = db["vector"][db["layers"].index(args.detector_layer)]
    threshold = args.gate_threshold
    if threshold is None:
        threshold = json.loads((Path(args.detector).parent / "gate_threshold.json").read_text())["threshold"]
    gate = {"layer": args.detector_layer, "vector": det_vec.to(device),
            "center": torch.zeros_like(det_vec).to(device), "scale": 1.0, "threshold": threshold}
    csd = {"layers": csd["layers"], "vector": csd["vector"].to(device)}
    layers = tuple(int(s) for s in args.layers.split(","))
    print(f"proposed.py: layers {layers}, strength {args.strength}, mode {args.mode}, "
          f"gate layer {args.detector_layer} threshold {threshold:.3f}, "
          f"initial_only {not args.no_initial_only}")

    rows = load_prompts(args.source, args.csv)[args.start: args.start + args.n]
    print(f"running {len(rows)} prompts from {args.source} (attack {args.attack})")

    print(f"loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True,
                                      dtype=torch.bfloat16).to(device).eval()

    policy = ProposedAdapter(
        model, gate, csd, layers=layers, total_steps=args.steps, strength=args.strength,
        mode=args.mode, max_remask_tokens=args.max_remask_tokens,
        max_parallel_commit=args.max_parallel_commit, initial_only=not args.no_initial_only)

    gen_config = {"steps": args.steps, "gen_length": args.gen_length,
                  "block_length": args.block_length, "temperature": args.temperature,
                  "cfg_scale": 0.0, "remasking": "low_confidence",
                  "attack": args.attack, "dija_steps": args.dija_steps, "dija_span": args.dija_span}
    steer_config = {"method": "proposed.py", "layers": list(layers), "strength": args.strength,
                    "transform": "project", "mode": args.mode, "detector_layer": args.detector_layer,
                    "gate_threshold": threshold, "max_remask_tokens": args.max_remask_tokens,
                    "max_parallel_commit": args.max_parallel_commit,
                    "initial_only": not args.no_initial_only, "single_forward": False}

    results = []
    t_start = time.time()
    for i, row in enumerate(rows):
        idx, prompt = row["index"], row["prompt"]
        user_message = build_user_message(row, args.attack, args.dija_steps, args.dija_span)
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}], add_generation_prompt=True, tokenize=False)
        input_ids = torch.tensor(tokenizer(formatted)["input_ids"], device=device).unsqueeze(0)

        t0 = time.time()
        out = generate_defended(model, input_ids, policy, steps=args.steps, gen_length=args.gen_length,
                                block_length=args.block_length, temperature=args.temperature)
        elapsed = time.time() - t0

        generation, extra = decode_response(tokenizer, out, input_ids, args.attack)
        results.append({"index": int(idx), "prompt": prompt, "attack_prompt": user_message,
                        "generation": generation, **extra,
                        "num_prompt_tokens": int(input_ids.shape[1]),
                        "num_prompt_masks": int((input_ids == MASK_ID).sum()),
                        "seconds": round(elapsed, 2), **policy.result_fields()})
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"model": MODEL_NAME, "config": gen_config, "steering": steer_config,
                       "results": results}, f, ensure_ascii=False, indent=2)
        r = results[-1]
        print(f"[{i + 1}/{len(rows)}] idx={idx} {elapsed:.1f}s armed={r['gate_open']}"
              f"{' [remask]' if r['remasked'] else ''} :: {generation[:100].replace(chr(10), ' ')}...",
              flush=True)

    print(f"gate opened on {sum(r['gate_open'] for r in results)}/{len(results)} prompts, "
          f"remasked {sum(r['remasked'] for r in results)}")
    print(f"\nDone in {time.time() - t_start:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
