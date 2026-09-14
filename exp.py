"""Experiment entry: --attack X --defense Y over a prompt source.

The attack decides how the user turn is built and what gets graded; the
defense is a policy object the sampler consults every denoising step.
Attack/defense-specific flags are registered by the selected class.

Usage:
    CUDA_VISIBLE_DEVICES=1 python exp.py --attack dija --defense ours \
        --source jbb_harmful --n 100 --out outputs/dija_ours.json
"""

import argparse
import time

from Attacker import ATTACKERS
from common import MODEL_NAME, MASK_ID, encode_prompt, load_llada, load_prompts, write_json
from Defender import DEFENDERS


def parse_args():
    # The attack/defense choice decides which extra flags exist, so it is
    # parsed first and the selected class registers its own arguments.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--attack", choices=sorted(ATTACKERS), default="none")
    pre.add_argument("--defense", choices=sorted(DEFENDERS), default="none")
    known, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(parents=[pre])
    p.add_argument("--source", choices=["csv", "jbb_harmful"], default="csv")
    p.add_argument("--csv", default="data/llada8b_wild_unsafe_only.csv")
    p.add_argument("--out", default="outputs/exp.json")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--gen-length", type=int, default=128)
    p.add_argument("--block-length", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--remasking", default="low_confidence")
    p.add_argument("--schedule", default="const", choices=["const", "linear", "cosine"])
    ATTACKERS[known.attack].add_args(p)
    DEFENDERS[known.defense].add_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    attacker = ATTACKERS[args.attack].from_args(args)

    print(f"loading {MODEL_NAME} ...")
    tokenizer, model = load_llada()
    device = next(model.parameters()).device

    defender = DEFENDERS[args.defense].from_args(args, model)
    rows = load_prompts(args.source, args.csv)[args.start: args.start + args.n]
    print(f"{len(rows)} prompts from {args.source}, attack={args.attack}, "
          f"defense={args.defense}")

    gen_config = {"steps": args.steps, "gen_length": args.gen_length,
                  "block_length": args.block_length, "temperature": args.temperature,
                  "remasking": args.remasking, "schedule": args.schedule}

    results = []
    t_start = time.time()
    for i, row in enumerate(rows):
        user_message = defender.transform_prompt(attacker.build_prompt(row))
        input_ids = encode_prompt(tokenizer, user_message, device)

        t0 = time.time()
        out = defender.defend(model, input_ids, **gen_config)
        elapsed = time.time() - t0

        generation, extra = attacker.decode(tokenizer, out, input_ids)
        results.append({"index": row["index"], "prompt": row["prompt"],
                        "attack_prompt": user_message, "generation": generation,
                        **extra, "num_prompt_tokens": int(input_ids.shape[1]),
                        "num_prompt_masks": int((input_ids == MASK_ID).sum()),
                        "seconds": round(elapsed, 2), **defender.result_fields()})

        write_json(args.out, {"model": MODEL_NAME, "config": gen_config,
                              "attack": attacker.describe(),
                              "defense": defender.describe(), "results": results})
        print(f"[{i + 1}/{len(rows)}] idx={row['index']} {elapsed:.1f}s :: "
              f"{generation[:100].replace(chr(10), ' ')}...", flush=True)

    summary = defender.summarize(results)
    if summary:
        print(summary)
    print(f"\nDone in {time.time() - t_start:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
