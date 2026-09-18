"""Experiment entry: --attack X --defense Y over a prompt source.

The attack decides how the user turn is built and what gets graded; the
defense is a policy object the sampler consults every denoising step.
Attack/defense-specific flags are registered by the selected class.

Usage:
    # DIJA with the paper's refined prompts (DIJA/run_*/refine_prompt); the
    # attack sets gen_length 0, temperature 0.2 and one mask per step.
    CUDA_VISIBLE_DEVICES=1 python exp.py --attack dija --defense ours \
        --source jbb_harmful --n 100 --out outputs/dija_ours.json

    # Same method on Dream-v0-Instruct-7B: --model picks the target (vectors
    # and detectors default to outputs/dream/, see script/build_vectors.sh).
    CUDA_VISIBLE_DEVICES=1 python exp.py --model dream --attack dija --defense ours \
        --remask v3 --source jbb_harmful --n 100 --out outputs/dream/JBB-dija-v3-42.json

    # utility / generalisation: graded sets, scored by eval_utility.py
    CUDA_VISIBLE_DEVICES=1 python exp.py --attack none --defense ours --remask v3 \
        --source mmlu --n 500 --out outputs/MMLU-none-v3-42.json
    python eval_utility.py --in outputs/MMLU-none-v3-42.json --out outputs/MMLU-none-v3-42_acc.json

    # over-refusal: benign prompts under the same defense, then judge refusals
    CUDA_VISIBLE_DEVICES=1 python exp.py --attack none --defense ours --remask v3 \
        --source truthfulqa --n 200 --out outputs/TQA-none-v3-42.json
    python steering/judge_refusal.py --in outputs/TQA-none-v3-42.json \
        --out outputs/TQA-none-v3-42_judged.json
"""

import argparse
import time

from Attacker import ATTACKERS
from common import (MODEL_KEY, MODEL_NAME, MASK_ID, PROMPT_SOURCES, add_model_arg,
                    encode_prompt, enable_reproducibility, force_math_attention,
                    load_model, load_prompts, parse_gpu_ids, run_eval_shards,
                    seed_all, write_json)
from Defender import DEFENDERS
from sampler import DECODERS, DREAM_ALGS


def parse_args():
    # The attack/defense choice decides which extra flags exist, so it is
    # parsed first and the selected class registers its own arguments.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--attack", choices=sorted(ATTACKERS), default="none")
    pre.add_argument("--defense", choices=sorted(DEFENDERS), default="none")
    known, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(parents=[pre])
    add_model_arg(p)   # already applied at import time; validated/recorded here
    p.add_argument("--source", choices=list(PROMPT_SOURCES), default="jbb_harmful",
                   help="harmful: jbb_harmful, advbench, harmbench, strongreject, xstest_unsafe; "
                        "benign (over-refusal): truthfulqa, xstest_safe, jbb_benign, wj_benign; "
                        "utility (accuracy via eval_utility.py): mmlu, gsm8k, truthfulqa_mc")
    p.add_argument("--out", default="outputs/exp.json")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--gen-length", type=int, default=128,
                   help="assistant tokens to append; DIJA attacks default this to 0 "
                        "(prompt-span infilling only)")
    p.add_argument("--block-length", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--remasking", default="low_confidence")
    p.add_argument("--schedule", default="const", choices=["const", "linear", "cosine"])
    p.add_argument("--decoder", choices=list(DECODERS),
                   default="dream" if MODEL_KEY == "dream" else "block",
                   help="block: LLaDA semi-AR blocks with --remasking (default for "
                        "llada). dream: Dream's own diffusion_generate rule (default "
                        "for --model dream); --block-length is then the audit "
                        "granularity in committed tokens.")
    p.add_argument("--alg", choices=list(DREAM_ALGS), default="origin",
                   help="decoder=dream: position order (Dream's alg). origin = "
                        "random order, diffusion_generate's config default and what "
                        "the DIJA and DiffuGuard authors run on Dream; entropy is "
                        "Dream's README recommendation.")
    p.add_argument("--alg-temp", type=float, default=None,
                   help="decoder=dream: soften the confidence ordering (Dream alg_temp)")
    p.add_argument("--top-p", type=float, default=None,
                   help="decoder=dream: nucleus sampling (default 0.95 as in Dream's "
                        "and the DIJA authors' Dream runs; 1 disables)")
    p.add_argument("--top-k", type=int, default=None,
                   help="decoder=dream: top-k truncation (default 50, which is what "
                        "diffusion_generate applies when top_k is not given; 0 disables)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reproduct", action="store_true",
                   help="Bitwise-deterministic generation: fixed seeds, deterministic "
                        "kernels, math SDPA backend. Slower but identical across GPUs.")
    p.add_argument("--gpus", default=None,
                   help="Comma-separated GPU ids (e.g. 0,1,2,3): shard the prompt "
                        "set into contiguous --start/--n slices, run one exp.py "
                        "subprocess per GPU, and merge the part JSONs into --out.")
    ATTACKERS[known.attack].add_args(p)
    DEFENDERS[known.defense].add_args(p)
    return p.parse_args()


def run_sharded(args):
    """Launcher path for --gpus: one exp.py subprocess per GPU, then merge.

    Each child reruns this file with its own --start/--n/--out slice and
    CUDA_VISIBLE_DEVICES set; per-prompt seeding (seed + row index) keeps
    rows identical no matter how they are partitioned. Attacks that drive a
    second LLM (pap/pair) get a (target, attack) GPU pair per shard so the
    two models never share a card.
    """
    devices = None
    if ATTACKERS[args.attack].needs_second_device:
        ids = parse_gpu_ids(args.gpus)
        if len(ids) < 2 or len(ids) % 2:
            raise SystemExit(
                f"--attack {args.attack} needs an even --gpus list: each "
                "shard runs the target on one GPU and the attack LLM on "
                "another (e.g. --gpus 0,1,2,3 -> 2 shards)")
        devices = [f"{a},{b}" for a, b in zip(ids[::2], ids[1::2])]
        print(f"attack needs a second device per shard: {len(devices)} "
              f"shards over pairs {devices}")
    results, payload = run_eval_shards(__file__, args,
                                       len(load_prompts(args.source)),
                                       devices=devices)
    write_json(args.out, {**payload, "results": results})
    print(f"merged {len(results)} results -> {args.out}")


def main():
    args = parse_args()
    if args.gpus:
        return run_sharded(args)
    attacker = ATTACKERS[args.attack].from_args(args)
    if args.reproduct:
        enable_reproducibility(args.seed)
    else:
        seed_all(args.seed)

    print(f"loading {MODEL_NAME} ...")
    tokenizer, model = load_model()
    if args.reproduct:
        force_math_attention()
    device = next(model.parameters()).device

    defender = DEFENDERS[args.defense].from_args(args, model)
    rows = load_prompts(args.source)[args.start: args.start + args.n]
    print(f"{len(rows)} prompts from {args.source}, attack={args.attack}, "
          f"defense={args.defense}")

    gen_config = {"steps": args.steps, "gen_length": args.gen_length,
                  "block_length": args.block_length, "temperature": args.temperature,
                  "decoder": args.decoder}
    if args.decoder == "dream":
        top_k = 50 if args.top_k is None else args.top_k
        gen_config.update(alg=args.alg, alg_temp=args.alg_temp,
                          top_p=0.95 if args.top_p is None else args.top_p,
                          top_k=top_k if top_k > 0 else None)
    else:
        gen_config.update(remasking=args.remasking, schedule=args.schedule)

    def respond(user_message):
        """One user turn through the defense -> (x, ids, cfg, shown)."""
        shown = defender.transform_prompt(user_message)
        ids = encode_prompt(tokenizer, shown, device)
        cfg = {**gen_config, **attacker.gen_overrides(ids)}
        return defender.defend(model, ids, **cfg), ids, cfg, shown

    results = []
    t_start = time.time()
    for i, row in enumerate(rows):
        # Per-prompt seed keeps generation identical under --start/--gpus sharding.
        seed_all(args.seed + int(row["index"]))
        vanilla_ids = (encode_prompt(tokenizer, defender.transform_prompt(row["prompt"]),
                                     device) if attacker.needs_vanilla else None)

        t0 = time.time()
        result = attacker.run(row, respond, tokenizer, vanilla_ids)
        elapsed = time.time() - t0

        generation, extra = result.generation, result.extra
        if result.cfg != gen_config:
            extra = {**extra, "gen_overrides": {k: v for k, v in result.cfg.items()
                                                if gen_config.get(k) != v}}
        # Graded sources carry their answer key through to eval_utility.py.
        graded = {k: row[k] for k in ("task", "answer", "subject", "category") if k in row}
        results.append({"index": row["index"], "prompt": row["prompt"], **graded,
                        "attack_prompt": result.attack_prompt, "generation": generation,
                        **extra, "num_prompt_tokens": int(result.prompt_ids.shape[1]),
                        "num_prompt_masks": int((result.prompt_ids == MASK_ID).sum()),
                        "seconds": round(elapsed, 2), **defender.result_fields()})

        # Rewrite the full results file periodically (crash safety) -- attack
        # histories make the payload large, so not every row.
        if (i + 1) % 5 == 0 or i == len(rows) - 1:
            write_json(args.out, {"model": MODEL_NAME, "config": gen_config,
                                  "attack": attacker.describe(),
                                  "defense": defender.describe(),
                                  "results": results})
        print(f"[{i + 1}/{len(rows)}] idx={row['index']} {elapsed:.1f}s :: "
              f"{generation[:100].replace(chr(10), ' ')}...", flush=True)

    summary = defender.summarize(results)
    if summary:
        print(summary)
    print(f"\nDone in {time.time() - t_start:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
