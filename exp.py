"""Experiment entry: --attack X --defense Y over a prompt source.

The attack decides how the user turn is built and what gets graded; the
defense is a policy object the sampler consults every denoising step.
Attack/defense-specific flags are registered by the selected class.

Usage:
    # DIJA with the paper's refined prompts (DIJA/run_*/refine_prompt); the
    # attack sets gen_length 0, temperature 0.2 and one mask per step.
    CUDA_VISIBLE_DEVICES=1 python exp.py --attack dija --defense ours \
        --source jbb_harmful --n 100 --out outputs/dija_ours.json

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
from common import (MODEL_NAME, MASK_ID, PROMPT_SOURCES, encode_prompt,
                    enable_reproducibility, force_math_attention, load_llada,
                    load_prompts, seed_all, write_json)
from Defender import DEFENDERS


def parse_args():
    # The attack/defense choice decides which extra flags exist, so it is
    # parsed first and the selected class registers its own arguments.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--attack", choices=sorted(ATTACKERS), default="none")
    pre.add_argument("--defense", choices=sorted(DEFENDERS), default="none")
    known, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(parents=[pre])
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


def _strip_flag(argv, name):
    """Drop --name value and --name=value occurrences from argv."""
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
        elif a == name:
            skip = True
        elif not a.startswith(name + "="):
            out.append(a)
    return out


def run_sharded(args):
    """Launcher path for --gpus: one exp.py subprocess per GPU, then merge.

    Each child reruns this file with its own --start/--n/--out slice and
    CUDA_VISIBLE_DEVICES set; per-prompt seeding (seed + row index) keeps
    rows identical no matter how they are partitioned.
    """
    import json
    import math
    import os
    import subprocess
    import sys
    from pathlib import Path

    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpu_ids:
        raise ValueError("--gpus needs at least one GPU id")
    total = min(args.n, len(load_prompts(args.source)) - args.start)
    if total <= 0:
        raise ValueError(f"no prompts in range: --start {args.start} --n {args.n}")
    per = math.ceil(total / len(gpu_ids))
    out = Path(args.out)
    argv = _strip_flag(sys.argv[1:], "--gpus")

    procs, parts = [], []
    for i, gpu in enumerate(gpu_ids):
        n_i = min(per, total - i * per)
        if n_i <= 0:
            break
        part = out.with_name(f"{out.stem}.part{i}{out.suffix}")
        log = part.with_suffix(".log")
        cmd = [sys.executable, str(Path(__file__).resolve()), *argv,
               "--start", str(args.start + i * per), "--n", str(n_i),
               "--out", str(part)]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
        procs.append(subprocess.Popen(cmd, stdout=open(log, "w"),
                                      stderr=subprocess.STDOUT, env=env))
        parts.append(part)
        print(f"part{i}: gpu={gpu} rows {args.start + i * per}..+{n_i} "
              f"-> {part} (log {log})", flush=True)

    rc = [p.wait() for p in procs]
    if any(rc):
        bad = ", ".join(f"part{i} rc={r}" for i, r in enumerate(rc) if r)
        raise SystemExit(f"shards failed: {bad} -- see part logs")
    results = sorted((r for p in parts for r in json.loads(p.read_text())["results"]),
                     key=lambda r: r["index"])
    payload = json.loads(parts[0].read_text())
    write_json(out, {**payload, "results": results})
    print(f"merged {len(results)} results from {len(parts)} parts -> {out}")


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
    tokenizer, model = load_llada()
    if args.reproduct:
        force_math_attention()
    device = next(model.parameters()).device

    defender = DEFENDERS[args.defense].from_args(args, model)
    rows = load_prompts(args.source)[args.start: args.start + args.n]
    print(f"{len(rows)} prompts from {args.source}, attack={args.attack}, "
          f"defense={args.defense}")

    gen_config = {"steps": args.steps, "gen_length": args.gen_length,
                  "block_length": args.block_length, "temperature": args.temperature,
                  "remasking": args.remasking, "schedule": args.schedule}

    results = []
    t_start = time.time()
    for i, row in enumerate(rows):
        # Per-prompt seed keeps generation identical under --start/--gpus sharding.
        seed_all(args.seed + int(row["index"]))
        user_message = defender.transform_prompt(attacker.build_prompt(row))
        input_ids = encode_prompt(tokenizer, user_message, device)
        vanilla_ids = encode_prompt(tokenizer, defender.transform_prompt(row["prompt"]), device)
        row_config = {**gen_config, **attacker.gen_overrides(input_ids)}

        t0 = time.time()
        out = defender.defend(model, input_ids, **row_config)
        elapsed = time.time() - t0

        generation, extra = attacker.decode(tokenizer, out, input_ids, vanilla_ids)
        if row_config != gen_config:
            extra = {**extra, "gen_overrides": {k: v for k, v in row_config.items()
                                                if gen_config.get(k) != v}}
        # Graded sources carry their answer key through to eval_utility.py.
        graded = {k: row[k] for k in ("task", "answer", "subject", "category") if k in row}
        results.append({"index": row["index"], "prompt": row["prompt"], **graded,
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
