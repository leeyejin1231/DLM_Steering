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
    python -m steering.judge_refusal --in outputs/TQA-none-v3-42.json \
        --out outputs/TQA-none-v3-42_judged.json
"""

import argparse
import threading
import time
import traceback

import torch

from Attacker import ATTACKERS
from common import (ERROR_SENTINEL, MODEL_NAME, MASK_ID, PROMPT_SOURCES,
                    encode_prompt, enable_reproducibility,
                    force_math_attention, load_llada, load_prompts,
                    plan_shards, run_eval_shards, seed_all, write_json)
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
    p.add_argument("--row-workers", type=int, default=1,
                   help="Interleave this many prompt rows per shard process: "
                        "while one row's target response denoises, another "
                        "row's attacker/judge calls fill the second GPU. "
                        "Validated for --attack pap.")
    ATTACKERS[known.attack].add_args(p)
    DEFENDERS[known.defense].add_args(p)
    return p.parse_args()


def run_sharded(args, devices):
    """Launcher path for --gpus: one exp.py subprocess per device group, merged.

    Each child reruns this file with its own --start/--n/--out slice and
    CUDA_VISIBLE_DEVICES set; per-prompt seeding (seed + row index) keeps
    rows identical no matter how they are partitioned. Attacks that drive a
    second LLM (pap/pair) get a (target, attack) GPU pair per shard so the
    two models never share a card.
    """
    if ATTACKERS[args.attack].needs_second_device:
        print(f"attack needs a second device per shard: {len(devices)} "
              f"shards over pairs {devices}")
    results, payload = run_eval_shards(__file__, args,
                                       len(load_prompts(args.source)),
                                       devices=devices)
    write_json(args.out, {**payload, "results": results})
    print(f"merged {len(results)} results -> {args.out}")
    # The children each summarised their own slice into their part log; the
    # merged number is the one worth printing (and what script/run_gated.sh
    # greps for).
    summary = DEFENDERS[args.defense].summarize(results)
    if summary:
        print(summary)


def main():
    args = parse_args()
    if args.gpus:
        # [] means one shard: plan_shards pinned this process to that GPU and
        # the run continues inline instead of spawning a single child.
        devices = plan_shards(
            args.gpus, pairs=ATTACKERS[args.attack].needs_second_device)
        if devices:
            return run_sharded(args, devices)
    if args.reproduct:
        enable_reproducibility(args.seed)
    else:
        seed_all(args.seed)

    print(f"loading {MODEL_NAME} ...")
    tokenizer, model = load_llada()
    if args.reproduct:
        force_math_attention()
    device = next(model.parameters()).device

    rows = load_prompts(args.source)[args.start: args.start + args.n]
    workers = max(1, args.row_workers)

    gen_config = {"steps": args.steps, "gen_length": args.gen_length,
                  "block_length": args.block_length, "temperature": args.temperature,
                  "remasking": args.remasking, "schedule": args.schedule}

    # One lane per row worker: attacker/defender instances hold per-response
    # state (PAIR convs, V3 recovery/audit), so interleaved rows each get a
    # private pair built on the shared (read-only) model.
    lanes = [(ATTACKERS[args.attack].from_args(args),
              DEFENDERS[args.defense].from_args(args, model))
             for _ in range(workers)]
    print(f"{len(rows)} prompts from {args.source}, attack={args.attack}, "
          f"defense={args.defense}, row_workers={workers}")

    def make_respond(att, dfn, rng):
        """Per-(lane, row) respond closures bound to that row's generator."""
        def respond(user_message):
            """One user turn through the defense -> (x, ids, cfg, shown)."""
            shown = dfn.transform_prompt(user_message)
            ids = encode_prompt(tokenizer, shown, device)
            cfg = {**gen_config, **att.gen_overrides(ids)}
            return dfn.defend(model, ids, rng=rng, **cfg), ids, cfg, shown

        def respond_batch(user_messages):
            """A batch of user turns -> list of (x, ids, cfg, shown).

            defender.defend_batch is a real batched denoising loop under
            --defense none and a sequential fallback otherwise; per-prompt
            gen_overrides that disagree also fall back to sequential calls.
            """
            shown = [dfn.transform_prompt(m) for m in user_messages]
            all_ids = [encode_prompt(tokenizer, s, device) for s in shown]
            cfgs = [{**gen_config, **att.gen_overrides(i)} for i in all_ids]
            if all(c == cfgs[0] for c in cfgs):
                outs = dfn.defend_batch(model, all_ids, rng=rng, **cfgs[0])
            else:
                outs = [dfn.defend(model, i, rng=rng, **c)
                        for i, c in zip(all_ids, cfgs)]
            return list(zip(outs, all_ids, cfgs, shown))
        return respond, respond_batch

    def graded_fields(row):
        """Answer key a graded source carries through to eval_utility.py."""
        return {k: row[k] for k in ("task", "answer", "subject", "category")
                if k in row}

    def record(result, row, elapsed, dfn):
        generation, extra = result.generation, result.extra
        if result.cfg != gen_config:
            extra = {**extra, "gen_overrides": {k: v for k, v in result.cfg.items()
                                                if gen_config.get(k) != v}}
        graded = graded_fields(row)
        return {"index": row["index"], "prompt": row["prompt"], **graded,
                "attack_prompt": result.attack_prompt, "generation": generation,
                **extra, "num_prompt_tokens": int(result.prompt_ids.shape[1]),
                "num_prompt_masks": int((result.prompt_ids == MASK_ID).sum()),
                "seconds": round(elapsed, 2), **dfn.result_fields()}

    results, lock = [], threading.Lock()
    jobs = iter(rows)
    t_start = time.time()

    def flush():
        """Write the full results file. Sorted by index so the output is
        identical regardless of completion order. Callers hold `lock`."""
        write_json(args.out, {
            "model": MODEL_NAME, "config": gen_config,
            "attack": lanes[0][0].describe(),
            "defense": lanes[0][1].describe(),
            "results": sorted(results, key=lambda r: r["index"])})

    # Crash-safety rewrites are spaced out as the payload grows instead of
    # firing every N rows: each row carries a per-step gate trace, so a fixed
    # cadence makes total write cost quadratic (817 rows x 128 steps = 31 MB,
    # ~97s of json.dumps). Backing off by ~10% keeps that near linear while
    # never risking more than a tenth of a run.
    next_flush = 5

    def job_loop(att, dfn):
        nonlocal next_flush
        for row in jobs:  # shared iterator: next() is atomic under the GIL
            # Per-prompt seed keeps generation identical under
            # --start/--gpus sharding; the per-row Generator keeps sampling
            # draws identical when rows interleave (--row-workers > 1).
            seed_all(args.seed + int(row["index"]))
            rng = torch.Generator(device=device)
            rng.manual_seed(args.seed + int(row["index"]))
            vanilla_ids = (encode_prompt(
                tokenizer, dfn.transform_prompt(row["prompt"]), device)
                if att.needs_vanilla else None)
            respond, respond_batch = make_respond(att, dfn, rng)

            t0 = time.time()
            try:
                result = att.run(row, respond, tokenizer, vanilla_ids,
                                 respond_batch=respond_batch)
                rec = record(result, row, time.time() - t0, dfn)
            except Exception:
                traceback.print_exc()
                # Every grader reads r["generation"] (eval_llamaguard,
                # run_sr_eval, judge_refusal, eval_utility), so a failed row
                # still carries one: the sentinel marks it as "nothing was
                # generated" and keeps it out of their denominators.
                rec = {"index": row["index"], "prompt": row["prompt"],
                       **graded_fields(row), "attack_prompt": None,
                       "generation": ERROR_SENTINEL,
                       "error": traceback.format_exc(limit=5)}

            with lock:
                results.append(rec)
                done = len(results)
                if done >= next_flush or done == len(rows):
                    flush()
                    next_flush = done + max(5, done // 10)
            preview = str(rec.get("generation", "<row failed>"))
            print(f"[{done}/{len(rows)}] idx={row['index']} "
                  f"{rec.get('seconds', -1)}s :: "
                  f"{preview[:100].replace(chr(10), ' ')}...", flush=True)

    if workers == 1:
        job_loop(*lanes[0])
    else:
        threads = [threading.Thread(target=job_loop, args=lane, daemon=True)
                   for lane in lanes]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        flush()

    ordered = sorted(results, key=lambda r: r["index"])
    summary = lanes[0][1].summarize(ordered)
    if summary:
        print(summary)
    print(f"\nDone in {time.time() - t_start:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
