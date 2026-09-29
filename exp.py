import argparse
import threading
import time
from pathlib import Path

from dlm_steering.attacks import ATTACKERS
from dlm_steering.runtime.constants import MODEL_KEY, MODEL_NAME, add_model_arg
from dlm_steering.runtime.data import PROMPT_SOURCES, load_prompts
from dlm_steering.runtime.models import (enable_reproducibility, force_math_attention, load_llada, release_cublas_env, seed_all)
from dlm_steering.runtime.execution import plan_shards, run_eval_shards, worker_devices
from dlm_steering.runtime.utils import write_json
from dlm_steering.defenses import DEFENDERS
from experiment_row import RowExecution
from dlm_steering.runtime.progress import task_progress


def parse_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--attack", choices=sorted(ATTACKERS), default="none")
    pre.add_argument("--defense", choices=sorted(DEFENDERS), default="none")
    known, _ = pre.parse_known_args(argv)

    p = argparse.ArgumentParser(parents=[pre])
    add_model_arg(p)
    p.add_argument("--source", choices=list(PROMPT_SOURCES), default="jbb_harmful",
                    help="harmful: jbb_harmful, advbench, harmbench, strongreject, xstest_unsafe; "
                        "benign (over-refusal): truthfulqa, xstest_safe, jbb_benign, wj_benign; "
                        "utility (accuracy via eval_utility.py): mmlu, gsm8k, truthfulqa_mc")
    p.add_argument("--out", default="outputs/exp.json")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--gen-length", type=int, default=128, 
                    help="assistant tokens to append (default 128); use 0 for DIJA prompt-span infilling only")
    p.add_argument("--block-length", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--remasking", default="low_confidence")
    p.add_argument("--schedule", default="const", choices=["const", "linear", "cosine"])
    p.add_argument("--decoder", choices=["block", "dream"], default="dream" if MODEL_KEY == "dream" else "block")
    p.add_argument("--alg", choices=["origin", "entropy", "maskgit_plus", "topk_margin"], default="origin")
    p.add_argument("--alg-temp", type=float, default=None)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reproduct", action="store_true")
    p.add_argument("--gpus", default=None,
                    help="Comma-separated GPU ids (e.g. 0,1): one subprocess per "
                        "GPU over a contiguous slice of the prompt set, parts "
                        "merged into --out. Rows are seeded by index, so the "
                        "result is identical to a single-GPU run.")
    p.add_argument("--procs-per-gpu", default="1",
                    help="Generation workers per --gpus card: an integer, or "
                        "'auto' for as many as each card's free memory holds "
                        "(max 2). Generations are identical for any value "
                        "(rows are seeded by index). Default 1: a second worker "
                        "measured ~10%% slower on ~330-token rows, where one "
                        "worker already saturates the GPU. Ignored by "
                        "two-model attacks.")
    p.add_argument("--row-workers", type=int, default=1,
                    help="Interleave this many prompt rows per shard process: "
                        "while one row's target response denoises, another "
                        "row's attacker/judge calls fill the second GPU. "
                        "Validated for --attack pap.")
    ATTACKERS[known.attack].add_args(p)
    DEFENDERS[known.defense].add_args(p)
    args = p.parse_args(argv)
    if args.procs_per_gpu != "auto" and not (args.procs_per_gpu.isdigit() and int(args.procs_per_gpu) >= 1):
        raise ValueError("--procs-per-gpu must be 'auto' or a positive integer")
    if args.model != MODEL_KEY:
        raise ValueError(f"Process loaded {MODEL_KEY}; launch exp.py --model {args.model} in a new process")
    if args.model != "dream" and args.decoder == "dream":
        raise ValueError("--decoder dream requires --model dream")
    if args.model == "dream" and args.defense == "diffuguard":
        raise ValueError("Bundled DiffuGuard supports LLaDA only; Dream needs its author's separate backend")
    if not 0 < args.top_p <= 1 or args.top_k < 0 or (args.alg_temp is not None and args.alg_temp < 0):
        raise ValueError("Invalid Dream top-p/top-k/alg-temp")
    return args


def run_sharded(args, devices):
    if ATTACKERS[args.attack].needs_second_device:
        print(f"attack needs a second device per shard: {len(devices)} shards over pairs {devices}")
    results, payload = run_eval_shards(__file__, args, len(load_prompts(args.source)), devices=devices)
    write_json(args.out, {**payload, "results": results}, compact=True)
    _summarize_shards(args, results)


def _summarize_shards(args, results):
    print(f"merged {len(results)} results -> {args.out}")
    summary = DEFENDERS[args.defense].summarize(results)
    if summary:
        print(summary)


def main(argv=None):
    args = parse_args(argv)
    if args.defense == "ours":
        required = [args.detector]
        if args.steer != "none":
            required.append(args.vector)
        if args.remask == "v3":
            required.append(args.response_detector)
        missing = [p for p in required if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(f"{args.model} defense checkpoints missing: {', '.join(missing)}; prepare model-specific bundles using steering fitters --model {args.model}")
    try:
        ATTACKERS[args.attack].validate_inputs(args)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from None
    if args.gpus:
        pairs = ATTACKERS[args.attack].needs_second_device
        devices = [] if pairs else worker_devices(args.gpus, args.procs_per_gpu)
        if len(devices) > len(set(devices)):
            print("workers per GPU: " + ", ".join(f"{g}x{devices.count(g)}" for g in dict.fromkeys(devices)))
        else:
            devices = plan_shards(args.gpus, pairs=pairs)
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
        if (not ATTACKERS[args.attack].needs_second_device and args.row_workers <= 1):
            release_cublas_env(model.device)

    all_rows = load_prompts(args.source)
    rows = all_rows[args.start: args.start + args.n]
    workers = max(1, args.row_workers)

    gen_config = {"steps": args.steps, "gen_length": args.gen_length, "block_length": args.block_length, "temperature": args.temperature, "remasking": args.remasking, "schedule": args.schedule}
    if args.decoder == "dream":
        gen_config.update(decoder="dream", alg=args.alg, alg_temp=args.alg_temp, top_p=args.top_p, top_k=args.top_k or None)

    lanes = [(ATTACKERS[args.attack].from_args(args), DEFENDERS[args.defense].from_args(args, model)) for _ in range(workers)]
    for att, _ in lanes:
        att.prepare_rows(all_rows)
    if ATTACKERS[args.attack].needs_second_device:
        lanes[0][0].llm.load()
        if args.reproduct:
            force_math_attention()
    print(f"{len(rows)} prompts from {args.source}, attack={args.attack}, defense={args.defense}, row_workers={workers}")

    results, lock = [], threading.Lock()
    jobs = iter(rows)
    t_start = time.time()

    def flush():
        write_json(args.out, {
            "model": MODEL_NAME, "config": gen_config,
            "attack": lanes[0][0].describe(),
            "defense": lanes[0][1].describe(),
            "results": sorted(results, key=lambda r: r["index"])}, compact=True)

    next_flush = 5
    progress = task_progress(total=len(rows), desc='Answer Generation', unit='row')

    def job_loop(att, dfn):
        nonlocal next_flush
        for row in jobs:  # shared iterator: next() is atomic under the GIL
            rec = RowExecution(row, att, dfn, model, tokenizer, gen_config, args.seed).run()

            with lock:
                results.append(rec)
                done = len(results)
                if done >= next_flush or done == len(rows):
                    flush()
                    next_flush = done + max(5, done // 10)
                progress.update(1)
            preview = str(rec.get("generation", "<row failed>"))
            if progress.disable:
                print(f"[{done}/{len(rows)}] idx={row['index']} {rec.get('seconds', -1)}s :: {preview[:100].replace(chr(10), ' ')}...", flush=True)
            else:
                progress.set_postfix(idx=row['index'], seconds=rec.get('seconds', -1))

    try:
        if workers == 1:
            job_loop(*lanes[0])
        else:
            threads = [threading.Thread(target=job_loop, args=lane, daemon=True) for lane in lanes]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            flush()
    finally:
        progress.close()

    ordered = sorted(results, key=lambda r: r["index"])
    summary = lanes[0][1].summarize(ordered)
    if summary:
        print(summary)
    print(f"\nDone in {time.time() - t_start:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
