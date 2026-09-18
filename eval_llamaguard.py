"""Score LLaDA generations with Llama-Guard-4-12B.

Classifies the assistant turn of each (prompt, generation) pair. Pass
--with-reference to additionally score a reference_response field, when the
input JSON carries one.

Usage:
    python eval_llamaguard.py [--in llada8b_generations.json] [--out llada8b_llamaguard4.json]
"""

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

from common import (plan_shards, read_jobs, run_eval_shards,
                    run_persistent_jobs, write_json)
from Evaluator import LlamaGuard4
from attack_evaluation import generation_items, is_iterative, summarize_attack


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="outputs/llada8b_generations.json")
    p.add_argument("--out", default="outputs/llada8b_llamaguard4.json")
    p.add_argument("--jsonl", default=None,
                   help="Streaming/resume file (default: <out>.jsonl).")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--n", type=int, default=None,
                   help="score only this many items (default all)")
    p.add_argument("--gpus", default=None,
                   help="Comma-separated GPU ids (e.g. 0,1): shard items into "
                        "one subprocess per GPU and merge the part JSONs")
    p.add_argument("--jobs", help="JSON list of grading CLI argument lists; "
                   "reuse each GPU's model across files (each job supplies --out)")
    p.add_argument("--max-new-tokens", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16,
                   help="Items per Llama Guard forward (default 16: 1.9x "
                        "end-to-end on a 100-row file, more on longer ones). "
                        "Items are grouped by length so a batch is not held up "
                        "by its longest member. Batching changes the reduction "
                        "order, which can move a verdict the model is torn on; "
                        "pass 1 to grade one at a time and reproduce an older "
                        "run exactly. Lower it if the grader OOMs.")
    p.add_argument(
        "--with-reference",
        action="store_true",
        help="Also score the reference_response field, if the input carries one.",
    )
    return p.parse_args(argv)


def main(argv=None, grader=None):
    args = parse_args(argv)
    if args.jobs:
        if grader is not None:
            raise ValueError("a worker cannot launch nested jobs")
        report = run_persistent_jobs("grade", read_jobs(args.jobs), args.gpus)
        write_json(args.jobs + ".timing.json", report)
        print(f"Done in {report['seconds']:.2f}s")
        return
    if grader is not None and args.gpus:
        raise ValueError("a reused grader must run on its worker's GPU")

    data = json.loads(Path(args.inp).read_text())
    rows = data["results"][args.start:
                           args.start + args.n if args.n is not None else None]
    items = generation_items(data, rows)
    print(f"Scoring {len(items)} items from {args.inp}")

    t_start = time.time()
    # [] means one shard: plan_shards pinned this process to that GPU, so the
    # work runs inline instead of spawning a single child.
    devices = plan_shards(args.gpus) if args.gpus else []
    if devices:
        scored, head = run_eval_shards(__file__, args, len(data["results"]),
                                       devices=devices)
        summary = LlamaGuard4.summarize(scored)
        model_id = head.get("guard_model")
    else:
        jsonl = args.jsonl or str(Path(args.out).with_suffix(".jsonl"))
        if grader is not None:
            grader.max_new_tokens = args.max_new_tokens
            grader.with_reference = args.with_reference
            grader.batch_size = args.batch_size
            if grader.batch_size < 1:
                raise ValueError("batch_size must be >= 1")
        context = (nullcontext(grader) if grader is not None else
                   LlamaGuard4(max_new_tokens=args.max_new_tokens,
                               with_reference=args.with_reference,
                               batch_size=args.batch_size))
        with context as grader:
            scored = grader.evaluate(items, output_path=jsonl)
            scored.sort(key=lambda r: r["index"])
            summary = grader.summarize(scored)
        model_id = grader.model_id

    payload = {
        "guard_model": model_id,
        "source": args.inp,
        "source_model": data.get("model"),
        "source_config": data.get("config"),
        "summary": summary,
        "results": scored,
    }
    if is_iterative(data):
        payload["attempt_summary"] = summary
        summary, outcomes = summarize_attack(rows, scored, "llamaguard4")
        payload.update(summary=summary, row_results=outcomes)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    print(f"\nDone in {time.time() - t_start:.1f}s -> {args.out}")
    print(f"response: {summary['unsafe']}/{summary['total']} unsafe")
    if "reference" in summary:
        print(f"reference : {summary['reference']['unsafe']}/"
              f"{summary['reference']['total']} unsafe")
    print("by category:", json.dumps(summary["by_category"], ensure_ascii=False))


if __name__ == "__main__":
    main()
