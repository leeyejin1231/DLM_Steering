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
from pathlib import Path

from Evaluator import LlamaGuard4


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="outputs/llada8b_generations.json")
    p.add_argument("--out", default="outputs/llada8b_llamaguard4.json")
    p.add_argument("--jsonl", default=None,
                   help="Streaming/resume file (default: <out>.jsonl).")
    p.add_argument("--max-new-tokens", type=int, default=20)
    p.add_argument(
        "--with-reference",
        action="store_true",
        help="Also score the reference_response field, if the input carries one.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    data = json.loads(Path(args.inp).read_text())
    items = [
        {"index": r["index"], "prompt": r["prompt"], "response": r["generation"],
         **({"reference_response": r["reference_response"]}
            if r.get("reference_response") else {})}
        for r in data["results"]
    ]
    print(f"Scoring {len(items)} items from {args.inp}")

    jsonl = args.jsonl or str(Path(args.out).with_suffix(".jsonl"))
    t_start = time.time()
    with LlamaGuard4(max_new_tokens=args.max_new_tokens,
                     with_reference=args.with_reference) as grader:
        scored = grader.evaluate(items, output_path=jsonl)
        scored.sort(key=lambda r: r["index"])
        summary = grader.summarize(scored)

    payload = {
        "guard_model": grader.model_id,
        "source": args.inp,
        "source_model": data.get("model"),
        "source_config": data.get("config"),
        "summary": summary,
        "results": scored,
    }
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    print(f"\nDone in {time.time() - t_start:.1f}s -> {args.out}")
    print(f"response: {summary['unsafe']}/{summary['total']} unsafe")
    if "reference" in summary:
        print(f"reference : {summary['reference']['unsafe']}/"
              f"{summary['reference']['total']} unsafe")
    print("by category:", json.dumps(summary["by_category"], ensure_ascii=False))


if __name__ == "__main__":
    main()
