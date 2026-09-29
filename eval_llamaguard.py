import argparse
import json
import time
from pathlib import Path
from dlm_steering.evaluation.attacks import generation_items
from dlm_steering.evaluation.cli import add_io_args, run_grading, slice_rows
from dlm_steering.evaluation.llamaguard import GUARD_MODEL, LlamaGuard4
from dlm_steering.evaluation.results import build_evaluation_payload


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    add_io_args(p, default_in="outputs/llada8b_generations.json", default_out="outputs/llada8b_llamaguard4.json")
    p.add_argument("--max-new-tokens", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--with-reference", action="store_true", help="Also score the reference_response field, if the input carries one.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data = json.loads(Path(args.inp).read_text())
    rows = slice_rows(data["results"], args)
    items = generation_items(data, rows)
    print(f"Scoring {len(items)} items from {args.inp}")
    t_start = time.time()
    make = lambda port, start: LlamaGuard4(max_new_tokens=args.max_new_tokens, with_reference=args.with_reference, batch_size=args.batch_size)
    scored, summary, head = run_grading(args, items, len(data["results"]), __file__, LlamaGuard4, make)
    payload = build_evaluation_payload(data, rows, scored, summary, source=args.inp, kind="llamaguard4", guard_model=head.get("guard_model") if head else GUARD_MODEL)
    summary = payload["summary"]
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nDone in {time.time() - t_start:.1f}s -> {args.out}")
    print(f"response: {summary['unsafe']}/{summary['total']} unsafe")
    if "reference" in summary:
        print(f"reference : {summary['reference']['unsafe']}/{summary['reference']['total']} unsafe")
    print("by category:", json.dumps(summary["by_category"], ensure_ascii=False))


if __name__ == "__main__":
    main()
