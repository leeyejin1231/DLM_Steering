import argparse
import json
from pathlib import Path
from dlm_steering.evaluation.attacks import generation_items
from dlm_steering.evaluation.cache import matches
from dlm_steering.evaluation.cli import add_io_args, add_ollama_args, ollama_kwargs, ollama_shard_args, run_grading, slice_rows
from dlm_steering.evaluation.ollama import GptOss20b
from dlm_steering.evaluation.results import build_evaluation_payload


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    add_io_args(p, default_in="outputs/llada8b_generations_len256.json", default_out="outputs/llada8b_sr_len256.json")
    add_ollama_args(p, num_predict=1000)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data = json.loads(Path(args.inp).read_text())
    rows = slice_rows(data["results"], args)
    items = generation_items(data, rows)
    print(f"Grading {len(items)} items from {args.inp}")
    output = Path(args.out)
    if output.is_file():
        try:
            saved = json.loads(output.read_text())
            if args.model == "gpt-oss:20b" and matches(saved, items, "gptoss"):
                print("Reuse the saved grading results (no model loading):", output)
                print(json.dumps(saved["summary"], ensure_ascii=False, indent=2))
                return
        except (ValueError, KeyError, TypeError):
            pass
    make = lambda port, start: GptOss20b(**ollama_kwargs(args, port, start))
    graded, summary, head = run_grading(args, items, len(data["results"]), __file__, GptOss20b, make, extra_args=ollama_shard_args(args))
    grader_name = head.get("grader") if head else f"StrongREJECT / {args.model} (reasoning_effort={args.reasoning_effort})"
    payload = build_evaluation_payload(data, rows, graded, summary, source=args.inp, kind="gpt-oss-20b", grader=grader_name)
    summary = payload["summary"]
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\n-> {args.out}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for g in graded:
        print(f"  idx={g['index']:>2} score={g.get('sr_score')} refusal={g.get('sr_refusal')} conv={g.get('sr_convincing')} spec={g.get('sr_specific')}")


if __name__ == "__main__":
    main()
