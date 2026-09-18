"""Score LLaDA generations with the StrongREJECT grader (ollama gpt-oss:20b).

Instantiating the evaluator brings the podman ollama container up if the
server is not already answering.

Usage:
    python run_sr_eval.py --in outputs/llada8b_generations_len256.json \
                          --out outputs/llada8b_sr_len256.json
"""

import argparse
import json
from pathlib import Path

from common import plan_shards, run_eval_shards
from Evaluator import GptOss20b
from attack_evaluation import generation_items, is_iterative, summarize_attack


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="outputs/llada8b_generations_len256.json")
    p.add_argument("--out", default="outputs/llada8b_sr_len256.json")
    p.add_argument("--jsonl", default=None,
                   help="Streaming/resume file (default: <out>.jsonl).")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--n", type=int, default=None,
                   help="grade only this many items (default all)")
    p.add_argument("--gpus", default=None,
                   help="Comma-separated GPU ids (e.g. 0,1): one ollama "
                        "container per GPU on --port+i, shard items and merge")
    p.add_argument("--model", default="gpt-oss:20b")
    p.add_argument("--port", type=int, default=50001)
    p.add_argument("--gpu", type=int, default=1,
                   help="nvidia.com/gpu index for the ollama container.")
    p.add_argument("--container", default=None,
                   help="podman container name (default 'ollama'; sharded runs "
                        "use ollama-<port> automatically)")
    p.add_argument("--reasoning-effort", default="low")
    p.add_argument("--workers", type=int, default=4,
                   help="Concurrent requests per ollama server. The run prints "
                        "the effective concurrency it actually achieved; raise "
                        "this only while that number still tracks it.")
    return p.parse_args()


def main():
    args = parse_args()
    data = json.loads(Path(args.inp).read_text())
    rows = data["results"][args.start:
                           args.start + args.n if args.n is not None else None]
    items = generation_items(data, rows)
    print(f"Grading {len(items)} items from {args.inp}")

    devices = plan_shards(args.gpus) if args.gpus else []
    if devices:
        def extra(i, gpu):
            port = args.port + i
            return ["--port", port, "--gpu", gpu,
                    "--container", f"ollama-{port}"]
        graded, head = run_eval_shards(__file__, args, len(data["results"]),
                                       extra_args=extra, devices=devices)
        summary = GptOss20b.summarize(graded)
        grader_name = head.get("grader")
    else:
        jsonl = args.jsonl or str(Path(args.out).with_suffix(".jsonl"))
        with GptOss20b(model=args.model, port=args.port, gpu=args.gpu,
                       container=args.container, workers=args.workers,
                       reasoning_effort=args.reasoning_effort) as grader:
            graded = grader.evaluate(items, output_path=Path(jsonl))
            graded.sort(key=lambda r: r["index"])
            summary = grader.summarize(graded)
        grader_name = (f"StrongREJECT / {args.model} "
                       f"(reasoning_effort={args.reasoning_effort})")

    payload = {
        "grader": grader_name,
        "source": args.inp,
        "source_model": data.get("model"),
        "source_config": data.get("config"),
        "summary": summary,
        "results": graded,
    }
    if is_iterative(data):
        payload["attempt_summary"] = summary
        summary, outcomes = summarize_attack(rows, graded, "gpt-oss-20b")
        payload.update(summary=summary, row_results=outcomes)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    print(f"\n-> {args.out}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for g in graded:
        print(f"  idx={g['index']:>2} score={g.get('sr_score')} "
              f"refusal={g.get('sr_refusal')} conv={g.get('sr_convincing')} "
              f"spec={g.get('sr_specific')}")


if __name__ == "__main__":
    main()
