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

from Evaluator import GptOss20b


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="outputs/llada8b_generations_len256.json")
    p.add_argument("--out", default="outputs/llada8b_sr_len256.json")
    p.add_argument("--jsonl", default=None,
                   help="Streaming/resume file (default: <out>.jsonl).")
    p.add_argument("--model", default="gpt-oss:20b")
    p.add_argument("--port", type=int, default=50001)
    p.add_argument("--gpu", type=int, default=1,
                   help="nvidia.com/gpu index for the ollama container.")
    p.add_argument("--reasoning-effort", default="low")
    return p.parse_args()


def main():
    args = parse_args()
    data = json.loads(Path(args.inp).read_text())
    items = [{"index": r["index"], "prompt": r["prompt"],
              "response": r["generation"]} for r in data["results"]]
    print(f"Grading {len(items)} items from {args.inp}")

    jsonl = args.jsonl or str(Path(args.out).with_suffix(".jsonl"))
    with GptOss20b(model=args.model, port=args.port, gpu=args.gpu,
                   reasoning_effort=args.reasoning_effort) as grader:
        graded = grader.evaluate(items, output_path=Path(jsonl))
        graded.sort(key=lambda r: r["index"])
        summary = grader.summarize(graded)

    payload = {
        "grader": f"StrongREJECT / {args.model} (reasoning_effort={args.reasoning_effort})",
        "source": args.inp,
        "source_model": data.get("model"),
        "source_config": data.get("config"),
        "summary": summary,
        "results": graded,
    }
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    print(f"\n-> {args.out}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for g in graded:
        print(f"  idx={g['index']:>2} score={g.get('sr_score')} "
              f"refusal={g.get('sr_refusal')} conv={g.get('sr_convincing')} "
              f"spec={g.get('sr_specific')}")


if __name__ == "__main__":
    main()
