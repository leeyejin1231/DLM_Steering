"""Judge refusal vs compliance with the XSTest three-way rubric, via ollama.

Substring matching is not good enough here: "I'm sorry to hear that you're going
through this" is a compliance that trips every refusal prefix list, and "I'd
rather not go into that" is a refusal that trips none. The XSTest paper uses an
LLM judge with a full_compliance / full_refusal / partial_refusal rubric, and
that is what Evaluator.Refusal reproduces against the local gpt-oss:20b server
(which it starts itself when down).

On a *safe* prompt set, over-refusal rate = (full_refusal + partial_refusal) / n.
On an *unsafe* set the same quantity is the desired refusal rate.

Usage:
    python steering/judge_refusal.py --in outputs/or_xstest_safe_steer.json \
                                     --out outputs/or_xstest_safe_steer_judged.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Evaluator import Refusal  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jsonl", default=None,
                    help="Streaming/resume file (default: <out>.jsonl).")
    ap.add_argument("--model", default="gpt-oss:20b")
    ap.add_argument("--port", type=int, default=50001)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--reasoning-effort", default="low")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--num-predict", type=int, default=512,
                    help="Must cover the judge's reasoning tokens too.")
    args = ap.parse_args()

    data = json.loads(Path(args.inp).read_text())
    items = [{**r, "response": r["generation"]} for r in data["results"]]
    print(f"judging {len(items)} items from {args.inp}")

    jsonl = args.jsonl or str(Path(args.out).with_suffix(".jsonl"))
    t_start = time.time()
    with Refusal(model=args.model, port=args.port, gpu=args.gpu,
                 reasoning_effort=args.reasoning_effort,
                 num_predict=args.num_predict, workers=args.workers) as judge:
        judged = judge.evaluate(items, output_path=jsonl)
        judged.sort(key=lambda r: r["index"])
        summary = judge.summarize(judged)

    payload = {"judge": f"XSTest 3-way / {args.model}", "source": args.inp,
               "source_set": data.get("source"), "steering": data.get("steering"),
               "summary": summary, "results": judged}
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"-> {args.out}  ({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
