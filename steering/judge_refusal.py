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
from common import run_eval_shards  # noqa: E402
from Evaluator import Refusal  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jsonl", default=None,
                    help="Streaming/resume file (default: <out>.jsonl).")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=None,
                    help="judge only this many items (default all)")
    ap.add_argument("--gpus", default=None,
                    help="Comma-separated GPU ids (e.g. 0,1): one ollama "
                         "container per GPU on --port+i, shard items and merge")
    ap.add_argument("--model", default="gpt-oss:20b")
    ap.add_argument("--port", type=int, default=50001)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--container", default=None,
                    help="podman container name (default 'ollama'; sharded runs "
                         "use ollama-<port> automatically)")
    ap.add_argument("--reasoning-effort", default="low")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--num-predict", type=int, default=512,
                    help="Must cover the judge's reasoning tokens too.")
    args = ap.parse_args()

    data = json.loads(Path(args.inp).read_text())
    items = [{**r, "response": r["generation"]} for r in data["results"]]
    print(f"judging {len(items)} items from {args.inp}")

    t_start = time.time()
    if args.gpus:
        def extra(i, gpu):
            port = args.port + i
            return ["--port", port, "--gpu", gpu,
                    "--container", f"ollama-{port}"]
        judged, _ = run_eval_shards(__file__, args, len(items),
                                    extra_args=extra)
        summary = Refusal.summarize(judged)
    else:
        items = items[args.start:
                      args.start + args.n if args.n is not None else None]
        jsonl = args.jsonl or str(Path(args.out).with_suffix(".jsonl"))
        with Refusal(model=args.model, port=args.port, gpu=args.gpu,
                     container=args.container,
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
