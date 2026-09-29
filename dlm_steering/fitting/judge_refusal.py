import argparse
import json
import time
from pathlib import Path

from dlm_steering.runtime.execution import plan_shards, run_eval_shards
from dlm_steering.evaluation.refusal import LocalRefusal, Refusal
from dlm_steering.evaluation.streaming import _resume_stream, _item_key, _run_graded
from ollama_runtime import OllamaServerPool

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
                    help="Comma-separated GPU ids (e.g. 0,1): one ollama container per GPU on --port+i, shard items and merge")
    ap.add_argument("--model", default="gpt-oss:20b")
    ap.add_argument("--auto-server", action="store_true",
                    help="Manage private local Ollama servers (one per --gpus id) instead of podman containers, as run_sr_eval.py does.")
    ap.add_argument("--port", type=int, default=50001)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--container", default=None,
                    help="podman container name (default 'ollama'; sharded runs use ollama-<port> automatically)")
    ap.add_argument("--judge", choices=["ollama", "local"], default="ollama",
                    help="'ollama' is gpt-oss:20b through the podman container; 'local' runs the same rubric on a local HF model "
                         "(--judge-model) and needs no podman. The two judges do not agree on every borderline answer -- pick one "
                         "and keep it for the whole comparison.")
    ap.add_argument("--judge-model", default="Qwen/Qwen3-14B",
                    help="HF model for --judge local.")
    ap.add_argument("--reasoning-effort", default="low")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--num-predict", type=int, default=512,
                    help="Must cover the judge's reasoning tokens too.")
    args = ap.parse_args()

    data = json.loads(Path(args.inp).read_text())
    items = [{**r, "response": r["generation"]} for r in data["results"]]
    print(f"judging {len(items)} items from {args.inp}")

    t_start = time.time()
    devices = plan_shards(args.gpus) if args.gpus else []
    if args.auto_server and args.judge == "ollama":
        items = items[args.start: args.start + args.n if args.n is not None else None]
        jsonl = Path(args.jsonl or Path(args.out).with_suffix(".jsonl"))
        _, done, stream = _resume_stream(jsonl, items)
        stream.close()

        def make_grader(port):
            return Refusal(model=args.model, port=port, workers=args.workers, reasoning_effort=args.reasoning_effort,
                           num_predict=args.num_predict, timeout_sec=600, start_container=False)

        with OllamaServerPool(devices or [str(args.gpu)], args.out, args.model, args.workers, make_grader, sum(_item_key(it) not in done for it in items)) as pool:
            judged = _run_graded(items, jsonl, pool.grade, workers=pool.capacity, desc=f"RefusalJudge ({args.model})")
        judged.sort(key=lambda r: r["index"])
        summary = Refusal.summarize(judged)
    elif devices:
        def extra(i, gpu):
            if args.judge == "local":
                return []
            port = args.port + i
            return ["--port", port, "--gpu", gpu,
                    "--container", f"ollama-{port}"]
        judged, _ = run_eval_shards("dlm_steering.fitting.judge_refusal", args, len(items),
                                    extra_args=extra, devices=devices)
        summary = Refusal.summarize(judged)
    else:
        items = items[args.start: args.start + args.n if args.n is not None else None]
        jsonl = args.jsonl or str(Path(args.out).with_suffix(".jsonl"))
        if args.judge == "local":
            judge = LocalRefusal(args.judge_model)
        else:
            judge = Refusal(model=args.model, port=args.port, gpu=args.gpu,
                            container=args.container,
                            reasoning_effort=args.reasoning_effort,
                            num_predict=args.num_predict, workers=args.workers)
        with judge:
            judged = judge.evaluate(items, output_path=jsonl)
            judged.sort(key=lambda r: r["index"])
            summary = judge.summarize(judged)

    payload = {"judge": f"XSTest 3-way / "
                        f"{args.judge_model if args.judge == 'local' else args.model}",
               "source": args.inp, "source_model": data.get("model"),
               "source_config": data.get("config"), "attack": data.get("attack"),
               "defense": data.get("defense"),
               "source_set": data.get("source"), "steering": data.get("steering"),
               "summary": summary, "results": judged}
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"-> {args.out}  ({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
