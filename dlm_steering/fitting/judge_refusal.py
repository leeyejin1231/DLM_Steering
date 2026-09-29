import argparse
import json
import time
from pathlib import Path
from dlm_steering.evaluation.cli import add_io_args, add_ollama_args, ollama_kwargs, ollama_shard_args, run_grading, slice_rows
from dlm_steering.evaluation.refusal import LocalRefusal, Refusal


def main():
    ap = argparse.ArgumentParser()
    add_io_args(ap)
    add_ollama_args(ap)
    ap.add_argument("--judge", choices=["ollama", "local"], default="ollama", help="'ollama' is gpt-oss:20b through the podman container; 'local' runs the same rubric on a local HF model (--judge-model) and needs no podman. The two judges do not agree on every borderline answer -- pick one and keep it for the whole comparison.")
    ap.add_argument("--judge-model", default="Qwen/Qwen3-14B", help="HF model for --judge local.")
    args = ap.parse_args()
    data = json.loads(Path(args.inp).read_text())
    items = [{**r, "response": r["generation"]} for r in data["results"]]
    print(f"judging {len(items)} items from {args.inp}")
    t_start = time.time()
    local = args.judge == "local"
    args.auto_server = args.auto_server and not local
    judge_cls = LocalRefusal if local else Refusal
    make = (lambda port, start: LocalRefusal(args.judge_model)) if local else (lambda port, start: Refusal(**ollama_kwargs(args, port, start)))
    judged, summary, _ = run_grading(args, slice_rows(items, args), len(items), "dlm_steering.fitting.judge_refusal", judge_cls, make, extra_args=None if local else ollama_shard_args(args))
    payload = {"judge": f"XSTest 3-way / {args.judge_model if local else args.model}", "source": args.inp, "source_model": data.get("model"), "source_config": data.get("config"), "attack": data.get("attack"), "defense": data.get("defense"), "source_set": data.get("source"), "steering": data.get("steering"), "summary": summary, "results": judged}
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"-> {args.out}  ({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
