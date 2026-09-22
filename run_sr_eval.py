"""Score LLaDA generations with the StrongREJECT grader (ollama gpt-oss:20b).

Instantiating the evaluator brings the podman ollama container up if the
server is not already answering.

Usage:
    python run_sr_eval.py --in outputs/llada8b_generations_len256.json \
                          --out outputs/llada8b_sr_len256.json
"""

from contextlib import nullcontext
import os
import argparse
import json
from pathlib import Path

from common import plan_shards, run_eval_shards
from Evaluator import GptOss20b
from attack_evaluation import generation_items
from dlm_steering.evaluation.results import build_evaluation_payload


def parse_args(argv=None):
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
    p.add_argument("--auto-server", action="store_true", help="Manage a private local Ollama server automatically")
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
    p.add_argument("--startup-workers", type=int, default=2,
                   help="Concurrent private server initializations (default 2). "
                        "CUDA discovery stays serialized; 1 serializes all loading.")
    p.add_argument("--num-predict", type=int, default=1000, help="Maximum judge output tokens")
    p.add_argument("--timeout-sec", type=float, default=None, help="Ollama request timeout (auto server: 600s; otherwise: 120s)")
    args = p.parse_args(argv)
    if args.startup_workers < 1:
        p.error("--startup-workers must be >= 1")
    if args.timeout_sec is None:
        args.timeout_sec = 600 if args.auto_server else 120
    return args


def main():
    args = parse_args()
    data = json.loads(Path(args.inp).read_text())
    rows = data["results"][args.start:
                           args.start + args.n if args.n is not None else None]
    items = generation_items(data, rows)
    print(f"Grading {len(items)} items from {args.inp}")

    output = Path(args.out)
    if output.is_file():
        from evaluation_cache import matches
        try:
            saved = json.loads(output.read_text())
            if args.model == 'gpt-oss:20b' and matches(saved, items, 'gptoss'):
                print('저장된 채점 결과 재사용 (모델 로딩 없음):', output)
                print(json.dumps(saved['summary'], ensure_ascii=False, indent=2))
                return
        except (ValueError, KeyError, TypeError):
            pass

    devices = plan_shards(args.gpus) if args.gpus else []
    grader_name = (f"StrongREJECT / {args.model} "
                   f"(reasoning_effort={args.reasoning_effort})")
    if devices and args.auto_server:
        # Private servers: one work queue over every GPU instead of static
        # shards, so items are graded while later servers are still starting.
        from Evaluator import _resume_stream, _item_key, _run_graded
        from ollama_runtime import OllamaServerPool
        jsonl = Path(args.jsonl or Path(args.out).with_suffix(".jsonl"))
        _, done, stream = _resume_stream(jsonl, items)
        stream.close()

        def make_grader(port):
            return GptOss20b(model=args.model, port=port, workers=args.workers,
                             reasoning_effort=args.reasoning_effort,
                             num_predict=args.num_predict,
                             timeout_sec=args.timeout_sec, start_container=False)

        with OllamaServerPool(devices, args.out, args.model, args.workers, make_grader,
                              sum(_item_key(it) not in done for it in items),
                              startup_workers=args.startup_workers) as pool:
            graded = _run_graded(items, jsonl, pool.grade, workers=pool.capacity,
                                 desc=f"SR-Ollama ({args.model})")
        graded.sort(key=lambda r: r['index'])
        summary = GptOss20b.summarize(graded)
    elif devices:
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
        if args.auto_server:
            from ollama_runtime import OllamaServer
            device = os.environ.get('CUDA_VISIBLE_DEVICES', str(args.gpu))
            server_context = OllamaServer(device, args.out, args.model, args.workers)
        else:
            server_context = nullcontext(None)
        with server_context as server:
            with GptOss20b(model=args.model, port=server.port if server else args.port, gpu=args.gpu,
                           container=args.container, workers=args.workers,
                           reasoning_effort=args.reasoning_effort, num_predict=args.num_predict,
                           timeout_sec=args.timeout_sec, start_container=not args.auto_server) as grader:
                graded = grader.evaluate(items, output_path=Path(jsonl))
                graded.sort(key=lambda r: r['index'])
                summary = grader.summarize(graded)

    payload = build_evaluation_payload(
        data, rows, graded, summary, source=args.inp,
        kind="gpt-oss-20b", grader=grader_name)
    summary = payload["summary"]
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    print(f"\n-> {args.out}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for g in graded:
        print(f"  idx={g['index']:>2} score={g.get('sr_score')} "
              f"refusal={g.get('sr_refusal')} conv={g.get('sr_convincing')} "
              f"spec={g.get('sr_specific')}")


if __name__ == "__main__":
    main()
