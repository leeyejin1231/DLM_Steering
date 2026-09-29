import os
from contextlib import nullcontext
from pathlib import Path
from dlm_steering.runtime.execution import plan_shards, run_eval_shards
from ollama_runtime import OllamaServer, OllamaServerPool
from .streaming import _item_key, _resume_stream, _run_graded


def add_io_args(p, default_in=None, default_out=None):
    p.add_argument("--in", dest="inp", default=default_in, required=default_in is None)
    p.add_argument("--out", default=default_out, required=default_out is None)
    p.add_argument("--jsonl", default=None, help="Streaming/resume file (default: <out>.jsonl).")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--n", type=int, default=None, help="grade only this many items (default all)")
    p.add_argument("--gpus", default=None, help="Comma-separated GPU ids (e.g. 0,1): one shard per GPU, merged into --out")


def add_ollama_args(p, num_predict=512):
    p.add_argument("--model", default="gpt-oss:20b")
    p.add_argument("--auto-server", action="store_true", help="Run private ollama servers (one per GPU) instead of the podman container")
    p.add_argument("--port", type=int, default=50001)
    p.add_argument("--gpu", type=int, default=1, help="nvidia.com/gpu index for the ollama container")
    p.add_argument("--container", default=None, help="podman container name (default 'ollama'; sharded runs use ollama-<port>)")
    p.add_argument("--reasoning-effort", default="low")
    p.add_argument("--workers", type=int, default=4, help="Concurrent requests per ollama server")
    p.add_argument("--startup-workers", type=int, default=2, help="Concurrent private server initializations")
    p.add_argument("--num-predict", type=int, default=num_predict, help="Maximum judge output tokens")
    p.add_argument("--timeout-sec", type=float, default=None, help="Ollama request timeout (auto server: 600s; otherwise the judge's default)")


def slice_rows(rows, args):
    return rows[args.start: args.start + args.n if args.n is not None else None]


def ollama_shard_args(args):
    return lambda i, gpu: ["--port", args.port + i, "--gpu", gpu, "--container", f"ollama-{args.port + i}"]


def ollama_kwargs(args, port=None, start_container=True):
    timeout = args.timeout_sec if args.timeout_sec is not None else (600 if args.auto_server else None)
    kw = dict(model=args.model, port=args.port if port is None else port, gpu=args.gpu, container=args.container, workers=args.workers, reasoning_effort=args.reasoning_effort, num_predict=args.num_predict, start_container=start_container)
    return kw if timeout is None else {**kw, "timeout_sec": timeout}


def run_grading(args, items, n_total, script, judge_cls, make_judge, extra_args=None):
    devices = plan_shards(args.gpus) if args.gpus else []
    auto = getattr(args, "auto_server", False)
    head = None
    if devices and auto:
        jsonl = Path(args.jsonl or Path(args.out).with_suffix(".jsonl"))
        _, done, stream = _resume_stream(jsonl, items)
        stream.close()
        with OllamaServerPool(devices, args.out, args.model, args.workers, lambda port: make_judge(port, False), sum(_item_key(it) not in done for it in items), startup_workers=args.startup_workers) as pool:
            graded = _run_graded(items, jsonl, pool.grade, workers=pool.capacity, desc=f"{judge_cls.DESC} ({args.model})")
    elif devices:
        graded, head = run_eval_shards(script, args, n_total, extra_args=extra_args, devices=devices)
    else:
        jsonl = args.jsonl or str(Path(args.out).with_suffix(".jsonl"))
        server = OllamaServer(os.environ.get("CUDA_VISIBLE_DEVICES", str(args.gpu)), args.out, args.model, args.workers) if auto else nullcontext(None)
        with server as running, make_judge(running.port if running else None, not auto) as judge:
            graded = judge.evaluate(items, output_path=jsonl)
    graded.sort(key=lambda r: r["index"])
    return graded, judge_cls.summarize(graded), head
