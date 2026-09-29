import os
import sys
import json
import subprocess
import subprocess
from pathlib import Path
from .progress import CHILD_PROGRESS_ENV, task_progress


def strip_argv_flag(argv, name):
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
        elif a == name:
            skip = True
        elif not a.startswith(name + "="):
            out.append(a)
    return out


def shard_slices(total, n_parts, start=0):
    per = -(-total // n_parts)
    return [(start + i * per, min(per, total - i * per))
            for i in range(n_parts) if total - i * per > 0]


def parse_gpu_ids(spec):
    """'0,1,2' -> ['0', '1', '2']; raises on empty."""
    gpu_ids = [g.strip() for g in spec.split(",") if g.strip()]
    if not gpu_ids:
        raise ValueError("--gpus needs at least one GPU id")
    return gpu_ids


WORKER_MIB = 19 * 1024


def _free_mib(gpu_id):
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", gpu_id, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10).stdout
        return int(out.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def worker_devices(spec, procs="1", limit=2):
    ids = parse_gpu_ids(spec)
    if str(procs) == "auto":
        free = [_free_mib(g) for g in ids]
        counts = [1 if f is None else max(1, min(limit, f // WORKER_MIB)) for f in free]
    else:
        if int(procs) < 1:
            raise ValueError("--procs-per-gpu must be 'auto' or >= 1")
        counts = [int(procs)] * len(ids)
    return [g for rank in range(max(counts))
            for g, n in zip(ids, counts) if rank < n]


def plan_shards(spec, pairs=False):
    ids = parse_gpu_ids(spec)
    if pairs:
        if len(ids) < 2 or len(ids) % 2:
            raise SystemExit("this attack needs an even --gpus list: each shard runs the target on one GPU and the attack LLM on another (e.g. --gpus 0,1,2,3 -> 2 shards)")
        groups = [f"{a},{b}" for a, b in zip(ids[::2], ids[1::2])]
    else:
        groups = ids
    if len(groups) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = groups[0]
        return []
    return groups


def child_launcher(entry):
    if str(entry).endswith(".py"):
        return [sys.executable, str(Path(entry).resolve())]
    return [sys.executable, "-m", str(entry)]


def spawn_shards(script, argv, gpu_ids, slices, out, extra_args=None):
    out = Path(out)
    procs = []
    for i, ((s, n), gpu) in enumerate(zip(slices, gpu_ids)):
        part = out.with_name(f"{out.stem}.part{i}{out.suffix}")
        log = part.with_suffix(".log")
        cmd = [*child_launcher(script), *argv, "--start", str(s), "--n", str(n), "--out", str(part)]
        if extra_args:
            cmd += [str(a) for a in extra_args(i, gpu)]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, **CHILD_PROGRESS_ENV}
        proc = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env)
        procs.append((proc, part))
        print(f"part{i}: gpu={gpu} items {s}..+{n} -> {part} (log {log})", flush=True)
    return procs


def wait_merge_shards(procs):
    with task_progress(procs, desc='GPU work', unit='Shared') as progress:
        rc = [p.wait() for p, _ in progress]
    if any(rc):
        bad = ", ".join(f"part{i} rc={r}" for i, r in enumerate(rc) if r)
        raise SystemExit(f"shards failed: {bad} -- see part logs")
    payloads = [json.loads(part.read_text()) for _, part in procs]
    results = sorted((r for p in payloads for r in p["results"]), key=lambda r: r["index"])
    return results, payloads[0]


def run_eval_shards(script, args, n_items, extra_args=None, devices=None):
    gpu_ids = devices if devices is not None else parse_gpu_ids(args.gpus)
    total = n_items - args.start
    if args.n is not None:
        total = min(total, args.n)
    if total <= 0:
        raise ValueError(f"no items in range: --start {args.start} --n {args.n}")
    argv = strip_argv_flag(sys.argv[1:], "--gpus")
    procs = spawn_shards(script, argv, gpu_ids, shard_slices(total, len(gpu_ids), args.start), args.out, extra_args)
    return wait_merge_shards(procs)
