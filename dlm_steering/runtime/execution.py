"""GPU assignment and subprocess sharding."""
import json
from pathlib import Path
from .progress import CHILD_PROGRESS_ENV, task_progress


def strip_argv_flag(argv, name):
    """Drop --name value and --name=value occurrences from argv."""
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
    """Contiguous (start, n) slices covering `total` items across n_parts."""
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
    """Free MiB on a physical GPU id/UUID, or None when nvidia-smi cannot say."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", gpu_id, "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10).stdout
        return int(out.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def worker_devices(spec, procs="1", limit=2):
    """Expand --gpus into one entry per generation worker, round-robin.

    Generations are identical for any layout: every row is seeded by its index,
    so which worker runs it is irrelevant. Whether a second worker on a card
    pays depends on how launch-bound decoding is. It was 2.0x while --reproduct
    still paid the per-matmul regex (see release_cublas_env); with that gone a
    ~330-token DIJA row keeps the GPU busy by itself and two workers measured
    10% SLOWER (100 JBB rows on 2 cards: 281s vs 310s; --reproduct 291s vs
    331s). Hence the default of 1; try 2 only for short prompts.

    procs: workers per card, or "auto" to give each card as many as its free
    memory holds (up to `limit`) -- a card another job is using gets fewer
    instead of an OOM. The result interleaves cards ("0,1,0,1", never
    "0,0,1,1") so a run with fewer tasks than workers still spreads out.
    """
    ids = parse_gpu_ids(spec)
    if str(procs) == "auto":
        free = [_free_mib(g) for g in ids]
        counts = [1 if f is None else max(1, min(limit, f // WORKER_MIB))
                  for f in free]
    else:
        if int(procs) < 1:
            raise ValueError("--procs-per-gpu must be 'auto' or >= 1")
        counts = [int(procs)] * len(ids)
    return [g for rank in range(max(counts))
            for g, n in zip(ids, counts) if rank < n]


def plan_shards(spec, pairs=False):
    """Resolve --gpus into per-shard device groups, or pin this process.

    Returns [] when the spec names a single group: there is nothing to run in
    parallel, so CUDA_VISIBLE_DEVICES is set here and the caller does the work
    inline. That skips a subprocess, a second model load and the .partN files
    a one-shard run would otherwise leave behind -- which is what `--gpus 0` on
    a single-GPU machine means.

    `pairs` groups the ids two at a time, for an attack that drives a second
    model and must keep it off the target's card.

    Must be called before the first CUDA op, since it may set
    CUDA_VISIBLE_DEVICES.
    """
    import os
    ids = parse_gpu_ids(spec)
    if pairs:
        if len(ids) < 2 or len(ids) % 2:
            raise SystemExit(
                "this attack needs an even --gpus list: each shard runs the "
                "target on one GPU and the attack LLM on another "
                "(e.g. --gpus 0,1,2,3 -> 2 shards)")
        groups = [f"{a},{b}" for a, b in zip(ids[::2], ids[1::2])]
    else:
        groups = ids
    if len(groups) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = groups[0]
        return []
    return groups


def child_launcher(entry):
    """The `python ...` prefix a shard child is invoked with.

    A path ending in .py is run directly; anything else is a dotted module
    name run with -m, which is what members of the steering package need for
    their absolute imports of common.py / Evaluator.py to resolve.
    """
    import sys
    if str(entry).endswith(".py"):
        return [sys.executable, str(Path(entry).resolve())]
    return [sys.executable, "-m", str(entry)]


def spawn_shards(script, argv, gpu_ids, slices, out, extra_args=None):
    """One `python <script>` subprocess per slice on its own GPU set.

    `script` is a file path or a dotted module name -- see child_launcher.

    gpu_ids entries may be comma-separated groups ("0,1"): a shard needing a
    second model (e.g. an attack LLM) gets two visible devices per process.
    argv should already have --gpus stripped; each child is invoked with
    --start/--n/--out for its slice plus extra_args(i, gpu) when given
    (e.g. a distinct ollama --port/--gpu/--container per shard). Child
    stdout/stderr go to <part>.log next to the part file.
    Returns [(proc, part_path)].
    """
    import os
    import subprocess
    import sys
    out = Path(out)
    procs = []
    for i, ((s, n), gpu) in enumerate(zip(slices, gpu_ids)):
        part = out.with_name(f"{out.stem}.part{i}{out.suffix}")
        log = part.with_suffix(".log")
        cmd = [*child_launcher(script), *argv,
               "--start", str(s), "--n", str(n), "--out", str(part)]
        if extra_args:
            cmd += [str(a) for a in extra_args(i, gpu)]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, **CHILD_PROGRESS_ENV}
        proc = subprocess.Popen(cmd, stdout=open(log, "w"),
                                stderr=subprocess.STDOUT, env=env)
        procs.append((proc, part))
        print(f"part{i}: gpu={gpu} items {s}..+{n} -> {part} (log {log})",
              flush=True)
    return procs


def wait_merge_shards(procs):
    """Wait on spawn_shards procs; return (merged results sorted by index,
    part0's payload dict)."""
    with task_progress(procs, desc='GPU 작업', unit='샤드') as progress:
        rc = [p.wait() for p, _ in progress]
    if any(rc):
        bad = ", ".join(f"part{i} rc={r}" for i, r in enumerate(rc) if r)
        raise SystemExit(f"shards failed: {bad} -- see part logs")
    payloads = [json.loads(part.read_text()) for _, part in procs]
    results = sorted((r for p in payloads for r in p["results"]),
                     key=lambda r: r["index"])
    return results, payloads[0]


def run_eval_shards(script, args, n_items, extra_args=None, devices=None):
    """Shared --gpus launcher for eval-style entry points.

    Shards `n_items` items over args.gpus into one subprocess per GPU, waits,
    and returns (merged results, part0 payload). Caller rewrites the merged
    summary (Evaluator.summarize is a staticmethod) and writes args.out.
    `devices` overrides the parsed --gpus list with per-shard device groups.
    """
    import sys
    gpu_ids = devices if devices is not None else parse_gpu_ids(args.gpus)
    total = n_items - args.start
    if args.n is not None:
        total = min(total, args.n)
    if total <= 0:
        raise ValueError(f"no items in range: --start {args.start} --n {args.n}")
    argv = strip_argv_flag(sys.argv[1:], "--gpus")
    procs = spawn_shards(script, argv, gpu_ids,
                         shard_slices(total, len(gpu_ids), args.start),
                         args.out, extra_args)
    return wait_merge_shards(procs)
