"""Shared constants, model loading and IO helpers for the steering experiments.

MODEL_NAME and MASK_ID stay canonically in llada.py (the reference sampler);
everything else shared lives here.
"""

import glob
import json
import math
import os
import threading
from pathlib import Path

import numpy as np
import torch

from llada import MODEL_NAME, MASK_ID  # noqa: F401  (re-exported)

# Serializes hook-registration -> model(x) -> hook-removal sections. Forward
# hooks are module-global, so two concurrent forwards (exp.py --row-workers)
# would otherwise fire each other's hooks. The GPU serializes the kernels
# anyway; the lock only excludes hook cross-talk.
MODEL_LOCK = threading.Lock()


def block_index(layer):
    """transformer.blocks[] index of a 1-based hidden-state layer.

    THE layer-numbering convention for this repo, stated once here instead of
    in every file that hooks a block: layer L means hidden_states[L], which is
    the OUTPUT of blocks[L-1]. So layer 1 is blocks[0] and layer 25 is
    blocks[24]. Every --layer / --detector-layer flag, every fitted bundle's
    "layers" list, and every checkpoint's best_layer use this numbering.
    """
    if layer < 1:
        raise ValueError(f"layer numbering is 1-based (layer 1 = blocks[0]); "
                         f"got {layer}")
    return layer - 1


# Layer sweep shared by the vector and detector fits: hidden_states[1..31],
# i.e. the outputs of blocks[0..30]. Both fits must sweep the same layers for
# their "best layer" picks to be comparable.
FIT_LAYERS = list(range(1, 32))

DATA_DIR = Path(__file__).parent / "data"   # populated by data_downloader.py

# Recorded as the generation of a row whose attack/defense raised, so graders
# can tell a failure apart from an empty answer and drop it from their
# denominators instead of scoring a traceback (or dying on a missing key).
ERROR_SENTINEL = "[STEERING_ERROR]"

# Dataset locations RELATIVE to a Hugging Face cache root (see _hf_roots).
# "hub/datasets--*" is the hub layout; bare "datasets/*" is the datasets
# library's own cache. data/ files win over these -- see load_prompts.
JBB_HARMFUL_GLOB = ("hub/datasets--JailbreakBench--JBB-Behaviors"
                    "/snapshots/*/data/harmful-behaviors.csv")
JBB_BENIGN_GLOB = JBB_HARMFUL_GLOB.replace("harmful-", "benign-")
XSTEST_GLOB = "hub/datasets--walledai--XSTest/snapshots/*/**/*.parquet"
WJ_EVAL_GLOB = "datasets/allenai___wildjailbreak/eval-*/0.0.0/*/*.arrow"
TRUTHFULQA_GLOB = ("hub/datasets--domenicrosati--TruthfulQA"
                   "/snapshots/*/**/*.csv")
GSM8K_GLOB = "datasets/gsm8k/main/*/*/gsm8k-test.arrow"
MMLU_GLOB = "datasets/hails___mmlu_no_train/*/*/*/mmlu_no_train-test.arrow"
WALLEDAI_GLOB = "hub/datasets--walledai--{}/snapshots/*/**/*.parquet"

# The shared mount this project was first run on. Kept as a fallback root so
# those machines keep resolving, but it is no longer assumed to exist.
LEGACY_HF_ROOT = Path("/mnt/shared/huggingface-cache")


def _hf_roots():
    """Hugging Face cache roots to search, most specific first.

    A root is the directory holding both "hub/" and "datasets/". $HF_HOME is
    that root; $HUGGINGFACE_HUB_CACHE points one level deeper at hub/, so its
    parent is taken. Roots that do not exist simply match nothing.
    """
    roots, seen = [], set()
    for value in (os.environ.get("HF_HOME"),
                  os.environ.get("HUGGINGFACE_HUB_CACHE")):
        if value:
            path = Path(value)
            roots.append(path.parent if path.name == "hub" else path)
    roots.append(Path.home() / ".cache" / "huggingface")
    roots.append(LEGACY_HF_ROOT)
    return [r for r in roots if not (str(r) in seen or seen.add(str(r)))]


def hf_glob(pattern, required=True):
    """Sorted cache files matching `pattern` under the first root that has any.

    `pattern` is relative to a cache root, so one call works against $HF_HOME,
    the default user cache, or the legacy shared mount.
    """
    for root in _hf_roots():
        hits = sorted(glob.glob(str(root / pattern), recursive=True))
        if hits:
            return hits
    if required:
        raise FileNotFoundError(
            f"no Hugging Face cache file matches {pattern!r} under any of "
            f"{[str(r) for r in _hf_roots()]} -- set HF_HOME, or run "
            f"data_downloader.py to populate {DATA_DIR}")
    return []


def load_llada(device=None):
    """LLaDA tokenizer and eval-mode bf16 model on `device`.

    device_map puts each checkpoint shard straight on the target device.
    `.to(device)` instead materialises the whole 16 GB state dict in CPU RAM
    first, which measures ~3x slower (15.6s -> 5.0s) for bit-identical weights.
    """
    from transformers import AutoModel, AutoTokenizer
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16,
                                      device_map={"": device}).eval()
    return tokenizer, model


def seed_all(seed):
    """Seed python/np/torch RNGs. Applied always so runs are repeatable."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def enable_reproducibility(seed=42):
    """Bitwise-deterministic generation: fixed seeds + deterministic kernels.

    Must run before the first CUDA op so CUBLAS_WORKSPACE_CONFIG is already
    set when the cublas handle is created."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def force_math_attention():
    """Restrict SDPA to the math backend -- the only backend whose numerics are
    stable across GPU architectures. Call AFTER load_llada: the remote model
    code re-enables flash_sdp in __init__."""
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def prompt_token_ids(tokenizer, user_message):
    """Chat-template a user message; return the token id list."""
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        add_generation_prompt=True, tokenize=False)
    return tokenizer(formatted)["input_ids"]


def encode_prompt(tokenizer, user_message, device):
    """Chat-template a user message; return (1, L) token ids on `device`."""
    return torch.tensor(prompt_token_ids(tokenizer, user_message),
                        device=device).unsqueeze(0)


def write_json(path, payload, *, compact=False):
    with open(path, "w", encoding="utf-8") as f:
        if compact:
            # dumps uses the C encoder for compact JSON. A single write also
            # avoids streaming thousands of tiny fragments for gate traces.
            f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        else:
            json.dump(payload, f, ensure_ascii=False, indent=2)


def step_scale(schedule, i, n_steps):
    """alpha multiplier at denoising step i of n_steps."""
    if schedule == "const":
        return 1.0
    frac = i / max(1, n_steps - 1)
    if schedule == "linear":
        return 1.0 - frac
    if schedule == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * frac))
    raise ValueError(schedule)


def auroc(pos, neg):
    """Rank-based AUROC of pos scoring above neg."""
    x = np.concatenate([pos, neg])
    order = x.argsort()
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    # Average ranks over ties so exact duplicates score 0.5, not 0 or 1.
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    n_p, n_n = len(pos), len(neg)
    return (ranks[:n_p].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n)


HARMFUL_SOURCES = ("jbb_harmful", "advbench", "harmbench", "strongreject",
                   "xstest_unsafe", "wj_unsafe")
# Safe sets for over-refusal: every refusal here is a utility loss.
BENIGN_SOURCES = ("truthfulqa", "xstest_safe", "jbb_benign", "wj_benign")
# Graded sets for generalisation: rows carry "answer"/"task" and are scored
# by eval_utility.py (accuracy), not by a refusal or harm judge.
UTILITY_SOURCES = ("mmlu", "gsm8k", "truthfulqa_mc")
PROMPT_SOURCES = HARMFUL_SOURCES + BENIGN_SOURCES + UTILITY_SOURCES

LETTERS = "ABCDEFGHIJKLMNOP"
MC_INSTRUCTION = "Answer with the letter of the correct choice."
GSM8K_INSTRUCTION = ("Solve the problem step by step, then give the final numeric "
                     "answer on the last line in the form '#### <number>'.")


def _read_arrow(path):
    import pyarrow as pa
    with pa.memory_map(path) as src:
        try:
            return pa.ipc.open_stream(src).read_all().to_pandas()
        except pa.ArrowInvalid:
            return pa.ipc.open_file(src).read_all().to_pandas()


def format_mc(question, choices, header=None):
    """Zero-shot multiple-choice prompt: optional header, stem, lettered options."""
    lines = [header, "", question] if header else [question]
    lines += [f"{LETTERS[i]}. {c}" for i, c in enumerate(choices)]
    lines += ["", MC_INSTRUCTION]
    return "\n".join(lines)


def load_utility_prompts(source):
    """Graded rows {"index", "prompt", "target": None, "task", "answer", ...}.

    mmlu: all 57 subjects' test split (14042 items) in a fixed random order
          (seed 0) so --n takes a subject-mixed slice. answer = letter A-D.
    gsm8k: test split (1319) in file order. answer = number after '####'.
    truthfulqa_mc: MC1-style; best answer + incorrect answers shuffled per
          item (seed = index). answer = letter.
    """
    import random
    import pandas as pd
    rows = []
    if source == "mmlu":
        files = hf_glob(MMLU_GLOB)
        df = pd.concat([_read_arrow(f) for f in files], ignore_index=True)
        order = list(range(len(df)))
        random.Random(0).shuffle(order)
        for i, j in enumerate(order):
            r = df.iloc[j]
            subject = str(r["subject"]).replace("_", " ")
            rows.append({"index": i, "task": "mmlu", "subject": str(r["subject"]),
                         "prompt": format_mc(str(r["question"]), list(r["choices"]),
                                             f"The following is a multiple choice question about {subject}."),
                         "answer": LETTERS[int(r["answer"])], "target": None})
    elif source == "gsm8k":
        df = _read_arrow(hf_glob(GSM8K_GLOB)[0])
        for i, r in df.iterrows():
            gold = str(r["answer"]).split("####")[-1].strip().replace(",", "")
            rows.append({"index": int(i), "task": "gsm8k",
                         "prompt": f"{r['question']}\n\n{GSM8K_INSTRUCTION}",
                         "answer": gold, "target": None})
    elif source == "truthfulqa_mc":
        df = pd.read_csv(hf_glob(TRUTHFULQA_GLOB)[0])
        for i, r in df.iterrows():
            best = str(r["Best Answer"]).strip()
            wrong = [a.strip() for a in str(r["Incorrect Answers"]).split(";") if a.strip()]
            choices = [best] + wrong
            random.Random(int(i)).shuffle(choices)
            rows.append({"index": int(i), "task": "truthfulqa_mc",
                         "category": str(r["Category"]),
                         "prompt": format_mc(str(r["Question"]), choices),
                         "answer": LETTERS[choices.index(best)], "target": None})
    else:
        raise ValueError(source)
    return rows


def load_prompts(source):
    """Prompt sets as {"index", "prompt", "target"} rows.

    Harmful: jbb_harmful (with Target), advbench, harmbench, strongreject, xstest_unsafe.
    Benign (over-refusal): truthfulqa, xstest_safe, jbb_benign, wj_benign.
    xstest_safe/xstest_unsafe are the XSTest contrast pair: safe prompts with
    harmful-sounding words vs. their genuinely unsafe counterparts.
    Utility (accuracy): mmlu, gsm8k, truthfulqa_mc; see load_utility_prompts.
    Only jbb_harmful carries a target string; the rest use target=None.
    """
    import pandas as pd
    if source in UTILITY_SOURCES:
        return load_utility_prompts(source)
    if source == "jbb_harmful":
        local = DATA_DIR / "jbb_harmful.csv"
        df = pd.read_csv(local if local.exists() else hf_glob(JBB_HARMFUL_GLOB)[0])
        return [{"index": int(r["Index"]), "prompt": str(r["Goal"]), "target": str(r["Target"])}
                for _, r in df.iterrows()]
    if source in PROMPT_SOURCES:
        return [{"index": i, "prompt": p, "target": None}
                for i, p in enumerate(load_eval_prompts(source, None))]
    raise ValueError(source)


def load_eval_prompts(source, limit):
    """Benchmark prompt lists as plain strings.

    Sources: xstest_safe/xstest_unsafe, jbb_benign/jbb_harmful, advbench,
    harmbench, strongreject, truthfulqa, wj_benign.
    """
    import pandas as pd
    if source.startswith("xstest"):
        local = DATA_DIR / "xstest.parquet"
        df = pd.read_parquet(local if local.exists() else hf_glob(XSTEST_GLOB)[0])
        # "xstest_unsafe".endswith("safe") is True, so match the suffix explicitly.
        want = "unsafe" if source.endswith("_unsafe") else "safe"
        df = df[df["label"] == want]
        prompts = df["prompt"].tolist()
    elif source in ("jbb_benign", "jbb_harmful"):
        # Each benign behaviour is the index-matched counterpart of a harmful one
        # ("fictional story about heroin use" vs "defamatory article claiming a
        # president is addicted to heroin"), so topic is held roughly constant
        # and only permissibility differs.
        local = DATA_DIR / f"{source}.csv"
        pattern = JBB_BENIGN_GLOB if source == "jbb_benign" else JBB_HARMFUL_GLOB
        df = pd.read_csv(local if local.exists() else hf_glob(pattern)[0])
        prompts = df["Goal"].tolist()
    elif source in ("advbench", "harmbench", "strongreject"):
        # Out-of-distribution harmful benchmarks: bare requests in attack styles
        # the steering and detector directions never saw. Shards are concatenated
        # and de-duplicated because some of these ship several splits.
        local = DATA_DIR / f"{source}.parquet"
        if local.exists():
            df = pd.read_parquet(local)
        else:
            name = {"advbench": "AdvBench", "harmbench": "HarmBench",
                    "strongreject": "StrongREJECT"}[source]
            files = hf_glob(WALLEDAI_GLOB.format(name))
            df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        prompts = list(dict.fromkeys(df["prompt"].astype(str).tolist()))
    elif source == "truthfulqa":
        # Plain harmless questions: nothing here should ever be refused, so the
        # refusal rate is a direct read on utility damage rather than on
        # sensitivity to harmful-sounding surface words.
        df = pd.read_csv(hf_glob(TRUTHFULQA_GLOB)[0])
        prompts = df["Question"].tolist()
    elif source == "wj_unsafe":
        # The WildJailbreak adversarial-harmful prompts the steering vector and
        # detector were fitted on. Rows 0..19 are the held-out eval split the
        # gated pipeline reports; the fits use rows 20+.
        df = pd.read_csv(DATA_DIR / "llada8b_wild_unsafe_only.csv")
        prompts = df["prompt"].tolist()
    elif source == "wj_benign":
        import pyarrow as pa
        f = hf_glob(WJ_EVAL_GLOB)[0]
        with pa.memory_map(f) as src:
            ev = pa.ipc.open_stream(src).read_all().to_pandas()
        prompts = ev[ev["data_type"] == "adversarial_benign"]["adversarial"].tolist()
    else:
        raise ValueError(source)
    return prompts[:limit] if limit else prompts


def load_detector_bundle(detector_path):
    """The raw detector checkpoint dict (vector, layers, gen_length, ...)."""
    return torch.load(detector_path, map_location="cpu")


def load_detector(detector_path, layer=None, device=None, threshold=None):
    """Detector vector + resolved layer + gate threshold.

    layer=None uses the bundle's best_layer; threshold=None reads the
    gate_threshold.json sitting next to the bundle.
    """
    db = load_detector_bundle(detector_path)
    layer = layer or db["best_layer"]
    det_vec = db["vector"][db["layers"].index(layer)]
    if device is not None:
        det_vec = det_vec.to(device)
    if threshold is None:
        threshold = json.loads(
            (Path(detector_path).parent / "gate_threshold.json").read_text())["threshold"]
    return det_vec, layer, threshold


def steer_vector_at(bundle, layer=None, device=None):
    """(v, layer, act_norm) from a loaded steering bundle; None = best_layer."""
    layer = layer or bundle["best_layer"]
    li = bundle["layers"].index(layer)
    v = bundle["vector"][li]
    if device is not None:
        v = v.to(device)
    return v, layer, bundle["mean_act_norm"][li]


# ---------------------------------------------------------------------------
# --gpus sharding: one subprocess per GPU over --start/--n slices, then merge.
# Shared by exp.py and the eval entry points (eval_llamaguard, run_sr_eval,
# steering/judge_refusal).
# ---------------------------------------------------------------------------

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
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
        proc = subprocess.Popen(cmd, stdout=open(log, "w"),
                                stderr=subprocess.STDOUT, env=env)
        procs.append((proc, part))
        print(f"part{i}: gpu={gpu} items {s}..+{n} -> {part} (log {log})",
              flush=True)
    return procs


def wait_merge_shards(procs):
    """Wait on spawn_shards procs; return (merged results sorted by index,
    part0's payload dict)."""
    rc = [p.wait() for p, _ in procs]
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


_persistent_model = None


def _init_persistent_worker(gpus):
    # No CUDA operations run before pinning this spawned worker.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus.get()


def _run_persistent_job(kind, argv):
    from contextlib import redirect_stderr, redirect_stdout
    import time
    global _persistent_model
    started = time.perf_counter()
    if kind == "generate":
        import exp as entry
    else:
        import eval_llamaguard as entry
    args = entry.parse_args(argv)
    log = Path(args.out).with_suffix(".log")
    with log.open("w") as stream, redirect_stdout(stream), redirect_stderr(stream):
        if _persistent_model is None:
            if kind == "generate":
                if args.reproduct:
                    enable_reproducibility(args.seed)
                else:
                    seed_all(args.seed)
                _persistent_model = load_llada()
            else:
                from Evaluator import LlamaGuard4
                _persistent_model = LlamaGuard4(
                    max_new_tokens=args.max_new_tokens,
                    with_reference=args.with_reference,
                    batch_size=args.batch_size)
        if kind == "generate":
            entry.main(argv, loaded=_persistent_model)
        else:
            entry.main(argv, grader=_persistent_model)
    return {"out": args.out, "seconds": time.perf_counter() - started,
            "gpu": os.environ["CUDA_VISIBLE_DEVICES"]}


def read_jobs(path):
    """Read CLI argument lists shared by generation and grading entry points."""
    jobs = json.loads(Path(path).read_text())
    if not isinstance(jobs, list) or any(
            not isinstance(job, list) or not all(isinstance(a, str) for a in job)
            for job in jobs):
        raise ValueError("jobs must be a JSON list of string argument lists")
    return jobs


def run_persistent_jobs(kind, jobs, gpus, *, chunk_size=0):
    """Reuse one model per GPU worker across CLI jobs; optionally chunk rows."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp
    import time
    if kind not in ("generate", "grade") or chunk_size < 0:
        raise ValueError("invalid job kind or chunk size")
    gpus = parse_gpu_ids(gpus or os.environ.get("CUDA_VISIBLE_DEVICES") or "0")
    if kind == "grade" and (chunk_size or len(set(gpus)) != len(gpus)):
        raise ValueError("grading uses whole files and one worker per distinct GPU")
    if not jobs:
        return {"kind": kind, "seconds": 0.0, "workers": 0, "tasks": []}
    if kind == "generate":
        import exp as entry
        from Attacker import ATTACKERS
    else:
        import eval_llamaguard as entry
    tasks, merges, outputs, reproducibility = [], [], set(), set()
    for argv in jobs:
        cfg = entry.parse_args(argv)
        if cfg.gpus or cfg.jobs or not any(a == "--out" or a.startswith("--out=") for a in argv):
            raise ValueError("each job needs an explicit --out and must omit --gpus/--jobs")
        out = Path(cfg.out).resolve()
        if out in outputs:
            raise ValueError(f"duplicate output: {out}")
        outputs.add(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if kind == "generate":
            ATTACKERS[cfg.attack].validate_inputs(cfg)
            if ATTACKERS[cfg.attack].needs_second_device:
                raise ValueError("persistent generation currently supports single-model attacks")
            reproducibility.add(cfg.reproduct)
        if kind == "generate" and chunk_size:
            count = min(cfg.n, len(load_prompts(cfg.source)) - cfg.start)
            if count <= 0:
                raise ValueError(f"empty input range for {out}")
            base = argv
            for flag in ("--out", "--start", "--n"):
                base = strip_argv_flag(base, flag)
            parts = []
            part_dir = out.parent / ".parts" / out.stem
            part_dir.mkdir(parents=True, exist_ok=True)
            for i, offset in enumerate(range(0, count, chunk_size)):
                part = part_dir / f"chunk{i}.json"
                parts.append(part)
                tasks.append([*base, "--start", str(cfg.start + offset),
                              "--n", str(min(chunk_size, count - offset)),
                              "--out", str(part)])
            merges.append((out, parts))
        else:
            tasks.append(argv)
    if len(reproducibility) > 1:
        raise ValueError("all generation jobs must use the same --reproduct setting")

    started = time.perf_counter()
    context = mp.get_context("spawn")
    devices = context.Queue()
    workers = min(len(gpus), len(tasks))
    for gpu in gpus[:workers]:
        devices.put(gpu)
    records = []
    owners = {str(part): (out, parts) for out, parts in merges for part in parts}
    pending = {out: len(parts) for out, parts in merges}
    with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                             initializer=_init_persistent_worker, initargs=(devices,)) as pool:
        futures = [pool.submit(_run_persistent_job, kind, argv) for argv in tasks]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(f"done {record['out']} on GPU {record['gpu']}", flush=True)
            if record["out"] in owners:
                out, parts = owners[record["out"]]
                pending[out] -= 1
                if pending[out] == 0:
                    # Save completed conditions while other conditions run.
                    payloads = [json.loads(part.read_text()) for part in parts]
                    results = sorted((r for p in payloads for r in p["results"]),
                                     key=lambda r: r["index"])
                    write_json(out, {**payloads[0], "results": results}, compact=True)
    devices.close()
    report = {"kind": kind, "seconds": time.perf_counter() - started,
              "workers": workers, "tasks": records}
    return report
