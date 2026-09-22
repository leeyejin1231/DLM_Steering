"""Shared constants, model loading and IO helpers for the steering experiments.

The target model (LLaDA-8B-Instruct or Dream-v0-Instruct-7B) is selected by
models.py from --model / DLM_MODEL before this module binds MODEL_NAME,
MASK_ID, MASK_TOKEN and EOT_ID; everything downstream imports those by value.
llada.py keeps its own LLaDA constants as the reference sampler.
"""

import glob
import json
import math
from pathlib import Path

import numpy as np
import torch

from models import MODEL, MODEL_KEY, add_model_arg  # noqa: F401  (re-exported)

MODEL_NAME = MODEL["name"]
MASK_ID = MODEL["mask_id"]
MASK_TOKEN = MODEL["mask_token"]     # text form, expanded into DIJA prompts
EOT_ID = MODEL["eot_id"]             # closes the user turn in the chat template
USER_HEADER_ID = MODEL["user_header_id"]
NEWLINE_ID = MODEL["newline_id"]
TURN_BREAKERS = tuple(MODEL["turn_breakers"])
N_LAYERS = MODEL["n_layers"]
OUT_DIR = MODEL["out_dir"]           # default home of fitted vectors/detectors
DETECTOR_LAYER = MODEL["detector_layer"]   # None: detector bundle best_layer
STEER_LAYERS = MODEL["steer_layers"]       # None: vector bundle best_layer
NEWLINE_ID = 198  # '\n' (same id in both tokenizers); locates the DIJA template

DATA_DIR = Path(__file__).parent / "data"   # populated by data_downloader.py

JBB_HARMFUL_GLOB = ("/mnt/shared/huggingface-cache/hub/datasets--JailbreakBench--JBB-Behaviors"
                    "/snapshots/*/data/harmful-behaviors.csv")
JBB_BENIGN_GLOB = JBB_HARMFUL_GLOB.replace("harmful-", "benign-")
XSTEST_GLOB = ("/mnt/shared/huggingface-cache/hub/datasets--walledai--XSTest"
               "/snapshots/*/**/*.parquet")
WJ_EVAL_GLOB = ("/mnt/shared/huggingface-cache/datasets/allenai___wildjailbreak"
                "/eval-*/0.0.0/*/*.arrow")
TRUTHFULQA_GLOB = ("/mnt/shared/huggingface-cache/hub/datasets--domenicrosati--TruthfulQA"
                   "/snapshots/*/**/*.csv")
GSM8K_GLOB = "/mnt/shared/huggingface-cache/datasets/gsm8k/main/*/*/gsm8k-test.arrow"
MMLU_GLOB = ("/mnt/shared/huggingface-cache/datasets/hails___mmlu_no_train"
             "/*/*/*/mmlu_no_train-test.arrow")
WALLEDAI_GLOB = ("/mnt/shared/huggingface-cache/hub/datasets--walledai--{}"
                 "/snapshots/*/**/*.parquet")


class ShiftedLogits(torch.nn.Module):
    """Dream's lm_head is trained with an autoregressive shift: logits[:, p]
    scores the token at position p+1. Dream's own sampler realigns them with
    cat([logits[:, :1], logits[:, :-1]], 1) before reading masked slots, and
    this wrapper does the same so model(x).logits[0, p] scores slot p exactly
    as LLaDA's does. Hidden states are untouched (they are per-position
    residual streams, which is what the detectors and steering hooks read),
    so hooks are registered on .blocks of the wrapped model as usual.
    """

    def __init__(self, model):
        super().__init__()
        self.inner = model

    def forward(self, input_ids, **kwargs):
        kwargs.setdefault("use_cache", False)
        out = self.inner(input_ids, **kwargs)
        logits = out.logits
        out.logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        return out

    @property
    def device(self):
        return self.inner.device

    @property
    def config(self):
        return self.inner.config

    @property
    def blocks(self):
        return model_blocks(self.inner)


def model_blocks(model):
    """The transformer block list hooks attach to (LLaDA, Dream, or a wrapper)."""
    if hasattr(model, "blocks"):
        return model.blocks
    if hasattr(model, "model") and hasattr(model.model, "transformer"):
        return model.model.transformer.blocks      # LLaDA
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers                  # Dream (Qwen2 layout)
    raise AttributeError("cannot locate transformer blocks on the model")


def load_model(device=None):
    """Tokenizer and eval-mode bf16 model for the selected target on `device`.

    Dream comes back wrapped in ShiftedLogits, and its tokenizer has the chat
    control tokens (<|im_start|>, <|im_end|>) marked special so
    skip_special_tokens drops them from decoded generations; their ids are
    unchanged.
    """
    from transformers import AutoModel, AutoTokenizer
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if MODEL["chat_control"]:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": list(MODEL["chat_control"])},
            replace_additional_special_tokens=False)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16).to(device).eval()
    if MODEL["shift_logits"]:
        model = ShiftedLogits(model).eval()
    return tokenizer, model


load_llada = load_model   # historical name; loads whichever model is selected


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
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def force_math_attention():
    """Restrict SDPA to the math backend -- the only backend whose numerics are
    stable across GPU architectures. Call AFTER load_model: the remote model
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


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
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


HARMFUL_SOURCES = ("jbb_harmful", "advbench", "harmbench", "strongreject", "xstest_unsafe")
# Safe sets for over-refusal: every refusal here is a utility loss.
BENIGN_SOURCES = ("truthfulqa", "xstest_safe", "jbb_benign", "wj_benign")
# Graded sets for generalisation: rows carry "answer"/"task" and are scored
# by eval_utility.py (accuracy), not by a refusal or harm judge.
UTILITY_SOURCES = ("mmlu", "gsm8k", "truthfulqa_mc", "math500")
PROMPT_SOURCES = HARMFUL_SOURCES + BENIGN_SOURCES + UTILITY_SOURCES

LETTERS = "ABCDEFGHIJKLMNOP"
MC_INSTRUCTION = "Answer with the letter of the correct choice."
MATH500_INSTRUCTION = ("Solve the problem step by step, then give the final answer on the "
                       "last line in the form 'Final answer: \\boxed{<answer>}'.")
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
        files = sorted(glob.glob(MMLU_GLOB))
        if not files:
            raise FileNotFoundError(MMLU_GLOB)
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
        local = DATA_DIR / "gsm8k_test.parquet"   # openai/gsm8k main/test, file order
        if local.exists():
            df = pd.read_parquet(local)
        else:
            files = glob.glob(GSM8K_GLOB)
            if not files:
                raise FileNotFoundError(GSM8K_GLOB)
            df = _read_arrow(files[0])
        for i, r in df.iterrows():
            gold = str(r["answer"]).split("####")[-1].strip().replace(",", "")
            rows.append({"index": int(i), "task": "gsm8k",
                         "prompt": f"{r['question']}\n\n{GSM8K_INSTRUCTION}",
                         "answer": gold, "target": None})
    elif source == "truthfulqa_mc":
        local = DATA_DIR / "truthfulqa.csv"   # domenicrosati/TruthfulQA
        df = pd.read_csv(local if local.exists() else glob.glob(TRUTHFULQA_GLOB, recursive=True)[0])
        for i, r in df.iterrows():
            best = str(r["Best Answer"]).strip()
            wrong = [a.strip() for a in str(r["Incorrect Answers"]).split(";") if a.strip()]
            choices = [best] + wrong
            random.Random(int(i)).shuffle(choices)
            rows.append({"index": int(i), "task": "truthfulqa_mc",
                         "category": str(r["Category"]),
                         "prompt": format_mc(str(r["Question"]), choices),
                         "answer": LETTERS[choices.index(best)], "target": None})
    elif source == "math500":
        # HuggingFaceH4/MATH-500 test.jsonl (500 items) in file order; answer is
        # the reference \boxed{} content, graded by eval_utility.py.
        local = DATA_DIR / "math500_test.jsonl"
        for i, line in enumerate(local.read_text().splitlines()):
            r = json.loads(line)
            rows.append({"index": i, "task": "math500", "subject": str(r["subject"]),
                         "level": int(r["level"]),
                         "prompt": f"{r['problem']}\n\n{MATH500_INSTRUCTION}",
                         "answer": str(r["answer"]), "target": None})
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
        df = pd.read_csv(local if local.exists() else glob.glob(JBB_HARMFUL_GLOB)[0])
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
        df = pd.read_parquet(glob.glob(XSTEST_GLOB, recursive=True)[0])
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
        df = pd.read_csv(local if local.exists() else glob.glob(pattern)[0])
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
            files = sorted(glob.glob(WALLEDAI_GLOB.format(name), recursive=True))
            df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        prompts = list(dict.fromkeys(df["prompt"].astype(str).tolist()))
    elif source == "truthfulqa":
        # Plain harmless questions: nothing here should ever be refused, so the
        # refusal rate is a direct read on utility damage rather than on
        # sensitivity to harmful-sounding surface words.
        df = pd.read_csv(glob.glob(TRUTHFULQA_GLOB, recursive=True)[0])
        prompts = df["Question"].tolist()
    elif source == "wj_benign":
        import pyarrow as pa
        f = glob.glob(WJ_EVAL_GLOB)[0]
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


def spawn_shards(script, argv, gpu_ids, slices, out, extra_args=None):
    """One `python <script>` subprocess per slice on its own GPU set.

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
        cmd = [sys.executable, str(Path(script).resolve()), *argv,
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
