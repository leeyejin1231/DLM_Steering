import os
import glob
import random
import pyarrow as pa
import pandas as pd
import pandas as pd
from pathlib import Path
from dlm_steering.paths import DATA_DIR


JBB_HARMFUL_GLOB = ("hub/datasets--JailbreakBench--JBB-Behaviors/snapshots/*/data/harmful-behaviors.csv")
JBB_BENIGN_GLOB = JBB_HARMFUL_GLOB.replace("harmful-", "benign-")
XSTEST_GLOB = "hub/datasets--walledai--XSTest/snapshots/*/**/*.parquet"
WJ_EVAL_GLOB = "datasets/allenai___wildjailbreak/eval-*/0.0.0/*/*.arrow"
TRUTHFULQA_GLOB = ("hub/datasets--domenicrosati--TruthfulQA/snapshots/*/**/*.csv")
GSM8K_GLOB = "datasets/gsm8k/main/*/*/gsm8k-test.arrow"
MMLU_GLOB = "datasets/hails___mmlu_no_train/*/*/*/mmlu_no_train-test.arrow"
WALLEDAI_GLOB = "hub/datasets--walledai--{}/snapshots/*/**/*.parquet"
LEGACY_HF_ROOT = Path("/mnt/shared/huggingface-cache")


def _hf_roots():
    roots, seen = [], set()
    for value in (os.environ.get("HF_HOME"), os.environ.get("HUGGINGFACE_HUB_CACHE")):
        if value:
            path = Path(value)
            roots.append(path.parent if path.name == "hub" else path)
    roots.append(Path.home() / ".cache" / "huggingface")
    roots.append(LEGACY_HF_ROOT)
    return [r for r in roots if not (str(r) in seen or seen.add(str(r)))]


def hf_glob(pattern, required=True):
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


HARMFUL_SOURCES = ("jbb_harmful", "advbench", "harmbench", "strongreject", "xstest_unsafe", "wj_unsafe")
BENIGN_SOURCES = ("truthfulqa", "xstest_safe", "jbb_benign", "wj_benign")
UTILITY_SOURCES = ("mmlu", "gsm8k", "math500", "truthfulqa_mc")
PROMPT_SOURCES = HARMFUL_SOURCES + BENIGN_SOURCES + UTILITY_SOURCES
LETTERS = "ABCDEFGHIJKLMNOP"
MC_INSTRUCTION = "Answer with the letter of the correct choice."
GSM8K_INSTRUCTION = ("Solve the problem step by step, then give the final numeric answer on the last line in the form '#### <number>'.")
MATH_INSTRUCTION = ("Solve the problem step by step, then put the final answer in \\boxed{}.")


def _read_arrow(path):
    with pa.memory_map(path) as src:
        try:
            return pa.ipc.open_stream(src).read_all().to_pandas()
        except pa.ArrowInvalid:
            return pa.ipc.open_file(src).read_all().to_pandas()


def format_mc(question, choices, header=None):
    lines = [header, "", question] if header else [question]
    lines += [f"{LETTERS[i]}. {c}" for i, c in enumerate(choices)]
    lines += ["", MC_INSTRUCTION]
    return "\n".join(lines)


def _truthfulqa_csv():
    local = DATA_DIR / "truthfulqa.csv"
    return local if local.exists() else hf_glob(TRUTHFULQA_GLOB)[0]


def load_utility_prompts(source):
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
                         "prompt": format_mc(str(r["question"]), list(r["choices"]), f"The following is a multiple choice question about {subject}."),
                         "answer": LETTERS[int(r["answer"])], "target": None})
    elif source == "gsm8k":
        local = DATA_DIR / "gsm8k.parquet"
        df = pd.read_parquet(local) if local.exists() else _read_arrow(hf_glob(GSM8K_GLOB)[0])
        for i, r in df.iterrows():
            gold = str(r["answer"]).split("####")[-1].strip().replace(",", "")
            rows.append({"index": int(i), "task": "gsm8k",
                         "prompt": f"{r['question']}\n\n{GSM8K_INSTRUCTION}",
                         "answer": gold, "target": None})
    elif source == "math500":
        df = pd.read_json(DATA_DIR / "math500.jsonl", lines=True, dtype=False)
        for i, r in df.iterrows():
            rows.append({"index": int(i), "task": "math500",
                         "subject": str(r["subject"]), "level": int(r["level"]),
                         "prompt": f"{r['problem']}\n\n{MATH_INSTRUCTION}",
                         "answer": str(r["answer"]), "target": None})
    elif source == "truthfulqa_mc":
        df = pd.read_csv(_truthfulqa_csv())
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
    if source.startswith("xstest"):
        local = DATA_DIR / "xstest.parquet"
        df = pd.read_parquet(local if local.exists() else hf_glob(XSTEST_GLOB)[0])
        want = "unsafe" if source.endswith("_unsafe") else "safe"
        df = df[df["label"] == want]
        prompts = df["prompt"].tolist()
    elif source in ("jbb_benign", "jbb_harmful"):
        local = DATA_DIR / f"{source}.csv"
        pattern = JBB_BENIGN_GLOB if source == "jbb_benign" else JBB_HARMFUL_GLOB
        df = pd.read_csv(local if local.exists() else hf_glob(pattern)[0])
        prompts = df["Goal"].tolist()
    elif source in ("advbench", "harmbench", "strongreject"):
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
        df = pd.read_csv(_truthfulqa_csv())
        prompts = df["Question"].tolist()
    elif source == "wj_unsafe":
        df = pd.read_csv(DATA_DIR / "llada8b_wild_unsafe_only.csv")
        prompts = df["prompt"].tolist()
    elif source == "wj_benign":
        f = hf_glob(WJ_EVAL_GLOB)[0]
        with pa.memory_map(f) as src:
            ev = pa.ipc.open_stream(src).read_all().to_pandas()
        prompts = ev[ev["data_type"] == "adversarial_benign"]["adversarial"].tolist()
    else:
        raise ValueError(source)
    return prompts[:limit] if limit else prompts
