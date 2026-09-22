"""Download the exp.py prompt datasets into ./data.

    jbb_harmful   -> data/jbb_harmful.csv     (JailbreakBench/JBB-Behaviors)
    advbench      -> data/advbench.parquet    (walledai/AdvBench)
    harmbench     -> data/harmbench.parquet   (walledai/HarmBench)
    strongreject  -> data/strongreject.parquet (the authors' own CSV)
    xstest         -> data/xstest.parquet      (the authors' own CSV)
    gsm8k         -> data/gsm8k.parquet       (openai/gsm8k, main/test, 1319 rows)
    math500       -> data/math500.jsonl       (HuggingFaceH4/MATH-500, 500 rows)
    truthfulqa    -> data/truthfulqa.csv      (domenicrosati/TruthfulQA)

The walledeval repos ship several parquet shards/splits; they are
concatenated and de-duplicated like common.load_eval_prompts does.
They are gated: accept access on the HF dataset page and authenticate
(huggingface-cli login or HF_TOKEN) before running.

StrongREJECT comes from the authors' repository rather than walledai's
mirror, which is gated. It is also the set DIJA refined: the 313 prompts
here are byte-identical, in the same order, to the "vanilla prompt" keys in
DIJA/run_strongreject/refine_prompt, and `--attack dija` looks its refined
prompt up by that exact string. Swapping in another edition of StrongREJECT
would break that lookup.

Usage:
    python data_downloader.py            # all four
    python data_downloader.py advbench   # one only
"""

import shutil
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download

DATA_DIR = Path(__file__).parent / "data"

WALLEDAI = {"advbench": "walledai/AdvBench", "harmbench": "walledai/HarmBench"}
STRONGREJECT_CSV = ("https://raw.githubusercontent.com/alexandrasouly/strongreject"
                    "/main/strongreject_dataset/strongreject_dataset.csv")
# Both halves live in one file; load_eval_prompts filters on the label column,
# so xstest_safe and xstest_unsafe both read this.
XSTEST_CSV = ("https://raw.githubusercontent.com/paul-rottger/xstest"
              "/main/xstest_prompts.csv")


def download_jbb_harmful():
    src = hf_hub_download("JailbreakBench/JBB-Behaviors",
                          "data/harmful-behaviors.csv", repo_type="dataset")
    df = pd.read_csv(src)
    out = DATA_DIR / "jbb_harmful.csv"
    df.to_csv(out, index=False)
    print(f"jbb_harmful: {len(df)} rows -> {out}")


def download_walledeval(source):
    src = snapshot_download(WALLEDAI[source], repo_type='dataset')
    files = sorted(Path(src).rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet shards under {src}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(subset="prompt")
    out = DATA_DIR / f"{source}.parquet"
    df.to_parquet(out, index=False)
    print(f"{source}: {len(df)} rows -> {out}")


def download_strongreject():
    df = pd.read_csv(STRONGREJECT_CSV).rename(columns={"forbidden_prompt": "prompt"})
    out = DATA_DIR / "strongreject.parquet"
    df.to_parquet(out, index=False)
    print(f"strongreject: {len(df)} rows -> {out}")


def download_xstest():
    df = pd.read_csv(XSTEST_CSV)
    out = DATA_DIR / "xstest.parquet"
    df.to_parquet(out, index=False)
    counts = df["label"].value_counts().to_dict()
    print(f"xstest: {len(df)} rows {counts} -> {out}")


def download_gsm8k():
    src = hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet",
                          repo_type="dataset")
    df = pd.read_parquet(src)
    out = DATA_DIR / "gsm8k.parquet"
    df.to_parquet(out, index=False)
    print(f"gsm8k: {len(df)} rows -> {out}")


def download_math500():
    src = hf_hub_download("HuggingFaceH4/MATH-500", "test.jsonl", repo_type="dataset")
    out = DATA_DIR / "math500.jsonl"
    shutil.copyfile(src, out)
    print(f"math500: {sum(1 for _ in out.open())} rows -> {out}")


def download_truthfulqa():
    src = hf_hub_download("domenicrosati/TruthfulQA", "train.csv", repo_type="dataset")
    out = DATA_DIR / "truthfulqa.csv"
    shutil.copyfile(src, out)
    print(f"truthfulqa: {len(pd.read_csv(out))} rows -> {out}")


def download_alpaca():
    """Ordinary instructions, used as the response detector's benign arm.

    WildJailbreak alone gives that detector only adversarial prompts, so plain
    question answering sits off-distribution; see steering/fit_response_detector.
    """
    src = hf_hub_download("tatsu-lab/alpaca",
                          "data/train-00000-of-00001-a09b74b3ef9c3b56.parquet",
                          repo_type="dataset")
    df = pd.read_parquet(src)
    out = DATA_DIR / "alpaca.parquet"
    df.to_parquet(out, index=False)
    print(f"alpaca: {len(df)} rows -> {out}")


def main():
    DATA_DIR.mkdir(exist_ok=True)
    wanted = sys.argv[1:] or ["jbb_harmful", *WALLEDAI, "strongreject", "xstest",
                              "gsm8k", "math500", "truthfulqa", "alpaca"]
    for source in wanted:
        if source == "jbb_harmful":
            download_jbb_harmful()
        elif source == "strongreject":
            download_strongreject()
        elif source == "xstest":
            download_xstest()
        elif source == "gsm8k":
            download_gsm8k()
        elif source == "math500":
            download_math500()
        elif source == "truthfulqa":
            download_truthfulqa()
        elif source == "alpaca":
            download_alpaca()
        elif source in WALLEDAI:
            download_walledeval(source)
        else:
            raise ValueError(f"unknown dataset: {source}")


if __name__ == "__main__":
    main()
