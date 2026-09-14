"""Download the exp.py prompt datasets into ./data.

    jbb_harmful  -> data/jbb_harmful.csv    (JailbreakBench/JBB-Behaviors)
    advbench     -> data/advbench.parquet   (walledai/AdvBench)
    harmbench    -> data/harmbench.parquet  (walledai/HarmBench)

The walledeval repos ship several parquet shards/splits; they are
concatenated and de-duplicated like common.load_eval_prompts does.
They are gated: accept access on the HF dataset page and authenticate
(huggingface-cli login or HF_TOKEN) before running.

Usage:
    python data_downloader.py            # all three
    python data_downloader.py advbench   # one only
"""

import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download

DATA_DIR = Path(__file__).parent / "data"

WALLEDAI = {"advbench": "walledai/AdvBench", "harmbench": "walledai/HarmBench"}


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


def main():
    DATA_DIR.mkdir(exist_ok=True)
    wanted = sys.argv[1:] or ["jbb_harmful", *WALLEDAI]
    for source in wanted:
        if source == "jbb_harmful":
            download_jbb_harmful()
        elif source in WALLEDAI:
            download_walledeval(source)
        else:
            raise ValueError(f"unknown dataset: {source}")


if __name__ == "__main__":
    main()
