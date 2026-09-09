"""Run LLaDA-8B-Instruct over prompts from a CSV and dump generations to JSON.

Usage:
    python run_dataset.py [--n 20] [--csv llada8b_wild_unsafe_only.csv] [--out results.json]
"""

import argparse
import json
import time

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from llada import MODEL_NAME, DEVICE, generate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/llada8b_wild_unsafe_only.csv")
    p.add_argument("--out", default="outputs/llada8b_generations.json")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--gen-length", type=int, default=256)
    p.add_argument("--block-length", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--cfg-scale", type=float, default=0.0)
    p.add_argument("--remasking", default="low_confidence")
    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.csv)
    rows = df.iloc[args.start : args.start + args.n]
    print(f"Loaded {len(df)} rows from {args.csv}, running {len(rows)} "
          f"(index {args.start}..{args.start + len(rows) - 1})")

    print(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(
            MODEL_NAME, trust_remote_code=True, dtype=torch.bfloat16
        )
        .to(DEVICE)
        .eval()
    )

    gen_config = {
        "steps": args.steps,
        "gen_length": args.gen_length,
        "block_length": args.block_length,
        "temperature": args.temperature,
        "cfg_scale": args.cfg_scale,
        "remasking": args.remasking,
    }

    results = []
    t_start = time.time()
    for i, (idx, row) in enumerate(rows.iterrows()):
        prompt = str(row["prompt"])
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        input_ids = torch.tensor(
            tokenizer(formatted)["input_ids"], device=DEVICE
        ).unsqueeze(0)

        t0 = time.time()
        out = generate(model, input_ids, **gen_config)
        elapsed = time.time() - t0

        generation = tokenizer.batch_decode(
            out[:, input_ids.shape[1] :], skip_special_tokens=True
        )[0]

        results.append(
            {
                "index": int(idx),
                "prompt": prompt,
                "generation": generation,
                "num_prompt_tokens": int(input_ids.shape[1]),
                "seconds": round(elapsed, 2),
            }
        )

        # Write after every sample so a crash doesn't lose completed work.
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(
                {"model": MODEL_NAME, "config": gen_config, "results": results},
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"[{i + 1}/{len(rows)}] idx={idx} {elapsed:.1f}s :: "
              f"{generation[:100].replace(chr(10), ' ')}...", flush=True)

    print(f"\nDone in {time.time() - t_start:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
