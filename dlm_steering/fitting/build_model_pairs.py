"""Generate target-model response contrasts from the prompt column alone.

The output is an auditable JSONL file. Only rows whose two target-model
generations receive opposite Llama Guard verdicts can train a response vector.
"""

import argparse
import json
from pathlib import Path
import re

import pandas as pd
from dlm_steering.runtime.constants import MODEL_KEY
from dlm_steering.runtime.models import encode_prompt, load_llada, seed_all
from models import add_model_arg
from sampler import generate


def response(model, tokenizer, prompt, device, seed):
    seed_all(seed)
    x_in = encode_prompt(tokenizer, prompt, device)
    x = generate(model, x_in, None, steps=128, gen_length=128,
                 block_length=32, temperature=0.2, decoder="block")
    suffix = x[0, x_in.shape[1]:]
    text = tokenizer.decode(suffix, skip_special_tokens=True)
    return re.sub(r"Step \d+:\s*", "", text).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_model_arg(ap)
    ap.add_argument("--csv", default="data/llada8b_wild_unsafe_only.csv")
    ap.add_argument("--out", default="data/llada1.5_model_pairs.jsonl")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=0, help="0 means all remaining rows")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--guard-device", default="cuda:1")
    args = ap.parse_args()
    if MODEL_KEY != "llada1.5":
        raise ValueError("This builder is intended for LLaDA 1.5")

    prompts = pd.read_csv(args.csv)["prompt"].astype(str).tolist()
    end = min(len(prompts), args.start + args.n) if args.n else len(prompts)
    if not 0 <= args.start < end:
        raise ValueError("Empty or invalid prompt range")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with out.open() as stream:
            done = {int(json.loads(line)["index"]) for line in stream if line.strip()}

    from dlm_steering.evaluation.llamaguard import LlamaGuard4
    tokenizer, model = load_llada(args.device)
    with LlamaGuard4(device=args.guard_device) as guard, out.open("a") as stream:
        for index in range(args.start, end):
            if index in done:
                continue
            prompt = prompts[index]
            safe_request = (
                "Read the request below. Give a clear refusal to provide harmful "
                "instructions, then offer a brief safe alternative.\n\nRequest: " + prompt)
            direct_request = prompt
            safe_text = response(model, tokenizer, safe_request, args.device,
                                 100000 + index)
            direct_text = response(model, tokenizer, direct_request, args.device,
                                   200000 + index)
            safe_label = guard._classify(prompt, safe_text)["label"]
            direct_label = guard._classify(prompt, direct_text)["label"]
            record = {"index": index, "prompt": prompt,
                      "safe_prompt": safe_request, "direct_prompt": direct_request,
                      "safe_text": safe_text, "direct_text": direct_text,
                      "safe_label": safe_label, "direct_label": direct_label,
                      "model": "GSAI-ML/LLaDA-1.5", "seed_safe": 100000 + index,
                      "seed_direct": 200000 + index}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"[{index + 1}/{end}] safe={safe_label} direct={direct_label}",
                  flush=True)


if __name__ == "__main__":
    main()
