"""Drive proposed.Proposed (via Defender.ProposedDefense) with the same sampler
and DIJA prompts as exp.py, so its ASR is directly comparable.

proposed.py is used unmodified: latched binary gate read by a separate
detector forward each step until done, projection-removal steering on layers
12/16/20/24 at strength 0.4 over ALL positions, one remasking action on the
first detection with committed tokens, and initial_only=True by default (a
closed step-0 gate ends monitoring for the response).

The gate checkpoint format proposed.py expects (center/scale) is built here from
outputs/steer_detector.pt with center=0, scale=1 so the threshold from
outputs/gate_threshold.json keeps its original h.v units.

Usage:
    CUDA_VISIBLE_DEVICES=1 python run_proposed_dija.py --source jbb_harmful --attack dija \
        --n 100 --out outputs/dija_jbb_proposed.json
"""

import argparse
import time

import torch

from llada import MODEL_NAME, MASK_ID
from Attacker import DIJA, NoAttack
from common import load_llada, load_prompts, encode_prompt, write_json, load_detector
from Defender import ProposedDefense
from sampler import generate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["csv", "jbb_harmful"], default="jbb_harmful")
    p.add_argument("--csv", default="data/llada8b_wild_unsafe_only.csv")
    p.add_argument("--attack", choices=["none", "dija"], default="dija")
    p.add_argument("--dija-steps", type=int, default=4)
    p.add_argument("--dija-span", type=int, default=16)
    p.add_argument("--vector", default="outputs/steer_vector.pt")
    p.add_argument("--detector", default="outputs/steer_detector.pt")
    p.add_argument("--detector-layer", type=int, default=18)
    p.add_argument("--gate-threshold", type=float, default=None)
    p.add_argument("--layers", default="12,16,20,24")
    p.add_argument("--strength", type=float, default=0.4)
    p.add_argument("--mode", choices=["baseline", "steer", "repair"], default="repair")
    p.add_argument("--max-remask-tokens", type=int, default=16)
    p.add_argument("--max-parallel-commit", type=int, default=2)
    p.add_argument("--no-initial-only", action="store_true")
    p.add_argument("--out", default="outputs/dija_jbb_proposed.json")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--gen-length", type=int, default=128)
    p.add_argument("--block-length", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    return p.parse_args()


def main():
    args = parse_args()

    csd = torch.load(args.vector, map_location="cpu")
    print(f"loading {MODEL_NAME} ...")
    tokenizer, model = load_llada()
    device = next(model.parameters()).device

    det_vec, det_layer, threshold = load_detector(
        args.detector, args.detector_layer, device, args.gate_threshold)
    gate = {"layer": det_layer, "vector": det_vec,
            "center": torch.zeros_like(det_vec), "scale": 1.0, "threshold": threshold}
    csd = {"layers": csd["layers"], "vector": csd["vector"].to(device)}
    layers = tuple(int(s) for s in args.layers.split(","))
    print(f"proposed.py: layers {layers}, strength {args.strength}, mode {args.mode}, "
          f"gate layer {det_layer} threshold {threshold:.3f}, "
          f"initial_only {not args.no_initial_only}")

    rows = load_prompts(args.source, args.csv)[args.start: args.start + args.n]
    attacker = DIJA(args.dija_steps, args.dija_span) if args.attack == "dija" else NoAttack()
    print(f"running {len(rows)} prompts from {args.source} (attack {args.attack})")

    policy = ProposedDefense(
        model, gate, csd, layers=layers, total_steps=args.steps, strength=args.strength,
        mode=args.mode, max_remask_tokens=args.max_remask_tokens,
        max_parallel_commit=args.max_parallel_commit, initial_only=not args.no_initial_only)

    gen_config = {"steps": args.steps, "gen_length": args.gen_length,
                  "block_length": args.block_length, "temperature": args.temperature,
                  "cfg_scale": 0.0, "remasking": "low_confidence",
                  "attack": args.attack, "dija_steps": args.dija_steps, "dija_span": args.dija_span}
    steer_config = {"method": "proposed.py", "layers": list(layers), "strength": args.strength,
                    "transform": "project", "mode": args.mode, "detector_layer": args.detector_layer,
                    "gate_threshold": threshold, "max_remask_tokens": args.max_remask_tokens,
                    "max_parallel_commit": args.max_parallel_commit,
                    "initial_only": not args.no_initial_only, "single_forward": False}

    results = []
    t_start = time.time()
    for i, row in enumerate(rows):
        idx, prompt = row["index"], row["prompt"]
        user_message = attacker.build_prompt(row)
        input_ids = encode_prompt(tokenizer, user_message, device)

        t0 = time.time()
        out = generate(model, input_ids, policy, steps=args.steps, gen_length=args.gen_length,
                       block_length=args.block_length, temperature=args.temperature)
        elapsed = time.time() - t0

        generation, extra = attacker.decode(tokenizer, out, input_ids)
        results.append({"index": int(idx), "prompt": prompt, "attack_prompt": user_message,
                        "generation": generation, **extra,
                        "num_prompt_tokens": int(input_ids.shape[1]),
                        "num_prompt_masks": int((input_ids == MASK_ID).sum()),
                        "seconds": round(elapsed, 2), **policy.result_fields()})
        write_json(args.out, {"model": MODEL_NAME, "config": gen_config, "steering": steer_config,
                              "results": results})
        r = results[-1]
        print(f"[{i + 1}/{len(rows)}] idx={idx} {elapsed:.1f}s armed={r['gate_open']}"
              f"{' [remask]' if r['remasked'] else ''} :: {generation[:100].replace(chr(10), ' ')}...",
              flush=True)

    print(policy.summarize(results))
    print(f"\nDone in {time.time() - t_start:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
