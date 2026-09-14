"""Single-forward adaptive steering with one-shot remasking for LLaDA.

Each denoising step runs ONE model forward. A detector hook on blocks[gate-1]
reads the current answer and sets a continuous gate strength
g = clamp((projection - threshold) / width, 0, 1); steering hooks on later
blocks (default blocks[24], i.e. --layer 25) read g in the same forward and push
the currently masked answer slots toward refusal. Every steering layer must sit
after the gate layer so the gate never sees the current step's injection.

Once committed answer tokens exist and g >= --remask-trigger, the policy makes a
single remasking attempt: candidate windows of committed tokens are probed with
detector-only forwards (no steering), the window whose removal lowers the
projection the most is reopened as [MASK], and the transfer schedule of the
current block is rebalanced. Steering keeps running after the repair, but it is
adaptive: g is re-read every step, so it decays as the committed answer turns
into a refusal.

Steering transform (u = unit refusal direction, ref = layer mean |h|, s = schedule):
    additive  h += g*s*strength*ref*u           (same as llada_steering_v2 alpha)
    project   h -= g*s*(max(h.(-u),0) + strength*ref)*(-u)

Modes:
    --steer none --remask none   no policy at all; the undefended reference
    --steer fixed     gate read once on the fully masked answer, g in {0,1}
    --steer adaptive  adaptive g every step (default)
    --remask fixed    one remasking attempt (default); none disables it

The sampler (sampler.generate), the policy itself (Defender.Ours), and the
attack templates (Attacker.DIJA) now live in their own modules; this file keeps
the original CLI as a thin wrapper. New experiments should use exp.py.

Usage:
    CUDA_VISIBLE_DEVICES=1 python llada_steering_remasking_v2.py --alpha 1.0 \
        --detector outputs/steer_detector.pt --detector-layer 18 --layer 25 \
        --gen-length 128 --steps 128 --n 20 \
        --out outputs/remask_v2_len128.json
"""

import argparse
import json
import time
from pathlib import Path

import torch

from Attacker import DIJA, NoAttack
from common import MODEL_NAME, MASK_ID, load_llada, load_prompts, encode_prompt, write_json
from Defender import V2
from sampler import generate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["jbb_harmful", "advbench", "harmbench"],
                   default="jbb_harmful")
    p.add_argument("--attack", choices=["none", "dija"], default="none")
    p.add_argument("--dija-steps", type=int, default=4)
    p.add_argument("--dija-span", type=int, default=16)
    p.add_argument("--vector", default="outputs/steer_vector.pt")
    p.add_argument("--detector", default="outputs/steer_detector.pt")
    p.add_argument("--out", default="outputs/remask_v2_len128.json")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--layer", default="25",
                   help="Comma-separated steering layers (hidden-state numbering; 25 = blocks[24]). "
                        "All must be after --detector-layer.")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Steering strength in units of the layer's mean activation norm.")
    p.add_argument("--transform", choices=["additive", "project"], default="additive")
    p.add_argument("--schedule", default="const", choices=["const", "linear", "cosine"])
    p.add_argument("--detector-layer", type=int, default=18)
    p.add_argument("--gate-threshold", type=float, default=None,
                   help="Projection threshold; defaults to outputs/gate_threshold.json.")
    p.add_argument("--gate-width", type=float, default=1.0,
                   help="Projection margin above threshold for full steering.")
    p.add_argument("--steer", choices=["none", "fixed", "adaptive"], default="adaptive",
                   help="none: no steering; fixed: step-0 binary gate; "
                        "adaptive: continuous gate every step.")
    p.add_argument("--remask", choices=["none", "fixed"], default="fixed",
                   help="none: never remask; fixed: one-shot committed-token repair.")
    p.add_argument("--max-remask-tokens", type=int, default=16)
    p.add_argument("--max-parallel-commit", type=int, default=2)
    p.add_argument("--remask-trigger", type=float, default=1.0,
                   help="Gate strength needed to attempt the one-shot repair.")
    p.add_argument("--initial-only", action="store_true",
                   help="Stop monitoring for the whole response if the step-0 gate is closed.")
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--gen-length", type=int, default=128)
    p.add_argument("--block-length", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--remasking", default="low_confidence")
    return p.parse_args()


def main():
    args = parse_args()

    steer_layers = [int(s) for s in args.layer.split(",")]
    site_specs = []
    if args.steer != "none":
        bundle = torch.load(args.vector, map_location="cpu")
        for layer in steer_layers:
            li = bundle["layers"].index(layer)
            site_specs.append((layer, bundle["vector"][li], bundle["mean_act_norm"][li]))
            print(f"steer: layer {layer} mean|h| {bundle['mean_act_norm'][li]:.1f} "
                  f"alpha {args.alpha} -> {args.alpha * bundle['mean_act_norm'][li]:.2f}")

    db = torch.load(args.detector, map_location="cpu")
    det_vec = db["vector"][db["layers"].index(args.detector_layer)]
    threshold = args.gate_threshold
    if threshold is None:
        threshold = json.loads((Path(args.detector).parent / "gate_threshold.json").read_text())["threshold"]
    print(f"gate: layer {args.detector_layer}, threshold {threshold:.3f}, width {args.gate_width}, "
          f"steer {args.steer}, remask {args.remask}, transform {args.transform}")

    rows = load_prompts(args.source)[args.start: args.start + args.n]
    attacker = DIJA(args.dija_steps, args.dija_span) if args.attack == "dija" else NoAttack()
    print(f"running {len(rows)} prompts from {args.source} (attack {args.attack})")

    print(f"loading {MODEL_NAME} ...")
    tokenizer, model = load_llada()
    device = next(model.parameters()).device

    policy = None if args.steer == "none" and args.remask == "none" else V2(
        model, gate_layer=args.detector_layer, gate_vector=det_vec.to(device),
        threshold=threshold, width=args.gate_width,
        sites=[(layer, v.to(device), ref) for layer, v, ref in site_specs],
        strength=args.alpha, transform=args.transform,
        steer=args.steer, remask=args.remask == "fixed",
        max_remask_tokens=args.max_remask_tokens, max_parallel_commit=args.max_parallel_commit,
        remask_trigger=args.remask_trigger, initial_only=args.initial_only,
    )

    gen_config = {"steps": args.steps, "gen_length": args.gen_length,
                  "block_length": args.block_length, "temperature": args.temperature,
                  "cfg_scale": 0.0, "remasking": args.remasking,
                  "attack": args.attack, "dija_steps": args.dija_steps, "dija_span": args.dija_span}
    steer_config = None if policy is None else {
        "layers": steer_layers, "alpha_rel": args.alpha, "transform": args.transform,
        "schedule": args.schedule, "vector": args.vector, "detector": args.detector,
        "detector_layer": args.detector_layer, "gate_threshold": threshold,
        "gate_width": args.gate_width, "steer": args.steer, "remask": args.remask,
        "max_remask_tokens": args.max_remask_tokens,
        "max_parallel_commit": args.max_parallel_commit,
        "remask_trigger": args.remask_trigger, "initial_only": args.initial_only,
        "single_forward": True,
    }

    results = []
    t_start = time.time()
    for i, row in enumerate(rows):
        idx, prompt = row["index"], row["prompt"]
        user_message = attacker.build_prompt(row)
        input_ids = encode_prompt(tokenizer, user_message, device)

        t0 = time.time()
        out = generate(
            model, input_ids, policy, steps=args.steps, gen_length=args.gen_length,
            block_length=args.block_length, temperature=args.temperature,
            remasking=args.remasking, schedule=args.schedule)
        elapsed = time.time() - t0

        generation, extra = attacker.decode(tokenizer, out, input_ids)
        results.append({"index": int(idx), "prompt": prompt, "attack_prompt": user_message,
                        "generation": generation, **extra,
                        "num_prompt_tokens": int(input_ids.shape[1]),
                        "num_prompt_masks": int((input_ids == MASK_ID).sum()),
                        "seconds": round(elapsed, 2),
                        **(policy.result_fields() if policy is not None else {})})

        write_json(args.out, {"model": MODEL_NAME, "config": gen_config,
                              "steering": steer_config, "results": results})

        r = results[-1]
        tag = "" if policy is None else f" gmax={r['gate_max_strength']:.2f}" + (" [remask]" if r["remasked"] else "")
        print(f"[{i + 1}/{len(rows)}] idx={idx} {elapsed:.1f}s{tag} :: "
              f"{generation[:100].replace(chr(10), ' ')}...", flush=True)

    if policy is not None:
        n_open = sum(r["gate_open"] for r in results)
        n_remask = sum(r["remasked"] for r in results)
        print(f"gate opened on {n_open}/{len(results)} prompts, remasked {n_remask}")
    print(f"\nDone in {time.time() - t_start:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
