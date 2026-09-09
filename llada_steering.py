"""Activation-steered LLaDA sampling, for suppressing jailbreak compliance.

The direction comes from steering/fit_vector.py: mean(h | writing a refusal) -
mean(h | writing a compliant answer), fitted on WildJailbreak pairs that exclude
the held-out eval prompts. Adding +alpha*v pushes the residual stream toward the
refusal continuation.

Three things are specific to a diffusion LM and are not choices an autoregressive
steering setup has to make:

  * position -- the forward pass sees the whole sequence every step, so the hook
    only touches currently-masked answer slots (what the sampler predicts, and
    what the vector was fitted on). The prompt is left alone.
  * schedule -- early denoising steps fix the high-confidence tokens that decide
    refusal vs compliance, so alpha can be decayed once the answer is committed.
  * blocks -- with semi-autoregressive block decoding, block 0 carries most of
    that decision; --first-block-only restricts steering to it.

Steering also shifts token confidences, which changes the *unmasking order* under
low_confidence remasking, not just the token values.

Usage:
    CUDA_VISIBLE_DEVICES=1 python llada_steering.py --alpha 1.0 \
        --gen-length 128 --steps 128 --n 20 --out outputs/steered_len128.json
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from llada import MODEL_NAME, MASK_ID, add_gumbel_noise, get_num_transfer_tokens


class Steerer:
    """Adds alpha * scale * v to selected positions at the output of one block.

    hidden_states[L] in the HF wrapper is the *input* to block L, so a vector
    fitted at layer L is injected by hooking blocks[L-1].
    """

    def __init__(self, model, vector, layer, alpha):
        self.v = vector
        self.alpha = alpha
        self.scale = 1.0
        self.pos = None  # bool mask over sequence positions, or None to disable
        # The hook stays registered on the module for the life of the run, so a
        # gate decision cannot be expressed by simply not passing the steerer to
        # generate_steered -- the hook would still fire, with whatever position
        # mask the previous prompt left behind. This flag is the real off switch.
        self.enabled = True
        block = model.model.transformer.blocks[layer - 1]
        self.handle = block.register_forward_hook(self._hook)

    def _hook(self, module, args, out):
        x, cache = out
        if not self.enabled or self.pos is None or self.scale == 0.0 or self.alpha == 0.0:
            return out
        x = x.clone()
        # Broadcasts over the batch, so a CFG-doubled batch is steered on both halves.
        x[:, self.pos] += (self.alpha * self.scale) * self.v.to(x.dtype)
        return x, cache

    def close(self):
        self.handle.remove()


@torch.no_grad()
def gate_projection(model, prompt, gen_length, layer, vec, steerer=None):
    """Read the prompt-side harmfulness detector before any token is committed.

    The answer region is fully masked here, exactly the state at the first
    denoising step, so the gate decides once per prompt rather than per step.
    A response-conditioned direction carries no signal in this state -- with
    everything masked its two arms are byte-identical -- which is why the
    detector is fitted prompt-side.
    """
    # The hook must be inert here: the detector has to read the unmodified
    # residual stream, and a position mask left over from the previous prompt
    # has the wrong length for this one.
    saved = None
    if steerer is not None:
        saved, steerer.enabled = steerer.enabled, False
    try:
        n_p = prompt.shape[1]
        x = torch.full((1, n_p + gen_length), MASK_ID, dtype=torch.long,
                       device=model.device)
        x[:, :n_p] = prompt
        hs = model(x, output_hidden_states=True).hidden_states
        h = hs[layer][0, n_p:, :].mean(dim=0).to(torch.float32)
        return float(h @ vec.to(h.dtype).to(h.device))
    finally:
        if steerer is not None:
            steerer.enabled = saved


def step_scale(schedule, i, n_steps):
    """alpha multiplier at denoising step i of n_steps."""
    if schedule == "const":
        return 1.0
    frac = i / max(1, n_steps - 1)
    if schedule == "linear":
        return 1.0 - frac
    if schedule == "cosine":
        import math
        return 0.5 * (1.0 + math.cos(math.pi * frac))
    raise ValueError(schedule)


@torch.no_grad()
def generate_steered(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=32,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    steerer=None,
    schedule="const",
    first_block_only=False,
):
    """llada.generate with a steering hook driven per block and per step."""
    x = torch.full(
        (1, prompt.shape[1] + gen_length), MASK_ID, dtype=torch.long, device=model.device
    )
    x[:, : prompt.shape[1]] = prompt.clone()
    prompt_index = x != MASK_ID

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = x[:, block_start:block_end] == MASK_ID
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = x == MASK_ID

            if steerer is not None:
                if first_block_only and num_block > 0:
                    steerer.pos = None
                else:
                    # Masked answer slots only -- never the prompt.
                    pos = mask_index[0].clone()
                    pos[: prompt.shape[1]] = False
                    steerer.pos = pos if pos.any() else None
                    steerer.scale = step_scale(schedule, i, steps_per_block)

            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = MASK_ID
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
            elif remasking == "random":
                x0_p = torch.rand_like(x0, dtype=torch.float64)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, block_end:] = -float("inf")
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(
                mask_index, x0_p, torch.tensor(-float("inf"), device=x0.device)
            )

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/llada8b_wild_unsafe_only.csv")
    p.add_argument("--vector", default="outputs/steer_vector.pt")
    p.add_argument("--out", default="outputs/steered_len128.json")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--layer", type=int, default=0, help="0 = vector's best layer.")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="In units of the layer's mean activation norm.")
    p.add_argument("--schedule", default="const", choices=["const", "linear", "cosine"])
    p.add_argument("--first-block-only", action="store_true")
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--gen-length", type=int, default=128)
    p.add_argument("--block-length", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--cfg-scale", type=float, default=0.0)
    p.add_argument("--remasking", default="low_confidence")
    p.add_argument("--detector", default=None,
                   help="Detector bundle; enables gating when given.")
    p.add_argument("--gate-threshold", type=float, default=None,
                   help="Projection threshold; defaults to outputs/gate_threshold.json.")
    p.add_argument("--detector-layer", type=int, default=0)
    return p.parse_args()


def main():
    from transformers import AutoModel, AutoTokenizer

    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bundle = torch.load(args.vector, map_location="cpu")
    layers = bundle["layers"]
    layer = args.layer or bundle["best_layer"]
    li = layers.index(layer)
    v = bundle["vector"][li]
    act_norm = bundle["mean_act_norm"][li]
    # alpha is expressed in units of the layer's typical residual norm, so the
    # same number means the same relative push at any layer.
    alpha = args.alpha * act_norm
    print(f"vector: layer {layer} (auroc {bundle['best_auroc']:.4f} at best layer "
          f"{bundle['best_layer']}), mean|h| {act_norm:.1f}, "
          f"alpha {args.alpha} -> {alpha:.2f}")

    df = pd.read_csv(args.csv)
    rows = df.iloc[args.start : args.start + args.n]
    print(f"running {len(rows)} prompts (index {args.start}..{args.start + len(rows) - 1})")

    print(f"loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_NAME, trust_remote_code=True, dtype=torch.bfloat16
    ).to(device).eval()

    steerer = Steerer(model, v.to(device), layer, alpha) if args.alpha != 0 else None

    det_vec = det_layer = threshold = None
    if args.detector:
        db = torch.load(args.detector, map_location="cpu")
        det_layer = args.detector_layer or db["best_layer"]
        det_vec = db["vector"][db["layers"].index(det_layer)].to(device)
        threshold = args.gate_threshold
        if threshold is None:
            threshold = json.loads(
                (Path(args.detector).parent / "gate_threshold.json").read_text()
            )["threshold"]
        print(f"gate: detector layer {det_layer}, threshold {threshold:.3f}")

    gen_config = {
        "steps": args.steps,
        "gen_length": args.gen_length,
        "block_length": args.block_length,
        "temperature": args.temperature,
        "cfg_scale": args.cfg_scale,
        "remasking": args.remasking,
    }
    steer_config = {
        "layer": layer, "alpha_rel": args.alpha, "alpha_abs": round(alpha, 3),
        "schedule": args.schedule, "first_block_only": args.first_block_only,
        "vector": args.vector, "auroc": bundle["best_auroc"],
        "detector": args.detector, "detector_layer": det_layer,
        "gate_threshold": threshold,
    }

    results = []
    t_start = time.time()
    for i, (idx, row) in enumerate(rows.iterrows()):
        prompt = str(row["prompt"])
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=False,
        )
        input_ids = torch.tensor(tokenizer(formatted)["input_ids"], device=device).unsqueeze(0)

        t0 = time.time()
        proj, gate_open = None, True
        if det_vec is not None:
            proj = gate_projection(model, input_ids, args.gen_length, det_layer,
                                   det_vec, steerer)
            gate_open = proj >= threshold
        if steerer is not None:
            steerer.enabled = gate_open
            steerer.pos = None
        active = steerer if gate_open else None
        out = generate_steered(
            model, input_ids, steerer=active, schedule=args.schedule,
            first_block_only=args.first_block_only, **gen_config
        )
        elapsed = time.time() - t0

        generation = tokenizer.batch_decode(
            out[:, input_ids.shape[1]:], skip_special_tokens=True
        )[0]
        results.append({
            "index": int(idx), "prompt": prompt, "generation": generation,
            "num_prompt_tokens": int(input_ids.shape[1]), "seconds": round(elapsed, 2),
            **({"gate_projection": round(proj, 4), "gate_open": bool(gate_open)}
               if proj is not None else {}),
        })

        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"model": MODEL_NAME, "config": gen_config,
                       "steering": steer_config, "results": results},
                      f, ensure_ascii=False, indent=2)

        print(f"[{i + 1}/{len(rows)}] idx={idx} {elapsed:.1f}s :: "
              f"{generation[:100].replace(chr(10), ' ')}...", flush=True)

    if steerer is not None:
        steerer.close()
    if det_vec is not None:
        n_open = sum(r["gate_open"] for r in results)
        print(f"gate opened on {n_open}/{len(results)} prompts")
    print(f"\nDone in {time.time() - t_start:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
