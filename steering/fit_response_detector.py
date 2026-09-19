"""Fit the response-side detector consumed by ``--remask v3``.

The prompt-side detector (``fit_detector.py``) reads fully masked answer
slots at the first denoising step; this detector instead reads *committed*
response tokens at a block boundary and asks whether the visible response is
unsafe. The pipeline mirrors what the reference ``response_detector.pt``
metadata describes:

    feature  = hidden states at --layer, mean-pooled over the committed
               template slots the model just filled
    label    = Llama-Guard-4 verdict on the generated text
               (unsafe -> 1, safe -> 0, parse errors dropped)
    model    = logistic regression, L2 with C (--C)
    prompts  = WildJailbreak adversarial prompts
               (data/llada8b_wild_unsafe_only.csv -- the unsafe-verdict half
               of the original generation set; we only need its prompts)

Samples are produced by the same sampler used at inference: the plain prompt
with no attack wrapper, so the committed slots the features pool over are
the generated response suffix -- the same distribution the boundary audit
reads. Both classes are judged, so the negative arm carries real GT labels
rather than assumed-safe regenerations.

The split is by prompt group: each row is one group and train/validation
group lists are stored in the checkpoint like the reference file.

Usage:
    CUDA_VISIBLE_DEVICES=1,2 python -m steering.fit_response_detector \
        --groups 384 --guard-device cuda:2
"""

from common import OUT_DIR, DETECTOR_LAYER, MODEL_KEY, add_model_arg

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import MODEL_NAME, MASK_ID, auroc, encode_prompt, load_llada

ROOT = Path(__file__).resolve().parent.parent

@torch.no_grad()
def sample_state(model, x, slot_positions, layer):
    """Mean-pooled hidden state over the committed template slots."""
    hs = model(x, output_hidden_states=True).hidden_states
    return hs[layer][0, slot_positions].mean(dim=0).to(torch.float32).cpu()


@torch.no_grad()
def generate_state(model, tokenizer, attacker, prompt, layer, device, *,
                   steps, gen_length, block_length, temperature):
    """Generate on one prompt; return (feature, filled-response text)."""
    from sampler import generate

    user_message = attacker.build_prompt({"prompt": prompt, "target": None})
    x_in = encode_prompt(tokenizer, user_message, device)
    slots = (x_in == MASK_ID)[0].nonzero().flatten()
    x = generate(model, x_in, None, steps=steps, gen_length=gen_length,
                 block_length=block_length, temperature=temperature,
                 decoder="dream" if MODEL_KEY == "dream" else "block")
    if slots.numel() == 0:  # no prompt masks: the generated suffix is the response
        slots = torch.arange(x_in.shape[1], x.shape[1], device=x.device)
    filled = tokenizer.decode(x[0, slots], skip_special_tokens=True)
    # The guard misfires on the "Step k:" scaffolding; judge the bare text.
    filled = re.sub(r"Step \d+:\s*", "", filled).strip()
    return sample_state(model, x, slots, layer), filled


def fit_logistic(X, y, C, steps=400):
    """L2-regularised logistic regression via LBFGS; returns (w, b)."""
    X = X.to(torch.float64)
    y = y.to(torch.float64)
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")
    lam = 1.0 / (C * len(X))

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            X @ w + b, y) + 0.5 * lam * (w @ w)
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().to(torch.float32), float(b.detach())




def main():
    ap = argparse.ArgumentParser()
    add_model_arg(ap)
    ap.add_argument("--csv", default=str(ROOT / "data/llada8b_wild_unsafe_only.csv"),
                    help="WildJailbreak prompt source (only the prompts are used)")
    ap.add_argument("--out", default=str(ROOT / OUT_DIR / "response_detector.pt"))
    ap.add_argument("--report", default=str(ROOT / OUT_DIR / "response_detector_report.json"))
    ap.add_argument("--groups", type=int, default=384)
    ap.add_argument("--layer", type=int, default=DETECTOR_LAYER)
    ap.add_argument("--C", type=float, default=0.01)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="probability cutoff V3 triggers on")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--gen-length", type=int, default=128)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--guard-device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.layer is None:
        bundle = torch.load(ROOT / OUT_DIR / "steer_detector.pt", map_location="cpu")
        args.layer = int(bundle["best_layer"])

    df = pd.read_csv(args.csv)
    rng = np.random.default_rng(args.seed)
    rows = df.iloc[rng.permutation(len(df))[: args.groups]]
    print(f"{len(rows)} prompt groups from {args.csv}")

    from Attacker import NoAttack
    attacker = NoAttack()
    gen_length = args.gen_length

    print(f"loading {MODEL_NAME} ...")
    tokenizer, model = load_llada(args.device)

    feats, texts, groups = [], [], []
    for k, (gi, r) in enumerate(rows.iterrows()):
        feat, filled = generate_state(
            model, tokenizer, attacker, r["prompt"], args.layer, args.device,
            steps=args.steps, gen_length=gen_length,
            block_length=args.block_length, temperature=args.temperature)
        feats.append(feat)
        texts.append(filled)
        groups.append(int(gi))
        if (k + 1) % 25 == 0:
            print(f"  generated [{k + 1}/{len(rows)}]", flush=True)

    from Evaluator import LlamaGuard4
    print(f"judging with {LlamaGuard4.__doc__.strip().splitlines()[0]} ...")
    labels = []
    with LlamaGuard4(device=args.guard_device) as guard:
        for k, (prompt, filled) in enumerate(zip(rows["prompt"], texts)):
            v = guard._classify(str(prompt), filled)
            labels.append(v["label"])
            if (k + 1) % 50 == 0:
                print(f"  judged [{k + 1}/{len(rows)}]", flush=True)

    # Verdicts occasionally come back like "safe/mkspecs"; prefix-match them.
    def _label(l):
        if l.startswith("unsafe"):
            return 1.0
        if l.startswith("safe"):
            return 0.0
        return None

    tagged = [_label(l) for l in labels]
    keep = [i for i, t in enumerate(tagged) if t is not None]
    n_drop = len(labels) - len(keep)
    y = torch.tensor([tagged[i] for i in keep])
    X = torch.stack([feats[i] for i in keep])
    groups = np.array([groups[i] for i in keep])
    print(f"labels: {int(y.sum())} unsafe / {int((~y.bool()).sum())} safe"
          + (f" ({n_drop} unparsed dropped)" if n_drop else ""))

    w, b = fit_logistic(X, y, args.C)
    logits = (X @ w + b).numpy()
    pos, neg = logits[y.numpy() == 1], logits[y.numpy() == 0]
    auc = auroc(pos, neg) if len(pos) and len(neg) else float("nan")
    threshold = float(args.threshold)
    t_logit = np.log(threshold / (1 - threshold))
    bal_acc = float(((pos >= t_logit).mean()
                     + (neg < t_logit).mean()) / 2)
    probs = 1 / (1 + np.exp(-logits))
    pos_p = probs[y.numpy() == 1]
    neg_p = probs[y.numpy() == 0]
    print(f"\nlayer {args.layer} C={args.C}: train AUROC={auc:.4f} "
          f"threshold={threshold:.4f} balanced-acc={bal_acc:.4f}")
    print(f"  unsafe P(trigger)={float((pos_p >= threshold).mean()):.3f}  "
          f"safe P(trigger)={float((neg_p >= threshold).mean()):.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"weight": w, "bias": b, "threshold": threshold,
                "layer": args.layer, "model": MODEL_NAME,
                "pool": "mean_committed_response_tokens",
                "target": "visible_response_unsafe",
                "source": "wildjailbreak_response_states",
                "train_groups": [int(g) for g in groups],
                "C": float(args.C)}, args.out)
    Path(args.report).write_text(json.dumps(
        {"groups": int(len(X)), "n_train": int(len(X)),
         "layer": args.layer, "C": args.C,
         "auroc": auc, "threshold": threshold, "balanced_accuracy": bal_acc,
         "steps": args.steps,
         "gen_length": gen_length, "seed": args.seed}, indent=2))
    print(f"-> {args.out}\n-> {args.report}")


if __name__ == "__main__":
    main()
