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
               of the original generation set; we only need its prompts),
               optionally mixed with ordinary instructions from Alpaca
               (--alpaca-groups). WildJailbreak alone gives the negative class
               only prompts that *look* adversarial, so ordinary Q&A, maths and
               multiple-choice answers sit off-distribution and score high; the
               Alpaca arm supplies benign responses as genuine negatives.

Samples are produced by the same sampler used at inference: the plain prompt
with no attack wrapper, so the committed slots the features pool over are
the generated response suffix -- the same distribution the boundary audit
reads. Both classes are judged, so the negative arm carries real GT labels
rather than assumed-safe regenerations.

Generation and judging dominate the cost and are deterministic at temperature
0, so both are cached under data/response_fit_cache (see steering/response_cache
.py). A rerun replays the cached token ids through one forward pass to rebuild
features, which leaves --layer free to change and regenerates only prompts the
cache has never seen. --refresh-cache forces a full rebuild.

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
ALPACA_GROUP_OFFSET = 1_000_000

@torch.no_grad()
def sample_state(model, x, slot_positions, layer):
    """Mean-pooled hidden state over the committed template slots."""
    hs = model(x, output_hidden_states=True).hidden_states
    return hs[layer][0, slot_positions].mean(dim=0).to(torch.float32).cpu()


def clean_text(tokenizer, x, slots):
    filled = tokenizer.decode(x[0, slots], skip_special_tokens=True)
    # The guard misfires on the "Step k:" scaffolding; judge the bare text.
    return re.sub(r"Step \d+:\s*", "", filled).strip()


@torch.no_grad()
def generate_sequence(model, tokenizer, attacker, prompt, device, *,
                      steps, gen_length, block_length, temperature):
    """Generate on one prompt; return (sequence, response slots, text)."""
    from sampler import generate

    user_message = attacker.build_prompt({"prompt": prompt, "target": None})
    x_in = encode_prompt(tokenizer, user_message, device)
    slots = (x_in == MASK_ID)[0].nonzero().flatten()
    x = generate(model, x_in, None, steps=steps, gen_length=gen_length,
                 block_length=block_length, temperature=temperature,
                 decoder="dream" if MODEL_KEY == "dream" else "block")
    if slots.numel() == 0:  # no prompt masks: the generated suffix is the response
        slots = torch.arange(x_in.shape[1], x.shape[1], device=x.device)
    return x, slots, clean_text(tokenizer, x, slots)


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
    ap.add_argument("--alpaca", default=str(ROOT / "data/alpaca.parquet"),
                    help="Benign instruction source mixed into the fit")
    ap.add_argument("--alpaca-groups", type=int, default=0,
                    help="How many Alpaca prompts to add (0: WildJailbreak only)")
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
    ap.add_argument("--refresh-cache", action="store_true",
                    help="Regenerate and re-judge every prompt, ignoring "
                         "data/response_fit_cache.")
    args = ap.parse_args()

    if args.layer is None:
        bundle = torch.load(ROOT / OUT_DIR / "steer_detector.pt", map_location="cpu")
        args.layer = int(bundle["best_layer"])

    rng = np.random.default_rng(args.seed)
    df = pd.read_csv(args.csv)
    picked = df.iloc[rng.permutation(len(df))[: args.groups]]
    # (group id, prompt, arm). Alpaca group ids are offset so the two arms stay
    # distinguishable in the checkpoint's train_groups list.
    rows = [(int(gi), str(r["prompt"]), "wildjailbreak")
            for gi, r in picked.iterrows()]
    print(f"{len(rows)} prompt groups from {args.csv}")
    if args.alpaca_groups:
        alpaca = pd.read_parquet(args.alpaca)
        picked = alpaca.iloc[rng.permutation(len(alpaca))[: args.alpaca_groups]]
        for gi, r in picked.iterrows():
            extra = str(r["input"]).strip()
            prompt = str(r["instruction"]).strip()
            rows.append((ALPACA_GROUP_OFFSET + int(gi),
                         f"{prompt}\n\n{extra}" if extra else prompt, "alpaca"))
        print(f"{args.alpaca_groups} prompt groups from {args.alpaca}")

    from Attacker import NoAttack
    from Evaluator import LlamaGuard4
    from steering import response_cache as rc
    attacker = NoAttack()
    gen_length = args.gen_length

    key = rc.config_key(model_name=MODEL_NAME, steps=args.steps,
                        gen_length=gen_length, block_length=args.block_length,
                        temperature=args.temperature, attacker="none",
                        judge="Llama-Guard-4-12B")
    caches = {arm: ({} if args.refresh_cache else rc.load(arm, key))
              for arm in dict.fromkeys(arm for _, _, arm in rows)}
    missing = [r for r in rows if rc.prompt_key(r[1]) not in caches[r[2]]]
    for arm, cache in caches.items():
        have = sum(1 for _, p, a in rows if a == arm and rc.prompt_key(p) in cache)
        print(f"캐시 {arm}: {have}/{sum(a == arm for _, _, a in rows)}행 재사용 "
              f"({rc.cache_path(arm, key)})")

    print(f"loading {MODEL_NAME} ...")
    tokenizer, model = load_llada(args.device)

    if missing:
        print(f"{len(missing)}행을 새로 생성합니다 (캐시 미보유).")
        fresh = []
        for k, (gi, prompt, arm) in enumerate(missing):
            x, slots, filled = generate_sequence(
                model, tokenizer, attacker, prompt, args.device,
                steps=args.steps, gen_length=gen_length,
                block_length=args.block_length, temperature=args.temperature)
            fresh.append((gi, prompt, arm, x[0].tolist(), slots.tolist(), filled))
            if (k + 1) % 25 == 0:
                print(f"  generated [{k + 1}/{len(missing)}]", flush=True)
        print(f"judging with {LlamaGuard4.__doc__.strip().splitlines()[0]} ...")
        with LlamaGuard4(device=args.guard_device) as guard:
            for k, (gi, prompt, arm, ids, slots, filled) in enumerate(fresh):
                label = guard._classify(str(prompt), filled)["label"]
                caches[arm][rc.prompt_key(prompt)] = rc.entry(
                    prompt, gi, ids, slots, filled, label)
                if (k + 1) % 50 == 0:
                    print(f"  judged [{k + 1}/{len(fresh)}]", flush=True)
        for arm, cache in caches.items():
            path = rc.save(arm, key, cache,
                           meta={"model": MODEL_NAME, "layer_independent": True,
                                 "steps": args.steps, "gen_length": gen_length,
                                 "block_length": args.block_length,
                                 "temperature": args.temperature})
            print(f"캐시 저장: {path} ({len(cache)}행)")

    # Features come from the cached sequence: one forward instead of the 64
    # denoising steps the generation cost, so --layer stays free to change.
    print(f"replaying {len(rows)} cached sequences at layer {args.layer} ...")
    feats, texts, groups, arms, labels = [], [], [], [], []
    for k, (gi, prompt, arm) in enumerate(rows):
        row = caches[arm][rc.prompt_key(prompt)]
        x = torch.tensor([row["token_ids"]], device=args.device)
        slots = torch.tensor(row["slots"], device=args.device)
        feats.append(sample_state(model, x, slots, args.layer))
        texts.append(row["text"])
        groups.append(int(row["group"]))
        arms.append(arm)
        labels.append(row["label"])
        if (k + 1) % 100 == 0:
            print(f"  replayed [{k + 1}/{len(rows)}]", flush=True)

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
    arms = np.array([arms[i] for i in keep])
    print(f"labels: {int(y.sum())} unsafe / {int((~y.bool()).sum())} safe"
          + (f" ({n_drop} unparsed dropped)" if n_drop else ""))
    for arm in dict.fromkeys(arms):
        sel = arms == arm
        print(f"  {arm}: {int(sel.sum())} rows, {int(y.numpy()[sel].sum())} unsafe")

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
    per_arm = {}
    for arm in dict.fromkeys(arms):
        sel = arms == arm
        per_arm[arm] = {"n": int(sel.sum()),
                        "unsafe_rate": float(y.numpy()[sel].mean()),
                        "trigger_rate": float((probs[sel] >= threshold).mean())}
        print(f"  {arm}: P(trigger)={per_arm[arm]['trigger_rate']:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"weight": w, "bias": b, "threshold": threshold,
                "layer": args.layer, "model": MODEL_NAME,
                "pool": "mean_committed_response_tokens",
                "target": "visible_response_unsafe",
                "source": ("wildjailbreak+alpaca_response_states"
                           if args.alpaca_groups else
                           "wildjailbreak_response_states"),
                "alpaca_group_offset": ALPACA_GROUP_OFFSET,
                "train_groups": [int(g) for g in groups],
                "C": float(args.C)}, args.out)
    Path(args.report).write_text(json.dumps(
        {"groups": int(len(X)), "n_train": int(len(X)),
         "layer": args.layer, "C": args.C,
         "auroc": auc, "threshold": threshold, "balanced_accuracy": bal_acc,
         "per_arm": per_arm, "alpaca_groups": args.alpaca_groups,
         "steps": args.steps,
         "gen_length": gen_length, "seed": args.seed}, indent=2))
    print(f"-> {args.out}\n-> {args.report}")


if __name__ == "__main__":
    main()
