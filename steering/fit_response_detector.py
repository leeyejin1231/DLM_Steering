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

Samples are produced by the same sampler used at inference. --attack dija
uses the synthetic DIJA template (Attacker.DIJATemplate: "Sure, here is ..."
plus "Step k:" mask spans inside the user turn) with --gen-length 0, so
features come from the same committed-slot distribution the boundary audit
reads under attack and the label set is not dominated by refusals; --attack
none samples plain answers (the over-refusal setting). Both classes are
judged, so the negative arm carries real GT labels rather than assumed-safe
regenerations.

The split is by prompt group (each CSV row is one group): the regression is
fitted on the train groups and the probability cutoff V3 triggers on is
chosen on the validation groups (--threshold auto: Youden's J, i.e. max
TPR - FPR), like the reference checkpoint whose cutoff is ~0.09 rather than
0.5. Train/validation group lists are stored in the checkpoint.

Generated features, texts and guard labels are cached next to --out
(--samples); --from-samples refits (C, cutoff, split) without regenerating.

Usage:
    CUDA_VISIBLE_DEVICES=1,2 python steering/fit_response_detector.py \
        --groups 384 --guard-device cuda:2
    CUDA_VISIBLE_DEVICES=0,1 python steering/fit_response_detector.py --model dream \
        --attack dija --guard-device cuda:1   # layer = outputs/dream detector best_layer
    python steering/fit_response_detector.py --model dream --attack dija \
        --from-samples outputs/dream/response_samples_dija.pt --C 0.1   # refit only
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common import (DETECTOR_LAYER, MODEL_KEY, MODEL_NAME, MASK_ID, N_LAYERS,  # noqa: E402
                    OUT_DIR, add_model_arg, auroc, encode_prompt, load_detector_bundle,
                    load_model, prompt_token_ids, seed_all)

ALL_LAYERS = list(range(1, N_LAYERS))   # hidden_states[L] == output of block L-1
# Alpaca group ids are offset so the two arms stay distinguishable in a
# checkpoint's group lists (same convention as the code-share branch).
ALPACA_GROUP_OFFSET = 1_000_000
ALPACA_PARQUET = ("/mnt/shared/huggingface-cache/hub/datasets--tatsu-lab--alpaca/snapshots/"
                  "dce01c9b08f87459cf36a430d809084718273017/data/"
                  "train-00000-of-00001-a09b74b3ef9c3b56.parquet")


@torch.no_grad()
def sample_state(model, x, slot_positions, layers=ALL_LAYERS):
    """Mean-pooled hidden state over the committed template slots, per layer
    -> [len(layers), hidden]. Every layer is kept so the detector layer can be
    chosen (or changed) from the cache without regenerating."""
    hs = model(x, output_hidden_states=True).hidden_states
    return torch.stack([hs[L][0, slot_positions].mean(dim=0) for L in layers]
                       ).to(torch.float32).cpu()


@torch.no_grad()
def generate_state(model, tokenizer, attacker, prompt, device, *,
                   steps, gen_length, block_length, temperature, sampling=None):
    """Generate on one prompt; return (per-layer features, filled text, ids, slots).

    sampling: extra sampler.generate kwargs (decoder, alg, top_p, top_k). Without
    it the sampler falls back to the LLaDA block decoder."""
    from sampler import generate

    user_message = attacker.build_prompt({"prompt": prompt, "target": None})
    x_in = encode_prompt(tokenizer, user_message, device)
    slots = (x_in == MASK_ID)[0].nonzero().flatten()
    x = generate(model, x_in, None, steps=steps, gen_length=gen_length,
                 block_length=block_length, temperature=temperature, **(sampling or {}))
    if slots.numel() == 0:  # no prompt masks: the generated suffix is the response
        slots = torch.arange(x_in.shape[1], x.shape[1], device=x.device)
    filled = tokenizer.decode(x[0, slots], skip_special_tokens=True)
    # The guard misfires on the "Step k:" scaffolding; judge the bare text.
    filled = re.sub(r"Step \d+:\s*", "", filled).strip()
    return sample_state(model, x, slots), filled, x[0].tolist(), slots.tolist()


def fit_logistic(X, y, C, steps=400, balanced=False):
    """L2-regularised logistic regression via LBFGS; returns (w, b).

    balanced=True reweights the positive class by n_neg / n_pos so a rare
    unsafe class is not drowned out (sklearn's class_weight="balanced").
    """
    X = X.to(torch.float64)
    y = y.to(torch.float64)
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")
    lam = 1.0 / (C * len(X))
    pos_weight = None
    if balanced and y.sum() > 0:
        pos_weight = torch.tensor([(len(y) - y.sum()) / y.sum()], dtype=torch.float64)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            X @ w + b, y, pos_weight=pos_weight) + 0.5 * lam * (w @ w)
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().to(torch.float32), float(b.detach())


def youden_cutoff(probs, y):
    """Probability cutoff maximising TPR - FPR; ties -> the higher cutoff."""
    order = np.argsort(-probs)
    best_j, best_t = -1.0, 0.5
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    tp = fp = 0
    for k in range(len(order)):
        i = order[k]
        tp += y[i] == 1
        fp += y[i] == 0
        if k + 1 < len(order) and probs[order[k + 1]] == probs[i]:
            continue  # only cut between distinct probabilities
        j = tp / max(n_pos, 1) - fp / max(n_neg, 1)
        if j > best_j:
            best_j = j
            nxt = probs[order[k + 1]] if k + 1 < len(order) else 0.0
            best_t = float((probs[i] + nxt) / 2)
    return best_t, float(best_j)


def rates(probs, y, cutoff):
    pos, neg = probs[y == 1], probs[y == 0]
    return {"tpr": float((pos >= cutoff).mean()) if len(pos) else float("nan"),
            "fpr": float((neg >= cutoff).mean()) if len(neg) else float("nan")}


def arm_rates(probs, y, arms, cutoff):
    """Trigger rates per arm and class: how often recovery would fire on
    WildJailbreak unsafe / safe rows and on Alpaca rows."""
    out = {}
    for arm in dict.fromkeys(arms):
        sel = arms == arm
        entry = {"n": int(sel.sum()), "n_unsafe": int(y[sel].sum())}
        for cls, name in ((1, "unsafe"), (0, "safe")):
            s = sel & (y == cls)
            entry[f"trigger_{name}"] = float((probs[s] >= cutoff).mean()) if s.any() else None
        out[arm] = entry
    return out


def cv_auroc_by_layer(X_all, y, arms, layers, C, folds, seed, balanced=False):
    """k-fold AUROC per layer; folds are stratified by (arm, label) like the split."""
    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(y), dtype=int)
    for arm in dict.fromkeys(arms):
        for cls in (0, 1):
            idx = np.flatnonzero((arms == arm) & (y == cls))
            fold_of[idx[rng.permutation(len(idx))]] = np.arange(len(idx)) % folds
    yt = torch.tensor(y, dtype=torch.float32)
    out = {}
    for li, L in enumerate(layers):
        aucs = []
        for f in range(folds):
            te, tr = fold_of == f, fold_of != f
            w, b = fit_logistic(X_all[tr, li], yt[tr], C, balanced=balanced)
            lg = (X_all[te, li] @ w + b).numpy()
            aucs.append(auroc(lg[y[te] == 1], lg[y[te] == 0]))
        out[L] = {"mean": float(np.mean(aucs)), "sd": float(np.std(aucs))}
    return out


def load_arm_rows(args):
    """(group id, prompt, arm) rows, drawn with the code-share branch's RNG order
    (WildJailbreak permutation first, then Alpaca), so --seed 0 reproduces the
    groups response_detector2.pt was trained on."""
    rng = np.random.default_rng(args.seed)
    df = pd.read_csv(args.csv)
    picked = df.iloc[rng.permutation(len(df))[: args.groups]]
    rows = [(int(gi), str(r["prompt"]), "wildjailbreak") for gi, r in picked.iterrows()]
    if args.alpaca_groups:
        alpaca = pd.read_parquet(args.alpaca)
        picked = alpaca.iloc[rng.permutation(len(alpaca))[: args.alpaca_groups]]
        for gi, r in picked.iterrows():
            extra, prompt = str(r["input"]).strip(), str(r["instruction"]).strip()
            rows.append((ALPACA_GROUP_OFFSET + int(gi),
                         f"{prompt}\n\n{extra}" if extra else prompt, "alpaca"))
    return rows



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "data/llada8b_wild_unsafe_only.csv"),
                    help="WildJailbreak prompt source (only the prompts are used)")
    add_model_arg(ap)
    ap.add_argument("--out", default=str(ROOT / OUT_DIR / "response_detector.pt"))
    ap.add_argument("--report", default=None,
                    help="default: <out without .pt>_report.json")
    ap.add_argument("--groups", type=int, default=384)
    ap.add_argument("--alpaca", default=ALPACA_PARQUET,
                    help="Alpaca parquet (instruction/input columns)")
    ap.add_argument("--alpaca-groups", type=int, default=0,
                    help="ordinary instructions mixed in as a benign arm; answered "
                         "without any attack wrapper")
    ap.add_argument("--arms", default=None,
                    help="comma list of arms to generate in this run "
                         "(wildjailbreak, alpaca); default all")
    ap.add_argument("--gen-only", action="store_true",
                    help="generate + judge + cache, then stop (merge with --from-samples)")
    ap.add_argument("--layer", type=int, default=DETECTOR_LAYER,
                    help="Must equal the gate layer V3 runs with. Default 18 for "
                         "llada, else the --detector bundle's best_layer.")
    ap.add_argument("--detector", default=str(ROOT / OUT_DIR / "steer_detector.pt"),
                    help="Prompt-side detector bundle; only read to resolve --layer.")
    ap.add_argument("--C", type=float, default=0.01)
    ap.add_argument("--threshold", default="auto",
                    help="probability cutoff V3 triggers on: a float, or 'auto' = "
                         "Youden's J on the validation groups")
    ap.add_argument("--balanced", action="store_true",
                    help="class-balanced logistic loss (rare unsafe class upweighted)")
    ap.add_argument("--val-frac", type=float, default=0.25,
                    help="fraction of prompt groups held out to pick the cutoff")
    ap.add_argument("--cv-folds", type=int, default=5,
                    help="k for the per-layer cross-validated AUROC (0 skips it)")
    ap.add_argument("--baseline", default=None,
                    help="existing checkpoint to score on the same validation groups")
    ap.add_argument("--samples", default=None,
                    help="cache of generated features/texts/labels "
                         "(default: <out dir>/response_samples_<attack>.pt)")
    ap.add_argument("--from-samples", default=None,
                    help="comma list of cached sample files to refit from; skips "
                         "generation and judging")
    ap.add_argument("--attack", choices=["none", "dija"], default="none",
                    help="prompt wrapper for the WildJailbreak arm: none = plain "
                         "answers, dija = synthetic DIJA template (DIJATemplate)")
    ap.add_argument("--dija-steps", type=int, default=4,
                    help="DIJATemplate 'Step k:' lines")
    ap.add_argument("--dija-span", type=int, default=16,
                    help="mask tokens per DIJATemplate step")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--gen-length", type=int, default=None,
                    help="default 0 for dija (span infilling only), else 128")
    ap.add_argument("--alpaca-steps", type=int, default=128)
    ap.add_argument("--alpaca-gen-length", type=int, default=128)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--decoder", default=None,
                    help="sampler decoder; default dream for --model dream, else block")
    ap.add_argument("--alg", default="origin")
    ap.add_argument("--top-p", type=float, default=None,
                    help="decoder=dream nucleus p (default 0.95)")
    ap.add_argument("--top-k", type=int, default=None,
                    help="decoder=dream top-k (default 50; 0 disables)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--guard-device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.layer is None:
        args.layer = int(load_detector_bundle(args.detector)["best_layer"])
        print(f"--layer not given: using {args.detector} best_layer {args.layer}")
    report_path = Path(args.report or str(Path(args.out).with_suffix("")) + "_report.json")

    decoder = args.decoder or ("dream" if MODEL_KEY == "dream" else "block")
    sampling = {"decoder": decoder}
    if decoder == "dream":
        top_k = 50 if args.top_k is None else args.top_k
        sampling.update(alg=args.alg, top_p=0.95 if args.top_p is None else args.top_p,
                        top_k=top_k if top_k > 0 else None)

    threshold = None if args.threshold == "auto" else float(args.threshold)
    samples_path = Path(args.samples or
                        Path(args.out).parent / f"response_samples_{args.attack}.pt")

    if args.from_samples:
        feats, texts, labels, groups, prompts, arms = [], [], [], [], [], []
        cached_layers, meta = None, {}
        for path in args.from_samples.split(","):
            cache = torch.load(path, map_location="cpu", weights_only=False)
            if cache.get("model") != MODEL_NAME:
                raise SystemExit(f"{path} was built for {cache.get('model')}, "
                                 f"not {MODEL_NAME}")
            f = cache["feats"]
            if f.ndim == 2:   # older caches hold one layer as [n, hidden]
                f = f[:, None, :]
            layers = list(cache.get("layers", [cache.get("layer")]))
            if cached_layers is None:
                cached_layers = layers
            elif layers != cached_layers:
                raise SystemExit(f"{path} caches layers {layers}, others {cached_layers}")
            n = len(f)
            feats += list(f)
            texts += cache["texts"]
            labels += cache["labels"]
            groups += [int(g) for g in cache["groups"]]
            prompts += cache["prompts"]
            arms += list(cache.get("arms", ["wildjailbreak"] * n))
            meta[path] = {k: cache.get(k) for k in
                          ("attack", "steps", "gen_length", "temperature", "decoder",
                           "sampling", "seed")}
            print(f"{n} cached samples from {path} "
                  f"(arms={sorted(set(cache.get('arms', ['wildjailbreak'])))}, "
                  f"attack={cache.get('attack')}, steps={cache.get('steps')}, "
                  f"decoder={cache.get('decoder', 'block')})")
    else:
        rows = load_arm_rows(args)
        want = set(args.arms.split(",")) if args.arms else {a for _, _, a in rows}
        rows = [r for r in rows if r[2] in want]
        print(f"{len(rows)} prompt groups: "
              + ", ".join(f"{a} {sum(r[2] == a for r in rows)}" for a in dict.fromkeys(
                  r[2] for r in rows)))

        from Attacker import DIJATemplate, NoAttack
        wj_attacker = (DIJATemplate(args.dija_steps, args.dija_span)
                       if args.attack == "dija" else NoAttack())
        wj_gen = args.gen_length
        if wj_gen is None:
            wj_gen = 0 if args.attack == "dija" else 128
        arm_cfg = {"wildjailbreak": (wj_attacker, args.steps, wj_gen),
                   "alpaca": (NoAttack(), args.alpaca_steps, args.alpaca_gen_length)}

        print(f"loading {MODEL_NAME} ... sampling={sampling}")
        tokenizer, model = load_model(args.device)

        cached_layers = ALL_LAYERS
        feats, texts, groups, prompts, ids, slot_lists, arms = [], [], [], [], [], [], []
        for k, (gi, prompt, arm) in enumerate(rows):
            attacker, steps, gen_length = arm_cfg[arm]
            seed_all(args.seed + k)   # per-prompt seed, as exp.py does
            feat, filled, x_ids, slots = generate_state(
                model, tokenizer, attacker, prompt, args.device,
                steps=steps, gen_length=gen_length, block_length=args.block_length,
                temperature=args.temperature, sampling=sampling)
            feats.append(feat)
            texts.append(filled)
            groups.append(gi)
            prompts.append(prompt)
            ids.append(x_ids)
            slot_lists.append(slots)
            arms.append(arm)
            if (k + 1) % 25 == 0:
                print(f"  generated [{k + 1}/{len(rows)}]", flush=True)
        del model
        torch.cuda.empty_cache()

        from Evaluator import LlamaGuard4
        print(f"judging with {LlamaGuard4.__doc__.strip().splitlines()[0]} ...")
        labels = []
        with LlamaGuard4(device=args.guard_device) as guard:
            for k, (prompt, filled) in enumerate(zip(prompts, texts)):
                v = guard._classify(prompt, filled)
                labels.append(v["label"])
                if (k + 1) % 50 == 0:
                    print(f"  judged [{k + 1}/{len(rows)}]", flush=True)

        samples_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"feats": torch.stack(feats), "layers": cached_layers,
                    "texts": texts, "labels": labels, "ids": ids, "slots": slot_lists,
                    "groups": groups, "prompts": prompts, "arms": arms,
                    "layer": args.layer, "model": MODEL_NAME, "attack": args.attack,
                    "steps": {a: arm_cfg[a][1] for a in set(arms)},
                    "gen_length": {a: arm_cfg[a][2] for a in set(arms)},
                    "block_length": args.block_length, "temperature": args.temperature,
                    "decoder": decoder, "sampling": sampling, "seed": args.seed},
                   samples_path)
        print(f"-> {samples_path} (cached samples; refit with --from-samples)")
        meta = {str(samples_path): {"attack": args.attack, "decoder": decoder,
                                    "sampling": sampling, "seed": args.seed}}
        if args.gen_only:
            return

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
    yn = y.numpy()
    X_all = torch.stack([feats[i] for i in keep])          # [n, layers, hidden]
    if args.layer not in cached_layers:
        raise SystemExit(f"layer {args.layer} not in the cached layers {cached_layers}")
    X = X_all[:, cached_layers.index(args.layer)]
    groups = np.array([groups[i] for i in keep])
    arms = np.array([arms[i] for i in keep])
    print(f"labels: {int(y.sum())} unsafe / {int((~y.bool()).sum())} safe"
          + (f" ({n_drop} unparsed dropped)" if n_drop else ""))
    for arm in dict.fromkeys(arms):
        sel = arms == arm
        print(f"  {arm}: {int(sel.sum())} rows, {int(yn[sel].sum())} unsafe")
    if y.sum() < 2 or (1 - y).sum() < 2:
        raise SystemExit("need at least two samples of each class to fit a detector; "
                         "try --attack dija for more unsafe completions")

    # Group-level split: the cutoff is chosen on groups the regression never saw.
    split_rng = np.random.default_rng(args.seed)
    perm = split_rng.permutation(len(keep))
    n_val = int(round(args.val_frac * len(keep)))
    val_idx, tr_idx = np.sort(perm[:n_val]), np.sort(perm[n_val:])
    yv, yt = yn[val_idx], yn[tr_idx]
    if len(val_idx) == 0 or yv.sum() < 1 or (1 - yv).sum() < 1 or yt.sum() < 1:
        print("validation split lacks a class; fitting and cutting on all groups")
        val_idx = tr_idx = np.arange(len(keep))
        yv = yt = yn
    print(f"fit on {len(tr_idx)} groups ({int(yt.sum())} unsafe), "
          f"cutoff on {len(val_idx)} held-out groups ({int(yv.sum())} unsafe)")

    # Per-layer AUROC (validation split + k-fold CV), so the detector layer can be
    # chosen; V3 needs it to equal the gate layer, so pick the two jointly.
    layer_auc, layer_cv = {}, {}
    if len(cached_layers) > 1:
        if args.cv_folds:
            layer_cv = cv_auroc_by_layer(X_all, yn, arms, cached_layers, args.C,
                                         args.cv_folds, args.seed, balanced=args.balanced)
        print(f"\nlayer  train AUROC  val AUROC  {args.cv_folds}-fold CV")
        for li, L in enumerate(cached_layers):
            wl, bl = fit_logistic(X_all[tr_idx, li], y[tr_idx], args.C, balanced=args.balanced)
            lg = (X_all[:, li] @ wl + bl).numpy()
            a_tr = auroc(lg[tr_idx][yt == 1], lg[tr_idx][yt == 0])
            a_va = auroc(lg[val_idx][yv == 1], lg[val_idx][yv == 0])
            layer_auc[L] = {"train": float(a_tr), "val": float(a_va)}
            cv = (f"  {layer_cv[L]['mean']:.4f}±{layer_cv[L]['sd']:.4f}"
                  if L in layer_cv else "")
            print(f"  {L:2d}     {a_tr:.4f}      {a_va:.4f}{cv}"
                  + ("  <-- --layer" if L == args.layer else ""))
        best_L = max(layer_auc, key=lambda L: layer_auc[L]["val"])
        print(f"best layer by val AUROC: {best_L} ({layer_auc[best_L]['val']:.4f})"
              + (f"; by CV: {max(layer_cv, key=lambda L: layer_cv[L]['mean'])}"
                 if layer_cv else "") + f"; fitting --layer {args.layer}")

    w, b = fit_logistic(X[tr_idx], y[tr_idx], args.C, balanced=args.balanced)
    probs_all = torch.sigmoid(X @ w + b).numpy()
    logits_all = (X @ w + b).numpy()
    auc_tr = auroc(logits_all[tr_idx][yt == 1], logits_all[tr_idx][yt == 0])
    auc_val = auroc(logits_all[val_idx][yv == 1], logits_all[val_idx][yv == 0])
    j = None
    if threshold is None:
        threshold, j = youden_cutoff(probs_all[val_idx], yv)
        print(f"cutoff by Youden's J on validation: {threshold:.4f} (J={j:.3f})")
    r_tr, r_val = rates(probs_all[tr_idx], yt, threshold), rates(probs_all[val_idx], yv, threshold)
    val_arms = arm_rates(probs_all[val_idx], yv, arms[val_idx], threshold)
    bal_acc = (r_val["tpr"] + 1 - r_val["fpr"]) / 2
    print(f"\nlayer {args.layer} C={args.C}{' balanced' if args.balanced else ''}: "
          f"train AUROC={auc_tr:.4f}  val AUROC={auc_val:.4f}  cutoff={threshold:.4f}  "
          f"val balanced-acc={bal_acc:.4f}")
    print(f"  train: unsafe P(trigger)={r_tr['tpr']:.3f}  safe P(trigger)={r_tr['fpr']:.3f}")
    print(f"  val  : unsafe P(trigger)={r_val['tpr']:.3f}  safe P(trigger)={r_val['fpr']:.3f}")
    for arm, e in val_arms.items():
        print(f"  val {arm}: n={e['n']} unsafe={e['n_unsafe']} "
              f"P(trigger|unsafe)={e['trigger_unsafe']} P(trigger|safe)={e['trigger_safe']}")

    baseline = None
    if args.baseline:
        bd = torch.load(args.baseline, map_location="cpu", weights_only=False)
        bL = int(bd["layer"])
        if bL in cached_layers:
            Xb = X_all[:, cached_layers.index(bL)]
            lb = (Xb @ torch.as_tensor(bd["weight"], dtype=torch.float32)
                  + float(bd["bias"])).numpy()
            pb = 1 / (1 + np.exp(-lb))
            bt = float(bd["threshold"])
            j_b, _ = youden_cutoff(pb[val_idx], yv)
            baseline = {"path": args.baseline, "layer": bL, "threshold": bt,
                        "val_auroc": float(auroc(lb[val_idx][yv == 1], lb[val_idx][yv == 0])),
                        "val_rates": rates(pb[val_idx], yv, bt),
                        "val_arm_rates": arm_rates(pb[val_idx], yv, arms[val_idx], bt),
                        "val_youden_cutoff": j_b,
                        "note": "trained on all groups, so its train set includes these "
                                "validation groups"}
            print(f"\nbaseline {args.baseline} (layer {bL}, cutoff {bt}): "
                  f"val AUROC={baseline['val_auroc']:.4f} "
                  f"TPR={baseline['val_rates']['tpr']:.3f} FPR={baseline['val_rates']['fpr']:.3f}")

    mixed = "alpaca" in set(arms)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"weight": w, "bias": b, "threshold": float(threshold),
                "layer": args.layer, "model": MODEL_NAME,
                "pool": ("mean_committed_response_tokens" if mixed
                         else "mean_committed_template_slots"),
                "target": "visible_response_unsafe",
                "source": ("wildjailbreak+alpaca_response_states" if mixed
                           else "wildjailbreak_response_states"),
                "attack": args.attack, "alpaca_group_offset": ALPACA_GROUP_OFFSET,
                "train_groups": [int(g) for g in groups[tr_idx]],
                "validation_groups": [int(g) for g in groups[val_idx]],
                "C": float(args.C), "balanced": bool(args.balanced)}, args.out)
    report_path.write_text(json.dumps(
        {"groups": int(len(X)), "n_train": int(len(tr_idx)), "n_val": int(len(val_idx)),
         "n_unsafe": int(y.sum()), "n_safe": int((1 - y).sum()),
         "per_arm_counts": {a: {"n": int((arms == a).sum()),
                                "unsafe": int(yn[arms == a].sum())}
                            for a in dict.fromkeys(arms)},
         "layer": args.layer, "C": args.C, "balanced": bool(args.balanced),
         "auroc_train": auc_tr, "auroc_val": auc_val,
         "threshold": float(threshold), "threshold_rule": args.threshold,
         "youden_j": j, "val_balanced_accuracy": bal_acc,
         "train_rates": r_tr, "val_rates": r_val, "val_arm_rates": val_arms,
         "auroc_by_layer": layer_auc, "cv_auroc_by_layer": layer_cv,
         "baseline": baseline, "samples": meta, "seed": args.seed}, indent=2))
    print(f"-> {args.out}\n-> {report_path}")


if __name__ == "__main__":
    main()
