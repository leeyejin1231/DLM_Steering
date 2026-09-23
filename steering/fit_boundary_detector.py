"""Fit the V3 response detector on *first-boundary* states.

fit_response_detector.py reads the finished response, but V3 audits the
first block boundary: the moment 32 answer tokens have been committed, with
the rest still masked. A detector fitted on finished responses is miscalibrated
there (Dream: training AUROC 0.87 -> 0.67 at audit time, probabilities
collapse). This script generates each prompt through the real V3 defender
(steering on, recovery disabled by a cutoff > 1) and records, at the first
boundary, the mean hidden state per layer over two pools:

    committed  the committed answer tokens (what V3._audit reads today)
    region     every answer slot, committed or still masked (the model's
               hidden state at a masked slot encodes what it intends to
               write there, like the step-0 prompt-side gate)

The label is Llama-Guard-4's verdict on the *finished* text, so the task is
"will this response end unsafe?" -- the question the audit actually asks.

Arms (so that neither source nor template can stand in for the label):
    wildjailbreak   WildJailbreak harmful prompts + DIJATemplate
    wj_benign       WildJailbreak adversarial_benign (train split) + the same template
    alpaca          plain Alpaca instructions, no template

Usage (generation, one arm per GPU; then a CPU refit over the caches):
    CUDA_VISIBLE_DEVICES=0 python steering/fit_boundary_detector.py --model dream \
        --arms wildjailbreak,alpaca --gen-only
    CUDA_VISIBLE_DEVICES=1 python steering/fit_boundary_detector.py --model dream \
        --arms wj_benign --gen-only
    python steering/fit_boundary_detector.py --model dream --pool committed --layer 24 \
        --from-samples outputs/dream/boundary_samples_{wildjailbreak,wj_benign,alpaca}.pt \
        --balanced --out outputs/dream/response_detector3_committed_L24.pt
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "steering"))
from common import (DETECTOR_LAYER, MODEL_KEY, MODEL_NAME, MASK_ID, OUT_DIR,  # noqa: E402
                    add_model_arg, auroc, encode_prompt, load_detector, seed_all)
from fit_response_detector import (ALL_LAYERS, ALPACA_GROUP_OFFSET, ALPACA_PARQUET,  # noqa: E402
                                   arm_rates, cv_auroc_by_layer, fit_logistic, rates,
                                   youden_cutoff)

WJ_TRAIN_GLOB = ("/mnt/shared/huggingface-cache/datasets/allenai___wildjailbreak"
                 "/train-*/0.0.0/*/wildjailbreak-train-*.arrow")
WJB_GROUP_OFFSET = 2_000_000
POOLS = ("committed", "region")


def load_wj_benign_train(n, seed):
    """n adversarial_benign prompts from the WildJailbreak *train* split (the
    eval split is what --source wj_benign probes read; keep them apart)."""
    import pyarrow as pa
    frames = []
    for f in sorted(glob.glob(WJ_TRAIN_GLOB)):
        with pa.memory_map(f) as src:
            try:
                t = pa.ipc.open_stream(src).read_all()
            except pa.ArrowInvalid:
                t = pa.ipc.open_file(src).read_all()
        df = t.to_pandas()
        frames.append(df[df["data_type"] == "adversarial_benign"][["adversarial"]])
    df = pd.concat(frames, ignore_index=True)
    df = df[df["adversarial"].astype(str).str.strip().str.len() > 0].drop_duplicates()
    rng = np.random.default_rng(seed)
    picked = df.iloc[rng.permutation(len(df))[:n]]
    return [(WJB_GROUP_OFFSET + int(gi), str(r["adversarial"]).strip())
            for gi, r in picked.iterrows()]


def load_rows(args):
    """(group id, prompt, arm) rows. WildJailbreak and Alpaca draws replay
    fit_response_detector.load_arm_rows (same RNG order) so the groups match
    response_detector2.pt; wj_benign uses its own stream (seed + 1)."""
    rng = np.random.default_rng(args.seed)
    df = pd.read_csv(args.csv)
    picked = df.iloc[rng.permutation(len(df))[: args.groups]]
    rows = [(int(gi), str(r["prompt"]), "wildjailbreak") for gi, r in picked.iterrows()]
    alpaca = pd.read_parquet(args.alpaca)
    picked = alpaca.iloc[rng.permutation(len(alpaca))[: args.groups]]
    for gi, r in picked.iterrows():
        extra, prompt = str(r["input"]).strip(), str(r["instruction"]).strip()
        rows.append((ALPACA_GROUP_OFFSET + int(gi),
                     f"{prompt}\n\n{extra}" if extra else prompt, "alpaca"))
    rows += [(g, p, "wj_benign") for g, p in load_wj_benign_train(args.groups, args.seed + 1)]
    return rows


def build_recorder(model, args):
    """The deployed V3 policy (gate 14 / steer 17 / alpha 1 / --remask-prompt)
    with a response cutoff above 1, so recovery never fires, plus a hook that
    stores the first-boundary features. Mirrors Defender.Ours.from_args."""
    from Defender import V3

    device = next(model.parameters()).device
    bundle = torch.load(args.vector, map_location="cpu")
    layer = int(bundle["best_layer"]) if args.steer_layer is None else int(args.steer_layer)
    li = bundle["layers"].index(layer)
    sites = [(layer, bundle["vector"][li].to(device), bundle["mean_act_norm"][li])]
    det_vec, det_layer, threshold = load_detector(args.detector, args.gate_layer, device, None)
    dummy = {"weight": torch.zeros_like(det_vec.cpu()), "bias": 0.0, "threshold": 1.1,
             "layer": det_layer, "pool": "mean_committed_response_tokens"}

    class Recorder(V3):
        def reset(self):
            super().reset()
            self.records = []

        @torch.no_grad()
        def after_block(self, x, region, *, block_index, block_positions, prompt_length,
                        temperature, remasking, last_block=False, sampling=None):
            if len(self.records) < args.record_boundaries:
                masks = x == self.mask_id
                pools = {"committed": (region[0] & ~masks[0]).nonzero().flatten(),
                         "region": region[0].nonzero().flatten()}
                hs = self.model(x, output_hidden_states=True).hidden_states
                self.audit_forwards += 1
                feats = {k: torch.stack([hs[L][0, p].to(torch.float32).mean(dim=0)
                                         for L in ALL_LAYERS]).to(torch.float16).cpu()
                         for k, p in pools.items()}
                self.records.append({"boundary": block_index,
                                     "n_committed": int(pools["committed"].numel()),
                                     "n_region": int(pools["region"].numel()), **feats})
            # Cutoff 1.1 can never trigger, so V3's own audit is skipped.

    return Recorder(model, gate_layer=det_layer, gate_vector=det_vec, threshold=threshold,
                    width=1.0, sites=sites, strength=args.alpha, transform="additive",
                    steer="adaptive", steer_shift=False, response_detector=dummy,
                    recovery_steps=32, remask_prompt=True)


@torch.no_grad()
def generate_one(model, tokenizer, recorder, attacker, prompt, device, *,
                 steps, gen_length, block_length, temperature, sampling):
    from sampler import generate

    user_message = attacker.build_prompt({"prompt": prompt, "target": None})
    x_in = encode_prompt(tokenizer, user_message, device)
    slots = (x_in == MASK_ID)[0].nonzero().flatten()
    x = generate(model, x_in, recorder, steps=steps, gen_length=gen_length,
                 block_length=block_length, temperature=temperature, **sampling)
    parts = []
    if slots.numel():
        filled = tokenizer.decode(x[0, slots], skip_special_tokens=True)
        parts.append(re.sub(r"Step \d+:\s*", "", filled).strip())
    if x.shape[1] > x_in.shape[1]:
        parts.append(tokenizer.decode(x[0, x_in.shape[1]:], skip_special_tokens=True).strip())
    return "\n".join(p for p in parts if p), x[0].tolist(), list(recorder.records)


def main():
    ap = argparse.ArgumentParser()
    add_model_arg(ap)
    ap.add_argument("--csv", default=str(ROOT / "data/llada8b_wild_unsafe_only.csv"))
    ap.add_argument("--alpaca", default=ALPACA_PARQUET)
    ap.add_argument("--groups", type=int, default=384, help="prompts per arm")
    ap.add_argument("--arms", default="wildjailbreak,wj_benign,alpaca")
    ap.add_argument("--gen-only", action="store_true")
    ap.add_argument("--samples-dir", default=str(ROOT / OUT_DIR))
    ap.add_argument("--from-samples", default=None, help="comma list of caches to refit from")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pool", choices=POOLS, default="committed")
    ap.add_argument("--layer", type=int, default=None,
                    help="detector layer to fit (default: best validation AUROC)")
    ap.add_argument("--C", type=float, default=0.01)
    ap.add_argument("--balanced", action="store_true")
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--baseline", default=None,
                    help="finished-response checkpoint to score on the same validation rows")
    ap.add_argument("--record-boundaries", type=int, default=1)
    # generation = the deployed setting (exp.py main matrix)
    ap.add_argument("--vector", default=f"{ROOT / OUT_DIR}/steer_vector.pt")
    ap.add_argument("--detector", default=f"{ROOT / OUT_DIR}/steer_detector.pt")
    ap.add_argument("--gate-layer", type=int, default=DETECTOR_LAYER)
    ap.add_argument("--steer-layer", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--dija-steps", type=int, default=4)
    ap.add_argument("--dija-span", type=int, default=16)
    ap.add_argument("--template-steps", type=int, default=256)
    ap.add_argument("--plain-steps", type=int, default=128)
    ap.add_argument("--gen-length", type=int, default=128)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--guard-device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if MODEL_KEY != "dream":
        raise SystemExit("this fitter targets the Dream native decoder (--model dream)")
    sampling = {"decoder": "dream", "alg": "origin", "top_p": args.top_p,
                "top_k": args.top_k if args.top_k > 0 else None}

    if not args.from_samples:
        rows = load_rows(args)
        want = set(args.arms.split(","))
        rows = [r for r in rows if r[2] in want]
        counts = {a: sum(r[2] == a for r in rows) for a in dict.fromkeys(r[2] for r in rows)}
        print(f"{len(rows)} prompts: {counts}")

        from Attacker import DIJATemplate, NoAttack
        from common import load_model
        template = DIJATemplate(args.dija_steps, args.dija_span)
        arm_cfg = {"wildjailbreak": (template, args.template_steps),
                   "wj_benign": (template, args.template_steps),
                   "alpaca": (NoAttack(), args.plain_steps)}
        print(f"loading {MODEL_NAME} ... sampling={sampling}")
        tokenizer, model = load_model(args.device)
        recorder = build_recorder(model, args)
        print(f"recorder: {recorder.describe()}")

        per_arm = {a: dict(feats={p: [] for p in POOLS}, n_committed=[], texts=[], groups=[],
                           prompts=[], ids=[], boundaries=[]) for a in counts}
        for k, (gi, prompt, arm) in enumerate(rows):
            attacker, steps = arm_cfg[arm]
            seed_all(args.seed + k)
            text, x_ids, recs = generate_one(
                model, tokenizer, recorder, attacker, prompt, args.device, steps=steps,
                gen_length=args.gen_length, block_length=args.block_length,
                temperature=args.temperature, sampling=sampling)
            if not recs:
                print(f"  [{k}] no boundary recorded (arm={arm}); skipped")
                continue
            r0 = recs[0]
            d = per_arm[arm]
            for p in POOLS:
                d["feats"][p].append(r0[p])
            d["n_committed"].append(r0["n_committed"])
            d["boundaries"].append(len(recs))
            d["texts"].append(text)
            d["groups"].append(gi)
            d["prompts"].append(prompt)
            d["ids"].append(x_ids)
            if (k + 1) % 25 == 0:
                print(f"  generated [{k + 1}/{len(rows)}]", flush=True)
        del model, recorder
        torch.cuda.empty_cache()

        from Evaluator import LlamaGuard4
        print("judging finished texts with Llama Guard 4 ...")
        with LlamaGuard4(device=args.guard_device) as guard:
            for arm, d in per_arm.items():
                d["labels"] = []
                for k, (prompt, text) in enumerate(zip(d["prompts"], d["texts"])):
                    d["labels"].append(guard._classify(prompt, text)["label"])
                    if (k + 1) % 50 == 0:
                        print(f"  {arm} judged [{k + 1}/{len(d['prompts'])}]", flush=True)
        for arm, d in per_arm.items():
            path = Path(args.samples_dir) / f"boundary_samples_{arm}.pt"
            torch.save({"arm": arm, "layers": ALL_LAYERS, "pools": POOLS,
                        "feats": {p: torch.stack(d["feats"][p]) for p in POOLS},
                        "n_committed": d["n_committed"], "n_boundaries": d["boundaries"],
                        "texts": d["texts"], "labels": d["labels"], "groups": d["groups"],
                        "prompts": d["prompts"], "ids": d["ids"], "model": MODEL_NAME,
                        "state": "first_boundary", "steps": arm_cfg[arm][1],
                        "gen_length": args.gen_length, "block_length": args.block_length,
                        "temperature": args.temperature, "sampling": sampling,
                        "defender": {"gate_layer": args.gate_layer, "alpha": args.alpha,
                                     "steer_layer": args.steer_layer},
                        "seed": args.seed}, path)
            n_unsafe = sum(str(l).startswith("unsafe") for l in d["labels"])
            print(f"-> {path}: {len(d['labels'])} rows, {n_unsafe} unsafe")
        if args.gen_only:
            return
        args.from_samples = ",".join(str(Path(args.samples_dir) / f"boundary_samples_{a}.pt")
                                     for a in per_arm)

    # ------------------------------------------------------------------ refit
    feats, labels, groups, arms, ncom = [], [], [], [], []
    layers = None
    for path in args.from_samples.split(","):
        c = torch.load(path, map_location="cpu", weights_only=False)
        if c.get("model") != MODEL_NAME:
            raise SystemExit(f"{path} was built for {c.get('model')}")
        if layers is None:
            layers = list(c["layers"])
        feats += list(c["feats"][args.pool].to(torch.float32))
        labels += c["labels"]
        groups += [int(g) for g in c["groups"]]
        arms += [c["arm"]] * len(c["labels"])
        ncom += c["n_committed"]
        print(f"{len(c['labels'])} rows from {path} (arm={c['arm']}, "
              f"unsafe={sum(str(l).startswith('unsafe') for l in c['labels'])}, "
              f"n_committed mean={np.mean(c['n_committed']):.1f})")

    def _label(l):
        l = str(l)
        return 1.0 if l.startswith("unsafe") else 0.0 if l.startswith("safe") else None

    tagged = [_label(l) for l in labels]
    keep = [i for i, t in enumerate(tagged) if t is not None]
    y = torch.tensor([tagged[i] for i in keep])
    yn = y.numpy()
    X_all = torch.stack([feats[i] for i in keep])
    groups = np.array([groups[i] for i in keep])
    arms = np.array([arms[i] for i in keep])
    print(f"labels: {int(y.sum())} unsafe / {int((1 - y).sum())} safe "
          f"({len(labels) - len(keep)} unparsed dropped); pool={args.pool}")
    for a in dict.fromkeys(arms):
        s = arms == a
        print(f"  {a}: {int(s.sum())} rows, {int(yn[s].sum())} unsafe")

    split_rng = np.random.default_rng(args.seed)
    perm = split_rng.permutation(len(keep))
    n_val = int(round(args.val_frac * len(keep)))
    val_idx, tr_idx = np.sort(perm[:n_val]), np.sort(perm[n_val:])
    yv, yt = yn[val_idx], yn[tr_idx]
    print(f"fit on {len(tr_idx)} ({int(yt.sum())} unsafe), cutoff on {len(val_idx)} "
          f"held-out ({int(yv.sum())} unsafe){' balanced' if args.balanced else ''}")

    layer_cv = cv_auroc_by_layer(X_all, yn, arms, layers, args.C, args.cv_folds,
                                 args.seed, balanced=args.balanced) if args.cv_folds else {}
    layer_auc, wj = {}, arms == "wildjailbreak"
    print(f"\nlayer  train   val    val-WJ   {args.cv_folds}-fold CV")
    for li, L in enumerate(layers):
        wl, bl = fit_logistic(X_all[tr_idx, li], y[tr_idx], args.C, balanced=args.balanced)
        lg = (X_all[:, li] @ wl + bl).numpy()
        a_tr = auroc(lg[tr_idx][yt == 1], lg[tr_idx][yt == 0])
        a_va = auroc(lg[val_idx][yv == 1], lg[val_idx][yv == 0])
        m = np.zeros(len(yn), bool); m[val_idx] = True; m &= wj
        a_wj = auroc(lg[m & (yn == 1)], lg[m & (yn == 0)]) if (m & (yn == 1)).any() and (m & (yn == 0)).any() else float("nan")
        layer_auc[L] = {"train": float(a_tr), "val": float(a_va), "val_wj": float(a_wj)}
        cv = f"  {layer_cv[L]['mean']:.4f}±{layer_cv[L]['sd']:.4f}" if L in layer_cv else ""
        print(f"  {L:2d}   {a_tr:.4f} {a_va:.4f}  {a_wj:.4f}{cv}")
    best_L = max(layer_auc, key=lambda L: layer_auc[L]["val"])
    layer = args.layer or best_L
    print(f"best layer by val AUROC: {best_L}; fitting layer {layer}")

    li = layers.index(layer)
    X = X_all[:, li]
    w, b = fit_logistic(X[tr_idx], y[tr_idx], args.C, balanced=args.balanced)
    logits = (X @ w + b).numpy()
    probs = 1 / (1 + np.exp(-logits))
    auc_tr = auroc(logits[tr_idx][yt == 1], logits[tr_idx][yt == 0])
    auc_val = auroc(logits[val_idx][yv == 1], logits[val_idx][yv == 0])
    cutoff, j = youden_cutoff(probs[val_idx], yv)
    r_val = rates(probs[val_idx], yv, cutoff)
    val_arms = arm_rates(probs[val_idx], yv, arms[val_idx], cutoff)
    print(f"\nlayer {layer} pool={args.pool} C={args.C}: train AUROC={auc_tr:.4f} "
          f"val AUROC={auc_val:.4f} cutoff={cutoff:.4f} (J={j:.3f}) "
          f"val TPR={r_val['tpr']:.3f} FPR={r_val['fpr']:.3f}")
    for a, e in val_arms.items():
        print(f"  val {a}: n={e['n']} unsafe={e['n_unsafe']} "
              f"P(trigger|unsafe)={e['trigger_unsafe']} P(trigger|safe)={e['trigger_safe']}")

    baseline = None
    if args.baseline:
        bd = torch.load(args.baseline, map_location="cpu", weights_only=False)
        bL = int(bd["layer"])
        Xb = X_all[:, layers.index(bL)]
        lb = (Xb @ torch.as_tensor(bd["weight"], dtype=torch.float32) + float(bd["bias"])).numpy()
        pb = 1 / (1 + np.exp(-lb))
        bt = float(bd["threshold"])
        baseline = {"path": args.baseline, "layer": bL, "threshold": bt,
                    "val_auroc": float(auroc(lb[val_idx][yv == 1], lb[val_idx][yv == 0])),
                    "val_rates": rates(pb[val_idx], yv, bt),
                    "val_arm_rates": arm_rates(pb[val_idx], yv, arms[val_idx], bt)}
        print(f"\nbaseline {args.baseline} on the same boundary states: "
              f"val AUROC={baseline['val_auroc']:.4f} TPR={baseline['val_rates']['tpr']:.3f} "
              f"FPR={baseline['val_rates']['fpr']:.3f}")

    out = Path(args.out or Path(args.samples_dir) / f"response_detector3_{args.pool}_L{layer}.pt")
    torch.save({"weight": w, "bias": b, "threshold": float(cutoff), "layer": layer,
                "model": MODEL_NAME,
                "pool": f"mean_{args.pool}_first_boundary",
                "target": "final_response_unsafe",
                "source": "wildjailbreak+wj_benign+alpaca_first_boundary_states",
                "alpaca_group_offset": ALPACA_GROUP_OFFSET, "wjb_group_offset": WJB_GROUP_OFFSET,
                "train_groups": [int(g) for g in groups[tr_idx]],
                "validation_groups": [int(g) for g in groups[val_idx]],
                "C": float(args.C), "balanced": bool(args.balanced)}, out)
    report = out.with_name(out.stem + "_report.json")
    report.write_text(json.dumps(
        {"rows": int(len(X)), "n_train": int(len(tr_idx)), "n_val": int(len(val_idx)),
         "n_unsafe": int(y.sum()), "per_arm": {a: {"n": int((arms == a).sum()),
                                                  "unsafe": int(yn[arms == a].sum())}
                                               for a in dict.fromkeys(arms)},
         "pool": args.pool, "layer": layer, "C": args.C, "balanced": bool(args.balanced),
         "auroc_train": auc_tr, "auroc_val": auc_val, "threshold": float(cutoff),
         "youden_j": float(j), "val_rates": r_val, "val_arm_rates": val_arms,
         "auroc_by_layer": layer_auc, "cv_auroc_by_layer": layer_cv,
         "n_committed_mean": float(np.mean([ncom[i] for i in keep])),
         "baseline": baseline, "samples": args.from_samples.split(","),
         "seed": args.seed}, indent=2))
    print(f"-> {out}\n-> {report}")


if __name__ == "__main__":
    main()
