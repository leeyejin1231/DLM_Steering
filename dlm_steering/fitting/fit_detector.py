"""Fit a prompt-side harmfulness detector and test whether it generalises.

This is the *detection* half of a split design: one direction decides whether a
request is harmful, a separate one (the plain refusal vector) drives the refusal.
Blending the two into a single direction failed, because a direction that reads
harmfulness out is not a direction that causes refusal.

The detector is read with the answer region fully masked, which is the state the
sampler is in at the first denoising step -- before any token is committed and
before a response-conditioned contrast such as the DiD vector carries any signal
at all (with everything masked, its two arms are byte-identical).

    v_detect = mean(h | harmful prompt, answer all MASK)
             - mean(h | benign prompt,  answer all MASK)

The benign arm is the length- and style-matched `adv_benign` prompt, so roleplay
wrapping and prompt length are already controlled by the pairing.

The decisive number is not the held-out AUROC on WildJailbreak pairs -- it is the
AUROC against the benign sets where steering actually over-refused. The detector
is fitted on long roleplay prompts; XSTest and TruthfulQA are short plain
questions, a different distribution entirely. If it cannot separate those, a gate
built on it will stay open and over-refusal will not improve.

Usage:
    CUDA_VISIBLE_DEVICES=1 python -m dlm_steering.fitting.fit_detector
"""

from dlm_steering.runtime.constants import OUT_DIR, add_model_arg

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dlm_steering.runtime.constants import FIT_LAYERS, MODEL_NAME, MASK_ID
from dlm_steering.runtime.utils import auroc
from dlm_steering.runtime.data import load_eval_prompts
from dlm_steering.runtime.models import load_llada, prompt_token_ids

from dlm_steering.paths import REPO as ROOT

@torch.no_grad()
def prompt_state(model, tokenizer, prompt, gen_length, layers, device):
    """Hidden states averaged over a fully masked answer region, per layer."""
    p_ids = prompt_token_ids(tokenizer, str(prompt))
    x = torch.full((1, len(p_ids) + gen_length), MASK_ID, dtype=torch.long, device=device)
    x[0, : len(p_ids)] = torch.tensor(p_ids, device=device)
    hs = model(x, output_hidden_states=True).hidden_states
    h = torch.stack([hs[L][0, len(p_ids):, :].mean(dim=0) for L in layers], dim=0)
    return h.to(torch.float32).cpu()


def collect(model, tokenizer, prompts, gen_length, layers, device, tag):
    out = []
    for i, p in enumerate(prompts):
        out.append(prompt_state(model, tokenizer, p, gen_length, layers, device))
        if (i + 1) % 50 == 0:
            print(f"  {tag} [{i + 1}/{len(prompts)}]", flush=True)
    return torch.stack(out)


def main():
    ap = argparse.ArgumentParser()
    add_model_arg(ap)
    ap.add_argument("--pairs", default=str(ROOT / "data/steer_pairs.json"))
    ap.add_argument("--csv", default=str(ROOT / "data/llada8b_wild_unsafe_only.csv"))
    ap.add_argument("--out", default=str(ROOT / OUT_DIR / "steer_detector.pt"))
    ap.add_argument("--report", default=str(ROOT / OUT_DIR / "steer_detector_report.json"))
    ap.add_argument("--gen-length", type=int, default=128)
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--n-eval", type=int, default=30)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    layers = FIT_LAYERS  # shared with fit_vector.py so the picks compare
    device = "cuda" if torch.cuda.is_available() else "cpu"

    meta = json.loads(Path(args.pairs).read_text())
    pairs = [p for p in meta["pairs"] if p["split"] == "fit"]
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    print(f"fit pairs: {len(pairs)}")

    print(f"loading {MODEL_NAME} ...")
    tokenizer, model = load_llada(device)

    H = collect(model, tokenizer, [p["adv_harmful"] for p in pairs],
                args.gen_length, layers, device, "harmful")
    B = collect(model, tokenizer, [p["adv_benign"] for p in pairs],
                args.gen_length, layers, device, "benign ")

    n = len(pairs)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n)
    n_val = max(8, int(round(args.val_frac * n)))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    print(f"fit on {len(tr_idx)} pairs, held-out {len(val_idx)}")

    v = (H[tr_idx] - B[tr_idx]).mean(dim=0)
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    in_dist = [round(float(auroc((H[val_idx, li] @ v[li]).numpy(),
                                (B[val_idx, li] @ v[li]).numpy())), 4)
               for li in range(len(layers))]

    # The prompts the gate must actually judge at inference time.
    df = pd.read_csv(args.csv)
    eval_harmful = df["prompt"].head(20).tolist()
    eval_sets = {
        "xstest_safe": load_eval_prompts("xstest_safe", args.n_eval),
        "truthfulqa": load_eval_prompts("truthfulqa", args.n_eval),
        "jbb_benign": load_eval_prompts("jbb_benign", args.n_eval),
    }
    print("\nprojecting held-out eval prompt sets ...")
    Eh = collect(model, tokenizer, eval_harmful, args.gen_length, layers, device, "eval-harm")
    Eb = {k: collect(model, tokenizer, ps, args.gen_length, layers, device, k)
          for k, ps in eval_sets.items()}

    ood = {k: [round(float(auroc((Eh[:, li] @ v[li]).numpy(),
                                 (M[:, li] @ v[li]).numpy())), 4)
               for li in range(len(layers))] for k, M in Eb.items()}

    print("\n            in-dist |  held-out 20 harmful vs ...")
    print("layer   WJ pairs | XSTest  TruthQA  JBB-ben  | mean")
    means = []
    for li, L in enumerate(layers):
        row = [ood[k][li] for k in ("xstest_safe", "truthfulqa", "jbb_benign")]
        means.append(sum(row) / 3)
        print(f"  {L:2d}    {in_dist[li]:.3f}  |  {row[0]:.3f}   {row[1]:.3f}    {row[2]:.3f}   | {means[-1]:.3f}")

    best_li = int(np.argmax(means))
    best_layer = layers[best_li]
    print(f"\nbest layer by mean OOD auroc: {best_layer}  mean={means[best_li]:.4f}")
    for k in ("xstest_safe", "truthfulqa", "jbb_benign"):
        print(f"  vs {k:12s} {ood[k][best_li]:.4f}")
    print(f"  in-dist (WildJailbreak held-out pairs) {in_dist[best_li]:.4f}")

    act_norm = [round(float(H[:, li].norm(dim=-1).mean()), 2) for li in range(len(layers))]
    report = {
        "n_pairs": n, "n_train": len(tr_idx), "n_val": len(val_idx),
        "gen_length": args.gen_length, "layers": layers,
        "auroc_in_distribution": dict(zip(map(str, layers), in_dist)),
        "auroc_vs_eval_sets": {k: dict(zip(map(str, layers), val)) for k, val in ood.items()},
        "mean_ood_auroc": dict(zip(map(str, layers), [round(m, 4) for m in means])),
        "best_layer_by_ood": best_layer,
        "mean_act_norm_by_layer": dict(zip(map(str, layers), act_norm)),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"vector": v, "layers": layers, "best_layer": best_layer,
                "auroc_in_distribution": in_dist, "auroc_vs_eval_sets": ood,
                "mean_ood_auroc": means, "mean_act_norm": act_norm,
                "gen_length": args.gen_length, "model": MODEL_NAME}, args.out)
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(f"\n-> {args.out}\n-> {args.report}")


if __name__ == "__main__":
    main()
