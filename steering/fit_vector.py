"""Fit the jailbreak steering direction from response-conditioned activations.

For each fit pair the prompt is held byte-identical and only the answer region
changes: once filled with a refusal, once with a compliant (unsafe) answer, both
truncated to the same token count and masked at the same positions. The residual
difference is therefore "which continuation am I writing", with prompt length and
roleplay style fully controlled.

Activations are read at several mask ratios t because a diffusion LM sees the
answer region at every noise level during sampling, and are averaged over the
masked positions only -- those are the positions the sampler actually predicts,
and the positions steering will later be injected at.

hidden_states[L] is the input to block L, i.e. the output of block L-1, so a
direction fitted at layer L is applied by hooking blocks[L-1] (see llada_steering).

Usage:
    CUDA_VISIBLE_DEVICES=1 python steering/fit_vector.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common import (  # noqa: E402
    MODEL_NAME, MASK_ID, auroc, load_llada, prompt_token_ids)


@torch.no_grad()
def collect(model, tokenizer, pairs, t_list, max_resp, layers, seed, device):
    """Return acts[t] -> (refusal, compliant), each (n_pairs, n_layers, d_model)."""
    acts = {t: ([], []) for t in t_list}
    kept = []
    for n, p in enumerate(pairs):
        p_ids = prompt_token_ids(tokenizer, p["adv_harmful"])
        ref = tokenizer(p["refusal_response"], add_special_tokens=False)["input_ids"]
        cmp_ = tokenizer(p["compliant_response"], add_special_tokens=False)["input_ids"]

        # Same token count for both arms removes response length as a confound.
        n_resp = min(len(ref), len(cmp_), max_resp)
        if n_resp < 16:
            continue
        ref, cmp_ = ref[:n_resp], cmp_[:n_resp]
        n_prompt = len(p_ids)

        base = torch.tensor([p_ids + ref, p_ids + cmp_], device=device)
        g = torch.Generator().manual_seed(seed + n)
        perm = torch.randperm(n_resp, generator=g)

        for t in t_list:
            k = max(1, int(round(t * n_resp)))
            pos = perm[:k] + n_prompt
            x = base.clone()
            x[:, pos] = MASK_ID
            hs = model(x, output_hidden_states=True).hidden_states
            # hidden_states[L] == output of blocks[L-1]; average over masked slots.
            h = torch.stack([hs[L][:, pos, :].mean(dim=1) for L in layers], dim=1)
            h = h.to(torch.float32).cpu()
            acts[t][0].append(h[0])
            acts[t][1].append(h[1])

        kept.append(p["pair_id"])
        if (n + 1) % 25 == 0:
            print(f"  [{n + 1}/{len(pairs)}] kept={len(kept)}", flush=True)

    return {t: (torch.stack(a), torch.stack(b)) for t, (a, b) in acts.items()}, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(ROOT / "data/steer_pairs.json"))
    ap.add_argument("--out", default=str(ROOT / "outputs/steer_vector.pt"))
    ap.add_argument("--report", default=str(ROOT / "outputs/steer_vector_report.json"))
    ap.add_argument("--max-pairs", type=int, default=0, help="0 = all fit pairs.")
    ap.add_argument("--max-resp", type=int, default=192, help="Max response tokens.")
    ap.add_argument("--t-list", default="0.3,0.5,0.7,0.9")
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t_list = [float(t) for t in args.t_list.split(",")]
    layers = list(range(1, 32))  # hidden_states[1..31] == blocks[0..30] outputs
    device = "cuda" if torch.cuda.is_available() else "cpu"

    meta = json.loads(Path(args.pairs).read_text())
    pairs = [p for p in meta["pairs"] if p["split"] == "fit"]
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    print(f"fit pairs: {len(pairs)} (held-out eval pairs excluded)")

    print(f"loading {MODEL_NAME} on {device} ...")
    tokenizer, model = load_llada(device)

    print(f"collecting activations at t={t_list}, layers 1..31 ...")
    acts, kept = collect(model, tokenizer, pairs, t_list, args.max_resp,
                         layers, args.seed, device)
    n = len(kept)
    print(f"collected {n} pairs")

    # Pair-level split: a pair's refusal and compliant arms stay on the same side.
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n)
    n_val = max(8, int(round(args.val_frac * n)))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    print(f"vector fit on {len(tr_idx)} pairs, validated on {len(val_idx)}")

    report = {"n_pairs": n, "n_train": len(tr_idx), "n_val": len(val_idx),
              "t_list": t_list, "layers": layers, "per_t": {}}
    dirs = []
    for t in t_list:
        ref, cmp_ = acts[t]
        d = (ref[tr_idx] - cmp_[tr_idx]).mean(dim=0)          # (n_layers, d_model)
        d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        dirs.append(d)
        aucs = []
        for li in range(len(layers)):
            pr = (ref[val_idx, li] @ d[li]).numpy()
            pc = (cmp_[val_idx, li] @ d[li]).numpy()
            aucs.append(round(float(auroc(pr, pc)), 4))
        report["per_t"][str(t)] = {"auroc_by_layer": dict(zip(map(str, layers), aucs)),
                                   "best_layer": layers[int(np.argmax(aucs))],
                                   "best_auroc": max(aucs)}
        print(f"  t={t}: best layer {layers[int(np.argmax(aucs))]} auroc {max(aucs):.4f}")

    # Average the per-t unit directions, then renormalise: a direction that is
    # consistent across noise levels survives, one that is t-specific cancels.
    v = torch.stack(dirs).mean(dim=0)
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    aucs, cos_t = [], []
    for li in range(len(layers)):
        pr = np.concatenate([(acts[t][0][val_idx, li] @ v[li]).numpy() for t in t_list])
        pc = np.concatenate([(acts[t][1][val_idx, li] @ v[li]).numpy() for t in t_list])
        aucs.append(round(float(auroc(pr, pc)), 4))
        cos_t.append(round(float(np.mean([
            float(torch.dot(dirs[i][li], dirs[j][li]))
            for i in range(len(t_list)) for j in range(i + 1, len(t_list))
        ])), 4))

    best_li = int(np.argmax(aucs))
    report["pooled"] = {"auroc_by_layer": dict(zip(map(str, layers), aucs)),
                        "cross_t_cosine_by_layer": dict(zip(map(str, layers), cos_t)),
                        "best_layer": layers[best_li], "best_auroc": aucs[best_li]}

    print("\nlayer  auroc  cross-t cos")
    for li, L in enumerate(layers):
        mark = "  <-- best" if li == best_li else ""
        print(f"  {L:2d}   {aucs[li]:.4f}   {cos_t[li]:+.3f}{mark}")

    # Norm of the residual stream at each layer, so alpha can be set relative to it.
    ref_norm = [float(torch.cat([acts[t][0][:, li] for t in t_list]).norm(dim=-1).mean())
                for li in range(len(layers))]
    report["mean_act_norm_by_layer"] = dict(zip(map(str, layers),
                                                [round(x, 2) for x in ref_norm]))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"vector": v, "layers": layers, "t_list": t_list,
                "best_layer": layers[best_li], "best_auroc": aucs[best_li],
                "mean_act_norm": ref_norm, "kept_pair_ids": kept,
                "model": MODEL_NAME}, args.out)
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(f"\n-> {args.out}\n-> {args.report}")
    print(f"best layer {layers[best_li]}  auroc {aucs[best_li]:.4f}  "
          f"mean |h| {ref_norm[best_li]:.1f}")


if __name__ == "__main__":
    main()
