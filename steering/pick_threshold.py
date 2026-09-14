"""Choose the gate threshold from fit-split prompts only.

The simulation picked 4.14 by reading the eval sets, which is test-set tuning.
This selects on the 366 fit-split prompts instead, so the held-out harmful
prompts and the benign benchmarks stay untouched.

The default rule is a percentile of the *harmful* fit distribution: it states a
safety-side guarantee directly ("this fraction of harmful prompts gets steered")
and does not depend on which benign set one happens to imagine. Youden's J on
harmful-vs-benign is reported as a cross-check, but it is calibrated on long
WildJailbreak roleplay benign prompts and transfers poorly to short questions.

Usage:
    CUDA_VISIBLE_DEVICES=1 python steering/pick_threshold.py --layer 18
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common import MODEL_NAME, load_detector_bundle, load_llada  # noqa: E402
from steering.fit_detector import collect  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default=str(ROOT / "outputs/steer_detector.pt"))
    ap.add_argument("--pairs", default=str(ROOT / "data/steer_pairs.json"))
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--percentile", type=float, default=15.0,
                    help="Gate opens for (100-p)%% of fit-split harmful prompts.")
    ap.add_argument("--out", default=str(ROOT / "outputs/gate_threshold.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    b = load_detector_bundle(args.detector)
    layers, gen_length = b["layers"], b["gen_length"]
    li = layers.index(args.layer)
    v = b["vector"][li]

    pairs = [p for p in json.loads(Path(args.pairs).read_text())["pairs"]
             if p["split"] == "fit"]
    print(f"fit pairs: {len(pairs)}")

    tokenizer, model = load_llada(device)

    H = collect(model, tokenizer, [p["adv_harmful"] for p in pairs],
                gen_length, layers, device, "harm")
    B = collect(model, tokenizer, [p["adv_benign"] for p in pairs],
                gen_length, layers, device, "ben ")
    ph = (H[:, li] @ v).numpy()
    pb = (B[:, li] @ v).numpy()
    print(f"\nfit harmful proj: mean {ph.mean():.2f} sd {ph.std():.2f}")
    print(f"fit benign  proj: mean {pb.mean():.2f} sd {pb.std():.2f}")

    th = float(np.percentile(ph, args.percentile))

    # Youden's J, as a cross-check on the percentile rule.
    cand = np.unique(np.concatenate([ph, pb]))
    j = [( (ph >= t).mean() - (pb >= t).mean(), t) for t in cand]
    best_j, th_j = max(j)

    print(f"\nthreshold @ p{args.percentile:g} of fit harmful : {th:.3f}"
          f"  (opens for {(ph >= th).mean():.2f} harmful, {(pb >= th).mean():.2f} benign)")
    print(f"threshold @ Youden's J                  : {th_j:.3f}  (J={best_j:.3f},"
          f" opens for {(ph >= th_j).mean():.2f} harmful, {(pb >= th_j).mean():.2f} benign)")

    Path(args.out).write_text(json.dumps({
        "layer": args.layer, "percentile": args.percentile,
        "threshold": round(th, 4), "threshold_youden": round(float(th_j), 4),
        "fit_harmful_mean": round(float(ph.mean()), 4),
        "fit_harmful_std": round(float(ph.std()), 4),
        "fit_benign_mean": round(float(pb.mean()), 4),
        "fit_benign_std": round(float(pb.std()), 4),
        "open_rate_harmful": round(float((ph >= th).mean()), 4),
        "open_rate_benign": round(float((pb >= th).mean()), 4),
        "n_fit_pairs": len(pairs),
    }, indent=2))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
