"""Fit or calibrate a V3 prompt gate from separate harmful and benign CSVs.

The harmful CSV's first 20 rows are reserved for evaluation. The remaining
rows and the benign CSV are split before fitting. Threshold calibration uses
only harmful fit rows, with no benchmark result leakage.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import MODEL_NAME, OUT_DIR, auroc, load_llada
from models import add_model_arg
from steering.fit_detector import collect


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_model_arg(ap)
    ap.add_argument("--harmful-csv", default="data/llada8b_wild_unsafe_only.csv")
    ap.add_argument("--benign-csv", default="data/jbb_benign.csv")
    ap.add_argument("--reuse-detector", action="store_true")
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--gen-length", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--percentile", type=float, default=15.0)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    if not 0 < args.percentile < 100:
        raise ValueError("percentile must be between 0 and 100")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    harmful = pd.read_csv(args.harmful_csv)["prompt"].astype(str).tolist()[20:]
    benign = pd.read_csv(args.benign_csv)["Goal"].astype(str).tolist()
    if len(harmful) < 20 or len(benign) < 20:
        raise ValueError("Both classes need at least 20 prompts")

    rng = np.random.default_rng(args.seed)
    hi, bi = rng.permutation(len(harmful)), rng.permutation(len(benign))
    hval = max(10, round(len(hi) * .2))
    bval = max(10, round(len(bi) * .2))
    hfit, htest = hi[hval:], hi[:hval]
    bfit, btest = bi[bval:], bi[:bval]
    tokenizer, model = load_llada(args.device)
    # collect() returns (N, 1, hidden_size) for the requested layer.
    H = collect(model, tokenizer, harmful, args.gen_length,
                [args.layer], args.device, "harmful")[:, 0]
    B = collect(model, tokenizer, benign, args.gen_length,
                [args.layer], args.device, "benign")[:, 0]

    detector_path = out_dir / "steer_detector.pt"
    if args.reuse_detector:
        bundle = torch.load(detector_path, map_location="cpu", weights_only=True)
        if bundle.get("model") != MODEL_NAME:
            raise ValueError("Existing detector belongs to a different model")
        vector = bundle["vector"][bundle["layers"].index(args.layer)].float()
    else:
        vector = (H[hfit].mean(0) - B[bfit].mean(0)).float()
        vector /= vector.norm().clamp_min(1e-8)
        bundle = {"vector": vector.unsqueeze(0), "layers": [args.layer],
                  "best_layer": args.layer, "gen_length": args.gen_length,
                  "model": MODEL_NAME, "fit_source": "wild_csv_prompt_vs_jbb_benign"}
        torch.save(bundle, detector_path)

    ph = (H @ vector).numpy()
    pb = (B @ vector).numpy()
    threshold = float(np.percentile(ph[hfit], args.percentile))
    val_auc = float(auroc(ph[htest], pb[btest]))
    result = {"layer": args.layer, "percentile": args.percentile,
              "threshold": round(threshold, 4),
              "open_rate_harmful": round(float((ph[htest] >= threshold).mean()), 4),
              "open_rate_benign": round(float((pb[btest] >= threshold).mean()), 4),
              "validation_auroc": round(val_auc, 4),
              "n_fit_harmful": len(hfit), "n_val_harmful": len(htest),
              "n_fit_benign": len(bfit), "n_val_benign": len(btest),
              "harmful_source": args.harmful_csv, "benign_source": args.benign_csv,
              "seed": args.seed, "model": MODEL_NAME,
              "detector_reused": args.reuse_detector}
    (out_dir / "gate_threshold.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
