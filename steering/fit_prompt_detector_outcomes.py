"""Fit the prompt gate to predict LLaDA 1.5's own unsafe response outcome.

Labels come from direct target-model generations on the user's CSV prompts,
judged by Llama Guard 4. The first 20 CSV rows remain held out; another
stratified 20% split tests the detector before threshold calibration.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common import MODEL_NAME, auroc, load_llada
from models import add_model_arg
from steering.fit_detector import collect


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_model_arg(ap)
    ap.add_argument("--in", dest="input", default="data/llada1.5_model_pairs.jsonl")
    ap.add_argument("--out-dir", default="outputs/llada1.5")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--gen-length", type=int, default=128)
    ap.add_argument("--percentile", type=float, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines()
            if line.strip()]
    if len(rows) != 386 or len({r["index"] for r in rows}) != len(rows):
        raise ValueError("Need 386 unique target-model prompt outcomes")
    rows = sorted((r for r in rows if r["index"] >= 20 and
                   (r["direct_label"].startswith("safe") or
                    r["direct_label"].startswith("unsafe"))),
                  key=lambda r: r["index"])
    labels = np.array([int(r["direct_label"].startswith("unsafe")) for r in rows])
    rng = np.random.default_rng(args.seed)
    fit, val = [], []
    for label in (0, 1):
        indices = rng.permutation(np.flatnonzero(labels == label))
        n_val = max(10, round(len(indices) * .2))
        val.extend(indices[:n_val])
        fit.extend(indices[n_val:])
    if min(int(labels[fit].sum()), int(len(fit) - labels[fit].sum())) < 20:
        raise ValueError("Too few examples of one outcome class")

    tokenizer, model = load_llada(args.device)
    states = collect(model, tokenizer, [r["prompt"] for r in rows],
                     args.gen_length, [args.layer], args.device, "prompt")[:, 0]
    bad = [i for i in fit if labels[i] == 1]
    safe = [i for i in fit if labels[i] == 0]
    vector = (states[bad].mean(0) - states[safe].mean(0)).float()
    vector /= vector.norm().clamp_min(1e-8)
    projection = (states @ vector).numpy()
    positive = np.array([projection[i] for i in val if labels[i] == 1])
    negative = np.array([projection[i] for i in val if labels[i] == 0])
    auc = float(auroc(positive, negative))
    threshold = float(np.percentile(projection[bad], args.percentile))
    report = {"model": MODEL_NAME, "layer": args.layer,
              "source": args.input, "label": "direct_response_lg4_unsafe",
              "n_train": len(fit), "n_val": len(val),
              "n_train_unsafe": len(bad), "n_train_safe": len(safe),
              "validation_auroc": round(auc, 4),
              "validation_unsafe_open_rate": round(float((positive >= threshold).mean()), 4),
              "validation_safe_open_rate": round(float((negative >= threshold).mean()), 4),
              "percentile": args.percentile, "threshold": round(threshold, 4),
              "seed": args.seed}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompt_outcome_detector_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if auc < .6:
        raise ValueError("Held-out outcome AUROC below 0.6; gate not fit for experiments")
    torch.save({"vector": vector.unsqueeze(0), "layers": [args.layer],
                "best_layer": args.layer, "gen_length": args.gen_length,
                "model": MODEL_NAME, "fit_source": args.input,
                "validation_auroc": auc}, out / "steer_detector.pt")
    (out / "gate_threshold.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
