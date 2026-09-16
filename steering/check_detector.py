"""Re-test the prompt-side detector under contrasts that control for length.

fit_detector.py compared held-out WildJailbreak harmful prompts (long roleplay,
~1000 chars) against XSTest and TruthfulQA (short plain questions, ~60 chars).
Early layers scored ~1.00 there while scoring only ~0.70 on the length-matched
WildJailbreak pairs -- a detector cannot be worse on the controlled contrast than
the uncontrolled one, so those numbers are reading prompt length and style, not
harmfulness.

These contrasts hold length and source fixed and vary only permissibility:

    xstest_unsafe vs xstest_safe   same benchmark, both short questions
    jbb_harmful   vs jbb_benign    index-matched counterparts, topic controlled

Usage:
    CUDA_VISIBLE_DEVICES=1 python steering/check_detector.py
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
    MODEL_NAME, OUT_DIR, add_model_arg, auroc, load_detector_bundle,
    load_eval_prompts, load_model)
from steering.fit_detector import collect  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    add_model_arg(ap)
    ap.add_argument("--detector", default=str(ROOT / OUT_DIR / "steer_detector.pt"))
    ap.add_argument("--out", default=str(ROOT / OUT_DIR / "detector_controlled.json"))
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    b = load_detector_bundle(args.detector)
    v, layers, gen_length = b["vector"], b["layers"], b["gen_length"]

    tokenizer, model = load_model(device)

    sets = {}
    for name in ("xstest_safe", "xstest_unsafe", "jbb_benign", "jbb_harmful"):
        ps = load_eval_prompts(name, args.n)
        chars = sum(len(str(p)) for p in ps) / len(ps)
        print(f"{name}: {len(ps)} prompts, mean {chars:.0f} chars")
        sets[name] = (collect(model, tokenizer, ps, gen_length, layers, device, name), chars)

    contrasts = [("XSTest  unsafe vs safe", "xstest_unsafe", "xstest_safe"),
                 ("JBB     harmful vs benign", "jbb_harmful", "jbb_benign")]
    results = {}
    print("\nlayer | XSTest u-vs-s | JBB h-vs-b")
    per_layer = {}
    for li, L in enumerate(layers):
        row = []
        for _, pos, neg in contrasts:
            a = float(auroc((sets[pos][0][:, li] @ v[li]).numpy(),
                            (sets[neg][0][:, li] @ v[li]).numpy()))
            row.append(a)
        per_layer[L] = row
        print(f"  {L:2d}  |    {row[0]:.3f}      |   {row[1]:.3f}")

    for i, (label, pos, neg) in enumerate(contrasts):
        vals = [per_layer[L][i] for L in layers]
        best = layers[int(np.argmax(vals))]
        results[label] = {"by_layer": {str(L): round(per_layer[L][i], 4) for L in layers},
                          "best_layer": best, "best_auroc": round(max(vals), 4),
                          "mean_chars_pos": round(sets[pos][1], 1),
                          "mean_chars_neg": round(sets[neg][1], 1)}
        print(f"\n{label}: best layer {best}, auroc {max(vals):.4f} "
              f"(chars {sets[pos][1]:.0f} vs {sets[neg][1]:.0f})")

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
