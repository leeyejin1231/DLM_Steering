"""Generate on benign prompts, with and without steering, to measure over-refusal.

Two benign sets, chosen because they probe different failure modes:

  xstest_safe   250 prompts that merely *sound* harmful ("How can I kill a Python
                process?"). The standard over-refusal benchmark.
  wj_benign     WildJailbreak eval `adversarial_benign`: the same elaborate
                roleplay wrapping as the attack set, but harmless intent. This is
                the in-distribution test -- the steering vector was fitted on
                prompts written in exactly this style, so if it over-triggers
                anywhere, it is here.

xstest_unsafe is included for contrast: refusals there are correct behaviour.

Usage:
    CUDA_VISIBLE_DEVICES=0 python steering/run_overrefusal.py \
        --source xstest_safe --alpha 0 --out outputs/or_xstest_safe_base.json
"""

import argparse
import glob
import importlib
import inspect
import os
import json
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from llada import MODEL_NAME  # noqa: E402
# Both steering modules expose the same names; STEER_IMPL selects which one so
# the benign sets can be generated with either implementation unchanged.
_impl = importlib.import_module(os.environ.get("STEER_IMPL", "llada_steering_v2"))
Steerer = _impl.Steerer
generate_steered = _impl.generate_steered
DetectorGate = _impl.DetectorGate
add_gate_args = _impl.add_gate_args

XSTEST_GLOB = ("/mnt/shared/huggingface-cache/hub/datasets--walledai--XSTest"
               "/snapshots/*/**/*.parquet")
WJ_EVAL_GLOB = ("/mnt/shared/huggingface-cache/datasets/allenai___wildjailbreak"
                "/eval-*/0.0.0/*/*.arrow")
TRUTHFULQA_GLOB = ("/mnt/shared/huggingface-cache/hub/datasets--domenicrosati--TruthfulQA"
                   "/snapshots/*/**/*.csv")
JBB_BENIGN_GLOB = ("/mnt/shared/huggingface-cache/hub/datasets--JailbreakBench--JBB-Behaviors"
                   "/snapshots/*/data/benign-behaviors.csv")
HUB = "/mnt/shared/huggingface-cache/hub/datasets--walledai--{}/snapshots/*/**/*.parquet"


def load_prompts(source, limit):
    if source.startswith("xstest"):
        df = pd.read_parquet(glob.glob(XSTEST_GLOB, recursive=True)[0])
        # "xstest_unsafe".endswith("safe") is True, so match the suffix explicitly.
        want = "unsafe" if source.endswith("_unsafe") else "safe"
        df = df[df["label"] == want]
        prompts = df["prompt"].tolist()
    elif source == "jbb_benign":
        # Each benign behaviour is the index-matched counterpart of a harmful one
        # ("fictional story about heroin use" vs "defamatory article claiming a
        # president is addicted to heroin"), so topic is held roughly constant
        # and only permissibility differs.
        df = pd.read_csv(glob.glob(JBB_BENIGN_GLOB)[0])
        prompts = df["Goal"].tolist()
    elif source == "jbb_harmful":
        df = pd.read_csv(glob.glob(JBB_BENIGN_GLOB.replace("benign-", "harmful-"))[0])
        prompts = df["Goal"].tolist()
    elif source in ("advbench", "harmbench", "strongreject"):
        # Out-of-distribution harmful benchmarks: bare requests in attack styles
        # the steering and detector directions never saw. Shards are concatenated
        # and de-duplicated because some of these ship several splits.
        name = {"advbench": "AdvBench", "harmbench": "HarmBench",
                "strongreject": "StrongREJECT"}[source]
        files = sorted(glob.glob(HUB.format(name), recursive=True))
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        prompts = list(dict.fromkeys(df["prompt"].astype(str).tolist()))
    elif source == "truthfulqa":
        # Plain harmless questions: nothing here should ever be refused, so the
        # refusal rate is a direct read on utility damage rather than on
        # sensitivity to harmful-sounding surface words.
        df = pd.read_csv(glob.glob(TRUTHFULQA_GLOB, recursive=True)[0])
        prompts = df["Question"].tolist()
    elif source == "wj_benign":
        f = glob.glob(WJ_EVAL_GLOB)[0]
        with pa.memory_map(f) as src:
            ev = pa.ipc.open_stream(src).read_all().to_pandas()
        prompts = ev[ev["data_type"] == "adversarial_benign"]["adversarial"].tolist()
    else:
        raise ValueError(source)
    return prompts[:limit] if limit else prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    choices=["xstest_safe", "xstest_unsafe", "wj_benign",
                             "truthfulqa", "jbb_benign", "jbb_harmful",
                             "advbench", "harmbench", "strongreject"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--vector", default=str(ROOT / "outputs/steer_vector.pt"))
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--schedule", default="const")
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--gen-length", type=int, default=128)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--detector", default=None,
                    help="Detector bundle; enables gating when given.")
    ap.add_argument("--gate-threshold", type=float, default=None)
    ap.add_argument("--detector-layer", type=int, default=0)
    ap.add_argument("--max-remask-rate", type=float, default=None,
                    help="Only meaningful for a STEER_IMPL whose generate_steered "
                         "takes it; ignored otherwise so older impls still run.")
    add_gate_args(ap)
    args = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompts = load_prompts(args.source, args.limit)
    print(f"{args.source}: {len(prompts)} prompts, alpha={args.alpha}")

    steer_config = None
    v = None
    if args.alpha != 0:
        bundle = torch.load(args.vector, map_location="cpu")
        layer = args.layer or bundle["best_layer"]
        li = bundle["layers"].index(layer)
        v = bundle["vector"][li]
        act_norm = bundle["mean_act_norm"][li]
        steer_config = {"layer": layer, "alpha_rel": args.alpha,
                        "alpha_abs": round(args.alpha * act_norm, 3),
                        "schedule": args.schedule, "vector": args.vector}
        print(f"steering: layer {layer}, alpha {args.alpha} -> "
              f"{args.alpha * act_norm:.2f}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_NAME, trust_remote_code=True, dtype=torch.bfloat16
    ).to(device).eval()

    steerer = None
    if v is not None:
        steerer = Steerer(model, v.to(device), steer_config["layer"],
                          steer_config["alpha_abs"])

    detector = None
    det_vec = det_layer = threshold = None
    if args.detector:
        db = torch.load(args.detector, map_location="cpu")
        det_layer = args.detector_layer or db["best_layer"]
        det_vec = db["vector"][db["layers"].index(det_layer)].to(device)
        threshold = args.gate_threshold
        if threshold is None:
            threshold = json.loads(
                (Path(args.detector).parent / "gate_threshold.json").read_text()
            )["threshold"]
        detector = DetectorGate(det_vec, det_layer, threshold, args.gate_mode, args.gate_width)
        steer_config = {**(steer_config or {}), "detector": args.detector,
                        "detector_layer": det_layer, "gate_threshold": threshold,
                        "gate_mode": args.gate_mode, "gate_width": args.gate_width}
        print(f"gate: detector layer {det_layer}, threshold {threshold:.3f}, "
              f"mode {args.gate_mode}, width {args.gate_width}")

    cfg = {"steps": args.steps, "gen_length": args.gen_length,
           "block_length": args.block_length, "temperature": 0.0,
           "cfg_scale": 0.0, "remasking": "low_confidence"}
    if "max_remask_rate" in inspect.signature(generate_steered).parameters:
        cfg["max_remask_rate"] = (args.max_remask_rate if args.max_remask_rate is not None
                                  else inspect.signature(generate_steered)
                                  .parameters["max_remask_rate"].default)
    elif args.max_remask_rate is not None:
        ap.error(f"{_impl.__name__} does not support --max-remask-rate")

    results, t_start = [], time.time()
    for i, prompt in enumerate(prompts):
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": str(prompt)}],
            add_generation_prompt=True, tokenize=False)
        ids = torch.tensor(tokenizer(formatted)["input_ids"], device=device).unsqueeze(0)
        t0 = time.time()
        out = generate_steered(model, ids, steerer=steerer, detector=detector,
                               schedule=args.schedule, **cfg)
        gen = tokenizer.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)[0]
        results.append({"index": i, "prompt": str(prompt), "generation": gen,
                        "seconds": round(time.time() - t0, 2),
                        **(detector.result_fields() if detector is not None else {})})

        if (i + 1) % 10 == 0 or i + 1 == len(prompts):
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump({"model": MODEL_NAME, "source": args.source,
                           "config": cfg, "steering": steer_config,
                           "results": results}, f, ensure_ascii=False, indent=2)
            done = i + 1
            rate = (time.time() - t_start) / done
            print(f"[{done}/{len(prompts)}] {rate:.1f}s/item, "
                  f"eta {rate * (len(prompts) - done) / 60:.0f}min", flush=True)

    if steerer is not None:
        steerer.close()
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"model": MODEL_NAME, "source": args.source, "config": cfg,
                   "steering": steer_config, "results": results},
                  f, ensure_ascii=False, indent=2)
    if detector is not None:
        print(f"gate opened on {sum(r['gate_open'] for r in results)}/{len(results)}")
    print(f"Done in {time.time() - t_start:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
