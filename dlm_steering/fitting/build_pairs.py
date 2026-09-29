"""Build the contrast dataset for the jailbreak steering vector.

The 386 prompts in data/llada8b_wild_unsafe_only.csv are all WildJailbreak
`adversarial_harmful` rows. Each such row carries the same-intent bare request
in `vanilla` and a *refusal* in `completion`; the CSV's own `response` column is
a *compliant* (unsafe) answer to the same prompt. That gives two contrasts:

  R-side (primary)   same prompt, refusal response  vs  compliant response
                     -> prompt is byte-identical, so length/roleplay style
                        cannot leak into the direction.
  P-side (secondary) adv_harmful vs van_harmful, minus the same difference
                     measured on length-matched adversarial_benign pairs
                     (difference-in-differences, removes the wrapping style).

The first --n-eval CSV rows are the prompts already used for baseline eval, and
are excluded from fitting entirely.

Usage:
    python -m dlm_steering.fitting.build_pairs [--n-eval 20] [--seed 0]
"""

import argparse
import glob
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa

WJ_GLOB = "/mnt/shared/huggingface-cache/datasets/allenai___wildjailbreak/train-*/0.0.0/*/*.arrow"
from dlm_steering.paths import REPO as ROOT


def norm(s):
    return " ".join(str(s).split())


def load_wildjailbreak():
    tables = []
    for f in sorted(glob.glob(WJ_GLOB)):
        with pa.memory_map(f) as src:
            tables.append(pa.ipc.open_stream(src).read_all())
    if not tables:
        raise FileNotFoundError(f"no wildjailbreak arrow shards under {WJ_GLOB}")
    return pa.concat_tables(tables).to_pandas()


def length_matched_sample(pool, targets, seed):
    """Pick one pool row per target length, nearest unused length first.

    Targets are consumed longest-first: long prompts are the scarce end of the
    benign pool, so matching them first keeps the total error small.
    """
    pool = pool.assign(_len=pool["adversarial"].str.len()).sort_values("_len")
    lengths = pool["_len"].to_numpy()
    used, picked = set(), []
    for t in sorted(targets, reverse=True):
        lo = int(lengths.searchsorted(t))
        best = None
        for j in range(len(pool)):
            for cand in (lo - j - 1, lo + j):
                if 0 <= cand < len(pool) and cand not in used:
                    if best is None or abs(int(lengths[cand]) - t) < abs(int(lengths[best]) - t):
                        best = cand
            if best is not None:
                break
        used.add(best)
        picked.append(pool.iloc[best])
    return pd.DataFrame(picked).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "data/llada8b_wild_unsafe_only.csv"))
    ap.add_argument("--out", default=str(ROOT / "data/steer_pairs.json"))
    ap.add_argument("--n-eval", type=int, default=20,
                    help="Leading CSV rows reserved for the held-out eval run.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    wj = load_wildjailbreak()
    df = pd.read_csv(args.csv)

    adv_index = {}
    for i, a in enumerate(wj["adversarial"]):
        key = norm(a)
        if key:
            adv_index.setdefault(key, i)

    rows, missing = [], 0
    for k, p in enumerate(df["prompt"]):
        i = adv_index.get(norm(p))
        if i is None:
            missing += 1
            continue
        r = wj.iloc[i]
        rows.append({
            "csv_index": k,
            "wj_index": int(i),
            "adv_harmful": r["adversarial"],
            "van_harmful": r["vanilla"],
            "refusal_response": str(r["completion"]),
            "compliant_response": str(df["response"].iloc[k]),
        })
    print(f"harmful rows matched : {len(rows)}/{len(df)} (missing {missing})")

    benign_pool = wj[wj.data_type == "adversarial_benign"]
    benign = length_matched_sample(
        benign_pool, [len(r["adv_harmful"]) for r in rows], args.seed
    ).to_dict("records")
    print(f"benign control pairs : {len(benign)} (length-matched from {len(benign_pool)})")

    pairs = []
    for n, (r, b) in enumerate(zip(rows, benign)):
        held_out = r["csv_index"] < args.n_eval
        pairs.append({
            **r,
            "pair_id": n,
            "adv_benign": b["adversarial"],
            "van_benign": b["vanilla"],
            # WildJailbreak's completion for a benign row is a *compliant*
            # answer, which is the fourth cell the difference-in-differences
            # vector needs: benign prompt + compliance.
            "benign_compliant_response": str(b["completion"]),
            "split": "held_out_eval" if held_out else "fit",
        })

    n_fit = sum(p["split"] == "fit" for p in pairs)
    n_held = len(pairs) - n_fit
    stats = {k: round(sum(len(p[k]) for p in pairs) / len(pairs), 1)
             for k in ("adv_harmful", "van_harmful", "adv_benign", "van_benign",
                       "refusal_response", "compliant_response",
                       "benign_compliant_response")}
    print("mean chars           :", stats)
    print(f"split                : fit={n_fit}  held_out_eval={n_held} "
          f"(csv_index < {args.n_eval})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"n_pairs": len(pairs), "n_fit": n_fit, "n_held_out": n_held,
                   "n_eval_excluded": args.n_eval, "seed": args.seed,
                   "mean_chars": stats, "pairs": pairs}, f,
                  ensure_ascii=False, indent=2)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
