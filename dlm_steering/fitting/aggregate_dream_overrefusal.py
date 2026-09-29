"""Aggregate the Dream over-refusal runs over seeds.

Reads <out-dir>/{XSTest-safe-none,TQA-none}-<tag>-<seed>.json (generation:
gate/detector/remask rates) and the matching *_judged.json (XSTest 3-way
refusal judge) and writes <out-dir>/OR-summary-<tag>.{json,md}.

Usage:
    python -m dlm_steering.fitting.aggregate_dream_overrefusal --out-dir outputs/dream --tag v3rp80
"""
import argparse
import json
import statistics as st
from pathlib import Path

SETS = {"XSTest-safe": "XSTest-safe-none", "TruthfulQA": "TQA-none"}


def gen_stats(rows):
    n = len(rows)
    trig = sum(1 for r in rows if any(a.get("trigger") for a in r.get("boundary_audits", [])))
    rem = sum(1 for r in rows if any(e.get("applied") for e in r.get("recovery_events", [])))
    gate = sum(1 for r in rows if r.get("gate_open"))
    secs = [r["seconds"] for r in rows if "seconds" in r]
    return {"n": n, "gate_open_rate": round(gate / n, 4), "detector_trigger_rate": round(trig / n, 4),
            "remask_rate": round(rem / n, 4),
            "mean_seconds": round(st.mean(secs), 2) if secs else None}


def ms(xs):
    return f"{st.mean(xs) * 100:.1f} ± {(st.stdev(xs) if len(xs) > 1 else 0) * 100:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/dream")
    ap.add_argument("--tag", default="v3rp80")
    ap.add_argument("--seeds", default="42,43,44")
    args = ap.parse_args()
    root, tag = Path(args.out_dir), args.tag
    seeds = [int(s) for s in args.seeds.split(",")]

    out = {}
    lines = ["| set | seed | n | over-refusal % (full+partial) | full refusal % | "
             "detector trigger % | remask % | gate open % |",
             "|---|---|---|---|---|---|---|---|"]
    for name, stem in SETS.items():
        per = []
        for s in seeds:
            g = root / f"{stem}-{tag}-{s}.json"
            j = root / f"{stem}-{tag}-{s}_judged.json"
            if not g.exists():
                continue
            gs = gen_stats(json.load(open(g))["results"])
            js = json.load(open(j))["summary"] if j.exists() else {}
            rec = {"seed": s, **gs,
                   **{k: js.get(k) for k in ("refusal_rate", "full_refusal_rate",
                                             "2_full_refusal", "3_partial_refusal",
                                             "n_parse_error")}}
            per.append(rec)
            rr = f"{rec['refusal_rate'] * 100:.1f}" if rec.get("refusal_rate") is not None else "-"
            fr = (f"{rec['full_refusal_rate'] * 100:.1f}"
                  if rec.get("full_refusal_rate") is not None else "-")
            lines.append(f"| {name} | {s} | {gs['n']} | {rr} | {fr} | "
                         f"{gs['detector_trigger_rate'] * 100:.1f} | {gs['remask_rate'] * 100:.1f} | "
                         f"{gs['gate_open_rate'] * 100:.1f} |")
        judged = [r for r in per if r.get("refusal_rate") is not None]
        summary = {"per_seed": per}
        if judged:
            summary["mean_std"] = {
                "over_refusal": ms([r["refusal_rate"] for r in judged]),
                "full_refusal": ms([r["full_refusal_rate"] for r in judged]),
                "detector_trigger": ms([r["detector_trigger_rate"] for r in per]),
                "remask": ms([r["remask_rate"] for r in per])}
            m = summary["mean_std"]
            lines.append(f"| **{name}** | mean±std | | **{m['over_refusal']}** | {m['full_refusal']} | "
                         f"{m['detector_trigger']} | {m['remask']} | |")
        out[name] = summary
    (root / f"OR-summary-{tag}.json").write_text(json.dumps(out, indent=2))
    md = "\n".join(lines)
    (root / f"OR-summary-{tag}.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
