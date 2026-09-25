"""Aggregate Dream over-refusal (ours, v3 + prompt remask 0.8) over seeds 42/43/44."""
import json, statistics as st, sys
from pathlib import Path
ROOT = Path("/home/yejin/contents/DLM_Steering/outputs/dream"); TAG = "v3rp80"
SETS = {"XSTest-safe": "XSTest-safe-none", "TruthfulQA": "TQA-none"}
SEEDS = [42, 43, 44]
def gen_stats(rows):
    n = len(rows)
    trig = sum(1 for r in rows if any(a.get("trigger") for a in r.get("boundary_audits", [])))
    rem = sum(1 for r in rows if any(e.get("applied") for e in r.get("recovery_events", [])))
    gate = sum(1 for r in rows if r.get("gate_open"))
    return {"n": n, "gate_open_rate": round(gate / n, 4), "detector_trigger_rate": round(trig / n, 4),
            "remask_rate": round(rem / n, 4), "mean_seconds": round(st.mean(r["seconds"] for r in rows), 2)}
def ms(xs): return f"{st.mean(xs)*100:.1f} ± {(st.stdev(xs) if len(xs)>1 else 0)*100:.1f}"
out, lines = {}, ["| set | seed | n | over-refusal % (full+partial) | full refusal % | detector trigger % | remask % | gate open % |", "|---|---|---|---|---|---|---|---|"]
for name, stem in SETS.items():
    per = []
    for s in SEEDS:
        g = ROOT / f"{stem}-{TAG}-{s}.json"; j = ROOT / f"{stem}-{TAG}-{s}_judged.json"
        if not g.exists(): continue
        gs = gen_stats(json.load(open(g))["results"])
        js = json.load(open(j))["summary"] if j.exists() else {}
        rec = {"seed": s, **gs, **{k: js.get(k) for k in ("refusal_rate", "full_refusal_rate", "2_full_refusal", "3_partial_refusal", "n_parse_error")}}
        per.append(rec)
        rr = f"{rec['refusal_rate']*100:.1f}" if rec.get("refusal_rate") is not None else "-"
        fr = f"{rec['full_refusal_rate']*100:.1f}" if rec.get("full_refusal_rate") is not None else "-"
        lines.append(f"| {name} | {s} | {gs['n']} | {rr} | {fr} | {gs['detector_trigger_rate']*100:.1f} | {gs['remask_rate']*100:.1f} | {gs['gate_open_rate']*100:.1f} |")
    judged = [r for r in per if r.get("refusal_rate") is not None]
    summary = {"per_seed": per}
    if judged:
        summary["mean_std"] = {"over_refusal": ms([r["refusal_rate"] for r in judged]),
                               "full_refusal": ms([r["full_refusal_rate"] for r in judged]),
                               "detector_trigger": ms([r["detector_trigger_rate"] for r in per]),
                               "remask": ms([r["remask_rate"] for r in per])}
        lines.append(f"| **{name}** | mean±std | | **{summary['mean_std']['over_refusal']}** | {summary['mean_std']['full_refusal']} | {summary['mean_std']['detector_trigger']} | {summary['mean_std']['remask']} | |")
    out[name] = summary
json.dump(out, open(ROOT / f"OR-summary-{TAG}.json", "w"), indent=2)
md = "\n".join(lines); (ROOT / f"OR-summary-{TAG}.md").write_text(md + "\n"); print(md)
