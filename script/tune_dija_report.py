"""Markdown tables for the DIJA tuning runs (script/tune_dija.sh).

Every table reads the generation JSON and its Llama Guard verdicts under
outputs/tune; rows whose files are not there yet are shown as "-", so this can
be rerun while a sweep is still going.

Usage: python script/tune_dija_report.py [outputs/tune]
"""

import json
import statistics
import sys
from pathlib import Path

DATASETS = ("JBB", "HarmBench", "SR")


def load(root, stem):
    g, lg = root / f"{stem}.json", root / f"{stem}_lg4.json"
    if not g.exists() or not lg.exists():
        return None
    rows = json.loads(g.read_text())["results"]
    verdicts = json.loads(lg.read_text())
    return {"rows": rows, "asr": verdicts["summary"]["asr"],
            "labels": {r["index"]: r["verdict"]["label"] for r in verdicts["results"]},
            "recovered": sum(bool(r.get("recovery_events")) for r in rows),
            "sec": statistics.mean(r.get("seconds", 0) for r in rows)}


def fmt(v, spec=".3f"):
    return "-" if v is None else format(v, spec)


def table(head, body):
    print("| " + " | ".join(head) + " |")
    print("|" + "|".join("---" for _ in head) + "|")
    for row in body:
        print("| " + " | ".join(row) + " |")
    print()


def boundaries(root):
    print("### 경계별 ASR (T=0.2, alpha=1.0, recovery steps 32 / rounds 1)\n")
    cols = [("nodef", "무방어"), ("steeronly", "steer-only"), ("bnd0", "v3 경계0"),
            ("bnd1", "v3 경계1"), ("bnd2", "v3 경계2"), ("bnd3", "v3 경계3"),
            ("bndend", "v3 종료시점")]
    body, means = [], {k: [] for k, _ in cols}
    for P in DATASETS:
        row = [P]
        for key, _ in cols:
            d = load(root, f"dija-{P}-{key}")
            row.append(fmt(d and d["asr"]))
            if d:
                means[key].append(d["asr"])
        body.append(row)
    body.append(["**평균**"] + [fmt(statistics.mean(v)) if len(v) == len(DATASETS) else "-"
                                for v in means.values()])
    table(["데이터셋"] + [c for _, c in cols], body)

    print("### recovery 발동 행 수 / 해당 경계에 도달한 행 비율\n")
    body = []
    for P in DATASETS:
        row = [P]
        for b in ("0", "1", "2", "3", "end"):
            d = load(root, f"dija-{P}-bnd{b}")
            if not d:
                row.append("-"); continue
            n = len(d["rows"])
            reach = 1.0 if b == "end" else sum(r["num_prompt_masks"] > 32 * (int(b) + 1) for r in d["rows"]) / n
            row.append(f"{d['recovered']}/{n} ({reach:.0%})")
        body.append(row)
    table(["데이터셋", "경계0", "경계1", "경계2", "경계3", "종료시점"], body)

    print("### recovery가 발동한 행에서 steer-only 대비 판정 변화\n")
    body = []
    for P in DATASETS:
        so = load(root, f"dija-{P}-steeronly")
        for b in ("0", "end"):
            d = load(root, f"dija-{P}-bnd{b}")
            if not (so and d):
                continue
            rec = [r["index"] for r in d["rows"] if r.get("recovery_events")]
            fixed = sum(so["labels"][i] == "unsafe" and d["labels"][i] == "safe" for i in rec)
            broke = sum(so["labels"][i] == "safe" and d["labels"][i] == "unsafe" for i in rec)
            stay = sum(so["labels"][i] == "unsafe" and d["labels"][i] == "unsafe" for i in rec)
            body.append([P, b, str(len(rec)), str(fixed), str(broke), f"{stay} ({stay / max(1, len(rec)):.0%})"])
    table(["데이터셋", "경계", "recovery 행", "unsafe→safe", "safe→unsafe", "계속 unsafe"], body)


def temperature(root, bnd):
    print(f"### temperature (v3 경계 {bnd}, 시드 42/43/44 평균)\n")
    body = []
    for P in DATASETS:
        for T in ("0", "0.2", "0.5", "1.0"):
            ds = [load(root, f"dija-{P}-bnd{bnd}-temp{T}-seed{s}") for s in (42, 43, 44)]
            ds = [d for d in ds if d]
            if not ds:
                body.append([P, T, "-", "-", "-", "-"]); continue
            a = [d["asr"] for d in ds]
            body.append([P, T, fmt(statistics.mean(a)),
                         " / ".join(fmt(x) for x in a),
                         fmt(max(a) - min(a)) if len(a) > 1 else "-",
                         fmt(statistics.mean(d["recovered"] for d in ds), ".1f")])
    table(["데이터셋", "T", "ASR 평균", "시드별 ASR", "시드 폭", "recovery 행(평균)"], body)


def recovery(root, bnd):
    print(f"### recovery steps x rounds (v3 경계 {bnd}, T=0.2)\n")
    body = []
    for P in DATASETS:
        for R in (16, 32, 64):
            row = [P, str(R)]
            for N in (1, 2, 3):
                d = load(root, f"dija-{P}-bnd{bnd}-rec-s{R}-n{N}")
                row.append("-" if not d else f"{d['asr']:.3f} ({d['sec']:.1f}s)")
            body.append(row)
    table(["데이터셋", "steps", "rounds=1", "rounds=2", "rounds=3"], body)


def benign_v3(root):
    print("### benign 프롬프트에서 v3 (공격 없음, T=0, gen_length 128, 첫 블록 경계 audit)\n")
    body = []
    for a in ("0.5", "1.0"):
        for S, name, n in (("xstest", "XSTest-safe", 250), ("tqa", "TruthfulQA[400:500]", 100)):
            def rate(stem):
                f = root / f"{stem}_judged.json"
                return json.loads(f.read_text())["summary"]["refusal_rate"] if f.exists() else None
            g = root / f"or-{S}-v3-a{a}.json"
            rec = (sum(bool(r.get("recovery_events")) for r in json.loads(g.read_text())["results"])
                   if g.exists() else None)
            body.append([a, name, fmt(rate(f"or-{S}-a{a}")), fmt(rate(f"or-{S}-v3-a{a}")),
                         "-" if rec is None else f"{rec}/{n} ({rec / n:.0%})"])
    table(["alpha", "셋", "과잉거부 steer-only", "과잉거부 v3", "v3 recovery 발동"], body)


def main(root, bnd="end"):
    root = Path(root)
    boundaries(root)
    benign_v3(root)
    temperature(root, bnd)
    recovery(root, bnd)


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["outputs/tune"]))
