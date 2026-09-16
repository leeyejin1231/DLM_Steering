"""Print the headline metrics of scored output files as one aligned table.

script/evaluate.sh calls this last; it also runs standalone over any grader
payloads:

    python script/report.py outputs/JBB-dija-v3-42_lg4.json outputs/*_sr.json

With no arguments it reports the files evaluate.sh produces. Every input is a
payload written by eval_llamaguard.py, run_sr_eval.py, steering/judge_refusal.py
or eval_utility.py; the grader is inferred from the keys of its "summary"
block, so no naming convention is assumed. Missing or unrecognised files are
listed rather than raised, so a partially finished pipeline still reports what
it has.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# What script/evaluate.sh writes, in reading order.
DEFAULT_FILES = [
    *(f"outputs/gated_lg4_len{n}.json" for n in (128, 256)),
    *(f"outputs/gated_sr_len{n}.json" for n in (128, 256)),
    *(f"outputs/gated_or30_{s}_judged.json" for s in ("xstest", "jbb", "tqa")),
]

# (grader, key that identifies it, [(column header, summary key)]). The first
# grader whose marker key is present wins, so markers must be unique.
GRADERS = [
    ("Llama Guard 4 (harmful)", "by_category",
     [("n", "total"), ("unsafe", "unsafe"), ("ASR", "asr"),
      ("errors", "n_errors")]),
    ("StrongREJECT (harmful)", "mean_sr_score",
     [("n", "total"), ("refused", "refused"), ("ASR", "asr"),
      ("mean score", "mean_sr_score"), ("errors", "n_errors")]),
    ("XSTest refusal judge", "2_full_refusal",
     [("n", "total"), ("refusal", "refusal_rate"),
      ("full refusal", "full_refusal_rate"), ("parse err", "n_parse_error"),
      ("gen err", "n_generation_error")]),
    ("Accuracy", "accuracy",
     [("n", "total"), ("accuracy", "accuracy"), ("unparsed", "unparsed"),
      ("errors", "n_errors")]),
]


def classify(summary):
    """(grader name, columns) for a summary block, or None if unrecognised."""
    for name, marker, columns in GRADERS:
        if marker in summary:
            return name, columns
    return None


def cell(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def render(name, columns, rows):
    """One grader's table: a label column plus that grader's metric columns."""
    headers = ["file", *(h for h, _ in columns)]
    table = [headers] + [[label, *(cell(s.get(k)) for _, k in columns)]
                         for label, s in rows]
    widths = [max(len(r[i]) for r in table) for i in range(len(headers))]
    print(f"\n{name}")
    for i, row in enumerate(table):
        print("  " + "  ".join(v.ljust(w) for v, w in zip(row, widths)))
        if i == 0:
            print("  " + "  ".join("-" * w for w in widths))


def main(argv):
    paths = [Path(a) for a in argv] or [ROOT / f for f in DEFAULT_FILES]
    grouped, missing, unknown = {}, [], []
    for path in paths:
        if not path.exists():
            missing.append(path)
            continue
        summary = json.loads(path.read_text()).get("summary")
        kind = classify(summary) if summary else None
        if kind is None:
            unknown.append(path)
            continue
        name, columns = kind
        grouped.setdefault((name, tuple(columns)), []).append(
            (path.name, summary))

    for (name, columns), rows in grouped.items():
        render(name, list(columns), rows)
    if unknown:
        print("\nno recognisable summary block:")
        for path in unknown:
            print(f"  {path}")
    if missing:
        print("\nnot scored yet:")
        for path in missing:
            print(f"  {path.relative_to(ROOT) if path.is_absolute() else path}")
    if not grouped:
        print("\nnothing to report.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
