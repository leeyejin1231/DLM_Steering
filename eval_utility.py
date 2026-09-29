import argparse
import collections
import json
import re
from pathlib import Path

from dlm_steering.runtime.constants import ERROR_SENTINEL
from dlm_steering.runtime.data import LETTERS

NUMBER = r"-?\d[\d,]*(?:\.\d+)?"


def extract_letter(text, n_choices):
    valid = LETTERS[:n_choices]
    patterns = [
        rf"answer(?:\s+is|:)?\s*(?:\(|\*\*)?\s*([{valid}])(?![a-zA-Z])",
        rf"^\s*(?:\(|\*\*)?([{valid}])(?:\)|\.|:|\*\*)(?![a-zA-Z])",
        rf"(?<![A-Za-z])([{valid}])(?:\)|\.)(?![a-zA-Z])",
        rf"(?<![A-Za-z])([{valid}])(?![A-Za-z])",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    return None


def extract_number(text):
    m = re.search(rf"####\s*\$?\s*({NUMBER})", text)
    if not m:
        nums = re.findall(NUMBER, text)
        if not nums:
            return None
        return nums[-1].replace(",", "")
    return m.group(1).replace(",", "")


def same_number(pred, gold):
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except (TypeError, ValueError):
        return False


def last_boxed(text):
    start = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if start < 0:
        return None
    i = text.find("{", start)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            return text[i + 1:j]
    return None


def _fix_fracs(s):
    # \frac12 -> \frac{1}{2}, \frac1{72} -> \frac{1}{72}
    parts = s.split("\\frac")
    out = parts[0]
    for part in parts[1:]:
        out += "\\frac"
        if part[:1] == "{" or len(part) < 2:
            out += part
            continue
        a, b, rest = part[0], part[1], part[2:]
        out += f"{{{a}}}{b}{rest}" if b == "{" else f"{{{a}}}{{{b}}}{rest}"
    return out


def normalize_math(s):
    s = s.strip().replace("\n", "").replace("\\!", "")
    s = s.replace("\\\\", "\\").replace("tfrac", "frac").replace("dfrac", "frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("\\$", "").replace("$", "")
    if "\\text{ " in s and s.split("\\text{ ")[0].strip():     # "5 \text{ cm}" -> "5"
        s = s.split("\\text{ ")[0]
    s = re.sub(r"\\text\{\s*([^{}]*)\}", r"\1", s)       # \text{Evelyn} -> Evelyn
    s = re.sub(r"\\mbox\{\s*([^{}]*)\}", r"\1", s)
    s = s.replace("\\%", "").replace("%", "")
    s = s.replace(" .", " 0.").replace("{.", "{0.")
    if s.startswith("."):
        s = "0" + s
    if len(s.split("=")) == 2 and len(s.split("=")[0]) <= 2:   # "x = 5" -> "5"
        s = s.split("=")[1]
    s = re.sub(r"\\sqrt(\w)", r"\\sqrt{\1}", s)
    s = s.replace(" ", "")
    s = _fix_fracs(s)
    if re.fullmatch(r"-?\d+/\d+", s):                        # 1/2 -> \frac{1}{2}
        a, b = s.split("/")
        s = f"\\frac{{{a}}}{{{b}}}"
    if s == "0.5":
        s = "\\frac{1}{2}"
    if re.fullmatch(r"-?[\d,]+(\.\d+)?", s):
        s = s.replace(",", "")
    return s.rstrip(".")


def same_math(pred, gold):
    a, b = normalize_math(pred), normalize_math(gold)
    return a == b or same_number(a, b)


def score_row(row):
    task, gold, gen = row["task"], row["answer"], row.get("generation", "")
    if gen == ERROR_SENTINEL:
        return {"pred": None, "correct": False, "parsed": False,
                "gen_error": True}
    if task == "gsm8k":
        pred = extract_number(gen)
        correct = pred is not None and same_number(pred, gold)
    elif task == "math500":
        pred = last_boxed(gen)
        correct = pred is not None and same_math(pred, gold)
    else:
        n_choices = len(re.findall(r"^[A-P]\. ", row["prompt"], flags=re.MULTILINE))
        pred = extract_letter(gen, n_choices)
        correct = pred == gold
    return {"pred": pred, "correct": bool(correct), "parsed": pred is not None,
            "gen_error": False}


def summarize(all_rows):
    n_errors = sum(r["gen_error"] for r in all_rows)
    rows = [r for r in all_rows if not r["gen_error"]]
    total = len(rows)
    correct = sum(r["correct"] for r in rows)
    unparsed = sum(not r["parsed"] for r in rows)
    summary = {"task": all_rows[0]["task"] if all_rows else None, "total": total,
                "correct": correct, "accuracy": round(correct / total, 4) if total else 0.0,
                "unparsed": unparsed, "n_errors": n_errors}
    group_key = {"mmlu": "subject", "math500": "subject", "truthfulqa_mc": "category"}.get(summary["task"])
    if group_key:
        groups = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            g = groups[r.get(group_key, "?")]
            g[0] += r["correct"]
            g[1] += 1
        summary[f"by_{group_key}"] = {k: {"correct": c, "total": t, "accuracy": round(c / t, 4)} for k, (c, t) in sorted(groups.items())}
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = json.loads(Path(args.inp).read_text())
    rows = data["results"]
    missing = [r["index"] for r in rows if "task" not in r or "answer" not in r]
    if missing:
        raise SystemExit(f"{len(missing)} rows lack task/answer fields; was --source a utility set (mmlu, gsm8k, math500, truthfulqa_mc)? e.g. index {missing[:5]}")
    scored = [{**r, **score_row(r)} for r in rows]
    summary = summarize(scored)
    payload = {"source": args.inp, "config": data.get("config"), "attack": data.get("attack"), "defense": data.get("defense"), "summary": summary, "results": scored}
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if not k.startswith("by_")}, indent=2))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
