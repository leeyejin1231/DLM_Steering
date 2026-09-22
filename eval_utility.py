"""Score exp.py generations on the graded sources (mmlu, gsm8k, truthfulqa_mc).

Each result row carries "task" and "answer" from common.load_utility_prompts;
this extracts the model's answer from "generation" and reports accuracy, plus
a per-subject (mmlu) or per-category (truthfulqa_mc) breakdown. Unparseable
generations count as wrong and are tallied separately, since a defense that
turns answers into refusals shows up there first.

Usage:
    python eval_utility.py --in outputs/MMLU-none-v3-42.json --out outputs/MMLU-none-v3-42_acc.json
"""

import argparse
import collections
import json
import re
from pathlib import Path

from common import LETTERS

NUMBER = r"-?\d[\d,]*(?:\.\d+)?"


def extract_letter(text, n_choices):
    """Predicted option letter, or None. Tries explicit answer phrasings first,
    then a bare option letter at the start of a line, then the first option
    letter that appears on its own anywhere."""
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
    """Number after '####' if present, else the last number in the text."""
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


def extract_boxed(text):
    """Content of the last \\boxed{...} (brace-balanced), else the text after the
    last 'Final answer:' line, else None."""
    starts = [m.end() for m in re.finditer(r"\\boxed\s*\{", text)]
    for start in reversed(starts):
        depth, i = 1, start
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        if depth == 0:
            return text[start:i - 1]
    m = None
    for m in re.finditer(r"[Ff]inal answer\s*[:：]\s*(.+)", text):
        pass
    return m.group(1).strip().rstrip(".").strip("$ ") if m else None


def normalize_math(ans):
    """Light LaTeX normalisation so equivalent spellings compare equal."""
    if ans is None:
        return None
    a = ans.strip().strip("$").strip()
    a = re.sub(r"\\(?:left|right|,|!|;|:|displaystyle)", "", a)
    a = a.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    a = re.sub(r"\\text\{([^}]*)\}", r"\1", a)
    a = re.sub(r"\^\{?\\circ\}?", "", a).replace("^\\circ", "")
    a = a.replace("\\%", "%").replace("%", "")
    a = re.sub(r"\s+", "", a).rstrip(".")
    a = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", a)   # \frac12 -> \frac{1}{2}
    a = re.sub(r"^([a-zA-Z])=", "", a)          # "x=5" -> "5"
    if re.fullmatch(r"-?\d[\d,]*(?:\.\d+)?", a):
        a = a.replace(",", "")
    return a


def same_math(pred, gold):
    p, g = normalize_math(pred), normalize_math(gold)
    if p is None or g is None:
        return False
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except ValueError:
        return False


def score_row(row):
    task, gold, gen = row["task"], row["answer"], row.get("generation", "")
    if task == "gsm8k":
        pred = extract_number(gen)
        correct = pred is not None and same_number(pred, gold)
    elif task == "math500":
        pred = extract_boxed(gen)
        correct = pred is not None and same_math(pred, gold)
    else:
        n_choices = len(re.findall(r"^[A-P]\. ", row["prompt"], flags=re.MULTILINE))
        pred = extract_letter(gen, n_choices)
        correct = pred == gold
    return {"pred": pred, "correct": bool(correct), "parsed": pred is not None}


def summarize(rows):
    total = len(rows)
    correct = sum(r["correct"] for r in rows)
    unparsed = sum(not r["parsed"] for r in rows)
    summary = {"task": rows[0]["task"] if rows else None, "total": total,
               "correct": correct, "accuracy": round(correct / total, 4) if total else 0.0,
               "unparsed": unparsed}
    group_key = {"mmlu": "subject", "truthfulqa_mc": "category", "math500": "subject"}.get(summary["task"])
    if group_key:
        groups = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            g = groups[r.get(group_key, "?")]
            g[0] += r["correct"]
            g[1] += 1
        summary[f"by_{group_key}"] = {k: {"correct": c, "total": t, "accuracy": round(c / t, 4)}
                                      for k, (c, t) in sorted(groups.items())}
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
        raise SystemExit(f"{len(missing)} rows lack task/answer fields; was --source a "
                         f"utility set (mmlu, gsm8k, truthfulqa_mc, math500)? e.g. index {missing[:5]}")
    scored = [{**r, **score_row(r)} for r in rows]
    summary = summarize(scored)
    payload = {"source": args.inp, "config": data.get("config"), "attack": data.get("attack"),
               "defense": data.get("defense"), "summary": summary, "results": scored}
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if not k.startswith("by_")}, indent=2))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
