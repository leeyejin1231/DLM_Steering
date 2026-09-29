"""eval_broken_sentence.py

Rule-based broken-sentence ratio over results.jsonl produced by eval_gsm8k.py.

Response-level rule:
  empty    : the whole response (after cutting at the first EOS) is blank

Sentence-level rules (a sentence is broken if ANY applies):
  repeat   : the same token appears >= 3 times consecutively
  script   : contains non-Latin script characters (CJK, Hangul, Cyrillic, ...)
  special  : residual special tokens (<|...|>, <mask>, [MASK], U+FFFD, ...)

Separator lines (---, ===, ***, ___) are not counted as sentences.

Usage:
  python eval_broken_sentence.py results/*/results.jsonl
  python eval_broken_sentence.py results.jsonl --dump broken.jsonl
"""
import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict

# ---------------------------------------------------------------- patterns
SPECIAL_TOKEN_RE = re.compile(
    r"<\|[^|>]*\|>"            # <|endoftext|>, <|mdm_mask|>, <|im_end|>
    r"|<mask>|\[MASK\]|<pad>|</s>|<s>|<unk>"
    r"|�",                # byte-fallback replacement char
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?|[^\sA-Za-z\d]")
EOS_RE = re.compile(r"<\|endoftext\|>|<\|eot_id\|>|<\|im_end\|>|</s>")
# Split on ./!/?/newline, but not a '.' inside a number (3.5).
SENT_SPLIT_RE = re.compile(r"(?<!\d)[.!?]+(?!\d)\s+|\n+")
# A line made only of separator symbols (with optional spaces), e.g. "---", "* * *".
SEPARATOR_RE = re.compile(r"^[\s\-=*_~#|]+$")
STRIP_CHARS = " \t\r\n.,;:!?\"'()[]{}*-_=#$\\"

ALLOWED_SCRIPTS = {"LATIN", "COMMON", "INHERITED"}


def _script_of(ch: str) -> str:
    try:
        return unicodedata.name(ch).split()[0]
    except ValueError:
        return "UNKNOWN"


def has_foreign_script(s: str) -> bool:
    for ch in s:
        if ch.isascii() or not ch.isalpha():
            continue
        if _script_of(ch) not in ALLOWED_SCRIPTS:
            return True
    return False


def has_token_repeat(s: str, k: int = 3) -> bool:
    toks = TOKEN_RE.findall(s.lower())
    run = 1
    for a, b in zip(toks, toks[1:]):
        run = run + 1 if a == b else 1
        if run >= k:
            return True
    return False


def truncate_at_eos(text: str) -> str:
    m = EOS_RE.search(text)
    return text[: m.start()] if m else text


def split_sentences(text: str):
    out = []
    for s in SENT_SPLIT_RE.split(text):
        if s is None or not s.strip(STRIP_CHARS):
            continue                       # blank fragment from the splitter
        if SEPARATOR_RE.match(s):
            continue                       # ---, ===, ***
        out.append(s)
    return out


def judge_sentence(s: str, repeat_k: int = 3):
    """Return list of violated rule names (empty list == fine)."""
    reasons = []
    if SPECIAL_TOKEN_RE.search(s):
        reasons.append("special")
    if has_foreign_script(s):
        reasons.append("script")
    if has_token_repeat(s, repeat_k):
        reasons.append("repeat")
    return reasons


def judge_text(text: str, repeat_k: int = 3, keep_eos_tail: bool = False):
    """Return dict with empty flag, n_sentences, n_broken, rule counter, examples."""
    body = text if keep_eos_tail else truncate_at_eos(text)
    if not body.strip(STRIP_CHARS):
        return {"empty": True, "n_sent": 0, "n_broken": 0,
                "rules": Counter({"empty": 1}), "examples": []}
    sents = split_sentences(body)
    n_broken, rules, examples = 0, Counter(), []
    for s in sents:
        r = judge_sentence(s, repeat_k)
        if r:
            n_broken += 1
            rules.update(r)
            examples.append({"sentence": s[:200], "rules": r})
    return {"empty": False, "n_sent": len(sents), "n_broken": n_broken,
            "rules": rules, "examples": examples}


# ------------------------------------------------------------------ driver
def load_rows(paths):
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    row.setdefault("_file", str(p))
                    yield row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="results.jsonl file(s)")
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--group-key", default="mode")
    ap.add_argument("--repeat-k", type=int, default=3)
    ap.add_argument("--keep-eos-tail", action="store_true",
                    help="do not cut text at the first EOS token")
    ap.add_argument("--dump", help="write broken examples to this jsonl")
    args = ap.parse_args()

    agg = defaultdict(lambda: {
        "responses": 0, "responses_empty": 0, "responses_broken": 0,
        "sentences": 0, "sentences_broken": 0, "rules": Counter(),
    })
    dump = open(args.dump, "w", encoding="utf-8") if args.dump else None

    for row in load_rows(args.paths):
        g = row.get(args.group_key, "all")
        res = judge_text(row.get(args.text_key) or "", args.repeat_k, args.keep_eos_tail)
        a = agg[g]
        a["responses"] += 1
        a["responses_empty"] += int(res["empty"])
        a["responses_broken"] += int(res["empty"] or res["n_broken"] > 0)
        a["sentences"] += res["n_sent"]
        a["sentences_broken"] += res["n_broken"]
        a["rules"].update(res["rules"])
        if dump and (res["empty"] or res["examples"]):
            dump.write(json.dumps({
                "file": row["_file"], "mode": g,
                "test_index": row.get("test_index"),
                "correct": row.get("correct"),
                "empty": res["empty"],
                "broken": res["examples"],
            }, ensure_ascii=False) + "\n")
    if dump:
        dump.close()

    header = (f"{'mode':<40} {'resp':>6} {'empty':>6} {'resp_brk%':>9} "
              f"{'sent':>7} {'sent_brk%':>9}  rules")
    print(header)
    print("-" * len(header))
    for g in sorted(agg):
        a = agg[g]
        rp = 100 * a["responses_broken"] / max(a["responses"], 1)
        sp = 100 * a["sentences_broken"] / max(a["sentences"], 1)
        rules = ",".join(f"{k}={v}" for k, v in sorted(a["rules"].items()))
        print(f"{g:<40} {a['responses']:>6} {a['responses_empty']:>6} {rp:>8.2f}% "
              f"{a['sentences']:>7} {sp:>8.2f}%  {rules}")

    summary = {
        g: {**a, "rules": dict(a["rules"]),
            "response_broken_ratio": a["responses_broken"] / max(a["responses"], 1),
            "sentence_broken_ratio": a["sentences_broken"] / max(a["sentences"], 1)}
        for g, a in agg.items()
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
