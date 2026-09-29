"""Select Guard-validated target-model contrasts for fit_vector.py."""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expected", type=int, default=386)
    args = ap.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines()
            if line.strip()]
    by_index = {int(row["index"]): row for row in rows}
    if len(by_index) != len(rows) or len(rows) != args.expected:
        raise ValueError(f"Need {args.expected} unique generated prompt rows; got {len(rows)}")
    pairs = []
    for index, row in sorted(by_index.items()):
        if not (row["safe_label"].startswith("safe") and
                row["direct_label"].startswith("unsafe")):
            continue
        if not row["safe_text"].strip() or not row["direct_text"].strip():
            continue
        pairs.append({"pair_id": index, "csv_index": index,
                      "adv_harmful": row["prompt"],
                      "refusal_response": row["safe_text"],
                      "compliant_response": row["direct_text"],
                      "split": "held_out_eval" if index < 20 else "fit"})
    n_fit = sum(p["split"] == "fit" for p in pairs)
    if n_fit < 40:
        raise ValueError(f"Only {n_fit} fit contrasts; need at least 40")
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"source": args.input,
                                  "selection": "Guard safe refusal vs unsafe direct target response",
                                  "rows": len(rows), "n_fit": n_fit,
                                  "pairs": pairs}, ensure_ascii=False, indent=2))
    print(f"{n_fit} fit contrasts, {len(pairs) - n_fit} held-out contrasts -> {output}")


if __name__ == "__main__":
    main()
