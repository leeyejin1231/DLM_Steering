"""Score LLaDA generations with StrongREJECT, using ollama_setting/evaluate.py as-is.

evaluate.py cannot be imported normally: its module-level _EVALUATORS dict
references eval_rr / eval_lg3 / eval_lg4, which that file does not define. The
source is therefore exec'd in a namespace where those names are pre-bound to
stubs, so eval_sr_ollama and compute_summary come from the project's own code
rather than a reimplementation here.

Requires the Ollama server from ollama_setting/podman to be up on --host.

Usage:
    python run_sr_eval.py --in outputs/llada8b_generations_len256.json \
                          --out outputs/llada8b_sr_len256.json
"""

import argparse
import json
from pathlib import Path

EVALUATE_PY = Path(__file__).parent / "ollama_setting" / "evaluate.py"


def load_project_evaluator():
    def _missing(*args, **kwargs):
        raise NotImplementedError("evaluator not defined in ollama_setting/evaluate.py")

    ns = {
        "__file__": str(EVALUATE_PY.resolve()),
        "__name__": "dlm_evaluate",
        "eval_rr": _missing,
        "eval_lg3": _missing,
        "eval_lg4": _missing,
    }
    exec(compile(EVALUATE_PY.read_text(), str(EVALUATE_PY), "exec"), ns)
    return ns


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="outputs/llada8b_generations_len256.json")
    p.add_argument("--out", default="outputs/llada8b_sr_len256.json")
    p.add_argument("--jsonl", default=None, help="Streaming/resume file for eval_sr_ollama.")
    p.add_argument("--model", default="gpt-oss:20b")
    p.add_argument("--host", default="http://localhost:50001")
    p.add_argument("--reasoning-effort", default="low")
    return p.parse_args()


def main():
    args = parse_args()
    ns = load_project_evaluator()
    eval_sr_ollama = ns["eval_sr_ollama"]
    compute_summary = ns["compute_summary"]

    with open(args.inp, encoding="utf-8") as f:
        data = json.load(f)

    # eval_sr_ollama grades the "response" field against prompt_key.
    items = [
        {"index": r["index"], "prompt": r["prompt"], "response": r["generation"]}
        for r in data["results"]
    ]
    print(f"Grading {len(items)} items from {args.inp}")

    jsonl = args.jsonl or str(Path(args.out).with_suffix(".jsonl"))
    graded = eval_sr_ollama(
        items,
        prompt_key="prompt",
        ollama_model=args.model,
        ollama_host=args.host,
        reasoning_effort=args.reasoning_effort,
        output_path=Path(jsonl),
    )
    graded.sort(key=lambda r: r["index"])

    summary = compute_summary(graded, metric="SR")
    scores = [g["sr_score"] for g in graded if g.get("sr_score") == g.get("sr_score")]
    summary["mean_sr_score"] = round(sum(scores) / len(scores), 4) if scores else None
    summary["score_scale"] = "0 = refused; otherwise (convincing + specific) / 2, range 1-5"

    payload = {
        "grader": f"StrongREJECT / {args.model} (reasoning_effort={args.reasoning_effort})",
        "source": args.inp,
        "source_model": data.get("model"),
        "source_config": data.get("config"),
        "summary": summary,
        "results": graded,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n-> {args.out}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for g in graded:
        print(f"  idx={g['index']:>2} score={g.get('sr_score')} "
              f"refusal={g.get('sr_refusal')} conv={g.get('sr_convincing')} "
              f"spec={g.get('sr_specific')}")


if __name__ == "__main__":
    main()
