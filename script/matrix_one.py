"""Run one requested matrix cell through interface.py, then both judges."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import interface
from common import load_prompts


def prior_result(model, source, attack, defense, seed, n):
    for plan_path in (ROOT / "outputs/interactive").glob("*/plan.json"):
        try:
            plan = json.loads(plan_path.read_text())
            for item in plan["commands"]:
                if item.get("status") != "complete" or Path(item["argv"][1]).name != "exp.py":
                    continue
                p = item["parameters"]
                if all(p.get(k) == v for k, v in dict(
                    model=model, source=source, attack=attack, defense=defense,
                    seed=seed, n=n, start=0, gen_length=128, steps=128,
                    block_length=32, temperature=.2, reproduct=True).items()):
                    argv = item["argv"]
                    path = Path(argv[argv.index("--out") + 1])
                    if path.is_file() and len(json.loads(path.read_text())["results"]) == n:
                        return path
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            continue
    return None


def new_folder(suffix):
    base = ROOT / "outputs/interactive" / suffix
    path = base
    attempt = 1
    while path.exists():
        path = Path(f"{base}_retry{attempt}")
        attempt += 1
    return path


def experiment(model, source, attack, defense, seed, gpu):
    n = len(load_prompts(source))
    old = prior_result(model, source, attack, defense, seed, n)
    if old:
        print(f"Reusing generation: {old}", flush=True)
        return old
    folder = new_folder(f"matrix_{model}_{source}_{attack}_{defense}_seed{seed}")
    out = folder / f"{model}_{source}_{attack}_{defense}_seed{seed}.json"
    args = ["--model", model, "--attack", attack, "--defense", defense,
            "--source", source, "--start", "0", "--n", str(n),
            "--gen-length", "128", "--steps", "128", "--block-length", "32",
            "--temperature", "0.2", "--gpus", str(gpu), "--reproduct"]
    if defense == "ours":
        args += ["--alpha", "1", "--remask", "v3", "--remask-prompt"]
    else:
        args += ["--remasking", "adaptive_step", "--repair-scope", "all"]
    args += ["--seed", str(seed), "--out", str(out)]
    cfg = interface.parse_experiment_args(args)
    interface.execute(folder, [interface.command("exp.py", args, vars(cfg))], [])
    return out


def evaluation(inp, gpu):
    folder = new_folder("matrix_eval_" + inp.parent.name)
    def choose(label, options, default=1):
        if label == "평가 모델":
            return "both"
        if label == "DIJA 평가 범위":
            return "dija_template"
        raise ValueError(f"Unexpected choice: {label}")

    def ask(label, default=None, convert=str):
        value = (str(gpu) if label.startswith("평가 GPU") else
                 "4" if label == "LG4 배치 크기" else
                 "4" if label == "GPT-OSS 동시 요청 수" else
                 "4096" if label == "GPT-OSS 최대 채점 출력 토큰" else None)
        if value is None:
            raise ValueError(f"Unexpected question: {label}")
        return convert(value)

    commands, prepared = interface.evaluation_plan(
        folder, ask=ask, choose=choose,
        select_result=lambda repo, ask, existing_file: inp)
    if all(item.get("cached_result") for item in commands):
        print(f"Both evaluations already cached: {inp}", flush=True)
        return
    interface.execute(folder, commands, prepared)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=["llada", "llada1.5"])
    ap.add_argument("--source", required=True,
                    choices=["jbb_harmful", "harmbench", "strongreject"])
    ap.add_argument("--attack", required=True, choices=["pap", "dija"])
    ap.add_argument("--defense", required=True, choices=["ours", "diffuguard"])
    ap.add_argument("--seed", required=True, type=int, choices=[42, 43, 44])
    ap.add_argument("--gpu", required=True, type=int, choices=range(8))
    args = ap.parse_args()
    inp = experiment(args.model, args.source, args.attack,
                     args.defense, args.seed, args.gpu)
    evaluation(inp, args.gpu)


if __name__ == "__main__":
    main()
