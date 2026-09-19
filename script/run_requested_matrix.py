"""Finish the 72-cell LLaDA/PAP/DIJA/V3/DiffuGuard matrix and both evaluations."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import json
from pathlib import Path
import queue
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv/bin/python"
OUTPUTS = ROOT / "outputs"


def missing_assets():
    paths = [OUTPUTS / "steer_vector.pt", OUTPUTS / "steer_detector.pt",
             OUTPUTS / "response_detector.pt", OUTPUTS / "gate_threshold.json"]
    paths += [OUTPUTS / "llada1.5" / name for name in
              ("steer_vector.pt", "steer_detector.pt", "response_detector.pt",
               "gate_threshold.json", "prompt_outcome_detector_report.json")]
    paths += [ROOT / "data/attacks/pap_better" / source / f"seed{seed}.json"
              for source in ("jbb_harmful", "harmbench", "strongreject")
              for seed in (42, 43, 44)]
    paths.append(OUTPUTS / "ollama/models/manifests/registry.ollama.ai/library/gpt-oss/20b")
    missing = [str(p.relative_to(ROOT)) for p in paths if not p.is_file()]
    outcome_report = OUTPUTS / "llada1.5/prompt_outcome_detector_report.json"
    if outcome_report.is_file():
        try:
            if json.loads(outcome_report.read_text())["validation_auroc"] < .6:
                missing.append("LLaDA 1.5 prompt detector validation AUROC < 0.6")
        except (OSError, ValueError, KeyError, TypeError):
            missing.append("LLaDA 1.5 prompt detector report invalid")
    if not missing:
        from pap_common import load_cache
        expected = {"jbb_harmful": 100, "harmbench": 393, "strongreject": 313}
        for source, n in expected.items():
            for seed in (42, 43, 44):
                path = ROOT / "data/attacks/pap_better" / source / f"seed{seed}.json"
                try:
                    if len(load_cache(path, source, seed, True)["results"]) != n:
                        missing.append(str(path.relative_to(ROOT)) + " (incomplete)")
                except (OSError, ValueError, KeyError, TypeError):
                    missing.append(str(path.relative_to(ROOT)) + " (invalid)")
    return missing


def wait_for_assets(hours=8):
    deadline = time.monotonic() + hours * 3600
    last_notice = 0
    while True:
        missing = missing_assets()
        if not missing:
            print("All model, defense, attack, and grader assets are ready", flush=True)
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("Assets still missing: " + ", ".join(missing))
        if time.monotonic() - last_notice > 300:
            print("Waiting for: " + ", ".join(missing), flush=True)
            last_notice = time.monotonic()
        time.sleep(60)


def run_cell(spec, available):
    gpu = available.get()
    try:
        model, source, attack, defense, seed = spec
        label = f"{model}_{source}_{attack}_{defense}_seed{seed}"
        log_path = OUTPUTS / "matrix_logs" / f"{label}.log"
        command = [str(PYTHON), "-u", "-m", "script.matrix_one",
                   "--model", model, "--source", source, "--attack", attack,
                   "--defense", defense, "--seed", str(seed), "--gpu", str(gpu)]
        print(f"GPU {gpu}: starting {label}", flush=True)
        with log_path.open("w") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log,
                                    stderr=subprocess.STDOUT)
        print(f"GPU {gpu}: {label} exit {result.returncode}", flush=True)
        return {"label": label, "gpu": gpu, "returncode": result.returncode,
                "log": str(log_path.relative_to(ROOT))}
    finally:
        available.put(gpu)


def main():
    OUTPUTS.mkdir(exist_ok=True)
    (OUTPUTS / "matrix_logs").mkdir(exist_ok=True)
    wait_for_assets()
    sources = ("harmbench", "strongreject", "jbb_harmful")
    specs = list(itertools.product(
        ("llada", "llada1.5"), sources, ("dija", "pap"),
        ("diffuguard", "ours"), (42, 43, 44)))
    available = queue.Queue()
    for gpu in range(8):
        available.put(gpu)
    records = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(run_cell, spec, available) for spec in specs]
        for future in as_completed(futures):
            records.append(future.result())
            (OUTPUTS / "matrix_summary.json").write_text(json.dumps(
                {"total": len(specs), "done": len(records),
                 "failed": sum(r["returncode"] != 0 for r in records),
                 "records": records}, indent=2))
    failed = [r for r in records if r["returncode"]]
    print(f"Matrix finished: {len(records) - len(failed)}/{len(records)} cells", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"Matrix runner failed: {exc}", file=sys.stderr)
        sys.exit(1)
