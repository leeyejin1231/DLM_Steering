"""Convert a DiffuGuard runner output (list of records with "response") into the
exp.py result payload that eval_llamaguard.py / run_sr_eval.py consume.
Rows are indexed by matching the vanilla prompt to common.load_prompts(source).

Usage: python script/convert_diffuguard.py --in out.json --out outputs/X.json --source jbb_harmful --config '{"...": ...}'
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import MODEL_NAME, load_prompts, write_json

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--source", default="jbb_harmful"); ap.add_argument("--config", default="{}")
ap.add_argument("--defense", default="{}")
a = ap.parse_args()
index = {r["prompt"].strip(): r["index"] for r in load_prompts(a.source)}
recs = json.loads(Path(a.inp).read_text())
results, seen = [], set()
for r in recs:
    van = (r.get("vanilla prompt") or r.get("goal") or r.get("Behavior")).strip()
    if van in seen:      # the HarmBench refined file repeats 7 behaviours; load_prompts de-duplicates
        continue
    seen.add(van)
    results.append({"index": index[van], "prompt": van, "attack_prompt": r.get("refined prompt"),
                    "generation": r["response"], "sp_hid_tail": r.get("sp_hid_tail"),
                    "template_attack": r.get("template_attack")})
results.sort(key=lambda x: x["index"])
write_json(a.out, {"model": MODEL_NAME, "config": json.loads(a.config), "attack": {"attack": "dija", "runner": "DiffuGuard"},
                   "defense": json.loads(a.defense), "results": results})
print(f"{len(results)} rows ({len(recs) - len(results)} duplicate prompts dropped) -> {a.out}")
