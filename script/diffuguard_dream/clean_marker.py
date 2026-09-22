"""Resolve the <<<ASSISTANT>>> marker left by run_dgm_dream_native_gen128.py.
  --mode dija    template + answer -> marker becomes a blank line (exp.py DIJA grading text)
  --mode prefix  keep only the assistant answer after the marker (exp.py grades the assistant turn)"""
import argparse, json
ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="inp"); ap.add_argument("--out"); ap.add_argument("--mode", choices=["dija", "prefix"])
a = ap.parse_args(); M = "\n<<<ASSISTANT>>>\n"
r = json.load(open(a.inp)); hit = 0
for x in r:
    if M in x["response"]:
        hit += 1
        x["response"] = x["response"].replace(M, "\n\n") if a.mode == "dija" else x["response"].split(M, 1)[1]
json.dump(r, open(a.out, "w"), ensure_ascii=False, indent=1)
print(f"marker found in {hit}/{len(r)} responses -> {a.out}")
