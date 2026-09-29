import hashlib
import json

from dlm_steering.paths import DATA_DIR

CACHE_DIR = DATA_DIR / "response_fit_cache"
VERSION = 1


def config_key(*, model_name, steps, gen_length, block_length, temperature, attacker, judge):
    """Short digest of every setting that changes a cached row."""
    payload = json.dumps({"v": VERSION, "model": model_name, "steps": steps,
                          "gen_length": gen_length, "block_length": block_length,
                          "temperature": temperature, "attacker": attacker,
                          "judge": judge}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def cache_path(arm, key):
    return CACHE_DIR / f"{arm}_{key}.json"


def prompt_key(prompt):
    return hashlib.sha256(str(prompt).encode()).hexdigest()[:16]


def load(arm, key):
    """{prompt digest: entry} for this arm and config; empty when absent."""
    path = cache_path(arm, key)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    if data.get("config_key") != key:
        return {}
    return {row["prompt_key"]: row for row in data.get("rows", [])}


def save(arm, key, entries, *, meta=None):
    """Write the arm's cache, newest metadata included, sorted for stable diffs."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path(arm, key)
    rows = sorted(entries.values(), key=lambda r: r["prompt_key"])
    path.write_text(json.dumps({"config_key": key, "arm": arm,
                                "version": VERSION, "meta": meta or {},
                                "rows": rows}, ensure_ascii=False, indent=1))
    return path


def entry(prompt, group, token_ids, slots, text, label):
    return {"prompt_key": prompt_key(prompt), "prompt": str(prompt),
            "group": int(group), "token_ids": [int(t) for t in token_ids],
            "slots": [int(s) for s in slots], "text": text, "label": label}
