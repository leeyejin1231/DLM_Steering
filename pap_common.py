import hashlib
import json
import random
import shlex
from pathlib import Path

from attack_prompts import PAP_BETTER_TEMPLATES_PATH
from dlm_steering.runtime.data import load_prompts
from dlm_steering.runtime.utils import atomic_write_json

TOP5 = ("Logical Appeal", "Authority Endorsement", "Misrepresentation",
        "Evidence-based Persuasion", "Expert Endorsement")
SAMPLING = {"temperature": 1, "top_p": 1, "max_new_tokens": 256}


def default_cache_path(source, seed):
    return Path(__file__).parent / "data" / "attacks" / "pap_better" / source / f"seed{seed}.json"


def require_cache(path, source, seed, reproduct):
    path = Path(path) if path else default_cache_path(source, seed)
    if not path.is_file():
        root = Path(__file__).parent
        command = [str(root / ".venv/bin/python"), str(root / "pap_generate.py"), "--source", source, "--seed", str(seed)]
        if reproduct:
            command.append("--reproduct")
        command += ["--out", str(path)]
        raise FileNotFoundError(
            f"There is no PAP attack prompt file: {path}\n"
            f"Please generate it first with the following command:\n{shlex.join(command)}")
    return path


def normalize_templates(templates):
    lowered = {name.lower(): value for name, value in templates.items()}
    if set(lowered) != {name.lower() for name in TOP5}:
        raise ValueError("PAP requires exactly the five better in-context templates")
    normalized = {name: lowered[name.lower()] for name in TOP5}
    if any(template.count("%s") != 1 for template in normalized.values()):
        raise ValueError("Each PAP template must have exactly one query placeholder")
    return normalized


def assign_techniques(rows, seed):
    indices = sorted(int(row["index"]) for row in rows)
    if len(indices) != len(set(indices)):
        raise ValueError("PAP technique assignment requires unique row indices")
    rng = random.Random(seed)
    rng.shuffle(indices)
    techniques = list(TOP5)
    rng.shuffle(techniques)
    return {index: techniques[position % len(techniques)]
            for position, index in enumerate(indices)}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, data)


def identity(source, seed, reproduct, model):
    return dict(source=source, seed=seed, reproduct=reproduct, model=model, template_sha256=digest(PAP_BETTER_TEMPLATES_PATH), sampling=dict(SAMPLING))


def validate(data, rows, source, seed, reproduct, complete=True):
    expected = identity(source, seed, reproduct, data['model'])
    if any(data.get(k) != v for k, v in expected.items()):
        raise ValueError('PAP cache metadata mismatch')
    assignments = assign_techniques(rows, seed)
    originals = {int(row['index']): row['prompt'] for row in rows}
    entries = data['results']
    indices = [int(e['index']) for e in entries]
    if len(indices) != len(set(indices)):
        raise ValueError('Duplicate PAP cache indices')
    if complete and set(indices) != set(originals):
        raise ValueError('PAP cache must contain the entire source dataset')
    for e in entries:
        i = int(e['index'])
        if (i not in originals or e['prompt'] != originals[i]
                or e['technique'] != assignments[i]
                or not isinstance(e['attack_prompt'], str) or not e['attack_prompt'].strip()):
            raise ValueError(f'Invalid PAP cache entry {i}')
    return {int(e['index']): e for e in entries}


def load_cache(path, source, seed, reproduct):
    data = json.loads(Path(path).read_text())
    data['by_index'] = validate(data, load_prompts(source), source, seed, reproduct)
    data['sha256'] = digest(path)
    return data
