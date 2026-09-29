import argparse
import json
import sys
from pathlib import Path

from attack_llms import HFChat
from attack_prompts import load_better_templates, extract_content
from dlm_steering.runtime.data import load_prompts
from dlm_steering.runtime.execution import parse_gpu_ids, shard_slices, spawn_shards, strip_argv_flag, wait_merge_shards
from dlm_steering.runtime.models import enable_reproducibility, seed_all, force_math_attention
from dlm_steering.runtime.progress import task_progress
from pap_common import SAMPLING, assign_techniques, normalize_templates, identity, validate, save


class PAPGenerator:
    def __init__(self, llm, templates):
        self.llm = llm
        self.templates = normalize_templates(templates)

    def generate(self, goal, technique, seed):
        raw = self.llm.generate([{"role": "user", "content": self.templates[technique].replace("%s", goal)}], **SAMPLING, seed=seed)
        mutation = extract_content("#mutated_text_with_same_specific_harmful_or_unlawful_intention: ", raw)
        if not mutation:
            raise ValueError("PAP output missing a nonempty mutation tag; no retry")
        return mutation


def generate_shard(a):
    rows = load_prompts(a.source)
    data = identity(a.source, a.seed, a.reproduct, a.model)
    entries = {}
    for path in [a.reuse, a.out]:
        if path and Path(path).exists():
            old = json.loads(Path(path).read_text())
            if old['model'] != a.model:
                raise ValueError('PAP cache model mismatch')
            entries.update(validate(old, rows, a.source, a.seed, a.reproduct, complete=False))
            data['backend'] = old['backend']
    selected = rows[a.start:a.start + a.n] if a.n is not None else rows[a.start:]
    entries = {int(row['index']): entries[int(row['index'])] for row in selected if int(row['index']) in entries}
    missing = [row for row in selected if int(row['index']) not in entries]
    if missing:
        (enable_reproducibility if a.reproduct else seed_all)(a.seed)
        llm = HFChat(a.model, device='cuda:0')
        llm.load()
        if a.reproduct:
            force_math_attention()
        generator = PAPGenerator(llm, load_better_templates())
        assignments = assign_techniques(rows, a.seed)
        data['backend'] = llm.describe()
        for row in task_progress(missing, total=len(selected), initial=len(entries), desc=f'PAP seed={a.seed}', unit='row'):
            i = int(row['index'])
            technique = assignments[i]
            text = generator.generate(row['prompt'], technique, a.seed + i * 1048576)
            entries[i] = dict(index=i, prompt=row['prompt'], technique=technique, attack_prompt=text)
            data['results'] = [entries[i] for i in sorted(entries)]
            save(a.out, data)
    data['results'] = [entries[i] for i in sorted(entries)]
    save(a.out, data)


def generate_sharded(a):
    rows = load_prompts(a.source)
    cache = Path(a.out).resolve()
    if cache.is_file():
        current = json.loads(cache.read_text())
        if current.get('model') != a.model:
            raise ValueError(f'PAP cache model mismatch: {cache}')
        if len(validate(current, rows, a.source, a.seed, a.reproduct, complete=False)) == len(rows):
            print(f'PAP cache complete: {cache} ({len(rows)} rows)', flush=True)
            return
    devices = parse_gpu_ids(a.gpus)
    extra = (lambda i, gpu: ['--reuse', str(cache)]) if cache.is_file() else None
    procs = spawn_shards(__file__, strip_argv_flag(sys.argv[1:], '--gpus'), devices, shard_slices(len(rows), len(devices)), cache, extra)
    results, head = wait_merge_shards(procs)
    merged = {**identity(a.source, a.seed, a.reproduct, a.model), **({'backend': head['backend']} if 'backend' in head else {}), 'results': results}
    validate(merged, rows, a.source, a.seed, a.reproduct, complete=True)
    save(cache, merged)
    print(f'PAP cache complete: {cache} ({len(results)}/{len(rows)} rows)', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--model', default='Qwen/Qwen3-14B')
    p.add_argument('--reproduct', action='store_true')
    p.add_argument('--out', required=True)
    p.add_argument('--gpus', help='Comma-separated GPU ids: one worker per GPU, shards merged into --out')
    p.add_argument('--reuse', help='Existing partial prompt cache with matching provenance')
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--n', type=int, default=None)
    a = p.parse_args()
    if a.start < 0 or (a.n is not None and a.n < 0):
        raise ValueError('--start and --n must be >= 0')
    if a.gpus:
        if a.start or a.n is not None or a.reuse:
            raise ValueError('--gpus drives the shards itself; do not combine it with --start, --n or --reuse')
        generate_sharded(a)
    else:
        generate_shard(a)


if __name__ == '__main__':
    main()
