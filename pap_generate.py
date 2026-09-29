import argparse
import json
from pathlib import Path

from attack_prompts import load_better_templates, extract_content
from dlm_steering.runtime.data import load_prompts
from dlm_steering.runtime.models import enable_reproducibility, seed_all, force_math_attention
from pap_common import (SAMPLING, assign_techniques, normalize_templates, identity, validate, save)
from dlm_steering.runtime.progress import task_progress


class PAPGenerator:
    def __init__(self, llm, templates):
        self.llm = llm
        self.templates = normalize_templates(templates)

    def generate(self, goal, technique, seed):
        raw = self.llm.generate(
            [{"role": "user", "content": self.templates[technique].replace("%s", goal)}],
            **SAMPLING, seed=seed)
        mutation = extract_content(
            "#mutated_text_with_same_specific_harmful_or_unlawful_intention: ", raw)
        if not mutation:
            raise ValueError("PAP output missing a nonempty mutation tag; no retry")
        return mutation


def generate_shard(a):
    """Fill the rows of this shard that --reuse / --out do not already hold."""
    from attack_llms import HFChat
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
    selected = rows[a.shard_index::a.shard_count]
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
    import os
    import subprocess
    import sys
    from dlm_steering.runtime.execution import parse_gpu_ids
    from dlm_steering.runtime.progress import CHILD_PROGRESS_ENV
    devices = parse_gpu_ids(a.gpus)
    rows = load_prompts(a.source)
    cache = Path(a.out).resolve()
    if cache.is_file():
        current = json.loads(cache.read_text())
        if current.get('model') != a.model:
            raise ValueError(f'PAP cache model mismatch: {cache}')
        done = validate(current, rows, a.source, a.seed, a.reproduct, complete=False)
        if len(done) == len(rows):
            print(f'PAP cache complete: {cache} ({len(rows)} rows)', flush=True)
            return
    shard_dir = cache.parent / '.shards' / f'seed{a.seed}'
    shard_dir.mkdir(parents=True, exist_ok=True)
    running = []
    for index, gpu in enumerate(devices):
        part = shard_dir / f'gpu{index}.json'
        log = shard_dir / f'gpu{index}.log'
        cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--source', a.source,
                '--seed', str(a.seed), '--model', a.model, '--out', str(part),
                '--shard-index', str(index), '--shard-count', str(len(devices))]
        if a.reproduct:
            cmd.append('--reproduct')
        if cache.is_file():
            cmd += ['--reuse', str(cache)]
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, **CHILD_PROGRESS_ENV}
        with log.open('w') as handle:
            proc = subprocess.Popen(cmd, cwd=Path(__file__).parent, env=env,
                                    stdout=handle, stderr=subprocess.STDOUT)
        running.append((proc, part, log))
        print(f'shard {index}: gpu={gpu} -> {part} (log {log})', flush=True)
    try:
        with task_progress(running, desc=f'PAP seed={a.seed}', unit='shard') as progress:
            codes = [proc.wait() for proc, _, _ in progress]
    except BaseException:
        for proc, _, _ in running:
            if proc.poll() is None:
                proc.terminate()
        for proc, _, _ in running:
            proc.wait()
        raise
    failed = [f'shard {i} rc={code} ({log})' for i, (code, (_, _, log)) in enumerate(zip(codes, running)) if code]
    if failed:
        raise SystemExit('PAP shards failed; shard files are kept: ' + ', '.join(failed))
    merged = identity(a.source, a.seed, a.reproduct, a.model)
    entries = {}
    for path in [cache, *(part for _, part, _ in running)]:
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        if data.get('model') != a.model:
            raise ValueError(f'PAP shard model mismatch: {path}')
        for i, entry in validate(data, rows, a.source, a.seed, a.reproduct, complete=False).items():
            if entries.get(i, entry) != entry:
                raise ValueError(f'PAP row {i} differs between {path} and an earlier file')
            entries[i] = entry
        if 'backend' in data:
            merged['backend'] = data['backend']
    merged['results'] = [entries[i] for i in sorted(entries)]
    validate(merged, rows, a.source, a.seed, a.reproduct, complete=True)
    save(cache, merged)
    print(f'PAP cache complete: {cache} ({len(entries)}/{len(rows)} rows)', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--model', default='Qwen/Qwen3-14B')
    p.add_argument('--reproduct', action='store_true')
    p.add_argument('--out', required=True)
    p.add_argument('--gpus', help='Comma-separated GPU ids: one worker per GPU, shards merged into --out')
    p.add_argument('--reuse', help='Existing partial prompt cache with matching provenance')
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--shard-count', type=int, default=1)
    a = p.parse_args()
    if a.shard_count < 1 or not 0 <= a.shard_index < a.shard_count:
        raise ValueError('Invalid shard index/count')
    if a.gpus:
        if a.shard_count != 1 or a.reuse:
            raise ValueError('--gpus drives the shards itself; do not combine it with --shard-count or --reuse')
        generate_sharded(a)
    else:
        generate_shard(a)


if __name__ == '__main__':
    main()
