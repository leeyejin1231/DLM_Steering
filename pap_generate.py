"""Prepare reusable PAP prompts without loading a target or a grader."""
import argparse
import json
from pathlib import Path

from attack_prompts import load_better_templates, extract_content
from common import load_prompts, enable_reproducibility, seed_all, force_math_attention
from pap_common import (SAMPLING, assign_techniques, normalize_templates,
                        identity, validate, save)


class PAPGenerator:
    """Generate one attack prompt using the assigned author template."""

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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--model', default='Qwen/Qwen3-14B')
    p.add_argument('--reproduct', action='store_true')
    p.add_argument('--out', required=True)
    p.add_argument('--reuse', help='Existing partial prompt cache with matching provenance')
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--shard-count', type=int, default=1)
    a = p.parse_args()
    if a.shard_count < 1 or not 0 <= a.shard_index < a.shard_count:
        raise ValueError('Invalid shard index/count')
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
        for row in missing:
            i = int(row['index'])
            technique = assignments[i]
            text = generator.generate(row['prompt'], technique, a.seed + i * 1048576)
            entries[i] = dict(index=i, prompt=row['prompt'], technique=technique, attack_prompt=text)
            data['results'] = [entries[i] for i in sorted(entries)]
            save(a.out, data)
            print(f'PAP seed={a.seed} index={i}: saved', flush=True)
    data['results'] = [entries[i] for i in sorted(entries)]
    save(a.out, data)


if __name__ == '__main__':
    main()
