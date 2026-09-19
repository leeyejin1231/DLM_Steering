"""Discover generation outputs without reading large traces in every file."""
import json
from pathlib import Path
import re


def preview(path):
    """Read metadata and the first row only; evaluate full input after selection."""
    try:
        with path.open() as stream:
            text = stream.read(256 * 1024)
        decoder = json.JSONDecoder()
        pos = text.index('{') + 1
        metadata = {}
        while True:
            while text[pos].isspace() or text[pos] == ',':
                pos += 1
            key, pos = decoder.raw_decode(text, pos)
            while text[pos].isspace() or text[pos] == ':':
                pos += 1
            if key == 'results':
                if text[pos] != '[':
                    return None
                pos += 1
                while text[pos].isspace():
                    pos += 1
                row, _ = decoder.raw_decode(text, pos)
                if not isinstance(row, dict) or not {'index','prompt','generation'} <= row.keys():
                    return None
                if 'summary' in metadata or 'source_model' in metadata:
                    return None
                return metadata
            value, pos = decoder.raw_decode(text, pos)
            metadata[key] = value
    except (OSError, ValueError, IndexError, TypeError):
        return None


def discover(repo):
    outputs = repo/'outputs'
    # Run-level files only: never list chunks, archive snapshots or model caches.
    paths = set(outputs.glob('*.json'))
    paths.update(outputs.glob('*/*.json'))
    paths.update((outputs/'interactive').glob('*/*.json'))
    entries = []
    plans = {}
    for path in paths:
        if path.is_symlink() or re.search(r'(\.part\d+|_combined|_retry|_input|_before_)', path.stem):
            continue
        metadata = preview(path)
        if metadata is None:
            continue
        if path.parent not in plans:
            plans[path.parent] = {}
            plan_path = path.parent/'plan.json'
            if plan_path.exists():
                try:
                    plan = json.loads(plan_path.read_text())
                    for command in plan['commands']:
                        argv = command['argv']
                        if Path(argv[1]).name == 'exp.py' and '--out' in argv:
                            plans[path.parent][Path(argv[argv.index('--out')+1]).resolve()] = command
                except (OSError, ValueError, KeyError, IndexError):
                    pass
        command = plans[path.parent].get(path.resolve())
        if command and command.get('status') != 'complete':
            continue
        cfg = command.get('parameters', {}) if command else {}
        attack = metadata.get('attack', {})
        defense = metadata.get('defense', {})
        seed = cfg.get('seed', attack.get('assignment_seed') if isinstance(attack, dict) else None)
        if seed is None:
            match = re.search(r'seed[_-]?(\d+)', path.stem)
            seed = match[1] if match else '?'
        model = str(metadata.get('model', cfg.get('model', '?'))).split('/')[-1]
        attack_name = attack.get('attack','?') if isinstance(attack, dict) else attack
        defense_name = defense.get('defense','?') if isinstance(defense, dict) else defense
        rows = str(cfg['n'])+'행' if 'n' in cfg else '행 수: 선택 후 확인'
        label = f'{model} | {attack_name} / {defense_name} | seed {seed} | {rows}'
        if '?' in label or 'None' in label:
            continue
        entries.append({'path':path.resolve(), 'label':label, 'relative':str(path.relative_to(repo)),
                        'mtime':path.stat().st_mtime})
    return sorted(entries, key=lambda e:(-e['mtime'],e['relative']))


def select_result(repo, ask, existing_file):
    print('\n저장된 실험 결과 검색 중… (실행 중인 대화형 작업·샤드·채점 결과는 제외)')
    entries = discover(repo)
    page, query = 0, ''
    while True:
        filtered = [e for e in entries if query.lower() in (e['label']+' '+e['relative']).lower()]
        pages = max(1, (len(filtered)+14)//15)
        page = min(page, pages-1)
        print(f'\n평가할 실험 결과 — {len(filtered)}개, {page+1}/{pages}페이지 (최신순)')
        for i, entry in enumerate(filtered[page*15:(page+1)*15], page*15+1):
            print(f'  {i}. {entry["label"]}')
        print('번호: 선택 | n/p: 다음/이전 | /검색어: 필터 | r: 새로고침 | 0: 경로 직접 입력')
        value = ask('결과 선택', '1' if filtered else '0')
        if value == '0':
            return ask('평가할 생성 결과 JSON 파일', convert=existing_file)
        if value == 'n':
            page = min(page+1, pages-1)
        elif value == 'p':
            page = max(page-1, 0)
        elif value == 'r':
            entries = discover(repo)
        elif value.startswith('/'):
            query, page = value[1:], 0
        elif value.isdigit() and 1 <= int(value) <= len(filtered):
            return filtered[int(value)-1]['path']
        else:
            print('목록의 번호 또는 안내된 명령을 입력하세요.')
