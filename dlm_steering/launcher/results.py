"""Discover generation outputs without reading large traces in every file."""
import json
from pathlib import Path
import re


def project_path(value, outputs):
    """Resolve paths recorded by a different checkout of this repository."""
    path = Path(value)
    if path.is_absolute() and not path.exists():
        parts = path.parts
        for marker in ('outputs', 'data'):
            if marker in parts:
                return outputs.parent.joinpath(*parts[parts.index(marker):]).resolve()
    return path.resolve()


ARCHIVE = 'archive'


def _archived(path, outputs):
    """True for anything under outputs/archive -- superseded runs kept on disk
    for provenance but deliberately hidden from the result list."""
    try:
        return path.resolve().relative_to(outputs.resolve()).parts[0] == ARCHIVE
    except ValueError:
        return False


def completed_grades(outputs):
    """Index completed interactive judge commands by their generation file."""
    grades = {}
    judges = {'eval_llamaguard.py': 'LG4', 'run_sr_eval.py': 'GPT-OSS',
              'eval_refusal.py': '거절판정', 'eval_utility.py': '정답률'}
    for plan_path in (outputs/'interactive').glob('*/plan.json'):
        try:
            commands = json.loads(plan_path.read_text())['commands']
            for item in commands:
                if item.get('status') != 'complete':
                    continue
                argv = item['argv']
                judge = judges.get(Path(argv[1]).name)
                params = item.get('parameters', {})
                source = params.get('input_original')
                if not judge or not source or '--out' not in argv:
                    continue
                if not project_path(argv[argv.index('--out')+1], outputs).is_file():
                    continue
                scope = params.get('evaluation_scope') or 'standard'
                grades.setdefault(project_path(source, outputs), {}).setdefault(scope, set()).add(judge)
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            continue
    return grades


def grade_label(scopes):
    names = {'dija_combined': '전체', 'dija_template': '템플릿만',
             'standard': ''}
    parts = []
    for scope, judges in sorted(scopes.items()):
        label = '+'.join(name for name in ('LG4', 'GPT-OSS', '거절판정', '정답률') if name in judges)
        parts.append(label + (f' ({names.get(scope, scope)})' if scope != 'standard' else ''))
    return ' | 채점 완료: ' + ', '.join(parts) if parts else ''


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
    grades = completed_grades(outputs)
    # Run-level files only: never list chunks, archive snapshots or model caches.
    paths = set(outputs.glob('*.json'))
    paths.update(outputs.glob('*/*.json'))
    paths.update((outputs/'interactive').glob('*/*.json'))
    entries = []
    plans = {}
    for path in paths:
        if path.is_symlink() or re.search(r'(\.part\d+|_combined|_retry|_input|_before_)', path.stem):
            continue
        if _archived(path, outputs):
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
                            plans[path.parent][project_path(argv[argv.index('--out')+1], outputs)] = command
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
        source = (cfg.get('source') or metadata.get('source') or
                  next((name for name in ('jbb_harmful', 'harmbench', 'strongreject')
                        if name in path.stem or name in path.parent.name), '데이터셋 미상'))
        label = f'{model} | {source} | {attack_name} / {defense_name} | seed {seed}'
        if '?' in label or 'None' in label:
            continue
        label += grade_label(grades.get(path.resolve(), {}))
        entries.append({'path':path.resolve(), 'label':label, 'relative':str(path.relative_to(repo)),
                        'mtime':path.stat().st_mtime})
    return sorted(entries, key=lambda e:(-e['mtime'],e['relative']))


def select_result(repo, ask, existing_file):
    print('\n저장된 실험 결과 검색 중… (실행 중인 대화형 작업·샤드·채점 결과는 제외)')
    entries = discover(repo)
    page, query = 0, ''
    while True:
        filtered = [e for e in entries if query in (e['label']+' '+e['relative']).casefold()]
        pages = max(1, (len(filtered)+14)//15)
        page = min(page, pages-1)
        search = f', 검색어: {query}' if query else ''
        print(f'\n평가할 실험 결과 — {len(filtered)}개, {page+1}/{pages}페이지 (최신순{search})')
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
            query, page = value.lstrip('/').strip().casefold(), 0
        elif value.isdigit() and 1 <= int(value) <= len(filtered):
            return filtered[int(value)-1]['path']
        else:
            print('목록의 번호 또는 안내된 명령을 입력하세요.')
