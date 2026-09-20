"""Reuse saved grading for the same input and judge, irrespective of runtime settings."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def saved_scope(payload):
    """Read explicit scope, or provenance from an older prepared DIJA input."""
    if payload.get('evaluation_scope'):
        return payload['evaluation_scope']
    source = payload.get('source')
    if source:
        try:
            data = json.loads(Path(source).read_text())
            if data.get('evaluation_scope'):
                return data['evaluation_scope']
            if (data.get('attack', {}).get('attack') == 'dija'
                    and 'assistant_text' in str(data.get('evaluation_input', ''))):
                return 'dija_combined'
        except (OSError, ValueError, TypeError):
            pass
    return None


def matches(payload, expected, kind):
    actual = payload.get('results', [])
    if not isinstance(payload.get('summary'), dict) or len(actual) != len(expected):
        return False
    scopes = {r.get('evaluation_scope') for r in expected}
    scope = next(iter(scopes)) if len(scopes) == 1 else None
    if scope and saved_scope(payload) != scope:
        return False
    expected = {r['index']: r for r in expected}
    if len({r['index'] for r in actual}) != len(actual):
        return False
    if any(r['index'] not in expected or any(r.get(k) != v for k,v in expected[r['index']].items()
                                           if k != 'evaluation_scope')
           or (r.get('evaluation_scope') is not None and r['evaluation_scope'] != scope)
           for r in actual):
        return False
    grader = payload.get('summary', {}).get('grader_kind', '')
    if kind == 'lg4':
        return payload.get('guard_model') == 'meta-llama/Llama-Guard-4-12B' or grader == 'llamaguard4'
    return 'gpt-oss:20b' in payload.get('grader', '') or grader == 'gpt-oss-20b'


def find_cached(repo, command, data):
    from attack_evaluation import generation_items
    from interface_results import project_path
    kind = 'lg4' if Path(command['argv'][1]).name == 'eval_llamaguard.py' else 'gptoss'
    expected = generation_items(data)
    # Completed interactive results plus earlier experiment-level evaluations.
    candidates = set()
    for path in (repo/'outputs'/'interactive').glob('*/plan.json'):
        try:
            for old in json.loads(path.read_text())['commands']:
                if old.get('status') == 'complete' and Path(old['argv'][1]).name == Path(command['argv'][1]).name:
                    # plan.json keeps absolute paths from the machine that ran the judge.
                    candidates.add(project_path(old['argv'][old['argv'].index('--out')+1], repo/'outputs'))
        except (OSError, ValueError, KeyError, IndexError):
            continue
    for pattern in (f'*/*_{kind}.json', f'*_{kind}.json'):
        candidates.update((repo/'outputs').glob(pattern))
    candidates = sorted((p for p in candidates if p.is_file()), key=lambda p:p.stat().st_mtime, reverse=True)
    for output in candidates:
        try:
            payload = json.loads(output.read_text())
            if matches(payload, expected, kind):
                return {'path':str(output.resolve()), 'sha256':sha256(output)}
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            continue
    return None


def load_cached(command):
    cache = command['cached_result']
    if sha256(cache['path']) != cache['sha256']:
        raise ValueError('선택한 저장 평가 결과가 변경됐습니다. 다시 선택하세요.')
    original = command['parameters']['input_original']
    if sha256(original) != command['parameters']['input_sha256']:
        raise ValueError('평가 입력이 변경됐습니다. 다시 선택하세요.')
    data = json.loads(Path(cache['path']).read_text())
    scope = command['parameters'].get('evaluation_scope')
    if scope:
        if saved_scope(data) != scope:
            raise ValueError('저장된 DIJA 평가 범위가 선택한 범위와 다릅니다.')
        data['evaluation_scope'] = scope
    return data


def show_summary(data):
    """Display the saved aggregate without loading a grading model."""
    summary = data.get('summary')
    if not isinstance(summary, dict):
        return
    names = {'llamaguard4': 'Llama Guard 4', 'gpt-oss-20b': 'GPT-OSS 20B'}
    kind = summary.get('grader_kind')
    if not kind:
        kind = ('llamaguard4' if data.get('guard_model') else
                'gpt-oss-20b' if 'gpt-oss' in str(data.get('grader', '')) else '평가')
    print(f'\n[{names.get(kind, kind)} 결과]', flush=True)
    scope = data.get('evaluation_scope')
    if scope in ('dija_combined', 'dija_template'):
        print('  DIJA 평가 범위: ' + ('assistant 포함' if scope == 'dija_combined'
                                     else 'assistant 제외'), flush=True)
    if 'asr' not in summary or 'total' not in summary:
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return
    total = summary['total']
    if 'successful' in summary:
        successful = summary['successful']
        failed = summary.get('failed')
    elif 'unsafe' in summary:
        successful, failed = summary['unsafe'], summary.get('safe')
    elif 'refused' in summary:
        successful, failed = total - summary['refused'], summary['refused']
    else:
        # Preserve unfamiliar summaries instead of guessing their semantics.
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return
    unresolved = summary.get('unresolved', summary.get('n_errors', 0))
    if summary['asr'] is None:
        print(f'  ASR: 계산 불가 | 유효 채점 {total}건 | 미확정 {unresolved}건', flush=True)
        return
    suffix = ' (미확정 포함 하한)' if summary.get('asr_is_lower_bound') else ''
    print(f'  ASR: {summary["asr"]:.2%}{suffix} | 성공 {successful}/{total}건', flush=True)
    print(f'  실패 {failed if failed is not None else "미기록"}건 | 미확정 {unresolved}건', flush=True)
    if unresolved and 'unresolved' not in summary:
        print(f'  위 ASR은 오류 {unresolved}건을 제외한 유효 채점 기준입니다.', flush=True)
    if unresolved and 'asr_upper_bound' in summary:
        print(f'  미확정에 따른 ASR 범위: {summary["asr"]:.2%} ~ '
              f'{summary["asr_upper_bound"]:.2%}', flush=True)


def show_cached(command):
    data = load_cached(command)
    print(f'\n이미 완료된 평가입니다. 저장 결과를 재사용합니다: {command["cached_result"]["path"]}')
    show_summary(data)
    return data
