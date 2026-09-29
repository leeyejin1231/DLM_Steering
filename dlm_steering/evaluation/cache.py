"""Reuse saved grading for the same input and judge, irrespective of runtime settings."""
import json
from pathlib import Path


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
    """True when a saved grading covers exactly `expected` with the judge `kind`."""
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
