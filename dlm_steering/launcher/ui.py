"""Interactive input validation and command construction."""
import math
import os
from pathlib import Path
import subprocess
from dlm_steering.paths import PYTHON


def ask(label, default=None, convert=str):
    while True:
        raw = input(f'{label}' + (f' [{default}]' if default is not None else '') + ': ').strip()
        try:
            value = raw if raw else default
            if value is None:
                raise ValueError('값을 입력하세요.')
            return convert(str(value))
        except (ValueError, OSError) as exc:
            print(f'입력을 확인하세요: {exc}')


def choose(label, options, default=1):
    print('\n' + label)
    for i, (_, title) in enumerate(options, 1):
        print(f'  {i}. {title}')
    def parse(value):
        number = int(value)
        if not 1 <= number <= len(options):
            raise ValueError('메뉴 번호 범위를 벗어났습니다.')
        return options[number - 1][0]
    return ask('선택', default, parse)


def integer(value, minimum=0):
    number = int(value)
    if number < minimum:
        raise ValueError(f'{minimum} 이상의 정수가 필요합니다.')
    return number


def real(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError('0 이상의 유한한 숫자가 필요합니다.')
    return number


def seeds(value):
    values = list(dict.fromkeys(integer(s.strip()) for s in value.split(',')))
    if not values:
        raise ValueError('시드를 입력하세요.')
    return values


def worker_count(value):
    value = value.strip().lower()
    if value != 'auto' and integer(value, 1) > 2:
        raise ValueError('auto, 1 또는 2를 입력하세요 (8B 모델은 작업자당 약 19GB).')
    return value


def gpu_ids(value):
    allowed = os.environ.get('CUDA_VISIBLE_DEVICES')
    if value.strip().lower() == 'all':
        if allowed is not None:
            value = allowed
        else:
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
                    capture_output=True, text=True, check=True, timeout=10)
            except (OSError, subprocess.SubprocessError) as exc:
                raise ValueError('GPU 자동 탐색에 실패했습니다. GPU 번호를 직접 입력하세요.') from exc
            value = ','.join(result.stdout.splitlines())
        if not value.strip() or value.strip() == '-1':
            raise ValueError('사용 가능한 GPU가 없습니다.')
    values = [s.strip() for s in value.split(',')]
    if not values or any(not s.isdigit() and not s.startswith(('GPU-', 'MIG-')) for s in values):
        raise ValueError('all 또는 GPU 번호/UUID를 쉼표로 구분해 입력하세요.')
    if allowed is not None and not set(values) <= {s.strip() for s in allowed.split(',')}:
        raise ValueError(f'현재 CUDA_VISIBLE_DEVICES={allowed} 범위에서 선택하세요.')
    return ','.join(values)


def command(script, args, config, env=None):
    return {'argv': [str(PYTHON), script, *map(str, args)],
            'env': env or {}, 'parameters': config}


def existing_file(value):
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError('존재하는 파일을 입력하세요.')
    return path
