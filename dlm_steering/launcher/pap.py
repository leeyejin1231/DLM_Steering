"""PAP preparation commands and shard merging."""
import json
import os
from pathlib import Path
import signal
import subprocess
from dlm_steering.paths import REPO, PYTHON
from dlm_steering.runtime.progress import CHILD_PROGRESS_ENV, task_progress
from .ui import command
from .storage import save, WORKER_ENV


def pap_preparation_command(source, seed, model, reproduct, gpus, cache):
    args = ['--pap-generate', '--source', source, '--seed', str(seed),
            '--pap-model', model, '--gpus', gpus, '--out', str(cache)]
    if reproduct:
        args.append('--reproduct')
    return command('interface.py', args,
                   {'source': source, 'seed': seed, 'model': model, 'gpus': gpus,
                    'scope': '전체 원본 데이터셋; Better top5 균등 배정; 행당 1회',
                    'temperature': 1, 'top_p': 1, 'max_new_tokens': 256,
                    'reproduct': reproduct, 'out': str(cache)})


def prepare_pap_sharded(source, seed, model, reproduct, gpus, cache):
    """Resume one seed on one worker per GPU, then validate and merge caches."""
    from dlm_steering.runtime.data import load_prompts
    from pap_common import identity, validate
    rows = load_prompts(source)
    devices = gpus.split(',')
    expected = identity(source, seed, reproduct, model)
    if cache.is_file():
        current = json.loads(cache.read_text())
        if current.get('model') != model:
            raise ValueError(f'PAP 캐시 모델 불일치: {cache}')
        validate(current, rows, source, seed, reproduct, complete=False)
        if len(current['results']) == len(rows):
            print(f'PAP 캐시 완료: {cache} ({len(rows)}행)', flush=True)
            return
    shard_dir = cache.parent / '.shards' / f'seed{seed}'
    shard_dir.mkdir(parents=True, exist_ok=True)
    running = []
    for index, gpu in enumerate(devices):
        output = shard_dir / f'gpu{index}.json'
        log_path = shard_dir / f'gpu{index}.log'
        args = [str(PYTHON), '-u', 'pap_generate.py', '--source', source,
                '--seed', str(seed), '--model', model, '--out', str(output),
                '--shard-index', str(index), '--shard-count', str(len(devices))]
        if reproduct:
            args.append('--reproduct')
        if cache.is_file():
            args += ['--reuse', str(cache)]
        env = {**os.environ, **WORKER_ENV, 'CUDA_VISIBLE_DEVICES': gpu, **CHILD_PROGRESS_ENV}
        with log_path.open('w') as log:
            process = subprocess.Popen(args, cwd=REPO, env=env,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        running.append((gpu, process, log_path, output))
        print(f'GPU {gpu}: PAP 샤드 {index+1}/{len(devices)} 시작, 로그 {log_path}', flush=True)
    previous_term = signal.getsignal(signal.SIGTERM)
    def interrupted(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        try:
            with task_progress(running, desc='PAP 프롬프트 준비', unit='샤드') as progress:
                failed = [(gpu, process.wait(), log_path) for gpu, process, log_path, _ in progress]
        except BaseException:
            for _, process, _, _ in running:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
            for _, process, _, _ in running:
                process.wait()
            raise
    finally:
        signal.signal(signal.SIGTERM, previous_term)
    failed = [(gpu, code, log) for gpu, code, log in failed if code]
    if failed:
        raise RuntimeError(f'PAP 샤드 실패; 부분 캐시는 유지됩니다: {failed}')
    merged = dict(expected)
    entries = {}
    for path in [cache, *(output for _, _, _, output in running)]:
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        if data.get('model') != model:
            raise ValueError(f'PAP 샤드 모델 불일치: {path}')
        valid = validate(data, rows, source, seed, reproduct, complete=False)
        for row_index, entry in valid.items():
            if row_index in entries and entries[row_index] != entry:
                raise ValueError(f'PAP 행 {row_index} 충돌: {path}')
            entries[row_index] = entry
        if 'backend' in data:
            merged['backend'] = data['backend']
    merged['results'] = [entries[index] for index in sorted(entries)]
    validate(merged, rows, source, seed, reproduct, complete=True)
    save(cache, merged)
    print(f'PAP 캐시 완료: {cache} ({len(entries)}/{len(rows)}행)', flush=True)
