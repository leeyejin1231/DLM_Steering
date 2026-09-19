"""Interactive experiment/evaluation launcher. Run: .venv/bin/python interface.py"""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import uuid

REPO = Path(__file__).resolve().parent
PYTHON = REPO / '.venv/bin/python'


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


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def command(script, args, config, env=None):
    return {'argv': [str(PYTHON), script, *map(str, args)],
            'env': env or {}, 'parameters': config}


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
    from common import load_prompts
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
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
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
            failed = [(gpu, process.wait(), log_path) for gpu, process, log_path, _ in running]
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


def extra_args():
    text = ask('추가 CLI 옵션 (기본값 유지: Enter)', '')
    args = shlex.split(text)
    reserved = {'--attack', '--defense', '--source', '--seed', '--gpus', '--out', '--in',
                '--jobs', '--n', '--start', '--pap-cache', '--reproduct', '--model',
                '--procs-per-gpu'}
    if any(token.startswith('--') and any(flag.startswith(token.split('=')[0]) for flag in reserved) for token in args):
        raise ValueError('공격·방어·입력·출력·시드·GPU·범위·캐시는 위 질문으로 설정하세요.')
    return args


def parse_experiment_args(argv):
    """Resolve model-specific CLI defaults in an isolated process."""
    from types import SimpleNamespace
    script = "import json; from exp import parse_args; print(json.dumps(vars(parse_args())))"
    result = subprocess.run([str(PYTHON), '-c', script, *argv], cwd=REPO,
                            text=True, capture_output=True)
    if result.returncode:
        raise ValueError(result.stderr.strip())
    return SimpleNamespace(**json.loads(result.stdout))


def experiment_plan(folder):
    from common import PROMPT_SOURCES, load_prompts
    from pap_common import default_cache_path, load_cache
    from models import MODELS
    model_key = choose('타깃 모델', [(key, spec['name']) for key, spec in MODELS.items()])
    attack = choose('공격', [(x, x) for x in ('none', 'dija', 'pap', 'pair', 'prefix')])
    defenses = [('ours', 'Ours V3'), ('none', '없음'), ('selfreminder', 'SelfReminder')]
    if MODELS[model_key]['family'] == 'llada':
        defenses.insert(1, ('diffuguard', 'DiffuGuard'))
        if model_key != 'llada':
            print(f'V3는 {MODELS[model_key]["out_dir"]}의 전용 체크포인트가 필요합니다.')
    else:
        print('Dream: V3는 outputs/dream의 전용 체크포인트가 필요합니다. DiffuGuard 통합은 LLaDA 전용입니다.')
    defense = choose('방어', defenses)
    sources = ('jbb_harmful', 'harmbench', 'strongreject') if attack == 'dija' else PROMPT_SOURCES
    source = choose('데이터셋', [(x, x) for x in sources])
    selected_seeds = ask('시드 (여러 개: 42,43,44)', '42', seeds)
    gpus = ask('GPU (all: 전체 자동 분할 / 0: 첫 GPU / 0,1: 지정)', 'all', gpu_ids)
    print(f'선택 GPU: {gpus} — 데이터를 작업자에 나누어 처리합니다.')
    devices = gpus.split(',')
    procs = '1'
    if attack != 'pair':
        # 행 단위 시드라 작업자 수와 무관하게 출력은 같다. 330토큰 안팎의 행은 작업자
        # 1개로 이미 GPU가 포화되어 2개가 약 10% 느렸으므로(실측) 기본값은 1.
        procs = ask('GPU당 생성 작업자 수 (1 권장 / 2: 짧은 프롬프트용 / auto: 카드별 여유 메모리로 1~2개)', '1', worker_count)
        from common import worker_devices
        layout = worker_devices(gpus, procs)
        print('작업자 배치: ' + ', '.join(f'GPU {g}×{layout.count(g)}' for g in dict.fromkeys(layout))
              + (' (실행 시점의 여유 메모리로 다시 계산됩니다)' if procs == 'auto' else ''))
    if attack == 'pair' and (len(devices) % 2 or any(a == b for a, b in zip(devices[::2], devices[1::2]))):
        raise ValueError('PAIR는 타깃/공격자 GPU 쌍이 필요합니다. 예: 0,1 또는 0,1,2,3')
    reproduct = choose('결정적 실행 (--reproduct)', [(True, '사용'), (False, '사용 안 함')])
    rows = load_prompts(source)
    start = ask('시작 행 (0부터)', 0, integer)
    if start >= len(rows):
        raise ValueError(f'데이터셋은 {len(rows)}행입니다.')
    n = ask(f'행 수 (남은 전체: {len(rows)-start})', len(rows)-start, lambda x: integer(x, 1))
    if start + n > len(rows):
        raise ValueError('선택 범위가 데이터셋을 벗어납니다.')
    length = ask('어시스턴트 생성 토큰 수', 128, integer)
    steps = ask('디퓨전 스텝', 128, lambda x: integer(x, 1))
    block = ask('경계 검사 간격 (생성 토큰 수)' if model_key == 'dream' else '블록 길이',
                32, lambda x: integer(x, 1))
    temperature = ask('Temperature', 0.2, real)
    options = ['--model', model_key, '--attack', attack, '--defense', defense, '--source', source,
               '--start', str(start), '--n', str(n), '--gen-length', str(length),
               '--steps', str(steps), '--block-length', str(block),
               '--temperature', str(temperature), '--gpus', gpus,
               '--procs-per-gpu', procs]
    if model_key == 'dream':
        options += ['--decoder', 'dream', '--alg', choose('Dream 토큰 선택 방식', [(x, x) for x in ('origin', 'entropy', 'maskgit_plus', 'topk_margin')]),
                    '--top-p', str(ask('Top-p', 0.95, real)), '--top-k', str(ask('Top-k (0: 제한 없음)', 50, integer))]
    if reproduct:
        options += ['--reproduct']
    if defense == 'ours':
        if model_key == 'dream' and choose('Dream 이전 위치에도 steering 적용', [(False, '사용 안 함 (브랜치 기본값)'), (True, '사용')]):
            options += ['--steer-shift']
        options += ['--alpha', str(ask('Alpha', 1, real)), '--remask', 'v3']
        if choose('프롬프트 전체 remasking', [(True, '허용'), (False, '허용하지 않음')]):
            options += ['--remask-prompt']
    if defense == 'diffuguard':
        options += ['--remasking', 'adaptive_step', '--repair-scope', 'all']
        print('DiffuGuard 기본: SAR 활성화, 전체 블록 복구')
    if attack == 'pair':
        options += ['--pair-llm', ask('PAIR 공격자 모델', 'Qwen/Qwen3-14B')]
    options += extra_args()
    commands = []
    for seed in selected_seeds:
        output = folder / f'{model_key + "_" if model_key != "llada" else ""}{source}_{attack}_{defense}_seed{seed}.json'
        argv = options + ['--seed', str(seed), '--out', str(output)]
        cfg = parse_experiment_args(argv)
        if (cfg.steps <= 0 or cfg.block_length <= 0 or cfg.gen_length < 0
                or not math.isfinite(cfg.temperature) or cfg.temperature < 0):
            raise ValueError('생성 파라미터 범위를 확인하세요.')
        if cfg.decoder == "block" and cfg.gen_length and (cfg.gen_length % cfg.block_length
                               or cfg.steps % (cfg.gen_length // cfg.block_length)):
            raise ValueError('생성 길이는 블록 길이의 배수, 스텝은 블록 수의 배수여야 합니다.')
        if not cfg.gen_length and attack != 'dija':
            raise ValueError('gen-length=0은 프롬프트 마스크가 있는 DIJA에서 사용하세요.')
        if defense == 'ours':
            required = [cfg.detector] + ([cfg.vector] if cfg.steer != 'none' else []) + ([cfg.response_detector] if cfg.remask == 'v3' else [])
            missing = [p for p in required if not (REPO / p).is_file()]
            if missing:
                raise ValueError(f'{model_key} 전용 방어 체크포인트가 없습니다: {missing}. steering 피팅 명령에 --model {model_key}를 사용해 준비하세요.')
        if attack == 'pap':
            cache = default_cache_path(source, seed)
            ready = False
            if cache.exists():
                try:
                    data = load_cache(cache, source, seed, reproduct)
                    print(f'PAP 캐시 재사용: {cache} (공격 모델: {data["model"]})')
                    ready = True
                except ValueError:
                    print(f'PAP 캐시 미완성: {cache}. 저장된 행부터 이어서 준비합니다.')
            if not ready:
                model = ask('PAP 공격 생성 모델', 'Qwen/Qwen3-14B')
                commands.append(pap_preparation_command(source, seed, model,
                                                        reproduct, gpus, cache))
            cfg.pap_cache = str(cache)
            argv += ['--pap-cache', str(cache)]
        commands.append(command('exp.py', argv, {**vars(cfg), 'model_name': MODELS[model_key]['name']}))
    return commands, []


def pap_plan():
    from common import load_prompts
    from pap_common import default_cache_path, load_cache
    source = choose('PAP 데이터셋', [(x, x) for x in
                    ('jbb_harmful', 'harmbench', 'strongreject')])
    selected_seeds = ask('시드 (여러 개: 42,43,44)', '42,43,44', seeds)
    gpus = ask('GPU (all: 전체 자동 분할 / 0,1: 지정)', 'all', gpu_ids)
    reproduct = choose('결정적 실행 (--reproduct)', [(True, '사용'), (False, '사용 안 함')])
    model = ask('PAP 공격 생성 모델', 'Qwen/Qwen3-14B')
    rows = load_prompts(source)
    commands = []
    for seed in selected_seeds:
        cache = default_cache_path(source, seed)
        if cache.is_file():
            try:
                data = load_cache(cache, source, seed, reproduct)
                if data['model'] == model and len(data['results']) == len(rows):
                    print(f'PAP 캐시 완료: {cache}')
                    continue
            except ValueError:
                pass
        commands.append(pap_preparation_command(source, seed, model,
                                                reproduct, gpus, cache))
    return commands, []


def existing_file(value):
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError('존재하는 파일을 입력하세요.')
    return path


def evaluation_plan(folder):
    from interface_results import select_result
    inp = select_result(REPO, ask, existing_file)
    data = json.loads(inp.read_text())
    if not isinstance(data.get('results'), list) or not data['results']:
        raise ValueError('results 행이 있는 생성 결과 JSON이 필요합니다.')
    if any(not all(k in r for k in ('index', 'prompt', 'generation')) for r in data['results']):
        raise ValueError('생성 결과의 index/prompt/generation 필드를 확인하세요.')
    print(f'선택한 결과: {inp} ({len(data["results"])}행)')
    judge = choose('평가 모델', [('lg4', 'Llama-Guard-4-12B (ASR)'),
                                 ('gptoss', 'GPT-OSS-20B / StrongREJECT (ASR)'),
                                 ('both', '두 모델 모두 (순서대로)')])
    prepared = []
    digest = hashlib.sha256(inp.read_bytes()).hexdigest()
    scope = None
    original = str(inp)
    if data.get('attack', {}).get('attack') == 'dija':
        scope = choose('DIJA 평가 범위', [
            ('dija_combined', '전체: 템플릿 채움 + assistant 답변'),
            ('dija_template', '템플릿 채움만')], default=2)
        required = ['filled_template'] + (['assistant_text'] if scope == 'dija_combined' else [])
        for row in data['results']:
            if row['generation'] == '[STEERING_ERROR]':
                continue
            if any(not isinstance(row.get(key), str) for key in required):
                raise ValueError(f'DIJA 평가에 필요한 필드가 없습니다: {required}')
            row['generation'] = row['filled_template']
            if scope == 'dija_combined':
                row['generation'] += '\n\n' + row['assistant_text']
        data['evaluation_scope'] = scope
        data['evaluation_input'] = ('filled_template + assistant_text' if scope == 'dija_combined'
                                    else 'filled_template')
        inp = folder / f'{scope}_input.json'
        prepared.append((inp, data))
        print('선택한 DIJA 평가 범위:', 'assistant 포함' if scope == 'dija_combined' else 'assistant 제외')
    def result_path(kind):
        return folder / f'{kind}_{scope}.json' if scope else folder / f'{kind}.json'
    commands = []
    shared = {'input_original': original, 'input_sha256': digest,
              'target_model': data.get('model'), 'generation_config': data.get('config'),
              'attack': data.get('attack'), 'rows': len(data['results']),
              'evaluation_scope': scope}
    from evaluation_cache import find_cached
    cached_commands = {}
    for kind, script in [('lg4','eval_llamaguard.py'), ('gptoss','run_sr_eval.py')]:
        if judge not in (kind, 'both'):
            continue
        probe = command(script, ['--in', str(inp), '--out', str(result_path(kind))], shared)
        cached = find_cached(REPO, probe, data)
        if cached:
            probe['cached_result'] = cached
            cached_commands[kind] = probe
    requested = ['lg4','gptoss'] if judge == 'both' else [judge]
    if len(cached_commands) == len(requested):
        return [cached_commands[k] for k in requested], []
    commands.extend(cached_commands.values())
    gpu = ask('평가 GPU (all: 전체 자동 분할 / 0 또는 0,1: 지정)', 'all', gpu_ids)
    print(f'선택 GPU: {gpu} — 평가 데이터를 나누어 처리합니다.')
    if judge in ('lg4', 'both') and 'lg4' not in cached_commands:
        batch = ask('LG4 배치 크기', 16, lambda x: integer(x, 1))
        args = ['--in', str(inp), '--out', str(result_path('lg4')), '--batch-size', str(batch), '--gpus', gpu]
        from eval_llamaguard import parse_args
        commands.append(command('eval_llamaguard.py', args, {**shared, **vars(parse_args(args))}))
    if judge in ('gptoss', 'both') and 'gptoss' not in cached_commands:
        workers = ask('GPT-OSS 동시 요청 수', 4, lambda x: integer(x, 1))
        tokens = ask('GPT-OSS 최대 채점 출력 토큰', 4096, lambda x: integer(x, 1))
        devices = gpu.split(',')
        if any(not device.isdigit() for device in devices):
            raise ValueError('기존 GPT-OSS CLI는 숫자 GPU 번호가 필요합니다.')
        args = ['--in', str(inp), '--out', str(result_path('gptoss')), '--auto-server',
                '--gpus', gpu, '--gpu', devices[0], '--model', 'gpt-oss:20b', '--workers', str(workers),
                '--reasoning-effort', 'low', '--num-predict', str(tokens)]
        from run_sr_eval import parse_args
        settings = vars(parse_args(args))
        for internal in ('port', 'container', 'timeout_sec'):
            settings.pop(internal)
        settings['server'] = 'GPU별 전용 서버 자동 시작/종료, 빈 포트 자동 배정'
        commands.append(command('run_sr_eval.py', args,
                                {**shared, **settings, 'temperature': 0.0}))
    for item in commands:
        if not item.get('cached_result') and item['argv'][1] == 'run_sr_eval.py':
            from ollama_runtime import runtime_paths
            runtime_paths()
    return commands, prepared


def report(folder, plan):
    lines = ['# 대화형 실행 결과', '', f'상태: {plan["status"]}', '',
             '전체 설정과 실행 이력: [plan.json](plan.json)', '']
    for i, item in enumerate(plan['commands'], 1):
        lines += [f'## {i}. {item["argv"][1]}', '', f'상태: {item.get("status", "pending")}', '',
                  '```bash', format_command(item), '```', '']
        scope = item.get('parameters', {}).get('evaluation_scope')
        if scope:
            label = 'assistant 포함 (템플릿 + 답변)' if scope == 'dija_combined' else 'assistant 제외 (템플릿만)'
            lines += [f'DIJA 평가 범위: {label}', '']
        if item.get('cached_result'):
            lines += [f'저장된 평가 재사용: `{item["cached_result"]["path"]}`', '']
        out = Path(item['argv'][item['argv'].index('--out')+1])
        if out.exists() and item.get('status') == 'complete':
            data = json.loads(out.read_text())
            if 'summary' in data:
                lines += ['```json', json.dumps(data['summary'], ensure_ascii=False, indent=2), '```', '']
    (folder/'RESULTS.md').write_text('\n'.join(lines))


def format_command(item):
    env = {**{'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'}, **item['env']}
    return shlex.join(['env', *[f'{k}={v}' for k, v in env.items()], *item['argv']])


def execute(folder, commands, prepared):
    from interface_progress import Progress, stream_process
    folder.mkdir(parents=True, exist_ok=False)
    plan = {'created': dt.datetime.now(dt.timezone.utc).isoformat(), 'status': 'running',
            'cwd': str(REPO), 'commands': commands}
    save(folder/'plan.json', plan)
    index = REPO/'RESULTS.md'
    with index.open('a') as stream:
        stream.write(f'\n- [{folder.name}]({folder.relative_to(REPO).as_posix()}/RESULTS.md)\n')
    for path, data in prepared:
        save(path, data)
    report(folder, plan)
    try:
        for i, item in enumerate(commands, 1):
            if item.get('cached_result'):
                from evaluation_cache import show_cached
                payload = show_cached(item)
                out = Path(item['argv'][item['argv'].index('--out')+1])
                save(out, payload)
                item.update(status='complete', returncode=0, reused=True)
                save(folder/'plan.json', plan)
                report(folder, plan)
                continue
            item['status'] = 'running'
            save(folder/'plan.json', plan)
            env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1', **item['env'])
            progress = Progress(item, i, len(commands))
            print(f'\n로그: {folder / f"{i:02d}.log"}', flush=True)
            with (folder/f'{i:02d}.log').open('w') as log:
                process = subprocess.Popen(item['argv'], cwd=REPO, env=env,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                try:
                    code = stream_process(process, log, progress)
                except BaseException:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise
                finally:
                    process.stdout.close()
            item.update(status='complete' if code == 0 else 'failed', returncode=code)
            if code:
                raise RuntimeError(f'{i}번 명령 실패. {folder / f"{i:02d}.log"} 확인')
            if Path(item['argv'][1]).name in ('eval_llamaguard.py', 'run_sr_eval.py'):
                from evaluation_cache import show_summary
                out = Path(item['argv'][item['argv'].index('--out')+1])
                show_summary(json.loads(out.read_text()))
        plan['status'] = 'complete'
    except BaseException as exc:
        plan.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
        if item.get('status') == 'running':
            item['status'] = plan['status']
        raise
    finally:
        save(folder/'plan.json', plan)
        report(folder, plan)
        print(f'\n결과와 실행 기록: {folder}')


def main():
    if not PYTHON.is_file():
        raise ValueError('.venv/bin/python 환경을 먼저 준비하세요.')
    if Path(sys.prefix).resolve() != (REPO/'.venv').resolve():
        os.execv(str(PYTHON), [str(PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])
    os.chdir(REPO)
    if len(sys.argv) == 3 and sys.argv[1] == '--watch':
        from interface_progress import watch
        watch(sys.argv[2])
        return
    if '--pap-generate' in sys.argv[1:]:
        from pap_common import default_cache_path
        parser = argparse.ArgumentParser(description='PAP 프롬프트를 GPU별로 분산 생성하고 합칩니다.')
        parser.add_argument('--pap-generate', action='store_true')
        parser.add_argument('--source', required=True,
                            choices=('jbb_harmful', 'harmbench', 'strongreject'))
        parser.add_argument('--seed', required=True, type=int)
        parser.add_argument('--gpus', default='all')
        parser.add_argument('--pap-model', default='Qwen/Qwen3-14B')
        parser.add_argument('--reproduct', action='store_true')
        parser.add_argument('--out')
        args = parser.parse_args()
        cache = Path(args.out).resolve() if args.out else default_cache_path(args.source, args.seed)
        prepare_pap_sharded(args.source, args.seed, args.pap_model,
                            args.reproduct, gpu_ids(args.gpus), cache)
        return
    print('실험 / 평가 인터페이스 — Enter: 기본값, Ctrl+C: 취소')
    mode = choose('작업 선택', [('experiment', '실험용'), ('evaluation', '평가용'),
                                 ('pap', 'PAP 공격 프롬프트 준비')])
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    folder = REPO/'outputs'/'interactive'/f'{stamp}_{mode}_{uuid.uuid4().hex[:8]}'
    commands, prepared = (experiment_plan(folder) if mode == 'experiment' else
                          pap_plan() if mode == 'pap' else evaluation_plan(folder))
    if not commands:
        print('요청한 PAP 캐시가 모두 완성돼 있습니다.')
        return
    if commands and all(item.get('cached_result') for item in commands):
        from evaluation_cache import show_cached
        for item in commands:
            show_cached(item)
        return
    print('\n===== 실행 전 최종 확인 =====')
    print(f'결과 폴더: {folder}')
    for i, item in enumerate(commands, 1):
        if item.get('cached_result'):
            from evaluation_cache import show_cached
            show_cached(item)
            continue
        print(f'\n[{i}] 최종 파라미터 (CLI 기본값 포함)')
        print(json.dumps(item['parameters'], ensure_ascii=False, indent=2))
        print(format_command(item))
    if ask('이 설정으로 실행하려면 yes 입력', 'yes').lower() != 'yes':
        print('취소했습니다. 실행하거나 결과 파일을 만들지 않았습니다.')
        return
    execute(folder, commands, prepared)


if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print('\n취소했습니다.')
        sys.exit(130)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f'오류: {exc}', file=sys.stderr)
        sys.exit(1)
