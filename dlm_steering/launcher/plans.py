"""Build experiment and evaluation commands before executing them."""
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
from dlm_steering.paths import REPO, PYTHON
from .ui import ask, choose, integer, real, seeds, worker_count, gpu_ids, command, existing_file
from .pap import pap_preparation_command


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
    from dlm_steering.runtime.data import PROMPT_SOURCES, load_prompts
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
        from dlm_steering.runtime.execution import worker_devices
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
    from dlm_steering.runtime.data import load_prompts
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


def threshold_plan(folder, *, ask=ask, choose=choose):
    """Build the response-threshold sweep over existing --remask v3 probe runs.

    The audit stores each row's probability before comparing it with the cutoff,
    so one probe run per dataset covers every candidate threshold at once.
    """
    from models import MODELS
    model_key = choose('프로브 모델', [(key, spec['name']) for key, spec in MODELS.items()])
    seed = ask('프로브 시드 (전체: all)', '42')
    ceiling = ask('허용 최대 오탐률 (무해·성능 세트 평균 트리거율)', 0.15, real)
    if not 0 < ceiling < 1:
        raise ValueError('오탐률 상한은 0과 1 사이여야 합니다.')
    args = ['--auto', '--model', model_key, '--max-false-positive', str(ceiling),
            '--out', str(folder/'threshold.json')]
    if seed != 'all':
        args += ['--seed', str(integer(seed))]
    print('공격(DIJA) 세트는 트리거가 많을수록, 무해·성능 세트는 적을수록 좋은 것으로 집계합니다.')
    return [command('tune_response_threshold.py', args,
                    {'model': model_key, 'seed': seed,
                     'max_false_positive': ceiling})], []


def _result_source(inp):
    """--source of the exp.py command that wrote `inp`, from its sibling plan.json."""
    try:
        for item in json.loads((inp.parent/'plan.json').read_text())['commands']:
            argv = item['argv']
            if '--out' in argv and Path(argv[argv.index('--out')+1]).name == inp.name:
                return item.get('parameters', {}).get('source')
    except (OSError, ValueError, KeyError):
        pass
    return None


def _benign_evaluation_plan(folder, inp, data, judge, ask):
    shared = {'input_original': str(inp), 'input_sha256': hashlib.sha256(inp.read_bytes()).hexdigest(),
              'target_model': data.get('model'), 'generation_config': data.get('config'),
              'attack': data.get('attack'), 'rows': len(data['results']), 'evaluation_scope': None}
    if judge == 'utility':
        if any('task' not in r or 'answer' not in r for r in data['results']):
            raise ValueError('정답 필드(task/answer)가 없습니다. gsm8k, math500, truthfulqa_mc, mmlu 결과를 선택하세요.')
        args = ['--in', str(inp), '--out', str(folder/'utility.json')]
        return [command('eval_utility.py', args, shared)], []
    gpu = ask('평가 GPU (all: 전체 자동 분할 / 0 또는 0,1: 지정)', 'all', gpu_ids)
    workers = ask('GPT-OSS 동시 요청 수', 4, lambda x: integer(x, 1))
    args = ['--in', str(inp), '--out', str(folder/'refusal.json'), '--auto-server', '--gpus', gpu,
            '--model', 'gpt-oss:20b', '--workers', str(workers), '--reasoning-effort', 'low',
            '--num-predict', '512']
    from ollama_runtime import runtime_paths
    runtime_paths()
    return [command('eval_refusal.py', args, {**shared, 'judge': 'XSTest 3-way / gpt-oss:20b',
                                               'temperature': 0.0})], []


def evaluation_plan(folder, *, ask=ask, choose=choose, select_result=None):
    if select_result is None:
        from .results import select_result
    inp = select_result(REPO, ask, existing_file)
    data = json.loads(inp.read_text())
    if not isinstance(data.get('results'), list) or not data['results']:
        raise ValueError('results 행이 있는 생성 결과 JSON이 필요합니다.')
    if any(not all(k in r for k in ('index', 'prompt', 'generation')) for r in data['results']):
        raise ValueError('생성 결과의 index/prompt/generation 필드를 확인하세요.')
    print(f'선택한 결과: {inp} ({len(data["results"])}행)')
    # 정답 필드가 있으면 성능(정답률), 무해 프롬프트 세트면 over-refusal 채점을 기본으로 제안한다.
    graded = all('task' in r and 'answer' in r for r in data['results'])
    benign = _result_source(inp) in ('xstest_safe', 'truthfulqa', 'jbb_benign', 'wj_benign')
    judge = choose('평가 모델', [('lg4', 'Llama-Guard-4-12B (ASR)'),
                                 ('gptoss', 'GPT-OSS-20B / StrongREJECT (ASR)'),
                                 ('both', '두 모델 모두 (순서대로)'),
                                 ('refusal', 'Over-refusal: XSTest 3-way 거절 판정 / GPT-OSS-20B'),
                                 ('utility', '성능: 정답률 (GSM8K / MATH500 / TruthfulQA-MC / MMLU)')],
                   5 if graded else 4 if benign else 1)
    if judge in ('refusal', 'utility'):
        return _benign_evaluation_plan(folder, inp, data, judge, ask)
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
    from dlm_steering.evaluation.cache import find_cached
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
