"""Interactive entry point for experiment and evaluation plans."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys
import uuid
from dlm_steering.paths import REPO, PYTHON
from .ui import ask, choose, gpu_ids
from .plans import (experiment_plan, evaluation_plan, pap_plan, threshold_plan,
                    detector_plan)
from .pap import prepare_pap_sharded
from .execution import format_command, execute


def main():
    if not PYTHON.is_file():
        raise ValueError('.venv/bin/python 환경을 먼저 준비하세요.')
    if Path(sys.prefix).resolve() != (REPO/'.venv').resolve():
        os.execv(str(PYTHON), [str(PYTHON), str(REPO / 'interface.py'), *sys.argv[1:]])
    os.chdir(REPO)
    if len(sys.argv) == 3 and sys.argv[1] == '--watch':
        from dlm_steering.launcher.progress import watch
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
                                 ('pap', 'PAP 공격 프롬프트 준비'),
                                 ('detector', 'V3 응답 검출기 학습'),
                                 ('threshold', 'V3 응답 검출기 임계값 탐색')])
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    folder = REPO/'outputs'/'interactive'/f'{stamp}_{mode}_{uuid.uuid4().hex[:8]}'
    commands, prepared = (experiment_plan(folder) if mode == 'experiment' else
                          pap_plan() if mode == 'pap' else
                          detector_plan(folder) if mode == 'detector' else
                          threshold_plan(folder) if mode == 'threshold' else
                          evaluation_plan(folder))
    if not commands:
        print('요청한 PAP 캐시가 모두 완성돼 있습니다.')
        return
    if commands and all(item.get('cached_result') for item in commands):
        from dlm_steering.evaluation.cache import show_cached
        for item in commands:
            show_cached(item)
        return
    print('\n===== 실행 전 최종 확인 =====')
    print(f'결과 폴더: {folder}')
    for i, item in enumerate(commands, 1):
        if item.get('cached_result'):
            from dlm_steering.evaluation.cache import show_cached
            show_cached(item)
            continue
        print(f'\n[{i}] 최종 파라미터 (CLI 기본값 포함)')
        print(json.dumps(item['parameters'], ensure_ascii=False, indent=2))
        print(format_command(item))
    if ask('이 설정으로 실행하려면 yes 입력', 'yes').lower() != 'yes':
        print('취소했습니다. 실행하거나 결과 파일을 만들지 않았습니다.')
        return
    execute(folder, commands, prepared)
