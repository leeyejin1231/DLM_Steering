"""Launch approved command plans and record progress and results."""
import datetime as dt
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
from dlm_steering.paths import REPO
from dlm_steering.runtime.progress import CHILD_PROGRESS_ENV
from .storage import save, WORKER_ENV


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
    env = {**WORKER_ENV, **item['env']}
    return shlex.join(['env', *[f'{k}={v}' for k, v in env.items()], *item['argv']])


def execute(folder, commands, prepared):
    from dlm_steering.launcher.progress import Progress, stream_process
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
                from dlm_steering.evaluation.cache import show_cached
                payload = show_cached(item)
                out = Path(item['argv'][item['argv'].index('--out')+1])
                save(out, payload)
                item.update(status='complete', returncode=0, reused=True)
                save(folder/'plan.json', plan)
                report(folder, plan)
                continue
            item['status'] = 'running'
            save(folder/'plan.json', plan)
            env = {**os.environ, **WORKER_ENV, 'PYTHONUNBUFFERED': '1',
                   **item['env'], **CHILD_PROGRESS_ENV}
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
                from dlm_steering.evaluation.cache import show_summary
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
