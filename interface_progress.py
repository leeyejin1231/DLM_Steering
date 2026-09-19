"""Read experiment artifacts to display progress without GPU synchronization."""
import codecs
import json
from pathlib import Path
import re
import selectors
import time


class Progress:
    def __init__(self, item, number, total):
        self.item, self.number, self.total = item, number, total
        self.argv = item['argv']
        self.output = Path(self.option('--out'))
        self.script = Path(self.argv[1]).name
        self.started = time.monotonic()
        self.cache = {}
        self.expected = None
        self.count = self.target_count()
        self.initial = None
        self.elapsed_label = "경과"

    def option(self, name, default=None):
        return self.argv[self.argv.index(name)+1] if name in self.argv else default

    def target_count(self):
        if self.script == 'exp.py':
            start, count = int(self.option('--start', 0)), int(self.option('--n', 20))
            self.expected = set(range(start, start+count))
            return count
        if self.script == 'pap_generate.py' or (self.script == 'interface.py' and '--pap-generate' in self.argv):
            from common import load_prompts
            rows = load_prompts(self.option('--source'))
            self.expected = {r['index'] for r in rows}
            return len(rows)
        if self.option('--in'):
            from attack_evaluation import generation_items
            data = json.loads(Path(self.option('--in')).read_text())
            rows = data['results']
            start = int(self.option('--start', 0))
            n = self.option('--n')
            items = generation_items(data, rows[start: start+int(n) if n else None])
            self.expected = {r['index'] for r in items}
            return len(items)
        return None

    def read_ids(self, path):
        """Cache unchanged JSON; incrementally read streaming logs and JSONL."""
        try:
            stat = path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
            old = self.cache.get(path)
            if old and old['stamp'] == stamp:
                return old['ids']
            if path.suffix == '.json':
                ids = {r['index'] for r in json.loads(path.read_text())['results']}
                state = {'stamp': stamp, 'ids': ids}
            else:
                state = old if old and old['offset'] <= stat.st_size else {'offset':0, 'ids':set()}
                with path.open('rb') as stream:
                    stream.seek(state['offset'])
                    content = stream.read()
                # Leave the final partial line for the next refresh.
                end = content.rfind(b'\n') + 1
                for line in content[:end].decode('utf-8', errors='replace').splitlines():
                    if path.suffix == '.jsonl':
                        try:
                            state['ids'].add(json.loads(line)['index'])
                        except (ValueError, KeyError):
                            continue
                    else:
                        match = re.match(r'^\[\d+/\d+\] idx=(\d+)\s', line)
                        if match:
                            state['ids'].add(int(match[1]))
                state.update(offset=state['offset']+end, stamp=stamp)
            self.cache[path] = state
            return state['ids']
        except (OSError, ValueError, KeyError, TypeError):
            return set()

    def completed(self):
        out = self.output
        paths = {out, out.with_suffix('.jsonl'), out.with_suffix('.log')}
        paths.update(out.parent.glob(out.stem+'.part*.json'))
        paths.update(out.parent.glob(out.stem+'.part*.jsonl'))
        paths.update(out.parent.glob(out.stem+'.part*.log'))
        chunk_dir = out.parent/'.parts'/out.stem
        paths.update(chunk_dir.glob('chunk*.json'))
        paths.update(chunk_dir.glob('chunk*.log'))
        if self.script == 'interface.py' and '--pap-generate' in self.argv:
            seed = self.option('--seed')
            paths.update((out.parent/'.shards'/f'seed{seed}').glob('gpu*.json'))
        ids = set()
        for path in paths:
            ids.update(self.read_ids(path))
        return len(ids if self.expected is None else ids & self.expected)

    def show(self, status='진행 중'):
        done = self.completed()
        elapsed = time.monotonic()-self.started
        if self.initial is None:
            self.initial = done
        if self.count is None:
            progress = f'{done}건'
        else:
            fraction = done/self.count if self.count else 1
            bar = '#' * int(fraction*20) + '-' * (20-int(fraction*20))
            progress = f'[{bar}] {done}/{self.count} ({fraction:.1%})'
        advanced = done-self.initial
        eta = ((self.count-done)*elapsed/advanced
               if self.count is not None and advanced > 0 and elapsed > 5 else None)
        if done == 0 and status == '진행 중':
            status = '모델 준비 또는 첫 결과 대기'
        suffix = f' | 남은 시간 약 {eta/60:.1f}분' if eta is not None else ''
        label = {'exp.py':'응답 생성', 'pap_generate.py':'PAP 프롬프트 준비',
                 'eval_llamaguard.py':'LG4 채점', 'run_sr_eval.py':'GPT-OSS 채점'}.get(self.script,self.script)
        if self.script == 'interface.py' and '--pap-generate' in self.argv:
            label = 'PAP 프롬프트 준비'
        cfg = self.item['parameters']
        details = ', '.join(f'{k}={cfg[k]}' for k in ('source','attack','defense','seed')
                            if isinstance(cfg.get(k), (str,int)))
        print(f'\n[작업 {self.number}/{self.total}] {label} {details}\n'
              f'  {progress} | {self.elapsed_label} {elapsed/60:.1f}분{suffix} | {status}', flush=True)


def stream_process(process, log, progress, interval=5):
    """Keep rendering while children are silent; preserve carriage-return output."""
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    progress.show()
    deadline = time.monotonic()+interval
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        while selector.get_map():
            for key, _ in selector.select(timeout=max(0, deadline-time.monotonic())):
                chunk = key.fileobj.read1(65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    text = decoder.decode(b'', final=True)
                else:
                    text = decoder.decode(chunk)
                print(text, end='', flush=True)
                log.write(text)
                log.flush()
            if time.monotonic() >= deadline:
                progress.show()
                deadline = time.monotonic()+interval
    code = process.wait()
    progress.show('완료' if code == 0 else f'실패 (종료 코드 {code})')
    return code


def watch(folder, interval=5):
    """Attach a read-only display to an already running interface experiment."""
    path = Path(folder).expanduser().resolve()/'plan.json'
    current = None
    progress = None
    while True:
        plan = json.loads(path.read_text())
        commands = plan['commands']
        index = next((i for i, c in enumerate(commands) if c.get('status') != 'complete'), len(commands)-1)
        if index != current:
            current = index
            progress = Progress(commands[index], index+1, len(commands))
            progress.elapsed_label = '관찰 경과'
        progress.show(plan['status'])
        if plan['status'] != 'running':
            return
        time.sleep(interval)
