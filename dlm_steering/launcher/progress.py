"""Read experiment artifacts to display progress without GPU synchronization."""
import codecs
import json
from pathlib import Path
import re
import selectors
import sys
import time
from tqdm import tqdm


class Progress:
    def __init__(self, item, number, total):
        self.item, self.number, self.total = item, number, total
        self.argv = item['argv']
        self.output = Path(self.option('--out'))
        self.script = Path(self.argv[1]).name
        self.started = time.monotonic()
        self.cache = {}
        self.live_ids = set()
        self.expected = None
        self.count = self.target_count()
        self.initial = None
        self.elapsed_label = "경과"
        self.bar = None

    def option(self, name, default=None):
        return self.argv[self.argv.index(name)+1] if name in self.argv else default

    def target_count(self):
        if self.script == 'exp.py':
            start, count = int(self.option('--start', 0)), int(self.option('--n', 20))
            self.expected = set(range(start, start+count))
            return count
        if self.script == 'pap_generate.py' or (self.script == 'interface.py' and '--pap-generate' in self.argv):
            from dlm_steering.runtime.data import load_prompts
            rows = load_prompts(self.option('--source'))
            self.expected = {r['index'] for r in rows}
            return len(rows)
        if self.option('--in'):
            from dlm_steering.evaluation.attacks import generation_items
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
        if self.option('--jsonl'):
            paths.add(Path(self.option('--jsonl')))
        paths.update(out.parent.glob(out.stem+'.part*.json'))
        paths.update(out.parent.glob(out.stem+'.part*.jsonl'))
        paths.update(out.parent.glob(out.stem+'.part*.log'))
        chunk_dir = out.parent/'.parts'/out.stem
        paths.update(chunk_dir.glob('chunk*.json'))
        paths.update(chunk_dir.glob('chunk*.log'))
        if self.script == 'interface.py' and '--pap-generate' in self.argv:
            seed = self.option('--seed')
            paths.update((out.parent/'.shards'/f'seed{seed}').glob('gpu*.json'))
        ids = set(self.live_ids)
        for path in paths:
            ids.update(self.read_ids(path))
        return len(ids if self.expected is None else ids & self.expected)

    def show(self, status='진행 중'):
        done = self.completed()
        if self.bar is None:
            label = {'exp.py':'응답 생성', 'pap_generate.py':'PAP 프롬프트 준비',
                     'eval_llamaguard.py':'LG4 채점', 'run_sr_eval.py':'GPT-OSS 채점'}.get(self.script,self.script)
            if self.script == 'interface.py' and '--pap-generate' in self.argv:
                label = 'PAP 프롬프트 준비'
            cfg = self.item['parameters']
            details = ', '.join(f'{k}={cfg[k]}' for k in ('source','attack','defense','seed')
                                if isinstance(cfg.get(k), (str,int)))
            if details:
                tqdm.write(details, file=sys.stdout)
            self.initial = done
            desc = f'[작업 {self.number}/{self.total}] {label}'
            if self.elapsed_label == '관찰 경과':
                desc += ' (관찰)'
            self.bar = tqdm(total=self.count, initial=done, desc=desc,
                            unit='건', dynamic_ncols=True, file=sys.stdout)
        if done == 0 and status == '진행 중':
            status = '모델 준비 또는 첫 결과 대기'
        self.bar.set_postfix_str(status, refresh=False)
        self.bar.update(max(0, done - self.bar.n))
        self.bar.refresh()

    def close(self):
        if self.bar is not None:
            self.bar.close()


def stream_process(process, log, progress, interval=5):
    """Render one bar while retaining the child's complete output in its log."""
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    pending = ''

    def display(text, final=False):
        nonlocal pending
        pending += text
        end = len(pending) if final else pending.rfind('\n') + 1
        if end:
            for line in pending[:end].splitlines(keepends=True):
                if progress.script == 'exp.py':
                    match = re.match(r'^\[\d+/\d+\] idx=(\d+)\s', line)
                    if match:
                        progress.live_ids.add(int(match[1]))
                        continue
                    if re.match(r'^done .+ on GPU .+\s*$', line):
                        continue
                tqdm.write(line, end='' if line.endswith(('\n', '\r')) else '\n', file=sys.stdout)
            pending = pending[end:]

    progress.show()
    deadline = time.monotonic()+interval
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while selector.get_map():
                for key, _ in selector.select(timeout=max(0, deadline-time.monotonic())):
                    chunk = key.fileobj.read1(65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                    text = decoder.decode(chunk, final=not chunk)
                    log.write(text)
                    log.flush()
                    display(text, final=not chunk)
                if time.monotonic() >= deadline:
                    progress.show()
                    deadline = time.monotonic()+interval
        code = process.wait()
        progress.show('완료' if code == 0 else f'실패 (종료 코드 {code})')
        return code
    finally:
        display('', final=True)
        progress.close()


def watch(folder, interval=5):
    """Attach a read-only display to an already running interface experiment."""
    path = Path(folder).expanduser().resolve()/'plan.json'
    current = None
    progress = None
    try:
        while True:
            plan = json.loads(path.read_text())
            commands = plan['commands']
            index = next((i for i, c in enumerate(commands) if c.get('status') != 'complete'), len(commands)-1)
            if index != current:
                if progress is not None:
                    progress.show('완료')
                    progress.close()
                current = index
                progress = Progress(commands[index], index+1, len(commands))
                progress.elapsed_label = '관찰 경과'
            progress.show(plan['status'])
            if plan['status'] != 'running':
                return
            time.sleep(interval)
    finally:
        if progress is not None:
            progress.close()
