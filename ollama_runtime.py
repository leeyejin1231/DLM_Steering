"""Private local Ollama servers for GPU evaluation, without podman."""
import fcntl
import json
import os
from pathlib import Path
import queue
import shutil
import socket
import subprocess
import threading
import time
import urllib.request

REPO = Path(__file__).resolve().parent


def runtime_paths():
    configured = os.environ.get('OLLAMA_BIN')
    candidates = ([Path(configured)] if configured else [])
    executable = shutil.which('ollama')
    if executable:
        candidates.append(Path(executable))
    candidates.extend(sorted((REPO/'outputs').glob('*/runtime/bin/ollama')))
    binary = next((p.resolve() for p in candidates if p.is_file() and os.access(p, os.X_OK)), None)
    if binary is None:
        raise ValueError('Ollama 실행 파일을 찾지 못했습니다. ollama를 설치하거나 OLLAMA_BIN을 지정하세요.')
    if os.environ.get('OLLAMA_MODELS'):
        models = Path(os.environ['OLLAMA_MODELS']).expanduser()
    else:
        candidates = [Path.home()/'.ollama/models', binary.parent.parent/'models']
        candidates.extend(sorted((REPO/'outputs').glob('*/models')))
        models = next((p for p in candidates if (p/'manifests/registry.ollama.ai/library/gpt-oss/20b').is_file()),
                      Path.home()/'.ollama/models')
    return binary, models.resolve()


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


class GPUDiscoveryError(RuntimeError):
    pass


class StartupCancelled(RuntimeError):
    pass


class OllamaServer:
    def __init__(self, gpu, output, model='gpt-oss:20b', workers=4, wanted=None):
        self.gpu, self.output, self.model, self.workers = str(gpu), Path(output), model, workers
        self.process = None
        self.log = None
        # Asked once this server reaches the front of the startup queue; a pool
        # whose work ran out while it waited answers False (see OllamaServerPool).
        self.wanted = wanted

    def request(self, route, body=None, timeout=3):
        encoded = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(f'http://127.0.0.1:{self.port}{route}', data=encoded,
                                         headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    def __enter__(self):
        # Ollama's CUDA discovery subprocess can crash/time out when several
        # servers initialize together. Serialize startup, not actual grading.
        lock_path = REPO/'outputs'/'.ollama-startup.lock'
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        print(f'GPU {self.gpu}: 서버 초기화 순서 대기 중…', flush=True)
        with lock_path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.wanted is not None and not self.wanted():
                raise StartupCancelled(f'GPU {self.gpu}: 남은 작업이 없어 서버를 띄우지 않습니다.')
            for attempt in range(1, 4):
                try:
                    return self._start()
                except GPUDiscoveryError:
                    if attempt == 3:
                        raise
                    print(f'GPU {self.gpu}: CUDA 탐색 실패, 서버 초기화 재시도 {attempt}/2', flush=True)

    def _start(self):
        binary, models = runtime_paths()
        self.port = free_port()
        self.output.parent.mkdir(parents=True, exist_ok=True)
        log_path = self.output.with_suffix('.ollama.log')
        self.log = log_path.open('a')
        log_offset = log_path.stat().st_size
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=self.gpu,
                   OLLAMA_HOST=f'127.0.0.1:{self.port}', OLLAMA_MODELS=str(models),
                   OLLAMA_NUM_PARALLEL=str(self.workers), OLLAMA_CONTEXT_LENGTH='16384',
                   OLLAMA_FLASH_ATTENTION='1', OLLAMA_KEEP_ALIVE='30m', OLLAMA_NO_CLOUD='1')
        print(f'GPU {self.gpu}: 평가 서버 준비 중…', flush=True)
        try:
            self.process = subprocess.Popen([str(binary), 'serve'], env=env,
                                            stdout=self.log, stderr=subprocess.STDOUT)
            deadline = time.monotonic()+120
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f'Ollama 시작 실패: {log_path}')
                try:
                    tags = self.request('/api/tags')['models']
                    break
                except (OSError, ValueError):
                    time.sleep(1)
            else:
                raise RuntimeError(f'Ollama 시작 시간 초과: {log_path}')
            with log_path.open() as stream:
                stream.seek(log_offset)
                startup_log = stream.read()
            if 'library=CUDA' not in startup_log:
                raise GPUDiscoveryError(f'GPU {self.gpu}: CUDA 탐색 실패. {log_path}')
            if not any(m['name'] == self.model for m in tags):
                print(f'{self.model} 모델 다운로드 중…', flush=True)
                subprocess.run([str(binary),'pull',self.model],env=env,check=True)
            print(f'GPU {self.gpu}: {self.model} 로딩 중…', flush=True)
            self.request('/api/generate', {'model':self.model,'prompt':'','stream':False,'keep_alive':'30m'}, timeout=600)
            loaded = self.request('/api/ps')['models']
            if not loaded or any(m.get('size_vram',0) < m['size'] for m in loaded):
                raise RuntimeError(f'GPU {self.gpu}: 모델 전체가 GPU에 적재되지 않았습니다. {log_path}')
            self.output.with_suffix('.ollama.json').write_text(json.dumps(
                {'gpu':self.gpu,'port':self.port,'pid':self.process.pid,'binary':str(binary),
                 'models':str(models),'loaded':loaded},ensure_ascii=False,indent=2))
            print(f'GPU {self.gpu}: 평가 준비 완료', flush=True)
            return self
        except BaseException:
            self.__exit__(None,None,None)
            raise

    def __exit__(self, *exc):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.log is not None:
            self.log.close()


class OllamaServerPool:
    """One server per GPU feeding a single work queue, usable as they come up.

    Server startup has to be serialized (see OllamaServer.__enter__) and costs
    ~20s per GPU, while a server grades about one item per second however many
    requests it is given. Static shards therefore made an 8-GPU run wait ~150s
    for its last server to grade a dozen items in 13s. Here every server that
    is ready starts taking items at once, later ones join as they come up, and
    one still queued when the remaining items are already in flight is never
    started. A GPU whose server fails to start only costs its share of the
    throughput instead of failing the whole run.

    grade(chunk) is the _run_graded callback; run it with workers=capacity.
    """

    def __init__(self, gpus, output, model, workers, make_grader, n_items):
        self.gpus, self.output, self.model = list(gpus), Path(output), model
        self.workers, self.make_grader = workers, make_grader
        self.capacity = len(self.gpus) * workers
        self.slots = queue.Queue()
        self.lock = threading.Lock()
        self.outstanding, self.ready, self.failed = n_items, 0, 0
        self.closing = False
        self.servers, self.threads = [], []

    def _wanted(self):
        with self.lock:
            return not self.closing and self.outstanding > self.ready

    def _serve(self, index, gpu):
        part = self.output.with_name(f'{self.output.stem}.gpu{index}{self.output.suffix}')
        server = OllamaServer(gpu, part, self.model, self.workers, wanted=self._wanted)
        with self.lock:
            self.servers.append(server)
        try:
            server.__enter__()
            grader = self.make_grader(server.port)
        except BaseException as exc:
            server.__exit__(None, None, None)
            if not isinstance(exc, StartupCancelled) and not self.closing:
                print(f'GPU {gpu}: 평가 서버를 사용할 수 없습니다: {exc}', flush=True)
            with self.lock:
                self.failed += 1
                nothing_left = self.failed == len(self.gpus)
            if nothing_left:    # unblock every waiting grade() so the run fails
                for _ in range(self.capacity):
                    self.slots.put(None)
            return
        with self.lock:
            self.ready += self.workers
        for _ in range(self.workers):
            self.slots.put(grader)

    def __enter__(self):
        for index, gpu in enumerate(self.gpus):
            thread = threading.Thread(target=self._serve, args=(index, gpu), daemon=True)
            thread.start()
            self.threads.append(thread)
        return self

    def grade(self, chunk):
        grader = self.slots.get()
        try:
            if grader is None:
                raise RuntimeError('사용 가능한 Ollama 평가 서버가 없습니다. *.ollama.log를 확인하세요.')
            return grader._grade(chunk)
        finally:
            self.slots.put(grader)
            with self.lock:
                self.outstanding -= len(chunk)

    def __exit__(self, *exc):
        with self.lock:
            self.closing = True
            servers = list(self.servers)
        for server in servers:      # also aborts a startup still in progress
            if server.process is not None and server.process.poll() is None:
                server.process.terminate()
        for thread in self.threads:
            thread.join()
        for server in servers:
            server.__exit__(None, None, None)
