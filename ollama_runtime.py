import fcntl
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import socket
import subprocess
import threading
import time
import urllib.request
import urllib.error

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
        raise ValueError('There is no Ollama execution file found. Please install ollama or specify OLLAMA_BIN.')
    if os.environ.get('OLLAMA_MODELS'):
        models = Path(os.environ['OLLAMA_MODELS']).expanduser()
    else:
        candidates = [Path.home()/'.ollama/models', binary.parent.parent/'models']
        candidates.extend(sorted((REPO/'outputs').glob('*/models')))
        models = next((p for p in candidates if (p/'manifests/registry.ollama.ai/library/gpt-oss/20b').is_file()), Path.home()/'.ollama/models')
    return binary, models.resolve()


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def ollama_up(host, timeout=3):
    try:
        with urllib.request.urlopen(f'{host}/api/tags', timeout=timeout):
            return True
    except Exception:
        return False


def start_ollama(port=50001, gpu=1, model=None, container=None, image='ollama/ollama', ready_timeout=90):
    container = container or 'ollama'
    host = f'http://localhost:{port}'
    if ollama_up(host):
        return False
    subprocess.run(['podman', 'stop', container], capture_output=True)
    subprocess.run(['podman', 'run', '--rm', '-d', '--init', '--security-opt=label=disable', '--mount=type=bind,src=.,dst=/mnt/workspace', f'--device=nvidia.com/gpu={gpu}', '--hooks-dir=/usr/share/containers/oci/hooks.d/', '-v', 'ollama:/root/.ollama', '-p', f'{port}:11434', '--name', container, image], check=True)
    deadline = time.time() + ready_timeout
    while time.time() < deadline:
        if ollama_up(host):
            break
        time.sleep(1)
    else:
        raise RuntimeError(f'ollama did not come up on :{port} within {ready_timeout}s')
    if model:
        subprocess.run(['podman', 'exec', container, 'ollama', 'pull', model], check=True)
    return True


class OllamaStartupError(RuntimeError):
    pass


class GPUDiscoveryError(OllamaStartupError):
    pass


class StartupCancelled(RuntimeError):
    pass


class OllamaServer:
    def __init__(self, gpu, output, model='gpt-oss:20b', workers=4, wanted=None, startup_limit=None):
        self.gpu, self.output, self.model, self.workers = str(gpu), Path(output), model, workers
        self.process = None
        self.log = None
        self.wanted = wanted
        self.startup_limit = startup_limit
        self.cancelled = threading.Event()

    def _check_wanted(self):
        if self.cancelled.is_set() or (self.wanted is not None and not self.wanted()):
            raise StartupCancelled(f'GPU {self.gpu}: Server initialization cancelled.')

    def request(self, route, body=None, timeout=3):
        encoded = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(f'http://127.0.0.1:{self.port}{route}', data=encoded, headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    def __enter__(self):
        lock_path = REPO/'outputs'/'.ollama-startup.lock'
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        print(f'GPU {self.gpu}: Wating for server initialization…', flush=True)
        for attempt in range(1, 4):
            self._check_wanted()
            acquired = False
            try:
                if self.startup_limit is not None:
                    while not self.startup_limit.acquire(timeout=0.2):
                        self._check_wanted()
                    acquired = True
                with lock_path.open('a') as lock:
                    while True:
                        self._check_wanted()
                        try:
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            break
                        except BlockingIOError:
                            self.cancelled.wait(0.2)
                    unlock = (lambda: fcntl.flock(lock, fcntl.LOCK_UN)) if attempt == 1 else None
                    return self._start(unlock_discovery=unlock)
            except (OllamaStartupError, urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                self._check_wanted()
                if isinstance(exc, urllib.error.HTTPError) and exc.code < 500:
                    raise
                if attempt == 3:
                    raise
                print(f'GPU {self.gpu}: Failed to initialize, retrying serial initialization {attempt}/2: {exc}', flush=True)
            finally:
                if acquired:
                    self.startup_limit.release()
            self.cancelled.wait(attempt)

    def _start(self, unlock_discovery=None):
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
        print(f'GPU {self.gpu}: Preparing evaluation server…', flush=True)
        try:
            self.process = subprocess.Popen([str(binary), 'serve'], env=env,
                                            stdout=self.log, stderr=subprocess.STDOUT,
                                            start_new_session=True)
            deadline = time.monotonic()+120
            while time.monotonic() < deadline:
                if self.cancelled.is_set():
                    raise StartupCancelled(f'GPU {self.gpu}: Server initialization cancelled')
                if self.process.poll() is not None:
                    raise OllamaStartupError(f'Ollama startup failed: {log_path}')
                if ollama_up(f'http://127.0.0.1:{self.port}'):
                    tags = self.request('/api/tags')['models']
                    break
                time.sleep(1)
            else:
                raise OllamaStartupError(f'Ollama startup timeout: {log_path}')
            with log_path.open() as stream:
                stream.seek(log_offset)
                startup_log = stream.read()
            if 'library=CUDA' not in startup_log:
                raise GPUDiscoveryError(f'GPU {self.gpu}: CUDA discovery failed. {log_path}')
            if unlock_discovery is not None:
                unlock_discovery()
            if not any(m['name'] == self.model for m in tags):
                print(f'{self.model} model downloading…', flush=True)
                subprocess.run([str(binary),'pull',self.model],env=env,check=True)
            print(f'GPU {self.gpu}: {self.model} loading…', flush=True)
            self.request('/api/generate', {'model':self.model,'prompt':'','stream':False,'keep_alive':'30m'}, timeout=600)
            if self.cancelled.is_set():
                raise StartupCancelled(f'GPU {self.gpu}: Server initialization cancelled')
            loaded = self.request('/api/ps')['models']
            if not loaded or any(m.get('size_vram',0) < m['size'] for m in loaded):
                raise RuntimeError(f'GPU {self.gpu}: Model not fully loaded on GPU. {log_path}')
            self.output.with_suffix('.ollama.json').write_text(json.dumps(
                {'gpu':self.gpu,'port':self.port,'pid':self.process.pid,'binary':str(binary), 'models':str(models),'loaded':loaded},ensure_ascii=False,indent=2))
            print(f'GPU {self.gpu}: Evaluation ready', flush=True)
            return self
        except BaseException:
            self.__exit__(None,None,None)
            raise

    def _signal(self, sig):
        if self.process is not None:
            try:
                os.killpg(self.process.pid, sig)
            except ProcessLookupError:
                pass

    def cancel(self):
        self.cancelled.set()
        self._signal(signal.SIGTERM)

    def __exit__(self, *exc):
        self._signal(signal.SIGTERM)
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self._signal(signal.SIGKILL)
                self.process.wait()
        if self.log is not None:
            self.log.close()
        self.process = None
        self.log = None


class OllamaServerPool:
    def __init__(self, gpus, output, model, workers, make_grader, n_items, startup_workers=2):
        if startup_workers < 1:
            raise ValueError('startup_workers must be >= 1')
        self.gpus, self.output, self.model = list(gpus), Path(output), model
        self.workers, self.make_grader = workers, make_grader
        self.capacity = len(self.gpus) * workers
        self.slots = queue.Queue()
        self.lock = threading.Lock()
        self.outstanding, self.ready, self.failed = n_items, 0, 0
        self.closing = False
        self.servers, self.threads = [], []
        self.startup_limit = threading.BoundedSemaphore(startup_workers)

    def _wanted(self):
        with self.lock:
            return not self.closing and self.outstanding > self.ready

    def _serve(self, index, gpu):
        part = self.output.with_name(f'{self.output.stem}.gpu{index}{self.output.suffix}')
        server = OllamaServer(gpu, part, self.model, self.workers, wanted=self._wanted, startup_limit=self.startup_limit)
        with self.lock:
            self.servers.append(server)
        try:
            server.__enter__()
            grader = self.make_grader(server.port)
        except BaseException as exc:
            server.__exit__(None, None, None)
            if not isinstance(exc, StartupCancelled) and not self.closing:
                print(f'GPU {gpu}: Evaluation server is not available: {exc}', flush=True)
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
                raise RuntimeError('No available Ollama evaluation server. Please check *.ollama.log.')
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
            server.cancel()
        for thread in self.threads:
            thread.join()
        for server in servers:
            server.__exit__(None, None, None)
