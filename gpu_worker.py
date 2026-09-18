"""Persistent GPU workers pinned before any torch/Transformers import."""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


class GPUJobPool:
    """One model per subprocess, with a shared queue of available workers."""

    def __init__(self, gpus):
        self.gpus = gpus
        self.processes = []
        self.available = queue.Queue()

    def __enter__(self):
        try:
            for gpu in self.gpus:
                process = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve())],
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu),
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    text=True, bufsize=1,
                )
                self.processes.append(process)
                self.available.put(process)
            self.executor = ThreadPoolExecutor(max_workers=len(self.processes))
        except BaseException:
            for process in self.processes:
                process.kill()
                process.wait()
            raise
        return self

    def submit(self, kind, argv):
        return self.executor.submit(self._run, kind, argv)

    def _run(self, kind, argv):
        process = self.available.get()
        try:
            process.stdin.write(json.dumps([kind, argv]) + '\n')
            process.stdin.flush()
            response = process.stdout.readline()
            if not response:
                raise RuntimeError(f'GPU worker {process.pid} exited; inspect job log and stderr')
            return json.loads(response)
        finally:
            self.available.put(process)

    def __exit__(self, exc_type, exc, tb):
        if exc_type:
            self.executor.shutdown(wait=False, cancel_futures=True)
            for process in self.processes:
                if process.poll() is None:
                    process.terminate()
            for process in self.processes:
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        self.executor.shutdown(wait=True, cancel_futures=True)
        for process in self.processes:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            process.stdout.close()


def main():
    # The parent supplied CUDA_VISIBLE_DEVICES at process creation. Importing
    # Transformers may query/cache device counts, so it must happen after that.
    from common import _run_persistent_job
    for line in sys.stdin:
        kind, argv = json.loads(line)
        record = _run_persistent_job(kind, argv)
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
