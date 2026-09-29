import json
import time
from pathlib import Path
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from dlm_steering.runtime.progress import task_progress


def _item_key(item):
    index = item.get("index")
    if index is None:
        return ("response", item.get("response", ""))
    return ("index", index)


def _resume_stream(output_path, items):
    wanted = {_item_key(it): it for it in items}
    results, done, stale = [], set(), 0
    if output_path is not None and Path(output_path).exists():
        with Path(output_path).open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                item = json.loads(line)
                key = _item_key(item)
                if (key not in wanted or key in done
                        or any(item.get(field) != wanted[key].get(field) for field in ("prompt", "response", "evaluation_scope"))):
                    stale += 1
                    continue
                results.append(item)
                done.add(key)
        if done:
            print(f"  [resume] {len(done)} items already graded, skipping.")
        if stale:
            print(f"  [resume] ignored {stale} row(s) duplicated or not matching this input/scope.")
    out_file = None
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out_file = open(output_path, "a", encoding="utf-8")
    return results, done, out_file


def _append(out_file, item, lock):
    if out_file is None:
        return
    with lock:
        out_file.write(json.dumps(item, ensure_ascii=False) + "\n")
        out_file.flush()


def _run_graded(items, output_path, grade, *, workers=0, chunk=1, order=None,
                desc=""):
    results, done, out_file = _resume_stream(output_path, items)
    remaining = [it for it in items if _item_key(it) not in done]
    if order is not None:
        remaining.sort(key=order)
    chunks = [remaining[i:i + chunk] for i in range(0, len(remaining), chunk)]
    lock = Lock()
    busy = 0.0

    def timed(c):
        nonlocal busy
        started = time.time()
        try:
            return grade(c)
        finally:
            with lock:
                busy += time.time() - started

    def collect(records):
        for rec in records:
            results.append(rec)
            _append(out_file, rec, lock)

    wall = time.time()
    try:
        if workers:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(timed, c) for c in chunks]
                for future in task_progress(as_completed(futures), total=len(futures),
                                            desc=desc):
                    collect(future.result())
        else:
            for c in task_progress(chunks, total=len(chunks), desc=desc):
                collect(timed(c))
    finally:
        if out_file is not None:
            out_file.close()
    wall = time.time() - wall
    if workers and chunks:
        print(f"  [{desc or 'graded'}] {len(chunks)} requests in {wall:.0f}s, "
              f"{busy / len(chunks):.1f}s each, effective concurrency "
              f"{busy / wall:.1f} of {workers} workers")
    return results
