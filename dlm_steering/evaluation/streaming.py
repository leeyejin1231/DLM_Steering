"""Shared result streaming, resume identity, and grading concurrency."""
import json
import time
from pathlib import Path
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from dlm_steering.runtime.progress import task_progress


def _item_key(item):
    """Resume identity of one item.

    The row index is the identity carried through from exp.py. Response TEXT
    must never be used here: distinct prompts routinely produce byte-identical
    responses -- refusals above all -- so keying on it silently drops every
    duplicate from a resumed run and quietly shrinks the metric denominators.
    Inputs without an index fall back to the text.
    """
    index = item.get("index")
    if index is None:
        return ("response", item.get("response", ""))
    return ("index", index)


def _resume_stream(output_path, items):
    """(prior results, done keys, append handle) -- JSONL streaming + resume.

    Only prior rows belonging to `items` are carried over, so a stale JSONL
    left by an earlier run over a different --in cannot leak foreign rows into
    this run's results, and a twice-resumed file cannot double-count a row.
    """
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
                        or any(item.get(field) != wanted[key].get(field)
                               for field in ("prompt", "response", "evaluation_scope"))):
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
    """Resume, grade whatever is left, stream each record out as it lands.

    The three graders differ only in `grade`, which maps a list of items to
    their graded records in the same order.

    chunk:   how many items `grade` gets at once; >1 only for a grader that
             batches (see LlamaGuard4.batch_size).
    order:   sort key applied to the pending items before they are chunked. A
             batching grader passes a length estimate, so each batch is padded
             to roughly one length -- that is most of the speedup, since a
             batch runs until its longest member finishes.
    workers: 0 grades inline, which is right for a local GPU model -- one call
             already saturates it. Higher fans chunks out over a thread pool,
             which is what the ollama-backed graders want. Records are appended
             as they complete, so an interrupted run resumes from the last one
             finished rather than the last one submitted.
    """
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
        # Effective concurrency: how many requests were really in flight. It
        # approaches `workers` while the server keeps up and flattens out once
        # the server is the bottleneck -- the number to read before raising
        # --workers, and the reason it is printed rather than guessed at.
        print(f"  [{desc or 'graded'}] {len(chunks)} requests in {wall:.0f}s, "
              f"{busy / len(chunks):.1f}s each, effective concurrency "
              f"{busy / wall:.1f} of {workers} workers")
    return results
