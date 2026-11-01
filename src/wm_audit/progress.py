"""Live stderr-only progress, including heartbeats for uncounted operations.

No CUDA queries, synchronization, RNG calls or result-file writes belong here.
WM_AUDIT_PROGRESS=0 disables project progress (not upstream library output).
"""
from contextlib import contextmanager
import os
import sys
from threading import Event, Thread
from time import monotonic

from tqdm import tqdm


def enabled():
    return os.environ.get("WM_AUDIT_PROGRESS", "1") != "0"


def note(message):
    if enabled():
        tqdm.write(f"[wm-audit] {message}", file=sys.stderr)
        sys.stderr.flush()


@contextmanager
def stage(label, *, heartbeat=30.0):
    """Report start/end/failure; heartbeat means alive, NOT measurable progress."""
    if not enabled():
        yield
        return
    begin = monotonic()
    stop = Event()
    note(f"START {label}")

    def pulse():
        while not stop.wait(heartbeat):
            note(f"RUNNING {label} | elapsed {monotonic() - begin:.0f}s | waiting for next progress update")

    worker = Thread(target=pulse, name="wm-audit-progress", daemon=True)
    worker.start()
    outcome = "DONE"
    try:
        yield
    except BaseException:
        outcome = "FAILED/INTERRUPTED"
        raise
    finally:
        stop.set()
        worker.join()
        note(f"{outcome} {label} | elapsed {monotonic() - begin:.1f}s")


def progress(items, *, desc, total=None, unit="item"):
    """Count only completed items; preserve iterable ordering and exceptions.

TTY uses tqdm; redirected/notebook streams use throttled newline snapshots
so saved logs remain readable. ETA is based on completed items, not a promise.
"""
    if not enabled():
        yield from items
        return
    if total is None:
        total = len(items)
    if sys.stderr.isatty():
        with tqdm(total=total, desc=desc, unit=unit, file=sys.stderr,
                  mininterval=2, dynamic_ncols=True, ascii=True) as bar:
            for item in items:
                yield item
                bar.update()
        return
    begin = last = monotonic()
    count = 0

    def snapshot():
        elapsed = monotonic() - begin
        fraction = count / total if total else 1.0
        filled = int(20 * fraction)
        eta = f"{elapsed * (total - count) / count:.0f}s" if count else "pending"
        note(f"{desc} [{'#' * filled}{'-' * (20 - filled)}] "
             f"{count}/{total} {unit} ({fraction:.0%}) | elapsed {elapsed:.1f}s | ETA {eta}")

    snapshot()
    for item in items:
        yield item
        count += 1
        now = monotonic()
        if count == total or now - last >= 5:
            snapshot()
            last = now