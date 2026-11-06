#!/usr/bin/env python3
"""hq#609: does the second check give back what it takes, over days of a noisy room?

SpeechCheck loads a fresh detector for every attempt, so a door knock loads a model. And the ggml
wheel brings its own OpenMP, which keeps a thread pool per calling thread, and every Ear has a new
worker thread. This runs both many times, and prints memory, threads and file descriptors.

Real SpeechCheck; no session code, no mic, no model call. HOME is scratch.
Usage: hq_609_churn.py <attempts> <worker threads>
"""

from __future__ import annotations

import gc
import os
import statistics
import sys
import tempfile
import threading
import time

os.environ["HOME"] = tempfile.mkdtemp(prefix="qa-609-churn-")
os.environ["XDG_CACHE_HOME"] = os.path.join(os.environ["HOME"], ".cache")

import numpy as np  # noqa: E402

from cyclops import ear  # noqa: E402

assert "cyclops.session" not in sys.modules, "session code was loaded"

TONE = (np.sin(np.arange(ear.BLOCK_FRAMES) / 8.0) * 8000).astype(np.int16).tobytes()
BLOCKS = ear.CHECK_BLOCKS  # an attempt that runs its full length, the worst case


def status() -> str:
    fields = dict(
        line.split(":", 1) for line in open("/proc/self/status").read().splitlines() if ":" in line
    )
    fds = len(os.listdir("/proc/self/fd"))
    return f"rss {int(fields['VmRSS'].split()[0]) / 1024:.1f} MB, threads {fields['Threads'].strip()}, fds {fds}"


attempts, workers = int(sys.argv[1]), int(sys.argv[2])
print(f"OMP_WAIT_POLICY={os.environ.get('OMP_WAIT_POLICY')} blocks per attempt={BLOCKS}", flush=True)
print(f"start: {status()}", flush=True)

loads: list[float] = []
step = max(1, attempts // 5)
for n in range(1, attempts + 1):
    begun = time.perf_counter()
    check = ear.SpeechCheck()
    loads.append(time.perf_counter() - begun)
    for _ in range(BLOCKS):
        check.feed(TONE)
    del check
    if n % step == 0:
        gc.collect()
        print(
            f"{n} attempts: {status()}, load median {statistics.median(loads) * 1000:.1f} ms, "
            f"max {max(loads) * 1000:.1f} ms",
            flush=True,
        )
        loads.clear()


def one_ear() -> None:
    check = ear.SpeechCheck()
    for _ in range(5):
        check.feed(TONE)


step = max(1, workers // 5)
for n in range(1, workers + 1):
    worker = threading.Thread(target=one_ear)
    worker.start()
    worker.join()
    if n % step == 0:
        gc.collect()
        print(f"{n} worker threads, one after another: {status()}", flush=True)

print("CHURN-DONE", flush=True)
