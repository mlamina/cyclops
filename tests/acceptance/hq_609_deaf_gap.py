#!/usr/bin/env python3
"""hq#609: a short word starting while the ear is still deaf after a session. Does the second check
lose it, where the first check alone wouldn't?

Real Ear, real SpeechGate and SpeechCheck; blocks go straight into its callback. No session, no mic,
no model call, and HOME is scratch. Usage: hq_609_deaf_gap.py <wav> [<wav> ...]
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ["HOME"] = tempfile.mkdtemp(prefix="qa-609-deaf-")
os.environ["XDG_CACHE_HOME"] = os.path.join(os.environ["HOME"], ".cache")

from cyclops import ear  # noqa: E402
from cyclops.audio import BLOCK_FRAMES, SAMPLE_RATE  # noqa: E402

assert "cyclops.session" not in sys.modules, "session code was loaded"

SIZE = BLOCK_FRAMES * 2
BLOCK_S = BLOCK_FRAMES / SAMPLE_RATE
REAL_FEED = ear.SpeechCheck.feed


class NoMic:
    def __init__(self, **kwargs) -> None:
        pass

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


ear.sd.RawInputStream = NoMic


def load(path: str) -> bytes:
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"],
        capture_output=True,
        check=True,
    ).stdout


def starts(pcm: bytes, hold_left_s: float) -> bool:
    """The word begins with this much of the deaf hold left. Does a session start?"""
    made = ear.Ear(gain_db=0.0, compress=True)
    deaf_blocks = int(hold_left_s / BLOCK_S)
    fed = {"n": 0}
    made.deaf = lambda: fed["n"] < deaf_blocks
    made.start()
    pcm += b"\0" * int(1.0 * SAMPLE_RATE) * 2
    try:
        for i in range(0, len(pcm) - SIZE + 1, SIZE):
            made._callback(pcm[i : i + SIZE], BLOCK_FRAMES, None, None)  # noqa: SLF001
            made._jobs.join()  # noqa: SLF001
            fed["n"] += 1
            if made.heard.is_set():
                return True
    finally:
        made.close()
    return False


waves = {Path(p).stem: load(p) for p in sys.argv[1:]}
print(f"clips: {', '.join(f'{k} {len(v) / 2 / SAMPLE_RATE:.2f} s' for k, v in waves.items())}")
print("hold left at the word | started, first check alone | started, with the second check")
for hold in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
    ear.SpeechCheck.feed = lambda self, block: True
    alone = [k for k, v in waves.items() if starts(v, hold)]
    ear.SpeechCheck.feed = REAL_FEED
    both = [k for k, v in waves.items() if starts(v, hold)]
    print(f"{hold:.1f} s | {len(alone)} of {len(waves)} | {len(both)} of {len(waves)}  lost: {sorted(set(alone) - set(both))}", flush=True)
print("DEAF-DONE")
