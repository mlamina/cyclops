"""Check for arc-eng/hq#609: run the real Ear over folders of recordings; does each start a session?

Uses Ear itself with a stand-in for the mic, through its real compressor, both checks and the
worker thread, block by block at full speed. Nothing but the Ear is imported from the session
code: no session is started, nothing is sent anywhere, nothing is written but stdout.

    uv run python tests/acceptance/hq_609_noise.py <folder> [<folder> ...]

Prints one line per folder (subfolders count as their own label), and the median of how long after
the first check fired the second one said yes, for what started. Needs ffmpeg for non-wav files.
"""

from __future__ import annotations

import socket
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from cyclops import ear
from cyclops.audio import BLOCK_FRAMES, SAMPLE_RATE

SIZE = BLOCK_FRAMES * 2
TAIL_S = 1.0  # quiet after each clip, so what it was deciding about gets decided


class NoMic:
    def __init__(self, **kwargs) -> None:
        pass

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


def load(path: Path) -> bytes:
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SAMPLE_RATE),
         "-f", "s16le", "-"],
        capture_output=True, check=True,
    ).stdout


def run_clip(pcm: bytes) -> float | None:
    """Seconds of sound between the first check firing and the second saying yes, or None."""
    made = ear.Ear(gain_db=0.0, compress=True)
    made.start()
    tail = b"\0" * int(TAIL_S * SAMPLE_RATE) * 2
    pcm += tail
    fired_at = None
    try:
        for n, i in enumerate(range(0, len(pcm) - SIZE + 1, SIZE)):
            made._callback(pcm[i:i + SIZE], BLOCK_FRAMES, None, None)
            made._jobs.join()
            if fired_at is None and made._attempt:
                fired_at = n
            if made.heard.is_set():
                return (n - fired_at) * BLOCK_FRAMES / SAMPLE_RATE
    finally:
        made.close()
    return None


TRIED_NETWORK: list[tuple] = []


def refuse_network(*args, **kwargs):
    # Noted as well as raised: the ear's fallback would swallow the error and hide it.
    TRIED_NETWORK.append(args)
    raise RuntimeError("something in the ear tried to open a network connection")


def main() -> None:
    # Safe on his device, and proves it each run: no session code is even loaded, so nothing can
    # be filed, and any attempt to reach a network (arc-eng/hq#609: nothing leaves) is fatal.
    loaded = {"cyclops.session", "cyclops.agent", "cyclops.ui", "cyclops.kiosk"} & set(sys.modules)
    assert not loaded, f"session code loaded: {loaded}"
    socket.socket = refuse_network
    ear.sd.RawInputStream = NoMic
    ear.pin_input_source = lambda device: None
    ear.NOISE_LOG_EVERY_S = 1e9
    started: dict[str, list[float]] = defaultdict(list)
    total: dict[str, int] = defaultdict(int)
    begun = time.perf_counter()
    for root in map(Path, sys.argv[1:]):
        for clip in sorted(p for p in root.rglob("*") if p.suffix in (".wav", ".ogg", ".flac")):
            label = clip.parent.name
            total[label] += 1
            after = run_clip(load(clip))
            if after is not None:
                started[label].append(after)
    for label in sorted(total):
        got = started[label]
        median = f", median {statistics.median(got):.2f} s after the first check" if got else ""
        print(f"{label}: {len(got)} of {total[label]} started{median}")
    print(f"({time.perf_counter() - begun:.0f} s)")
    assert not TRIED_NETWORK, f"tried the network: {TRIED_NETWORK}"
    print("no network attempts", flush=True)


if __name__ == "__main__":
    main()
