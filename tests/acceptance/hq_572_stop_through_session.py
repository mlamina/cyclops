#!/usr/bin/env python3
"""hq#572: a stop mid-answer, through the real SessionController._session, with a slow close.

    uv run python tests/acceptance/hq_572_stop_through_session.py
    uv run python tests/acceptance/hq_572_stop_through_session.py --old-stop   # must fail

No network, no sound card, no records: nothing here can reach his notes. Real: SessionController,
_session's teardown order, Speaker, Microphone, Ear._callback. Stand-ins: the sound streams, the
agent (feeds an answer, then waits to be cancelled), and SessionLog (its exit holds the thread for
CLOSE_S, where a real close saves its video). hq_572_stuck_speaker.py sets up the stopped speaker by
hand; this one shows _session gets there before the slow part of the close, which is what the ear
depends on. --old-stop puts back round 2's Speaker.stop, so the check can be seen to fail.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

SCRATCH = Path(tempfile.mkdtemp(prefix="cyclops-qa-572-"))
os.environ["HOME"] = str(SCRATCH)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402

from cyclops import audio, ear, ui  # noqa: E402
from cyclops.audio import AUDIBLE_TAIL_S, BLOCK_FRAMES, BYTES_PER_FRAME  # noqa: E402
from cyclops.config import Settings  # noqa: E402

CLOSE_S = 2.0
BLOCK_S = BLOCK_FRAMES / audio.SAMPLE_RATE


class Stream:
    def __init__(self, *, callback=None, **_kw) -> None:  # noqa: ANN001, ANN003
        self.callback, self.run = callback, False

    def start(self) -> None:
        self.run = True

    def stop(self) -> None:
        self.run = False

    def close(self) -> None:
        self.run = False


class Output(Stream):
    def start(self) -> None:
        self.run = True
        threading.Thread(target=self.tick, daemon=True).start()

    def tick(self) -> None:
        while self.run:
            self.callback(bytearray(BLOCK_FRAMES * BYTES_PER_FRAME), BLOCK_FRAMES, None, None)
            time.sleep(BLOCK_S)


class Agent:
    """Enough of VoiceAgent for _session: an answer far longer than the test, then a cancel."""

    def __init__(self, _live, *, mic, speaker, guard, ear, earlier) -> None:  # noqa: ANN001
        self.speaker, self.spoken = speaker, []
        self.ready = threading.Event()
        self.activity = ""
        self.tool_active = self.search_active = self.drawing_active = False
        self.on_end = None

    def local_barge_in(self, *_a) -> None:  # noqa: ANN002
        pass

    async def run(self) -> None:
        self.speaker.feed(b"\x10\x10" * BLOCK_FRAMES * 3000)  # a minute of answer
        self.ready.set()
        await asyncio.Event().wait()


class Log:
    """SessionLog's shape: its exit is where a real close spends its time, saving the video."""

    entered = threading.Event()
    closing_from = [0.0]

    def __init__(self, *_a, **_k) -> None:  # noqa: ANN002, ANN003
        pass

    def __enter__(self) -> Log:
        Log.entered.set()
        return self

    def __exit__(self, *_a) -> None:  # noqa: ANN002
        Log.closing_from[0] = time.monotonic()
        time.sleep(CLOSE_S)


def main() -> None:
    audio.sd.RawOutputStream = Output
    audio.sd.RawInputStream = Stream
    ear.sd.RawInputStream = Stream
    audio.pin_input_source = lambda device: "room"
    ear.pin_input_source = lambda device: "room"
    ui.VoiceAgent = Agent
    ui.SessionLog = Log
    ui.resolve_device = lambda d: None
    ui.output_is_speaker = lambda name: True
    ui.default_output_name = lambda dev: "fake"

    if "--old-stop" in sys.argv:
        def old_stop(self) -> None:  # noqa: ANN001
            self._stream.stop()
            self._stream.close()

        audio.Speaker.stop = old_stop

    made = ui.SessionController(Settings(api_key=""), frames=None, entrypoint="qa")
    made.start()
    assert Log.entered.wait(5), f"session never came up: {made._error or made.status()}"  # noqa: SLF001
    deadline = time.monotonic() + 5
    while not made.speaking and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.5)
    print(f"1. mid-answer: speaking={made.speaking}")
    was_speaking = made.speaking

    made.stop()
    deadline = time.monotonic() + 5
    while not Log.closing_from[0] and time.monotonic() < deadline:
        time.sleep(0.005)
    into_save = time.monotonic()
    time.sleep(AUDIBLE_TAIL_S + 0.1)
    inside = time.monotonic() - into_save < CLOSE_S and made.closing
    speaker = made._speaker  # noqa: SLF001
    print(f"2. {AUDIBLE_TAIL_S + 0.1:.1f} s into the slow part of the close:"
          f" closing={made.closing} speaking={made.speaking}"
          f" buffered={len(speaker._buffer) if speaker else 'gone'}")  # noqa: SLF001
    stuck = made.speaking

    listening = ear.Ear(device=None, gain_db=0.0, compress=False)
    listening.loud = lambda: made.speaking
    words = np.full((BLOCK_FRAMES, 1), 3000, dtype=np.int16)
    for _ in range(25):  # half a second of him talking, still inside the close
        listening._callback(words, BLOCK_FRAMES, None, None)  # noqa: SLF001
    kept = len(listening.take())
    listening.close()
    print(f"3. half a second of him talking in the close: {kept} bytes kept")
    made.join(CLOSE_S + 5)

    ok = was_speaking and inside and not stuck and kept > 0
    print("PASS: the ear hears him while the close saves" if ok
          else "FAIL: after a stop mid-answer, what he says during the close is dropped")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
