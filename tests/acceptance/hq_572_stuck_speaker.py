#!/usr/bin/env python3
"""hq#572: a session stopped mid-answer leaves the ear deaf and dropping what it hears for the close.

    uv run python tests/acceptance/hq_572_stuck_speaker.py

No network, no sound card. Real: Speaker, SessionController.speaking, Ear._callback. Speaker.stop()
leaves its unplayed audio in the buffer and is_audible is True while that buffer is non-empty, so
nothing drains it. The controller keeps publishing the speaker until its thread's last moments,
which can be the whole video save.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402

from cyclops import audio, ear, ui  # noqa: E402
from cyclops.audio import BLOCK_FRAMES, AUDIBLE_TAIL_S  # noqa: E402
from cyclops.config import Settings  # noqa: E402


class Stream:
    def __init__(self, **_kw) -> None:  # noqa: ANN003
        pass

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


def main() -> None:
    audio.sd.RawOutputStream = Stream
    ear.sd.RawInputStream = Stream
    ear.pin_input_source = lambda device: "room"

    controller = ui.SessionController(Settings(api_key=""), frames=None, entrypoint="qa")
    speaker = audio.Speaker()
    speaker.feed(b"\x10\x10" * BLOCK_FRAMES * 50)  # the rest of an answer it hasn't played yet
    speaker.stop()
    controller._speaker = speaker  # noqa: SLF001 - as it is until the thread's last moments

    time.sleep(AUDIBLE_TAIL_S + 0.5)
    stuck = controller.speaking
    print(f"1. {AUDIBLE_TAIL_S + 0.5:.1f} s after speaker.stop(), with no sound coming out:"
          f" speaking={stuck}")

    listening = ear.Ear(device=None, gain_db=0.0, compress=False)
    listening.loud = lambda: controller.speaking
    words = np.full((BLOCK_FRAMES, 1), 3000, dtype=np.int16)
    for _ in range(50):  # one second of him talking, during the close
        listening._callback(words, BLOCK_FRAMES, None, None)  # noqa: SLF001
    kept = len(listening.take())
    print(f"2. one second of him talking during the close: {kept} bytes kept")
    listening.close()

    ok = not stuck and kept > 0
    print("PASS: the ear hears him once its voice has stopped" if ok
          else "FAIL: after a stop mid-answer, what he says during the close is dropped")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
