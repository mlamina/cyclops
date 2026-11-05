#!/usr/bin/env python3
"""hq#572: what the kiosk's ear sees in the last moment of a close. No network, no sound card.

    uv run python tests/acceptance/hq_572_close_window.py [closes]

Part 1 asks the real SessionController what it reports once `closing` has gone False while its
thread is still alive. Part 2 hands that state to the real Kiosk._tend_ear holding an ear that has
heard him. Part 3 measures how long the window lasts over real closes.
"""

from __future__ import annotations

import asyncio
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cyclops import kiosk as kiosk_module  # noqa: E402
from cyclops import ui  # noqa: E402
from cyclops.config import Settings  # noqa: E402


class StubEar:
    def __init__(self) -> None:
        self.heard = threading.Event()
        self.heard.set()
        self.closed = False

    def close(self) -> None:
        self.closed = True


def controller() -> ui.SessionController:
    made = ui.SessionController(Settings(api_key=""), frames=None, entrypoint="qa")

    async def forever(self, loop) -> None:  # noqa: ANN001
        await asyncio.sleep(3600)

    made._session = forever.__get__(made)  # noqa: SLF001
    return made


def part_1_and_2() -> bool:
    made = controller()
    gate = threading.Event()
    real_new_loop = asyncio.new_event_loop

    class HeldLoop(asyncio.SelectorEventLoop):
        def close(self) -> None:  # the thread's last act: hold it here so the window can be read
            gate.wait(5)
            super().close()

    asyncio.new_event_loop = HeldLoop
    try:
        made.start()
        while made._task is None:  # noqa: SLF001
            time.sleep(0.001)
        made.stop()
        while made.closing:
            time.sleep(0.001)
        state = str(made.status()["state"])
        print(f"1. closing={made.closing} running={made.running} state={state!r}"
              f" session_up={kiosk_module.session_up(state)}")

        kiosk = object.__new__(kiosk_module.Kiosk)
        kiosk.controller = made
        kiosk._cues = type("C", (), {"waiting": lambda self: 0.0})()  # noqa: SLF001
        kiosk._menu, kiosk._wifi, kiosk._online = False, None, True  # noqa: SLF001
        kiosk._pending, kiosk._pending_at = None, 0.0  # noqa: SLF001
        ear = StubEar()
        kiosk._ear = ear  # noqa: SLF001
        kiosk._ear_state, kiosk._ear_quiet_until, kiosk._ear_retry_at = "", 0.0, 0.0  # noqa: SLF001
        kiosk._tend_ear(state, time.monotonic())  # noqa: SLF001
        print(f"2. ear holding what he said, after one frame in that state: closed={ear.closed}")
        gate.set()
        made.join(5)
        return not ear.closed
    finally:
        asyncio.new_event_loop = real_new_loop


def part_3(closes: int) -> None:
    widths: list[float] = []
    for _ in range(closes):
        made = controller()
        made.start()
        while made._task is None:  # noqa: SLF001
            time.sleep(0.0005)
        made.stop()
        opened = None
        while made.running:
            if not made.closing and opened is None:
                opened = time.perf_counter()
        widths.append(time.perf_counter() - opened if opened else 0.0)
    ms = sorted(w * 1000 for w in widths)
    frame = 1000 / kiosk_module.TARGET_FPS
    print(f"3. window over {closes} closes: median {statistics.median(ms):.3f} ms,"
          f" max {ms[-1]:.3f} ms; one frame is {frame:.1f} ms,"
          f" so about {statistics.median(ms) / frame:.1%} of closes land a frame in it")


def part_4(closes: int) -> None:
    """Part 3's poller spins and competes for the GIL, which may stretch the window. This times the
    thread's last act, loop.close(), with nothing spinning: a lower bound on the window."""
    widths: list[float] = []

    class TimedLoop(asyncio.SelectorEventLoop):
        def close(self) -> None:
            began = time.perf_counter()
            super().close()
            widths.append(time.perf_counter() - began)

    real = asyncio.new_event_loop
    asyncio.new_event_loop = TimedLoop
    try:
        for _ in range(closes):
            made = controller()
            made.start()
            while made._task is None:  # noqa: SLF001
                time.sleep(0.002)
            made.stop()
            made.join(5)
    finally:
        asyncio.new_event_loop = real
    ms = sorted(w * 1000 for w in widths)
    print(f"4. loop.close() alone, nothing spinning: median {statistics.median(ms):.3f} ms,"
          f" max {ms[-1]:.3f} ms")


def main() -> None:
    held = part_1_and_2()
    part_3(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
    part_4(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
    print("PASS: the ear survives the end of a close" if held
          else "FAIL: the ear and what it heard are dropped at the end of a close")
    sys.exit(0 if held else 1)


if __name__ == "__main__":
    main()
