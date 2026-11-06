"""Check for arc-eng/hq#609: what a noisy room costs the idle ear, in real time, on the Pi.

Feeds the real Ear a loop of recordings 20 ms at a time with a stand-in for the mic, so the second
check keeps being woken the way sound in the room would. Prints this process's share of a core,
plus the board's temperature and fan, every 30 s. No session code is loaded, nothing is played out
loud, and the network is refused, so it can't file anything or reach anyone.

    uv run python tests/acceptance/hq_609_idle_cost.py <seconds> <sound> [<sound> ...]
"""

from __future__ import annotations

import glob
import socket
import subprocess
import sys
import time
from pathlib import Path

from cyclops import ear
from cyclops.audio import BLOCK_FRAMES, SAMPLE_RATE

SIZE = BLOCK_FRAMES * 2
BLOCK_S = BLOCK_FRAMES / SAMPLE_RATE


class NoMic:
    def __init__(self, **kwargs) -> None:
        pass

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


TRIED_NETWORK: list[tuple] = []


def refuse_network(*args, **kwargs):
    # Noted as well as raised: the ear's fallback would swallow the error and hide it.
    TRIED_NETWORK.append(args)
    raise RuntimeError("something in the ear tried to open a network connection")


def board() -> str:
    temp = int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
    fans = glob.glob("/sys/devices/platform/cooling_fan/hwmon/*/fan1_input")
    fan = Path(fans[0]).read_text().strip() if fans else "?"
    return f"{temp:.1f} C, fan {fan} rpm"


def main() -> None:
    loaded = {"cyclops.session", "cyclops.agent", "cyclops.ui", "cyclops.kiosk"} & set(sys.modules)
    assert not loaded, f"session code loaded: {loaded}"
    socket.socket = refuse_network
    ear.sd.RawInputStream = NoMic
    ear.pin_input_source = lambda device: None
    seconds = float(sys.argv[1])
    pcm = b"".join(
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", p, "-ac", "1", "-ar", str(SAMPLE_RATE),
             "-f", "s16le", "-"], capture_output=True, check=True,
        ).stdout for p in sys.argv[2:]
    )
    blocks = [pcm[i:i + SIZE] for i in range(0, len(pcm) - SIZE + 1, SIZE)]
    made = ear.Ear(gain_db=0.0, compress=True)
    made.start()
    print(f"before: {board()}", flush=True)
    begun = next_tick = time.monotonic()
    cpu_mark, wall_mark, attempts_mark, n = time.process_time(), begun, 0, 0
    report_at = begun + 30
    starts = 0
    while time.monotonic() - begun < seconds:
        made._callback(blocks[n % len(blocks)], BLOCK_FRAMES, None, None)
        n += 1
        if made.heard.is_set():  # a false start: count it and go on listening with a fresh ear
            starts += 1
            made.close()
            made = ear.Ear(gain_db=0.0, compress=True)
            made.start()
            attempts_mark = 0
        next_tick += BLOCK_S
        time.sleep(max(0.0, next_tick - time.monotonic()))
        now = time.monotonic()
        if now >= report_at:
            cpu = (time.process_time() - cpu_mark) / (now - wall_mark)
            print(f"{now - begun:4.0f} s: {100 * cpu:.1f}% of a core, {board()}, "
                  f"{made._attempts - attempts_mark} checks woken, {starts} false starts",
                  flush=True)
            cpu_mark, wall_mark, attempts_mark = time.process_time(), now, made._attempts
            report_at = now + 30
    made.close()
    assert not TRIED_NETWORK, f"tried the network: {TRIED_NETWORK}"
    print("no network attempts", flush=True)


if __name__ == "__main__":
    main()
