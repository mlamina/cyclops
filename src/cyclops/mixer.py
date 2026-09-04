"""The speaker's one volume knob, and the note the admin page leaves about it.

There is exactly one place the output volume lives: the sink's own control. Asking the sink
rather than the card is what makes that true on either box - the Pi's I2S amplifier exposes no
hardware mixer at all, so PipeWire scales the samples itself and ``amixer`` has nothing to show,
while a device with a real hardware control is driven directly instead. Both answer ``pactl``.
So this module never stacks a second gain of its own, and cyclops' per-buffer gain
(:attr:`cyclops.config.Settings.volume`) stays at unity.

Two halves, because two processes are involved. ``cyclops-admin`` serves the page but is
hardened with ``PrivateDevices=yes`` - it has no ``/dev/snd`` at all, and it starts at boot
long before there is a user session to reach PipeWire through - so all it does is write the
level it wants to :data:`~cyclops.config.VOLUME_FILE`. The kiosk lives inside the desktop
session and applies it. Same division of labour as the browser-close flag: the page asks, the
process that can act acts.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .config import VOLUME_FILE

PACTL = "pactl"
SINK = "@DEFAULT_SINK@"  # follows whatever speaker is plugged in, rather than a pinned card
TIMEOUT_S = 2.0
MINIMUM, MAXIMUM = 0, 100


def clamp(percent: int) -> int:
    return max(MINIMUM, min(MAXIMUM, int(percent)))


# ---- the knob (kiosk side: needs the user session) ----


def level() -> int | None:
    """The sink's volume as a whole percent, or None where there is no PipeWire to ask."""
    out = pactl("get-sink-volume", SINK)
    if out is None:
        return None
    found = re.search(r"(\d+)%", out)  # "front-left: 52607 /  80% / -5.73 dB"
    return None if found is None else clamp(int(found.group(1)))


def set_level(percent: int) -> bool:
    """Set the sink's volume. False if there is nothing here that can be set."""
    return pactl("set-sink-volume", SINK, f"{clamp(percent)}%") is not None


def pactl(*args: str) -> str | None:
    """Run pactl, or return None on any machine or moment where that is not possible.

    Public because it is the one careful way this project talks to PipeWire, and the volume is
    not the only thing worth asking about - :mod:`cyclops.audio` uses it to list capture sources.
    """
    try:
        done = subprocess.run([PACTL, *args], capture_output=True, text=True, timeout=TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


# ---- the note (both sides) ----


def requested(path: Path = VOLUME_FILE) -> int | None:
    """The level the page last asked for, or None if nobody has ever asked."""
    try:
        return clamp(int(path.read_text().strip()))
    except (OSError, ValueError):
        return None


def request(percent: int, path: Path = VOLUME_FILE) -> int:
    """Leave the level for the kiosk to pick up, and return what was actually written."""
    wanted = clamp(percent)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{wanted}\n")
    return wanted
