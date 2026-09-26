"""The box on its head: FLIP SCREEN in the power menu turns the panel over, and the camera with it.

Sometimes the box has to stand upside down. What that takes is a 180 degree turn of the panel -
not a mirror, which was tried on the glass and reads as mirror writing - done by the compositor
with ``wlr-randr``, which needs the Wayland socket and no privilege at all. The camera module is
the other half: it went into the case upside down and is corrected by ``--rotation 180``, so with
the box turned over that correction comes off (:mod:`cyclops.webcam`). The screen recording is
the third: ``wf-recorder`` hands over the physical framebuffer, so :mod:`cyclops.screen` turns
it back.

A note in ``~/.cache``, the same pair as :mod:`cyclops.steady`, so the choice outlives a reboot.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .config import FLIP_FILE

WLR_RANDR_TIMEOUT_S = 5.0  # a listing or a turn is instant; this is for a compositor that hangs

_ON = {"on", "1", "true", "yes"}


def enabled(path: Path | None = None) -> bool:
    """Whether the box is meant to be upside down. Off unless the note says on."""
    try:
        return (path or FLIP_FILE).read_text().strip().lower() in _ON
    except OSError:
        return False


def request(on: bool, path: Path | None = None) -> bool:
    """Write the note down, and return what was written."""
    path = path or FLIP_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("on\n" if on else "off\n")
    return on


def connector(listing: str) -> str | None:
    """The panel's output name out of ``wlr-randr``'s listing: the first enabled one.

    Not baked in, because it is not stable - fitting the Camera Module renumbered DSI-1 to DSI-2.
    Output blocks start in column 0; their properties are indented under them.
    """
    name, first = None, None
    for line in listing.splitlines():
        if line and not line[0].isspace():
            name = line.split()[0]
            first = first or name
        elif name and line.strip().lower() == "enabled: yes":
            return name
    return first


def _wlr_randr(args: list[str]) -> tuple[int, str]:
    try:
        done = subprocess.run(["wlr-randr", *args], capture_output=True, text=True,
                              timeout=WLR_RANDR_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, str(exc)
    return done.returncode, done.stdout + done.stderr


def apply(on: bool) -> bool:
    """Turn the panel over, or back. Blocking; never call it from the render loop."""
    code, listing = _wlr_randr([])
    name = connector(listing) if code == 0 else None
    if name is None:
        print(f"· flip: no output to turn ({listing.strip()[:80] or 'no wlr-randr'})", flush=True)
        return False
    code, said = _wlr_randr(["--output", name, "--transform", "180" if on else "normal"])
    if code != 0:
        print(f"· flip: wlr-randr refused {name}: {said.strip()[:80]}", flush=True)
        return False
    print(f"· flip: {name} {'upside down' if on else 'right way up'}", flush=True)
    return True
