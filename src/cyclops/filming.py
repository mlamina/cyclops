"""What a session's video is a recording of, and the note the settings screen leaves about it.

There are two honest answers and the box cannot know which one you want.

**SCREEN** records the panel: the frame the kiosk composited and put on the glass, sharpened and
cropped to the panel's 5:3, with the halo, the timer, the caption, the REC tag and the tab row on
it. Watching it back is watching the session happen - you can see when Cyclops was thinking, when
you interrupted him, and where the shutter went off. It is the default because it is what you
were looking at. Including the minutes a drawing or a photo has the whole panel, which the kiosk
is not the one painting: the picture is rebuilt for the recording from what the page was handed
(:mod:`cyclops.still`), because a video that went black over the wiring diagram would be missing
the part worth watching twice.

**CAMERA** records the sensor: what the lens saw, at its own resolution and its own shape, with
nothing drawn over it and nothing trimmed off the sides. Both read the right way round - the
panel stopped mirroring - so what this one offers is the frame entire and every pixel of it spent
on the room. It is the one to reach for when the recording is evidence rather than a memory - a
part number, a wiring colour, a serial you will squint at later.

Two halves, exactly like :mod:`cyclops.barge` and :mod:`cyclops.mixer`. The settings screen writes
down what it wants; the process holding the camera and the panel picks it up. The page can no more
reach into a live recording than it can reach the speaker's mixer, and the file is what they agree
through - so a switch flipped mid-session changes the *next* one. That is not a limitation worked
around but the honest shape of the thing: an encoder is opened once, at a fixed frame size, and a
video that changed what it was of halfway through would be a worse answer than either.
"""

from __future__ import annotations

from pathlib import Path

from .config import RECORD_SOURCE_FILE, Settings

CAMERA = "camera"
SCREEN = "screen"
_SOURCES = {CAMERA, SCREEN}


def requested(path: Path = RECORD_SOURCE_FILE) -> str | None:
    """What the page last asked for, or None if nobody has ever asked.

    An unreadable or unrecognised note is the same as no note: this is a preference, and the
    worst thing it could do is cost you a recording over a typo in a cache file.
    """
    try:
        raw = path.read_text().strip().lower()
    except OSError:
        return None
    return raw if raw in _SOURCES else None


def request(source: str, path: Path = RECORD_SOURCE_FILE) -> str:
    """Leave the answer for the next session to pick up, and return what was written."""
    wanted = source.strip().lower()
    if wanted not in _SOURCES:
        raise ValueError(f"record source must be one of {sorted(_SOURCES)}, got {source!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{wanted}\n")
    return wanted


def chosen(settings: Settings, path: Path = RECORD_SOURCE_FILE) -> str:
    """Which source the next recording takes, ``CAMERA`` or ``SCREEN``.

    The switch wins over ``CYCLOPS_RECORD_SOURCE``, because it is the one of the two you can
    reach without an ssh session. The variable still decides on a box where nobody has ever
    touched the switch, which is every box on its first boot.
    """
    return requested(path) or settings.record_source


def on_screen(settings: Settings, path: Path = RECORD_SOURCE_FILE) -> bool:
    """The switch's own position, for a page that has to draw it."""
    return chosen(settings, path) == SCREEN
