"""May you talk over Cyclops, and the note the settings screen leaves about it.

Barge-in is the thing that lets you cut in mid-sentence over an open speaker: the
:class:`~cyclops.audio.EchoGuard` listens for a mic that is clearly louder than the echo of
what is playing and stops the playback there. Its judgement is a guess about a room, so it is
sometimes wrong in the one way that is worst to be wrong - Cyclops hears its own voice as you
and cuts itself off - and this is the switch that turns it off. With it off the mic is simply
shut while Cyclops is speaking: you wait for it to finish, and nothing can interrupt it.

Two halves, exactly like :mod:`cyclops.mixer`. The settings screen writes down what it wants;
the process holding the microphone picks it up. The page cannot reach into a live session any
more than it can reach the speaker's mixer, and the file is what they agree through.
"""

from __future__ import annotations

from pathlib import Path

from .config import BARGE_IN_FILE, Settings

_ON = {"on", "1", "true", "yes"}
_OFF = {"off", "0", "false", "no"}


def requested(path: Path = BARGE_IN_FILE) -> bool | None:
    """What the page last asked for, or None if nobody has ever asked.

    An unreadable or unrecognised note is the same as no note: this is a preference, and the
    worst thing it could do is take a session down over a typo in a cache file.
    """
    try:
        raw = path.read_text().strip().lower()
    except OSError:
        return None
    if raw in _ON:
        return True
    if raw in _OFF:
        return False
    return None


def request(on: bool, path: Path = BARGE_IN_FILE) -> bool:
    """Leave the answer for the session to pick up, and return what was written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("on\n" if on else "off\n")
    return on


def margin_db(settings: Settings, path: Path = BARGE_IN_FILE) -> float | None:
    """How far over the echo you must be for this session - or None, meaning "do not listen".

    The switch wins over ``CYCLOPS_BARGE_IN_DB``, because it is the one of the two you can
    reach while Cyclops is talking. The variable still sets *how hard* barge-in is when it is
    on, and still decides on a box where nobody has ever touched the switch.
    """
    wanted = requested(path)
    if wanted is None:
        return settings.barge_in_db
    if not wanted:
        return None
    # Asked back on. If ``CYCLOPS_BARGE_IN_DB=off`` was what turned it off there is no number
    # in the environment to come back to, and the built-in one is the only answer left.
    return settings.barge_in_db if settings.barge_in_db is not None else Settings.barge_in_db


def enabled(settings: Settings, path: Path = BARGE_IN_FILE) -> bool:
    """The switch's own position, for a page that has to draw it."""
    return margin_db(settings, path) is not None
