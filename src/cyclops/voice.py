"""Which of the ten voices Cyclops speaks in, and the note the settings screen leaves about it.

The Realtime API offers ten, and they are not variations on one another - they differ in age,
in pace, and in how much they perform the sentence. Which of them you want to be answered by is
not a thing the box can work out, and it is not a thing you can decide off a list of names
either. It is decided by hearing them, which is why the control on the panel plays one every
time it is touched (:mod:`cyclops.kiosk`, ``_sync_voice``).

``marin`` and ``cedar`` are the two the API added for itself and the two OpenAI recommends; the
other eight are older and shared with the text-to-speech endpoint. ``marin`` is the default and
the one this box has always used, so an untouched Pi sounds exactly as it did before this module
existed.

Two halves, exactly like :mod:`cyclops.filming` and :mod:`cyclops.barge`. The settings screen
writes down what it wants; the process that opens the conversation picks it up. Here the seam is
not a choice but a fact about the protocol: the voice rides in the ``session.update`` that opens
a websocket and there is no event that changes it afterwards, so a voice chosen mid-sentence is
a voice you hear on the next wake. See ``agent.py`` around ``RECONNECT_ATTEMPTS`` for why a
silent redial - which is the only thing that could apply it sooner - is deliberately impossible.
"""

from __future__ import annotations

from pathlib import Path

from .config import VOICE_FILE, Settings

# In the order the panel steps through them: the two worth reaching for first, then the eight
# older ones alphabetically. Verified against the installed SDK's own literal rather than the
# docs - ``openai.types.realtime.realtime_audio_config_output_param.VoiceID``, which is the
# thing the API will actually be holding us to.
VOICES = ("marin", "cedar", "alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse")
NAMES = frozenset(VOICES)


def requested(path: Path = VOICE_FILE) -> str | None:
    """What the page last asked for, or None if nobody has ever asked.

    An unreadable or unrecognised note is the same as no note: this is a preference, and the
    worst thing it could do is cost you a session over a typo in a cache file. A name that is
    not one of the ten is exactly that - OpenAI would refuse the session, and refusing it here
    on the box's own default is the quieter failure.
    """
    try:
        raw = path.read_text().strip().lower()
    except OSError:
        return None
    return raw if raw in NAMES else None


def request(name: str, path: Path = VOICE_FILE) -> str:
    """Leave the answer for the next session to pick up, and return what was written."""
    wanted = name.strip().lower()
    if wanted not in NAMES:
        raise ValueError(f"voice must be one of {', '.join(VOICES)}, got {name!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{wanted}\n")
    return wanted


def chosen(settings: Settings, path: Path = VOICE_FILE) -> str:
    """Which voice the next session opens with.

    The note wins over ``CYCLOPS_VOICE``, because it is the one of the two you can reach without
    an ssh session. The variable still decides on a box where nobody has ever touched the
    stepper, which is every box on its first boot.
    """
    return requested(path) or settings.voice


def cue(name: str) -> str:
    """The :mod:`cyclops.sfx` cue that is this voice saying its line.

    One function rather than an f-string at each call site, because the sample files, the table
    in ``sfx.SAMPLES`` and the thing the kiosk plays all have to agree about the spelling and
    only one of the three would say so if they stopped.
    """
    return f"voice-{name}"
