"""What was being said when a picture was taken, and whether those words have stopped arriving.

A caption written from the pixels alone calls an electric tricycle "a blue motorcycle", because
that is what it looks like. The word *trike* was said out loud twice in the ten seconds around
the shutter and is sitting in ``session.jsonl`` beside the photo. This module is the reader that
fetches it.

The filing curator has had this for months - :class:`cyclops.projects.deps.PhotoLine` builds the
same thing, and its prompt says outright that who took a picture and what was being said then
tells you more than the pixels would. The curator has the words and not the pixels; the captioner
has the pixels and not the words. This is the half that was missing.

It owns reading, and only reading: it never opens an image, never calls a model and never writes
anything. It imports ``card`` and nothing else of ours, deliberately - ``projects.deps`` reaches
``session``, which reaches ``record``, which drags OpenCV into a background service that wants to
read directories. That is the same line :mod:`cyclops.recall` holds and for the same reason, and
it is why ``PhotoLine`` is not simply imported here.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from . import card

# How far either side of the shutter to listen. Lopsided on purpose, and this is the measurement
# the whole module exists for: a photograph opens a topic more often than it closes one. In
# `2026-09-08_19-37-12_front-brake-caliper-assembly` the first photo lands at t=77.9s with only
# "Hej / Thank you / Hey Marco" before it, and the sentence that names what is in the frame
# arrives 1.7s after it. Listening only backwards - which is all a caption written five seconds
# after the shutter can do - hears the wrong half.
BEFORE_S = 60.0
AFTER_S = 120.0
TURNS = 2  # each side. More is the rest of the conversation, which the summary already covers.

MAX_SAID_CHARS = 240  # the same order as deps.MAX_SAID_CHARS; a turn, not a monologue

# How long to wait for a session that ended without ever being described. Naming happens in a
# detached child that can be killed, or offline, or looking at a conversation too short to name -
# and a picture must not stay undescribed forever because of any of them.
PATIENCE_S = 600.0

SPEECH = frozenset({"you", "cyclops"})

# What each role did, in the words somebody would use about it. deps.PhotoLine.line() says the
# same four things to the filing curator; kept in step by hand rather than shared, because
# sharing them costs the import this module is written to avoid.
TOOK = {
    "you": "they pressed the shutter",
    "cyclops": "Cyclops took it",
    "drawn": "a diagram Cyclops drew, asked for out loud and read at the bench",
    "edit": "Cyclops redrew an earlier photo: an illustration, not a record",
}

TITLE = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def session_of(images: Path) -> Path | None:
    """The session a folder of pictures belongs to, or ``None`` when it belongs to nobody.

    Structural, not by name. A project's is ``Photos/`` against a session's ``photos/``, which is
    one name on a case-insensitive filesystem; the log lying beside it is what actually decides.
    """
    folder = images.parent
    return folder if (folder / card.LOG_NAME).is_file() else None


def ready(images: Path) -> bool:
    """Have the words about these pictures stopped arriving?

    False only for a folder whose conversation is still going. Everything else is ready: a
    project's own folders were never waiting on anything, and a session that ended is not going
    to say more. The one awkward case is a session that ended and was never described - the
    naming child died, or the card was offline - and patience answers it: describe them from the
    transcript alone rather than leave them silent.
    """
    folder = session_of(images)
    if folder is None:
        return True
    if card.locked(folder):
        return False
    if card.written(folder / card.SUMMARY_NAME):
        return True
    try:
        last = (folder / card.LOG_NAME).stat().st_mtime
    except OSError:
        return True
    return time.time() - last > PATIENCE_S


def _title(folder: Path) -> str:
    """The one line somebody already wrote about this whole session, or nothing."""
    try:
        found = TITLE.search((folder / card.SUMMARY_NAME).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        return ""
    return " ".join(found.group(1).split()) if found else ""


def _project(records: list[dict]) -> str:
    """What this session said it was about, when it named a project out loud.

    The single most valuable line here and the cheapest: a `project` record already carries the
    name, and it is the difference between "a blue motorcycle" and "the Burning Man trike".
    """
    named = [r for r in records if r.get("type") == "project" and r.get("name")]
    return str(named[-1]["name"]) if named else ""


def _near(speech: list[dict], at: float) -> tuple[str, str]:
    """What was said just before a photo and just after it, as two blocks of quoted turns."""
    sides = []
    for lo, hi, take in ((at - BEFORE_S, at, -TURNS), (at, at + AFTER_S, TURNS)):
        turns = [r for r in speech if lo <= float(r.get("t") or 0.0) <= hi]
        turns = turns[take:] if take < 0 else turns[:take]
        sides.append(
            " ".join(
                f"{'They' if r.get('type') == 'you' else 'Cyclops'}: "
                f'"{" ".join(str(r.get("text", "")).split())[:MAX_SAID_CHARS]}"'
                for r in turns
                if str(r.get("text", "")).strip()
            )
        )
    return sides[0], sides[1]


def shots(images: Path) -> dict[str, str]:
    """Every picture in one folder, keyed by filename, described by everything except the pixels.

    ``{}`` when no session put them there, which is the answer for a project's ``Photos/``, an
    ``Eye Designs/``, and anything dropped on the card from a laptop.
    """
    folder = session_of(images)
    if folder is None:
        return {}
    records, _ = card.read_log(folder / card.LOG_NAME)
    records.sort(key=lambda r: float(r.get("t") or 0.0))
    speech = [r for r in records if r.get("type") in SPEECH]
    project, title = _project(records), _title(folder)

    out: dict[str, str] = {}
    for record in records:
        if record.get("type") != "photo" or not record.get("file"):
            continue
        before, after = _near(speech, float(record.get("t") or 0.0))
        # `focus` is what Cyclops was told to look at; `request` is what it was told to draw or
        # change. Only ever one of them, decided by who took the picture - deps.read_session
        # reads the pair the same way.
        asked = str(record.get("focus") or record.get("request") or "").strip()
        lines = [
            f"Project: {project}" if project else "",
            f"What this session was: {title}" if title else "",
            TOOK.get(str(record.get("by", "")), ""),
            f"Asked for: {asked}" if asked else "",
            f"Said just before: {before}" if before else "",
            f"Said just after: {after}" if after else "",
        ]
        block = "\n".join(line for line in lines if line)
        if block:
            out[Path(str(record["file"])).name] = block
    return out
