"""What a session folder is on the card, and how to write into one so a power cut cannot lie.

This module exists because of one incident and two sentences that turned out to be false.

The incident: the Pi lost power ten seconds after a session's teardown, and the folder came back
holding an intact ``session.jsonl``, a 27 MB ``video.mp4``, and a ``session.md`` and
``summary.md`` that were both **zero bytes long**. Nothing had failed. Every write returned
successfully. ext4 journals the inode when you create a file and writes the data back whenever
it feels like it, so a crash in between leaves exactly that: a file that exists, with a name,
containing nothing. The one file that survived - ``project.md``, written seven seconds *later* -
survived because :func:`cyclops.projects.store.write_receipt` fsyncs and nothing else did.

The two false sentences follow from it, and both were written down in this codebase:

* **"Written last, which is what makes its presence mean 'finished'."** Presence means a name
  exists. It does not mean a file has bytes in it, and ten separate ``.is_file()`` checks across
  four modules believed otherwise. A zero-byte ``session.md`` read as a finished session to the
  repair path, to the projects sweep and to the admin page - so the damage made itself invisible
  to the code whose whole job was to repair it.
* **"One file, written whole."** ``Path.write_text`` and ``open("w")`` truncate first and write
  second. There is a window, and on a box that gets unplugged the window is where the bug lives.

So: two rules, and every write and every question about a session folder goes through them.

* **Every write lands whole or not at all.** :func:`write_text` and :func:`write_bytes` write to
  a dotfile beside the target, fsync it, ``os.replace`` it into place, and fsync the directory.
  A reader sees the old file or the new one, never a half of either, and after the call returns
  the bytes are on the card rather than in RAM. This is the actual fix: a zero-byte file stops
  being something to detect and starts being something that cannot be created.
* **Presence is not existence.** :func:`written` asks whether a file is there *and* has bytes in
  it, and :func:`triage` answers every question anything asks about a session folder from its
  contents rather than from a name. The size checks are defence-in-depth once the writes are
  atomic - but every card that ever ran the old build has the damage on it already, and this is
  what finds it.

**Nothing here imports anything.** Standard library only: no ``cv2``, no ``openai``, no
``pydantic_ai``, and above all no :mod:`cyclops.session`, which pulls OpenCV in through
:mod:`cyclops.record`. That is not incidental. :mod:`cyclops.stats` used to keep its own copies
of ``PAGE_NAME`` and ``LOG_NAME`` with a comment explaining that importing ``session`` would
drag OpenCV into a status page that only wanted to count folders - and duplicated constants are
how two modules quietly stop agreeing about what a finished session looks like. This is the
module they can both have. Keep it dependency-free and they both keep it.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import stat
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# The names of everything a session folder can hold. One copy, imported by session.py, stats.py
# and projects/store.py, so "what is a finished session called" has a single answer.
LOG_NAME = "session.jsonl"
PAGE_NAME = "session.md"
SUMMARY_NAME = "summary.md"
RECEIPT_NAME = "project.md"  # written into the session folder by projects/store.py
# What each picture in this folder is of, written by the index service - see cyclops.captions.
# Lands in whichever folder holds the images, so a session has one in photos/ and a project has
# one in Photos/ and another in any folder you dropped pictures into.
CAPTIONS_NAME = "captions.json"
PHOTOS = "photos"
PARTS = "parts"
VIDEO = "video.mp4"
# The edited video somebody asked for, and the two files it is made from - see cyclops.cut.
# CUT existing means one ffmpeg run returned zero, exactly as VIDEO does, because both land
# through tmp_for/land rather than being written where they lie.
#
# Two markers rather than one, and that is the whole retry story: CUT_PLAN present means the
# model has been asked, CUT present means the encode is done. A deploy that kills a render
# halfway costs the encode and never the model call. CUT_REQUEST is the queue - the button
# writes it, the worker removes it - and it lives here rather than in ~/.cache so that renaming
# a session folder carries its request along with it.
CUT_REQUEST = "cut.request"
CUT_PLAN = "cut.json"
CUT_SUBS = "cut.ass"
CUT = "cut.mp4"
# The clips cyclops.cut finds by itself, one directory rather than a fixed set of names: a
# session yields between zero and three of them and KNOWN is a frozenset of literals, so the
# directory name is what goes in it. Inside: plan.json - whose presence is the whole "this
# session has been considered" gate - and then 1.mp4/1.ass, 2.mp4/2.ass, numbered from 1,
# best first.
#
# The four CUT_* names above are the one-video-per-session design this replaced. They stay
# here, and in session._remove, so a folder still holding one can still be deleted.
CLIPS = "clips"
CLIP_PLAN = "plan.json"

STAMP = "%Y-%m-%d_%H-%M-%S"
STAMPED = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")  # a folder nobody has named yet

# Everything a session folder is allowed to contain. Only ever consulted before deleting one:
# a folder holding anything not on this list is something a person put there, and is never
# removed however empty triage thinks it is.
KNOWN = frozenset(
    {
        LOG_NAME, PAGE_NAME, SUMMARY_NAME, RECEIPT_NAME, VIDEO, PHOTOS, PARTS, CLIPS,
        CUT_REQUEST, CUT_PLAN, CUT_SUBS, CUT,
    }
)

# What a record has to be for something to have happened here. Two sets rather than one, because
# they are two different arguments for keeping a folder and whoever adds the next record type
# needs to know which one they are adding to: somebody spoke, or something was made.
#
# Everything not in either - `session`, `video`, `end`, `error` - is bookkeeping *about* a
# session rather than anything that happened *in* one. So is every file the folder holds except
# its photos: `video.mp4`, `session.md` and `summary.md` are all made from the log.
#
# `transcript_failed` is in SPOKEN and that is the whole reason this comment is long. It is what
# the log gets when the realtime API cannot transcribe a turn (`session._observe`), so a session
# whose transcription failed throughout has a real conversation and a real recording and not one
# `you` record. Leaving it out would delete that video, and nothing downstream could have caught
# it - `session.transcript_text` reads only `you` and `cyclops`, so the model is never even asked.
SPOKEN = frozenset({"you", "cyclops", "transcript_failed"})
MADE = frozenset({"photo", "screen", "project", "data", "recall", "search"})


# ------------------------------------------------------------------ writing


def tmp_for(path: Path) -> Path:
    """The scratch name :func:`land` renames from - beside the target, on the same filesystem.

    Deterministic rather than unique: ``os.replace`` is only atomic within one filesystem, so
    this must be a sibling, and a fixed name means a killed write leaves at most one stray per
    target which the next write reuses. A pid-suffixed name would accumulate one per crash
    forever. Dotted so no directory scan in the codebase picks it up as content - every one of
    them either filters ``is_dir()`` or globs ``*.jpg``.
    """
    return path.with_name(f".{path.name}.tmp")


def sync_dir(directory: Path) -> None:
    """fsync a directory, so a name that appeared in it is on the card and not just in RAM.

    The step everyone forgets. ``os.replace`` makes the *content* swap atomic, but the rename
    is itself a change to the directory, and an unsynced directory can come back from a power
    cut without it. Best-effort: a filesystem that will not let us open a directory read-only
    is not a reason to fail a write that has already landed.
    """
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def land(tmp: Path, dst: Path) -> None:
    """Rename a finished scratch file onto its target, durably. Assumes ``tmp`` is fsynced."""
    os.replace(tmp, dst)
    sync_dir(dst.parent)


def write_bytes(path: Path, data: bytes, *, mode: int = 0o644) -> None:
    """One file, written whole or not at all.

    There is deliberately no ``sync=`` switch. The version of this that lived in
    ``projects/store.py`` had one, and a flag whose wrong setting silently costs you the file
    after a power cut is not a knob, it is a trap - the receipt was the only caller that ever
    passed it. These files are kilobytes, on a teardown path that has just written megabytes of
    video and is waited on by nobody.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tmp_for(path)
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
        with os.fdopen(fd, "wb") as handle:
            os.fchmod(fd, mode)  # the mode sticks even if the scratch file pre-existed
            handle.write(data)
            handle.flush()
            os.fsync(fd)
        land(tmp, path)
    finally:
        # Only reached with the tmp still present when something raised; land() has renamed it
        # away on every good path. Never leave our own litter for strays() to find.
        tmp.unlink(missing_ok=True)


def write_text(path: Path, text: str, *, mode: int = 0o644) -> None:
    """:func:`write_bytes`, for the markdown and JSON everything here actually writes."""
    write_bytes(path, text.encode("utf-8"), mode=mode)


def write_stream(path: Path, chunks: Iterable[bytes], *, limit: int, mode: int = 0o644) -> int:
    """:func:`write_bytes`, for bytes that arrive a block at a time. Returns how many landed.

    The same scratch-fsync-rename dance, and it exists for the one case ``write_bytes`` cannot
    cover: that one takes the whole payload as a ``bytes``, and what this was written for is a
    file somebody dragged onto the admin page from a laptop. The payload is then however large
    that file is, and a 40 MB video would be 40 MB of a Pi's RAM held for the length of an
    upload - on the box whose whole design note is that memory and heat are the scarce things.

    ``limit`` refuses rather than truncates. A file cut off at the ceiling would land whole, look
    whole in the browser, and be broken in whatever opened it later; raising leaves the scratch
    file to the ``finally`` below and nothing on the target at all, which is a failure you can
    see. The caller is what turns that into an answer the page can show.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tmp_for(path)
    written = 0
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
        with os.fdopen(fd, "wb") as handle:
            os.fchmod(fd, mode)  # the mode sticks even if the scratch file pre-existed
            for block in chunks:
                written += len(block)
                if written > limit:
                    raise ValueError(f"more than {limit} bytes")
                handle.write(block)
            handle.flush()
            os.fsync(fd)
        land(tmp, path)
    finally:
        # Same contract as write_bytes: only reached with the tmp still there when something
        # raised, which for this one includes the caller sending too much.
        tmp.unlink(missing_ok=True)
    return written


def strays(folder: Path) -> list[Path]:
    """The scratch files a killed write left behind, so recovery can sweep them.

    :func:`write_bytes` cleans up after an *exception*. It cannot clean up after a SIGKILL or a
    power cut, and that is exactly the case this module is about.
    """
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.glob(".*.tmp") if p.is_file())


# ------------------------------------------------------------------ reading


def written(path: Path) -> bool:
    """Is this file there **and** does it have anything in it?

    The whole bug, in one function. ``path.is_file()`` was the test everywhere, and it cannot
    tell a finished session from a zero-byte file wearing its name. One ``stat`` answers both
    halves of it - a directory has a size too, and that is never the answer to "is this file
    written" - so callers that must stay cheap (the admin page polls, the projects sweep stats
    every folder on the card) can use this without reading anything.
    """
    try:
        info = path.stat()
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode) and info.st_size > 0


def read_log(path: Path) -> tuple[list[dict], int]:
    """Every record in a ``session.jsonl``, and how many lines were not one.

    Lines are flushed as they land and fsynced once, at the close (see
    :mod:`cyclops.session`), so a power cut mid-session can leave the last one half-written.
    That is the only corruption this format can suffer, and nothing is swallowed silently: the
    count comes back and ``render_markdown`` says so on the page.

    Lives here rather than in ``session.py`` because :func:`triage` needs it and this module may
    not import that one. ``session.py`` re-exports the name.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return [], 0
    records: list[dict] = []
    dropped = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:  # JSONDecodeError is a subclass
            dropped += 1
            continue
        if isinstance(record, dict):
            records.append(record)
        else:
            dropped += 1
    return records, dropped


# ------------------------------------------------------------------ who is writing right now


def claim(handle) -> bool:
    """Take this folder's lock, by flocking the session's own open log handle.

    A live session already holds ``session.jsonl`` open for the whole of its life, so the lock
    is a flag on a file descriptor that exists anyway - no new path, nothing on the card (a lock
    on the card would travel with a card image, which is the argument ``store.LOCK_FILE``
    already makes), and it follows the inode through the rename that ends a session.

    ``flock`` for the reason it is used in ``store.py``: the kernel drops it when the holder
    dies. A box that loses power mid-session comes back with no lock at all - nothing stale to
    detect, no heuristic to get wrong, nobody to unwedge.

    Never raises and never matters. The lock is a courtesy to a repair process that might run
    later; a conversation is never blocked on being able to take it.
    """
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, ValueError, AttributeError):
        return False
    return True


def locked(folder: Path) -> bool:
    """Is a session writing into this folder right now?

    Asked by recovery before it touches anything. Getting this wrong in the optimistic direction
    is the worst failure this feature has: naming a folder *renames* it, and renaming one out
    from under a running session leaves its ``SessionLog.dir`` pointing at a path that no longer
    exists, after which the next shutter tap re-creates it and one session becomes two folders.
    """
    log = folder / LOG_NAME
    try:
        handle = log.open("r")
    except OSError:
        return False
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return True  # somebody else holds it
    else:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return False
    finally:
        handle.close()


# ------------------------------------------------------------------ what a folder amounts to


def said_or_made(records: list[dict]) -> bool:
    """Did anything happen in this session, as opposed to a session merely having happened?

    The cheap half of the question deletion asks. It reads the log and nothing else - no stat, no
    ffprobe, no network - which is what lets it run on the teardown path.

    A ``photo`` record has to carry a ``file``: an edit that failed leaves the record and no
    picture, and :meth:`cyclops.session.SessionLog.__exit__` already counts photos that way, so
    counting them differently here is how the two start to disagree.
    """
    for record in records:
        kind = record.get("type")
        if kind in SPOKEN:
            return True
        if kind in MADE and (kind != "photo" or record.get("file")):
            return True
    return False


@dataclass(frozen=True)
class State:
    """What one session folder on the card actually amounts to.

    Every question anything asks about a session folder is answered here, once, from content
    rather than from a name existing. What this replaces was ten separate ``.is_file()`` calls
    spread over four modules, none of which could tell a finished session from a zero-byte file
    with the right name - and three of which then *skipped* the damaged folder precisely because
    the damage was there.
    """

    path: Path
    verdict: str  # "live" | "finished" | "unfinished" | "empty"
    records: int
    dropped: int
    photos: int
    video: bool
    parts: bool
    page: bool
    summary: bool
    filed: bool
    named: bool
    content: bool  # somebody spoke or something was made - see said_or_made
    salvage: bool  # is there anything here worth keeping? the only question deletion ever asks


def triage(folder: Path) -> State:
    """Read a session folder and say what it is.

    Reads the log, so it is not free - callers that only need "is this one finished" should ask
    :func:`written` about ``session.md`` instead. ``_summarise`` and recovery both read the log
    anyway, which is what makes this the right shape for them.
    """
    records, dropped = read_log(folder / LOG_NAME)
    photos_dir = folder / PHOTOS
    photos = sum(1 for p in photos_dir.glob("*.jpg") if written(p)) if photos_dir.is_dir() else 0
    page = written(folder / PAGE_NAME)
    parts = (folder / PARTS).is_dir()
    video = written(folder / VIDEO)
    summary = written(folder / SUMMARY_NAME)

    content = said_or_made(records)
    # Four ways to be worth keeping, and only the first is about the session. The other three are
    # this module's own doctrine applied to itself: judge a folder by its log, and when there is
    # no log to judge it by, keep whatever it has. A `video.mp4` beside a zero-byte log is one
    # where the recording is now the only surviving copy of the session - which is the incident
    # at the top of this file - and a torn log is one where the line we lost is the line that
    # could have been a `you` turn. A guess is never a reason to delete.
    salvage = bool(
        content
        or photos
        or dropped
        or (not records and (video or parts or page or summary))
    )

    if locked(folder):
        verdict = "live"  # asked first: a folder being written to is not judged at all
    elif not salvage:
        verdict = "empty"
    elif parts or not page:
        verdict = "unfinished"  # parts/ means the mux never finished, whatever else is here
    else:
        verdict = "finished"

    return State(
        path=folder,
        verdict=verdict,
        records=len(records),
        dropped=dropped,
        photos=photos,
        video=video,
        parts=parts,
        page=page,
        summary=summary,
        filed=written(folder / RECEIPT_NAME),
        named=not STAMPED.match(folder.name),
        content=content,
        salvage=salvage,
    )


def surprises(folder: Path) -> list[str]:
    """Anything in this folder that cyclops did not put there. Empty means we know it all.

    Consulted only before deleting, which is the one destructive thing in the program - so it
    asks a second time and from a different direction. :func:`triage` says there is nothing of
    value here; this says there is nothing here we do not recognise. A folder someone dropped a
    note into keeps the note, and therefore keeps the folder, whatever the first answer was.
    """
    try:
        return sorted(p.name for p in folder.iterdir() if p.name not in KNOWN)
    except OSError:
        return ["unreadable"]  # cannot enumerate it, so cannot promise it is safe to remove
