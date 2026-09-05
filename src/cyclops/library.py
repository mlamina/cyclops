"""What is on the card, for the page that shows it: sessions, and every picture in them.

Stdlib and :mod:`cyclops.card`, and nothing else. The same rule ``stats.py`` states at its own
:func:`session_counts` applies here and for the same reason: ``session.py`` imports ``record.py``
and with it OpenCV, and the admin service has no business loading a video encoder to list a
directory. Everything a listing needs - what a folder is called, what marks one finished, how to
read its log - already lives in ``card.py``, which is the copy those two modules share.

For everything this module answers, the card is still the index: a session is a folder, and the
answer to any question a listing asks is in that folder. What this module adds is a small read
cache, because the page asks the same question every time you switch views.

This used to say "there is no index and no database", flatly and about the whole codebase, and
that stopped being true when :mod:`cyclops.recall` arrived. Semantic search cannot be answered by
reading one folder - you cannot embed a query against a directory listing - so there is now a
vector index under ``~/.cache/cyclops/``, kept by its own service. Nothing here uses it and nothing
here needs to.

What survived the change is the half that was actually load-bearing, and it is still worth
stating: **the card holds every fact.** The index is derived from these folders and holds nothing
that is not in them, so deleting it costs a rebuild and never an answer.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

from .card import (
    LOG_NAME,
    PHOTOS,
    STAMP,
    SUMMARY_NAME,
    VIDEO,
    locked,
    read_log,
    triage,
    written,
)

# A picture is named for the moment it was made, to the second - see session.py's photo_target.
# That prefix is the only timestamp the file carries.
AT = re.compile(r"^(\d{2})-(\d{2})-(\d{2})_")

STAMP_LEN = len("0000-00-00_00-00-00")

# What the transcript shows. `session`, `video` and `end` are the bookkeeping records that the
# frontmatter is made of; they say nothing a reader wants in the middle of a conversation.
#
# `screen` and `transcript_failed` joined on 2026-09-05, when this transcript stopped being only
# a thing you read afterwards. A scratchpad is the one thing Cyclops writes *while he is talking*,
# so a companion following along live and not being told about it is the one gap you would
# actually notice; and a turn that could not be transcribed leaves a hole in the conversation that
# is better labelled than silently missing. Both have always been in `session.md`
# (`session._render_record`) - this is the web transcript catching up with it.
SPOKEN = frozenset(
    {
        "you",
        "cyclops",
        "photo",
        "search",
        "project",
        "data",
        "recall",
        "screen",
        "transcript_failed",
        "error",
    }
)

STREAM_LIMIT = 500


@dataclass(frozen=True)
class Entry:
    """One session, as much of it as a list needs to show."""

    name: str  # the folder, which is also its id in every URL below
    title: str
    summary: str
    started: str  # ISO-8601, or "" for a folder whose name is not a stamp
    seconds: float
    entrypoint: str
    video: bool
    video_bytes: int
    photos: int
    verdict: str  # card.State.verdict: live | finished | unfinished | empty
    filed: str  # the project it was filed under, "" if none


@dataclass(frozen=True)
class Item:
    """One picture, for the stream that runs across every session."""

    kind: str  # "photo" - the only kind there is; a diagram is a photo too
    session: str
    session_title: str
    title: str
    when: str  # ISO-8601
    url: str


# ------------------------------------------------------------------ reading one folder


def _started(folder: Path) -> datetime | None:
    """When a session began, read off the timestamp its folder name still starts with."""
    try:
        return datetime.strptime(folder.name[:STAMP_LEN], STAMP)
    except ValueError:
        return None


def _slug(folder: Path) -> str:
    """The part of the folder name the namer chose, or "" while it is still just a stamp."""
    return folder.name[STAMP_LEN + 1 :] if len(folder.name) > STAMP_LEN else ""


def _summary(folder: Path) -> tuple[str, str]:
    """``(title, paragraph)`` out of ``summary.md``, or two empty strings.

    The same shape as ``session.read_summary``, which cannot be imported from here - see the
    module docstring. Ten lines is the cheaper half of that trade.
    """
    try:
        text = (folder / SUMMARY_NAME).read_text(encoding="utf-8")
    except OSError:
        return "", ""
    title, body = "", []
    for line in text.splitlines():
        stripped = line.strip()
        if not title:
            if stripped.startswith("#"):
                title = " ".join(stripped.lstrip("#").split())
            continue
        body.append(stripped)
    return title, " ".join(" ".join(body).split())


def _project(folder: Path) -> str:
    """The project a session was filed under, off ``project.md``'s frontmatter."""
    try:
        text = (folder / "project.md").read_text(encoding="utf-8")
    except OSError:
        return ""
    for line in text.splitlines()[:8]:  # it is frontmatter or it is nothing
        if line.startswith("project:"):
            # A sweep that filed a session under nothing writes an empty *quoted* string, and
            # `""` on the end of a meta line reads as a bug in the page rather than as no answer.
            return line.split(":", 1)[1].strip().strip("\"'")
    return ""


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _read(folder: Path) -> Entry:
    """Everything a listing shows about one folder, from the folder."""
    state = triage(folder)
    records, _ = read_log(folder / LOG_NAME)
    head = next((r for r in records if r.get("type") == "session"), {})
    tail = next((r for r in reversed(records) if r.get("type") == "end"), {})

    when = _started(folder)
    seconds = tail.get("seconds")
    if not isinstance(seconds, int | float):
        # No end record: it was interrupted, or it is running right now. The last thing that
        # happened is the best answer either way, and it is never worse than zero.
        seconds = max((r.get("t", 0) for r in records), default=0.0)

    title, summary = _summary(folder)
    if not title:
        slug = _slug(folder)
        if slug:
            title = slug.replace("-", " ").title()
        elif when:
            title = f"Session {when:%Y-%m-%d %H:%M}"
        else:
            title = folder.name

    return Entry(
        name=folder.name,
        title=title,
        summary=summary,
        started=when.isoformat() if when else "",
        seconds=round(float(seconds), 2),
        entrypoint=str(head.get("entrypoint", "")),
        video=state.video,
        video_bytes=_size(folder / VIDEO) if state.video else 0,
        photos=state.photos,
        verdict=state.verdict,
        filed=_project(folder) if state.filed else "",
    )


# ------------------------------------------------------------------ the cache

# Keyed on the folder's own mtime, which moves whenever anything lands in it or it is renamed -
# which is exactly when what we said about it stopped being true. Per gunicorn worker, read-only,
# and gone on restart: this is a way to not re-read 23 logs when you tap between two tabs, not a
# store. A live session's folder is never cached, because its mtime is about to change again.
_cache: dict[Path, tuple[int, Entry]] = {}


def _cached(folder: Path) -> Entry:
    try:
        stamp = folder.stat().st_mtime_ns
    except OSError:
        return _read(folder)
    found = _cache.get(folder)
    if found is not None and found[0] == stamp:
        return found[1]
    entry = _read(folder)
    if entry.verdict != "live":
        _cache[folder] = (stamp, entry)
    return entry


# ------------------------------------------------------------------ the whole card


def _folders(sessions_dir: Path) -> list[Path]:
    """Session folders, newest first. ``iterdir`` does not descend, so ``photos/`` is never one."""
    directory = sessions_dir.expanduser()
    try:
        found = [entry for entry in directory.iterdir() if entry.is_dir()]
    except OSError:
        return []
    # The name starts with the stamp, so it sorts chronologically as a string and there is no
    # date to parse for the ordering - only for the display.
    return sorted(found, key=lambda p: p.name, reverse=True)


def entries(sessions_dir: Path) -> list[Entry]:
    """Every session on the card, newest first."""
    return [_cached(folder) for folder in _folders(sessions_dir)]


def entry(sessions_dir: Path, name: str) -> Entry | None:
    """One session by folder name, or None if there is no such folder."""
    folder = resolve(sessions_dir, name)
    return None if folder is None else _cached(folder)


def resolve(sessions_dir: Path, name: str) -> Path | None:
    """The folder called ``name``, or None - and never anything outside the sessions directory.

    By construction rather than by inspecting the string. Whatever ``name`` is - dotted, slashed,
    absolute, a symlink - it is resolved first and then required to be a direct child of the
    resolved root, which is one comparison for every way out of the tree at once.
    """
    root = sessions_dir.expanduser().resolve()
    try:
        found = (root / name).resolve()
    except OSError:
        return None
    if found.parent != root or not found.is_dir():
        return None
    return found


def live(sessions_dir: Path) -> str | None:
    """The session being recorded right now, or None. Cheap enough to poll: no log is read.

    A running session holds an exclusive ``flock`` on its own ``session.jsonl`` for its whole life
    (``card.claim``), which is the same question :func:`card.triage` answers with ``verdict ==
    "live"`` - asked here without the log read that goes with it.

    The newest few folders rather than the newest one. Names begin with a sortable stamp and the
    end-of-session rename only appends a slug, so the running one is first by construction; three
    costs two more ``flock`` probes and stops that being an assumption.
    """
    for folder in _folders(sessions_dir)[:3]:
        if locked(folder):
            return folder.name
    return None


def records(sessions_dir: Path, name: str) -> list[dict]:
    """A session's transcript: the log, filtered to what is worth reading, media resolved to URLs.

    The JSONL and not ``session.md``, because every line carries the ``t`` that is also its
    position in the video, and the markdown is only a rendering of these.
    """
    folder = resolve(sessions_dir, name)
    if folder is None:
        return []
    found, _ = read_log(folder / LOG_NAME)
    out = []
    for record in found:
        kind = record.get("type")
        if kind not in SPOKEN:
            continue
        line = dict(record)
        if kind == "photo" and record.get("file"):
            line["url"] = media_url(name, str(record["file"]))
        out.append(line)
    return out


def media_url(name: str, relative: str) -> str:
    """Where the page fetches one file out of one session."""
    return f"/media/{name}/{relative}"


# ------------------------------------------------------------------ the picture stream


def _at(folder_start: datetime | None, filename: str) -> datetime | None:
    """The moment a photo or a diagram was made, off the time of day in its name."""
    match = AT.match(filename)
    if match is None or folder_start is None:
        return None
    hour, minute, second = (int(part) for part in match.groups())
    day = folder_start.replace(hour=0, minute=0, second=0, microsecond=0)
    made = day + timedelta(hours=hour, minutes=minute, seconds=second)
    # A session that crosses midnight names its later files with a smaller clock than it started
    # with. The folder's date plus a day is the only reading of that which moves forwards.
    return made + timedelta(days=1) if made < folder_start else made


def stream(sessions_dir: Path, limit: int = STREAM_LIMIT) -> list[Item]:
    """Every photo and every drawing on the card, newest first, as one run.

    Built from the folders rather than from the logs: a picture is a file, and a session whose log
    was truncated by a power cut still has its pictures. The sessions are walked newest-first and
    the walk stops at ``limit``, so the cost of the first page does not grow with the card.
    """
    out: list[Item] = []
    for folder in _folders(sessions_dir):
        found = _cached(folder)
        start = _started(folder)
        made: list[Item] = []

        photos = folder / PHOTOS
        if photos.is_dir():
            for picture in photos.glob("*.jpg"):
                if not written(picture):
                    continue
                when = _at(start, picture.name)
                made.append(
                    Item(
                        kind="photo",
                        session=folder.name,
                        session_title=found.title,
                        title=found.title,
                        when=when.isoformat() if when else found.started,
                        url=media_url(folder.name, f"{PHOTOS}/{picture.name}"),
                    )
                )

        made.sort(key=lambda item: item.when, reverse=True)
        out.extend(made)
        if len(out) >= limit:
            return out[:limit]
    return out


def as_dicts(found: list[Entry] | list[Item]) -> list[dict]:
    """For JsonResponse, which will not take a dataclass."""
    return [asdict(one) for one in found]
