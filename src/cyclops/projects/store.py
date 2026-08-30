"""The projects folder on the card: its shape, its locks, and every byte written into it.

This is the only module in the package that writes. Everything else - the orchestrator, its four
subagents, the CLI - decides things and hands them here. The rule is meant to be checkable rather
than merely intended::

    rg -n 'write_text|write_bytes|mkdir|card\\.land|os\\.replace|shutil\\.(copy|move)' \\
       src/cyclops/projects/ -g '!store.py'

and it should come back empty.

The layout::

    projects/Pelican Display Mount/
        README.md   rewritten every run: frontmatter is ours, the prose under it is the model's
        Log.md      appended to and never rewritten, one dated entry per session
        Photos/2026-08-26_16-48-48_cyclops.jpg

One sentence holds the whole design up: **Log.md is the truth, README.md is a view of it, and
projects/ can be rebuilt from sessions/ alone.** That is the same split session.py already runs
on - ``session.jsonl`` is the record and ``session.md`` is only ever ``render_markdown`` of it -
and it is what lets this have no database. When the two disagree, Log.md wins and the README is
regenerated; the README is never a source of anything except the frontmatter we put there.

Three rules follow from it, and each is load-bearing:

* **Identity is the ``key`` in the frontmatter, never the folder name.** Frozen when the project
  is created and never recomputed. This is what lets someone rename a folder in Finder without
  the next session quietly starting a second project beside it.
* **Code owns the frontmatter; the model owns the prose.** A model that rewrote the frontmatter
  would eventually mangle ``key`` on one bad run, and the project would fork in two permanently.
* **Not filing is always recoverable; filing wrongly is not.** So every failure path here leaves
  the receipt unwritten, which is exactly how the next sweep knows to try again.

This module is importable on a box with no key, no network and no ``pydantic_ai`` - which is what
lets the live agent's two tools, ``cyclops-projects --check`` and plain listing all work offline.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .. import card
from ..config import Settings
from ..slug import fold, safe_folder_name
from . import data

README_NAME = "README.md"
LOG_NAME = "Log.md"
DATA_NAME = "Project Data.xlsx"  # the numbers; see projects/data.py for what is in it
PHOTOS = "Photos"
DIAGRAMS = "Diagrams"  # drawings, kept whole: the spec Cyclops re-reads and the svg you open
RECEIPT_NAME = "project.md"  # written into the *session* folder, not the project

# The end of one entry in Log.md, and the only thing in that file a program reads. An HTML comment
# because it has to be invisible in every markdown viewer on the card, and it closes with "-->"
# because that is what makes a torn tail detectable: the one corruption an appended file can
# suffer is a half-written last line, exactly as in session.jsonl, and an entry without its
# terminator is one that did not finish.
ENTRY_END = re.compile(r"<!--\s*cyclops:session\s+([0-9a-fA-F-]{36})[^>]*-->")

# A lock is a fact about this machine, not about the card: put it on the card and it travels with
# a card image, and it shows up in a Finder browse of a tree whose entire purpose is to be
# browsable. config.py already put the browser-close flag and the volume file here, for adjacent
# reasons and with the same argument about /tmp.
LOCK_FILE = Path.home() / ".cache" / "cyclops" / "projects.lock"
LOG_FILE = Path.home() / ".cache" / "cyclops" / "projects.log"
LOCK_WAIT_S = 300.0  # a backstop against a wedged sweep, not a budget anything plans against
LOCK_POLL_S = 0.5

MAX_ALIASES = 8
MAX_KEYWORDS = 12
MAX_TAGLINE_CHARS = 320  # a dense sentence naming specifics; see ProjectPage.tagline
STATUSES = ("active", "paused", "done")


class Unfilable(RuntimeError):
    """A name that cannot become a folder, or a project that cannot be reached."""


class Busy(RuntimeError):
    """Another sweep holds the lock and would not let go."""


# ------------------------------------------------------------------ front matter

# A leading one of these and YAML stops reading a plain scalar, so those values get quoted.
_QUOTE_ME = "\"'#-?:,[]{}&*!|>%@`"


def _scalar(value: str) -> str:
    """One value, plain if it reads back as itself and quoted if it would not.

    Quoting is decided rather than universal: ``updated: 2026-08-26`` should look like a date to
    someone reading the card, not like a string. The three ways a plain scalar goes wrong are a
    leading indicator character, a ``": "`` inside it and a ``" #"`` inside it - so those, and
    only those, get quotes.
    """
    value = " ".join(str(value).split())  # never a newline inside a scalar, whatever we were given
    if value and value[0] not in _QUOTE_ME and ": " not in value and " #" not in value:
        return value
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        body = value[1:-1]
        # One left-to-right pass, deliberately: unescaping quotes before backslashes mishandles a
        # literal backslash that precedes a quote. It matters approximately never and costs
        # nothing to get right.
        return re.sub(r"\\(.)", r"\1", body) if value[0] == '"' else body
    return value.split(" #", 1)[0].rstrip()  # a trailing comment someone added by hand


def render_front(fields: dict[str, object]) -> str:
    """The block, in the order given - the shape ``session.render_markdown`` already builds."""
    out = ["---"]
    for name, value in fields.items():
        if isinstance(value, list | tuple):
            out.append(f"{name}:")
            out.extend(f"  - {_scalar(str(item))}" for item in value)
        else:
            out.append(f"{name}: {_scalar(str(value))}".rstrip())
    out.append("---")
    return "\n".join(out)


def parse_front(text: str) -> dict[str, str | list[str]]:
    """The block back out of a file, for the small subset we ever write.

    Hand-rolled rather than a YAML dependency, and safe because the emitter above is constrained:
    this only has to read back what :func:`render_front` wrote, plus be unsurprised by whatever a
    person typed into it in a text editor. What we emit is a strict subset that is also valid
    YAML, so ``yaml.safe_load`` would read it unchanged if a real parser is ever wanted.

    A file with no block, or one whose block never closes, comes back empty - the same answer
    :func:`cyclops.session.read_summary` gives to a missing summary, and it means the same thing
    here: there is nothing to trust, so rebuild from Log.md.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    end = next((n for n, line in enumerate(lines[1:], 1) if line.strip() == "---"), None)
    if end is None:
        return {}  # half a file: a power cut caught the write, or someone deleted a line
    fields: dict[str, str | list[str]] = {}
    current = ""
    for line in lines[1:end]:
        stripped = line.strip()
        if stripped.startswith("- ") and current:
            if not isinstance(fields.get(current), list):
                fields[current] = []
            fields[current].append(_unquote(stripped[2:]))  # type: ignore[union-attr]
            continue
        name, sep, value = stripped.partition(":")
        if not sep or not name.strip():
            continue
        current = name.strip()
        fields[current] = _unquote(value)
    return fields


def split_front(text: str) -> tuple[dict[str, str | list[str]], str]:
    """``(frontmatter, everything after it)``. The body is the model's half of the README."""
    front = parse_front(text)
    if not front:
        return {}, text
    lines = text.splitlines()
    end = next(n for n, line in enumerate(lines[1:], 1) if line.strip() == "---")
    return front, "\n".join(lines[end + 1 :]).strip()


# ------------------------------------------------------------------ a project


@dataclass
class Project:
    """One project as it is on the card. Built from a README's frontmatter, never from its path."""

    key: str  # the identity, frozen at creation. Never recomputed from the folder name.
    name: str  # the display name; the folder is a sanitised form of it
    path: Path  # where it is right now. Never shown to a model.
    status: str = "active"
    started: str = ""
    updated: str = ""
    sessions: int = 0
    photos: int = 0
    last: str = ""  # uuid of the most recently filed session
    tagline: str = ""  # the first line of the body, kept for the index and the catalog
    aliases: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    @property
    def log(self) -> Path:
        return self.path / LOG_NAME

    @property
    def readme(self) -> Path:
        return self.path / README_NAME

    @property
    def photos_dir(self) -> Path:
        return self.path / PHOTOS

    @property
    def diagrams_dir(self) -> Path:
        return self.path / DIAGRAMS

    @property
    def data(self) -> Path:
        return self.path / DATA_NAME

    def names(self) -> set[str]:
        """Every spelling that should resolve here, folded."""
        found = {self.key, fold(self.name), fold(self.path.name)}
        found.update(fold(alias) for alias in self.aliases)
        return {name for name in found if name}

    def line(self) -> str:
        """One line for the index handed to a model.

        The tagline leads, because it is the only field that says what the project actually *is*.
        A name is a label someone chose once and is often useless on its own - "Cyclops" and
        "Falcon" tell a model nothing about whether today's work belongs to either - so the
        sentence describing the thing does the distinguishing, and the keywords underneath it
        catch the vocabulary the name and tagline both miss.
        """
        bits = [self.name]
        if self.tagline:
            bits.append(self.tagline)
        bits += [self.status, f"{self.sessions} sessions"]
        if self.updated:
            bits.append(f"last worked on {self.updated}")
        if self.aliases:
            bits.append("also called: " + ", ".join(self.aliases))
        if self.keywords:
            bits.append("about: " + ", ".join(self.keywords))
        return f"{self.key} | " + " | ".join(bits)


def _project_from(folder: Path) -> Project | None:
    """Read one project off the card, or None if this folder is not one."""
    try:
        text = (folder / README_NAME).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    front, body = split_front(text)
    key = str(front.get("key", "") or "")
    if not key:
        # No key means no identity, and inventing one from the folder name is how a Finder rename
        # forks a project. Fall back to the folder's own fold, which is what a project someone
        # made by hand looks like, and let the next write freeze it properly.
        key = fold(folder.name)
    if not key:
        return None

    def one(name: str) -> str:
        value = front.get(name, "")
        return value if isinstance(value, str) else ""

    def many(name: str) -> list[str]:
        value = front.get(name, [])
        return [item for item in value if item] if isinstance(value, list) else []

    def count(name: str) -> int:
        try:
            return int(one(name) or 0)
        except ValueError:
            return 0

    return Project(
        key=key,
        name=one("project") or folder.name,
        path=folder,
        status=one("status") or "active",
        started=one("started"),
        updated=one("updated"),
        sessions=count("sessions"),
        photos=count("photos"),
        last=one("last"),
        tagline=_tagline(body),
        aliases=many("aliases")[:MAX_ALIASES],
        keywords=many("keywords")[:MAX_KEYWORDS],
    )


def _tagline(body: str) -> str:
    """The first real line under the heading - the sentence saying what the project is.

    Cut at a word boundary rather than mid-character, the way :func:`cyclops.slug._clean` does.
    This string is the matcher's main evidence, and "...writes up every session int" is both
    uglier and less useful than a clean stop.
    """
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if len(stripped) <= MAX_TAGLINE_CHARS:
            return stripped
        return stripped[: MAX_TAGLINE_CHARS - 1].rsplit(" ", 1)[0] + "…"
    return ""


def index(projects_dir: Path) -> list[Project]:
    """Every project on the card, most recently worked on first.

    A handful of small reads, one per project - the same shape and the same cost as
    :func:`cyclops.session.recent_context`, and cheap because it never opens a Log or a body.
    """
    folder = projects_dir.expanduser()
    if not folder.is_dir():
        return []
    found: dict[str, Project] = {}
    for entry in sorted(folder.iterdir()):
        if not entry.is_dir():
            continue
        project = _project_from(entry)
        if project is None:
            continue
        # Two folders claiming one key - someone duplicated a folder in Finder. Prefer the one
        # worked on most recently and say so rather than picking arbitrarily and hiding it.
        clash = found.get(project.key)
        if clash is not None:
            print(
                f"· two folders share the key {project.key!r}: "
                f"{clash.path.name!r} and {entry.name!r}; using the newer",
                flush=True,
            )
            if clash.updated >= project.updated:
                continue
        found[project.key] = project
    return sorted(found.values(), key=lambda p: (p.updated, p.name), reverse=True)


def catalog(settings: Settings) -> list[Project]:
    """The projects a running session should know about: the ones not finished with."""
    return [p for p in index(settings.projects_dir) if p.status != "done"]


def find(projects: list[Project], name: str) -> Project | None:
    """The project someone means by ``name`` - its key, its display name, or any alias."""
    wanted = fold(name)
    if not wanted:
        return None
    for project in projects:
        if wanted in project.names():
            return project
    # Nothing exact. A containment match catches "the falcon" against "Lego Millennium Falcon",
    # which is how people actually talk about their own projects.
    for project in projects:
        if any(wanted in known or known in wanted for known in project.names() if known):
            return project
    return None


def search(projects: list[Project], query: str, limit: int = 5) -> list[str]:
    """Projects that overlap ``query``, best first. Deterministic - no model involved."""
    wanted = set(fold(query).split())
    if not wanted:
        return []
    scored = []
    for project in projects:
        haystack = " ".join([project.name, project.tagline, *project.aliases, *project.keywords])
        words = set(fold(haystack).split())
        shared = len(wanted & words)
        if shared:
            scored.append((shared / len(wanted | words), shared, project))
    scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return [project.line() for _, _, project in scored[:limit]]


def read_body(project: Project) -> str:
    """The prose half of a project's README - what ``open_project`` hands the voice agent."""
    try:
        text = project.readme.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    _, body = split_front(text)
    return body


# One project's numbers are read, changed and written back, and the three steps must not
# interleave. They can: `_on_response_done` spawns every tool call of one response as its own
# task, so a photo of a spec plate yielding two `save_data` calls has two threads in here at
# once, and the second would write a book built before the first one's rows existed. The
# cross-process flock from `held()` is deliberately not taken - the sweep never opens this file,
# and that lock is about sweeps.
_DATA_LOCK = threading.Lock()


def read_data(project: Project) -> data.Book:
    """A project's numbers. A missing or unreadable file is an empty book, as in `read_body`."""
    try:
        return data.load(project.data.read_bytes())
    except OSError:
        return data.load(None)


def write_data(project: Project, book: data.Book) -> None:
    """Write the numbers back, whole or not at all.

    Through :func:`card.write_bytes` like every other byte on the card, so a power cut costs the
    save and never the file. It is a spreadsheet somebody opens in Excel; a half-written zip is
    not a degraded version of one, it is a file that will not open.
    """
    try:
        blob: bytes | None = project.data.read_bytes()
    except OSError:
        blob = None
    card.write_bytes(project.data, data.dump(book, blob))


@contextmanager
def data_held() -> Iterator[None]:
    """Hold the numbers of every project for one read-change-write. See :data:`_DATA_LOCK`."""
    with _DATA_LOCK:
        yield


def history(project: Project, entries: int = 5) -> str:
    """Where a project stands, and the headings of the last few things done to it."""
    body = read_body(project)
    headings = [line.strip() for line in _log_text(project).splitlines() if line.startswith("## ")]
    tail = "\n".join(headings[-entries:])
    return f"{body}\n\n## Recent entries\n{tail}".strip() if tail else body


# ------------------------------------------------------------------ the log, and its ledger


def _log_text(project: Project) -> str:
    try:
        return project.log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def filed_uuids(project: Project) -> set[str]:
    """Every session already written into this project's log.

    This is the ledger, and it is the correctness guarantee behind the receipt: a crash between
    appending an entry and writing the receipt leaves a session that looks unfiled, and this is
    what stops the next sweep appending it a second time. Only ever consulted for a session the
    receipt says is unfiled, so it costs one read on a rare path.
    """
    return {match.group(1).lower() for match in ENTRY_END.finditer(_log_text(project))}


def photo_count(project: Project) -> int:
    """How many photos the folder actually holds.

    Counted, never accumulated. A ``+=`` here drifts the moment a session is filed twice - which
    is a normal thing to happen, because losing a receipt is exactly what the ledger is there to
    survive - and a number in the frontmatter that disagrees with the folder beside it is the
    kind of small lie that makes a person stop trusting the rest of the page.
    """
    if not project.photos_dir.is_dir():
        return 0
    return sum(1 for item in project.photos_dir.iterdir() if item.is_file())


def entry_count(project: Project) -> int:
    """How many complete entries the log holds - what ``sessions:`` is checked against."""
    return len(ENTRY_END.findall(_log_text(project)))


def torn(project: Project) -> bool:
    """Does the log end mid-entry? The power probably went; see :func:`cyclops.session.read_log`.

    An entry that began and never got its terminator - not merely text after the last one. A log
    with no entries at all is a new project, not a damaged file, and the file's own ``#`` heading
    is not an entry: entries are ``##``. Getting that wrong makes every empty project report
    damage, which teaches people to ignore the one warning that matters.
    """
    text = _log_text(project)
    if not text.strip():
        return False
    done = list(ENTRY_END.finditer(text))
    tail = text[done[-1].end() :] if done else text
    return any(line.startswith("## ") for line in tail.splitlines())


def open_log():
    """The append handle for the detached sweep's own output, directory and all.

    Here rather than in the caller so that the "only store.py writes" rule stays literally true
    and the grep in the module docstring keeps coming back empty. A rule you have to remember an
    exception to is not a rule.
    """
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    return LOG_FILE.open("ab")


# ------------------------------------------------------------------ the lock


@contextmanager
def held(*, announce: bool = False) -> Iterator[None]:
    """Exclusive use of ``projects/``, or a clean give-up.

    ``flock`` rather than a PID file or a lock directory, for the one reason that decides it here:
    the kernel drops it when the holder dies. A box that loses power mid-sweep comes back with no
    lock at all - nothing stale to detect, no heuristic to get wrong, nobody to unwedge it.

    One lock for the whole tree rather than one per project. What two sweeps collide over is
    creating a folder, resolving a name collision and adopting a rename, and all three are facts
    about the directory rather than about any one project. It over-serialises on purpose: the most
    concurrency there can ever be here is two.

    Model calls happen outside this. Only the writes are inside it, so a manual sweep of fifty
    sessions never holds a global lock across fifty network round trips.
    """
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    handle = LOCK_FILE.open("a+")
    deadline, said = time.monotonic() + LOCK_WAIT_S, False
    try:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise Busy(f"another sweep held {LOCK_FILE} for {LOCK_WAIT_S:.0f}s") from None
                if announce and not said:  # once, so nobody is left staring at nothing
                    print("· another sweep is running; waiting for it", flush=True)
                    said = True
                time.sleep(LOCK_POLL_S)
        yield
    finally:
        handle.close()  # closing releases it, whichever way we left


# ------------------------------------------------------------------ creating one


class Exists(Unfilable):
    """The name given is already a project. Carries the one it collided with."""

    def __init__(self, project: Project) -> None:
        super().__init__(f"{project.name!r} is already being tracked")
        self.project = project


def folder_for(projects_dir: Path, name: str) -> Path:
    """Where a new project would live. Raises rather than inventing anything.

    Belt and braces over a loop that already cannot emit a separator: the resolved path must sit
    directly inside ``projects/``. That catches an absolute path, a ``..`` that somehow survived,
    and a symlink someone left on the card - none of which :func:`safe_folder_name` can see,
    because they are properties of the filesystem rather than of the string.
    """
    safe = safe_folder_name(name)
    if not safe:
        raise Unfilable(f"nothing survives sanitising {name!r} into a folder name")
    root = projects_dir.expanduser()
    target = root / safe
    if target.resolve().parent != root.resolve():
        raise Unfilable(f"{safe!r} would land outside {root}")
    # Two genuinely different projects that sanitise to one name must still not collide on a
    # case-insensitive filesystem, which is what the card and every Mac that reads it will be.
    taken = {fold(p.name) for p in root.iterdir() if p.is_dir()} if root.is_dir() else set()
    if fold(safe) in taken:
        for n in range(2, 100):
            if fold(f"{safe} ({n})") not in taken:
                return root / f"{safe} ({n})"
    return target


def create(settings: Settings, name: str, *, tagline: str = "") -> Project:
    """Start tracking something. Called from the voice agent's ``track_project`` tool.

    This is the only function anywhere that brings a project into existence, which is the whole
    point: nothing downstream invents one, so the filing pipeline only ever chooses between
    projects a person has already agreed to. The ambiguity that would otherwise fill the card with
    "Pelican Case", "Pelican Display" and "Pelican Mount" is answered in the conversation instead.

    The folder is complete the moment it returns - a README with real frontmatter and a stub body,
    an empty log with its heading, and a Photos directory - so a session that ends badly still
    leaves something a person can read.
    """
    projects_dir = settings.projects_dir.expanduser()
    with held():
        existing = find(index(projects_dir), name)
        if existing is not None:
            raise Exists(existing)
        target = folder_for(projects_dir, name)
        display = target.name
        today = datetime.now().astimezone().strftime("%Y-%m-%d")
        project = Project(
            key=fold(display),  # frozen here, and never recomputed for the life of the project
            name=display,
            path=target,
            status="active",
            started=today,
            updated=today,
            sessions=0,
        )
        target.mkdir(parents=True, exist_ok=True)
        project.photos_dir.mkdir(parents=True, exist_ok=True)
        _write(
            project.log,
            f"# {display} - log\n\n"
            "One entry per session, oldest first. Appended when a session is filed and never\n"
            f"rewritten; `{README_NAME}` is the picture built from this file and can be deleted\n"
            "and made again at any time.\n",
        )
        body = tagline.strip() or "Just started - nothing written down about it yet."
        project.tagline = body
        _write(project.readme, f"{_front_for(project)}\n\n# {display}\n\n{body}\n")
        return project


def _front_for(project: Project) -> str:
    """The frontmatter block for a project, in a fixed field order. Ours, never the model's."""
    fields: dict[str, object] = {
        "project": project.name,
        "key": project.key,
    }
    if project.aliases:
        fields["aliases"] = project.aliases[:MAX_ALIASES]
    if project.keywords:
        fields["keywords"] = project.keywords[:MAX_KEYWORDS]
    fields["status"] = project.status if project.status in STATUSES else "active"
    fields["started"] = project.started
    fields["updated"] = project.updated
    fields["sessions"] = project.sessions
    fields["photos"] = project.photos
    if project.last:
        fields["last"] = project.last
    return render_front(fields)


def _write(path: Path, text: str) -> None:
    """One file, written whole - :func:`cyclops.card.write_text`, under the name used here.

    This used to open the target with ``"w"``, which truncates before it writes, and to fsync
    only when a ``sync=`` argument said so. Both halves were wrong for the same reason and in
    the same place: the receipt was the only caller that ever passed ``sync=True``, and it is
    also the file whose mere presence stops a session ever being read again - so a power cut
    mid-write could leave a truncated ``project.md`` that gated its own retry permanently. The
    flag is gone rather than defaulted; the cost is one fsync on files measured in kilobytes.
    """
    card.write_text(path, text)


# ------------------------------------------------------------------ writing a session in


def copy_photos(
    project: Project, session_dir: Path, picks: list[tuple[str, str]], *, limit: int
) -> list[tuple[str, str]]:
    """Copy the chosen photos in, and hand back ``(relative path, caption)`` for the log.

    Copies rather than symlinks: exFAT has no symlinks, and the session folder may be deleted
    while the project outlives it. The target name is the session's date plus the photo's own
    name, which is what makes this idempotent with no ledger - a re-run computes the same target,
    finds it already there, and does nothing.

    Never pruned, here or anywhere: a photo removed later would break a link already written into
    Log.md, and Log.md is never rewritten.
    """
    if limit <= 0:
        return []
    date = session_dir.name[:10]
    written = []
    for name, caption in picks[:limit]:
        source = session_dir / name
        if not source.is_file() or source.parent.resolve() != (session_dir / "photos").resolve():
            continue  # a name the model invented, or one pointing outside the session's photos
        target = project.photos_dir / f"{date}_{Path(name).name}"
        try:
            if not target.is_file() or target.stat().st_size != source.stat().st_size:
                project.photos_dir.mkdir(parents=True, exist_ok=True)
                # Through a scratch name, so a kill mid-copy cannot leave a truncated jpg on the
                # final one. The size check above already made the next sweep repair it, but in
                # between, photo_count() counted an orphan no Log.md entry pointed at.
                tmp = card.tmp_for(target)
                shutil.copy2(source, tmp)
                card.land(tmp, target)
        except OSError:
            continue  # a photo is never worth failing a filing over
        written.append((f"{PHOTOS}/{target.name}", caption))
    return written


def copy_diagrams(project: Project, session_dir: Path) -> list[tuple[str, str]]:
    """Copy every diagram this session drew in, and hand back ``(relative path, title)``.

    Every one, with no curator and no limit - the one place this deliberately parts company with
    :func:`copy_photos`. A photo is a frame caught in passing and three of forty are worth
    keeping, so a model picks. A diagram was asked for out loud, drawn on purpose and looked at;
    there is no version of "which of these did you mean" worth asking about it.

    Both halves travel: the ``.json`` is what Cyclops reads to put it back on the panel, and the
    ``.svg`` beside it is what a person opens. The svg is what gets linked from the log, because
    a log entry is read by people.

    Idempotent the same way, by the same trick - the target name is the session's date plus the
    diagram's own - and never pruned, for the same reason: Log.md is never rewritten.
    """
    source_dir = session_dir / card.DIAGRAMS
    if not source_dir.is_dir():
        return []
    date = session_dir.name[:10]
    written = []
    for spec in sorted(source_dir.glob("*.json")):
        if not card.written(spec):
            continue  # a drawing interrupted mid-write; the next sweep finds it whole or not
        stem = f"{date}_{spec.stem}"
        picture = spec.with_suffix(".svg")
        for source, target in ((spec, project.diagrams_dir / f"{stem}.json"),
                               (picture, project.diagrams_dir / f"{stem}.svg")):
            if not card.written(source):
                continue  # a diagram the panel never sent a picture back for keeps its spec
            try:
                if not target.is_file() or target.stat().st_size != source.stat().st_size:
                    project.diagrams_dir.mkdir(parents=True, exist_ok=True)
                    tmp = card.tmp_for(target)
                    shutil.copy2(source, tmp)
                    card.land(tmp, target)
            except OSError:
                continue  # a drawing is never worth failing a filing over
        if card.written(project.diagrams_dir / f"{stem}.svg"):
            title = _diagram_title(spec)
            written.append((f"{DIAGRAMS}/{stem}.svg", title))
    return written


def _diagram_title(spec: Path) -> str:
    """What a diagram calls itself, for the log entry's caption. Its filename if it will not say."""
    try:
        found = json.loads(spec.read_text(encoding="utf-8"))
        title = str(found.get("title", "")).strip()
    except (OSError, ValueError):
        title = ""
    return title or spec.stem


def append_entry(
    project: Project,
    *,
    title: str,
    body: str,
    when: datetime,
    uuid: str,
    stamp: str,
    span: str,
    photos: list[tuple[str, str]],
    diagrams: list[tuple[str, str]] | None = None,
) -> None:
    """Add one session to the log. Appended, fsynced, and never touched again.

    Oldest first, so this is a real ``open("a")`` and a power cut can only ever damage the tail.
    The terminator goes last for the same reason it exists: an entry without one is an entry that
    did not finish, and the next sweep will not count it as filed.

    This is the one write in the whole feature that nothing else can reconstruct, which is why it
    is the one place that fsyncs. session.py's "flushed, never fsynced" rule is about an fsync per
    *line* grinding an SD card; this is one per session, next to a video that already wrote three
    megabytes a minute.
    """
    blocks = [f"## {when:%A %-d %B %Y} - {title}".rstrip(" -"), body.strip()]
    blocks += [f"![{caption}]({path})" for path, caption in photos]
    # After the photos: a photo is what the bench looked like and a diagram is what was worked
    # out, and the second reads better as the conclusion of an entry than as its illustration.
    blocks += [f"![{caption}]({path})" for path, caption in (diagrams or [])]
    blocks.append(f"<!-- cyclops:session {uuid} · {stamp} · {span} -->")
    text = "\n\n".join(block for block in blocks if block)
    project.log.parent.mkdir(parents=True, exist_ok=True)
    with project.log.open("a", encoding="utf-8") as handle:
        if project.log.stat().st_size:
            handle.write("\n")
        handle.write(text + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def render_page(
    project: Project,
    page: dict[str, object],
    photos: list[tuple[str, str]],
    diagrams: list[tuple[str, str]] | None = None,
) -> str:
    """The whole README: our frontmatter, then the model's prose in our sections.

    The split is not tidiness. If a model wrote the frontmatter, one malformed run would mangle
    ``key`` and the project would fork in two, silently and permanently. So the model supplies
    sentences and lists and this decides where they go - which also means ``## Where it stands``
    is always the first section, and always the paragraph worth handing to the next session.
    """

    def lines(name: str) -> list[str]:
        value = page.get(name) or []
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    blocks = [_front_for(project), f"# {project.name}"]
    if tagline := str(page.get("tagline", "")).strip():
        blocks.append(tagline)
    if stands := str(page.get("where_it_stands", "")).strip():
        blocks += ["## Where it stands", stands]
    sections = (("Still open", "still_open"), ("Decided", "decided"), ("Details", "details"))
    for heading, key in sections:
        if items := lines(key):
            blocks += [f"## {heading}", "\n".join(f"- {item}" for item in items)]
    if photos:
        blocks += ["## Photos", "\n".join(f"![{caption}]({path})" for path, caption in photos)]
    if diagrams:
        blocks += ["## Diagrams",
                   "\n".join(f"![{caption}]({path})" for path, caption in diagrams)]

    plural = "" if project.sessions == 1 else "s"
    footer = f"*{project.sessions} session{plural}"
    if since := _spoken(project.started):
        footer += f" since {since}"
    blocks += ["---", f"{footer}. The full history is in [{LOG_NAME}]({LOG_NAME}).*"]
    return "\n\n".join(blocks) + "\n"


def _spoken(date: str) -> str:
    """``2026-08-26`` as "26 August 2026". The footer is read by a person, not parsed."""
    try:
        return f"{datetime.strptime(date, '%Y-%m-%d'):%-d %B %Y}"
    except ValueError:
        return ""


def rewrite_readme(project: Project, text: str) -> None:
    """Replace the README, whole or not at all.

    This already did the tmp-and-rename half - the atomicity was always the point. What it did
    not do was fsync, which on ext4 is what makes the atomicity mean anything across a power
    cut, and it hand-rolled the temp file, so a process killed between the write and the replace
    stranded a ``.README.md.tmp`` that nothing ever cleaned up. Both are card.write_text's job.
    """
    card.write_text(project.readme, text)


# ------------------------------------------------------------------ the receipt


def receipt(session_dir: Path) -> Path:
    return session_dir / RECEIPT_NAME


def is_filed(session_dir: Path) -> bool:
    """Has this session been read already? One stat, which is what makes a sweep cheap.

    The same argument ``session.md`` makes for "this session finished", and the reason the ledger
    in Log.md is not the only check: consulting that would mean reading every project's log on
    every sweep, instead of one stat per folder.
    """
    # written(), not is_file(): the receipt is the only thing standing between a session
    # and a re-read, so a truncated one must not be able to gate it forever. Tightening
    # this is safe by construction - a re-file is caught by filed_uuids() and reported as
    # "already in the log" rather than written twice.
    return card.written(receipt(session_dir))


def write_receipt(session_dir: Path, project: Project | None, why: str = "") -> None:
    """Say where this session went, in the session's own folder. Written last, and fsynced.

    Its ``project:`` and ``key:`` are prose for a person, never an index - nothing ever reads them
    back. That matters: merging two projects by hand leaves every receipt naming a folder that no
    longer exists, and none of it breaks, because the only machine-load-bearing thing about this
    file is that it is here.
    """
    now = datetime.now().astimezone()
    if project is None:
        front = {
            "project": "",
            "skipped": why or "not project work",
            "filed": f"{now:%Y-%m-%d %H:%M:%S %z}",
        }
        body = (
            "# Not filed under any project\n\n"
            f"Read on {now:%-d %B %Y} and left where it is.\n\n"
            f"{why or 'This was not work on anything being tracked.'}\n\n"
            "Nothing was written to `projects/`. Delete this file to have it read again.\n"
        )
    else:
        front = {
            "project": project.name,
            "key": project.key,
            "filed": f"{now:%Y-%m-%d %H:%M:%S %z}",
        }
        where = f"{project.path.parent.name}/{project.path.name}/{LOG_NAME}"
        link = f"../../{where}".replace(" ", "%20")
        body = (
            f"# Filed under {project.name}\n\n"
            f"Read on {now:%-d %B %Y} and added to [{project.path.name}/{LOG_NAME}]({link}).\n\n"
            "This file is a receipt: its presence is what stops the next sweep reading this\n"
            "session again. Delete it, and the entry it points at, to have it filed elsewhere.\n"
        )
    _write(receipt(session_dir), f"{render_front(front)}\n\n{body}")
