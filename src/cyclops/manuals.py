"""The manuals folder on the card: what a manual is, and every byte written into it.

A manual is a PDF somebody dropped on the admin page, plus what reading it produced. It lives in
its own folder under ``manuals/``, beside ``sessions/`` and ``projects/`` rather than inside any
one of them, because **a manual documents a part and not a project**. The same m.unit manual
serves two bikes; filed under one of them it would be invisible to the other, and "do we have the
manual for this?" would start depending on which project the conversation happened to be near.

So this is a domain object, built the way :class:`cyclops.projects.store.Project` is built - **from
the frontmatter of a file, never from the path**. A folder renamed in Finder is the same manual;
the ``key`` frozen at first read is the identity. That is also why :meth:`Manual.names` exists:
Marco says "m.unit", motogadget's cover says "mo.unit blue / basic", and the file the web server
sent is called ``mo.unit_blue_%26_basic_manual.pdf``. All three have to resolve to one thing.

What this module must never do:

* **No pymupdf and no openai.** Rendering a page and reading it belong to :mod:`cyclops.reading`;
  this module only ever touches text and bytes already on the card. ``cyclops-admin`` imports this
  to list manuals and must keep working on a box with no key, which is the rule ``shelf.py`` and
  ``store.py`` both keep for the same reason.
* **Nothing raises for a file it cannot read.** A hand-edited ``Manual.md`` is a skipped manual,
  never a failed sweep - the rule :mod:`cyclops.recall` states and for the same reason.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from . import card
from .config import Settings
from .projects.store import render_front, split_front
from .slug import fold, safe_folder_name

# The identity file. Not README.md: recall.SKIP_NAMES already special-cases that name for
# projects, and the whole point of this folder is that a manual is not a project.
MANUAL_NAME = "Manual.md"
# What reading produced, one entry per page. Derived - delete it and the next sweep rebuilds it,
# at the cost of one vision call per page.
PAGES_NAME = "pages.json"
# Where the page renders go. Capitalised like a project's Photos/, because it is the same kind of
# thing: a folder of pictures somebody might open in a file browser.
PAGES_DIR = "Pages"

# A manual is only a manual if there is something to read. A folder holding a Manual.md and no
# PDF is a half-finished upload, and listing it would offer the model a manual with no pages.
SOURCE_SUFFIX = ".pdf"

MAX_ALIASES = 8
MAX_LINE_CHARS = 220  # one manual's line, and it is read every session - keep it short


@dataclass
class Manual:
    """One manual as it is on the card. Built from ``Manual.md``, never from its path."""

    key: str  # the identity, frozen at creation. Never recomputed from the folder name.
    name: str  # what the document calls itself - "mo.unit blue / basic Instruction Manual"
    path: Path  # where it is right now. Never shown to a model.
    part: str = ""  # the thing it documents - "motogadget mo.unit blue body control module"
    maker: str = ""
    revision: str = ""
    # One dense sentence: what the thing is, and what this manual can answer about it. The
    # tagline's twin - `Project.tagline` exists so a session can decide where work belongs, and
    # this exists so a session can decide whether a question is one this manual answers.
    about: str = ""
    source: str = ""  # the PDF's filename within the folder
    pages: int = 0  # how many the PDF has
    read: int = 0  # how many have been through reading.py; read < pages means still working
    added: str = ""
    aliases: list[str] = field(default_factory=list)

    @property
    def pdf(self) -> Path:
        return self.path / self.source

    @property
    def pages_dir(self) -> Path:
        return self.path / PAGES_DIR

    @property
    def identity(self) -> Path:
        return self.path / MANUAL_NAME

    @property
    def scope(self) -> str:
        """What a page of this manual is scoped to in the index, beside project: and session:."""
        return f"manual:{self.path.name}"

    @property
    def ready(self) -> bool:
        return self.read > 0 and self.read >= self.pages

    def names(self) -> set[str]:
        """Every spelling that should resolve here, folded.

        The part is in here as well as the title, which is the one difference from
        ``Project.names``: nobody asks for "the mo.unit blue / basic Instruction Manual", they ask
        about the m.unit. What the manual is *for* is the name it gets called by.
        """
        found = {self.key, fold(self.name), fold(self.path.name), fold(self.part)}
        found.update(fold(alias) for alias in self.aliases)
        return {name for name in found if name}

    def line(self) -> str:
        """One line for the list handed to a model at the start of a session.

        Shaped like ``Project.line`` and for its reason: a label and a sentence that says what the
        thing *is*. A document's own title is usually useless for deciding relevance - half of
        these call themselves "Instruction Manual" - and so is a page count. What decides whether
        to open a manual is knowing what it covers, which is what :attr:`about` holds.

        Aliases come last and only when there are any. They are not description; they are the
        spellings somebody says out loud, and they are here so the model can match what it just
        heard against this line without a lookup.
        """
        label = self.part or self.name
        if self.revision:
            label += f" {self.revision}"
        line = f"- {label}"
        if about := _unrepeated(self.about, label):
            line += f" — {about}"
        if self.aliases:
            line += f" (also: {', '.join(self.aliases[:3])})"
        # Cut at a comma, so a line that runs long ends on a whole topic rather than mid-word
        # with a dangling separator. This is a list; half an item in it is worse than one fewer.
        if len(line) > MAX_LINE_CHARS:
            line = line[:MAX_LINE_CHARS].rsplit(",", 1)[0].rstrip(" ,;—-")
        return line


def _unrepeated(about: str, label: str) -> str:
    """The description with any restatement of the label trimmed off its front.

    The identity prompt says not to repeat the product name and the model does it anyway about
    half the time - "mo.unit blue basic body control module specs, installation, wiring" against a
    label that already said all of that. Asking twice does not fix it; this does, and it costs
    nothing. Only a *leading* repeat is trimmed: the same words later in the list are a topic.
    """
    if not about:
        return ""
    known = {w for w in re.split(r"[^a-z0-9]+", label.lower()) if len(w) > 2}
    words = about.split()
    cut = 0
    for n, word in enumerate(words):
        if re.sub(r"[^a-z0-9]", "", word.lower()) in known:
            cut = n + 1
        elif n - cut > 1:  # two words in a row that are not the label: the list has started
            break
    trimmed = " ".join(words[cut:]).lstrip(" ,;:-—")
    return trimmed or about


# ------------------------------------------------------------------ reading the card


def source_in(folder: Path) -> str:
    """The name of the PDF in this folder, or "" if there is not exactly one to read."""
    try:
        found = sorted(
            p.name
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() == SOURCE_SUFFIX and not p.name.startswith(".")
        )
    except OSError:
        return ""
    return found[0] if found else ""


def _manual_from(folder: Path) -> Manual | None:
    """Read one manual off the card, or None if this folder is not one."""
    found = source_in(folder)
    if not found:
        return None  # no PDF, nothing to read; a half-finished upload is not a manual

    front: dict[str, str | list[str]] = {}
    try:
        front, _ = split_front((folder / MANUAL_NAME).read_text(encoding="utf-8", errors="replace"))
    except OSError:
        pass  # not read yet. It is still a manual - it just has nothing but its filename.

    def one(name: str) -> str:
        value = front.get(name, "")
        return value if isinstance(value, str) else ""

    def many(name: str) -> list[str]:
        value = front.get(name, [])
        return [item for item in value if item][:MAX_ALIASES] if isinstance(value, list) else []

    def count(name: str) -> int:
        try:
            return int(one(name) or 0)
        except ValueError:
            return 0

    # Same fallback as _project_from: no key means no identity, and inventing one from the folder
    # name is how a rename forks a manual. The folder's own fold is what an unread one looks like.
    key = one("key") or fold(folder.name)
    if not key:
        return None

    return Manual(
        key=key,
        name=one("manual") or folder.name,
        path=folder,
        part=one("part"),
        maker=one("maker"),
        revision=one("revision"),
        about=one("about"),
        source=one("file") or found,
        pages=count("pages"),
        read=count("read"),
        added=one("added"),
        aliases=many("aliases"),
    )


def index(manuals_dir: Path) -> list[Manual]:
    """Every manual on the card, newest first. A folder that is not one is skipped."""
    root = manuals_dir.expanduser()
    if not root.is_dir():
        return []
    found: dict[str, Manual] = {}
    try:
        folders = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError:
        return []
    for folder in folders:
        manual = _manual_from(folder)
        if manual is not None:
            found[manual.key] = manual
    return sorted(found.values(), key=lambda m: (m.added, m.name), reverse=True)


def catalog(settings: Settings) -> list[Manual]:
    """The manuals a running session should know about."""
    return index(settings.manuals_dir)


def find(manuals: list[Manual], name: str) -> Manual | None:
    """The manual someone means by ``name`` - its key, its title, the part, or any alias."""
    wanted = fold(name)
    if not wanted:
        return None
    for manual in manuals:
        if wanted in manual.names():
            return manual
    # Nothing exact. Containment catches "the m unit" against "motogadget mo.unit blue", which is
    # how people actually name the thing in front of them - the same fallback store.find makes.
    for manual in manuals:
        if any(wanted in known or known in wanted for known in manual.names() if known):
            return manual
    return None


# ------------------------------------------------------------------ the pages sidecar


def read_pages(folder: Path) -> dict[str, dict]:
    """What reading this manual produced, keyed by page number as a string.

    Never raises, for the reason :func:`cyclops.captions.read` gives about its own sidecar: a file
    somebody hand-edited into invalid JSON must cost the pages and not the sweep.
    """
    try:
        found = json.loads((folder / PAGES_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return {}
    if not isinstance(found, dict):
        return {}
    return {str(n): page for n, page in found.items() if isinstance(page, dict)}


def write_pages(folder: Path, pages: dict[str, dict]) -> None:
    """Land the sidecar whole. Sorted numerically so a diff of it reads like the manual."""
    ordered = dict(sorted(pages.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0))
    card.write_text(folder / PAGES_NAME, json.dumps(ordered, indent=1, ensure_ascii=False))


def write_identity(manual: Manual, body: str = "") -> None:
    """Land ``Manual.md``: the frontmatter is ours, the prose underneath is the model's.

    The body is kept if one is already there and none is offered, so re-reading a manual to pick
    up new pages never silently drops what the first read wrote about it.
    """
    if not body:
        try:
            _, body = split_front(manual.identity.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            body = ""
    front = render_front(
        {
            "manual": manual.name,
            "key": manual.key,
            "part": manual.part,
            "maker": manual.maker,
            "revision": manual.revision,
            "about": manual.about,
            "aliases": manual.aliases,
            "file": manual.source,
            "pages": manual.pages,
            "read": manual.read,
            "added": manual.added or date.today().isoformat(),
        }
    )
    card.write_text(manual.identity, f"{front}\n\n{body.strip()}\n")


# ------------------------------------------------------------------ taking one in


def folder_for(manuals_dir: Path, filename: str) -> Path | None:
    """Where a newly uploaded PDF should go. Named from the file, because that is all we know.

    The identity comes later, from reading page one - and it never renames this folder, for the
    reason the module docstring gives: the key in the frontmatter is what identity means here.
    """
    stem = safe_folder_name(Path(filename).stem)
    if not stem:
        return None
    return manuals_dir.expanduser() / stem


def receive(manuals_dir: Path, filename: str, chunks: Iterator[bytes], *, limit: int) -> str | None:
    """Land an uploaded PDF in a folder of its own. Returns the folder name, or None if refused.

    Refuses anything not a ``.pdf``: unlike a project folder, which is somewhere a person keeps
    whatever they like, this directory has exactly one meaning and a ``.docx`` in it would be a
    manual that can never be read.
    """
    if Path(filename).suffix.lower() != SOURCE_SUFFIX:
        return None
    folder = folder_for(manuals_dir, filename)
    if folder is None:
        return None
    # write_stream makes the parent, so a refused upload leaves no empty folder behind.
    card.write_stream(folder / f"{folder.name}{SOURCE_SUFFIX}", chunks, limit=limit)
    card.sync_dir(folder.parent)
    return folder.name
