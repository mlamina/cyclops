"""What is in the projects folder, for the page that shows it: the shelf, and what is on it.

The sibling of :mod:`cyclops.library`, and it keeps that module's two rules. No ``session.py``
import, so the admin service never loads a video encoder to list a directory; and nothing here
raises for a file it cannot read, because a project someone half-edited in a text editor must
still open in the browser rather than take the page down with it.

The layout it walks is ``projects/store.py``'s, described in full at the top of that module::

    projects/Pelican Display Mount/
        README.md   frontmatter is ours, the prose under it is the model's
        Log.md      one dated entry per session, appended and never rewritten
        Project Data.xlsx
        Photos/     what the sweep copied out of sessions, plus a captions.json beside it

Read-only, all of it. ``store.py`` is the only thing in this repo that writes into ``projects/``
and that invariant is checkable with a grep - so this module reads, classifies and renders, and
owns no path that anything writes to. The admin page can now make a folder and take an upload,
and that changed nothing here: the two endpoints resolve their destination with :func:`inside`,
the same function every read on the page goes through, and then hand it to ``store.py`` to write.
Resolving is this module's job; writing is still not.

Two things are worth knowing about the identity used here. A project's real identity is the
``key`` in its frontmatter, but what appears in every URL below is the **folder name**, exactly as
a session's folder name is its id. The key is for deciding which project a conversation belongs
to; a browser is browsing folders, and the folder is what a person renamed in Finder and expects
to find. And ``store.index`` is what decides what counts as a project at all: a directory with a
readable ``README.md``. A folder without one is not listed, which is the same answer the voice
agent gets.
"""

from __future__ import annotations

import html
import json
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import markdown

from . import card
from .projects import data, store

# What may be handed to a browser as bytes, and as what. An allow-list by suffix for the reason
# admin/views.py gives for its own: the set of legal answers is short enough to write down, and
# writing it down is the end of every argument about what some other name might resolve to. PNG
# is here and not in the session list because a project folder is a place a person drops files.
MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".mp4": "video/mp4",
}
# What the gallery can hold. Derived from the set above rather than written out again, so a new
# picture format only has to be admitted once - and .mp4 is out because a grid of <img> cannot
# show a recording, not because a project may not have one. It is still browsable in FILES.
IMAGE_TYPES = frozenset(suffix for suffix in MEDIA_TYPES if suffix != ".mp4")
MAX_PICTURES = 500  # library.STREAM_LIMIT's sibling: a ceiling, not a page size

# Shown as words rather than served as bytes. `.md` is not here: it has its own renderer, and a
# markdown file that failed to render falls back to this set's treatment anyway.
TEXT_TYPES = frozenset({".txt", ".csv", ".json", ".log", ".yml", ".yaml", ".md"})
MAX_TEXT_BYTES = 256 * 1024  # a Log.md with a hundred entries is ~40 KB; this is a ceiling

MARKDOWN_EXTENSIONS = ["tables", "fenced_code", "sane_lists"]

# Every HTML comment. Log.md's `<!-- cyclops:session ... -->` terminators are bookkeeping that no
# reader wants in the middle of an entry, and taking them out first also means the escape below
# has nothing legitimate left to swallow.
COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
# A relative src or href in the rendered output - one that is not already absolute, a scheme, or
# a fragment. Those are the ones that mean "a file next to me in this project".
RELATIVE = re.compile(r'\b(src|href)="(?!https?:|/|#|mailto:|data:)([^"]*)"')


# ------------------------------------------------------------------ the shelf


def _thumb(folder: Path) -> str:
    """The first photo in a project, for its row in the list - or "" if it has none."""
    photos = folder / store.PHOTOS
    try:
        found = sorted(p for p in photos.iterdir() if p.is_file() and not p.name.startswith("."))
    except OSError:
        return ""
    for picture in found:
        if picture.suffix.lower() in MEDIA_TYPES:
            return media_url(folder.name, f"{store.PHOTOS}/{picture.name}")
    return ""


def projects(projects_dir: Path) -> list[dict]:
    """Every project on the card, most recently worked on first.

    ``store.index`` does the reading and the ordering; this only decides what a page is shown.
    ``Project.path`` deliberately does not survive the trip - the folder name is the id, and an
    absolute path in a JSON body is a detail of this box that the page has no use for.
    """
    out = []
    for project in store.index(projects_dir):
        out.append(
            {
                "name": project.path.name,  # the id in every URL below
                "title": project.name,
                "tagline": project.tagline,
                "status": project.status,
                "updated": project.updated,
                "sessions": project.sessions,
                "photos": project.photos,
                "thumb": _thumb(project.path),
            }
        )
    return out


def resolve(projects_dir: Path, name: str) -> Path | None:
    """The project folder called ``name``, or None. ``library.resolve``, on the other root."""
    root = projects_dir.expanduser().resolve()
    try:
        found = (root / name).resolve()
    except OSError:
        return None
    if found.parent != root or not found.is_dir():
        return None
    return found


def inside(folder: Path, relative: str) -> Path | None:
    """One path inside one project, or None - and never a path outside it.

    The one place this parts company with :func:`cyclops.library.resolve`, which requires a direct
    child. A browser has to descend: ``Photos/`` is a folder you open, and a project may hold
    folders of your own that nothing here put there. So the containment is ``is_relative_to``
    rather than a parent comparison - still one check after one resolution, and still closing
    ``..``, an absolute name and a symlink out of the tree all at once.
    """
    try:
        found = (folder / relative).resolve() if relative else folder.resolve()
    except OSError:
        return None
    root = folder.resolve()
    if found != root and not found.is_relative_to(root):
        return None
    return found


def media_url(name: str, relative: str) -> str:
    """Where the page fetches one file out of one project."""
    return f"/project-media/{quote(name)}/{quote(relative)}"


def file_route(name: str, relative: str) -> str:
    """Where the page *goes* to show one file - a hash route, never an href.

    Both segments are encoded whole, slashes included, so ``Photos/x.jpg`` stays one segment and
    the router's split is unambiguous however deep the file is.
    """
    return f"#/f/{quote(name, safe='')}/{quote(relative, safe='')}"


# ------------------------------------------------------------------ one directory


def _when(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    except OSError:
        return ""


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def listing(folder: Path, relative: str = "") -> dict | None:
    """What is in one directory of one project: folders first, then files, each by name.

    Dotfiles are left out. The card's atomic writes go through ``card.write_bytes``, which stages
    at ``.<name>.tmp`` beside the target - so a power cut mid-save leaves one behind, and a file
    browser that showed it would be reporting our own bookkeeping as the project's contents.
    """
    here = inside(folder, relative)
    if here is None or not here.is_dir():
        return None
    try:
        found = list(here.iterdir())
    except OSError:
        return None

    entries = []
    for item in found:
        if item.name.startswith("."):
            continue
        directory = item.is_dir()
        entries.append(
            {
                "name": item.name,
                "path": f"{relative}/{item.name}" if relative else item.name,
                "kind": "dir" if directory else "file",
                "size": 0 if directory else _size(item),
                "when": _when(item),
                "suffix": "" if directory else item.suffix.lower(),
            }
        )
    entries.sort(key=lambda e: (e["kind"] != "dir", e["name"].lower()))
    return {
        "path": relative,
        "parent": relative.rsplit("/", 1)[0] if "/" in relative else "",
        "entries": entries,
    }


# ------------------------------------------------------------------ every picture


def _captions(folder: Path) -> dict[str, str]:
    """What the index service wrote about the pictures in one folder, or nothing at all.

    :func:`cyclops.captions.read` is this function, and it is deliberately not imported: that
    module reaches for ``openai`` at import time, and the admin service has no more business
    loading an API client to caption a grid than it has loading a video encoder to list a
    directory. Same argument the module docstring makes about ``session.py``, one import along.
    """
    try:
        found = json.loads((folder / card.CAPTIONS_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return {}
    if not isinstance(found, dict):
        return {}
    return {str(k): str(v) for k, v in found.items() if isinstance(v, str) and v.strip()}


def pictures(name: str, folder: Path) -> list[dict]:
    """Every picture anywhere in one project, newest first.

    Not only ``Photos/``. That folder holds what the sweep copied out of sessions, and a project
    somebody actually works in also has an ``Eye Designs/`` and a ``UI Evolution/`` full of PNGs -
    which are pictures of the project by any reading a person would give the word. So the whole
    tree is walked, and the only thing kept out is what a grid of ``<img>`` cannot show.

    ``rglob`` does not descend a symlinked directory, so this cannot walk out of the project; the
    bytes themselves are still guarded by :func:`inside` when the page comes back for them.

    The shape is :class:`cyclops.library.Item`'s, so the gallery on the page is one renderer and
    not two.
    """
    captions: dict[Path, dict[str, str]] = {}
    out = []
    for found in folder.rglob("*"):
        if found.suffix.lower() not in IMAGE_TYPES or not found.is_file():
            continue
        relative = found.relative_to(folder)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if found.parent not in captions:
            captions[found.parent] = _captions(found.parent)
        # Two fields and not one. ``title`` is alt text and always says something; ``caption`` is
        # only there when the index service has actually looked at the picture, so the page can
        # print it without having to guess whether it is a description or a filename repeated.
        caption = captions[found.parent].get(found.name, "")
        out.append(
            {
                "kind": "photo",
                "title": caption or found.name,
                "caption": caption,
                "when": _when(found),
                "url": media_url(name, relative.as_posix()),
                "path": relative.as_posix(),
            }
        )
    out.sort(key=lambda one: one["when"], reverse=True)
    return out[:MAX_PICTURES]


# ------------------------------------------------------------------ one file


def render_markdown(text: str, name: str) -> str:
    """One markdown file as HTML, with its links pointed somewhere this page can follow.

    The frontmatter goes first. It is our machine state - ``key``, ``sessions``, ``last`` - and
    ``store.py`` is emphatic that the model never owns it; a reader has no more use for it than
    the model does, and rendered it is just a paragraph of field names.

    Then every ``<`` that survives the comment strip is escaped, so the only tags in the output
    are ones Python-Markdown generated itself. These files are ours and yours, on a private LAN,
    and they are still the one thing here a browser would happily execute - the same argument the
    SVG response makes for carrying a CSP header it will almost certainly never need.
    """
    _, body = store.split_front(text)
    body = COMMENT.sub("", body)
    body = body.replace("<", "&lt;")
    rendered = markdown.markdown(body, extensions=MARKDOWN_EXTENSIONS)

    def point(match: re.Match[str]) -> str:
        attribute, target = match.group(1), html.unescape(match.group(2))
        if attribute == "src":
            return f'src="{media_url(name, target)}"'
        # A link to another file in the project - the README's footer links Log.md - stays inside
        # the page. The kiosk's browser must never navigate (kiosk.py:112-116), so this is a hash
        # route and not an href to anywhere.
        return f'href="{file_route(name, target)}"'

    return RELATIVE.sub(point, rendered)


def _sheet(path: Path) -> list[dict]:
    """A workbook as its tabs, in sheet order, each with its rows in theirs.

    :mod:`cyclops.projects.data` already holds exactly this shape and already promises that an
    unreadable workbook is an empty book rather than an exception - and it takes bytes and never
    touches the disk, which is what lets this read any workbook in the folder and not only
    ``Project Data.xlsx``. All that is added here is the grouping: ``tabs()`` for the order, which
    keeps a sheet someone made and has not filled in yet, and ``rows()`` for the contents.
    """
    book = data.load(_bytes(path))
    grouped: dict[str, list[dict]] = {tab: [] for tab in book.tabs()}
    for row in book.rows():
        grouped.setdefault(row.tab, []).append(
            {"key": row.key, "value": row.value, "note": row.note}
        )
    return [{"name": tab, "rows": rows} for tab, rows in grouped.items()]


def _bytes(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except OSError:
        return None


def _text(path: Path) -> str:
    """A file as words, capped. ``errors="replace"`` for the same reason ``store`` uses it."""
    try:
        with path.open("rb") as handle:
            blob = handle.read(MAX_TEXT_BYTES)
    except OSError:
        return ""
    return blob.decode("utf-8", errors="replace")


def view(name: str, folder: Path, relative: str) -> dict | None:
    """One file, in whatever shape it is worth showing in - or None if there is no such file.

    The suffix decides, and every branch has an answer: a format with no viewer here is still a
    file with a size, and saying so is better than a blank screen or a download the panel's
    browser has nowhere to put.
    """
    found = inside(folder, relative)
    if found is None or not found.is_file():
        return None
    suffix = found.suffix.lower()
    base = {"name": found.name, "path": relative, "size": _size(found)}

    if suffix == ".md":
        return {**base, "kind": "markdown", "html": render_markdown(_text(found), name)}
    if suffix == ".xlsx":
        return {**base, "kind": "sheet", "tabs": _sheet(found)}
    if suffix in MEDIA_TYPES:
        kind = "video" if suffix == ".mp4" else "image"
        return {**base, "kind": kind, "url": media_url(name, relative)}
    if suffix in TEXT_TYPES:
        return {**base, "kind": "text", "text": _text(found)}
    return {**base, "kind": "none"}
