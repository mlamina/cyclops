"""Everything on the card, flattened into things you could ask for, and the index over them.

This is the read half of recall. :mod:`cyclops.indexer` is the write half - it watches the card,
captions what has no caption, and rewrites the index file this module loads. Nothing here writes
anything, and the voice agent only ever calls into this side: a recall costs one embedding call
for the query and a dot product, and never a disk walk or a model call for anything else.

**This codebase used to say it had no index.** ``library.py`` said so in as many words and
``store.py`` said it from the other side, and both were right for as long as every question could
be answered by reading one folder. "Show me the picture of the torque spec" is not such a question
- you cannot embed a query against a directory listing - so the rule is retired and those two
docstrings are updated rather than left arguing with the code.

What survives is the property the rule was protecting: **the card is still the only thing that
holds a fact.** Every item below is derived from a file in ``sessions/`` or ``projects/``, the
index holds nothing that is not, and ``rm ~/.cache/cyclops/recall.npz`` costs a rebuild and
nothing else. That is worth keeping true, because it is what lets the whole of this be wrong
without anything being lost.

Two constraints inherited from the neighbours, and for their reasons:

* **No** :mod:`cyclops.session` **import.** It pulls in ``record.py`` and with it OpenCV, and this
  module is loaded by a background service that only wants to read directories. ``library.py`` and
  ``shelf.py`` both keep this rule; ``card.py`` is the dependency-free module all three share.
* **Nothing raises for a file it cannot read.** A half-edited ``Log.md`` is a skipped item, never
  a failed sweep.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import card
from .config import RECALL_FILE, Settings

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMS = 1536
# The API takes a list; this is how many items go in one request. Well above anything this card
# will hold in a year, so a first build is one call - which is the point, on a Pi over wifi.
EMBED_BATCH = 128
EMBED_TIMEOUT_S = 60.0

MAX_TEXT_BYTES = 256 * 1024  # shelf.MAX_TEXT_BYTES, duplicated for the no-shelf-import reason
CHUNK_CHARS = 1000  # one paragraph-ish run of a dropped text file
MAX_CHUNKS = 24  # per file, so one large paste cannot dominate the index
MAX_ITEM_CHARS = 4000  # what any one item contributes to an embedding

# What can be read as words, and what can be shown as a picture. Deliberately not shelf.py's
# tables: that module decides what a *browser* may be handed, which is a different question with a
# different answer (it serves .mp4 and .svg, neither of which is searchable or showable here).
TEXT_SUFFIXES = frozenset({".txt", ".md", ".csv", ".json", ".log", ".yml", ".yaml"})
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})

# Files that are somebody's bookkeeping rather than content. README.md and Log.md are read as
# structured sources below and must not also be swept up as plain text, or every project would be
# in the index twice with the second copy chunked at arbitrary boundaries.
SKIP_NAMES = frozenset({
    card.CAPTIONS_NAME,
    "README.md",
    "Log.md",
    card.LOG_NAME,
    card.PAGE_NAME,
    card.RECEIPT_NAME,
})

# One `![caption](path)` out of a Log.md or a README. This is where the filing curator's caption
# for a hero shot lives, and it is often better than anything a picture alone would produce -
# it says why the photo was taken, which no amount of looking at it reveals.
IMAGE_LINK = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
# One entry in a Log.md: a `## ` heading and everything under it up to the next one.
ENTRY_SPLIT = re.compile(r"^## ", re.MULTILINE)
# The bookkeeping terminator store.py appends to every log entry. Never wanted in index text.
COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


@dataclass(frozen=True)
class Item:
    """One thing on the card you could ask for, and everything needed to find and show it."""

    kind: str  # "photo" | "image" | "entry" | "file"
    path: str  # absolute, as a string, because this round-trips through JSON
    scope: str  # "project:<folder name>" | "session:<folder name>"
    title: str  # what to call it out loud
    text: str  # what gets embedded
    mtime_ns: int
    size: int
    mark: str = ""  # content_hash, for pictures only: the same photo in two folders is one thing

    @property
    def key(self) -> str:
        """Identity for the reconcile diff. A chunked file yields several items per path."""
        return f"{self.kind}:{self.path}:{self.title}"

    @property
    def stamp(self) -> tuple[int, int]:
        """What changed-ness is judged on - the same pair ``library._cache`` keys on."""
        return (self.mtime_ns, self.size)

    @property
    def showable(self) -> bool:
        return self.kind in {"photo", "image"}


# ------------------------------------------------------------------ reading the card


# One picture's content, so that the same picture in two places is one thing. Keyed on the pair
# that already means "this file changed", so a reconcile that finds nothing new re-reads nothing.
# Per process and gone on restart, exactly like ``library._cache`` and for the same reasons.
_hashes: dict[tuple[str, int, int], str] = {}


def content_hash(path: Path) -> str:
    """What is *in* a file, as a short digest. ``""`` if it cannot be read.

    The filing curator copies a hero shot into a project, so the identical JPEG lives in both the
    session folder and the project folder. Before this existed they were two items: they took the
    top two slots of one search between them, and - worse - each got its own caption call, whose
    two readings of one dense manual page disagreed about the numbers on it. Content is the only
    identity that sees through that; a name cannot, because ``copy_photos`` renames as it copies.
    """
    try:
        info = path.stat()
    except OSError:
        return ""
    key = (str(path), info.st_mtime_ns, info.st_size)
    found = _hashes.get(key)
    if found is not None:
        return found
    digest = hashlib.blake2b(digest_size=16)
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError:
        return ""
    _hashes[key] = mark = digest.hexdigest()
    return mark


def _stat(path: Path) -> tuple[int, int]:
    try:
        info = path.stat()
    except OSError:
        return (0, 0)
    return (info.st_mtime_ns, info.st_size)


def _text_of(path: Path) -> str:
    """One text file, read defensively and capped. ``""`` for anything that will not read."""
    try:
        if path.stat().st_size > MAX_TEXT_BYTES:
            return path.read_bytes()[:MAX_TEXT_BYTES].decode("utf-8", errors="replace")
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _chunks(text: str) -> list[str]:
    """A text file split into runs small enough to embed one meaning at a time.

    On blank lines first, because a person's own notes are paragraphs and a paragraph is the unit
    of "one thing said". Runs are then packed up to ``CHUNK_CHARS`` rather than emitted one per
    paragraph, so a file of one-line bullets does not become forty items each too short to carry
    any signal.
    """
    out: list[str] = []
    current = ""
    for block in re.split(r"\n\s*\n", text):
        block = " ".join(block.split())
        if not block:
            continue
        if len(current) + len(block) + 1 <= CHUNK_CHARS:
            current = f"{current} {block}".strip()
            continue
        if current:
            out.append(current)
        # A single paragraph longer than the budget is cut on a word boundary rather than dropped.
        while len(block) > CHUNK_CHARS:
            cut = block[:CHUNK_CHARS].rsplit(" ", 1)[0] or block[:CHUNK_CHARS]
            out.append(cut)
            block = block[len(cut) :].strip()
        current = block
    if current:
        out.append(current)
    return out[:MAX_CHUNKS]


def _captions_for(folder: Path) -> dict[str, str]:
    from . import captions

    return captions.read(folder)


def _log_captions(project: Path) -> dict[str, str]:
    """Every ``![caption](Photos/x.jpg)`` in a project's Log.md and README, keyed by filename.

    The curator's own words about a picture, which say why it mattered rather than what is in it.
    Both halves are worth having and they are concatenated: "Discussing the missing torque spec"
    and "a torque table showing 60-65 Nm" answer different questions about the same photo.
    """
    found: dict[str, str] = {}
    for name in ("Log.md", "README.md"):
        for caption, target in IMAGE_LINK.findall(_text_of(project / name)):
            caption = " ".join(caption.split())
            if not caption:
                continue
            key = Path(target.replace("%20", " ")).name
            if key and key not in found:
                found[key] = caption
    return found


def _images_in(folder: Path, scope: str, log_captions: dict[str, str], kind: str) -> list[Item]:
    """Every picture in one folder, described by whatever words exist about it."""
    if not folder.is_dir():
        return []
    captions = _captions_for(folder)
    out: list[Item] = []
    try:
        listing = sorted(folder.iterdir())
    except OSError:
        return []
    for path in listing:
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.name.startswith("."):
            continue
        parts = [captions.get(path.name, ""), log_captions.get(path.name, "")]
        text = ". ".join(p.strip(" .") for p in parts if p.strip())
        if not text:
            # Nothing has described it yet - the index service will, on a later pass. Its folder
            # and filename are the only words that exist about it, and they are worth something:
            # "Eye Designs/eye-round6.png" is a real answer to "the sixth eye sketch".
            text = f"{folder.name} {path.stem}".replace("-", " ").replace("_", " ")
        mtime, size = _stat(path)
        out.append(
            Item(
                kind=kind,
                path=str(path),
                scope=scope,
                title=captions.get(path.name, "") or log_captions.get(path.name, "") or path.name,
                text=text[:MAX_ITEM_CHARS],
                mtime_ns=mtime,
                size=size,
                mark=content_hash(path),
            )
        )
    return out


def _entries_of(project: Path, scope: str) -> list[Item]:
    """One item per ``## `` entry in a project's Log.md - the prose, not just the heading."""
    log = project / "Log.md"
    text = COMMENT.sub("", _text_of(log))
    if not text.strip():
        return []
    mtime, size = _stat(log)
    out: list[Item] = []
    for block in ENTRY_SPLIT.split(text)[1:]:  # [0] is the file's own "# Name - log" preamble
        lines = block.strip().splitlines()
        if not lines:
            continue
        heading = " ".join(lines[0].split())
        body = " ".join(" ".join(lines[1:]).split())
        # The image links are the curator's photo captions; they are already carried by the photo
        # items themselves, and leaving the raw markdown in here would embed a path as if it were
        # a sentence.
        body = IMAGE_LINK.sub(r"\1", body)
        out.append(
            Item(
                kind="entry",
                path=str(log),
                scope=scope,
                title=heading,
                text=f"{heading}. {body}"[:MAX_ITEM_CHARS],
                mtime_ns=mtime,
                size=size,
            )
        )
    return out


def _readme_of(project: Path, scope: str) -> list[Item]:
    """The project's own page: what it is and where it stands."""
    readme = project / "README.md"
    text = _text_of(readme)
    if not text.strip():
        return []
    from .projects.store import split_front

    _, body = split_front(text)
    body = IMAGE_LINK.sub(r"\1", COMMENT.sub("", body))
    body = " ".join(body.split())
    if not body:
        return []
    mtime, size = _stat(readme)
    return [
        Item(
            kind="entry",
            path=str(readme),
            scope=scope,
            title=f"Where {project.name} stands",
            text=body[:MAX_ITEM_CHARS],
            mtime_ns=mtime,
            size=size,
        )
    ]


def _files_in(root: Path, scope: str) -> list[Item]:
    """Every readable text file anywhere under a folder - the things you dropped there yourself.

    Walked recursively, because ``Datasheets/bolts.txt`` is the shape a person actually files
    things in. ``Photos/`` is skipped because its contents are read as pictures above.
    """
    out: list[Item] = []
    try:
        listing = sorted(root.rglob("*"))
    except OSError:
        return []
    for path in listing:
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.name in SKIP_NAMES or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if any(part in {"Photos", card.PHOTOS} for part in path.parts):
            continue
        text = _text_of(path)
        if not text.strip():
            continue
        mtime, size = _stat(path)
        where = path.relative_to(root)
        for n, chunk in enumerate(_chunks(text)):
            out.append(
                Item(
                    kind="file",
                    path=str(path),
                    scope=scope,
                    title=str(where) if n == 0 else f"{where} ({n + 1})",
                    text=f"{where}. {chunk}"[:MAX_ITEM_CHARS],
                    mtime_ns=mtime,
                    size=size,
                )
            )
    return out


def project_items(folder: Path) -> list[Item]:
    """Everything in one project folder that could be asked for."""
    scope = f"project:{folder.name}"
    log_captions = _log_captions(folder)
    items = _readme_of(folder, scope) + _entries_of(folder, scope)
    items += _images_in(folder / "Photos", scope, log_captions, "photo")
    items += _files_in(folder, scope)
    # Pictures a person dropped into a folder of their own - "Eye Designs/", say. Photos/ is
    # already done above, so it is skipped rather than re-read.
    try:
        others = [p for p in sorted(folder.iterdir()) if p.is_dir()]
    except OSError:
        others = []
    for sub in others:
        if sub.name == "Photos":
            continue
        items += _images_in(sub, scope, log_captions, "image")
    return items


def session_items(folder: Path) -> list[Item]:
    """A session's photos, and the summary somebody already wrote of the conversation."""
    scope = f"session:{folder.name}"
    items = _images_in(folder / card.PHOTOS, scope, {}, "photo")
    summary = folder / card.SUMMARY_NAME
    text = " ".join(_text_of(summary).split())
    if text:
        mtime, size = _stat(summary)
        items.append(
            Item(
                kind="entry",
                path=str(summary),
                scope=scope,
                title=text.lstrip("# ").split(".")[0][:120],
                text=text.lstrip("# ")[:MAX_ITEM_CHARS],
                mtime_ns=mtime,
                size=size,
            )
        )
    return items


def corpus(settings: Settings, *, collapse: bool = True) -> list[Item]:
    """Everything on the card, as items. One walk, no model, no network.

    ``collapse=False`` keeps both copies of a duplicated picture. Only :func:`image_folders` wants
    that, and it needs it: the folders are what the captioner is pointed at, and a folder whose
    one photograph happens to be a duplicate of another still has to be visited - to caption it,
    and to prune captions for pictures that have been deleted from it.
    """
    items: list[Item] = []
    for root, reader in (
        (settings.projects_dir, project_items),
        (settings.sessions_dir, session_items),
    ):
        folder = root.expanduser()
        if not folder.is_dir():
            continue
        try:
            found = sorted(p for p in folder.iterdir() if p.is_dir())
        except OSError:
            continue
        for one in found:
            try:
                items.extend(reader(one))
            except OSError:
                continue  # one unreadable folder is never a reason to lose the rest
    return dedupe(items) if collapse else items


def dedupe(items: list[Item]) -> list[Item]:
    """One row per picture, however many folders it sits in. Everything else passes through.

    ``copy_photos`` copies a hero shot into its project, so the identical JPEG is on the card
    twice and was in the index twice - taking the first two places of one search between them and
    telling the model about an alternative that was the photograph it had just put on the screen.

    The project's copy wins. Both are the same pixels, so the tie is broken on which is the better
    thing to hand somebody: a project folder is curated and kept, while a session folder is a
    working record that :func:`cyclops.session.main` can be asked to delete.
    """
    best: dict[str, Item] = {}
    out: list[Item] = []
    for item in items:
        if not item.showable or not item.mark:
            out.append(item)
            continue
        found = best.get(item.mark)
        if found is None or (
            not found.scope.startswith("project:") and item.scope.startswith("project:")
        ):
            best[item.mark] = item
    return out + list(best.values())


def image_folders(settings: Settings) -> list[Path]:
    """Every folder on the card that holds pictures - what the captioner is pointed at.

    Derived from the same walk as :func:`corpus` rather than from a second set of rules, so a
    picture that can be indexed is by construction a picture that can be captioned.
    """
    seen: dict[Path, None] = {}
    for item in corpus(settings, collapse=False):
        if item.showable:
            seen.setdefault(Path(item.path).parent, None)
    return list(seen)


# ------------------------------------------------------------------ the index file


@dataclass(frozen=True)
class Index:
    """The items and their vectors, in step. Row ``n`` of ``vectors`` describes ``items[n]``."""

    items: list[Item]
    vectors: np.ndarray  # float32 [n, dims], L2-normalised

    def __len__(self) -> int:
        return len(self.items)


def empty() -> Index:
    return Index(items=[], vectors=np.zeros((0, EMBED_DIMS), dtype=np.float32))


def dump(index: Index) -> bytes:
    """One index as the bytes of an ``.npz``. Built in memory so it can be landed atomically."""
    buffer = io.BytesIO()
    np.savez(
        buffer,
        vectors=index.vectors.astype(np.float32, copy=False),
        items=np.array(json.dumps([asdict(i) for i in index.items]), dtype=object),
    )
    return buffer.getvalue()


def load(path: Path | None = None) -> Index:
    """The index off the disk, or an empty one.

    ``path`` is resolved at call time rather than defaulted in the signature. A default argument is
    evaluated once, at import, which would freeze the location for the life of the process and make
    the module untestable against a temporary file - and would have silently kept reading the real
    index while a reconcile wrote somewhere else.

    Never raises. A missing file is the ordinary state before the service has ever run, and a
    corrupt one is a cache that will be rebuilt - neither is a reason to fail a recall, and both
    look the same to the caller: nothing found.
    """
    try:
        blob = (path or RECALL_FILE).read_bytes()
    except OSError:
        return empty()
    try:
        with np.load(io.BytesIO(blob), allow_pickle=True) as found:
            vectors = np.asarray(found["vectors"], dtype=np.float32)
            raw = json.loads(str(found["items"].item()))
    except Exception:  # noqa: BLE001 - every way a cache file can be wrong ends here
        return empty()
    if not isinstance(raw, list):
        return empty()
    items = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            items.append(Item(**entry))
        except TypeError:
            continue  # written by an older build with a different shape; it will be rebuilt
    if len(items) != vectors.shape[0]:
        return empty()  # the two halves disagree, which must not be possible - rebuild
    return Index(items=items, vectors=vectors)


def normalise(vectors: np.ndarray) -> np.ndarray:
    """Unit rows, so a cosine similarity is a dot product and nothing divides at query time."""
    if vectors.size == 0:
        return vectors.astype(np.float32, copy=False)
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    return (vectors / lengths).astype(np.float32, copy=False)


# ------------------------------------------------------------------ searching it


@dataclass(frozen=True)
class Hit:
    item: Item
    score: float


def rank(index: Index, query: np.ndarray, *, scopes: set[str] | None, limit: int) -> list[Hit]:
    """The best items for an already-embedded query. Pure numpy - microseconds at this size.

    ``scopes`` of ``None`` searches everything; otherwise only items whose scope is in the set.
    The filter is applied to the scores rather than to the matrix, because slicing a 40-row array
    costs more than scoring all of it.
    """
    if len(index) == 0 or query.size == 0:
        return []
    scores = index.vectors @ normalise(query.reshape(1, -1))[0]
    order = np.argsort(-scores)
    out: list[Hit] = []
    for n in order:
        item = index.items[int(n)]
        if scopes is not None and item.scope not in scopes:
            continue
        out.append(Hit(item=item, score=float(scores[int(n)])))
        if len(out) >= limit:
            break
    return out


# ------------------------------------------------------------------ turning words into vectors

# Below this a "best match" is not a match. Cosine on text-embedding-3-small puts unrelated pairs
# around 0.1-0.25 and genuinely related ones from about 0.35 up, so this sits just under the gap.
# Getting it wrong in the generous direction is the worse failure: a confidently wrong picture on
# the panel is worse than "I could not find that", because only one of the two can be argued with.
MIN_SCORE = 0.30


async def embed(texts: list[str], client) -> np.ndarray:
    """Embed a list of strings, in batches. ``[n, dims]``, unit rows.

    Takes an open client rather than settings, for the reason
    :func:`cyclops.captions.describe` gives: the caller embeds a run of things in one pass, and a
    TLS handshake per item is the wrong shape on a Pi over wifi.

    Raises on failure rather than returning empty. Unlike a caption, a half-finished embedding
    would put rows in the index that do not describe the items beside them, and the caller must
    be able to tell "nothing to do" from "this did not work".
    """
    if not texts:
        return np.zeros((0, EMBED_DIMS), dtype=np.float32)
    rows: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH):
        batch = [text or " " for text in texts[start : start + EMBED_BATCH]]
        response = await client.embeddings.create(model=EMBED_MODEL, input=batch)
        rows.extend(item.embedding for item in response.data)
    return normalise(np.asarray(rows, dtype=np.float32))
