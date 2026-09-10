"""What a picture is of, in words, so that something can search for it later.

A photo on this card carries no description of itself. The ``photo`` record in ``session.jsonl``
says how many bytes it is and what Cyclops was asked to look at - ``focus``, which is usually
something like "current camera view" - and nothing at all about what is in the frame. The filing
curator writes a real caption for the two or three hero shots it copies into a project, and those
land in ``Log.md``; every other picture on the card is undescribed, and an image you drag onto the
project folder yourself has nothing whatsoever.

That is the gap this closes, under one rule:

    **An image with no caption gets one, whatever put it there.**

Stated once and applied by one caller - the index service - because there are three ways a picture
reaches the card (the shutter, the filing curator, your laptop) and a hook on any one of them
would catch a third of the problem. See :mod:`cyclops.indexer`.

The caption is *content*, not machine state, so it lives in the data tree beside the picture it
describes rather than in the index. Delete the index and the captions survive; that is the point.

One file per folder rather than a sidecar per image. A session's ``photos/`` holds a handful of
JPEGs and reading eight small files to caption eight pictures is eight times the syscalls for no
gain - and a folder full of ``16-48-48_cyclops.json`` next to ``16-48-48_cyclops.jpg`` is exactly
the kind of clutter ``sessions/`` exists not to have.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from openai import APIError, AsyncOpenAI

from . import around, card
from .config import Settings

CAPTION_MODEL = "gpt-5.4-nano"
# Above the worst case seen for a single image and well under the reconcile's own patience. This
# runs niced in a background service that nobody is waiting on, so the budget is about not wedging
# a sweep on one unreadable file rather than about anyone's attention.
CAPTION_TIMEOUT_S = 20.0
MAX_CAPTION_CHARS = 400

# What a picture may be, for a caption. The panel and the shelf both keep their own lists for
# their own reasons; this one is about what an image model will accept.
SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})

MEDIA_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}

# A picture bigger than this is downscaled before it is sent. Nothing here needs the full frame -
# the caption is a sentence - and a 1024px edge is what webcam.py already decided was enough for a
# model to read a label off.
MAX_EDGE = 1024

CAPTION_PROMPT = """\
Describe this picture in one or two sentences, for someone who will later search for it by
describing it from memory. You are not talking to them: what you write is index text.

So write down what is *visible and specific*, and nothing else:

- Every number, unit, part name, model code and measurement you can actually read in the frame.
- If there is text on a label, a page, a plate or a screen, say what it says. That is usually the
  single most searchable thing in the picture.
- Name the object, and its state if that is what the picture is about - stripped, half-assembled,
  mounted, leaking.
- Plain prose. No markdown, no preamble, no "this image shows", no guessing at what it is for.

"A photo of a motorcycle" is useless. "A torque table in a BMW R80RT manual, front caliper cap
screws 60-65 Nm" is the whole job. If the picture is genuinely of nothing - a blurred frame, a
dark room - say that plainly in a few words rather than inventing detail."""

# What was going on at the bench, when the session log knows. Sent as its own block after the
# prompt above rather than folded into it, so it stays obvious which half is instruction and
# which half is data - and so a picture nobody said anything about is described by the exact
# bytes that described every picture before this existed.
#
# The framing carries the whole risk here. A model handed a transcript will write down what it
# was told rather than what it can see, which is the failure CAPTION_PROMPT spends four bullets
# preventing, and a caption that invents a part number is worse than one that says "a blue
# motorcycle". Hence: a glossary, not a source of facts; a worked example that is a partial view
# named correctly, because that is the actual case - one corner of a trike, one caliper off a
# bike; and one line settling the conflict a model would otherwise settle the wrong way.
CONTEXT_PROMPT = """\
Here is what was going on at the bench when this was taken. It is a glossary, not a source of
facts. Use it only to call what you can see by its right name - the machine, the project, the
part, the model - in the words they used for it, and to say which of them this belongs to.

Everything you write must still be visible in the frame. Where the words name a whole thing and
the frame holds one corner of it, write the corner and say whose it is: "the front hub motor on
the Burning Man trike, wheel out and fork legs bare". Write down nothing that is only in the
words - no plans, no numbers you cannot read here, no parts that are out of shot - and where the
words and the picture disagree, the picture is right.

{words}"""


def prompt_for(words: str) -> str:
    """What to send with one picture. The bare prompt, byte for byte, when nothing was said."""
    if not words.strip():
        return CAPTION_PROMPT
    return f"{CAPTION_PROMPT}\n\n{CONTEXT_PROMPT.format(words=words.strip())}"


def read(folder: Path) -> dict[str, str]:
    """Every caption written for one folder. A missing or damaged file is an empty answer.

    Never raises, for the reason :func:`cyclops.panel.read` gives about its own sidecars: a
    caption file somebody hand-edited into invalid JSON must cost the captions and not the sweep.
    """
    try:
        found = json.loads((folder / card.CAPTIONS_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return {}
    if not isinstance(found, dict):
        return {}
    return {str(k): str(v) for k, v in found.items() if isinstance(v, str) and v.strip()}


def write(folder: Path, captions: dict[str, str]) -> None:
    """Replace a folder's captions, whole or not at all.

    Through :func:`cyclops.card.write_text` like every other byte in the data tree. Rewritten
    rather than merged on disk: the caller holds the whole map anyway, and a merge here would
    need a lock that the "written whole or not at all" guarantee is precisely there to avoid.
    """
    card.write_text(folder / card.CAPTIONS_NAME, json.dumps(captions, indent=1, sort_keys=True))


def uncaptioned(folder: Path) -> list[Path]:
    """The images in one folder that nothing has described yet. The queue, and it is the folder.

    The same shape as every other pending-work question in this codebase - a session with no
    ``summary.md`` is one still to name, a session with no ``project.md`` is one still to file -
    and it has the same property: a crash costs a retry and never a lost picture.
    """
    known = read(folder)
    return [p for p in images(folder) if p.name not in known and card.written(p)]


def images(folder: Path) -> list[Path]:
    """Every picture in one folder, described or not. What ``--recaption`` works through."""
    if not folder.is_dir():
        return []
    try:
        return sorted(
            p
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in SUFFIXES and not p.name.startswith(".")
        )
    except OSError:
        return []


def _mark(path: Path) -> str:
    """What this picture contains, for matching it against one described under another name."""
    from .recall import content_hash

    return content_hash(path)


def departed(folder: Path, captions: dict[str, str]) -> list[str]:
    """Captions describing pictures that are no longer in the folder. The mirror of the above.

    This file's whole claim is that it describes what is in this folder, so an entry for a deleted
    photo makes that claim false - the same small lie :func:`cyclops.projects.store.photo_count`
    refuses to tell about its own number. Cheap to avoid, because the map is rewritten whole
    anyway.

    Judged on the name being absent from the listing rather than on :func:`cyclops.card.written`:
    a picture halfway through being written is still a picture, and dropping its caption would
    mean paying a model to make the same one again a second later.
    """
    try:
        here = {p.name for p in folder.iterdir()}
    except OSError:
        # Cannot list it, so prune nothing. This is the case that matters: a folder that has been
        # renamed out from under a sweep - which session.describe does to every session it names -
        # would otherwise look like a folder whose every picture had just been deleted, and one
        # pass would throw away every caption in it.
        return []
    return [name for name in captions if name not in here]


def _data_url(path: Path) -> str | None:
    """One image, small enough to send, as a data URL. None if it will not open at all.

    Downscaled here rather than by the caller because every caller wants the same thing - a
    picture a model can read a label off - unlike :func:`cyclops.imagine.for_panel`, whose size is
    decided by the panel it is going to. Pillow is already a dependency and already does this
    exact resize in two other modules.
    """
    from io import BytesIO

    from PIL import Image, UnidentifiedImageError

    try:
        blob = path.read_bytes()
    except OSError:
        return None
    media = MEDIA_TYPES.get(path.suffix.lower(), "image/jpeg")
    try:
        with Image.open(BytesIO(blob)) as image:
            width, height = image.size
            if max(width, height) > MAX_EDGE:
                scale = MAX_EDGE / max(width, height)
                small = image.convert("RGB").resize(
                    (max(1, round(width * scale)), max(1, round(height * scale))), Image.LANCZOS
                )
                out = BytesIO()
                small.save(out, format="JPEG", quality=85)
                blob, media = out.getvalue(), "image/jpeg"
    except (UnidentifiedImageError, OSError, ValueError):
        # It would not open as an image. Send it as it is and let the API refuse it: a file with
        # a .jpg suffix that Pillow cannot parse is more likely truncated than it is malicious,
        # and the next reconcile will find it whole.
        pass
    return f"data:{media};base64," + base64.b64encode(blob).decode("ascii")


async def describe(path: Path, client: AsyncOpenAI, words: str = "") -> str:
    """One caption for one picture, or ``""``. ``words`` is what was being said around it.

    Takes the client rather than the settings, because the caller captions a run of pictures in
    one pass and a connection pool and TLS handshake per photo is the wrong shape on a Pi over
    wifi - the argument ``projects/agents.Models.build`` already makes for the filing sweep.

    Never raises. A picture that cannot be described is one that stays uncaptioned and is tried
    again on the next reconcile, which is the whole of the error handling this needs.
    """
    url = _data_url(path)
    if url is None:
        return ""
    try:
        response = await client.responses.create(
            model=CAPTION_MODEL,
            reasoning={"effort": "none"},  # a sentence about a picture needs no deliberation
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt_for(words)},
                        {"type": "input_image", "image_url": url, "detail": "auto"},
                    ],
                }
            ],
        )
    except (APIError, OSError, ValueError):
        return ""
    except Exception:  # noqa: BLE001 - one picture is never worth taking the service down
        return ""
    text = " ".join((getattr(response, "output_text", "") or "").split())
    if len(text) <= MAX_CAPTION_CHARS:
        return text
    return text[: MAX_CAPTION_CHARS - 1].rsplit(" ", 1)[0] + "…"


async def fill(
    folder: Path,
    settings: Settings,
    client: AsyncOpenAI | None,
    *,
    known: dict[str, str] | None = None,
    again: bool = False,
) -> int:
    """Caption everything in one folder that has none. Returns how many were written.

    Writes once at the end rather than per picture: the map is rewritten whole (see :func:`write`),
    so a write per photo would be N rewrites of a growing file for no added durability. A crash
    part-way through costs the captions made since the last folder and nothing that was already on
    the card.

    The one race worth naming: :func:`cyclops.session.describe` renames a finished session folder,
    so this can be holding a path that has just moved. Best-effort is the entire answer - the write
    fails, nothing is lost, and the next reconcile finds the folder under its new name with its
    pictures still uncaptioned.
    """
    captions = read(folder)
    stale = departed(folder, captions)

    pending = images(folder) if again else uncaptioned(folder)
    if not settings.api_key and known is None:
        pending = []
    if not around.ready(folder):
        # The conversation that produced these is still going, and the words that say what they
        # are have not all arrived - most of them arrive *after* the shutter, because a photo
        # opens a topic more often than it closes one. See cyclops.around. Pruning still runs:
        # it costs nothing and owes nothing to the session.
        pending = []
    if not pending and not stale:
        return 0
    words = around.shots(folder)

    made = 0
    for path in pending:
        # A picture this card has already described somewhere else, under another name. Reusing
        # the words costs nothing and is the only way the two copies can agree - see
        # ``indexer._known_captions`` for what happens when they do not.
        text = (known or {}).get(_mark(path), "")
        if not text and client is not None:
            text = await describe(path, client, words.get(path.name, ""))
            if text and known is not None:
                known[_mark(path)] = text
        if text:
            captions[path.name] = text
            made += 1
    for name in stale:
        captions.pop(name, None)
    if not made and not stale:
        return 0
    try:
        if captions:
            write(folder, captions)
        else:
            # Nothing left to describe. An empty map is litter in a folder somebody browses.
            (folder / card.CAPTIONS_NAME).unlink(missing_ok=True)
    except OSError:
        return 0  # the folder moved, or the card is full; the next reconcile tries again
    return made


def client_for(settings: Settings) -> AsyncOpenAI:
    """One client for a whole reconcile. Closed by the caller."""
    return AsyncOpenAI(api_key=settings.api_key, timeout=CAPTION_TIMEOUT_S, max_retries=0)
