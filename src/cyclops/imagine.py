"""Imagining a change: the photo they just took, redrawn with one thing different.

Cyclops can already look at what is held up and say something about it, and it can draw a
diagram when the answer is a set of connections. What it could not do is show the thing on the
bench *changed* - the doors painted matt black, the bracket moved to the other end of the rail,
the half-built thing shown finished. Those are the answers where a sentence is a paragraph and a
picture is a glance, and there is a screen sitting right there.

**What comes back is an illustration, never evidence.** ``images.edit`` with no mask redraws the
whole frame, so every pixel in the result is the model's, including the ones that look untouched.
Nothing in it is measured and nothing in it is a fact about anybody's hardware. That is the one
thing about this module that must not be forgotten, because the output *looks* like a photograph
of the user's own bench - and a picture of their loom "shown wired correctly" is the single worst
thing this feature could produce. The rule is enforced where the model can read it, in
``EDIT_PHOTO_TOOL``'s description over in :mod:`cyclops.agent`: colour, finish, placement and
"shown finished" are for here; connections, orientation and order of assembly are for
``draw_diagram``, whose output is checked against a schema we own.

**No mask, and that is the same fact from the other side.** There is no way to draw one from a
voice interface, so there is no way to tell the model "leave this half alone" other than by
asking it nicely in the prompt. The door is open if a future panel gains a finger to draw with.

**And no retry**, unlike :func:`cyclops.diagram.draw`. A diagram gets one because its failure is
a schema violation with a complaint you can hand back, and the second attempt usually fixes it.
An image has no schema to fail: a retry is another half-minute of somebody's patience spent on
the same dice.

Nothing here imports :mod:`cyclops.session` or :mod:`cyclops.webcam` - it takes a ``Path`` and
returns bytes - which is what keeps ``tests/test_imagine.py`` in the pure-logic suite with no
camera and no key, and what keeps OpenCV out of a module the admin service may one day import.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from openai import APIError, AsyncOpenAI
from PIL import Image, UnidentifiedImageError

from . import card
from .config import Settings

IMAGE_MODEL = "gpt-image-2"  # newest GA image model; the id pins snapshot gpt-image-2-2026-04-21

# Low, and not a compromise. Measured from a laptop on 2026-09-02 against a real 1024x576
# capture: 13.1 s and 13.2 s for "paint every wooden surface matt black" and "put a full
# bookshelf along the empty wall on the left", ~100 KB out, both correct and both keeping the
# rest of the room exactly where it was. The picture is then shown on an 800x480 panel, so it is
# halved again on the way there (see for_panel), which hides most of what "low" gives up - while
# what it buys is the thing a conversation notices. OpenAI's own guidance is that low is the
# fast-iteration setting and that a complex prompt at higher quality can take two minutes: an
# answer in thirteen seconds beats a better answer in sixty when somebody is standing there.
QUALITY = "low"

# JPEG rather than the default PNG, and this is load-bearing twice over. The result lands in the
# session's photos/ beside every other jpg, so the card, the admin service's MEDIA_TYPES
# allow-list and library.stream all need no changes at all; and the picture travels to the panel
# as base64 inside a file that is re-read several times a second, where a 1.5 MP PNG would be
# megabytes.
OUTPUT_FORMAT = "jpeg"

# The documented worst case for a complex prompt. Academic at QUALITY = "low", and generous for
# the same reason diagram.py's is: nothing is covering the panel while we wait, the caption says
# what is happening, and the alternative to waiting is no picture at all.
EDIT_TIMEOUT_S = 120.0

MAX_REQUEST_CHARS = 600
MAX_TITLE_CHARS = 70

# gpt-image-2 takes an arbitrary WIDTHxHEIGHT as long as both sides divide by 16 and the aspect
# is between 1:3 and 3:1, so the edit can keep the photograph's own framing instead of being
# cropped into one of three standard shapes. Keeping the framing is the whole point: a crop
# changes what you are looking at, which is the one thing an edit of your own photo must not do.
#
# There is a fourth rule the documentation does not mention, and it is the one that bites: a
# *minimum pixel budget*. Measured on 2026-09-02, 1024x576 - the exact shape a C920 capture comes
# out of webcam.py as - is refused with "Requested resolution is below the current minimum pixel
# budget", while 1152x640 renders fine. So the size is chosen by area and not by edge: keep the
# aspect, spend PIXEL_BUDGET pixels on it, and round every edge *up* to a multiple of 16 so the
# rounding can only ever take us further above the floor. The word "current" in that error is why
# BUDGET_HINT and the one retry in edit() exist - OpenAI can move this, and a feature that dies
# silently when they do is worse than one that spends an extra half-second finding out.
PIXEL_BUDGET = 786_432  # 1024x768. The floor sits between 0.59 and 0.74 MP; this clears it.
BUDGET_GROWTH = 1.6  # what one retry multiplies the budget by, if the floor has moved under us
MAX_EDGE = 2048  # a guard, not a target: at this budget no legal aspect ratio comes near it
MIN_EDGE = 256
STEP = 16
MAX_ASPECT = 3.0

# What goes to the panel. See for_panel.
PANEL_MAX_EDGE = 1024
PANEL_JPEG_QUALITY = 82

EDIT_PROMPT = """\
Edit the attached photograph as instructed, and change nothing else.

It was taken over a workbench by the person working on the thing in it, and they will look at
your result on a small screen beside that bench. Keep the camera position, the framing, the
lighting and every object that was not mentioned exactly as they are. Change only what is asked
for, and make the change sit in this photograph rather than look pasted onto it.

Add no text, labels, arrows, captions or watermarks unless the instruction asks for one. Do not
tidy, improve or beautify anything you were not asked about.

The instruction: {request}"""


class ImagineError(RuntimeError):
    """The edit could not be made. The message is meant for the voice model to relay."""


@dataclass(frozen=True)
class Edit:
    """One imagined picture on the card: what was asked for, and where it landed."""

    ident: str  # the filename stem
    request: str
    created: str
    bytes: int
    path: Path | None = None  # None for one nothing could keep


def _text(value: object, limit: int) -> str:
    """One field, collapsed and trimmed - the same helper, for the same reason, as diagram.py."""
    out = " ".join(str(value or "").split())
    return out[:limit].rstrip()


def _up(value: float) -> int:
    """One edge, rounded *up* to something gpt-image-2 will accept.

    Up rather than nearest, so that rounding can only ever move us further above the minimum
    pixel budget. Rounding down is how you send 1024x576 and get a 400 back.
    """
    return max(MIN_EDGE, math.ceil(value / STEP) * STEP)


def size_for(width: int, height: int, *, budget: int = PIXEL_BUDGET) -> str:
    """The output size that keeps this photo's framing, as ``"WIDTHxHEIGHT"``.

    Clamp the aspect into what the model allows, scale *both* edges together until the picture
    is worth ``budget`` pixels, hold it under MAX_EDGE, and round each edge up to a multiple of
    16. Both edges together every time, because scaling one is a crop.
    """
    width, height = max(1, int(width)), max(1, int(height))
    if width > height * MAX_ASPECT:
        width = round(height * MAX_ASPECT)
    elif height > width * MAX_ASPECT:
        height = round(width * MAX_ASPECT)
    scale = math.sqrt(budget / (width * height))
    scale = min(scale, MAX_EDGE / max(width, height))
    return f"{_up(width * scale)}x{_up(height * scale)}"


def _dimensions(jpeg: bytes) -> tuple[int, int]:
    """How big the source is. Only the header is read; the pixels are never decoded."""
    try:
        with Image.open(io.BytesIO(jpeg)) as image:
            return image.size
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImagineError("that photo could not be read") from exc


def decode(response: object) -> bytes:
    """The picture out of one ``images.edit`` response, or :class:`ImagineError` saying there
    wasn't one.

    Its own function so that the one piece of API-shape knowledge in this module is covered by a
    test that needs no key - the same reason :func:`cyclops.diagram.validate` is not inlined into
    ``draw()``.
    """
    data = getattr(response, "data", None) or []
    b64 = getattr(data[0], "b64_json", "") if data else ""
    if not b64:
        raise ImagineError("the edit came back empty")
    try:
        return base64.b64decode(b64, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ImagineError("the edit came back unreadable") from exc


async def edit(source: Path, request: str, settings: Settings) -> bytes:
    """Redraw one photo with one change. Raises :class:`ImagineError`, never a traceback.

    ``max_retries=0`` matters more here than anywhere else in the codebase: the SDK's default of
    two would turn one slow minute into three, silently, with somebody waiting.
    """
    request = _text(request, MAX_REQUEST_CHARS)
    if not request:
        raise ImagineError("no change was described")
    try:
        data = await asyncio.to_thread(source.read_bytes)
    except OSError as exc:
        raise ImagineError("that photo is not on the card any more") from exc
    if not data:
        raise ImagineError("that photo is empty")

    width, height = _dimensions(data)
    client = AsyncOpenAI(api_key=settings.api_key, timeout=EDIT_TIMEOUT_S, max_retries=0)
    try:
        budget = PIXEL_BUDGET
        for attempt in range(2):
            size = size_for(width, height, budget=budget)
            try:
                response = await client.images.edit(
                    model=IMAGE_MODEL,
                    image=("photo.jpg", data, "image/jpeg"),
                    prompt=EDIT_PROMPT.format(request=request),
                    size=size,
                    quality=QUALITY,
                    output_format=OUTPUT_FORMAT,
                    # input_fidelity is deliberately absent: gpt-image-2 is always high and
                    # rejects being told so.
                )
            except APIError as exc:
                message = exc.message or str(exc)
                # The one retry this module allows, and it is not a second roll of the dice: the
                # first attempt never rendered anything. The floor is documented nowhere and the
                # error calls it "current", so the day it moves this asks once for more pixels
                # instead of taking the feature down. Anything else - a moderation refusal above
                # all, which is the commonest real failure - is handed straight on, because the
                # message goes to the voice model to say out loud.
                if attempt == 0 and "pixel budget" in message.lower():
                    print(f"· imagine: {size} is under the floor now; asking for more", flush=True)
                    budget = int(budget * BUDGET_GROWTH)
                    continue
                raise ImagineError(f"the edit failed: {message}") from exc
            except TimeoutError as exc:
                raise ImagineError(f"the edit took longer than {EDIT_TIMEOUT_S:.0f}s") from exc
            return decode(response)
        raise ImagineError("the edit failed: no picture size the model would accept")
    finally:
        await client.close()


def for_panel(jpeg: bytes) -> bytes:
    """A copy small enough to travel to the page. The card keeps the full-size one.

    The panel is 800x480, so most of the pixels that arrive exist only to be thrown away by the
    browser - after being base64'd into ``DIAGRAM_FILE``, fsynced to the SD card, and then read
    and JSON-parsed by ``/api/panel``, which is polled every 400 ms and whose whole promise is
    that it stays one read and nothing else. This is the same trade
    ``webcam._resize_to_max_edge`` makes on the way to the model, for the same reason.

    Never raises: a picture that will not shrink is still a picture worth showing, so the
    original comes back unchanged.
    """
    try:
        with Image.open(io.BytesIO(jpeg)) as image:
            width, height = image.size
            if max(width, height) <= PANEL_MAX_EDGE:
                return jpeg
            scale = PANEL_MAX_EDGE / max(width, height)
            small = image.convert("RGB").resize(
                (max(1, round(width * scale)), max(1, round(height * scale))), Image.LANCZOS
            )
            out = io.BytesIO()
            small.save(out, format="JPEG", quality=PANEL_JPEG_QUALITY)
            return out.getvalue()
    except (UnidentifiedImageError, OSError, ValueError):
        return jpeg


def write(jpeg: bytes, request: str, folder: Path, *, when: datetime | None = None) -> Edit:
    """Put one imagined picture in a session's ``photos/`` and hand back what it became.

    Into ``photos/`` rather than an ``edits/`` of its own, and the argument is worth writing
    down. ``card.KNOWN`` is the closed set of what a session folder may hold, so a fourth name
    means touching ``card.triage``, ``views.MEDIA_DIRS``, ``library.stream``, the projects copier
    and two places in the README - five files of plumbing for a directory. In ``photos/`` it is
    in the picture stream, the lightbox, the Media view and the transcript for free, and the
    convention that already carries "who made this" is the filename's own suffix: ``_you`` for
    the shutter, ``_cyclops`` for the older cards, and now ``_edit``. ``session._render_photo``
    was already branching on that.

    0600 because ``webcam._write_private`` is: this is a picture of somebody's room, whoever
    drew it. The admin service runs as the same user and can still serve it.
    """
    when = when or datetime.now()
    stem = f"{when.strftime('%H-%M-%S')}_edit"
    path = folder / f"{stem}.jpg"
    card.write_bytes(path, jpeg, mode=0o600)
    return Edit(
        ident=stem,
        request=_text(request, MAX_TITLE_CHARS),
        created=when.strftime("%Y-%m-%d %H:%M:%S"),
        bytes=len(jpeg),
        path=path,
    )
