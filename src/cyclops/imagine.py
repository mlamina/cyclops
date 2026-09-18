"""Pictures a model made: the photo redrawn with one thing different, and the diagram drawn from
nothing but a sentence.

Two jobs, one module, because they are the same job underneath - the same model, the same pixel
arithmetic, the same :func:`decode`, and the same jpg landing in the same ``photos/``.
:func:`edit` is the older half: the thing on the bench shown *changed* - the doors painted matt
black, the bracket moved to the other end of the rail, the half-built thing shown finished.
:func:`draw` is the newer one and has no photograph at all: a wiring diagram, a pinout, a block
diagram, drawn from a description because speech is a bad way to describe a wiring loom and there
is a screen sitting right there.

Both are answers where a sentence is a paragraph and a picture is a glance.

**The diagram used to be checked and is not any more.** It was drawn by a text model into a JSON
schema we owned and validated, and the panel rendered that with JointJS. The pictures were worse -
labels colliding, boxes clipped off the edge - so on 2026-09-04 the schema went and this took over.
What went with it was the only thing that could tell a wire drawn to the wrong pin from a wire
drawn to the right one. ``DRAW_QUALITY`` is the whole of the mitigation, and it is not a proof.
The tool description in :mod:`cyclops.agent` is where the model is told to say a connection out
loud when getting it wrong would cost somebody.

**What comes back is an illustration, never evidence.** ``images.edit`` with no mask redraws the
whole frame, so every pixel in the result is the model's, including the ones that look untouched.
Nothing in it is measured and nothing in it is a fact about anybody's hardware. That is the one
thing about this module that must not be forgotten, because the output *looks* like a photograph
of the user's own bench - and a picture of their loom "shown wired correctly" is the single worst
thing this feature could produce. The rule is enforced where the model can read it, in
``EDIT_PHOTO_TOOL``'s description over in :mod:`cyclops.agent`: colour, finish, placement and
"shown finished" are for :func:`edit`; connections, orientation and order of assembly are for
:func:`draw`. Both are drawn rather than measured, so the line between them is no longer "one is
checked" - it is that only :func:`edit` starts from a photograph of their bench, and a photograph
redrawn is the one that can be mistaken for a record of it.

**No mask, and that is the same fact from the other side.** There is no way to draw one from a
voice interface, so there is no way to tell the model "leave this half alone" other than by
asking it nicely in the prompt. The door is open if a future panel gains a finger to draw with.

**And no retry, in either half.** The drawing used to get one, because its failure was a schema
violation with a complaint you could hand back and the second attempt usually fixed it. There is
no schema now and so nothing to hand back: a retry is only another half-minute of somebody's
patience spent on the same dice. The one exception is the pixel-budget probe in :func:`edit`,
which is not a second roll - the first attempt never rendered anything.

Nothing here imports :mod:`cyclops.session` or :mod:`cyclops.webcam` - it takes a ``Path`` or two
strings and returns bytes - which is what keeps ``tests/test_imagine.py`` in the pure-logic suite
with no camera and no key, and what keeps OpenCV out of a module the admin service may one day
import.
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

# Sunburst, and not Flare and not the gpt-image-2 this used to be. Measured on 2026-09-15 from a
# laptop, through draw() and edit() themselves at production settings - the real DRAW_PROMPT and
# DEFAULT_STYLE, the same request each time:
#
#                             drawing, 1200x720 "high"   edit, 1152x640 "low"   output tokens
#   gpt-image-2                      76-88 s                 13.1 / 13.2 s          7024
#   gpt-image-2.5-flare               20.5 s                     10.8 s             1756
#   gpt-image-2.5-sunburst            26.0 s                     13.1 s             1756
#
# A drawing lands in a third of the time and a quarter of the price. The edit was already "low"
# and already quick: it does not move, and it did not need to. Everything downstream that says
# how long to expect to wait was rewritten for that first column - see DRAW_DIAGRAM_TOOL and the
# system prompt in cyclops.agent - because a picture announced as taking three times as long as
# it does sends the conversation off somewhere else and leaves it there.
#
# Sunburst over Flare on the picture and not on the clock: asked for the same diagram, Flare
# padded it with a "Colour Key (Wires)" legend nothing had asked for, and Sunburst drew what was
# asked. The two edits were indistinguishable - the tank painted, every label untouched.
#
# The saved half-minute is not to be spent back on quality. 2.5 adds "xhigh" and "max" above
# "high"; Sunburst took 34.4 s at xhigh and 62.3 s at max, and xhigh drew crisper lines and then
# invented a colour key *and* misspelled a label. See DRAW_QUALITY.
IMAGE_MODEL = "gpt-image-2.5-sunburst"

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
# the same reason panel.py's was: nothing is covering the panel while we wait, the caption says
# what is happening, and the alternative to waiting is no picture at all.
EDIT_TIMEOUT_S = 120.0

MAX_REQUEST_CHARS = 600
MAX_TITLE_CHARS = 70
MAX_STYLE_CHARS = 300  # a style note, not a second request; see DRAW_DIAGRAM_TOOL

# High, and it buys less than you would like. Measured on 2026-09-04 over five requests against
# gpt-image-2 - a relay wiring, the NS4168 amp on the Pi's header, the 40-pin pinout, a block
# diagram and a state machine: "low" took 19-22 s and put two of the I2S wires on the wrong rows
# of the header; "high" took 76-88 s and made one such mistake in the same drawing (LRCLK
# labelled "pin 35" and drawn from pin 36, on the Pi, first try). Fewer, not none. The same
# request through 2.5 sunburst is 26 s at "high" - see IMAGE_MODEL - so what follows about the
# wait costs a third of what it did, and every word of the argument still holds.
#
# So this is a reduction in a failure rate and not a fix, and the rest of the mitigation is
# elsewhere on purpose: DRAW_DIAGRAM_TOOL tells the model to say a connection out loud when
# getting it wrong would cost somebody a part. Nothing downstream can check a generated diagram -
# there is no schema left to check it against - and a wire drawn to the pin next to the right one
# is the one error a person cannot catch by looking, because the label beside it still reads
# correctly.
#
# Half a minute is bearable only because nothing is waiting on it: the tool has returned, the
# overlay says "drawing…", and the conversation carries on. That was written as a description and
# was not true - `_run_draw_diagram` awaited this call inside the tool handler, so the call stayed
# open for the whole of it and the model could not say another word. It is true now, and it is
# what pays for this setting: the drawing is a task (cyclops.tasks), the tool answers in a moment,
# and the model is told when the picture lands. Anything that puts a caller back in front of this
# await takes the argument for "high" with it.
#
# Upwards is not the answer either, now that there is an upwards: 2.5 adds "xhigh" and "max",
# and xhigh cost another eight seconds and drew a legend nobody asked for with a word misspelled
# in it. If the error rate turns out not to justify the wait, "low" is a one-word change and the
# honest one.
DRAW_QUALITY = "high"
DRAW_TIMEOUT_S = 180.0  # the documented worst case, with room; see EDIT_TIMEOUT_S

# Exactly 5:3, which is the panel's own shape, so nothing is letterboxed on the way to the
# glass. Both edges divide by 16 and the area clears the pixel-budget floor described below;
# checked against the API on 2026-09-04 and again on 2026-09-15 against 2.5, because a size it
# will not take is a 400 arriving half a minute late. Fixed rather than computed by size_for,
# because size_for exists to preserve a *source photograph's* framing and a drawing has no source.
PANEL_SIZE = "1200x720"

# The image model takes an arbitrary WIDTHxHEIGHT as long as both sides divide by 16 and the aspect
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
# ...and what a manual page goes to it as: the panel's full width, height following. See for_page.
PAGE_PANEL_WIDTH = 800

EDIT_PROMPT = """\
Edit the attached photograph as instructed, and change nothing else.

It was taken over a workbench by the person working on the thing in it, and they will look at
your result on a small screen beside that bench. Keep the camera position, the framing, the
lighting and every object that was not mentioned exactly as they are. Change only what is asked
for, and make the change sit in this photograph rather than look pasted onto it.

Add no text, labels, arrows, captions or watermarks unless the instruction asks for one. Do not
tidy, improve or beautify anything you were not asked about.

The instruction: {request}"""


# What the drawing is asked to look like when the voice model sends no style note. It should not
# happen - the tool makes the parameter required - but a diagram drawn in the wrong style still
# beats a tool that refuses over a missing adjective.
DEFAULT_STYLE = (
    "Draw it as a printed workshop service-manual diagram: black line-art on off-white paper, "
    "colour-coded wires with a small colour key, standard schematic symbols."
)

# The fixed half of the drawing instruction. The style note sits in the middle because that is
# where it reads as a description of the picture rather than an afterthought, and the request
# goes last so the most specific thing is the last thing read.
DRAW_PROMPT = """\
A technical diagram, drawn to be read on a small 800x480 workshop panel at arm's length.

{style}

Favour few, large, clearly separated elements over dense detail: a diagram of nine things
somebody can read beats one of thirty they cannot. Label every component and terminal in a crisp
technical sans-serif, set horizontally. Use the exact values, pin numbers and part names given
and no others - never "a resistor" where a value was said.

No perspective, no shading, no gradients, no glow, no photographic style, no decoration, no
watermark.

The diagram: {request}"""


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
    """One field, collapsed and trimmed - the same helper, for the same reason, as panel.py."""
    out = " ".join(str(value or "").split())
    return out[:limit].rstrip()


def _up(value: float) -> int:
    """One edge, rounded *up* to something the image model will accept.

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
    test that needs no key - the same reason :func:`cyclops.panel.validate` is not inlined into
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
    # Whatever is on the panel is what gets edited, and that is no longer always a camera JPEG:
    # a recalled picture can be a PNG somebody dropped into a project folder. The bytes go up as
    # image/jpeg below, so a mislabelled blob would reach the API as something it is not.
    # as_jpeg checks the format itself and hands a JPEG straight back, so this is not a branch.
    data = as_jpeg(data)

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
                    # input_fidelity is deliberately absent: the image model is always high and
                    # rejects being told so. Re-checked on 2026-09-15 against 2.5 sunburst, which
                    # answers "does not support the 'input_fidelity' parameter".
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


def as_jpeg(blob: bytes) -> bytes:
    """Any picture, as JPEG bytes. Unchanged if it will not open.

    :func:`for_panel` hands back its input untouched when it is already small enough, and
    :func:`cyclops.panel.offer_image` labels whatever it is given ``data:image/jpeg``. For the
    pictures this module makes that pairing is fine - they are always JPEG and always large. It is
    wrong for a small PNG somebody dropped into a project folder, which would reach the page
    labelled as something it is not and render as nothing at all, silently, on the one screen
    nobody can see from here. So anything not already a JPEG comes through here first.

    Never raises, for the reason :func:`for_panel` gives about the same case: a picture that will
    not convert is still a picture worth trying to show.
    """
    try:
        with Image.open(io.BytesIO(blob)) as image:
            if (image.format or "").upper() in {"JPEG", "JPG"}:
                return blob
            out = io.BytesIO()
            # RGBA and palette images both have to lose their alpha to become a JPEG. Flattening
            # onto white rather than black: these are photographs and screenshots, and a
            # transparent margin reads as paper far more often than it reads as night.
            picture = image.convert("RGBA")
            flat = Image.new("RGB", picture.size, (255, 255, 255))
            flat.paste(picture, mask=picture.split()[-1])
            flat.save(out, format="JPEG", quality=PANEL_JPEG_QUALITY)
            return out.getvalue()
    except (UnidentifiedImageError, OSError, ValueError):
        return blob


def for_panel(jpeg: bytes) -> bytes:
    """A copy small enough to travel to the page. The card keeps the full-size one.

    The panel is 800x480, so most of the pixels that arrive exist only to be thrown away by the
    browser - after being base64'd into ``PANEL_FILE``, fsynced to the SD card, and then read
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


def for_page(jpeg: bytes) -> bytes:
    """A manual page for the panel: exactly as wide as the panel, however tall that makes it.

    :func:`for_panel` caps the long edge, which for a portrait page is the height - an A4 page
    came out 724x1024, and the page stretches it across 800 px of glass. Capping the width
    instead means filling the width never upscales: a 150 dpi A4 page leaves as 800x1132 and
    scrolls. Never raises, and never enlarges a page that is already narrower than the panel.
    """
    try:
        with Image.open(io.BytesIO(jpeg)) as image:
            width, height = image.size
            if width <= PAGE_PANEL_WIDTH:
                return jpeg
            scale = PAGE_PANEL_WIDTH / width
            small = image.convert("RGB").resize(
                (PAGE_PANEL_WIDTH, max(1, round(height * scale))), Image.LANCZOS
            )
            out = io.BytesIO()
            small.save(out, format="JPEG", quality=PANEL_JPEG_QUALITY)
            return out.getvalue()
    except (UnidentifiedImageError, OSError, ValueError):
        return jpeg


async def draw(request: str, style: str, settings: Settings) -> bytes:
    """Draw one technical diagram from a description. Raises :class:`ImagineError`, never a
    traceback.

    The counterpart to :func:`edit`, and deliberately in the same module: everything the drawing
    needs - the model, the pixel-budget arithmetic, :func:`decode`, :func:`write` - is already
    here, and a diagram is the same kind of object an edit is. What differs is that there is no
    source photograph, so there is no framing to preserve and no :func:`size_for` call: the panel
    is 800x480 and the picture is drawn to fit it.

    ``style`` arrives from the voice model rather than from a constant here, because the right
    look is a property of the subject and only the conversation knows the subject. See
    ``DRAW_DIAGRAM_TOOL`` in :mod:`cyclops.agent` for what it is told to send.

    No retry, unlike :func:`cyclops.panel.draw` before it and for the reason this module's
    docstring already gives about :func:`edit`: an image has no schema to fail, so a second
    attempt is another half-minute of somebody's patience spent on the same dice.
    """
    request = _text(request, MAX_REQUEST_CHARS)
    if not request:
        raise ImagineError("no diagram was described")
    style = _text(style, MAX_STYLE_CHARS)

    client = AsyncOpenAI(api_key=settings.api_key, timeout=DRAW_TIMEOUT_S, max_retries=0)
    try:
        try:
            response = await client.images.generate(
                model=IMAGE_MODEL,
                prompt=DRAW_PROMPT.format(request=request, style=style or DEFAULT_STYLE),
                size=PANEL_SIZE,
                quality=DRAW_QUALITY,
                output_format=OUTPUT_FORMAT,
            )
        except APIError as exc:
            raise ImagineError(f"the drawing failed: {exc.message or exc}") from exc
        except TimeoutError as exc:
            raise ImagineError(f"the drawing took longer than {DRAW_TIMEOUT_S:.0f}s") from exc
        return decode(response)
    finally:
        await client.close()


def write(
    jpeg: bytes,
    request: str,
    folder: Path,
    *,
    role: str = "edit",
    when: datetime | None = None,
) -> Edit:
    """Put one imagined picture in a session's ``photos/`` and hand back what it became.

    Into ``photos/`` rather than an ``edits/`` of its own, and the argument is worth writing
    down. ``card.KNOWN`` is the closed set of what a session folder may hold, so a fourth name
    means touching ``card.triage``, ``views.MEDIA_DIRS``, ``library.stream``, the projects copier
    and two places in the README - five files of plumbing for a directory. In ``photos/`` it is
    in the picture stream, the lightbox, the Media view and the transcript for free, and the
    convention that already carries "who made this" is the filename's own suffix: ``_you`` for
    the shutter, ``_cyclops`` for the older cards, ``_edit`` for a photo redrawn with a change,
    and ``_drawn`` for a diagram :func:`draw` made from nothing but a sentence.
    ``session._render_photo`` was already branching on that.

    ``role`` is the only reason a diagram needs no plumbing of its own: it is a jpg in
    ``photos/`` like any other, so the captioner, the recall index, the Media view and the
    projects copier all pick it up with no change at all. That is the whole argument for having
    deleted the ``diagrams/`` folder.

    0600 because ``webcam._write_private`` is: this is a picture of somebody's room, whoever
    drew it. The admin service runs as the same user and can still serve it.
    """
    when = when or datetime.now()
    stem = f"{when.strftime('%H-%M-%S')}_{role}"
    path = folder / f"{stem}.jpg"
    card.write_bytes(path, jpeg, mode=0o600)
    return Edit(
        ident=stem,
        request=_text(request, MAX_TITLE_CHARS),
        created=when.strftime("%Y-%m-%d %H:%M:%S"),
        bytes=len(jpeg),
        path=path,
    )
