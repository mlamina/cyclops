"""What is printed on a page of a PDF, in words - so that something can search for it later.

The manual half of :mod:`cyclops.captions`, and deliberately its twin: one vision call per page,
never raising, writing a sidecar beside the thing it describes. This module owns ``pymupdf`` and
the page-reading model, and nothing else in the tree imports either.

**Every page goes through vision, even when the PDF has a text layer.** That looks wasteful and is
not. Measured on the m.unit manual, 14 of its 45 pages carry broken ligatures - the extracted text
holds ``conﬁ guration`` and ``certiﬁ ed``, so searching it for "configuration" or "certified"
returns nothing at all, while the vision transcription finds every one. The illustrations are
worse: they are drawn in vector, so ``page.get_images()`` returns **0** on the wiring-diagram
pages, and no amount of reading the file will tell you a diagram is there. A model looking at the
page knows both. So the page is rendered and looked at, and the text layer is never consulted.

That decision also keeps this module out of the trap waiting in the file: a manual's diagram pages
are usually ``rotation=90`` - landscape artwork in a portrait document - and ``get_drawings()``
reports display coordinates while ``get_text()`` reports unrotated ones. Mixing them raises
nothing; it silently yields a box with the labels sheared off. ``get_pixmap`` applies the rotation
itself, so **v1 cannot hit this**. Whoever adds figure-cropping later will, and must multiply by
``page.rotation_matrix``.

What this module must never do:

* **Never raise for a page it cannot read.** A failed page stays unread and is retried on the next
  sweep; a whole manual is never lost to one bad page.
* **Never decide where anything lives.** :mod:`cyclops.manuals` owns the folder and every write
  into it. This renders, asks, and hands back.
"""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass, field
from pathlib import Path

from openai import APIError, AsyncOpenAI

from . import manuals
from .config import Settings
from .manuals import Manual

# The same model the captioner uses, for the same reason: this is index text, nobody is waiting on
# it, and a 45-page manual through it costs about a penny. Measured 2026-09-11 on an image-only
# copy of the m.unit manual - median 0.96 bag-of-words recall against the original text layer.
PAGE_MODEL = "gpt-5.4-nano"
# 150 dpi is one render doing two jobs. The model reads it, comfortably above the 120 dpi the 0.96
# was measured at; the panel gets it downscaled. A4 at 150 is 1240x1754 and about 40 KB.
PAGE_DPI = 150
PAGE_TIMEOUT_S = 45.0
# Four at a time. Serial is ~2-3 s a page, so a 45-page manual would take two minutes of a sweep
# that has photos to caption too; measured fan-out did 45 pages in 23.5 s. Above four the gain is
# small and the Pi is sharing this with a kiosk.
PAGE_WORKERS = 4
# A backstop, not a budget. The longest thing likely to land here is a printer manual at ~670
# pages; past this the cost stops being invisible and somebody should choose deliberately.
MAX_PAGES = 400
# Written every this many pages, so a power cut costs a few calls rather than the whole manual.
FLUSH_EVERY = 10
# How many pages' text the identity call gets. The cover, the legal page and a contents page is
# enough to name a part, and three is cheap because the transcriptions are already paid for.
IDENTITY_PAGES = 3
MAX_FIGURES = 6
# The description in the session prompt. Short because it is paid for in every session
# whether a manual comes up or not - see agent.MANUALS_HEADER.
ABOUT_CHARS = 140
MAX_PAGE_CHARS = 6000  # one dense page of specifications, generously

PAGE_PROMPT = """\
This is one page of a product manual. Transcribe and describe it so somebody can find this page
later by searching for what is on it. Reply with JSON only, no markdown fence:

{"heading": "the page's topic in at most 8 words",
 "text": "all readable text, verbatim, in reading order, tables as lines",
 "figures": [{"what": "what this illustration shows, one sentence",
              "kind": "wiring diagram|photo|exploded view|table|pinout|schematic"}]}

Transcribe the text, do not summarise it - every number, unit, torque, size and part code exactly
as printed. If the page has no illustration, figures is []. If the page is blank or holds nothing
readable, return empty strings and an empty list rather than guessing."""

IDENTITY_PROMPT = """\
These are the opening pages of a product manual. Say what it documents. JSON only, no fence:

{"manual": "what the document calls itself",
 "part": "the product it documents, named as somebody with one on the bench would name it",
 "maker": "the manufacturer",
 "revision": "the version or revision, or an empty string",
 "about": "the topics this manual covers, as a bare comma-separated list",
 "aliases": ["other names this gets called out loud, including sloppy spoken ones"]}

Two of these do the real work.

"about" is read at the top of every conversation by an assistant deciding one thing only: is the
question in front of me one this manual could answer? So it is **the subjects covered, as a bare
comma-separated list, at most 14 words**. The part name is printed beside it already, so never
repeat the product, its category or its maker - only what you could look up in it. No sentence,
no verb, no "this manual covers".
Good: "loom wiring, load circuits, indicators, fuses, keyless entry, app setup, fault codes"
Bad: "The mo.unit blue body control module manual, covering safety and installation."
It is read whether or not a manual ever comes up, so every word is paid for in every session.

"aliases" are how somebody finds this by voice, so include what a person would actually say.
Always include the part name with its punctuation removed and with it replaced by spaces, plus any
shortened form. For "mo.unit blue" that is "mo unit", "mounit", "mo.unit", "mo unit blue". At most
6, all lowercase, never the maker's name on its own, and never a typo you invented.

Pages:
"""


@dataclass
class Page:
    """One page, read. An instance with empty text is a page that genuinely holds nothing."""

    heading: str = ""
    text: str = ""
    figures: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"heading": self.heading, "text": self.text, "figures": self.figures}


# ------------------------------------------------------------------ the PDF itself


def page_count(pdf: Path) -> int:
    """How many pages, or 0 if this is not a PDF we can open. Never raises."""
    try:
        import pymupdf
    except ImportError:
        return 0
    try:
        with pymupdf.open(pdf) as doc:
            return doc.page_count
    except Exception:
        return 0  # encrypted, truncated, or not a PDF at all


def render(pdf: Path, numbers: list[int], into: Path) -> dict[int, Path]:
    """Render these 1-based page numbers to JPEGs under ``into``. Returns what landed.

    Pure CPU and no network - 57 ms a page on a Pi, measured - so this runs in a thread while the
    model calls for the pages before it are still in flight.
    """
    try:
        import pymupdf
    except ImportError:
        return {}
    done: dict[int, Path] = {}
    try:
        doc = pymupdf.open(pdf)
    except Exception:
        return {}
    with doc:
        for n in numbers:
            target = into / f"{n:04d}.jpg"
            try:
                pixels = doc[n - 1].get_pixmap(dpi=PAGE_DPI)
                card_write(target, pixels.tobytes("jpeg", jpg_quality=80))
                done[n] = target
            except Exception:
                continue  # one unrenderable page must not cost the rest of the manual
    return done


def card_write(path: Path, blob: bytes) -> None:
    """Land one render. Split out so :func:`render` stays about pymupdf and nothing else."""
    from . import card

    card.write_bytes(path, blob)


# ------------------------------------------------------------------ asking the model


def _data_url(image: Path) -> str | None:
    """One render as a data URL, or None if the bytes will not read. Sync, like its caller's."""
    try:
        blob = image.read_bytes()
    except OSError:
        return None
    return f"data:image/jpeg;base64,{base64.b64encode(blob).decode()}"


def _loads(answer: str) -> dict | None:
    """The model's JSON, or None. Tolerates the fence it was told not to use."""
    text = answer.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        found = json.loads(text)
    except ValueError:
        return None
    return found if isinstance(found, dict) else None


async def read_page(image: Path, client: AsyncOpenAI) -> Page | None:
    """Read one rendered page. ``None`` means ask again; a ``Page`` means do not.

    That distinction is the whole contract and it is not the one :func:`cyclops.captions.describe`
    makes. A photo is one call, so conflating "the call failed" with "there is nothing here" costs
    a retry. A manual is hundreds, and a page whose answer never parses would otherwise be re-read
    every fifteen minutes for as long as the card exists.

    So a transport failure returns ``None``. A reply that arrives and will not parse is **not** a
    failure: the words in it are still the words on the page, so they are kept as the transcription
    with no figures. Measured on the m.unit manual, exactly one page in 45 did this - a dense legal
    page whose JSON came back truncated - and its text was perfectly good.
    """
    url = _data_url(image)
    if url is None:
        return None
    try:
        answer = await client.responses.create(
            model=PAGE_MODEL,
            reasoning={"effort": "none"},
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": PAGE_PROMPT},
                        {"type": "input_image", "image_url": url, "detail": "auto"},
                    ],
                }
            ],
        )
    except (APIError, OSError, ValueError):
        return None
    except Exception:
        return None

    said = (getattr(answer, "output_text", "") or "").strip()
    if not said:
        return None  # an empty reply is a failed call, not a blank page
    found = _loads(said)
    if found is None:
        return Page(text=said[:MAX_PAGE_CHARS])  # unparseable, but the words are still the words

    figures = [
        {"what": str(f.get("what", ""))[:200], "kind": str(f.get("kind", ""))[:40]}
        for f in found.get("figures", [])[:MAX_FIGURES]
        if isinstance(f, dict) and f.get("what")
    ]
    return Page(
        heading=str(found.get("heading", ""))[:120],
        text=str(found.get("text", ""))[:MAX_PAGE_CHARS],
        figures=figures,
    )


async def identify(pages: dict[str, dict], client: AsyncOpenAI) -> dict:
    """What this manual is, from the pages already read. ``{}`` if it could not be worked out.

    Text only, and no second render: the opening pages have already been transcribed and paid for,
    so this is one cheap call over words we hold rather than another trip through the images.
    """
    opening = [
        pages[str(n)].get("text", "")
        for n in range(1, IDENTITY_PAGES + 1)
        if str(n) in pages and pages[str(n)].get("text", "").strip()
    ]
    if not opening:
        return {}
    # Every page's heading, which is the contents page this document may not have. The opening
    # pages alone name the product well and describe it badly: on the m.unit manual they are a
    # cover, a safety warning and a legal page, so the first `about` written from them said the
    # manual covered "warranty and liability exclusions" - true of pages 2 and 3 and useless as a
    # description of a wiring manual. The headings say what is actually in it, and they are
    # already paid for.
    headings = [
        h
        for n in sorted(pages, key=lambda k: int(k) if k.isdigit() else 0)
        if (h := str(pages[n].get("heading", "")).strip())
    ]
    # Concatenated, never str.format: both prompts in this module carry a literal JSON example,
    # and a brace in a format string is a field to be substituted. It fails as a KeyError that the
    # catch-all below turns into a silent "could not work it out" - which is exactly how it got
    # past a first run here.
    try:
        answer = await client.responses.create(
            model=PAGE_MODEL,
            reasoning={"effort": "none"},
            input=(
                IDENTITY_PROMPT
                + "\n\n---\n\n".join(opening)[:6000]
                + "\n\nEvery page's heading, in order - this is what the manual covers:\n"
                + "\n".join(f"- {h}" for h in headings)[:3000]
            ),
        )
    except Exception:
        return {}
    found = _loads((getattr(answer, "output_text", "") or "").strip())
    return found or {}


# ------------------------------------------------------------------ one manual, start to finish


def unread(manual: Manual, pages: dict[str, dict], total: int) -> list[int]:
    """Which page numbers still need reading. The queue is the sidecar, as it is for captions."""
    return [n for n in range(1, min(total, MAX_PAGES) + 1) if str(n) not in pages]


async def fill(manual: Manual, settings: Settings, client: AsyncOpenAI) -> int:
    """Read what is still unread of one manual. Returns how many pages were added.

    Resumable by construction: the sidecar says what is done, so a power cut mid-manual costs at
    most ``FLUSH_EVERY`` pages of work and never the manual.
    """
    total = manual.pages or page_count(manual.pdf)
    if not total:
        return 0
    pages = manuals.read_pages(manual.path)
    todo = unread(manual, pages, total)
    if not todo:
        if manual.pages != total or manual.read != len(pages):
            manual.pages, manual.read = total, len(pages)
            manuals.write_identity(manual)
        # A manual finished reading before `about` existed has every page and no description, and
        # nothing else will ever come and give it one: there are no pages left to add, so every
        # later sweep would take this exit. The identity call is the one piece of work a complete
        # manual can still be owed.
        await _name(manual, pages, client)
        return 0

    manual.pages = total
    limit = asyncio.Semaphore(PAGE_WORKERS)

    async def one(n: int, image: Path) -> tuple[int, Page | None]:
        async with limit:
            return n, await read_page(image, client)

    made = 0
    for start in range(0, len(todo), FLUSH_EVERY):
        batch = todo[start : start + FLUSH_EVERY]
        rendered = await asyncio.to_thread(render, manual.pdf, batch, manual.pages_dir)
        if not rendered:
            break  # the PDF stopped being readable; leave what is done and try again next sweep
        results = await asyncio.gather(*(one(n, path) for n, path in sorted(rendered.items())))
        for n, page in results:
            if page is None:
                continue  # a failed call, not a blank page - it stays in the queue
            pages[str(n)] = page.as_dict()
            made += 1
        manual.read = len(pages)
        manuals.write_pages(manual.path, pages)
        manuals.write_identity(manual)

    await _name(manual, pages, client)
    return made


async def _name(manual: Manual, pages: dict[str, dict], client: AsyncOpenAI) -> None:
    """Work out what this manual is, if that is not known yet. One call, once, per manual.

    Gated on the description rather than on new pages: what decides whether to open a manual is
    knowing what it covers, and a manual read before that field existed has a name and nothing
    else. Gating on work done would mean the sweep skipped it for ever, because a complete manual
    adds no pages.
    """
    if not pages or manual.about:
        return
    found = await identify(pages, client)
    if not found:
        return
    manual.name = str(found.get("manual") or manual.name)[:160]
    manual.part = str(found.get("part", ""))[:160]
    manual.maker = str(found.get("maker", ""))[:80]
    manual.revision = str(found.get("revision", ""))[:40]
    manual.about = str(found.get("about", ""))[:ABOUT_CHARS]
    aliases = found.get("aliases", [])
    if isinstance(aliases, list):
        manual.aliases = [str(a)[:60] for a in aliases if a][: manuals.MAX_ALIASES]
    manuals.write_identity(manual)


def client_for(settings: Settings) -> AsyncOpenAI:
    """One client for a whole pass. Closed by the caller - the shape ``captions`` uses."""
    return AsyncOpenAI(api_key=settings.api_key, timeout=PAGE_TIMEOUT_S, max_retries=1)
