"""The librarian: a subagent that answers one question from the manuals on the card.

Cyclops hands it a question in plain words and gets back a few speakable lines, the pages they
came from, or the exact reason there is no answer. Everything in between - which manual, which
page, reading three of them, looking at a wiring diagram - happens in here, in a context of its
own, so none of it lands in the realtime conversation that pays for every token on every turn.

It reads only what is already on the card: ``Manual.md`` for what each manual is, ``pages.json``
for what each page says, ``Pages/NNNN.jpg`` for what it looks like, and the manual rows of the
recall index for searching by meaning. Nothing is re-indexed and nothing is written.

What this module must never do:

* **Never write.** Every tool returns text or a picture. The answer goes back to the caller and
  the caller decides what to say, show and note.
* **Never be imported eagerly.** It pulls in ``pydantic_ai``, a second on the Pi; the voice agent
  imports it in a thread when a session starts, the way ``projects.agents`` is kept off the boot.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent, RunContext, UsageLimits
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

from . import manuals, recall
from .config import Settings
from .manuals import Manual

MODEL = "gpt-5.4-mini"
# Low effort, because somebody is standing there waiting: the work is finding the page, and the
# page does the thinking. Measured against the eval in tools/manual_eval.py.
SETTINGS = OpenAIResponsesModelSettings(openai_reasoning_effort="low")
REQUEST_TIMEOUT_S = 20.0
# A loop backstop. A good lookup is two or three requests: read what the first search found,
# maybe one more search or a look at a diagram, answer.
LIMITS = UsageLimits(request_limit=8, tool_calls_limit=16, cost_limit=Decimal("0.05"))

# Full resolution. "high" scales a page to 768 px on its short side, and on Revival's 3300 px
# diagram that read right turn as green and start as yellow; "original" read all five wires right,
# for about 10k input tokens.
IMAGE_DETAIL = "original"
SEARCH_HITS = 8
PRE_READ = 3  # the first search's best pages, read into the opening prompt
FIND_HITS = 10
SNIPPET_CHARS = 220
MAX_PAGE_CHARS = 6000
MAX_QUESTION_CHARS = 600

INSTRUCTIONS = """\
You look things up in product manuals for Cyclops, a voice assistant working with someone whose
hands are full. Cyclops sends you one question; you find the answer in the manuals and send back
a few lines it can say out loud.

How to look:
- The first search has already been run for you and its top pages read; both are under the
  question. If those pages do not hold the whole answer, read more - a search hit is a pointer,
  never the answer.
- Read neighbouring pages when a procedure or a menu list runs over a page break.
- Use find for exact words: a menu number ("Menu 8"), a terminal ("AUX1"), a button name, a
  setting. Use search again, worded differently, when the first hits miss.
- Use look only when the text is not enough: a wiring diagram, wire colours, a scrambled table,
  a figure the text only names. The text of a diagram page lists labels, not connections.
- Several manuals can cover one part: a maker's manual and a kit's wiring diagram. If the
  question names a kit, a bike or a diagram, use the manual that matches that.

How to answer:
- answer: two to four short spoken sentences. Lead with the answer itself. Give the exact value,
  option letter, menu number, wire colour or press sequence as printed - never "see the manual".
  A press sequence starts from where they are: every button in order, including the one that
  opens the menu or mode the control lives in.
  No page numbers and no manual titles in it; Cyclops cites them only when asked.
- pages: every page the answer came from.
- Never answer from general knowledge. If the manuals do not say, answered is false and why_not
  says exactly why: what you searched for, which pages you read, and whether the manuals do not
  cover it or the question was ambiguous (and between what).
"""


class Source(BaseModel):
    manual: str = Field(description="The manual's key, as listed by manuals().")
    page: int


class ManualAnswer(BaseModel):
    answered: bool
    answer: str = Field(default="", description="Short and speakable, with the exact values.")
    pages: list[Source] = Field(default_factory=list)
    why_not: str = Field(
        default="",
        description="Only when not answered: what was searched and read, and why that failed.",
    )


@dataclass
class Shelf:
    """What one lookup can see, and the trail it leaves. One per question."""

    held: list[Manual]
    index: recall.Index  # the manual rows of the recall index, nothing else
    client: AsyncOpenAI | None  # for embedding a search; None means search by words only
    steps: list[str] = field(default_factory=list)
    _pages: dict[str, dict[str, dict]] = field(default_factory=dict)

    def manual(self, name: str) -> Manual | None:
        wanted = name.strip().lower()
        for one in self.held:
            if one.key == wanted:
                return one
        return manuals.find(self.held, name)

    def pages(self, manual: Manual) -> dict[str, dict]:
        if manual.key not in self._pages:
            self._pages[manual.key] = manuals.read_pages(manual.path)
        return self._pages[manual.key]

    def by_scope(self, scope: str) -> Manual | None:
        return next((one for one in self.held if one.scope == scope), None)


def _words(page: dict) -> str:
    figures = " ".join(str(f.get("what", "")) for f in page.get("figures", []) if isinstance(f, dict))
    return " ".join(f"{page.get('heading', '')} {page.get('text', '')} {figures}".split())


def snippet(text: str, query: str, size: int = SNIPPET_CHARS) -> str:
    """The stretch of ``text`` with the most of the query's words in it."""
    terms = {t for t in re.findall(r"[a-z0-9.]+", query.lower()) if len(t) > 2}
    if len(text) <= size or not terms:
        return text[:size]
    low = text.lower()
    step = size // 4
    best, at = -1, 0
    for start in range(0, len(text) - size + step, step):
        hits = sum(low.count(t, start, start + size) for t in terms)
        if hits > best:
            best, at = hits, start
    return ("…" if at else "") + text[at : at + size].strip() + "…"


def _missing(shelf: Shelf, name: str) -> str:
    known = ", ".join(one.key for one in shelf.held) or "none"
    return f"No manual called {name!r}. The manuals are: {known}"


librarian = Agent(
    name="librarian",
    deps_type=Shelf,
    output_type=ManualAnswer,
    instructions=INSTRUCTIONS,
    retries=2,
)


@librarian.instructions
def _catalogue(ctx: RunContext[Shelf]) -> str:
    return "The manuals on the card:\n" + _list(ctx.deps)


def _list(shelf: Shelf) -> str:
    lines = []
    for one in shelf.held:
        line = f"- {one.key}: {one.part or one.name}"
        if one.maker:
            line += f" by {one.maker}"
        line += f", {one.pages} pages"
        if one.about:
            line += f" - {one.about}"
        if one.aliases:
            line += f" (also called: {', '.join(one.aliases)})"
        lines.append(line)
    return "\n".join(lines) or "(none)"


@librarian.tool
async def contents(ctx: RunContext[Shelf], manual: str) -> str:
    """Every page of one manual with its heading - the table of contents.

    Args:
        manual: the manual's key.
    """
    found = ctx.deps.manual(manual)
    ctx.deps.steps.append(f"contents {manual}")
    if found is None:
        return _missing(ctx.deps, manual)
    pages = ctx.deps.pages(found)
    return "\n".join(f"p{n}: {page.get('heading', '')}" for n, page in pages.items())


async def _ranked(shelf: Shelf, query: str, scope: str | None) -> list[tuple[Manual, int, float]]:
    """The best pages for a query by meaning, as (manual, page, score)."""
    vector = await recall.embed([query], shelf.client)
    out = []
    # Deeper than SEARCH_HITS: each manual has one row for itself as well as one per page.
    limit = SEARCH_HITS + len(shelf.held)
    for hit in recall.rank(shelf.index, vector[0], scopes={scope} if scope else None, limit=limit):
        owner = shelf.by_scope(hit.item.scope)
        if owner is None or hit.item.kind != "page":
            continue
        out.append((owner, int(Path(hit.item.path).stem), hit.score))
    return out[:SEARCH_HITS]


async def _search(shelf: Shelf, query: str, manual: str | None) -> tuple[str, list]:
    """The search tool's text, and the ranked pages behind it."""
    one = shelf.manual(manual) if manual else None
    if manual and one is None:
        return _missing(shelf, manual), []
    if shelf.client is None or not len(shelf.index):
        return "Search by meaning is not available; use find or contents.", []
    ranked = await _ranked(shelf, query, one.scope if one else None)
    lines = []
    for owner, number, score in ranked:
        page = shelf.pages(owner).get(str(number), {})
        lines.append(
            f"{owner.key} p{number} ({score:.2f}) {page.get('heading', '')}: "
            f"{snippet(_words(page), query)}"
        )
    return "\n".join(lines) or "Nothing matches.", ranked


@librarian.tool
async def search(ctx: RunContext[Shelf], query: str, manual: str | None = None) -> str:
    """Search pages by meaning. The best pages, each with its score, heading and a snippet.

    Args:
        query: what you are looking for, as a phrase.
        manual: a manual's key to search only that one; leave out to search all of them.
    """
    ctx.deps.steps.append(f"search {query!r}" + (f" in {manual}" if manual else ""))
    return (await _search(ctx.deps, query, manual))[0]


@librarian.tool
async def find(ctx: RunContext[Shelf], words: str, manual: str | None = None) -> str:
    """Text match, case-insensitive. Pages with the exact phrase come first, then pages holding
    every one of the words anywhere, the most mentions first.

    Args:
        words: the words - "Menu 8", "AUX1", "lower notes".
        manual: a manual's key; leave out to look in all of them.
    """
    ctx.deps.steps.append(f"find {words!r}" + (f" in {manual}" if manual else ""))
    held = ctx.deps.held
    if manual:
        one = ctx.deps.manual(manual)
        if one is None:
            return _missing(ctx.deps, manual)
        held = [one]
    hits = sorted(_matches(ctx.deps, held, words), key=lambda hit: hit[0], reverse=True)
    out = [line for _, line in hits[:FIND_HITS]]
    if len(hits) > FIND_HITS:
        out.append(f"({len(hits) - FIND_HITS} more pages match; add a word to narrow it)")
    return "\n".join(out) or f"No page contains {words!r}."


def _matches(shelf: Shelf, held: list[Manual], words: str) -> list[tuple[float, str]]:
    """(rank, line) for every page that holds the phrase, or failing that, all of its words."""
    phrase = " ".join(words.lower().split())
    terms = phrase.split()
    found = []
    for one in held:
        for n, page in shelf.pages(one).items():
            text = _words(page)
            low = text.lower()
            exact = low.count(phrase)
            if not exact and not all(term in low for term in terms):
                continue
            rank = exact * 100 + sum(low.count(term) for term in terms)
            line = f"{one.key} p{n} {page.get('heading', '')}: {snippet(text, phrase)}"
            found.append((rank, line))
    return found


@librarian.tool
async def read(ctx: RunContext[Shelf], manual: str, page: int) -> str:
    """One page's full text and what its illustrations show.

    Args:
        manual: the manual's key.
        page: the page number.
    """
    ctx.deps.steps.append(f"read {manual} p{page}")
    one = ctx.deps.manual(manual)
    if one is None:
        return _missing(ctx.deps, manual)
    return _page_text(ctx.deps, one, page)


def _page_text(shelf: Shelf, one: Manual, page: int) -> str:
    found = shelf.pages(one).get(str(page))
    if found is None:
        return f"{one.key} has no page {page}; it has {one.pages}."
    figures = [
        f"[Illustration, {f.get('kind', '')}: {f.get('what', '')}]"
        for f in found.get("figures", [])
        if isinstance(f, dict) and f.get("what")
    ]
    body = f"{one.key} p{page} - {found.get('heading', '')}\n{found.get('text', '')}"
    return "\n".join([body, *figures])[:MAX_PAGE_CHARS]


@librarian.tool
async def look(ctx: RunContext[Shelf], manual: str, page: int) -> BinaryContent | str:
    """The page as an image. Only when the text is not enough: diagrams, wire colours, tables.

    Args:
        manual: the manual's key.
        page: the page number.
    """
    ctx.deps.steps.append(f"look {manual} p{page}")
    one = ctx.deps.manual(manual)
    if one is None:
        return _missing(ctx.deps, manual)
    try:
        blob = page_image(one, page).read_bytes()
    except OSError:
        return f"{one.key} has no picture of page {page}."
    return BinaryContent(blob, media_type="image/jpeg", vendor_metadata={"detail": IMAGE_DETAIL})


def page_image(manual: Manual, page: int) -> Path:
    """Where the render of one page is. The same file the panel shows."""
    return manual.pages_dir / f"{page:04d}.jpg"


# ------------------------------------------------------------------ one question


@dataclass
class Lookup:
    """One question, answered or not, and everything worth writing down about it."""

    question: str
    answer: ManualAnswer
    steps: list[str]
    seconds: float
    requests: int
    cost: float

    def sources(self, held: list[Manual]) -> list[tuple[Manual, int]]:
        out = []
        for source in self.answer.pages:
            one = next((m for m in held if m.key == source.manual), None) or manuals.find(
                held, source.manual
            )
            if one is not None:
                out.append((one, source.page))
        return out


def build_model(settings: Settings) -> tuple[Model, AsyncOpenAI]:
    client = AsyncOpenAI(api_key=settings.api_key, timeout=REQUEST_TIMEOUT_S, max_retries=1)
    return OpenAIResponsesModel(MODEL, provider=OpenAIProvider(openai_client=client)), client


def shelf_for(settings: Settings, client: AsyncOpenAI | None) -> Shelf:
    held = [one for one in manuals.catalog(settings) if one.read]
    return Shelf(held=held, index=recall.only_manuals(recall.load()), client=client)


async def ask(
    question: str, settings: Settings, *, model: Model | None = None, shelf: Shelf | None = None
) -> Lookup:
    """Answer one question from the manuals. Raises on a failed call; the caller says so."""
    started = time.monotonic()
    question = question.strip()[:MAX_QUESTION_CHARS]
    client = None
    if model is None:
        model, client = build_model(settings)
    if shelf is None:
        shelf = shelf_for(settings, client)
    try:
        if not shelf.held:
            answer = ManualAnswer(answered=False, why_not="There are no manuals on the card.")
            return Lookup(question, answer, [], 0.0, 0, 0.0)
        prompt = await _opening(shelf, question)
        result = await librarian.run(
            prompt, deps=shelf, model=model, usage_limits=LIMITS, model_settings=SETTINGS
        )
        usage = result.usage
        return Lookup(
            question=question,
            answer=result.output,
            steps=shelf.steps,
            seconds=round(time.monotonic() - started, 2),
            requests=usage.requests,
            cost=_cost(result.all_messages()),
        )
    finally:
        if client is not None:
            try:
                await client.close()
            except Exception:  # noqa: BLE001 - closing is never worth an exception
                pass


async def _opening(shelf: Shelf, question: str) -> str:
    """The question, the first search, and the top pages already read.

    Read here rather than left to the model, because a request saved is a second saved, and
    the misses in the eval were answers written off a snippet.
    """
    first, ranked = await _search(shelf, question, None)
    shelf.steps.append(f"search {question!r}")
    pages = []
    for one, number, _ in ranked[:PRE_READ]:
        shelf.steps.append(f"read {one.key} p{number}")
        pages.append(_page_text(shelf, one, number))
    read = "\n\n".join(pages)
    return (
        f"{question}\n\nFirst search, already run:\n{first}"
        + (f"\n\nThe top {len(pages)} pages, already read:\n\n{read}" if pages else "")
    )


def _cost(messages: list) -> float:
    total = 0.0
    for message in messages:
        try:
            total += float(message.cost().total_price)
        except Exception:  # noqa: BLE001 - a price table that does not know the model costs 0
            continue
    return round(total, 5)
