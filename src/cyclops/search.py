"""Web search for the voice agent, bridged from the Responses API.

OpenAI's hosted ``web_search`` tool is real - you do not implement searching yourself - but it
only exists on the Responses API. A Realtime session accepts custom ``function`` tools and
nothing else: no hosted tools, no remote MCP. OpenAI's stated reason is lifecycle, and it is a
fair one: a voice user can talk over a search in progress, so somebody has to decide whether to
cancel the lookup, what to do with a late answer, and how to keep a stale result out of the
conversation. That decision belongs to the app, so it lives here.

The bridge is small: the Realtime model calls our ``web_search`` function, we run a Responses
call with the hosted tool, and hand the text back as ``function_call_output``.

What comes back is *source material*, not a script. It lands in the conversation as tool output
and the Realtime model then composes its own spoken reply from it - nothing here is read out
verbatim. So this asks for a dense, factual answer with its numbers and caveats intact and
leaves the job of sounding human to the voice model, which is better at it. It also stays
deliberately terse: a search that takes half a minute is useless mid-conversation, so this is
tuned to answer in under five seconds even though a slower model would answer more richly.

Source URLs are kept rather than stripped. They are never spoken - the session instructions
forbid reading URLs aloud - but they stay in the conversation so a later fetch tool can pull up
a manual or spec sheet the search only summarised.
"""

from __future__ import annotations

import asyncio

from openai import APIError, AsyncOpenAI

from .config import Settings

# Model choice barely moves the needle: latency here is the hosted tool fetching pages, not the
# model thinking. Measured sequentially with reasoning disabled, gpt-5.4-nano ran 3.5-7.4 s
# (median 5.0 s), gpt-5.4-mini 4.9 s mean, gpt-5.6-terra 6.2 s and never under five. So nano is
# the pick - fastest of the current generation - but a sub-five-second guarantee is not on offer
# at this layer. What makes the wait bearable is the assistant saying "let me look that up"
# before calling, and the UI showing a searching state; both are handled by the caller.
# `reasoning={"effort": "none"}`: "minimal" is rejected outright alongside web_search, and
# anything higher costs seconds. `return_token_budget` is not a lever - it takes only
# "default" or "unlimited", never a number.
SEARCH_MODEL = "gpt-5.4-nano"
# Above the 7.4 s worst case seen, because a timeout is the worst outcome available: the user
# waited and got nothing. This catches a genuinely hung search, not a merely slow one.
SEARCH_TIMEOUT_S = 12.0
MAX_QUERY_CHARS = 300
MAX_ANSWER_CHARS = 1500  # context for the voice model to condense, not a line to be spoken

SEARCH_PROMPT = """\
Search the web and answer in at most 3 sentences. Your answer is not shown to a person - it is
handed to a voice assistant as reference material, and that assistant decides what to say. So
optimise for facts per second, not for style:

- Lead with the direct answer. Keep every number, unit, size, tolerance and model code exact.
- Add source URLs in parentheses. They are kept deliberately: the assistant can fetch one later
  to read a spec sheet in full, so a good link is worth more than a bare site name.
- If sources disagree or it depends on the variant, say so in a clause, not a paragraph.
- If you could not find a reliable answer, say that plainly rather than guessing.
- Plain prose only. No markdown whatsoever - no bold, no asterisks, no headings, no bullets.

Question: {query}"""


class SearchError(RuntimeError):
    """The search could not be completed."""


def _shorten(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rsplit(" ", 1)[0] + "…"


async def search_web(query: str, settings: Settings) -> str:
    """Run one hosted web search and return a dense factual answer for the voice model to use.

    Raises :class:`SearchError` with a message meant for the model to relay, never a traceback -
    the voice session must always get a tool result back, even a disappointing one.
    """
    query = _shorten(query, MAX_QUERY_CHARS)
    if not query:
        raise SearchError("no search query was given")

    client = AsyncOpenAI(api_key=settings.api_key)
    try:
        async with asyncio.timeout(SEARCH_TIMEOUT_S):
            response = await client.responses.create(
                model=SEARCH_MODEL,
                tools=[{"type": "web_search"}],
                reasoning={"effort": "none"},  # seconds matter more than deliberation here
                input=SEARCH_PROMPT.format(query=query),
            )
    except TimeoutError as exc:
        raise SearchError(f"the search took longer than {SEARCH_TIMEOUT_S:.0f}s") from exc
    except APIError as exc:
        raise SearchError(f"the search failed: {exc.message or exc}") from exc
    finally:
        await client.close()

    text = (getattr(response, "output_text", "") or "").strip()
    if not text:
        raise SearchError("the search came back empty")
    return _shorten(text, MAX_ANSWER_CHARS)
