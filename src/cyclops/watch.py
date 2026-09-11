"""Which video, and where in it. The half of watching that needs a model.

:mod:`cyclops.youtube` fetches; this chooses. Two side-car calls to the Responses API - the
same bridge :mod:`cyclops.search` builds for web search, and for the same reason: a Realtime
session takes custom ``function`` tools and nothing else, so thinking a tool needs has to
happen here and arrive as a result.

**The point of this module is what it does not return.** A video's transcript runs to
thousands of tokens. It is read here, by a model that answers with a single number, and then
it is dropped. What goes back is a title, a channel and a second - about thirty tokens,
which a Realtime session can carry for the rest of a conversation without noticing. Put a
transcript in a ``function_call_output`` and every later turn pays for it again.

The two calls are asked in this order deliberately. Ranking is a judgement over titles, so
it happens before anything per-video is fetched: pulling chapters for five losers is five
round trips to inform one decision, and it quietly biases the choice towards whoever writes
the nicest chapter titles.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from openai import APIError, AsyncOpenAI

from . import youtube
from .config import Settings


class WatchError(RuntimeError):
    """No video could be found or played. The message is meant to be relayed, not logged."""


# gpt-5.4-nano is the house default for a side-car (search.py, captions.py) and it is the
# wrong pick here. Measured 2026-09-11 over four questions, scored against one video's own
# chapter boundaries: mini answered 4/4 at a 0.94 s median, nano 3/4 at 0.99 s - it put
# "where does he hone the edge" four minutes late, in the stropping section. Four questions
# is a thin sample, but nano is not faster here, so nothing is being traded for the miss.
# MOMENT_PROMPT's two-sentence middle paragraph was measured, not written. Scored against
# chapter boundaries over nine questions on three videos, four wordings ran 6/9, 7/9, 6/9 and
# 8/9; this is the 8/9, and its one "miss" was a question the video does not answer, where 0
# is the honest reply. Without that paragraph the two whole-job questions both landed in the
# middle of a video that was entirely about them - the worst failure available here, because
# it silently skips the part somebody actually needed.
PICK_MODEL = "gpt-5.4-mini"
PICK_TIMEOUT_S = 12.0
MAX_REQUEST_CHARS = 300
# How many the ranking call may name. Every one past the first costs a metadata fetch only
# if the one before it turned out to be unplayable, so three is cheap insurance.
SHORTLIST = 3

RANK_PROMPT = """\
Somebody at a workbench asked: {request}

Here are YouTube search results. Pick the ones most likely to actually show the job being
done - a clear demonstration by somebody who knows it, rather than a review, a compilation,
a reaction or an advert. Prefer one long enough to explain and short enough to watch.

{candidates}

Reply with the ids of the best {count}, best first, one per line. Ids only, nothing else."""

MOMENT_PROMPT = """\
Somebody at a workbench asked: {request}

Below is one video in sections, each with the second it begins at. Find where the answer to
their question actually starts - where the doing begins, not where it is introduced.

{sections}

Their question names either one step of the job or the whole job. If it names a step, give
the second that step begins. If it names the whole job, give 0.

Reply with only the number of seconds. No units, no other words."""


@dataclass(frozen=True)
class Watch:
    """One video, opened at one moment."""

    id: str
    title: str
    channel: str
    start: int
    stream: str
    thumb: str

    @property
    def clock(self) -> str:
        """The start as somebody would say it aloud: "5:12"."""
        return _clock(self.start)


def _clock(seconds: int) -> str:
    return f"{seconds // 60}:{seconds % 60:02d}"


def _shorten(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


def ids_named(reply: str, known: set[str]) -> list[str]:
    """The ids a ranking reply named, in its order, ignoring anything it invented."""
    out: list[str] = []
    for line in reply.splitlines():
        found = line.strip().strip("-•.\"' \t")
        if found in known and found not in out:
            out.append(found)
    return out


def seconds_named(reply: str, duration: int) -> int:
    """The number a localising reply named, clamped inside the video. 0 when it named none."""
    lines = reply.strip().splitlines()
    digits = "".join(c for c in lines[0] if c.isdigit())[:7] if lines else ""
    if not digits:
        return 0
    # Clamped, because a model that answers in milliseconds - or hallucinates a number off
    # the end - would otherwise seek past the end and leave the panel on a black frame.
    return max(0, min(int(digits), max(0, duration - 1)))


def sections_of(video: youtube.Video, spoken: list[tuple[int, str]]) -> str:
    """The video as numbered sections for :data:`MOMENT_PROMPT`, chapters first if it has them.

    Chapters are preferred over the transcript, and not only because they are two thousand
    tokens cheaper. A chapter is where the person who made the video says a part begins,
    which is a better answer than where a sentence about it happens to fall.
    """
    if video.chapters:
        return "\n".join(f"[{at}] {title}" for at, title in video.chapters)
    return "\n".join(f"[{at}] {said}" for at, said in spoken)


async def _ask(client: AsyncOpenAI, prompt: str) -> str:
    try:
        async with asyncio.timeout(PICK_TIMEOUT_S):
            response = await client.responses.create(
                model=PICK_MODEL,
                reasoning={"effort": "none"},  # seconds matter more than deliberation here
                input=prompt,
            )
    except TimeoutError as exc:
        raise WatchError(f"deciding which video took longer than {PICK_TIMEOUT_S:.0f}s") from exc
    except APIError as exc:
        raise WatchError(f"the video lookup failed: {exc.message or exc}") from exc
    return (getattr(response, "output_text", "") or "").strip()


async def _shortlist(
    client: AsyncOpenAI, request: str, found: list[youtube.Candidate]
) -> list[str]:
    lines = "\n".join(f"{c.id} · {_clock(c.duration)} · {c.channel} · {c.title}" for c in found)
    reply = await _ask(
        client, RANK_PROMPT.format(request=request, candidates=lines, count=SHORTLIST)
    )
    named = ids_named(reply, {c.id for c in found})
    # A reply naming nothing we offered is not worth a second round trip. The search already
    # put these in an order and its own first results are a fair answer.
    return (named or [c.id for c in found])[:SHORTLIST]


async def _playable(shortlist: list[str]) -> youtube.Video:
    """The first of the shortlist that has a stream we can actually play."""
    for ident in shortlist:
        video = await asyncio.to_thread(youtube.details, ident)
        if video is not None and video.stream:
            return video
    raise WatchError("none of the videos found could be played on this screen")


async def _moment(client: AsyncOpenAI, request: str, video: youtube.Video) -> int:
    """Where in this video the answer starts, or 0 when there is nothing to go on.

    A video with neither chapters nor captions is not a failure - it is a video that starts
    at the beginning, which is where it would have started anyway.
    """
    spoken = [] if video.chapters else await asyncio.to_thread(youtube.windows, video.id)
    if not video.chapters and not spoken:
        return 0
    sections = sections_of(video, spoken)
    return seconds_named(
        await _ask(client, MOMENT_PROMPT.format(request=request, sections=sections)), video.duration
    )


async def find_video(request: str, settings: Settings) -> Watch:
    """One video and the second to start it at, for a request said out loud.

    Raises :class:`WatchError` with a sentence meant for the model to relay, never a
    traceback - the voice session must always get a tool result back, even a disappointing
    one.
    """
    request = _shorten(request, MAX_REQUEST_CHARS)
    if not request:
        raise WatchError("no idea what to look for was given")

    found = await asyncio.to_thread(youtube.search, request)
    if not found:
        raise WatchError("nothing on YouTube matched that")

    client = AsyncOpenAI(api_key=settings.api_key)
    try:
        video = await _playable(await _shortlist(client, request, found))
        start = await _moment(client, request, video)
    finally:
        await client.close()

    return Watch(
        id=video.id,
        title=video.title,
        channel=video.channel,
        start=start,
        stream=video.stream,
        thumb=video.thumb,
    )


async def restream(video_id: str) -> str:
    """A fresh playable URL for a video we already know about. "" when it cannot be had.

    Signed googlevideo URLs last about six hours, so a reference kept on the card stores the
    id and the second and comes back through here. It costs one metadata fetch, which is why
    a saved reference carries its own thumbnail: the picture can be on the glass before this
    has answered.
    """
    video = await asyncio.to_thread(youtube.details, video_id)
    return video.stream if video is not None else ""
