"""Describing a finished session: what to call its folder, and what happened in it.

A session's folder is created the moment it starts, when nobody yet knows what the
conversation will be - so it is named for the clock: ``2026-08-26_14-32-05``. That is already
a good name, sortable and honest. At the end, one small model reads the transcript and turns it
into a better one - ``2026-08-26_14-32-05_lego-falcon`` - and, in the same breath, writes the
session down: a one-sentence title and a paragraph of what actually happened.

One round trip for both, deliberately. The teardown budget is a single call, and a folder name
that disagreed with the summary inside it would be worse than either alone.

The paragraph is not decoration. It is what the *next* session is told about this one (see
:func:`cyclops.session.recent_context`), which is why the prompt below asks for the numbers and
the loose ends rather than a tidy precis.

Everything here is best-effort by design. No network, no key, an empty conversation, a slow
answer, a model that ignores the format - all of them land on an empty :class:`Description`, and
the session keeps the name it already had and gets no summary. A folder that is merely dated is a
perfectly good outcome; a kiosk that hangs on shutdown waiting for prose is not. The three parts
are parsed independently, so a mangled reply loses one of them rather than all three.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from openai import APIError, OpenAI

from .config import Settings

# The same pick, for the same reason, as ``search.SEARCH_MODEL``: fastest of the current
# generation, and this is a sentence and a paragraph, not an essay. Kept as its own constant
# because a describing model and a search model happening to be the same string today is a
# coincidence.
DESCRIBE_MODEL = "gpt-5.4-nano"
# Was six seconds when the answer was four words. A paragraph needs the headroom, and this still
# runs alongside the mux rather than after it, so it is normally free.
DESCRIBE_TIMEOUT_S = 10.0
MAX_WORDS = 4
MAX_CHARS = 40
MAX_TITLE_CHARS = 160  # a headline; the prompt asks for a dozen words, this is the backstop
MAX_SUMMARY_CHARS = 1400  # a generous six sentences; also what the next session is handed
# The whole conversation, near enough. The old 3000-char cap was sized for naming, which a
# session does in its opening minute - but a summary written from the first two minutes of a
# forty-minute session is worse than none, because it reads as if it knows.
MAX_TRANSCRIPT_CHARS = 20000
ELISION = "\n[... the middle of this conversation is omitted ...]\n"

# Spelled out line by line, and tediously so, because it had to be: given the shape as a
# template the model read the middle line as an instruction and answered with two lines, which
# cost the summary while leaving the slug looking fine. Numbered rules it obeys.
DESCRIBE_PROMPT = """\
Below is the transcript of a conversation between someone working on something practical -
making, fixing, or figuring a thing out - and Cyclops, the assistant helping them.

Reply with exactly three lines, and nothing else. Not two. Three.

LINE 1 starts with "slug: " and then two to four lowercase words joined by hyphens. It becomes
a folder name on a memory card, so no quotes, no punctuation, no date, no file extension. Name
what the session was about the way someone labelling the folder later would:
    slug: lego-falcon
    slug: bike-brake-torque
    slug: kitchen-tap-leak

LINE 2 starts with "# " and is then ONE short sentence - a dozen words or so, never more than
twenty - saying what this session was. It is a headline, not a precis: it is what gets listed
when someone scans a card full of these. Write it about the work, never about the conversation:
    # Cut the three shelf uprights and found the wall is brick, not plasterboard.
    # Bled the rear brake and replaced the pads.
not "The user and Cyclops discussed shelving" and not "This session covered...".

LINE 3 is ONE paragraph, four to six sentences, of what actually happened: what was worked on,
what was decided or measured, what was tried, and what was left unfinished or unanswered. Keep
the numbers, sizes, part names and decisions - they are the whole point. Write it for two
readers: the person themselves, reading it back weeks later, and Cyclops, reading it at the
start of their next session on this - so end with whatever is still open. Plain prose in the
past tense. No markdown, no bullets, no line breaks inside it, and do not say "the session",
"the user" or "we discussed" - say what was done.

If the conversation was too short or too vague to be about anything, reply with exactly one
line and nothing else:
slug: chat

Conversation:
{transcript}"""


@dataclass(frozen=True)
class Description:
    """What a small model made of a session. Any field may be empty; empty means "say nothing"."""

    slug: str = ""  # "" leaves the folder with the date-stamped name it already has
    title: str = ""  # one sentence; the ``#`` line of summary.md, and one line of the next recap
    summary: str = ""  # one paragraph; the body of summary.md, and the bulk of the next recap

    def __bool__(self) -> bool:
        return bool(self.slug or self.title or self.summary)

    @property
    def page(self) -> str:
        """``summary.md``, or ``""`` when there is not enough to write one."""
        if not self.title:
            return ""  # a paragraph with no sentence over it is not the file we promised
        body = f"# {self.title}\n"
        if self.summary:
            body += f"\n{self.summary}\n"
        return body


def describe_session(transcript: str, settings: Settings) -> Description:
    """Ask a small model what this session was and what happened in it.

    An empty :class:`Description` is a first-class outcome, not a failure, and this never
    raises: the caller is a teardown path that must finish either way.
    """
    transcript = _fit(transcript.strip())
    if not transcript or not settings.slug or not settings.api_key:
        return Description()  # nothing to describe, or describing is off - don't call at all

    # max_retries=0 is load-bearing: the SDK retries twice by default, which would silently turn
    # a 10 s budget into 30 s of a kiosk that looks hung after you tapped stop.
    client = OpenAI(api_key=settings.api_key, timeout=DESCRIBE_TIMEOUT_S, max_retries=0)
    try:
        response = client.responses.create(
            model=DESCRIBE_MODEL,
            reasoning={"effort": "none"},  # a folder name and a paragraph need no deliberation
            input=DESCRIBE_PROMPT.format(transcript=transcript),
        )
        answer = (getattr(response, "output_text", "") or "").strip()
    except (APIError, OSError, ValueError):
        return Description()  # offline, refused, timed out - the date-stamped name stands
    except Exception:  # noqa: BLE001 - a name is never worth failing a teardown over
        return Description()
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass
    return parse(answer)


def _fit(transcript: str) -> str:
    """The conversation, trimmed to fit - from the middle, so both ends survive.

    Trimming from the front would drop how it ended, which is the half a summary is mostly
    about: what was left open, what was decided last. Trimming from the back would drop what it
    was, which is what the slug needs. So the middle goes.
    """
    if len(transcript) <= MAX_TRANSCRIPT_CHARS:
        return transcript
    keep = (MAX_TRANSCRIPT_CHARS - len(ELISION)) // 2
    return transcript[:keep] + ELISION + transcript[-keep:]


def parse(answer: str) -> Description:
    """Pull the three parts out of a reply, each independently of the other two.

    Deliberately forgiving about everything except structure: a model that adds a preamble,
    drops the slug line, or forgets the ``#`` still gives us whatever it did get right.

    The heading is the part most often missed - the model writes the paragraph and considers the
    job done - so a reply without one is not thrown away: its first sentence becomes the title,
    which is what the heading was going to be anyway. Prose that arrives *before* a heading is
    dropped as preamble; prose in a reply that has no heading at all is the summary.
    """
    slug = title = ""
    before: list[str] = []
    after: list[str] = []
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not slug and stripped.lower().startswith("slug:"):
            slug = slugify(stripped[len("slug:") :])
            continue
        if not title and stripped.startswith("#"):
            title = _clean(stripped.lstrip("#"), MAX_TITLE_CHARS)
            continue
        (after if title else before).append(stripped)
    summary = _clean(" ".join(after if title else before), MAX_SUMMARY_CHARS)
    return Description(slug=slug, title=title or _first_sentence(summary), summary=summary)


def _first_sentence(text: str) -> str:
    """The opening sentence, for when the model gave us a paragraph but no heading over it.

    Splits only on a full stop that is followed by a space, so "1980mm." ends a sentence and
    "12.5" does not.
    """
    if not text:
        return ""
    end = re.search(r"(?<=[.!?])\s", text)
    return _clean(text if end is None else text[: end.start()], MAX_TITLE_CHARS)


def _clean(text: str, limit: int) -> str:
    """One line of plain text, collapsed and capped at a word boundary."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rsplit(" ", 1)[0] + "…"


def slugify(text: str) -> str:
    """A model's answer, reduced to something safe to put in a filename.

    Applied unconditionally and never skipped: this is the one place a filename is built out of
    text a model chose, and a model that decides to answer with a path, a quote or an emoji must
    not be able to reach the filesystem with it.
    """
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    if not text:
        return ""
    text = "-".join(text.split("-")[:MAX_WORDS])[:MAX_CHARS].strip("-")
    return "" if text in {"", "chat"} else text
