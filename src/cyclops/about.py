"""What Cyclops knows about the person it works with, and how it comes to know it.

Cyclops already remembers two things across sessions: what happened (the last few ``summary.md``
cards) and what is being built (``projects/``). Neither of them is about *who is on the other
side of the bench*, so every session used to open on a stranger - no name, no idea what they do,
no memory of what they said last week they wanted help with.

This is the third kind of memory and the smallest: a short list of standing facts, in one file
at the root of the card.

    # About you

    - Goes by Marco.
    - Embedded work by trade; comfortable with a soldering iron and a scope.

**The list is rewritten, not appended to.** At the end of a session a small model is handed the
list as it stands plus the transcript, and answers with the whole list back - so a fact that
changed gets corrected and one that is finished with gets dropped, rather than the file growing
a contradictory second copy of everything.

That trade has one sharp edge, and :func:`remember` is built around it: **an empty answer never
empties the file**. Every failure here - no key, offline, refused, timed out, a reply in the
wrong shape, a model that decided to say nothing - lands on ``[]``, and ``[]`` means *change
nothing* rather than *forget everything*. The only thing that can shorten the list is a reply
that is well-formed and shorter, or a person opening the file and deleting a line. Emptying it
altogether is deliberately something only a person can do.

The header is a constant here and is never read back: :func:`read` takes the bullets and nothing
else, and :func:`write` lays the prose over them again. So the file is rebuilt from its data
every time and the explanation at the top of it can never drift from what the code does.
"""

from __future__ import annotations

import re

from openai import APIError, OpenAI

from . import card
from .config import Settings
from .slug import fit

# NOT ``slug.DESCRIBE_MODEL``'s nano, and the difference between the two calls is the reason.
# Naming a session is a transcription: read a conversation, say what it was. This one is a
# reconciliation - hold a list of sentences about somebody against a new conversation and work
# out which of them it has just made untrue - and that is a judgement, which is why this reaches
# for the same model the matcher and the diagram writer do rather than the cheapest one going.
#
# Measured, because "smarter is better" is not a reason. Handed a list saying "uses Fusion 360"
# and a conversation saying they have moved to OnShape, five times over: nano got it clean 4/5,
# once leaving a spare line about sketching being quicker in OnShape - a sentence about a week,
# in a list that is supposed to be about a person. mini and this both went 5/5. Over the 32 real
# sessions on the card the two were also not the trade the price list implies: nano ran a median
# 1.4 s and a slowest 12.3 s, over the timeout below; this one 1.0 s and 5.3 s. mini is the other
# right answer and was dropped for splitting one stated goal across three lines.
REMEMBER_MODEL = "gpt-5.6-terra"
# Matches slug.DESCRIBE_TIMEOUT_S, because it is the same budget: both calls go out together at
# the top of a teardown and share the one join below the mux.
REMEMBER_TIMEOUT_S = 10.0

MAX_FACTS = 12  # a page of somebody, not a dossier; the model is told the number too
MAX_FACT_CHARS = 140  # one short sentence

# Everything a line may be introduced by. Written down rather than guessed at because the model
# is asked for "- " and a person editing the file by hand will write whatever their editor does.
BULLET = re.compile(r"^\s*(?:[-*•–]|\d+[.)])\s+")

HEADER = """\
# About you

What Cyclops has picked up about the person it works with. It reads this at the start of every
session and rewrites it at the end of one. Edit or delete any line by hand - it will not put a
line back unless it hears it again. Delete the file and it starts over knowing nobody.
"""

# Braces are deliberately absent from the prose below: this goes through str.format.
REMEMBER_PROMPT = """\
Below is the transcript of a conversation between someone working on something practical -
making, fixing, or figuring a thing out - and Cyclops, the assistant on the bench beside them.

Cyclops keeps a short list of standing facts about this person, so that the next conversation
starts with it already knowing who it is talking to. Here is that list as it stands:

{facts}

Your job is to hand back that list as it should read after this conversation. Work through it
line by line against the transcript:

- If the transcript contradicts a line or moves past it, REPLACE that line. Do not keep the old
  one as well. The list must never hold two lines that disagree with each other.
- If a line is about something they have finished with, take it out.
- If the transcript says nothing about a line, keep it exactly as it is, word for word.
- If they said something new about themselves, add a line for it.

A line earns its place only if it would change how Cyclops talks to them in three weeks:
- what they are called
- what they do, and how experienced they are with this kind of work
- what they have and work with: the space, the tools, the machines, the software
- what kind of work they want to get better at - a skill, never a job: "wants to get better
  at brazing" belongs here, "wants help with the gate hinge" does not
- how they want to be talked to: which language, which units, how much detail, and anything
  about hearing, sight or handedness that changes what help looks like

Ask of every line: would this still be worth knowing if today's job had never happened? If not,
it does not belong here. So nothing about what was on the bench, what it measured, what was
decided, or how the job went - all of that is written down elsewhere. Nothing they did not say
about themselves. Nothing you are inferring, guessing at, or being polite about. Nothing private
they did not offer as a standing fact: health, money, who they live with, where they live.

Most conversations add nothing to this list. That is the ordinary outcome and not a failure to
try harder: people talk about the job, not about themselves. Before adding a line, find the
words they said it in - if you cannot point at them, it does not go in. A line naming the thing
on the bench is about the job however much of the conversation it was, and a line saying they
want help with that thing is about the job too. A project they are working on is never a fact
about them: projects are kept elsewhere, and Cyclops is already told their names alongside this
list, so a line about one is the same sentence twice. Adding nothing is nearly always the right
answer, and a list of four true sentences is worth more than a list of ten that are mostly today.

One short plain sentence per line, ten words or so. Write ABOUT them, in the third person.
Never "I" and never "you": this list is read by Cyclops, so an "I" in it means Cyclops. Write
each line the way you would introduce them to somebody - "Goes by Priya.", "Restores furniture
at weekends.", "Deaf in the left ear, so say things once." At most {max_facts} lines, the most
useful first.

Reply with the COMPLETE list and nothing else: plain lines each starting with "- ", no heading,
no preamble, no numbering, no blank lines between them, no note about what you changed. If the
transcript adds nothing and corrects nothing, reply with the list exactly as it stands above. If
there is nothing to say about them at all, reply with one line and nothing else:
none

Conversation:
{transcript}"""

NOTHING_YET = "(nothing written down about them yet)"


# ------------------------------------------------------------------ the file


def read(settings: Settings) -> list[str]:
    """The facts on the card, or an empty list. Never raises for anything a card can do.

    Forgiving about the bullet, because the file is meant to be edited by hand and nobody should
    have to know that this parser exists. Strict about everything else: prose that is not a
    bullet is the header, and the header is not a fact.
    """
    try:
        text = settings.about_file.expanduser().read_text(encoding="utf-8")
    except OSError:
        return []
    return _facts(text)


def write(settings: Settings, facts: list[str]) -> None:
    """Lay the prose back over the list and land it whole. Never raises.

    Through :mod:`cyclops.card`, like everything else that lands on a card that gets unplugged:
    scratch file, fsync, rename. A power cut leaves the list as it was or as it now is, never a
    half of either and never the zero-byte file that used to be the failure mode here.

    **An empty list is refused**, and the check lives here rather than only in the caller. The
    module's one rule is that nothing a model says can empty this file, and a rule enforced only
    at the call site is one the next caller does not know about - which is exactly what happened
    the first time this was run against a real session on the card: the conversation was entirely
    about a photograph, taught us nothing about anybody, and a header with no list under it
    landed anyway. Emptying it stays what it should be: a person, with an editor.
    """
    if not facts:
        return
    body = "\n".join(f"- {fact}" for fact in facts)
    try:
        card.write_text(settings.about_file.expanduser(), f"{HEADER}\n{body}\n")
    except OSError as exc:
        print(f"· could not write {settings.about_file} ({exc})", flush=True)


# ------------------------------------------------------------------ the model call


def remember(transcript: str, facts: list[str], settings: Settings) -> list[str]:
    """The list as it should be after this conversation, or ``[]`` for "leave it alone".

    Never raises: the caller is a teardown path that has to finish either way. And every way
    this can go wrong returns the same empty list, which the caller must read as *change
    nothing* - see the module docstring. A conversation that taught us nothing about somebody is
    not a failure, it is most conversations.
    """
    transcript = fit(transcript.strip())
    if not transcript or not settings.remember or not settings.api_key:
        return []

    listing = "\n".join(f"- {fact}" for fact in facts) or NOTHING_YET
    prompt = REMEMBER_PROMPT.format(
        facts=listing, max_facts=MAX_FACTS, transcript=transcript
    )
    # max_retries=0 for the reason slug.describe_session gives: the SDK retries twice by default,
    # which would quietly turn a 10 s budget into 30 s of a kiosk that looks hung after you
    # tapped GO TO SLEEP.
    client = OpenAI(api_key=settings.api_key, timeout=REMEMBER_TIMEOUT_S, max_retries=0)
    try:
        response = client.responses.create(
            model=REMEMBER_MODEL,
            # "low", not the "none" slug.py uses, and the difference is visible in the output:
            # reconciling a new sentence against a list already holding an older one is a
            # judgement, not a transcription. At "none" it appended "moved to OnShape" under a
            # standing "uses Fusion 360" and left the list contradicting itself - which is the
            # one failure a rewritten list is supposed to be immune to.
            reasoning={"effort": "low"},
            input=prompt,
        )
        answer = (getattr(response, "output_text", "") or "").strip()
    except (APIError, OSError, ValueError):
        return []  # offline, refused, timed out - the list stands
    except Exception:  # noqa: BLE001 - knowing somebody is never worth failing a teardown over
        return []
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass
    return parse(answer)


def parse(answer: str) -> list[str]:
    """The bullets out of a reply, ignoring everything that is not one.

    As forgiving as :func:`cyclops.slug.parse` and for the same reason: a model that opens with
    "Here is the updated list:" or wraps the thing in a heading has still done the job, and
    throwing the answer away over its packaging would cost a session's worth of knowing someone.
    """
    return _facts(answer)


def _facts(text: str) -> list[str]:
    """Every bullet in a block of text: collapsed, capped, deduped, in the order given."""
    out: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not (marker := BULLET.match(stripped)):
            continue
        fact = _clean(stripped[marker.end() :])
        # "none" is the prompt's own word for "nothing to say", and a model that answers it as a
        # bullet rather than a bare line must not leave us with a fact called none.
        if not fact or fact.rstrip(".").casefold() in {"none", "nothing"}:
            continue
        if (key := fact.casefold()) in seen:
            continue
        seen.add(key)
        out.append(fact)
        if len(out) >= MAX_FACTS:
            break
    return out


def _clean(text: str) -> str:
    """One line of plain text, collapsed and cut at a word boundary. As ``slug._clean``."""
    text = " ".join(text.split())
    if len(text) <= MAX_FACT_CHARS:
        return text
    return text[: MAX_FACT_CHARS - 1].rsplit(" ", 1)[0] + "…"
