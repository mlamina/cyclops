"""The Director and the three subagents it works with, and the prompts that make them useful.

One agent is handed a finished recording and works out what stories it holds by calling the
others. Delegation is Pydantic AI's agent-as-tool pattern, the same as the filing pipeline:
``deps=ctx.deps`` and ``usage=ctx.usage`` on every subagent run, so one budget covers the whole
session and a runaway loop runs out of money rather than out of patience.

Two things this module deliberately cannot do.

**It cannot write a file.** Every agent returns a typed answer; :mod:`cyclops.cut` shapes the
beats against measured speech and does every byte and every encode afterwards.

**It cannot pass a story it has not tested.** The Editor is wired as the Director's output
validator rather than as a step after it, so "the clip does not stand alone" comes back as a
:class:`ModelRetry` naming the missing beat instead of as a line in a log nobody reads. One
repair; a story that still cannot be told is dropped and its neighbours are kept.

Importing this module pulls in ``pydantic_ai``, which is a second or two on a Pi, so
:mod:`cyclops.cut` imports it lazily and only after the two cheap gates have let a session
through - exactly as ``session.py`` imports ``slug``, and for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from openai import AsyncOpenAI
from pydantic_ai import Agent, BinaryContent, ModelRetry, RunContext, UsageLimits
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

from ..config import Settings
from ..cut import MIN_SHOT_S, STORY_CAP_S, STORY_LOW_S, TURN_HOLD_S, WINDOW_S
from .deps import Telling
from .models import NOTHING_TO_SEE, Arc, Arcs, Beat, Direction, Look, Retelling, Told

# The smart model directs; the cheap one reads the transcript and writes the retelling; the
# smallest one looks at a single frame and names what is on it. Nothing here has to be clever
# about pictures - the question the Looker is asked has five possible answers.
DIRECTOR_MODEL = "gpt-5.6-terra"
READER_MODEL = "gpt-5.4-mini"
LOOKER_MODEL = "gpt-5.4-nano"
EDITOR_MODEL = "gpt-5.4-mini"

REQUEST_TIMEOUT_S = 60.0
SDK_RETRIES = 2

# At most three candidates reach the Director and at most two stories leave it. Both caps are
# also in the types (``Arcs.stories`` and ``Direction.stories`` are bounded lists), so a model
# that ignores the prompt is corrected by the schema rather than by a slice here.
MAX_CANDIDATES = 3

# One budget for one session, and it has to be one: ``usage`` is shared down the delegation
# chain, so a limit passed to a subagent is a limit on the whole run. See the same note in
# projects/agents.py. Fifteen requests is the Reader, the Director, four frames and four
# retellings with room for one repair; the cost limit is the thing actually holding the purse.
RUN_LIMITS = UsageLimits(request_limit=15, cost_limit=Decimal("0.30"))


@dataclass(frozen=True)
class Models:
    """One client and four bound models, built once per session."""

    client: AsyncOpenAI
    director: OpenAIResponsesModel
    reader: OpenAIResponsesModel
    looker: OpenAIResponsesModel
    editor: OpenAIResponsesModel

    @classmethod
    def build(cls, settings: Settings) -> Models:
        client = AsyncOpenAI(
            api_key=settings.api_key, timeout=REQUEST_TIMEOUT_S, max_retries=SDK_RETRIES
        )
        provider = OpenAIProvider(openai_client=client)
        return cls(
            client=client,
            director=OpenAIResponsesModel(DIRECTOR_MODEL, provider=provider),
            reader=OpenAIResponsesModel(READER_MODEL, provider=provider),
            looker=OpenAIResponsesModel(LOOKER_MODEL, provider=provider),
            editor=OpenAIResponsesModel(EDITOR_MODEL, provider=provider),
        )

    async def close(self) -> None:
        try:
            await self.client.close()
        except Exception:  # noqa: BLE001 - tearing down is never worth an exception
            pass


# ------------------------------------------------------------------ the Reader

READER_INSTRUCTIONS = f"""\
Below is the timeline of a recorded session between somebody making or fixing something at their
workbench and Cyclops, the assistant on the bench beside them. Every time is a position in the
recording, in seconds.

Your job is to find the STORIES in it. Not moments - stories. A story is three beats:

  SETUP   The person asks for something, or holds something up. Their WHOLE line, from the first
          word to the last. Never a fragment, and never Cyclops talking.
  TURN    The thing happens. The picture arrives, the drawing lands, the number goes up on the
          panel. This beat must hold the new picture on screen for at least {TURN_HOLD_S:.0f}
          seconds after it appears, so a viewer sees what arrived.
  PAYOFF  How it ended. The reaction, the closing line, or the result sitting on the screen.

Read the marked lines exactly this way. They are the only way you can tell what a viewer sees.

  [photo] ...              THE PICTURE CHANGED. A new image is on the screen.
  [wrote on the panel] ... THE PICTURE CHANGED. A number, a diagram or a few words fill it.
  [drew on the panel] ...  THE PICTURE CHANGED. Cyclops drew something on the screen.
  [found something on the card] ...  An OLD picture out of storage. Never a TURN.
  [searched the web] ...   NOTHING HAPPENS ON SCREEN. Never a TURN.

The TURN beat must contain one of the first three. A session with no line of those three holds no
story, and saying so is the right answer.

REJECT, and say nothing rather than stretch. These are most of what you are for - every story you
propose costs real money downstream, and a bad one costs it twice:

- THE PICTURE IS OF NOTHING. A view out of a window, a wall, a doorway, a ceiling, an empty
  bench, a bare face. There is no object in it, so there is nothing to look at.
- THE PHOTO FAILED and Cyclops's own next line says so: blurry, too close, off angle, cannot tell
  what it is. However interesting the object was.
- A DEMO OF THE TOOL rather than a picture of the work. Boxes reading INPUT, PROCESS, OUTPUT,
  CONTROLLER or MODULE. "Show me what the scratchpad can do." Anything drawn to prove a feature
  works. Judge a drawing by what is written IN it: real parts, real values and the actual machine
  on the bench make it real, and nothing else does.
- A REPEAT. A second snap of something already on the screen this session, or a tidier version of
  a diagram already drawn. The first one was the story.
- NOBODY ASKED AND NOBODY REACTED. The picture just appears and the session moves on. With no
  SETUP and no PAYOFF there is no story, only a moment.
- PRIVATE PAPERWORK or an INJURY. A home address, a VIN or serial number, a registration or an
  insurance document, or a close-up of somebody's injury. These clips get shared with people who
  were not there. Never propose one.

Rules for the beats themselves:

- In session order: setup before turn before payoff, never overlapping.
- First number to last number no more than {WINDOW_S:.0f} seconds apart. A story that needs more
  of the session than that is not one clip.
- No beat shorter than {MIN_SHOT_S:.1f} seconds. The TURN holds at least {TURN_HOLD_S:.0f}.
- The three together should come to {STORY_LOW_S:.0f}-{STORY_CAP_S:.0f} seconds, aiming for 20-35.
- Overshooting the END of a beat costs nothing; the silence is measured and trimmed off before
  anything is rendered. Undershooting cuts somebody off mid-word and cannot be repaired.
- The SETUP beat must start on the person's own line, a second or so before their first word.

At most {MAX_CANDIDATES} candidates, best first. An empty list is a normal, common, cheap and
correct answer - most sessions here are somebody checking the microphone works. When you return
none, say in one sentence why.
"""

reader = Agent(
    name="stories-reader",
    deps_type=Telling,
    output_type=Arcs,
    instructions=READER_INSTRUCTIONS,
    retries=2,
)


# ------------------------------------------------------------------ the Looker

LOOKER_INSTRUCTIONS = """\
You are shown one frame from a recording of a small screen sitting on a workbench. Say what is on
that screen, in one word from the list and a few words of description.

  photo       a photograph - the bench, a part, a tool, a machine, a person, a page of a manual.
  drawing     a drawing, a diagram, a schematic, or a photograph that has plainly been redrawn,
              cut out, recoloured or turned into an illustration.
  panel_text  words, numbers, a list or a table filling the screen. A written answer.
  eye         Cyclops's own face: a single round eye on a dark background, and nothing else.
  black       nothing. A black or near-black frame, or a frame too dark to make anything out.

The eye is what the screen shows when there is no picture up, so `eye` and `black` both mean
nothing arrived. Be strict about those two: if a picture is up at all, however plain, it is one
of the first three.
"""

looker = Agent(
    name="stories-looker",
    deps_type=Telling,
    output_type=Look,
    instructions=LOOKER_INSTRUCTIONS,
    retries=1,
)


# ------------------------------------------------------------------ the Editor

EDITOR_INSTRUCTIONS = """\
You are watching one short clip cold. You were not there, you have not read the session, and what
is below is everything you get: the shots it is cut from, what is on the screen in each, and
every word spoken inside them.

Tell it back in three parts, and only from what is below:

  asked   what the person wanted.
  did     what Cyclops did about it.
  ended   how it ended - the reaction, the result, or the closing line.

Then say whether it is `clear`: true only if all three are genuinely there and a stranger walking
past would follow this without being told anything.

Be strict, and say false when you have to. False is the useful answer - it is what stops a clip
that opens mid-conversation and stops before the picture lands from being published. In
particular, false when:

- nobody ever says what they want, so the clip opens on an answer to a question nobody heard
- nothing appears on the screen and nothing in the words says what happened
- it stops before the end: the picture arrives and the clip cuts away with nobody reacting

When it is false, `missing` names WHICH of the three is absent and what the clip would need -
"no ask: the clip opens on Cyclops answering; it needs their question, about 20 seconds earlier".
That sentence is handed straight back to the Director to repair, so make it about the clip.

Never guess. If the words do not say it, it is not there, and that is the finding.
"""

editor = Agent(
    name="stories-editor",
    deps_type=Telling,
    output_type=Retelling,
    instructions=EDITOR_INSTRUCTIONS,
    retries=1,
)


# ------------------------------------------------------------------ the Director

DIRECTOR_INSTRUCTIONS = f"""\
You are cutting a short clip out of a recorded session at somebody's workbench. It goes on a reel
that autoplays one clip after another, often muted, to somebody walking past. Every clip on that
reel has to stand alone: one mini story a stranger follows cold, with no title card, nothing
written on the frame, and nothing explained.

A clip is three beats in session order, and never anything else:

  SETUP   the person asks for something or holds something up - their whole line.
  TURN    the thing happens: the picture arrives, the drawing lands, the number goes up. It holds
          for at least {TURN_HOLD_S:.0f} seconds so a viewer sees what appeared.
  PAYOFF  how it ended: the reaction, the closing line, or the result sitting on screen.

You cannot see the recording. You have the session's own summary, its timeline, and candidate
stories somebody else found in it. What you can do is LOOK: `look_at` pulls one frame out of the
recording at a second you name and tells you what is on the screen there.

How to work:

1. Read the candidates. Pick the ones that are really stories - a person wanting something, the
   thing happening, and an ending - and drop the rest. One good story beats two thin ones.
2. For each one you are keeping, call `look_at` on the middle of its TURN beat. The recording is
   Cyclops's own screen, and the screen shows its face when nothing is up: a TURN that comes back
   `eye` or `black` is a beat aimed at the wrong second, and the picture is usually a little
   later than the line that announced it. Move the beat and look again, or drop the story.
3. Check the PAYOFF the same way when you are unsure what is on screen at the end.
4. Answer with the stories you are keeping. At most two, and never two that would look alike on
   one reel - the same object snapped twice, or a drawing and a written version of it, is one
   story and it is the first of them.

The rules your beats must obey, because they are checked and a break comes back to you:

- setup before turn before payoff, in session order, never overlapping
- first number to last number no more than {WINDOW_S:.0f} seconds apart
- the SETUP beat opens on the person's own line, not on Cyclops
- no beat under {MIN_SHOT_S:.1f} seconds; the TURN holds at least {TURN_HOLD_S:.0f}
- the three together come to {STORY_LOW_S:.0f}-{STORY_CAP_S:.0f} seconds

Every story you send is then watched cold by somebody who knows nothing about the session and
asked to say what was wanted, what happened and how it ended. If they cannot, it comes back to
you saying which beat is missing. Fix that beat and send it again.

Returning NO stories is a normal answer and costs nobody anything. Most sessions here are not
stories. Say so in one sentence rather than sending a thin one.
"""

director = Agent(
    name="stories-director",
    deps_type=Telling,
    output_type=Direction,
    instructions=DIRECTOR_INSTRUCTIONS,
    retries=1,
)


@director.instructions
def _director_brief(ctx: RunContext[Telling]) -> str:
    """The summary and the timeline, always wanted, so never behind a round trip."""
    return (
        f"The recording is {ctx.deps.seconds:.0f} seconds long.\n\n"
        f"The session:\n{ctx.deps.brief}\n\n"
        f"Its timeline:\n{ctx.deps.timeline}"
    )


@director.tool
async def look_at(ctx: RunContext[Telling], at: float) -> str:
    """Look at one frame of the recording and say what is on the screen there.

    Args:
        at: a position in the recording, in seconds. Use the middle of a beat, never its start -
            a beat is stamped from the instant the picture changed, and a frame grabbed there is
            as likely to catch the screen being left as the one arriving.
    """
    look = await _look(ctx.deps, round(float(at), 2), ctx.usage)
    if look is None:
        return "Could not get a frame at that second."
    return f"{look.saw}: {look.what}"


# ------------------------------------------------------------------ the gate on the way out


@director.output_validator
async def _every_story_must_stand_alone(
    ctx: RunContext[Telling], out: Direction
) -> Direction:
    """No story leaves without a frame on its TURN and a stranger's retelling of the whole thing.

    Wired here rather than as a step afterwards on purpose: a validator that raises
    :class:`ModelRetry` hands the Director the missing beat and lets it move one number, which is
    the cheapest possible repair. One of those per run - :attr:`Telling.repaired` - and after it
    a story that still cannot be told is dropped while the ones that passed are kept.
    """
    kept: list[Told] = []
    broken: list[str] = []
    for told in out.stories:
        why = await _cannot_stand_alone(ctx, told)
        if why:
            broken.append(f"{told.title!r}: {why}")
            continue
        kept.append(told)
        ctx.deps.passed.append((told, ctx.deps.retellings[_key(told)]))
    if broken and not ctx.deps.repaired:
        ctx.deps.repaired = True
        ctx.deps.passed.clear()  # the whole answer is coming back; these arrive again with it
        raise ModelRetry(
            "These do not stand alone yet. You get one repair.\n"
            + "\n".join(broken)
            + "\n\nRepair means MOVING OR WIDENING the beat that is named, in the timeline you "
            "were given. 'No ask' almost always means the person's request is a line or two "
            "earlier than the beat starts - find that line and open the SETUP on it, even if it "
            "is thirty seconds back, as long as the whole story still spans no more than "
            f"{WINDOW_S:.0f} seconds. 'No ending' means the PAYOFF stops too early - push it out "
            "to whatever was said after. If a beat genuinely does not exist in this session, drop "
            "that story and send the others."
        )
    ctx.deps.notes += broken
    return out.model_copy(update={"stories": kept})


async def _cannot_stand_alone(ctx: RunContext[Telling], told: Told) -> str:
    """Why this story cannot be published, or "". Shape first, then a frame, then the Editor."""
    shape = _misshapen(told, ctx.deps.seconds)
    if shape:
        return shape
    look = await _look(ctx.deps, Telling.instant(told.turn), ctx.usage)
    if look is not None and look.saw in NOTHING_TO_SEE:
        return (
            f"the TURN beat holds nothing - at {Telling.instant(told.turn):.1f}s the screen is "
            f"{look.saw}. Move it to where the picture is actually up, or drop the story"
        )
    await _look(ctx.deps, Telling.instant(told.payoff), ctx.usage)
    told_back = await _retell(ctx.deps, told, ctx.usage)
    if not told_back.clear:
        return told_back.missing or "a stranger watching this cold could not say what happened"
    return ""


def _misshapen(told: Told, seconds: float) -> str:
    """What is mechanically wrong with these three beats, or "". Never a judgement."""
    setup, turn, payoff = told.beats()
    if not (setup.start < setup.end <= turn.start < turn.end <= payoff.start < payoff.end):
        return "the beats are out of order or overlap; they must run setup, turn, payoff"
    if payoff.end > seconds + 0.5:
        return f"the payoff ends past the end of the recording, which is {seconds:.0f}s long"
    if payoff.end - setup.start > WINDOW_S:
        return (
            f"this reaches across {payoff.end - setup.start:.0f}s of the session; a story spans "
            f"at most {WINDOW_S:.0f}s"
        )
    if turn.seconds < TURN_HOLD_S - 0.5:
        return (
            f"the TURN beat is only {turn.seconds:.1f}s; it must hold the new picture for at "
            f"least {TURN_HOLD_S:.0f}s"
        )
    return ""


# ------------------------------------------------------------------ running the two small ones


def _key(told: Told) -> str:
    """One story's identity, for the retelling cache. Its beats, which is what the Editor saw."""
    return "|".join(f"{b.start:.2f}-{b.end:.2f}" for b in told.beats())


async def _look(deps: Telling, at: float, usage) -> Look | None:
    """What is on screen at one instant. Cached, so the same second is never paid for twice."""
    if at in deps.looks:
        return deps.looks[at]
    frame = deps.frame(at)
    if not frame:
        return None
    result = await looker.run(
        [
            f"One frame from {at:.1f} seconds into the recording.",
            BinaryContent(data=frame, media_type="image/jpeg"),
        ],
        model=deps.models.looker,
        deps=deps,
        usage=usage,
        usage_limits=RUN_LIMITS,
    )
    deps.looks[at] = result.output
    return result.output


async def _retell(deps: Telling, told: Told, usage) -> Retelling:
    """Watch one story cold and say it back. Cached on the beats, so a repair pays once."""
    key = _key(told)
    if key in deps.retellings:
        return deps.retellings[key]
    result = await editor.run(
        deps.as_watched(told),
        model=deps.models.editor,
        deps=deps,
        usage=usage,
        usage_limits=RUN_LIMITS,
    )
    deps.retellings[key] = result.output
    return result.output


async def find_arcs(deps: Telling, usage) -> Arcs:
    """The Reader's pass: the candidate stories, or none. The second resource gate."""
    result = await reader.run(
        f"The recording is {deps.seconds:.0f} seconds long.\n\nTimeline:\n{deps.timeline}",
        model=deps.models.reader,
        deps=deps,
        usage=usage,
        usage_limits=RUN_LIMITS,
    )
    return result.output


async def direct(deps: Telling, arcs: list[Arc], usage) -> Direction:
    """The Director's pass: which of the candidates are really stories, checked on the way out."""
    result = await director.run(
        "Candidate stories somebody else found in this session:\n\n" + _candidates(arcs),
        model=deps.models.director,
        deps=deps,
        usage=usage,
        usage_limits=RUN_LIMITS,
    )
    return result.output


def _candidates(arcs: list[Arc]) -> str:
    blocks = []
    for n, arc in enumerate(arcs[:MAX_CANDIDATES], start=1):
        beats = "\n".join(
            f"  {name}: {beat.start:.1f}-{beat.end:.1f} - {beat.what}"
            for name, beat in zip(("SETUP", "TURN", "PAYOFF"), arc.beats(), strict=False)
        )
        blocks.append(f"CANDIDATE {n}: {arc.line}\n{beats}")
    return "\n\n".join(blocks)


__all__ = [
    "Beat",
    "Models",
    "RUN_LIMITS",
    "direct",
    "director",
    "editor",
    "find_arcs",
    "looker",
    "reader",
]
