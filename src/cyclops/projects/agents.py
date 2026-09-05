"""The orchestrator and the four subagents it delegates to.

One agent is handed a finished session and a short prompt, and works out the rest by calling the
others. Delegation is Pydantic AI's agent-as-tool pattern: each tool below runs a subagent with
``deps=ctx.deps`` and ``usage=ctx.usage``, so one budget covers the whole filing and a runaway
loop runs out of money rather than out of patience.

Two things this module deliberately cannot do.

**It cannot create a project.** The set of projects is closed before the pipeline starts - they
come into existence only when someone says yes to Cyclops asking, in the conversation itself
(see :func:`cyclops.projects.store.create`). So the hardest judgement in a filing pipeline, "is
this a new thing or the same thing as last month", is never made by a model reading a transcript
afterwards. What is left is a membership question with a known answer set, and the one output
validator here is a set lookup rather than a heuristic.

**It cannot write.** Every tool returns text. The subagents' answers are cached onto the mutable
:class:`~cyclops.projects.deps.Filing`, the orchestrator's own answer is only whether it filed and
why, and :mod:`cyclops.projects.store` does every byte afterwards.

Importing this module pulls in ``pydantic_ai``, which is a second or two on a Pi, so the pipeline
imports it lazily and only after it knows there is a key - exactly as ``session.py`` imports
``slug``, and for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from openai import AsyncOpenAI
from pydantic_ai import Agent, ModelRetry, RunContext, UsageLimits
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

from ..config import Settings
from . import store
from .deps import MAX_PREVIOUS_PAGE_CHARS, Filing
from .models import Belongs, Outcome, PhotoPicks, Scribed, SessionDigest, Verdict

# The smart model orchestrates and matches; the cheap one reads; the middle one writes. Naming a
# project is not in any of their gifts, which is why none of them needs to be cleverer than this.
ORCHESTRATOR_MODEL = "gpt-5.6-terra"
READER_MODEL = "gpt-5.4-nano"
MATCHER_MODEL = "gpt-5.6-terra"
SCRIBE_MODEL = "gpt-5.4-mini"
CURATOR_MODEL = "gpt-5.4-mini"

# Nobody is waiting on any of this - it runs detached, after the session folder is already
# complete and correct - so unlike slug.describe_session these retry. What carries over from that
# module is the contract, not the constant: never raise into the caller, and treat "nothing
# happened" as a first-class outcome.
REQUEST_TIMEOUT_S = 60.0
SDK_RETRIES = 2

# One budget for one filing, and it has to be one: ``usage`` is shared down the delegation chain
# so that the whole filing costs what it costs, and a ``UsageLimits`` is checked against whatever
# usage object it is handed. Give the reader "request_limit=5" and you have not capped the reader
# - you have capped the entire run at five requests, and it trips on whichever agent happens to
# be holding the parcel. So there is one cumulative limit, passed to every agent, and it means
# exactly what it says.
#
# The request limit is a backstop against a loop; ``cost_limit`` is the thing actually holding
# the purse. A filing is normally one request when the session was chatter and six or seven when
# it was not, so forty is very wide - which is the point, because the alternative failure is
# stopping halfway through a session that deserved filing.
RUN_LIMITS = UsageLimits(
    request_limit=40, tool_calls_limit=20, cost_limit=Decimal("0.40")
)


@dataclass(frozen=True)
class Models:
    """One client and five bound models, built once per sweep.

    One client, deliberately: a single connection pool and a single TLS handshake for a whole
    card's worth of filing matters more on a Pi over wifi than it looks.
    """

    client: AsyncOpenAI
    orchestrator: OpenAIResponsesModel
    reader: OpenAIResponsesModel
    matcher: OpenAIResponsesModel
    scribe: OpenAIResponsesModel
    curator: OpenAIResponsesModel

    @classmethod
    def build(cls, settings: Settings) -> Models:
        client = AsyncOpenAI(
            api_key=settings.api_key, timeout=REQUEST_TIMEOUT_S, max_retries=SDK_RETRIES
        )
        provider = OpenAIProvider(openai_client=client)
        return cls(
            client=client,
            orchestrator=OpenAIResponsesModel(ORCHESTRATOR_MODEL, provider=provider),
            reader=OpenAIResponsesModel(READER_MODEL, provider=provider),
            matcher=OpenAIResponsesModel(MATCHER_MODEL, provider=provider),
            scribe=OpenAIResponsesModel(SCRIBE_MODEL, provider=provider),
            curator=OpenAIResponsesModel(CURATOR_MODEL, provider=provider),
        )

    async def close(self) -> None:
        try:
            await self.client.close()
        except Exception:  # noqa: BLE001 - tearing down is never worth an exception
            pass


# ------------------------------------------------------------------ the reader

READER_INSTRUCTIONS = """\
Below is a transcript between someone working on something practical and Cyclops, the assistant
helping them. Turn it into a record two other programs will read: one that decides which project
folder this belongs in, and one that writes that folder's page.

Extract, do not interpret. Keep the numbers - sizes, torques, part numbers, model codes, prices,
quantities, temperatures. They are the entire point of the record and prose without them is
worthless here.

`subject` is the thing, named the way its owner names it, never a category. "Lego Millennium
Falcon 75192", not "a lego set". "The kitchen tap", not "plumbing".

`terms` is what makes this findable in three weeks. Put the sloppy names in as well as the proper
one: if they said "the falcon", "the lego" and "75192", all three belong there.

`worked` is false for a conversation where nothing was actually worked on - a question answered,
a chat, a plan with no doing in it. Be strict; false is a normal answer.
"""

reader = Agent(
    name="projects-reader",
    deps_type=Filing,
    output_type=SessionDigest,
    instructions=READER_INSTRUCTIONS,
    retries=2,
)


# ------------------------------------------------------------------ the matcher

MATCHER_INSTRUCTIONS = """\
You are filing one piece of work onto a shelf of projects that already exist. You cannot create a
project and you are not being asked whether one should exist - that was settled by the person
themselves, out loud, at the bench. Your only question is whether this session was work on one of
the projects listed below, and if so which.

Each project is listed with a sentence saying what it is. Read those before the names: a name is
a label someone picked once, and "Cyclops" or "the falcon" tells you nothing on its own. The
sentence is what tells you whether today's work belongs there.

Prefer matching. Names drift badly: "the falcon", "the lego" and "75192" are one project; the
same bike is "the bike", "the Ribble" and "the rear brake". Before concluding nothing matches,
use find_projects on the subject and on two or three of the terms, and use project_history on
anything that comes back even close.

Say no project when the work genuinely is not any of these - including when it is real work on
something nobody has started tracking yet. That is a normal, correct answer, and it costs
nothing: the next session where they say "yes, track this" will pick the thread up.

Never stretch a session onto a project because it is the only one on the shelf.
"""

matcher = Agent(
    name="projects-matcher",
    deps_type=Filing,
    output_type=Verdict,
    instructions=MATCHER_INSTRUCTIONS,
    retries=3,
)


@matcher.instructions
def _matcher_index(ctx: RunContext[Filing]) -> str:
    return f"The projects being tracked:\n{ctx.deps.index()}"


@matcher.tool
async def find_projects(ctx: RunContext[Filing], query: str) -> str:
    """Search the tracked projects by name, nickname, part number or subject.

    Args:
        query: a thing or a nickname - "falcon", "rear brake", "75192".
    """
    hits = store.search(list(ctx.deps.projects.values()), query[:120], limit=5)
    return "\n".join(hits)[:800] if hits else f"Nothing matches {query[:60]!r}."


@matcher.tool
async def project_history(ctx: RunContext[Filing], key: str) -> str:
    """Where one project stands, and the last few things done to it.

    Args:
        key: one of the keys from the list of projects you were given.
    """
    project = ctx.deps.projects.get(key.strip().lower())
    if project is None:
        known = ", ".join(sorted(ctx.deps.projects)) or "none"
        return f"There is no project {key!r}. The ones being tracked are: {known}"
    return store.history(project, entries=5)[:1200]


@matcher.output_validator
async def _key_must_exist(ctx: RunContext[Filing], out: Verdict) -> Verdict:
    """A key a model invented must never reach the filesystem. It becomes a retry instead.

    The whole reason this is a set lookup and not a similarity score is that nothing downstream
    can create a folder: the answer is either one of these keys or it is "no project".
    """
    if isinstance(out, Belongs) and out.key not in ctx.deps.projects:
        known = ", ".join(sorted(ctx.deps.projects)) or "none"
        raise ModelRetry(
            f"There is no project with the key {out.key!r}. The keys that exist are: {known}. "
            "Use one of those exactly, or say this belongs to no project."
        )
    return out


# ------------------------------------------------------------------ the scribe

SCRIBE_INSTRUCTIONS = """\
You keep a folder on a memory card that someone reads to remember what they are doing. Two things
come out of you at once and they must agree with each other, because they sit six inches apart in
the same folder.

THE LOG ENTRY is what happened in this one session, past tense, and it is never rewritten. Three
to six sentences: what was worked on, what was decided or measured, what was tried, what was left
hanging. Keep every number. Do not say "the session", "the user" or "we discussed" - say what was
done.

THE PAGE is the whole project as it stands today, and it replaces the page that is there now.
Someone who reads only this should know where the thing is without reading twelve dated entries.
- tagline: one dense sentence saying what this project IS. This is the hardest-working line in
  the whole folder: it is what a later session is shown when deciding whether new work belongs
  here, so it has to distinguish this project from every other one on the shelf. Name the
  specifics - what the thing is, what it is built from or on, and what it is for. "A Raspberry
  Pi voice assistant with a camera that helps with workshop projects and writes up each session"
  works. "A Pi project", "the build", or "an ongoing effort" are useless and will cause work to
  be filed in the wrong place. Never just restate the project's name.
- where_it_stands: one paragraph, PRESENT tense, the state of the thing right now.
- decided: what has been settled, and why. Cumulative.
- still_open: what is unresolved. The old page's list, minus what got done, plus what is new.
- details: the numbers worth keeping - part numbers, sizes, torques, paint codes, prices.
  Cumulative. This is the part that is genuinely useful in six months. Numbers and named parts
  only: if nothing was measured or specified, leave it empty. A list of bare nouns ("webcam",
  "screws") is noise on the page and buries the real figures when they do arrive.

Carrying forward matters more than freshness. If the old page knew the wall was brick and this
session never mentioned the wall, the new page still knows the wall is brick. Never drop a fact
just because today was about something else.

Write for the person, not for a machine. Plain sentences, no headings, no bold, no bullets inside
a field - the lists are the bullets.

aliases and keywords are not for the reader: they are how the next session finds this folder. Put
the sloppy spoken names in aliases ("the falcon", "the lego") and the searchable nouns in
keywords. No commas inside an item.
"""

scribe = Agent(
    name="projects-scribe",
    deps_type=Filing,
    output_type=Scribed,
    instructions=SCRIBE_INSTRUCTIONS,
    retries=2,
)


# ------------------------------------------------------------------ the curator

CURATOR_INSTRUCTIONS = """\
Pick at most three photos from this session to keep in the project's folder, and caption them.

You are not looking at the pictures. You are looking at who took each one, what Cyclops was asked
to look at, and what was being said at that moment - which for this purpose tells you more than
the pixels would. A photo Cyclops took with a focus of "the mitre joint" while someone was saying
"look at where these two meet, it's not flush" is the one that matters. A photo taken thirty
seconds in with nothing being said is not.

Prefer the thing itself once, a detail that was being argued about, a problem, or visible
progress. Avoid near-duplicates taken seconds apart, and shots with no conversation around them.

One of these may be a diagram Cyclops drew rather than a photograph anybody took - the line says
so. A diagram was asked for out loud, drawn on purpose and read, so it is nearly always worth one
of the slots, ahead of a bench shot that merely happened. Caption it by what it shows.

Captions are for someone scrolling a folder in a year: "The mitre joint, not flush" beats "a
photo of the model". No dates, no filenames, no "photo of".

Use the filenames exactly as they are given to you. Picking nothing is a fine answer and often
the right one.
"""

curator = Agent(
    name="projects-curator",
    deps_type=Filing,
    output_type=PhotoPicks,
    instructions=CURATOR_INSTRUCTIONS,
    retries=1,
)


# ------------------------------------------------------------------ the orchestrator

ORCHESTRATOR_INSTRUCTIONS = """\
You are filing one finished conversation into the notes someone keeps about the things they are
building. The card those notes live on is browsed by a person who is not technical and will never
tidy it up, so what you write is what they live with.

You cannot start a project and you are never asked to. Projects exist because the person agreed
to one out loud, at the bench. Your job is narrower and has a definite answer: did this session
advance one of the projects being tracked, and if so, write it up.

How to work:
1. You are given the session's own summary. Read it. If it is obviously chatter - a question
   answered, a story, nothing made or fixed - stop and file nothing. Most sessions that get here
   are real work, but some are not, and a wrong entry is permanent where a missing one is not.
2. If the session says it started tracking a project, that is the project. Do not second-guess it
   and do not call match_project.
3. Otherwise call read_session, then match_project.
4. If it belongs to a project, call write_up. Then call choose_photos if the session took any.
5. Answer with whether you filed it and one sentence saying why. That sentence is written onto
   the card in the session's own folder, so write it for the person: "Filed under Kitchen Tap"
   or "A question about tyre pressure, not work on anything being tracked."

Do not call a tool twice. Do not call write_up unless match_project found a project or the
session started tracking one.
"""

orchestrator = Agent(
    name="projects-orchestrator",
    deps_type=Filing,
    output_type=Outcome,
    instructions=ORCHESTRATOR_INSTRUCTIONS,
    retries=2,
)


@orchestrator.instructions
def _orchestrator_context(ctx: RunContext[Filing]) -> str:
    """Everything cheap enough to always hand over: the summary, the shelf, and what it did.

    The index goes in the instructions rather than behind a tool because it is always wanted, and
    a round trip to fetch something you always need is a round trip bought for nothing.
    """
    session = ctx.deps.session
    blocks = [
        f"The session:\n{session.brief()}",
        f"The projects being tracked:\n{ctx.deps.index()}",
    ]
    if session.tracked:
        blocks.append(
            "During this session they agreed to start tracking: "
            + ", ".join(repr(name) for name in session.tracked)
            + ". That is what this session was about."
        )
    if session.opened:
        blocks.append(
            "During this session Cyclops opened the notes for: "
            + ", ".join(repr(name) for name in session.opened)
            + ". Strong evidence, though not proof, that this is the project."
        )
    if session.photos:
        blocks.append(f"{len(session.photos)} photo(s) were taken.")
    return "\n\n".join(blocks)


@orchestrator.tool
async def read_session(ctx: RunContext[Filing]) -> str:
    """Read the whole transcript of this session, not just the summary you were given."""
    if ctx.deps.digest is None:
        result = await reader.run(
            f"Transcript:\n{ctx.deps.session.transcript}",
            model=ctx.deps.models.reader,
            deps=ctx.deps,
            usage=ctx.usage,
            usage_limits=RUN_LIMITS,
        )
        ctx.deps.digest = result.output
    return ctx.deps.digest.model_dump_json()


@orchestrator.tool
async def match_project(ctx: RunContext[Filing]) -> str:
    """Decide which tracked project this session belongs to, if any."""
    if ctx.deps.digest is None:
        await read_session(ctx)
    result = await matcher.run(
        f"This session:\n{ctx.deps.digest.model_dump_json()}",
        model=ctx.deps.models.matcher,
        deps=ctx.deps,
        usage=ctx.usage,
        usage_limits=RUN_LIMITS,
    )
    ctx.deps.verdict = result.output
    if isinstance(result.output, Belongs):
        project = ctx.deps.projects[result.output.key]
        return f"Belongs to {project.name!r} ({result.output.confidence}): {result.output.evidence}"
    return f"No project: {result.output.why}"


@orchestrator.tool
async def write_up(ctx: RunContext[Filing], key: str) -> str:
    """Write this session's log entry and the project's new page.

    Args:
        key: the key of the project to file under, from the list you were given.
    """
    project = ctx.deps.projects.get(key.strip().lower())
    if project is None:
        known = ", ".join(sorted(ctx.deps.projects)) or "none"
        return f"There is no project {key!r}. The ones being tracked are: {known}"
    if ctx.deps.digest is None:
        await read_session(ctx)
    previous = store.read_body(project)[:MAX_PREVIOUS_PAGE_CHARS] or "(nothing written yet)"
    result = await scribe.run(
        f"The project is {project.name!r}.\n\n"
        f"Its page as it stands:\n{previous}\n\n"
        f"What happened in this session:\n{ctx.deps.digest.model_dump_json()}",
        model=ctx.deps.models.scribe,
        deps=ctx.deps,
        usage=ctx.usage,
        usage_limits=RUN_LIMITS,
    )
    ctx.deps.scribed = result.output
    ctx.deps.verdict = Belongs(
        key=project.key, evidence="chosen by the orchestrator", confidence="high"
    )
    return f"Written up for {project.name!r}: {result.output.entry.title}"


async def choose_photos_for(deps: Filing, models: Models, usage) -> PhotoPicks:
    """Run the curator over this session's photos. Idempotent - cached onto the filing."""
    if deps.picks is not None:
        return deps.picks
    session = deps.session
    result = await curator.run(
        f"The work: {deps.digest.subject if deps.digest else session.title}\n\n"
        f"The photos:\n{session.photo_lines()}",
        model=models.curator,
        deps=deps,
        usage=usage,
        usage_limits=RUN_LIMITS,
    )
    deps.picks = result.output
    return result.output


@orchestrator.tool
async def choose_photos(ctx: RunContext[Filing]) -> str:
    """Pick which of this session's photos to keep in the project folder, and caption them."""
    if not ctx.deps.session.photos or ctx.deps.settings.project_photos <= 0:
        return "There are no photos to choose from."
    picks = await choose_photos_for(ctx.deps, ctx.deps.models, ctx.usage)
    if not picks.picks:
        return "Kept none of them."
    return "Keeping: " + "; ".join(f"{p.file} ({p.caption})" for p in picks.picks)
