"""Turning one recorded session into the stories it holds, or into a reason it holds none.

Two of the three resource gates live here. :func:`cyclops.cut.worth_asking` is the first and is
free - it reads the log and stops the session that was a microphone check. This module owns the
other two, and the order is the whole economy of the thing:

* **Tier 2** is one Reader call over the transcript. It answers with candidate stories or with
  none, and none is the common answer. A session that stops here has cost about a cent.
* **Tier 3** is the Director, and it runs *only* when the Reader came back with a candidate. It
  looks at real frames, has every story it keeps watched cold by the Editor, and is capped by one
  shared budget for the whole session.

Nothing here writes a file and nothing here renders. What comes back is an :class:`Outcome`:
which tier the session stopped at, why, and the stories that survived with the retelling that
proved each one. :mod:`cyclops.cut` shapes them against measured speech and does every byte.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from pydantic_ai.exceptions import (
    AgentRunError,
    ModelHTTPError,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
)
from pydantic_ai.usage import RunUsage

from ..config import Settings
from .deps import Telling
from .models import Retelling, Told

# Which tier a session stopped at, for the journal line and the plan file. The numbers are the
# plan's own, and they are what makes "36 of 87 never reach a model" a thing anybody can count.
GATE, READER, CREW = 1, 2, 3


@dataclass(frozen=True)
class Shot:
    """One beat of a finished story, flattened. What :mod:`cyclops.cut` shapes and renders.

    A plain dataclass rather than the pydantic :class:`~cyclops.stories.models.Beat` it came from,
    deliberately: it is the boundary of this package, and ``cut.py`` is imported by the admin
    server on every page load and has no business loading pydantic-ai to read three floats.
    """

    kind: str  # "setup" | "turn" | "payoff"
    start: float
    end: float
    what: str
    saw: str = ""  # what the Looker found on the screen here. "" when it was never looked at


@dataclass(frozen=True)
class Story:
    """One story that passed: three shots, the line, and the retelling that proved it."""

    title: str
    line: str
    retelling: str
    shots: tuple[Shot, ...]


@dataclass(frozen=True)
class Outcome:
    """What one session came to. ``asked`` is the only field that decides whether a plan lands."""

    tier: int = CREW
    why: str = ""  # why there are no stories. Written into the plan's note, never a failure
    stories: tuple[Story, ...] = ()
    asked: bool = False  # did a model actually answer? "no network" must never read as "boring"
    cost: Decimal = field(default_factory=lambda: Decimal(0))
    took: float = 0.0  # wall clock, seconds


def tell(
    folder: Path,
    records: list[dict],
    timeline: str,
    brief: str,
    seconds: float,
    settings: Settings,
) -> Outcome:
    """What stories this session holds. The synchronous front door; never raises.

    Called from the index service's worker thread, so it owns its event loop the way
    :mod:`cyclops.after` does when it runs the filing sweep.
    """
    import asyncio

    if not settings.api_key or not timeline.strip() or seconds <= 0:
        return Outcome(tier=CREW, asked=False)
    deps = Telling(
        settings=settings,
        folder=folder,
        records=records,
        timeline=timeline,
        brief=brief or "(no summary was written for this session)",
        seconds=seconds,
    )
    try:
        return asyncio.run(_tell(deps))
    except Exception:  # noqa: BLE001 - a clip is never worth taking the index service down
        return Outcome(tier=CREW, asked=False)


async def _tell(deps: Telling) -> Outcome:
    """The two paid tiers, with one budget and one client across both."""
    from .agents import Models, direct, find_arcs

    started = time.monotonic()
    usage = RunUsage()
    deps.models = Models.build(deps.settings)
    try:
        try:
            arcs = await find_arcs(deps, usage)
        except (ModelHTTPError, UnexpectedModelBehavior, AgentRunError, OSError):
            return _outcome(READER, "", (), asked=False, usage=usage, started=started)
        if not arcs.stories:
            return _outcome(
                READER,
                arcs.why_not or "there is no story in that one",
                (),
                asked=True,
                usage=usage,
                started=started,
            )
        try:
            told = await direct(deps, arcs.stories, usage)
            kept, why = list(deps.passed), told.why
        except UsageLimitExceeded:
            # Over budget keeps what already passed: every story on deps.passed has had its frame
            # looked at and its retelling read, so it is finished work and not a half-answer.
            kept, why = list(deps.passed), "ran out of budget partway through"
        except UnexpectedModelBehavior:
            # The Director answered, repeatedly, and could not be made to answer well. That is a
            # result - an empty plan - and not a reason to ask again tomorrow at the same price.
            kept, why = list(deps.passed), "could not make a story out of that one"
        except (ModelHTTPError, AgentRunError, OSError):
            return _outcome(CREW, "", (), asked=False, usage=usage, started=started)
        return _outcome(
            CREW,
            why or "nothing in that one stands alone",
            _flatten(deps, kept),
            asked=True,
            usage=usage,
            started=started,
        )
    finally:
        await deps.models.close()


def _outcome(tier, why, stories, *, asked, usage, started) -> Outcome:
    return Outcome(
        tier=tier,
        why=why if not stories else "",
        stories=tuple(stories),
        asked=asked,
        cost=Decimal(str(usage.cost)) if usage.cost else Decimal(0),
        took=round(time.monotonic() - started, 1),
    )


def _flatten(deps: Telling, passed: list[tuple[Told, Retelling]]) -> tuple[Story, ...]:
    """The stories that passed, with the Looker's verdict folded in beside each beat."""
    from .models import Beat

    def shot(kind: str, beat: Beat) -> Shot:
        look = deps.looked_at(beat)
        return Shot(
            kind=kind,
            start=beat.start,
            end=beat.end,
            what=beat.what,
            saw=f"{look.saw}: {look.what}" if look else "",
        )

    return tuple(
        Story(
            title=told.title,
            line=told.line,
            retelling=retelling.line(),
            shots=tuple(
                shot(kind, beat)
                for kind, beat in zip(("setup", "turn", "payoff"), told.beats(), strict=False)
            ),
        )
        for told, retelling in passed
    )


__all__ = ["CREW", "GATE", "READER", "Outcome", "tell"]
