"""What each agent in the story crew is allowed to say back.

Every field here is prose, an enum, a flag or a number of seconds. Nothing in this module is a
path, a filename, a filter argument or a piece of ffmpeg grammar - :mod:`cyclops.cut` owns every
one of those and is the only thing that renders. A model that cannot express a path cannot
choose one.

The three beats are the whole design. A clip is SETUP, TURN, PAYOFF in session order, and the
types below make any other answer unrepresentable rather than merely discouraged.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# What the Looker is allowed to have seen. An enum rather than a sentence because the Director
# and the validator both branch on it, and "mostly black with something in the corner" is not a
# branch anybody can write.
Saw = Literal["photo", "drawing", "panel_text", "eye", "black"]

# The two the TURN beat may not land on: the bare face, and nothing at all.
NOTHING_TO_SEE: frozenset[str] = frozenset({"eye", "black"})


class Beat(BaseModel):
    """One shot of a story: a span of the recording and what happens in it."""

    start: float = Field(ge=0, description="Seconds into the recording.")
    end: float = Field(ge=0, description="Seconds into the recording. After start.")
    what: str = Field(max_length=120, description="What happens here, in one clause.")

    @property
    def seconds(self) -> float:
        return max(0.0, self.end - self.start)


class Arc(BaseModel):
    """A story somebody could tell about this session, with the three beats it is made of."""

    line: str = Field(
        max_length=140,
        description="The story in one sentence: what was wanted, what happened, how it ended.",
    )
    setup: Beat = Field(description="The person asks for something or holds something up.")
    turn: Beat = Field(description="The thing happens: the picture arrives, the number goes up.")
    payoff: Beat = Field(description="How it ended: the reaction, the closing line, the result.")

    def beats(self) -> tuple[Beat, Beat, Beat]:
        return (self.setup, self.turn, self.payoff)


class Arcs(BaseModel):
    """The Reader's whole answer. ``stories`` empty is the ordinary answer and a cheap one."""

    stories: list[Arc] = Field(default_factory=list, max_length=3)
    why_not: str = Field(
        default="",
        max_length=200,
        description="When there are no stories, the one reason. Written into the plan file.",
    )


class Look(BaseModel):
    """What was on the screen at one instant of the recording."""

    saw: Saw
    what: str = Field(max_length=100, description="What is on the screen, in a few words.")


class Retelling(BaseModel):
    """The Editor's blind account of a finished clip. The standalone test, and the only one."""

    asked: str = Field(max_length=160, description="What the person wanted. '' if you cannot say.")
    did: str = Field(max_length=160, description="What Cyclops did about it. '' if you cannot.")
    ended: str = Field(max_length=160, description="How it ended. '' if you cannot say.")
    clear: bool = Field(
        description="True only if all three are there and a stranger would follow this cold."
    )
    missing: str = Field(
        default="",
        max_length=120,
        description="When clear is false, which beat is missing and what it needs.",
    )

    def line(self) -> str:
        """The three beats as one sentence, which is what goes in the plan file."""
        return " ".join(part for part in (self.asked, self.did, self.ended) if part)


class Told(BaseModel):
    """One story the Director decided to keep."""

    line: str = Field(max_length=140, description="The story in one sentence.")
    title: str = Field(
        max_length=70,
        description="What a person would call this out loud. The person's own words are best. "
        "No quotes, no emoji, no 'In this video'.",
    )
    setup: Beat
    turn: Beat
    payoff: Beat

    def beats(self) -> tuple[Beat, Beat, Beat]:
        return (self.setup, self.turn, self.payoff)


class Direction(BaseModel):
    """The Director's whole answer: the stories this session holds, at most two."""

    stories: list[Told] = Field(default_factory=list, max_length=2)
    why: str = Field(
        default="",
        max_length=200,
        description="One sentence. When there are no stories this is why, and it is written "
        "onto the card.",
    )
