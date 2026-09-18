"""A walkthrough: the steps somebody asked to be taken through, which one is up, and what the
model is told after every move.

Owns the state and the words, and nothing else - the bar is drawn by :mod:`cyclops.overlay` and
the three tools are wired in :mod:`cyclops.agent`. Pure: no I/O, no clock, no session.

The rules for walking somebody through a job live here and not in the system prompt. They are
read in the tool return, at the moment they apply - the waiting rule arrives as the step it is
about goes up - and a session in which nobody asks for a walkthrough reads none of it. So the
note is *composed from the step index*: the first step carries the whole brief, a middle one a
line, and the model reads the most recent of them, which is the one it acts on.

Never persisted. A tutorial belongs to the session that started it: one begun on Tuesday must not
be on the glass on Thursday.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

MIN_STEPS = 2  # one step is an instruction, not a walkthrough
# Ten segments across the ~322 px of glass is ~29 px each with the gaps, and still counts at a
# glance; twenty is mush. A list that long out of a manual is not something anybody can follow
# with their hands full either, so the refusal asks for it grouped rather than for a bigger bar.
MAX_STEPS = 10
MAX_STEP_CHARS = 60  # a label is read on one row of the terminal; the glass elides past ~33

NONE_RUNNING = "No tutorial is running. If they want one, call start_tutorial with the steps."
FINISHED = "That was the last step. Say so in a few words. The tutorial is over."
ENDED = "The tutorial is off their screen. Say so in a few words, then carry on."
# Said with every step, not only the first: "forget it, stop this" comes four steps after the
# brief, and the note the model acts on is the last one it read.
STOP = (
    "The walkthrough stays up until you call end_tutorial. Call it as soon as they want to stop."
)


@dataclass(frozen=True)
class Tutorial:
    """The steps, in order, and the index of the one on the glass."""

    steps: tuple[str, ...]
    index: int = 0

    @property
    def current(self) -> str:
        return self.steps[self.index]

    @property
    def number(self) -> int:
        """The step up, counted from one - how it is said and how it is drawn."""
        return self.index + 1

    @property
    def total(self) -> int:
        return len(self.steps)

    def advanced(self) -> Tutorial | None:
        """The next step up, or None once the last one is done."""
        if self.number >= self.total:
            return None
        return replace(self, index=self.index + 1)


def clean(raw: object) -> list[str]:
    """The steps out of a tool call's ``steps`` argument: strings, collapsed, capped, non-empty.

    Every malformed thing is dropped and nothing raises, in the spirit of the other argument
    readers in :mod:`cyclops.agent`. The count is not capped here - how many there were is what
    the refusal has to say.
    """
    if not isinstance(raw, list):
        return []
    steps = (" ".join(str(item).split())[:MAX_STEP_CHARS] for item in raw if item is not None)
    return [step for step in steps if step]


def refused(count: int) -> str:
    """Why a list of *count* steps was not put up. Nothing is on the glass when this is sent."""
    if count > MAX_STEPS:
        return (
            f"Nothing went up: {count} steps is too many to follow. Group them into "
            f"{MAX_STEPS} or fewer, a few words each, and call start_tutorial again. Do not "
            "mention this to them."
        )
    return (
        "Nothing went up: a walkthrough needs at least two steps. If it is one thing, just say it."
    )


def note(tutorial: Tutorial) -> str:
    """What the model is told when *tutorial*'s current step has just gone up.

    Step one carries the whole brief, because it is the only moment the brief is needed and the
    model has not read it yet. After that a line is enough: the rule it needs is the last one it
    read, and a middle step that repeated the whole brief would be the prompt, three times over.
    """
    up = f"Step {tutorial.number} of {tutorial.total} is on their screen"
    if tutorial.index == 0:
        return (
            f"{up}. Your first words are the step itself: one or two sentences, and only what the "
            "screen does not already say. Then stop talking and wait: they will tell you when it "
            "is done, and only then call advance_tutorial. A question mid-step gets an answer and "
            "the step stays where it is. Do not read the steps out, and do not say how many there "
            f"are unless they ask. {STOP}"
        )
    last = ", the last one" if tutorial.number == tutorial.total else ""
    return f"{up}{last}. Your first words are the step itself; then wait. {STOP}"
