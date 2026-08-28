"""What each agent in the pipeline is allowed to say back.

Every field here is prose, a flag or a name. Nothing in this module is a path, a date, a count,
a heading or a piece of markdown structure - all of those belong to :mod:`cyclops.projects.store`,
which is the only thing that writes. A model that cannot express a path cannot choose one.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SessionDigest(BaseModel):
    """One session, in the terms a project folder cares about."""

    subject: str = Field(max_length=80, description="The thing itself, named plainly.")
    activity: str = Field(max_length=140, description="What was done to it, in one clause.")
    worked: bool = Field(
        description="True only if practical work actually happened on a real thing. "
        "A question answered, a chat, or a plan with no doing in it is False."
    )
    facts: list[str] = Field(default_factory=list, max_length=8)
    decisions: list[str] = Field(default_factory=list, max_length=5)
    open_threads: list[str] = Field(default_factory=list, max_length=5)
    terms: list[str] = Field(
        default_factory=list,
        max_length=12,
        description="Lowercase nouns, part numbers and tool names - what someone would search "
        "for. Include the sloppy spoken names as well as the proper one.",
    )


class Belongs(BaseModel):
    """This session was work on a project that already has a folder."""

    key: str = Field(description="Exactly one key from the list of projects you were given.")
    evidence: str = Field(max_length=240)
    confidence: Literal["low", "medium", "high"]


class NoProject(BaseModel):
    """No folder should be touched. A normal answer, and the right one more often than not."""

    why: str = Field(max_length=200)


Verdict = Belongs | NoProject


class LogEntry(BaseModel):
    """One dated entry in ``Log.md``. The date, the photos and the marker are added by code."""

    title: str = Field(max_length=70, description="No date. What this session did.")
    body: str = Field(max_length=1200, description="Three to six sentences, past tense.")


class ProjectPage(BaseModel):
    """The prose of ``README.md``. The frontmatter and every heading around it are ours."""

    tagline: str = Field(
        max_length=300,
        description="One dense sentence saying what this project is, naming the specifics that "
        "tell it apart from other projects - what the thing is, what it is built on, what it is "
        "for. It is shown to later sessions deciding where work belongs, so a vague one causes "
        "misfiling. Never merely restate the name.",
    )
    where_it_stands: str = Field(max_length=900)
    still_open: list[str] = Field(default_factory=list, max_length=6)
    decided: list[str] = Field(default_factory=list, max_length=8)
    details: list[str] = Field(default_factory=list, max_length=10)


class Scribed(BaseModel):
    """The scribe's whole answer: one round trip for the entry and the page together.

    Both at once, deliberately, and for the reason :mod:`cyclops.slug` gives for doing the same:
    a log entry that disagreed with the page sitting six inches from it in the same folder would
    be worse than either alone.
    """

    entry: LogEntry
    page: ProjectPage
    aliases: list[str] = Field(
        default_factory=list,
        max_length=6,
        description="Other names this gets called out loud. These are how the next session finds "
        "this folder, so include the sloppy ones. No commas inside an item.",
    )
    keywords: list[str] = Field(default_factory=list, max_length=12)


class PhotoPick(BaseModel):
    file: str = Field(description="Exactly one filename from the list you were offered.")
    caption: str = Field(max_length=100, description="No dates, no filenames, no 'photo of'.")


class PhotoPicks(BaseModel):
    picks: list[PhotoPick] = Field(default_factory=list, max_length=3)


class Outcome(BaseModel):
    """What the orchestrator decided, once it has finished delegating."""

    filed: bool
    why: str = Field(max_length=200, description="One sentence. It is written onto the card.")
