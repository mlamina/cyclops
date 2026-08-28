"""What the pipeline carries: one session read off the card, and the projects it might belong to.

One ``deps_type`` for all five agents, which is what makes ``deps=ctx.deps`` in a delegating tool
trivial. :class:`Filing` is mutable on purpose - each delegating tool writes its subagent's answer
back into it, and the code that does the writing reads them off afterwards. That is the seam
between "the agents decide" and "the code writes": the orchestrator's own answer is only ever
whether it filed and why.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import Settings
from ..session import LOG_NAME, PHOTOS, read_log, read_summary, transcript_text
from .models import PhotoPicks, Scribed, SessionDigest, Verdict
from .store import Project

if TYPE_CHECKING:  # importing for real would be a cycle - agents.py needs Filing
    from .agents import Models

# The whole conversation, near enough - the same budget slug.py uses, trimmed the same way and
# for the same reason. See cyclops.slug.fit: the middle goes, so both ends survive.
MAX_TRANSCRIPT_CHARS = 20000
MAX_SAID_CHARS = 160
MAX_PREVIOUS_PAGE_CHARS = 6000


@dataclass(frozen=True)
class PhotoLine:
    """One photo, described by everything except the pixels.

    Which is, for choosing between them, more than the pixels: a shot Cyclops took with a focus of
    "the mitre joint" while someone was saying "look at where these two meet, it's not flush" is
    obviously the one worth keeping, and no amount of looking at the JPEG says that.
    """

    file: str  # "photos/16-48-48_cyclops.jpg", as the record spells it
    by: str  # "cyclops" | "you"
    at: float
    focus: str
    said: str

    def line(self) -> str:
        who = "Cyclops took it" if self.by == "cyclops" else "you pressed the shutter"
        bits = [self.file, who]
        if self.focus:
            bits.append(f"asked to look at: {self.focus}")
        if self.said:
            bits.append(f'said around then: "{self.said}"')
        return " | ".join(bits)


@dataclass(frozen=True)
class SessionFolder:
    """A finished session, read off the card. Everything here came out of a file."""

    path: Path
    uuid: str
    started: str  # ISO, from the "session" record
    date: str  # YYYY-MM-DD, off the folder name
    span: str
    title: str  # summary.md's "# " line
    paragraph: str  # summary.md's body
    transcript: str
    photos: tuple[PhotoLine, ...] = ()
    tracked: tuple[str, ...] = ()  # projects this session started tracking, by name
    opened: tuple[str, ...] = ()  # projects it opened with the open_project tool

    def brief(self) -> str:
        """What the orchestrator is shown first: the summary a small model already wrote."""
        lines = [f"Session of {self.date}, {self.span}."]
        if self.title:
            lines.append(self.title)
        if self.paragraph:
            lines.append(self.paragraph)
        if not self.title and not self.paragraph:
            lines.append("(no summary was written for this session)")
        return "\n".join(lines)

    def photo_lines(self) -> str:
        return "\n".join(photo.line() for photo in self.photos)


@dataclass
class Filing:
    """One session being filed, and everywhere the agents leave their answers."""

    settings: Settings
    session: SessionFolder
    projects: dict[str, Project]  # key -> project. The only way a model can name one.
    models: Models | None = None  # the five bound models; set by the pipeline before any run
    digest: SessionDigest | None = None
    verdict: Verdict | None = None
    scribed: Scribed | None = None
    picks: PhotoPicks | None = None
    notes: list[str] = field(default_factory=list)  # console breadcrumbs, never sent to a model

    def index(self) -> str:
        """The projects, one line each, frontmatter only. Twenty is about three kilobytes."""
        if not self.projects:
            return "(nothing is being tracked yet - this card has no projects)"
        ordered = sorted(self.projects.values(), key=lambda p: p.updated, reverse=True)
        return "\n".join(project.line() for project in ordered)


# ------------------------------------------------------------------ reading one off the card


def _said_near(records: list[dict], at: float) -> str:
    """The last thing anybody said before a photo was taken."""
    spoken = [r for r in records if r.get("type") in {"you", "cyclops"} and (r.get("t") or 0) <= at]
    if not spoken:
        return ""
    text = str(spoken[-1].get("text", "")).strip()
    return " ".join(text.split())[:MAX_SAID_CHARS]


def read_session(folder: Path) -> SessionFolder:
    """Everything the pipeline needs about one finished session, in one pass over its files."""
    from ..slug import fit  # local, so this module stays importable wherever slug is

    records, _ = read_log(folder / LOG_NAME)
    head = next((r for r in records if r.get("type") == "session"), {})
    tail = next((r for r in reversed(records) if r.get("type") == "end"), {})
    title, paragraph = read_summary(folder)

    photos = tuple(
        PhotoLine(
            file=str(r.get("file", "")),
            by=str(r.get("by", "")),
            at=float(r.get("t") or 0.0),
            focus=" ".join(str(r.get("focus", "")).split())[:MAX_SAID_CHARS],
            said=_said_near(records, float(r.get("t") or 0.0)),
        )
        for r in records
        if r.get("type") == "photo" and r.get("file")
    )
    tracked = tuple(
        str(r.get("name", ""))
        for r in records
        if r.get("type") == "project" and r.get("action") == "tracked" and r.get("name")
    )
    opened = tuple(
        dict.fromkeys(  # in order, without repeats: one project opened three times is one project
            str(r.get("name", ""))
            for r in records
            if r.get("type") == "project" and r.get("action") == "opened" and r.get("name")
        )
    )

    seconds = tail.get("seconds")
    return SessionFolder(
        path=folder,
        uuid=str(head.get("uuid", "")),
        started=str(head.get("started", "")),
        date=folder.name[:10],
        span=_span(seconds),
        title=title,
        paragraph=paragraph,
        transcript=fit(transcript_text(records), MAX_TRANSCRIPT_CHARS),
        photos=photos,
        tracked=tracked,
        opened=opened,
    )


def _span(seconds: object) -> str:
    """How long it ran, in the words session.py already uses for the same number."""
    if not isinstance(seconds, int | float) or seconds <= 0:
        return "unknown length"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m {secs:02d}s"


def has_photos(folder: Path) -> bool:
    return (folder / PHOTOS).is_dir()
