"""What the story crew carries: one recorded session, and everywhere the agents leave answers.

One ``deps_type`` for all four agents, which is what makes ``deps=ctx.deps`` in a delegating tool
trivial - the same shape :mod:`cyclops.projects.deps` has, for the same reason.

Two jobs beyond carrying. It is the **edge**: the one frame the Looker is handed comes out of
``video.mp4`` through :meth:`Telling.frame`, and that is the only subprocess the crew runs. And it
is what keeps the Editor **blind**: :meth:`Telling.as_watched` builds the only view of a story the
Editor is ever shown - the lines that survive the cut and what the Looker saw - which is what
makes its retelling a test of the clip rather than of the session.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import Settings
from .models import Beat, Look, Retelling, Told

if TYPE_CHECKING:  # importing for real would be a cycle - agents.py needs Telling
    from .agents import Models

# One 640-wide JPEG. The Looker is being asked which of five things is on the screen, and that
# survives a heavy re-encode - where a full-size frame costs tokens on every candidate.
FRAME_WIDTH = 640
FRAME_QUALITY = 6
FRAME_TIMEOUT_S = 20.0

# Two decimals is a tenth of a frame at 15 fps: the key that makes "the Director looked here
# already" true without asking whether 12.400001 is the same instant as 12.4.
FRAME_KEY = 2


@dataclass
class Telling:
    """One session being made into stories, and everywhere the agents leave their answers."""

    settings: Settings
    folder: Path
    records: list[dict]
    timeline: str
    brief: str  # the summary a small model already wrote. The Director's brief; never the Editor's
    seconds: float
    models: Models | None = None  # set by the pipeline before any run
    looks: dict[float, Look] = field(default_factory=dict)
    retellings: dict[str, Retelling] = field(default_factory=dict)
    passed: list[tuple[Told, Retelling]] = field(default_factory=list)
    repaired: bool = False  # one ModelRetry per run; after that a broken story is dropped
    notes: list[str] = field(default_factory=list)  # console breadcrumbs, never sent to a model

    # -------------------------------------------------------------- the frame

    def frame(self, at: float) -> bytes:
        """One JPEG out of the recording at ``at`` seconds, or ``b""`` if ffmpeg cannot.

        ``-ss`` before ``-i`` is input seeking, so this decodes from the preceding keyframe and
        stops - a constant cost whatever minute of the session is asked for.
        """
        at = max(0.0, min(at, max(0.0, self.seconds - 0.05)))
        try:
            done = subprocess.run(  # noqa: S603 - the command is ours; `at` is a float
                [
                    "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error",
                    "-ss", f"{at:.3f}", "-i", "video.mp4",
                    "-frames:v", "1", "-vf", f"scale={FRAME_WIDTH}:-2",
                    "-q:v", str(FRAME_QUALITY), "-f", "mjpeg", "-",
                ],
                cwd=self.folder, capture_output=True, timeout=FRAME_TIMEOUT_S, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return b""
        return done.stdout if done.returncode == 0 else b""

    # -------------------------------------------------------------- what the viewer gets

    def said_in(self, beats: tuple[Beat, ...]) -> str:
        """Every line spoken inside these beats, stamped, in order. What survives the cut.

        Only ``you`` and ``cyclops``: the marked lines that tell the timeline what appeared are
        the session's own record of itself, and handing them to the Editor would let it retell a
        picture nobody watching the clip can see.
        """
        lines: list[str] = []
        for record in self.records:
            if record.get("type") not in {"you", "cyclops"}:
                continue
            at = float(record.get("t", 0.0) or 0.0)
            if not any(beat.start <= at < beat.end for beat in beats):
                continue
            who = "THEM" if record.get("type") == "you" else "CYCLOPS"
            text = " ".join(str(record.get("text", "") or "").split())
            if text:
                lines.append(f"[{at:.1f}] {who}: {text}")
        return "\n".join(lines)

    def as_watched(self, told: Told) -> str:
        """One story exactly as a viewer gets it: the shots, what was on screen, what was said.

        The Editor sees this and nothing else. No summary, no transcript, no timeline - which is
        the entire point: if the story cannot be told from here, it cannot be told from the clip.
        """
        blocks = []
        for name, beat in zip(("SHOT 1", "SHOT 2", "SHOT 3"), told.beats(), strict=False):
            look = self.looked_at(beat)
            seen = f" On screen: {look.what} ({look.saw})." if look else ""
            blocks.append(f"{name} ({beat.seconds:.0f}s).{seen}")
        said = self.said_in(told.beats()) or "(nobody speaks in this clip)"
        return "\n".join(blocks) + "\n\nWhat is said, in order:\n" + said

    # -------------------------------------------------------------- the Looker's cache

    @staticmethod
    def instant(beat: Beat) -> float:
        """Where in a beat the Looker is pointed: the middle, where the picture is up.

        Never the start. A beat is stamped from the instant the picture *changed*, and a frame
        grabbed there is as likely to catch the screen it is leaving as the one it is arriving at.
        """
        return round((beat.start + beat.end) / 2.0, FRAME_KEY)

    def looked_at(self, beat: Beat) -> Look | None:
        return self.looks.get(self.instant(beat))
