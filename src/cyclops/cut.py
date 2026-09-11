"""Finding the moments of a session worth showing somebody who was not there.

A session leaves ``video.mp4`` - the panel, exactly as it was on the glass, for however long the
conversation ran. That is an archive, not a video: nobody sits through six minutes of somebody
waiting for an answer. **Nobody sits through ninety seconds of it either.** This module used to
cut one long video per session, on a button, with a title card and a description sized for a
YouTube upload; what it makes now is between zero and three clips of about fifteen seconds each,
every one of them a single moment with the pauses taken out.

**Zero is the ordinary answer and the most important one.** Four sessions in five are somebody
checking a torque figure or saying good morning, and a highlights reel is only worth opening if
everything in it is worth watching. So there are two filters before an encode: :func:`worth_asking`
reads the log and declines to spend anything at all on a session that is plainly a test - the card
here held ninety-eight sessions and fifty-five of them never needed a model - and then the model
itself is asked to answer NOTHING, which the prompt spends four paragraphs making easy to say.

**What makes the clips tight is the audio, not the transcript.** ``session.jsonl`` stamps every
turn against the same ``t0`` the video was muxed on, so the model can name a moment in seconds
with no transcription to do. But a turn is not a shot: a logged thirteen-second answer is really
eight bursts of speech with 2.7 seconds of pause inside it, and server VAD overshoots the tail of
every one by half a second to a second and a third. :func:`listen` runs ``silencedetect`` over each
channel - measured at 0.55 s for a six-minute recording, because ``-vn`` means the video packets
are never decoded - and :func:`tighten` trims the model's ranges down onto the spans where somebody
was actually audible. Across this card that is 65% of the running time removed. Filler words are
*not* removed and cannot be: the transcription model normalises them out, so across 220 real turns
the whole corpus holds one "uh" and no "um" at all. What a viewer perceives as the fillers going is
the holes around them closing.

The queue is the filesystem, as everywhere else here::

    clips/plan.json   present means this session has been considered. That is the whole gate
    clips/1.mp4       one clip. Existing means one ffmpeg run returned zero

**Plan-present-means-considered is the entire retry story.** ``deploy/push.sh`` restarts the index
service on every deploy and the default ``KillMode`` takes the whole cgroup with it, so a render
*will* be killed halfway - and when it is, the plan is still on the card, so the next sweep re-runs
ffmpeg for the clip that has no file yet and never pays for the model twice. Each clip is its own
retryable unit; a session with three of them is three encodes that can be interrupted independently.

It also makes a bad cut fixable by hand: edit the ranges in ``clips/plan.json``, delete that clip's
``.mp4``, and the next sweep renders what you wrote. :func:`sanitize` runs over a hand-edited plan
exactly as it runs over the model's answer, so that is safe to do.

Unlike the design this replaced, there is no rule-based fallback. A cut chosen by rules was a
defensible ninety-second archive; it is not a moment, and a highlights reel padded with material
nothing chose is worse than a short one. With no key the plan is simply not written, the session
stays pending, and the next sweep with a key picks up the whole backlog - "nobody looked" and
"somebody looked and there was nothing" are deliberately different files.

Stdlib, :mod:`cyclops.card`, :mod:`cyclops.library` and :mod:`cyclops.stats` - all three of which
are themselves stdlib-only. ``openai`` is imported inside :func:`decide` rather than at the top,
the move :mod:`cyclops.after` and :mod:`cyclops.captions` already make: ``cyclops.admin.views``
imports this module to ask what state a session's clips are in, and loading an SDK to answer that
would put a second and a half into a request handler.
"""

from __future__ import annotations

import fcntl
import json
import math
import re
import subprocess
import time
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from . import card, library, stats
from .config import CUT_LOCK, Settings

# Not -nano, and this is the only call in the codebase that is not. Deciding which fifteen seconds
# of a six-minute conversation somebody would want to watch is a judgement about the whole
# transcript rather than a lookup, and it is the difference between a moment and a montage.
CUT_MODEL = "gpt-5.4-mini"
# Generous, because nobody is standing in front of it: this runs niced inside the index service.
# max_retries=0 still applies, for slug.describe_session's reason.
CUT_TIMEOUT_S = 60.0
# Five times the worst measured for a clip of this size. Wide enough that a slow encode is never
# mistaken for a wedged one, tight enough that a wedged one is not forever.
RENDER_TIMEOUT_S = 240.0

EDGE_FADE_S = 0.02  # 20 ms across every join, because cutting mid-waveform clicks

MAX_CLIPS = 3       # a bench session does not hold four memorable moments
MIN_CLIP_S = 5.0    # after tightening. Deliberately BELOW the prompt's floor: a clip the model
                    # sized at twelve seconds loses a fifth of itself to the pauses, and a clip
                    # that evaporates between the prompt and sanitize() is a queue that never drains
MAX_CLIP_S = 25.0   # on a muted autoplaying reel the viewer has already moved on
TARGET_CLIP_S = 15.0
WINDOW_S = 60.0     # first number to last number, in the source. This is what "one specific
                    # moment" means as an enforceable rule, and it bounds the decode as well

MIN_KEEP_S = 0.6    # "Yeah, that's it" is a legitimate shot in a reel
MERGE_GAP_S = 0.25  # bigger than this and sanitize re-joins the pauses tighten just removed
LEAD_IN_S = 0.15    # the boundary is measured audio now, not a model's guess at one
LEAD_OUT_S = 0.25
MAX_RANGES = 24     # per clip. Tightening turns two model ranges into six to twenty pieces
FPS = 15.0          # what the panel records at; boundaries are snapped to it

SNAP_TO_TURN_S = 1.5  # a boundary this close to a logged turn was meant to be that turn

# Silence detection - see listen(). Two floors because they are two instruments: the mic is an
# open mic in a workshop, the right channel is the speaker's own zero-filled output.
MIC_FLOOR_DB = -38
CYC_FLOOR_DB = -50
SILENCE_MIN_S = 0.20
SILENCE_TIMEOUT_S = 120.0
BRIDGE_S = 0.30   # a hole this small between two audible spans is a breath, not a pause
FLOOR_S = 0.50    # what is left of a range after trimming, below which it was never a shot

# What a session has to have before the model is asked about it at all - see worth_asking. These
# are the cheapest filter there is and they run off the log, so a test session costs one read.
WORTH_YOU = 1       # turns from the person. One is the archetype here, not a mic check: "make
                    # this photo a cartoon" is one request, one picture and a whole clip. Two cost
                    # six of the best moments on this card and saved six model calls.
WORTH_CYCLOPS = 2
WORTH_SECONDS = 30.0
WORTH_CHARS = 300   # of transcript, both voices. Kills a mic storm: thirty turns in four seconds

MAX_TIMELINE_CHARS = 24000
MAX_LINE_CHARS = 200  # a timeline line is trimmed rather than the timeline being elided
MAX_TITLE_CHARS = 70

# The panel's own resolution, and the general scale/pad form rather than the literal numbers
# because CYCLOPS_RECORD_SOURCE=camera makes the recording the sensor's shape and not 5:3.
OUT_W, OUT_H = 800, 480

# What this module calls its rows in the background ledger. Written once so that _drop_stale_rows
# below can recognise its own work and nobody else's.
CHOOSING = "Choosing what to keep of"
CUTTING = "Cutting the video of"

_RANGE_LINE = re.compile(r"^\s*[-*]?\s*(\d+(?:\.\d+)?)\s*[-–—]\s*(\d+(?:\.\d+)?)\s*$")
_CLIP_LINE = re.compile(r"^\s*(?:[-*]\s*)?CLIP\b\s*:?\s*$", re.IGNORECASE)
_SIL = re.compile(r"silence_(start|end):\s*(-?\d+(?:\.\d+)?)")


# ------------------------------------------------------------------ the plan


@dataclass(frozen=True)
class Clip:
    """One moment, and the pieces of the recording it is made of."""

    title: str = ""
    ranges: tuple[tuple[float, float], ...] = ()
    why: str = ""  # why this one will never be rendered. Empty means it is still owed.

    @property
    def seconds(self) -> float:
        return sum(end - start for start, end in self.ranges)


@dataclass(frozen=True)
class Plan:
    """What was decided for one session. The whole of ``clips/plan.json``.

    Present means the session has been considered, which is what makes it the queue's marker;
    ``clips: []`` is the ordinary answer and takes a session off the queue for good.
    """

    decided: str = ""             # ISO-8601, when it was looked at
    by: str = "model"             # "model" | "gate" - which of the two filters answered
    seconds: float = 0.0          # the source recording's length, from ffprobe
    why: str = ""                 # why nothing could be decided at all. A failure, and shown
    note: str = ""                # why nothing was looked for. Not a failure - see worth_asking
    speech: tuple[tuple[float, float], ...] = ()   # measured once; a retry never re-measures
    clips: tuple[Clip, ...] = ()
    source: dict[str, Any] = field(default_factory=dict)  # (size, mtime_ns) of video.mp4


def plan_path(folder: Path) -> Path:
    return folder / card.CLIPS / card.CLIP_PLAN


def clip_path(folder: Path, n: int) -> Path:
    return folder / card.CLIPS / f"{n}.mp4"


def read_plan(folder: Path) -> Plan | None:
    """The plan on the card, or None if there is not one worth believing.

    Bad JSON reads as no plan rather than as an error: the only thing that can produce one is a
    hand edit, and the honest response to a file somebody broke is to decide again.
    """
    try:
        raw = json.loads(plan_path(folder).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None

    def pairs(got: Any) -> tuple[tuple[float, float], ...]:
        rows = got if isinstance(got, list | tuple) else ()
        out = []
        for one in rows:
            if isinstance(one, list | tuple) and len(one) == 2:
                with suppress(TypeError, ValueError):
                    out.append((float(one[0]), float(one[1])))
        return tuple(out)

    clips = []
    for one in raw.get("clips", []) if isinstance(raw.get("clips"), list) else []:
        if isinstance(one, dict):
            clips.append(
                Clip(
                    title=str(one.get("title", "")),
                    ranges=pairs(one.get("ranges")),
                    why=str(one.get("why", "")),
                )
            )
    return Plan(
        decided=str(raw.get("decided", "")),
        by=str(raw.get("by", "")),
        seconds=float(raw.get("seconds", 0.0) or 0.0),
        why=str(raw.get("why", "")),
        note=str(raw.get("note", "")),
        speech=pairs(raw.get("speech")),
        clips=tuple(clips),
        source=raw.get("source") if isinstance(raw.get("source"), dict) else {},
    )


def write_plan(folder: Path, plan: Plan) -> None:
    """Land the plan, whole or not at all - card.write_text, like every byte here."""
    with suppress(OSError):
        (folder / card.CLIPS).mkdir(parents=True, exist_ok=True)
    body = asdict(plan) | {
        "speech": [list(s) for s in plan.speech],
        "clips": [asdict(c) | {"ranges": [list(r) for r in c.ranges]} for c in plan.clips],
    }
    with suppress(OSError, ValueError):
        card.write_text(plan_path(folder), json.dumps(body, indent=2, ensure_ascii=False))


# ------------------------------------------------------------------ state, and the queue


@dataclass(frozen=True)
class Progress:
    """One session's clip situation, in numbers the page can show."""

    state: str = ""      # "" | "asked" | "clipping" | "done" | "none" | "failed"
    made: int = 0
    total: int = 0
    failed: int = 0


def progress(folder: Path) -> Progress:
    """What has happened to this session's clips, from the folder and never from a ledger."""
    plan = read_plan(folder)
    if plan is None:
        return Progress("asked" if card.written(folder / card.VIDEO) else "")
    if plan.why:
        return Progress("failed")
    total = len(plan.clips)
    if not total:
        return Progress("none")
    made = sum(1 for n in range(1, total + 1) if card.written(clip_path(folder, n)))
    failed = sum(1 for one in plan.clips if one.why)
    # The scratch file is ffmpeg's, and it exists only while ffmpeg is writing into it. A SIGKILL
    # leaves one behind, which session.recover sweeps and the next attempt overwrites - so this is
    # allowed to be briefly wrong after a power cut and is never wrong for long.
    if any(card.tmp_for(clip_path(folder, n)).exists() for n in range(1, total + 1)):
        return Progress("clipping", made, total, failed)
    if made + failed >= total:
        return Progress("done" if made else "failed", made, total, failed)
    return Progress("asked", made, total, failed)


def state(folder: Path) -> str:
    """One word, because most call sites want one. See :class:`Progress` for the counts."""
    return progress(folder).state


@dataclass(frozen=True)
class Work:
    """The single next unit. ``index`` -1 means decide; otherwise render that clip."""

    folder: Path
    index: int = -1


def work(sessions_dir: Path) -> Work | None:
    """The next unit of work, newest session first, or None.

    Newest first so this afternoon's session jumps the backfill. Rendering outranks deciding
    within one folder, and a folder that is fully rendered is skipped in one read.

    **There is no separate rate limit on the backfill, and that is deliberate.** One unit per
    sweep is the rate limit: a card of ninety-eight sessions drains newest-first, one decide or
    one encode at a time, each one behind :func:`busy`'s heat and live-conversation gates and
    behind :func:`worth_asking`, which on this card answered "no" for fifty-five of them without
    spending anything. A cap on top of that would only decide which sessions are never looked at,
    which is not a decision worth making automatically.

    Every path that declines to render a clip writes a ``why`` into that clip (see
    :func:`_render_one`), so a unit can never sit at the head of this walk forever. That is not an
    assertion about ffmpeg - it is enforced by the two callers below.
    """
    try:
        folders = library._folders(sessions_dir)  # noqa: SLF001 - newest first, already sorted
    except OSError:
        return None
    for folder in folders:
        # A recording and a finished summary. The summary is the "this session is over and the
        # index service has been through it" marker, which keeps this off a folder still settling.
        if not card.written(folder / card.VIDEO) or not card.written(folder / card.PAGE_NAME):
            continue
        if card.locked(folder):
            continue
        plan = read_plan(folder)
        if plan is None:
            if plan_path(folder).exists():
                # A plan somebody broke by hand. Replaced with an empty one rather than decided
                # again, which would pay the model on every bell for as long as the file is bad.
                write_plan(folder, Plan(why="the plan on the card could not be read"))
                continue
            return Work(folder)
        if plan.why:
            continue
        for n, clip in enumerate(plan.clips, start=1):
            if clip.why or card.written(clip_path(folder, n)):
                continue
            return Work(folder, n)
    return None


def forget(folder: Path) -> None:
    """Throw away what was decided for this session, so the next sweep decides again.

    What the Find clips button does. Deletes rather than versions, which is what "look again"
    honestly means and what keeps :func:`progress` a handful of stats instead of an mtime
    comparison. It also removes what the one-video-per-session design left behind, because a
    session being reconsidered is the moment its old cut stops meaning anything.
    """
    clips = folder / card.CLIPS
    if clips.is_dir():
        for made in clips.iterdir():
            made.unlink(missing_ok=True)
    for old in (card.CUT, card.CUT_PLAN, card.CUT_SUBS, card.CUT_REQUEST):
        (folder / old).unlink(missing_ok=True)


@contextmanager
def _held():
    """One render at a time on this box, or nothing at all.

    A second lock rather than RECALL_LOCK, because they answer different questions - see
    ``config.CUT_LOCK``. Non-blocking: the right answer to "somebody else is rendering" is to
    come back on the next sweep, not to queue a second ffmpeg behind the first.
    """
    CUT_LOCK.parent.mkdir(parents=True, exist_ok=True)
    handle = CUT_LOCK.open("w")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        yield True
    finally:
        handle.close()  # the kernel drops the lock with the fd, however this process ends


# ------------------------------------------------------------------ is this one worth anything


def worth_asking(records: list[dict]) -> str:
    """Why this session is not worth spending anything on, or "" when it might be.

    **The cheapest filter there is, and the only one that runs before a subprocess.** The card
    this was written against held ninety-eight sessions; fifty-five of them are a button pressed
    to check the microphone, a session that ended before anybody spoke, or a mic storm - twenty-six
    have no turns in them at all. Asking a model about those costs a call each and gets NOTHING
    back every time, and running ffmpeg over them costs more than the call does.

    So this reads the log and nothing else: no ffprobe, no silencedetect, no network. A session
    that fails here gets an empty plan with the reason in it, which takes it off the queue for
    good and is exactly as good an answer as the model would have given.

    The thresholds are measured rather than chosen. ``WORTH_SECONDS`` is what rejects the session
    with thirty "you" turns inside four seconds, which is a fan and not a conversation.
    """
    you = [r for r in records if r.get("type") == "you"]
    cyclops = [r for r in records if r.get("type") == "cyclops"]
    if len(you) < WORTH_YOU or len(cyclops) < WORTH_CYCLOPS:
        return "not enough was said in that one"
    spoken = max((float(r.get("t", 0.0) or 0.0) for r in records), default=0.0)
    tail = next((r for r in reversed(records) if r.get("type") == "end"), {})
    said = tail.get("seconds")
    if isinstance(said, int | float) and said > 0:
        spoken = max(spoken, float(said))
    if spoken < WORTH_SECONDS:
        return "that one was over too quickly to hold a moment"
    chars = sum(len(str(r.get("text", "") or "")) for r in (*you, *cyclops))
    if chars < WORTH_CHARS:
        return "there is barely any conversation in that one"
    return ""


# ------------------------------------------------------------------ how long the recording is


def probe(folder: Path, records: list[dict]) -> tuple[float, bool]:
    """How long the recording runs, and whether it has any picture in it at all.

    The length comes from ffprobe rather than the log's ``end`` record, because the two disagree:
    ``-shortest`` trims the mux to whichever stream ran out first, so a session whose last seconds
    were audio-only has a video shorter than the conversation was. Every range is clamped against
    this, so it has to be the file's number and not the transcript's. The log is the fallback for
    a box with no ffprobe, because being slightly wrong beats refusing.

    The second half of the answer is here because three sessions on the card turned out to have a
    ``video.mp4`` holding **only sound** - a recording where the camera never produced a frame
    still muxes, and still leaves a file of several megabytes. Pointed at one of those, ffmpeg
    fails on ``[0:v]`` with half a page of filtergraph and the words "matches no streams", which
    is a true thing to write into the plan and a useless thing to show somebody who pressed a
    button. So it is asked here instead, once, and answered in a sentence.
    """
    seconds, has_video = 0.0, True
    try:
        done = subprocess.run(  # noqa: S603 - the command is ours
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration:stream=codec_type",
                "-of", "default=noprint_wrappers=1:nokey=1", str(folder / card.VIDEO),
            ],
            capture_output=True, timeout=30.0, check=False,
        )
        if done.returncode == 0:
            out = done.stdout.decode(errors="replace").split()
            has_video = "video" in out
            for word in out:
                try:
                    found = float(word)
                except ValueError:
                    continue
                if math.isfinite(found) and found > 0:
                    seconds = found
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    if seconds:
        return seconds, has_video
    tail = next((r for r in reversed(records) if r.get("type") == "end"), {})
    said = tail.get("seconds")
    if isinstance(said, int | float) and said > 0:
        return float(said), has_video
    return float(max((r.get("t", 0.0) or 0.0 for r in records), default=0.0)), has_video


# ------------------------------------------------------------------ where the speech actually is


def silence_command(channel: int) -> list[str]:
    """One channel of the recording, as ffmpeg's list of the holes in it. ``cwd`` is the folder.

    ``-v info``, not the render's ``-loglevel error``: silencedetect reports on stderr at info
    level, and the quiet setting would throw the answer away.
    """
    floor = MIC_FLOOR_DB if channel == 0 else CYC_FLOOR_DB
    return [
        "ffmpeg", "-hide_banner", "-nostdin", "-v", "info",
        "-i", card.VIDEO, "-vn",
        "-af", f"pan=mono|c0=c{channel},silencedetect=n={floor}dB:d={SILENCE_MIN_S}",
        "-f", "null", "-",
    ]


def listen(folder: Path, seconds: float) -> tuple[tuple[float, float], ...]:
    """When anybody was audible, from both channels. ``()`` when ffmpeg cannot say.

    Two passes rather than one over a downmix, because they are two instruments: the left channel
    is an open mic in a workshop and the right is the speaker's own zero-filled output, which is
    exactly zero between utterances. One threshold would apply the mic's floor to a channel that
    needs none and lose the quiet ends of Cyclops's sentences.

    Cheap enough not to need its own politeness: measured on the Pi at 62 °C, 0.55 s of wall clock
    for a 362-second recording, ``speed=853x``. ``-vn`` is what does it - the demuxer skips the
    video packets without decoding them, so the whole cost is one AAC stream.
    """
    if seconds <= 0:
        return ()
    spans: list[list[float]] = []
    for channel in (0, 1):
        try:
            done = subprocess.run(  # noqa: S603 - the command is ours
                silence_command(channel), cwd=folder, capture_output=True,
                timeout=SILENCE_TIMEOUT_S, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return ()
        if done.returncode != 0:
            return ()
        spans += [list(s) for s in read_silence(done.stderr.decode(errors="replace"), seconds)]
    return tuple((a, b) for a, b in _bridge(spans, 0.0))


def read_silence(text: str, seconds: float) -> tuple[tuple[float, float], ...]:
    """silencedetect's report, inverted: the spans where something was audible.

    Pure text in and pairs out, which is what lets this be tested without a subprocess.
    """
    holes: list[list[float]] = []
    for kind, value in _SIL.findall(text):
        if kind == "start":
            holes.append([max(0.0, float(value)), seconds])
        elif holes:
            holes[-1][1] = min(seconds, float(value))
    out: list[tuple[float, float]] = []
    at = 0.0
    for a, b in holes:
        if a - at > 0.05:
            out.append((at, a))
        at = max(at, b)
    if seconds - at > 0.05:
        out.append((at, seconds))
    return tuple(out)


def _bridge(pairs: list[list[float]], gap: float) -> list[list[float]]:
    """Spans in, spans out, with anything closer together than ``gap`` joined."""
    merged: list[list[float]] = []
    for start, end in sorted(pairs):
        if merged and start - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


# ------------------------------------------------------------------ what the model is shown


def timeline(records: list[dict], captions: dict[str, str]) -> str:
    """The session as a list of stamped lines, which is what makes the ranges nominable.

    Not ``session.transcript_text``: that drops ``t``, and ``t`` is the entire answer here. And
    deliberately not ``slug.fit`` either - eliding the middle of the conversation would make the
    middle of the session unpickable, which is exactly the half most worth keeping. Over budget,
    every *line* is trimmed instead, so every moment stays reachable.
    """
    lines: list[str] = []
    for record in records:
        kind = record.get("type")
        at = float(record.get("t", 0.0) or 0.0)
        if kind == "you":
            end = at + float(record.get("dur", 0.0) or 0.0)
            lines.append(f"[{at:.1f}-{end:.1f}] YOU: {_flat(record.get('text', ''))}")
        elif kind == "cyclops":
            # Interruption is the strongest signal in the file: it marks the moment somebody
            # decided the answer was going the wrong way, which is either the best thing in the
            # session or the most cuttable, and the model can only tell if it is told.
            cut_off = " (cut off)" if record.get("interrupted") else ""
            lines.append(f"[{at:.1f}] CYCLOPS{cut_off}: {_flat(record.get('text', ''))}")
        elif kind == "photo":
            what = captions.get(str(record.get("file", "")).split("/")[-1], "")
            lines.append(f"[{at:.1f}] [photo] {what or 'the camera was pointed at something'}")
        elif kind == "screen":
            # ``html``, not ``text``: agent.py writes session.note("screen", html=...). Reading
            # the wrong key meant every panel write - the most visual thing that happens in a
            # session, and the only thing besides a photo that changes the picture - reached the
            # model as a bare "[wrote on the panel]" with nothing after it.
            lines.append(f"[{at:.1f}] [wrote on the panel] {_flat(record.get('html', ''))}")
        elif kind == "search":
            lines.append(f"[{at:.1f}] [searched the web] {_flat(record.get('query', ''))}")
        elif kind == "recall":
            found = _flat(record.get("title", ""))
            lines.append(f"[{at:.1f}] [found something on the card] {found}")
    trimmed = [line[:MAX_LINE_CHARS] for line in lines]
    while sum(len(line) + 1 for line in trimmed) > MAX_TIMELINE_CHARS and trimmed:
        trimmed.pop(len(trimmed) // 2)  # from the middle, so both ends of the session survive
    return "\n".join(trimmed)


def _flat(text: Any) -> str:
    return " ".join(str(text or "").split())


CLIP_PROMPT = """\
Below is the timeline of a recorded session between a person making or fixing something at their
workbench and Cyclops, the assistant that sits on the bench beside them. The recording is
{seconds:.0f} seconds long and every time below is a position in it, in seconds from the start.

What the recording looks like: one fixed camera view of the bench with Cyclops's own screen drawn
over it. The camera never moves and never follows anybody. Exactly one thing in that frame ever
changes - the picture on the screen. Apart from that it is a still with two voices over it.

These clips go on a reel that autoplays one after another, often muted, to somebody walking past.
There is only one question about any moment: is it fun or interesting TO LOOK AT.

Read the timeline's marked lines exactly this way. They are the only way you can tell what a
viewer would see.

  [photo] ...              THE PICTURE CHANGED. A new image is on the screen - a snap of the
                           bench, or an edit or a drawing that was just made. This is the good one.
  [wrote on the panel] ... THE PICTURE CHANGED. A number, a diagram or a few words now fill the
                           screen. Also good.
  [found something on the card] ...  An OLD picture dragged back out of storage. However handsome
                           it is, Cyclops did not make it here: it has been on this panel before
                           and the same few stored images come back session after session, so
                           keeping one means the reel shows one picture three times. On its own it
                           is never a clip, and the session it appears in is usually one where
                           Cyclops could not find what was asked for anyway.
  [searched the web] ...   NOTHING HAPPENS ON SCREEN AT ALL. A search runs inside the machine.
                           Never a clip, however interesting the result sounded. What a search
                           finds counts only if it then turns into its own [photo] or
                           [wrote on the panel] line.

WHAT MAKES A MOMENT WORTH WATCHING is not the assistant's answer. It is the person. Somebody
asking for something with an attitude, showing off what they built, introducing a friend to a
machine, getting impatient with it, teasing it, being pleased, being let down, changing their
mind out loud. The trick the machine does is only fun because of the human next to it. A clip is
a person being a person, and the picture landing is what they are being a person AT.

YOUR WORKING, FIRST. Before you decide anything, write one line for every moment in this session
that changed the picture, plus a line for any moment you were tempted to keep for what was said.
One line each, this shape and nothing fancier:

MARK 3 - a cartoon version of the photo appears at 70.7 because he asked for one

Write the mark, then the one thing that happened. The time in a mark line is the instant the
picture changed. It is NOT the clip; the clip gets built later and is far longer.

Mark on this scale.

MARK 3 - something is on the screen that did not exist anywhere before this moment, and it is
there because somebody asked for it. A photograph edited into a different picture: a background
cut away, a person straightened and centred, an object recoloured, a snapshot turned into a
catalogue shot or into a cartoon. A drawing made to order that names real parts and real values.
A figure written across the panel in answer to a question about the work. A stranger sees the
before and the after with the sound off and gets it. This is the commonest and best thing that
happens here.

MARK 2 - the picture changes to a real THING, and somebody is showing it to the camera on
purpose. The machine on the bench, the part in a hand, the tool, the board, the case, the page of
the manual - it arrives on screen and the person says what it is or what they mean to do with it
while Cyclops names it back. One change, one payoff. A snap like this is a 2 even when it is
completely plain: the object is the reveal and nobody had to ask for a transformation. These are
easy to walk past because nobody asked for anything, and they are some of the best moments here.

MARK 1 - the picture changes, but not into anything new. All of these are 1 however good the
thing on screen looks:
- a [found something on the card] line. An old picture out of storage. Always a 1.
- a search that came back with nothing to show, or an answer of the form "I could not find it".
- a photograph that failed, and Cyclops's own next line tells you it failed: it calls the picture
  blurry, or too close, or off angle, or says it cannot see the thing or cannot tell what it is
  looking at, or asks for another one. A 1 however interesting the object was.
- a drawing with nothing real inside it. Judge it by what is written IN it, never by what was
  asked for. Boxes reading INPUT, PROCESS, OUTPUT, CONTROLLER or MODULE are a picture of nothing.
  So is anything drawn only to demonstrate that Cyclops can draw - when the request is to show off
  the panel, or to prove a feature works, or to put up a sample, the picture is a demo of the tool
  rather than a picture of the work, and it is a 1 even when the person says it looks cool. A
  drawing that names actual parts, actual values, or the actual machine on the bench is a 3,
  however casually it was asked for.
- a snap of a person's face, or a selfie, with nothing done to it afterwards. The person is not a
  workpiece and a face on its own is not a reveal. If that same face is later cut out,
  straightened or restyled, the EDIT is a 3 and that is the moment.
- a snap of a VIEW rather than of a thing: out of a window, at a wall, a doorway, a ceiling, an
  empty corner, the room in general, the bench with nothing on it. There is no object in it, so
  there is nothing to look at, whatever was asked about it.
- a second snap of something this session has already put on the screen. The first one was the
  reveal; this one is the camera being checked again.
- a SECOND panel write repeating what this session already drew: the same diagram laid out again
  more simply or more cleanly, a written or text version of a picture already on the panel, a list
  of what that drawing already showed, a summary of it. The first one was the moment and it keeps
  its 3; every repeat after it is a 1 however neat it is.

MARK 0 - the frame does not change at all. Whatever is being said over it, this is two voices
over a still. These are all 0, and it is not close:
- a correction, however sharp
- a warning, however important
- the best explanation in the session
- a plan, a list of parts on order, somebody describing what they are about to build
- working out loud about what to do next
- a greeting, a microphone check, small talk, a good joke, a touching line
- a question answered out of memory with nothing shown
Every one of those reads well written down. None of them is anything to look at. This is the
mistake to be most careful about: the strongest-reading thing in the session is usually a 0.

One thing is never kept whatever it marks: a picture carrying somebody's private paperwork - a
home address, a VIN or serial number, a registration or insurance document - or a close-up of a
person's face showing an injury. This reel gets shared with people who were not there.

THE BAR IS 2, AND APPLYING IT IS MECHANICAL, not a second opinion. You already made the
judgement when you wrote the mark down; do not now talk yourself back out of it. Work through
these in order. They only ever choose WHICH moment to cut, never whether to cut anything at all.

1. IS THERE A 3 IN YOUR LIST? Then you are writing at least one clip. Take the 3 where the
   picture changed most and build a clip around it. That moment's own time has to sit inside the
   clip's ranges.
2. IS THERE A SECOND 3 THAT LOOKS PLAINLY DIFFERENT FROM THE FIRST? A photograph that became a
   cartoon where the other became a cut-out, a second drawing of a different thing - something a
   stranger would see as a different picture. Then write a second clip for it.
   A 3 that only NUDGES the picture you already cut - brighter, warmer, a shelf under it, moved
   over a bit - does not get a clip. Skip it and stop. A session marked 3, 3, 3 ends with one
   clip or two. IT NEVER ENDS WITH NOTHING.
3. NO 3s AT ALL? Then your best 2 becomes the clip. A second 2 gets a clip only when it is a
   picture of a completely different thing, asked for separately, minutes away, with its own line
   from the person.
4. TWO CLIPS FROM ONE SESSION AT THE MOST, and never two that look alike. All these clips land on
   one reel together, so two clips that look alike are worse than one. The same picture nudged
   twice, the same object snapped twice, a drawing and then a written version of it: one clip,
   and it is the first and the more picture-like of them.
5. At most {max_clips} clips in the answer, best mark first.

NOTHING is the answer when, and only when, every line you wrote is a 1 or a 0. That is a session
of zeroes, not a failure to find something, and saying so costs nothing - most sessions here are
somebody checking the microphone works.

DO NOT HOLD OUT FOR DELIGHT. Nobody in this workshop whoops. The ordinary case is a person asking
for a picture and the picture showing up, and THAT IS THE MOMENT - it is never too plain to keep.
Nothing has to be discovered, argued about, got wrong or admitted for a clip to be worth
watching. A flat "cool, thanks" over a picture that just changed is a real reaction and counts.
Small and visible beats big and invisible every time.

FOUR SHAPES THAT PASS. Look for these.

  A. SOMEBODY ASKS AND THE MACHINE DELIVERS. A person asks for a change to a picture, or for a
     number, or for a drawing of the thing in front of them, and the changed picture, the number
     or the drawing arrives. The best shape here.
  B. SOMEBODY HOLDS UP WHAT THEY ARE BUILDING. A snap of the machine, the part, the tool, the
     case, and the person says what it is or what they mean to do with it. The photo landing plus
     their own line about it is the whole clip.
  C. SOMEBODY REACTS TO A PICTURE THAT JUST CHANGED. Pleased, impatient, teasing, unimpressed,
     pushing for more. The picture change is the setup and their line is the punchline.
  D. THE WAIT PAYS OFF. A drawing can take a minute and a half to arrive, and while it does the
     person gets restless - wondering aloud how long this will take, asking whether it is still
     working on it, saying they will wait. Put that line next to the moment the picture lands,
     cut out everything in between, and you have the best joke this machine makes. Do not let a
     long wait talk you out of a session: the request may be too far back to reach, but the
     restless line and the arrival are almost always close enough.

If nothing here reached the bar, write your MARK lines, then one word on its own line, and stop:

NOTHING

Otherwise one block per clip, at most {max_clips} of them, best first:

CLIP
TITLE: Make the bracket blue
KEEP:
88.5-96.0
99.5-104.0

TITLE is one line under {title_chars} characters. Name what appeared and why somebody asked for
it, in the words a person would use out loud - the person's own words are the best title there
is. Never the passive voice of an archive label. But the transcript mishears things, so never
quote a line that reads as garbled or misheard; when the words are broken, say what happened
instead. No quotes, no emoji, no "In this video".

The KEEP lines are the seconds of the recording the clip is built from, smallest number first.
Choose them like this.

1. HOLD ON THE PICTURE, AND THIS RULE BEATS EVERY RULE BELOW IT. A [photo] or [wrote on the
   panel] line is stamped at the instant the picture changed, but the picture stays up afterwards,
   and that shot is the whole payoff. So the range containing an arrival runs from about a second
   BEFORE that stamp to at least SIX SECONDS AFTER it. A photo stamped at 122.0 is kept as
   121.0-128.0 at the very least, and longer if the person says something about it. Do this even
   when nobody speaks in those six seconds and even when the only voice is Cyclops saying it is up
   on the screen: the recording is still running, the picture is still there, and quiet seconds on
   something that just appeared are the shot, not dead air. Cutting away one second after a
   picture lands is the single commonest way to wreck one of these clips - on the reel it is a
   flicker and the viewer never sees what arrived. Watch the END of the clip especially: whatever
   else the ranges do, THE LAST NUMBER YOU WRITE must be at least six seconds past the last
   picture stamp inside the clip. A clip that runs 102.3-122.8 around a photo stamped at 122.0 has
   done all the work and then cut away before the payoff - it needs to run to 128 or later.
2. THE WAITING IS NOT THE MOMENT. Ten to ninety seconds can pass between the asking and the
   picture landing. Build the clip out of separate ranges - the asking, then the arrival with its
   hold, then what the person says to it - and throw away the dead middle. Two ranges are normal.
   If a range you have written has more than about three seconds inside it where nobody speaks and
   nothing appears, split it in two rather than carrying the dead air. If the asking is further
   back than {window:.0f} seconds, let it go and open instead on whatever the person said while
   they were waiting, or simply on the arrival.
3. Overshooting the END of a range costs nothing: the silence is measured and trimmed off
   automatically before anything is rendered. Undershooting cuts somebody off mid-word and cannot
   be repaired. So when a range ends on something Cyclops is saying, run it past the end of that
   sentence rather than stopping on the timestamp the line starts at.
4. Everything in one clip comes from the same part of the session. First number to last number is
   never more than {window:.0f} seconds apart.
5. {low:.0f} to {high:.0f} seconds of kept material per clip, aiming for {target:.0f}. Add your
   ranges up before you write them down. Under {low:.0f} seconds is a flash, not a moment: fix it
   by lengthening the hold on the picture or taking in the whole line on either side, never by
   dropping the clip. Over {high:.0f} and one range is carrying dead air - trim that range.
6. Open on a person talking wherever you can, and close on one when there is one. Trim Cyclops's
   trailing offer to crop it tighter or asking what is next - but only when trimming it still
   leaves the new picture its six seconds. When it does not, keep talking over the picture
   instead. Rule 1 wins.
7. Cut in the silence between turns and never inside a sentence. Start about half a second before
   the first word kept and end about half a second after the last.
8. Ranges are ascending and never overlap - not inside one clip and not between two clips. No
   second of the recording is used twice.

BEFORE YOU ANSWER, take each clip you have written and check it against this list. These are
repairs to the ranges, not a chance to reconsider the mark you gave.

  - Does a [photo] or a [wrote on the panel] stamp actually fall INSIDE one of its ranges? Not
    nearby, not just before the range starts - INSIDE, with the range beginning a second or so
    earlier than the stamp. This goes wrong most often when the picture lands first and the person
    speaks after it: the clip opens on their line and the arrival is a second outside the range,
    so nobody watching ever sees the picture appear. Pull the start back past the stamp. If no
    range can be made to contain a stamp, delete the clip.
  - Is the clip's very last number at least six seconds past the last picture stamp in it? If it
    is not, push it out until it is. This is the one that keeps going wrong.
  - Do the ranges add up to between {low:.0f} and {high:.0f} seconds?
  - Are the ranges ascending, and does no range overlap another, in this clip or in any other clip
    you are sending?

Timeline:
{timeline}"""


def decide(shown: str, seconds: float, settings: Settings) -> tuple[tuple[Clip, ...], bool]:
    """Ask what is worth clipping. ``(clips, asked)``. Never raises.

    ``asked`` says whether the model actually answered, so "no key" and "nothing here" are
    recorded differently: the first writes no plan and leaves the session pending, the second
    writes an empty plan and is never asked again. Collapsing the two would mean a night without
    a network quietly deciding that the whole card is boring.

    Imported here rather than at the top of the module for the reason in the module docstring:
    the admin service asks this module for a folder's state on every listing and has no business
    loading an SDK to do it.
    """
    if not shown.strip() or not settings.cut or not settings.api_key:
        return (), False

    from openai import APIError, OpenAI

    # max_retries=0 for slug.describe_session's reason, and one more: three retries of a
    # sixty-second budget is three minutes of an index service not indexing.
    client = OpenAI(api_key=settings.api_key, timeout=CUT_TIMEOUT_S, max_retries=0)
    try:
        response = client.responses.create(
            model=CUT_MODEL,
            reasoning={"effort": "low"},  # unlike a folder name, this is a judgement
            input=CLIP_PROMPT.format(
                seconds=seconds, max_clips=MAX_CLIPS, title_chars=MAX_TITLE_CHARS,
                window=WINDOW_S, low=12, high=int(MAX_CLIP_S), target=int(TARGET_CLIP_S),
                timeline=shown,
            ),
        )
        answer = (getattr(response, "output_text", "") or "").strip()
    except (APIError, OSError, ValueError):
        return (), False
    except Exception:  # noqa: BLE001 - a clip is never worth taking the index service down
        return (), False
    finally:
        with suppress(Exception):
            client.close()
    return parse(answer), True


def parse(answer: str) -> tuple[Clip, ...]:
    """One block per clip, each read on its own.

    slug.parse's doctrine one level up: a block that goes wrong loses that block and not the
    other two. NOTHING anywhere in the reply means nothing, because a model that has decided
    that has decided it - and being able to say so is most of what makes the reel watchable.
    """
    blocks: list[dict] = []
    for line in answer.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper in {"NOTHING", "NONE"}:
            return ()
        if _CLIP_LINE.match(stripped) or upper.startswith("CLIP:"):
            title = stripped.split(":", 1)[1].strip().strip('"') if ":" in stripped else ""
            blocks.append({"title": title, "ranges": []})
            continue
        if not blocks:
            continue  # anything before the first CLIP is preamble
        if upper.startswith("TITLE:"):
            blocks[-1]["title"] = stripped[6:].strip().strip('"')
            continue
        if upper.startswith("KEEP:"):
            stripped = stripped.split(":", 1)[1].strip()
            if not stripped:
                continue
        found = _RANGE_LINE.match(stripped)
        if found:
            blocks[-1]["ranges"].append((float(found.group(1)), float(found.group(2))))
    return tuple(
        Clip(title=_clip(one["title"], MAX_TITLE_CHARS), ranges=tuple(one["ranges"]))
        for one in blocks[:MAX_CLIPS]
        if one["ranges"]
    )


def _clip(text: str, limit: int) -> str:
    """One line, no control characters, cut at a word boundary if it has to be cut."""
    flat = " ".join(str(text or "").replace(" ", " ").split())
    if len(flat) <= limit:
        return flat
    return flat[:limit].rsplit(" ", 1)[0].rstrip(",.;:") + "…"


# ------------------------------------------------------------------ making it renderable


def snap_to_turns(ranges, records: list[dict]) -> tuple[tuple[float, float], ...]:
    """Each start pulled onto a logged turn when it is within ``SNAP_TO_TURN_S`` of one.

    A model asked for seconds is routinely a second or two out, and two seconds out means the
    clip opens on the tail of the previous sentence - which is the one thing that kills a cold
    open. The log knows where every turn started, so this is three lines rather than a subsystem
    for making the model name quotes instead of numbers.

    Only the start moves. Pulling the end onto a turn boundary would as often cut off the word
    the clip exists for, and :func:`tighten` is about to trim the tail against measured audio.
    """
    marks = sorted(
        float(r.get("t", 0.0) or 0.0) for r in records if r.get("type") in {"you", "cyclops"}
    )
    if not marks:
        return tuple((float(a), float(b)) for a, b in ranges)

    def pull(t: float) -> float:
        near = min(marks, key=lambda m: abs(m - t))
        return near if abs(near - t) <= SNAP_TO_TURN_S else t

    return tuple((pull(float(a)), float(b)) for a, b in ranges)


def tighten(ranges, speech) -> tuple[tuple[float, float], ...]:
    """The chosen ranges, with the silence taken out of them.

    Only ever a TRIM of ranges the model named, never a source of new ones. That ordering is the
    whole safety argument: silence detection cannot tell an interesting moment from a dull one,
    and a threshold two decibels wrong can only cost a fraction of a second off an edge rather
    than putting a stranger's dead air into the reel.

    No spans means no opinion: the ranges pass through untouched and the clip renders at turn
    granularity, which is exactly what the design before this one did. A quality multiplier, not
    a dependency - a box with no ffmpeg still gets clips, just looser ones.
    """
    if not speech:
        return tuple((float(a), float(b)) for a, b in ranges)
    out: list[tuple[float, float]] = []
    for a, b in ranges:
        hits = [[max(a, s), min(b, e)] for s, e in speech if e > a and s < b]
        for piece_a, piece_b in _bridge(hits, BRIDGE_S):
            if piece_b - piece_a >= FLOOR_S:
                out.append((piece_a, piece_b))
    return tuple(out)


def sanitize(raw: Any, seconds: float) -> tuple[tuple[float, float], ...]:
    """Whatever was suggested, turned into ranges that cannot render badly.

    Run over the model's answer *and* over whatever is read back out of ``plan.json``, so that a
    plan somebody edited by hand is held to the same rules. The plan stores what comes out of
    here, never what went in, so the file always says exactly what was rendered.

    Returns () when nothing survives, which the caller reads as "there is no clip here".
    """
    if seconds <= 0:
        return ()
    pairs: list[list[float]] = []
    for item in raw if isinstance(raw, list | tuple) else ():
        if not isinstance(item, list | tuple) or len(item) != 2:
            continue  # dropped, never repaired: a three-element range meant nothing knowable
        try:
            start, end = float(item[0]), float(item[1])
        except (TypeError, ValueError):
            continue
        if isinstance(item[0], bool) or isinstance(item[1], bool):
            continue
        if not (math.isfinite(start) and math.isfinite(end)):
            continue
        if end < start:
            start, end = end, start  # written backwards still meant a range
        start = max(0.0, start - LEAD_IN_S)
        end = min(seconds, end + LEAD_OUT_S)
        # Padding before merging is deliberate: it is what makes two ranges either side of a
        # breath touch, so the step below joins them instead of leaving a frame of black between.
        if end - start >= MIN_KEEP_S:
            pairs.append([start, end])

    pairs.sort(key=lambda p: p[0])
    merged: list[list[float]] = []
    for start, end in pairs:
        # A negative gap is an overlap, so this is also where overlapping keeps become one.
        if merged and start - merged[-1][1] < MERGE_GAP_S:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    # One clip is one moment, and that is enforced here as well as asked for in the prompt: a
    # model that answered with the start of the session and the end of it gets the start.
    if merged and merged[-1][1] - merged[0][0] > WINDOW_S:
        first = merged[0][0]
        merged = [p for p in merged if p[1] <= first + WINDOW_S] or merged[:1]

    if len(merged) > MAX_RANGES:
        # In time order. Keeping the longest - which is what the one-video design did - scatters
        # a tightened clip across its own holes and deletes the short reactions that are the point.
        merged = merged[:MAX_RANGES]

    kept: list[tuple[float, float]] = []
    total = 0.0
    for start, end in merged:
        if total >= MAX_CLIP_S:
            break
        end = min(end, start + (MAX_CLIP_S - total))
        if end - start < MIN_KEEP_S:
            continue
        kept.append((_snap(start), _snap(end)))
        total += end - start
    return tuple(kept)


def _snap(t: float) -> float:
    """To the nearest frame. An unsnapped boundary makes concat's arithmetic unreproducible."""
    return round(t * FPS) / FPS


# ------------------------------------------------------------------ what to call it


def naming(folder: Path, title: str) -> str:
    """A title that is never empty, whatever the model did or did not say.

    The fallbacks below the model's own answer are ``library._read``'s chain, reused rather than
    written again, so the caption on the reel and the row in the session list cannot disagree.
    """
    was_title, _ = library._summary(folder)  # noqa: SLF001 - see the docstring
    when = library._started(folder)  # noqa: SLF001
    if not title:
        title = was_title
    if not title:
        slug = library._slug(folder)  # noqa: SLF001
        if slug:
            title = slug.replace("-", " ").capitalize()
        elif when:
            title = f"Session {when:%Y-%m-%d %H:%M}"
        else:
            title = folder.name
    return _clip(title, MAX_TITLE_CHARS)


def _span(seconds: float) -> str:
    minutes, rest = divmod(int(max(0.0, seconds)), 60)
    return f"{minutes}m {rest:02d}s" if minutes else f"{rest}s"


# ------------------------------------------------------------------ the subtitle file


def shift(t: float, ranges: tuple[tuple[float, float], ...], lead: float = 0.0) -> float | None:
    """Where a moment of the recording ends up in the finished clip, or None if it was cut out.

    The captions stand or fall on this. A record's ``t`` is a position in ``video.mp4``; after
    the trims and the concat it is somewhere else entirely, and a caption at the wrong somewhere
    is worse than no caption at all.
    """
    out = lead
    for start, end in ranges:
        if t < start:
            return None
        if t < end:
            return out + (t - start)
        out += end - start
    return None


# ------------------------------------------------------------------ the render


@dataclass(frozen=True)
class Cut:
    """What one render came to. ``why`` is ffmpeg's last line, and empty when it worked."""

    ok: bool
    why: str = ""


def clip_command(clip: Clip, out_name: str) -> list[str]:
    """The one ffmpeg call that makes one clip. Run with ``cwd`` set to ``<session>/clips``.

    **Nothing a model wrote reaches the filtergraph.** There used to be a long argument here about
    ``subtitles=`` - the one argument whose value ffmpeg parses as filter grammar, where ``:``
    ``'`` ``\\`` ``,`` ``[`` ``]`` all mean something - and about keeping a session folder's
    part-model-written name away from it by passing a bare ``1.ass`` and running in the folder.
    Captions are not burned in any more, so that whole surface is gone rather than guarded: every
    value in the graph below is a number this module computed. The input is ``../video.mp4``,
    which is an argv element and never reaches the parser either.

    ``-ss`` before ``-i`` is input seeking, which is accurate when re-encoding - ffmpeg decodes
    from the preceding keyframe and discards - and it is what keeps the decode proportional to
    the clip rather than to the session. A fifteen-second clip out of six minutes is a fifteen
    second decode, which is most of why three clips cost less than the one video did.
    """
    base = clip.ranges[0][0]  # -ss rebases every timestamp to zero
    span = clip.ranges[-1][1] - base
    graph: list[str] = []
    for n, (start, end) in enumerate(clip.ranges):
        a, b = start - base, end - base
        fade_out = max(0.0, (b - a) - EDGE_FADE_S)
        graph.append(f"[0:v]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS[v{n}]")
        graph.append(
            f"[0:a]atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS,"
            # Cutting mid-waveform clicks. Twenty milliseconds either side of every join is
            # inaudible as a fade and is the difference between "edited" and "chopped".
            f"afade=t=in:st=0:d={EDGE_FADE_S},afade=t=out:st={fade_out:.3f}:d={EDGE_FADE_S}[a{n}]"
        )
    chain = "".join(f"[v{n}][a{n}]" for n in range(len(clip.ranges)))
    graph.append(f"{chain}concat=n={len(clip.ranges)}:v=1:a=1[vb][ab]")
    graph.append(
        f"[vb]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease,"
        f"pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,"
        f"format=yuv420p[v]"
    )
    graph.append(
        # The microphone is on the left and Cyclops on the right, which is right for an archive
        # and unlistenable in one earbud. This flattens one derived copy; record.py's note about
        # not flattening every recording ever made still stands.
        "[ab]pan=stereo|c0=0.5*c0+0.5*c1|c1=0.5*c0+0.5*c1,aresample=48000[a]"
    )
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-ss", f"{base:.3f}", "-t", f"{span + 0.2:.3f}",
        "-i", f"../{card.VIDEO}",
        "-filter_complex", ";".join(graph),
        "-map", "[v]", "-map", "[a]",
        # Two threads, not four. Even in the seconds between a conversation starting and the
        # sweep below noticing, this must not be able to take the whole box.
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
        "-threads", "2",
        "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart",
        "-f", "mp4",  # the output is a scratch name; ffmpeg cannot infer the container from it
        out_name,
    ]


def busy(sessions_dir: Path) -> str:
    """Why this is not the moment to render, or "" when it is.

    The most important few lines in the module. A libx264 job at 200% of a core, on a box with
    four of them, while a realtime audio thread and a live encoder already have work to do,
    produces dropouts in a conversation - and they get blamed on the wifi, not on this.
    """
    try:
        live = any(
            card.locked(entry)
            for entry in sessions_dir.expanduser().iterdir()
            if entry.is_dir() and (entry / card.LOG_NAME).exists()
        )
    except OSError:
        live = False
    if live:
        return "waiting for the conversation to end"
    temp = stats.cpu_temp_c()
    if temp is not None and temp >= stats.HOT_C:
        return "too hot to render"
    return ""


def render(folder: Path, clip: Clip, n: int, sessions_dir: Path) -> Cut:
    """Make one clip, landing the file only if ffmpeg said it worked.

    ``record.mux``'s discipline and for its incident: the encode happens under a scratch name and
    is renamed into place, so a clip existing means *a render returned zero* - which is the only
    reading under which the retry above can be trusted.

    ``Popen`` rather than ``subprocess.run`` is a deliberate deviation from the house pattern, for
    one reason: a render will routinely overlap a conversation that starts thirty seconds into it,
    so being polite only at the start is not being polite. This watches, and stands down. A
    stand-down leaves the unit owed - it is a pause, not a failure.
    """
    out = clip_path(folder, n)
    tmp = card.tmp_for(out)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(  # noqa: S603 - the command is ours, from clip_command
            clip_command(clip, tmp.name),
            cwd=out.parent, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        # Notably FileNotFoundError: systemd's PATH is not a login shell's.
        tmp.unlink(missing_ok=True)
        return Cut(False, f"{type(exc).__name__}: {exc}")

    deadline = time.monotonic() + RENDER_TIMEOUT_S
    stood_down = ""
    while proc.poll() is None:
        if time.monotonic() > deadline:
            stood_down = f"gave up after {RENDER_TIMEOUT_S:.0f}s"
            break
        temp = stats.cpu_temp_c()
        if temp is not None and temp >= stats.THROTTLE_C:
            stood_down = "stopped to let the board cool"
            break
        if busy(sessions_dir):
            stood_down = "paused for a conversation"
            break
        time.sleep(2.0)

    if stood_down:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            with suppress(subprocess.SubprocessError):
                proc.wait(timeout=5.0)
        tmp.unlink(missing_ok=True)
        return Cut(False, stood_down)

    detail = (proc.stderr.read() if proc.stderr else b"").decode(errors="replace").strip()
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        lines = detail.splitlines()
        return Cut(False, lines[-1] if lines else f"exit {proc.returncode}")
    try:
        card.land(tmp, out)
    except OSError as exc:
        # Notably ENOENT, when session.describe renamed the folder while this was encoding. The
        # plan is still there under the new name, so the next sweep picks it up - the same
        # best-effort answer captions.fill gives to the same race.
        tmp.unlink(missing_ok=True)
        return Cut(False, f"{type(exc).__name__}: {exc}")
    return Cut(True)


# ------------------------------------------------------------------ the sweep


def one(settings: Settings) -> str:
    """Do at most one unit of work, and say what happened. What the index service calls.

    One unit rather than all of them: a finished clip lands in a folder the index service is
    watching, which rings its bell, which schedules the next sweep, which does the next unit.
    The queue drains itself, within a session as well as across sessions, and no single wake
    ever burns two encodes back to back.
    """
    if not settings.cut:
        return ""
    sessions_dir = settings.sessions_dir.expanduser()

    # The lock is taken before the queue is even looked at, and that ordering is the fix to a
    # bug Marco found by reading the header: tidying the ledger only when there was something to
    # render meant a killed render's row sat there claiming to be working right up until
    # tasks.running() aged it out a quarter of an hour later. Taking it first costs an open, a
    # flock and a close on a sweep that was going to walk the card anyway.
    with _held() as mine:
        if not mine:
            return ""  # somebody else is rendering; the next sweep is soon enough
        _drop_stale_rows()
        todo = work(sessions_dir)
        if todo is None:
            return ""
        held_up = busy(sessions_dir)
        if held_up:
            return f"not clipping yet - {held_up}"
        if todo.index < 0:
            return _decide_one(todo.folder, settings)
        return _render_one(todo.folder, todo.index, sessions_dir)


def _drop_stale_rows() -> None:
    """Close any of our own ledger rows left open by a render that was killed.

    Only ever called while this process holds CUT_LOCK, and that is what makes it safe: the lock
    is exclusive and is held for the whole of a render, so a row of ours still saying `running`
    while we hold it belongs to a process that is gone. Nothing has to guess at an age.

    It exists because the deploy path kills renders *by design* - push.sh restarts the index
    service, the cgroup takes ffmpeg with it, and the plan on the card is what makes that free.
    Free for the render, but the ledger row went with the process too, and cyclops.tasks has no
    way to know: nothing writes "this process died". So the panel and every phone went on saying
    "Cutting the video of…" for the fifteen minutes it takes tasks.running() to give up on it.
    Found by Marco reading the header and asking whether it was true. It was not.
    """
    from . import tasks

    for task in tasks.running():
        if task.what.startswith((CHOOSING, CUTTING)):
            tasks.fail(task.id, "interrupted")


def _decide_one(folder: Path, settings: Settings) -> str:
    """Look at one session once: check it is worth anything, measure the silence, ask.

    The order is the point. :func:`worth_asking` reads the log and nothing else, so a session
    that is plainly a test costs one read and one small file - no ffprobe, no silencedetect, no
    network - and is never considered again.
    """
    from . import tasks

    records, _ = card.read_log(folder / card.LOG_NAME)
    thin = worth_asking(records)
    if thin:
        # An empty plan with the reason in `note` rather than in `why`: this session was not
        # looked at and did not fail, and the page must not offer to try again.
        write_plan(folder, Plan(decided=_now(), by="gate", note=thin, clips=()))
        _sweep_old(folder)
        return f"nothing worth a clip in {folder.name}: {thin}"

    seconds, has_video = probe(folder, records)
    if not has_video or seconds <= 0:
        write_plan(
            folder,
            Plan(decided=_now(), seconds=seconds,
                 why="that recording has sound but no picture in it"),
        )
        return f"cannot clip {folder.name}: it has no picture in it"

    task = tasks.start(f"{CHOOSING} {folder.name}…")
    speech = listen(folder, seconds)
    found, asked = decide(timeline(records, _captions(folder)), seconds, settings)
    if not asked:
        # No key, no network, a refusing model. No plan is written, so this session is still
        # pending and the next sweep with a key processes the whole backlog. "Nobody looked" and
        # "somebody looked and there was nothing" must never be the same file.
        tasks.fail(task, "could not ask")
        return f"could not decide {folder.name}: no answer from the model"

    clips: list[Clip] = []
    for got in found:
        ranges = sanitize(
            [list(r) for r in tighten(snap_to_turns(got.ranges, records), speech)], seconds
        )
        if sum(b - a for a, b in ranges) < MIN_CLIP_S:
            continue  # nothing left after the pauses came out; not a clip
        clips.append(Clip(title=naming(folder, got.title), ranges=ranges))
    write_plan(
        folder,
        Plan(decided=_now(), by="model", seconds=seconds, speech=speech,
             clips=tuple(clips[:MAX_CLIPS]), source=_source(folder)),
    )
    _sweep_old(folder)
    tasks.finish(task, f"{len(clips)} clips" if clips else "nothing worth a clip")
    return (
        f"{folder.name}: {len(clips)} clips" if clips
        else f"nothing worth a clip in {folder.name}"
    )


def _sweep_old(folder: Path) -> None:
    """Remove what the one-video-per-session design left, as the walk passes each folder.

    A migration that needs no migration script: every session is considered exactly once, and
    the moment it is, its old ``cut.mp4`` stops meaning anything. The names stay in ``card.KNOWN``
    forever so a folder still holding one can still be deleted.
    """
    for old in (card.CUT, card.CUT_PLAN, card.CUT_SUBS, card.CUT_REQUEST):
        (folder / old).unlink(missing_ok=True)


def _render_one(folder: Path, n: int, sessions_dir: Path) -> str:
    """Render one clip. Every path out of here either lands a file or writes a why.

    That is the queue's only escape hatch and it is enforced here rather than asserted: a clip
    that neither renders nor records a reason would be returned by :func:`work` on every bell
    forever, and because that walk is newest-first it would starve every older session behind it.
    """
    from . import tasks

    plan = read_plan(folder)
    if plan is None or n > len(plan.clips):
        return ""
    clip = plan.clips[n - 1]
    ranges = sanitize([list(r) for r in clip.ranges], plan.seconds)
    if sum(b - a for a, b in ranges) < MIN_CLIP_S:
        _blame(folder, plan, n, "nothing left of that one after the pauses came out")
        return f"{folder.name} clip {n}: nothing left to render"
    clip = replace(clip, ranges=ranges)

    task = tasks.start(f"{CUTTING} {clip.title}…")
    done = render(folder, clip, n, sessions_dir)
    if not done.ok:
        tasks.fail(task, done.why)
        if done.why in {"paused for a conversation", "stopped to let the board cool"}:
            return f"{folder.name}: {done.why}"  # the unit stays owed; this is a pause
        _blame(folder, plan, n, done.why)
        return f"could not clip {folder.name} ({n}): {done.why}"
    tasks.finish(task, _span(clip.seconds))
    return f"clipped {folder.name} ({n}): {_span(clip.seconds)}"


def _blame(folder: Path, plan: Plan, n: int, why: str) -> None:
    """Write a reason into one clip, which takes it off the queue for good."""
    clips = list(plan.clips)
    clips[n - 1] = replace(clips[n - 1], why=why)
    write_plan(folder, replace(plan, clips=tuple(clips)))


def _now() -> str:
    return f"{datetime.now().astimezone():%Y-%m-%dT%H:%M:%S%z}"


def _captions(folder: Path) -> dict[str, str]:
    """What the index service said each of this session's photos is of, by filename."""
    try:
        raw = json.loads((folder / card.PHOTOS / card.CAPTIONS_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {k: str(v) for k, v in raw.items() if isinstance(v, str)}


def _source(folder: Path) -> dict[str, Any]:
    """The freshness pair the rest of the codebase keys on, for the recording this was cut from."""
    try:
        info = (folder / card.VIDEO).stat()
    except OSError:
        return {}
    return {"size": info.st_size, "mtime_ns": info.st_mtime_ns}


# ------------------------------------------------------------------ the listing


@dataclass(frozen=True)
class Made:
    """One finished clip, as much of it as the reel needs to play it."""

    id: str        # "<session>/<n>", unique and stable
    name: str      # the session folder
    n: int
    title: str
    started: str
    seconds: float
    bytes: int
    src: str       # /media/<session>/clips/<n>.mp4


def clips(sessions_dir: Path, limit: int = 200) -> tuple[list[Made], dict]:
    """Every finished clip, newest session first and best first within a session.

    Finished only, which reverses the old listing's decision on purpose: an in-progress row is
    what somebody who just pressed a button came to see, and nobody presses a button to get here
    any more. A reel is for watching, so a clip is in it when it can be played.

    The counts beside it are what stop an empty reel being a mystery - "nobody has looked yet"
    and "everything has been looked at and none of it was interesting" are different sentences.
    """
    out: list[Made] = []
    looked = found = waiting = 0
    try:
        folders = library._folders(sessions_dir)  # noqa: SLF001 - newest first
    except OSError:
        return [], {"looked": 0, "found": 0, "waiting": 0}
    for folder in folders:
        plan = read_plan(folder)
        if plan is None:
            if card.written(folder / card.VIDEO) and card.written(folder / card.PAGE_NAME):
                waiting += 1
            continue
        looked += 1
        if plan.clips:
            found += 1
        when = library._started(folder)  # noqa: SLF001
        for n, clip in enumerate(plan.clips, start=1):
            path = clip_path(folder, n)
            if len(out) >= limit or not card.written(path):
                continue
            out.append(
                Made(
                    id=f"{folder.name}/{n}",
                    name=folder.name,
                    n=n,
                    title=clip.title or naming(folder, ""),
                    started=when.isoformat() if when else "",
                    seconds=round(clip.seconds, 2),
                    bytes=_size(path),
                    src=f"/media/{folder.name}/{card.CLIPS}/{n}.mp4",
                )
            )
    return out, {"looked": looked, "found": found, "waiting": waiting}


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0
