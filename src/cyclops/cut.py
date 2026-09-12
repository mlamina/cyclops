"""Cutting one recorded session into the short stories it holds, and rendering them.

A session leaves ``video.mp4`` - the panel, exactly as it was on the glass, for however long the
conversation ran. That is an archive, not a video: nobody sits through six minutes of somebody
waiting for an answer. What this module makes out of it is between zero and two clips, and every
one of them is a **story** rather than a moment.

**Three beats, in session order, and never anything else.** SETUP is the person asking for
something or holding something up - their whole line. TURN is the thing happening: the picture
arrives, the drawing lands, the number goes up, and it is held long enough that a viewer sees what
appeared. PAYOFF is how it ended. That shape is why these clips can autoplay muted on a reel with
no title card and no text on the frame: a stranger walking past follows one cold. The design this
replaced cut the *instant* instead - it was told in capitals that "THE WAITING IS NOT THE MOMENT" -
and the clips it made opened mid-conversation, had no ask and no ending, and Marco could not tell
what was going on in any of them.

**Zero is still the ordinary answer, and it is now bought in three tiers.** :func:`worth_asking`
reads the log and nothing else: a session with no turns, no length, no words - or with no new
picture anywhere in it - costs one read and is never considered again. What survives that goes to
:mod:`cyclops.stories`, where one cheap Reader call answers with candidate arcs or with none, and
only a session holding a candidate reaches the Director, the Looker and the Editor. Measured over
the eighty-seven sessions on this card, forty stop at the first tier for nothing.

**What makes the cuts land is the audio, not the transcript.** ``session.jsonl`` stamps every turn
against the same ``t0`` the video was muxed on, so a beat can be named in seconds with no
transcription to do. :func:`listen` runs ``silencedetect`` over each channel - measured at 0.55 s
for a six-minute recording, because ``-vn`` means the video packets are never decoded - and
:func:`shape` uses it to pull the head and tail of each beat onto real speech. **It only ever
trims the edges.** The pauses *inside* a beat stay, which is the other half of the repair: the old
:func:`tighten` intersected every range with measured speech and bridged only holes under 0.3 s,
so the six-second hold on a new picture came back as six to twenty splices and the clip read as a
glitch rather than as an edit.

The queue is the filesystem, as everywhere else here::

    clips/plan.json   present means this session has been considered. That is the whole gate
    clips/1.mp4       one clip. Existing means one ffmpeg run returned zero

**Plan-present-means-considered is the entire retry story.** ``deploy/push.sh`` restarts the index
service on every deploy and the default ``KillMode`` takes the whole cgroup with it, so a render
*will* be killed halfway - and when it is, the plan is still on the card, so the next sweep re-runs
ffmpeg for the clip that has no file yet and never pays for the model twice.

It also makes a bad cut fixable by hand: edit the ranges in ``clips/plan.json``, delete that clip's
``.mp4``, and the next sweep renders what you wrote. :func:`sanitize` runs over a hand-edited plan
exactly as it runs over what the crew decided, so that is safe to do. The plan carries the story
line, what the Looker saw at each beat and the Editor's blind retelling beside the numbers, so a
clip that reads wrong can be understood before it is repaired.

**A session is considered once, and only after it has a name.** :func:`work` waits for
``summary.md`` rather than for ``session.md``, because :mod:`cyclops.after` *renames* the folder
while it writes one - and a plan written into the old path recreated a stamp-only folder holding
nothing but ``clips/plan.json``, leaving the named folder to be decided all over again. Two of
those orphans and eleven double calls were on the card. ``SETTLE_S`` is the way out for a session
that never gets named at all.

Stdlib, :mod:`cyclops.card`, :mod:`cyclops.library` and :mod:`cyclops.stats` - all three of which
are themselves stdlib-only. :mod:`cyclops.stories` is imported inside :func:`_decide_one` rather
than at the top, the move :mod:`cyclops.after` and :mod:`cyclops.captions` already make:
``cyclops.admin.views`` imports this module to ask what state a session's clips are in, and
loading pydantic-ai to answer that would put a second and a half into a request handler.
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

# Five times the worst measured for a clip of this size. Wide enough that a slow encode is never
# mistaken for a wedged one, tight enough that a wedged one is not forever.
RENDER_TIMEOUT_S = 240.0

EDGE_FADE_S = 0.02  # 20 ms across every audio join, because cutting mid-waveform clicks
DIP_S = 0.12        # the dip to black BETWEEN beats. See clip_command: a jump in time that fades
                    # reads as an edit, and the same jump on a hard cut reads as a dropped frame

MAX_CLIPS = 2       # a bench session does not hold three stories, and two that look alike on one
                    # reel are worse than one

# The shape of a story, and the only numbers in this module a viewer would notice. MIN_SHOT_S is
# the repair for the splice storm: the old floor was 0.6 s, which is what let a twelve-second clip
# be twenty shots. TURN_HOLD_S is how long the new picture stays up before anything cuts away -
# under four seconds and nobody walking past ever sees what arrived.
MIN_SHOT_S = 2.5
TURN_HOLD_S = 4.0
STORY_LOW_S = 15.0   # below this it is a flash, not a story
STORY_CAP_S = 45.0   # above this the reel has lost them
STORY_TARGET_S = 27.0  # the middle of the 20-35 s the crew is asked to aim for
WINDOW_S = 60.0      # first number to last number, in the source. A story may reach across the
                     # whole session, and this is how far "the whole session" is allowed to be

LEAD_IN_S = 0.15     # the boundary is measured audio now, not a model's guess at one
LEAD_OUT_S = 0.25
FPS = 15.0           # what the panel records at; boundaries are snapped to it

SNAP_TO_TURN_S = 2.5  # a setup this close to one of the person's own turns was meant to be it

# Silence detection - see listen(). Two floors because they are two instruments: the mic is an
# open mic in a workshop, the right channel is the speaker's own zero-filled output.
MIC_FLOOR_DB = -38
CYC_FLOOR_DB = -50
SILENCE_MIN_S = 0.20
SILENCE_TIMEOUT_S = 120.0
BRIDGE_S = 0.30   # a hole this small between two audible spans is a breath, not a pause

# What a session has to have before anything is spent on it at all - see worth_asking. The
# cheapest filter there is, and it runs off the log, so a test session costs one read.
WORTH_YOU = 1       # turns from the person. One is the archetype here, not a mic check: "make
                    # this photo a cartoon" is one request, one picture and a whole clip.
WORTH_CYCLOPS = 2
WORTH_SECONDS = 30.0
WORTH_CHARS = 300   # of transcript, both voices. Kills a mic storm: thirty turns in four seconds

# A folder with no summary yet is one cyclops.after is still naming, and naming renames it. Ten
# minutes is far longer than a naming call and far shorter than a backlog - see work().
SETTLE_S = 600.0

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

_SIL = re.compile(r"silence_(start|end):\s*(-?\d+(?:\.\d+)?)")


# ------------------------------------------------------------------ the plan


# The three beats, in the order they must appear in. Written down once, because the shape is the
# whole design and a fourth kind of beat is not a feature anybody should be able to add by typing.
SETUP, TURN, PAYOFF = "setup", "turn", "payoff"
BEATS = (SETUP, TURN, PAYOFF)


@dataclass(frozen=True)
class Beat:
    """What one shot of a clip is, beside the numbers. One of these per range, in the same order.

    Deliberately holds no times. ``Clip.ranges`` is the renderable truth and the thing a hand
    repair edits; keeping a second copy of the same seconds here would mean a hand edit that
    silently disagreed with itself.
    """

    kind: str = ""  # one of BEATS
    what: str = ""  # what happens in it, in one clause
    saw: str = ""   # what the Looker found on the screen here. "" when it was never looked at


@dataclass(frozen=True)
class Clip:
    """One story, and the three pieces of the recording it is made of."""

    title: str = ""
    story: str = ""   # the story in one sentence, as the Director put it
    ranges: tuple[tuple[float, float], ...] = ()
    beats: tuple[Beat, ...] = ()  # one per range. () on a plan somebody edited into a new shape
    retelling: str = ""  # what the Editor got out of it watching cold, which is what let it pass
    why: str = ""  # why this one will never be rendered. Empty means it is still owed.

    @property
    def seconds(self) -> float:
        return sum(end - start for start, end in self.ranges)

    @property
    def shaped(self) -> bool:
        """Do the beats still describe the ranges? False after a hand edit changed their number."""
        return len(self.beats) == len(self.ranges) and len(self.ranges) > 1


@dataclass(frozen=True)
class Plan:
    """What was decided for one session. The whole of ``clips/plan.json``.

    Present means the session has been considered, which is what makes it the queue's marker;
    ``clips: []`` is the ordinary answer and takes a session off the queue for good.
    """

    decided: str = ""             # ISO-8601, when it was looked at
    by: str = "model"             # "model" | "gate" - which of the filters answered
    tier: int = 0                 # 1 the log, 2 the Reader, 3 the crew. Which one it got to
    cost: str = ""                # what the crew spent on it, in dollars, as written
    took: float = 0.0             # how long the crew took, in seconds
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

    def beats(got: Any) -> tuple[Beat, ...]:
        rows = got if isinstance(got, list | tuple) else ()
        return tuple(
            Beat(
                kind=str(one.get("kind", "")),
                what=str(one.get("what", "")),
                saw=str(one.get("saw", "")),
            )
            for one in rows
            if isinstance(one, dict)
        )

    clips = []
    for one in raw.get("clips", []) if isinstance(raw.get("clips"), list) else []:
        if isinstance(one, dict):
            clips.append(
                Clip(
                    title=str(one.get("title", "")),
                    story=str(one.get("story", "")),
                    ranges=pairs(one.get("ranges")),
                    beats=beats(one.get("beats")),
                    retelling=str(one.get("retelling", "")),
                    why=str(one.get("why", "")),
                )
            )
    return Plan(
        decided=str(raw.get("decided", "")),
        by=str(raw.get("by", "")),
        tier=int(raw.get("tier", 0) or 0) if isinstance(raw.get("tier"), int | float) else 0,
        cost=str(raw.get("cost", "")),
        took=float(raw.get("took", 0.0) or 0.0),
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
        if not card.written(folder / card.VIDEO) or not card.written(folder / card.PAGE_NAME):
            continue
        if card.locked(folder):
            continue
        if not settled(folder):
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


def settled(folder: Path) -> bool:
    """Has this session stopped moving? Asked before anything is spent on deciding about it.

    A finished ``summary.md`` is the marker, and the reason is a race that was costing every
    session on this card a second model call. ``cyclops.after`` writes the summary and then
    **renames the folder** to include the slug; the index service, meanwhile, was happy with
    ``session.md``, which exists from the moment the session ends. So a decision that began before
    the rename finished wrote its plan back to a path that no longer existed - recreating the
    stamp-only folder with nothing in it but ``clips/plan.json`` - while the renamed folder, with
    no plan, was picked up and decided all over again. Two orphans and eleven doubled sessions.

    ``SETTLE_S`` is the escape hatch for a session that never gets a summary at all - no key that
    night, or a model that declined - because "wait for the naming" must not mean "wait forever".
    Measured against the recording's own mtime: the mux happens at teardown and never again, where
    a directory's mtime changes every time anything is written beside it.
    """
    if card.written(folder / card.SUMMARY_NAME):
        return True
    try:
        age = time.time() - (folder / card.VIDEO).stat().st_mtime
    except OSError:
        return False
    return age > SETTLE_S


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

    The last clause is the newest and the biggest: **a session where the picture never changed
    cannot hold a TURN**, so it cannot hold a story, and no amount of cleverness downstream will
    find one. Twenty of the eighty-seven sessions on this card are exactly that - a question
    answered out of memory, a search that came back with prose - and they are twenty model calls
    that were being paid for an answer nobody could have given.
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
    if not any(made_a_picture(r) for r in records):
        return "nothing new ever appeared on the screen in that one"
    return ""


def made_a_picture(record: dict) -> bool:
    """Did this record put something new on the screen? The TURN beat has to be one of these.

    ``recall`` is not here and must not be: it drags an OLD picture back out of storage, which is
    the same few images session after session, and a story built on one is a story about the
    filing cabinet. Neither is ``search`` - nothing happens on screen at all.
    """
    kind = record.get("type")
    if kind == "photo":
        return bool(record.get("file"))
    return kind in {"screen", "sketch"}


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
        elif kind == "sketch":
            # What replaced ``screen`` when the panel started taking Prefab rather than HTML, and
            # for a while the only record type that changes the picture was invisible here: every
            # session after 2026-09-11 reached the model with its drawings missing, which is most
            # of why the newest clips on the card had nothing to hold on. The code is Python, not
            # prose, so it is squeezed harder than a line of HTML would be.
            lines.append(f"[{at:.1f}] [drew on the panel] {_flat(record.get('code', ''))}")
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


def _clip(text: str, limit: int) -> str:
    """One line, no control characters, cut at a word boundary if it has to be cut."""
    flat = " ".join(str(text or "").replace("\u00a0", " ").split())
    if len(flat) <= limit:
        return flat
    return flat[:limit].rsplit(" ", 1)[0].rstrip(",.;:") + "\u2026"


# ------------------------------------------------------------------ making it renderable


def opens_on_a_person(start: float, records: list[dict]) -> float:
    """A setup's start, pulled back onto one of the person's own turns when one is near.

    Only ever backwards, and only onto a ``you`` turn. "Opens on the person's own line" is the
    first thing a viewer notices about one of these clips and the one thing the crew is routinely
    a second or two out on; the log knows where every turn began, so this is five lines rather
    than a subsystem for making a model name quotes instead of numbers.

    Pulling *forwards* would be a different thing entirely - it would trim the first word off the
    ask - so a start that has already crept past the turn it belongs to stays where it is and
    :func:`shape` trims it against real audio instead.
    """
    marks = [float(r.get("t", 0.0) or 0.0) for r in records if r.get("type") == "you"]
    behind = [m for m in marks if start - SNAP_TO_TURN_S <= m <= start]
    return min(behind) if behind else start


def tighten(ranges, speech) -> tuple[tuple[float, float], ...]:
    """The chosen ranges, with the silence trimmed off their **edges** and nowhere else.

    An edge trim, deliberately, and this is the half of the repair nothing else could do. The
    version before this intersected each range with measured speech and bridged only holes under
    ``BRIDGE_S``, so every pause longer than a breath became a cut: the median clip on the card
    was twelve seconds of six to twenty splices, and the six-second hold on a new picture - the
    whole payoff - came back as whatever speech happened to land inside it. A beat is one shot
    now. The silence inside it is somebody looking at a picture, which is the shot.

    No spans means no opinion: the ranges pass through untouched and the clip renders at the
    beats the crew named. A quality multiplier, not a dependency - a box with no ffmpeg still
    gets clips, just looser ones.
    """
    if not speech:
        return tuple((float(a), float(b)) for a, b in ranges)
    out: list[tuple[float, float]] = []
    for a, b in ranges:
        hits = [(max(a, s), min(b, e)) for s, e in speech if e > a and s < b]
        if not hits:
            out.append((float(a), float(b)))  # a silent hold on a picture is a shot, not dead air
            continue
        out.append((hits[0][0], hits[-1][1]))
    return tuple(out)


def shape(beats, speech, seconds: float, records: list[dict]) -> tuple[tuple[float, float], ...]:
    """Three beats as three renderable ranges: SETUP, TURN, PAYOFF, in session order.

    The one place a story becomes a cut, and the only function here that may make a range
    *longer*. Four things happen, in this order and for these reasons:

    1. The setup is pulled back onto the person's own line, so the clip opens on the ask.
    2. Each beat's edges are trimmed onto measured speech - the pauses inside it stay.
    3. Every beat is grown back to its floor: ``MIN_SHOT_S`` for a shot anybody can read, and
       ``TURN_HOLD_S`` for the turn, because the picture arriving is what the clip is *for* and a
       viewer who does not see it has watched nothing. Growth is forwards, into the recording, and
       never into the beat that follows.
    4. The whole thing is grown towards ``STORY_LOW_S`` and then capped at ``STORY_CAP_S``.

    Returns ``()`` when these three beats cannot be made into a clip at all - beats out of order,
    or a story that has nowhere left to grow into. The caller drops the story and says why.
    """
    if seconds <= 0 or len(beats) != len(BEATS):
        return ()
    spans = [(max(0.0, float(a)), min(seconds, float(b))) for a, b in beats]
    if any(b <= a for a, b in spans) or any(
        spans[n][1] > spans[n + 1][0] for n in range(len(spans) - 1)
    ):
        return ()

    first = opens_on_a_person(spans[0][0], records)
    spans[0] = (min(first, spans[0][0]), spans[0][1])
    trimmed = list(tighten(spans, speech))

    # A trim that pulled a beat past its neighbour's start cannot have been a trim of this beat.
    for n, (a, b) in enumerate(trimmed):
        low = trimmed[n - 1][1] if n else 0.0
        trimmed[n] = (max(low, min(a, spans[n][1] - 0.1)), max(min(b, seconds), a + 0.1))

    floors = [TURN_HOLD_S if kind == TURN else MIN_SHOT_S for kind in BEATS]
    grown = _grow(trimmed, floors, seconds)
    if grown is None:
        return ()
    capped = _cap(_stretch(grown, _towards(grown, floors), seconds), floors)
    if capped[-1][1] - capped[0][0] > WINDOW_S + LEAD_OUT_S:
        return ()
    snapped = [(_snap(a), _snap(b)) for a, b in capped]
    # Six boundaries rounded to the nearest frame can lose three frames between them, which is
    # what put two clips on this card at 14.93 s against a floor of 15. Paid back on the tail.
    short = STORY_LOW_S - sum(b - a for a, b in snapped)
    if short > 0:
        a, b = snapped[-1]
        snapped[-1] = (a, min(_snap(b + short + 1.0 / FPS), _snap(seconds)))
    return tuple(snapped)


def _grow(spans, floors, seconds: float) -> list[tuple[float, float]] | None:
    """Every span out to its floor, forwards first and backwards only if it has to be.

    ``None`` when one of them has nowhere to go, which means these three beats are sitting on top
    of each other and no arithmetic here can separate them.
    """
    out = list(spans)
    for n, (a, b) in enumerate(out):
        want = floors[n]
        if b - a >= want:
            continue
        ceiling = out[n + 1][0] if n + 1 < len(out) else seconds
        b = min(ceiling, a + want)
        if b - a < want:  # no room ahead; take it out of the pause in front instead
            floor = out[n - 1][1] if n else 0.0
            a = max(floor, b - want)
        if b - a < want - 0.05:
            return None
        out[n] = (a, b)
    return out


def _stretch(spans, wants, seconds: float) -> list[tuple[float, float]]:
    """Every span towards its target, as far as there is room, and never past its neighbour.

    Best-effort where :func:`_grow` is all-or-nothing, and the difference matters: a story whose
    payoff has nowhere to go should still come out as long as it can be, not as short as it
    started. Measured over the card, that is the difference between a clip at 14.9 seconds and
    one inside the 15-45 band it is supposed to be in.
    """
    out = list(spans)
    for n, (a, b) in enumerate(out):
        ceiling = out[n + 1][0] if n + 1 < len(out) else seconds
        out[n] = (a, max(b, min(ceiling, a + wants[n])))
    for n in range(len(out) - 1, -1, -1):  # backwards, into the pause in front of the beat
        a, b = out[n]
        if b - a >= wants[n]:
            continue
        floor = out[n - 1][1] if n else 0.0
        out[n] = (max(floor, b - wants[n]), b)
    return out


def _towards(spans, floors) -> list[float]:
    """Per-beat targets that would bring a short story up to ``STORY_LOW_S``.

    The turn gets the first shot's worth and the setup the last, because holding the new picture
    longer is the one way of making a clip longer that makes it better, and the ask is the one
    beat where extra seconds are somebody waiting to speak.
    """
    want = [max(floors[n], b - a) for n, (a, b) in enumerate(spans)]
    short = STORY_LOW_S - sum(want)
    order = (BEATS.index(TURN), BEATS.index(PAYOFF), BEATS.index(SETUP))
    # A shot at a time, round and round, rather than the whole shortfall onto one beat: a
    # ten-second hold and a two-second reaction add up to the same number as three even shots and
    # read as a still with a caption. Bounded by STORY_TARGET_S, so this always terminates.
    while short > 0.05:
        for n in order:
            if short <= 0.05:
                break
            add = min(short, MIN_SHOT_S)
            want[n] += add
            short -= add
        if sum(want) >= STORY_TARGET_S:
            break
    return want


def _cap(spans, floors) -> list[tuple[float, float]]:
    """Back under ``STORY_CAP_S``, taken off the longest beat first and never below its floor."""
    out = [list(one) for one in spans]
    for _ in range(len(out) * 2):
        over = sum(b - a for a, b in out) - STORY_CAP_S
        if over <= 0:
            break
        # The longest beat above its own floor. Trimming the tail rather than the head, because a
        # head is where somebody starts talking and a tail is where they have stopped.
        room = [(out[n][1] - out[n][0] - floors[n], n) for n in range(len(out))]
        slack, n = max(room)
        if slack <= 0:
            break
        out[n][1] -= min(over, slack)
    return [(a, b) for a, b in out]


def sanitize(raw: Any, seconds: float) -> tuple[tuple[float, float], ...]:
    """Whatever is in the plan file, turned into ranges that cannot render badly.

    Run over what :func:`shape` produced *and* over whatever is read back out of ``plan.json``, so
    that a plan somebody edited by hand is held to the same rules. The plan stores what comes out
    of here, never what went in, so the file always says exactly what was rendered.

    **It does not merge.** The version before this joined any two ranges closer together than a
    quarter of a second, which was right when a clip was one moment cut into twenty pieces and is
    wrong now that a range *is* a beat: two beats that happened to touch would silently become one
    and the clip would stop being three shots. Overlaps are resolved by shortening the earlier
    range instead, which keeps the count and the order a hand edit wrote down.

    Returns () when nothing survives, which the caller reads as "there is no clip here".
    """
    if seconds <= 0:
        return ()
    pairs: list[list[float]] = []
    for item in raw if isinstance(raw, list | tuple) else ():
        if not isinstance(item, list | tuple) or len(item) != 2:
            continue  # dropped, never repaired: a three-element range meant nothing knowable
        if isinstance(item[0], bool) or isinstance(item[1], bool):
            continue
        try:
            start, end = float(item[0]), float(item[1])
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(start) and math.isfinite(end)):
            continue
        if end < start:
            start, end = end, start  # written backwards still meant a range
        start = max(0.0, start - LEAD_IN_S)
        end = min(seconds, end + LEAD_OUT_S)
        if end - start >= MIN_SHOT_S:
            pairs.append([start, end])

    pairs.sort(key=lambda p: p[0])
    kept: list[tuple[float, float]] = []
    total = 0.0
    for start, end in pairs:
        if kept:
            start = max(start, kept[-1][1])
            if start - kept[0][0] > WINDOW_S:
                break  # one story, one part of the session. A plan naming both ends gets the first
        if total >= STORY_CAP_S:
            break
        end = min(end, start + (STORY_CAP_S - total))
        if end - start < MIN_SHOT_S:
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

    **The dip to black is the only thing on the frame.** A clip is three beats that may be a
    minute apart in the source, and a hard cut between two of them reads as a dropped frame on a
    recording that is otherwise a fixed camera on a still bench. ``DIP_S`` at each end of the
    jump says "time passed" in the one visual language a muted reel has. It is skipped where two
    beats are actually contiguous, because there the picture genuinely does continue.
    """
    base = clip.ranges[0][0]  # -ss rebases every timestamp to zero
    span = clip.ranges[-1][1] - base
    jumps = [
        n
        for n in range(len(clip.ranges) - 1)
        # Half a frame: a gap smaller than that is two beats the shaping grew into each other.
        if clip.ranges[n + 1][0] - clip.ranges[n][1] > 0.5 / FPS
    ]
    graph: list[str] = []
    for n, (start, end) in enumerate(clip.ranges):
        a, b = start - base, end - base
        fade_out = max(0.0, (b - a) - EDGE_FADE_S)
        dips = ""
        if n - 1 in jumps:
            dips += f",fade=t=in:st=0:d={DIP_S}"
        if n in jumps:
            dips += f",fade=t=out:st={max(0.0, (b - a) - DIP_S):.3f}:d={DIP_S}"
        graph.append(f"[0:v]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS{dips}[v{n}]")
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

    The order is the point, and it is the whole cost argument. :func:`worth_asking` reads the log
    and nothing else, so a session that is plainly a test - or one where the picture never changed
    - costs one read and one small file and is never considered again. Only what survives that
    pays for ffprobe, silencedetect and a Reader call, and only a session the Reader found a
    candidate arc in pays for the Director, the Looker and the Editor.

    Which tier a session stopped at goes in the plan and in the line this returns, because the
    line is the journal: ``journalctl -u cyclops-index`` is where the question "what is this
    costing" gets answered, and "nothing worth a clip" three hundred times does not answer it.
    """
    from . import stories, tasks

    records, _ = card.read_log(folder / card.LOG_NAME)
    thin = worth_asking(records)
    if thin:
        # An empty plan with the reason in `note` rather than in `why`: this session was not
        # looked at and did not fail, and the page must not offer to try again.
        write_plan(folder, Plan(decided=_now(), by="gate", tier=stories.GATE, note=thin, clips=()))
        _sweep_old(folder)
        return f"tier 1, free: {folder.name} - {thin}"

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
    title, paragraph = library._summary(folder)  # noqa: SLF001 - the Director's brief
    outcome = stories.tell(
        folder=folder,
        records=records,
        timeline=timeline(records, _captions(folder)),
        brief=f"{title}\n{paragraph}".strip(),
        seconds=seconds,
        settings=settings,
    )
    if not outcome.asked:
        # No key, no network, a refusing model. No plan is written, so this session is still
        # pending and the next sweep with a key processes the whole backlog. "Nobody looked" and
        # "somebody looked and there was nothing" must never be the same file.
        tasks.fail(task, "could not ask")
        return f"could not decide {folder.name}: no answer from the model"

    clips = [made for made in (_as_clip(folder, one, speech, seconds, records)
                               for one in outcome.stories) if made is not None]
    write_plan(
        folder,
        Plan(decided=_now(), by="model", tier=outcome.tier, cost=f"{outcome.cost}",
             took=outcome.took, seconds=seconds, speech=speech, note=outcome.why,
             clips=tuple(clips[:MAX_CLIPS]), source=_source(folder)),
    )
    _sweep_old(folder)
    tasks.finish(task, f"{len(clips)} stories" if clips else "no story in that one")
    spent = f"${outcome.cost} in {outcome.took:.0f}s"
    if clips:
        return f"tier {outcome.tier}, {spent}: {folder.name} - {len(clips)} story/ies"
    return f"tier {outcome.tier}, {spent}: {folder.name} - {outcome.why or 'no story in it'}"


def _as_clip(folder: Path, story, speech, seconds: float, records: list[dict]) -> Clip | None:
    """One story, shaped against measured speech, or None when the beats cannot be cut.

    The only place the crew's seconds become a clip's seconds. A story that cannot be shaped is
    dropped rather than repaired: the crew had its one repair already, and a story whose beats sit
    on top of each other is not a numbers problem.
    """
    ranges = shape([(one.start, one.end) for one in story.shots], speech, seconds, records)
    if not ranges or sum(b - a for a, b in ranges) < STORY_LOW_S:
        return None
    return Clip(
        title=naming(folder, story.title),
        story=story.line,
        ranges=ranges,
        beats=tuple(Beat(kind=one.kind, what=one.what, saw=one.saw) for one in story.shots),
        retelling=story.retelling,
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
    if sum(b - a for a, b in ranges) < STORY_LOW_S - MIN_SHOT_S:
        _blame(folder, plan, n, "there is not enough of that one left to be a story")
        return f"{folder.name} clip {n}: nothing left to render"
    # The beats describe the ranges by position, so a hand edit that changed their number has
    # thrown that away. Renders as plain hard cuts rather than guessing which beat is which.
    clip = replace(clip, ranges=ranges, beats=clip.beats if len(ranges) == len(clip.beats) else ())

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
