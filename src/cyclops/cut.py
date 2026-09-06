"""Turning a session's recording into a video somebody would watch.

A session leaves ``video.mp4`` - the panel, exactly as it was on the glass, for however long the
conversation ran. That is an archive, not a video: nobody sits through six minutes of somebody
waiting for an answer. What makes it watchable is knowing which ninety seconds of it carried the
work, and everything needed to know that is already on the card.

**That is the whole reason this module is small.** ``session.jsonl`` stamps every turn in seconds
against the same ``t0`` the video was muxed on, so there is no transcription to do, no word
timing to recover and no speaker to identify - the microphone is already on the left channel and
Cyclops on the right. The expensive half of automatic video editing was done while the
conversation was happening. What is left is choosing spans and running ffmpeg.

Asked for, never automatic. Most sessions are thirty seconds of checking a torque figure and are
not videos; the button on the session screen is somebody saying this one was worth watching.

Four files, and two of them are markers::

    cut.request   somebody pressed the button. The queue, and it is the folder
    cut.json      what was chosen: the ranges, the title, the description
    cut.ass       the captions and the two cards, as one subtitle file
    cut.mp4       the deliverable. Existing means one ffmpeg run returned zero

**Two markers rather than one is the entire retry story.** ``cut.json`` present means the model
has been asked; ``cut.mp4`` present means the encode is done. ``deploy/push.sh`` restarts the
index service on every deploy and the default ``KillMode`` takes the whole cgroup with it, so a
render *will* be killed halfway - and when it is, the request is still on the card and the plan
beside it, so the next sweep re-runs ffmpeg and never pays for the model twice. That is the same
bargain :mod:`cyclops.after` makes: the queue is the filesystem, and a crash costs a retry.

It also makes a bad cut fixable by hand, which nothing else here could: edit the ranges in
``cut.json``, delete ``cut.mp4``, and the next sweep renders what you wrote. :func:`sanitize` runs
over a hand-edited plan exactly as it runs over the model's answer, so that is safe to do.

**The model is an optimisation, not a dependency.** No key, no network, a refusing model, a reply
in the wrong shape - :func:`rules` cuts the session from its own log instead, and you still get a
video. There is no path through this module where the button lies.

Stdlib, :mod:`cyclops.card`, :mod:`cyclops.library` and :mod:`cyclops.stats` - all three of which
are themselves stdlib-only. ``openai`` is imported inside :func:`decide` rather than at the top,
the move :mod:`cyclops.after` and :mod:`cyclops.captions` already make: ``cyclops.admin.views``
imports this module to ask what state a session's video is in, and loading an SDK to answer that
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

# Not -nano, and this is the only call in the codebase that is not. Choosing which ninety seconds
# of a six-minute conversation carry it is a judgement about the whole transcript rather than a
# lookup, and it is the difference between a video and a montage of somebody saying "um".
CUT_MODEL = "gpt-5.4-mini"
# Generous, because nobody is standing in front of it: this runs niced inside the index service,
# behind a button whose honest answer is "queued". max_retries=0 still applies, for
# slug.describe_session's reason.
CUT_TIMEOUT_S = 60.0
# Five times the worst measured on the Pi - 1m47s for a 362 s source at 720p. Wide enough that a
# slow encode is never mistaken for a wedged one, tight enough that a wedged one is not forever.
RENDER_TIMEOUT_S = 900.0

CARD_S = 2.0  # the title card at the front, and the end card at the back
FADE_S = 0.4  # in from the title card, out into the end card
EDGE_FADE_S = 0.02  # 20 ms across every join, because cutting mid-waveform clicks

MIN_KEEP_S = 1.2  # anything shorter is a flicker, not a shot
MERGE_GAP_S = 0.6  # a hole this small between two keeps is a stutter; join them instead
LEAD_IN_S = 0.25  # a model picking "the sentence" always clips the first syllable
LEAD_OUT_S = 0.45
MAX_RANGES = 16
MAX_TOTAL_S = 600.0
MIN_TOTAL_S = 8.0  # under this it is not an edit; fall back to the rules
TARGET_S = 90.0  # what the prompt asks for
FPS = 15.0  # what the panel records at; boundaries are snapped to it

# The rule cut, for when the model is not available or not usable.
RULE_HEAD_S = 1.0  # before the first thing said
RULE_TAIL_S = 2.0  # after the last
RULE_GAP_S = 12.0  # a silence longer than this is dead air and comes out
RULE_KEEP_S = 1.0  # left either side of a dropped gap

MAX_TIMELINE_CHARS = 24000
MAX_LINE_CHARS = 200  # a timeline line is trimmed rather than the timeline being elided
MAX_TITLE_CHARS = 100  # YouTube's own limit
MAX_DESC_CHARS = 4500
MAX_SUB_CHARS = 84  # two readable lines at this size in a 1280-wide frame
MAX_SUB_S = 6.0
MIN_SUB_S = 1.2
SUB_CPS = 16.0  # measured against a real log: a 267-character turn ran about fifteen seconds

# 1280x720, and the panel scaled into it. 800x480 is 5:3, so x1.5 is exactly 1200x720 and nothing
# rounds - but the general form is written out because a session recorded from the camera is the
# sensor's shape and not 5:3, where the literal numbers would stretch it.
OUT_W, OUT_H = 1280, 720

# BBGGRR, not RGB. The green is the kiosk's own phosphor (--green in base.css), so the two voices
# are told apart the same way on the video as they are on the panel.
GREEN_ASS = "&H008CFF56"

_RANGE_LINE = re.compile(r"^\s*[-*]?\s*(\d+(?:\.\d+)?)\s*[-–—]\s*(\d+(?:\.\d+)?)\s*$")


# ------------------------------------------------------------------ the plan


@dataclass(frozen=True)
class Plan:
    """What was asked for and what was chosen. The whole of ``cut.json``."""

    asked: str = ""  # ISO-8601, when the button was pressed
    decided: str = ""  # ISO-8601, when the ranges were settled
    by: str = ""  # "model" | "rules" - which of the two chose them
    seconds: float = 0.0  # the source recording's length, from ffprobe
    ranges: tuple[tuple[float, float], ...] = ()
    title: str = ""
    desc: str = ""
    why: str = ""  # ffmpeg's last line, when the last attempt failed
    source: dict[str, Any] = field(default_factory=dict)  # (size, mtime_ns) of video.mp4

    @property
    def body_s(self) -> float:
        return sum(end - start for start, end in self.ranges)

    @property
    def total_s(self) -> float:
        return self.body_s + 2 * CARD_S


def read_plan(folder: Path) -> Plan | None:
    """The plan on the card, or None if there is not one worth believing.

    Bad JSON reads as no plan rather than as an error: the only thing that can produce one is a
    hand edit, and the honest response to a file somebody broke is to decide again.
    """
    try:
        raw = json.loads((folder / card.CUT_PLAN).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    got = raw.get("ranges", [])
    pairs = [r for r in got if isinstance(r, list | tuple) and len(r) == 2] if got else []
    ranges = tuple((float(a), float(b)) for a, b in pairs)
    return Plan(
        asked=str(raw.get("asked", "")),
        decided=str(raw.get("decided", "")),
        by=str(raw.get("by", "")),
        seconds=float(raw.get("seconds", 0.0) or 0.0),
        ranges=ranges,
        title=str(raw.get("title", "")),
        desc=str(raw.get("desc", "")),
        why=str(raw.get("why", "")),
        source=raw.get("source") if isinstance(raw.get("source"), dict) else {},
    )


def write_plan(folder: Path, plan: Plan) -> None:
    """Land the plan, whole or not at all - card.write_text, like every byte here."""
    body = asdict(plan) | {"ranges": [list(r) for r in plan.ranges]}
    with suppress(OSError, ValueError):
        card.write_text(folder / card.CUT_PLAN, json.dumps(body, indent=2, ensure_ascii=False))


# ------------------------------------------------------------------ state, and the queue


def state(folder: Path) -> str:
    """``"" | "asked" | "cutting" | "done" | "failed"`` - from the folder, never from a ledger.

    Cheap in the case a listing asks seventy times: a folder with a finished video is one stat,
    and one that has never been asked is two.
    """
    if card.written(folder / card.CUT):
        return "done"
    if (folder / card.CUT_REQUEST).exists():
        # The scratch file is ffmpeg's, and it exists only while ffmpeg is writing into it. A
        # SIGKILL leaves one behind, which the next attempt overwrites - so this is allowed to be
        # briefly wrong after a power cut and is never wrong for long.
        return "cutting" if card.tmp_for(folder / card.CUT).exists() else "asked"
    plan = read_plan(folder)
    return "failed" if plan is not None and plan.why else ""


def request(folder: Path) -> None:
    """Ask for a video of this session, replacing any it already has.

    Deletes rather than versions, which is what "cut it again" honestly means and what keeps
    :func:`state` one predicate instead of an mtime comparison. The plan goes with it: keeping it
    would deterministically re-render the identical file, and somebody pressing this button is
    saying the choice was wrong, not the encode.
    """
    (folder / card.CUT).unlink(missing_ok=True)
    (folder / card.CUT_PLAN).unlink(missing_ok=True)
    stamp = f"{datetime.now().astimezone():%Y-%m-%dT%H:%M:%S%z}"
    card.write_text(folder / card.CUT_REQUEST, json.dumps({"at": stamp}) + "\n")


def waiting(sessions_dir: Path) -> list[Path]:
    """Every folder with a request outstanding, oldest first.

    Oldest first so two people pressing the button in the same minute get their videos in the
    order they asked, which is the only ordering anyone could predict.
    """
    try:
        found = [
            entry
            for entry in sessions_dir.expanduser().iterdir()
            if entry.is_dir() and (entry / card.CUT_REQUEST).exists()
        ]
    except OSError:
        return []
    return sorted(found, key=lambda p: _mtime(p / card.CUT_REQUEST))


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


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


# ------------------------------------------------------------------ how long the recording is


def duration(folder: Path, records: list[dict]) -> float:
    """How long ``video.mp4`` actually runs, for the clamp in :func:`sanitize`."""
    return probe(folder, records)[0]


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
    is a true thing to write into ``cut.json`` and a useless thing to show somebody who pressed a
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


# ------------------------------------------------------------------ what the model is shown


def timeline(records: list[dict], captions: dict[str, str]) -> str:
    """The session as a list of stamped lines, which is what makes the ranges nominable.

    Not ``session.transcript_text``: that drops ``t``, and ``t`` is the entire answer here. And
    deliberately not ``slug.fit`` either - eliding the middle of the conversation would make the
    middle of the video unpickable, which is exactly the half most worth keeping. Over budget,
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
            lines.append(f"[{at:.1f}] [wrote on the panel] {_flat(record.get('text', ''))}")
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


CUT_PROMPT = """\
Below is the timeline of a recorded session between someone working on something practical -
making, fixing, or figuring a thing out - and Cyclops, the assistant helping them. The recording
is {seconds:.0f} seconds long, and every time below is a position in it, in seconds from the start.

Turn it into a short video for someone who was not there. Reply with exactly this shape and
nothing else - no preamble, no sign-off, no markdown.

TITLE: one line, under 90 characters, saying what this video is. Write it about the work, never
about the conversation: "Bled the rear brake and found the pads were glazed", not "A discussion
about brakes". No quotes, no emoji, no "In this video".

DESC: one paragraph, four to six sentences, of what happens in it - what was worked on, what was
measured or decided, what was tried, what was left open. Keep the numbers, the sizes and the part
names; they are the whole point. No bullets, no hashtags, no line breaks inside it.

KEEP: and then one range per line, and nothing after them. Each line is two numbers of seconds,
smallest first, joined by a hyphen:
KEEP:
12.5-31.0
48.0-73.5

Those ranges are what is kept; everything else is cut out. Choose them like this:

1. In order, never overlapping, and nothing shorter than three seconds.
2. Cut in the silence between turns and never inside a sentence. Start a range about half a
   second before the line you are keeping and end it about half a second after.
3. Keep what actually happened: a measurement, a number, a part name, a decision, a photograph
   of the real thing, a disagreement, a thing that did not work.
4. Drop the greeting, the false starts, the waiting, the question asked twice, and anything
   Cyclops said that was cut off before it landed.
5. Aim for about {target:.0f} seconds of finished video, and never more than {ceiling:.0f}.

If there is nothing in this session worth keeping, write KEEP: with no ranges under it.

Timeline:
{timeline}"""


def decide(shown: str, seconds: float, settings: Settings) -> tuple[str, str, tuple, bool]:
    """Ask what to keep and what to call it. ``(title, desc, ranges, asked)``. Never raises.

    ``asked`` says whether the model actually answered, so the caller can record whether these
    were its ranges or the rules', without having to guess from an empty tuple.

    Imported here rather than at the top of the module for the reason in the module docstring:
    the admin service asks this module for a folder's state on every listing and has no business
    loading an SDK to do it.
    """
    if not shown.strip() or not settings.cut or not settings.api_key:
        return "", "", (), False

    from openai import APIError, OpenAI

    # max_retries=0 for slug.describe_session's reason, and one more: three retries of a
    # sixty-second budget is three minutes of an index service not indexing.
    client = OpenAI(api_key=settings.api_key, timeout=CUT_TIMEOUT_S, max_retries=0)
    try:
        response = client.responses.create(
            model=CUT_MODEL,
            reasoning={"effort": "low"},  # unlike a folder name, this is a judgement
            input=CUT_PROMPT.format(
                seconds=seconds, target=TARGET_S, ceiling=MAX_TOTAL_S, timeline=shown
            ),
        )
        answer = (getattr(response, "output_text", "") or "").strip()
    except (APIError, OSError, ValueError):
        return "", "", (), False
    except Exception:  # noqa: BLE001 - a video is never worth taking the index service down
        return "", "", (), False
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass
    title, desc, ranges = parse(answer)
    return title, desc, ranges, True


def parse(answer: str) -> tuple[str, str, tuple[tuple[float, float], ...]]:
    """The three parts, each read independently.

    slug.parse's doctrine: a reply that goes wrong in one place should lose that one part rather
    than all three. A title with no ranges under it still gives a video, because the rules will
    cut it; ranges with no title still give a video, because ``summary.md`` names it.
    """
    title, desc, ranges = "", "", []
    for line in answer.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("TITLE:"):
            title = stripped[6:].strip().strip('"')
        elif upper.startswith("DESC:") or upper.startswith("DESCRIPTION:"):
            desc = stripped.split(":", 1)[1].strip()
        elif upper.startswith("KEEP:"):
            rest = stripped.split(":", 1)[1].strip()
            if rest:
                found = _RANGE_LINE.match(rest)
                if found:
                    ranges.append((float(found.group(1)), float(found.group(2))))
        else:
            found = _RANGE_LINE.match(stripped)
            if found:
                ranges.append((float(found.group(1)), float(found.group(2))))
            elif desc and not stripped.startswith(("TITLE", "KEEP")) and stripped:
                desc = f"{desc} {stripped}".strip()
    return _clip(title, MAX_TITLE_CHARS), _clip(desc, MAX_DESC_CHARS), tuple(ranges)


def _clip(text: str, limit: int) -> str:
    """One line, no control characters, cut at a word boundary if it has to be cut."""
    flat = " ".join(str(text or "").replace(" ", " ").split())
    if len(flat) <= limit:
        return flat
    return flat[:limit].rsplit(" ", 1)[0].rstrip(",.;:") + "…"


# ------------------------------------------------------------------ making it renderable


def sanitize(raw: Any, seconds: float) -> tuple[tuple[float, float], ...]:
    """Whatever was suggested, turned into ranges that cannot render badly.

    Run over the model's answer *and* over whatever is read back out of ``cut.json``, so that a
    plan somebody edited by hand is held to the same rules. ``cut.json`` stores what comes out of
    here, never what went in, so the file always says exactly what was rendered.

    Returns () when nothing survives, which the caller reads as "use the rules".
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
        # breath touch, so step three joins them instead of leaving a frame of black between.
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

    if len(merged) > MAX_RANGES:
        merged = sorted(sorted(merged, key=lambda p: p[1] - p[0], reverse=True)[:MAX_RANGES])

    kept: list[tuple[float, float]] = []
    total = 0.0
    for start, end in merged:
        if total >= MAX_TOTAL_S:
            break
        end = min(end, start + (MAX_TOTAL_S - total))
        if end - start < MIN_KEEP_S:
            continue
        kept.append((_snap(start), _snap(end)))
        total += end - start

    if sum(end - start for start, end in kept) < MIN_TOTAL_S:
        return ()  # not an edit; the caller falls back to the rules
    return tuple(kept)


def _snap(t: float) -> float:
    """To the nearest frame. An unsnapped boundary makes concat's arithmetic unreproducible."""
    return round(t * FPS) / FPS


def rules(records: list[dict], seconds: float) -> tuple[tuple[float, float], ...]:
    """A cut from the log alone, for when the model is not there or not usable.

    This is what makes the model an optimisation rather than a dependency. It is not as good as a
    chosen cut and does not try to be: it takes off the silence at either end and drops the long
    gaps where nothing was said, which is most of what an unedited recording is.
    """
    spoken = [
        float(r.get("t", 0.0) or 0.0) for r in records if r.get("type") in {"you", "cyclops"}
    ]
    if not spoken or seconds <= 0:
        return ()
    start = max(0.0, min(spoken) - RULE_HEAD_S)
    end = min(seconds, max(spoken) + RULE_TAIL_S)
    if end - start < MIN_KEEP_S:
        return ()

    marks = sorted(t for t in spoken if start <= t <= end)
    kept: list[list[float]] = [[start, end]]
    for before, after in zip(marks, marks[1:], strict=False):
        if after - before > RULE_GAP_S:
            hole_from, hole_to = before + RULE_KEEP_S, after - RULE_KEEP_S
            last = kept[-1]
            if last[0] < hole_from < hole_to < last[1]:
                kept[-1] = [last[0], hole_from]
                kept.append([hole_to, last[1]])
    # Back through the same door, so the rules cannot produce something the model could not.
    return sanitize([list(r) for r in kept], seconds)


# ------------------------------------------------------------------ what to call it


def naming(folder: Path, title: str, desc: str, seconds: float) -> tuple[str, str]:
    """A title and a description that are never empty, whatever the model did or did not say.

    The fallbacks below the model's own answer are ``library._read``'s chain, reused rather than
    written again, so the row in the list and the words on the title card cannot disagree.
    """
    was_title, was_summary = library._summary(folder)  # noqa: SLF001 - see the docstring
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
    if not desc:
        desc = was_summary
    if not desc:
        stamp = f"{when:%-d %B %Y}" if when else "an afternoon"
        desc = f"Recorded on {stamp}. {_span(seconds)} of it, cut down from the whole session."
    return _clip(title, MAX_TITLE_CHARS), _clip(desc, MAX_DESC_CHARS)


def _span(seconds: float) -> str:
    minutes, rest = divmod(int(max(0.0, seconds)), 60)
    return f"{minutes}m {rest:02d}s" if minutes else f"{rest}s"


# ------------------------------------------------------------------ the subtitle file


def shift(t: float, ranges: tuple[tuple[float, float], ...], lead: float = CARD_S) -> float | None:
    """Where a moment of the recording ends up in the finished video, or None if it was cut out.

    The captions stand or fall on this. A record's ``t`` is a position in ``video.mp4``; after the
    trims, the concat and the title card it is somewhere else entirely, and a caption at the wrong
    somewhere is worse than no caption at all.
    """
    out = lead
    for start, end in ranges:
        if t < start:
            return None
        if t < end:
            return out + (t - start)
        out += end - start
    return None


def _clip_into(a: float, b: float, ranges, lead: float = CARD_S):
    """A span of the recording, as the spans of the video it survives into.

    A caption straddling a join is clipped to the part that was kept rather than dropped whole;
    a sliver left after clipping is dropped rather than flashed for two frames.
    """
    out, at = [], lead
    for start, end in ranges:
        if b > start and a < end:
            piece_a = max(a, start) - start + at
            piece_b = min(b, end) - start + at
            if piece_b - piece_a >= 0.35:
                out.append((piece_a, piece_b))
        at += end - start
    return out


def _ass_time(t: float) -> str:
    t = max(0.0, t)
    hours, rest = divmod(t, 3600.0)
    minutes, seconds = divmod(rest, 60.0)
    return f"{int(hours)}:{int(minutes):02d}:{seconds:05.2f}"


def _ass_text(text: str) -> str:
    """Plain text, made safe for a subtitle line.

    Three substitutions and that is the whole escaping surface, which is the argument for putting
    the text in a file at all: ``drawtext`` would need this done against ffmpeg's filter grammar,
    its own escaping and ``%`` expansion, on a string a model chose.
    """
    flat = " ".join(str(text or "").split())
    return flat.replace("\\", "/").replace("{", "(").replace("}", ")")


def _chunks(text: str, limit: int = MAX_SUB_CHARS) -> list[str]:
    """One turn, split into caption-sized pieces on word boundaries."""
    words, out, line = text.split(), [], ""
    for word in words:
        if line and len(line) + 1 + len(word) > limit:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out or [""]


def script(records: list[dict], plan: Plan, when: datetime | None) -> str:
    """The title card, the captions and the end card, as one ASS file.

    One file rather than a filter per caption: libass wraps, centres and positions, and forty
    ``drawtext`` filters would each evaluate an ``enable`` expression on every frame.

    ``BorderStyle: 3`` - an opaque box rather than an outline - because the body of this video is
    a screen recording of a green-on-black interface, and outlined white text over it is
    unreadable in exactly the places somebody would be trying to read it.
    """
    head = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {OUT_W}",
        f"PlayResY: {OUT_H}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Card,DejaVu Sans,52,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
        "1,0,0,0,100,100,0,0,1,0,0,5,90,90,0,1",
        "Style: Foot,DejaVu Sans,26,&H00909090,&H000000FF,&H00000000,&H00000000,"
        "0,0,0,0,100,100,0,0,1,0,0,5,90,90,0,1",
        "Style: You,DejaVu Sans,32,&H00FFFFFF,&H000000FF,&H00000000,&H78000000,"
        "0,0,0,0,100,100,0,0,3,2,0,2,90,90,42,1",
        f"Style: Cyc,DejaVu Sans,32,{GREEN_ASS},&H000000FF,&H00000000,&H78000000,"
        "0,0,0,0,100,100,0,0,3,2,0,2,90,90,42,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    events: list[str] = []

    def say(start: float, end: float, style: str, text: str, pos: str = "") -> None:
        events.append(
            f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},{style},,0,0,0,,{pos}{text}"
        )

    total = plan.total_s
    stamp = f"{when:%-d %B %Y}" if when else ""
    foot = " · ".join(part for part in (stamp, _span(plan.body_s)) if part)

    say(0.0, CARD_S, "Card", _ass_text(plan.title), "{\\pos(640,320)}")
    if foot:
        say(0.0, CARD_S, "Foot", _ass_text(foot), "{\\pos(640,420)}")

    order = [r for r in records if r.get("type") in {"you", "cyclops"}]
    for index, record in enumerate(order):
        text = _ass_text(record.get("text", ""))
        if not text:
            continue
        at = float(record.get("t", 0.0) or 0.0)
        if record.get("type") == "you":
            ends = at + float(record.get("dur", 0.0) or 0.0)
        else:
            # Cyclops's records carry no duration - they are stamped when generation began. The
            # next thing that happened is the upper bound, and how long the words take to say is
            # the estimate, whichever is shorter.
            nxt = order[index + 1] if index + 1 < len(order) else None
            ceiling = float(nxt.get("t", at)) if nxt else at + MAX_SUB_S
            ends = min(ceiling, at + len(text) / SUB_CPS)
        ends = max(at + MIN_SUB_S, ends)
        style = "You" if record.get("type") == "you" else "Cyc"
        pieces = _chunks(text)
        for piece, (piece_at, piece_ends) in zip(
            pieces, _spread(at, ends, pieces), strict=False
        ):
            for from_t, to_t in _clip_into(piece_at, piece_ends, plan.ranges):
                say(from_t, min(to_t, from_t + MAX_SUB_S), style, piece)

    say(total - CARD_S, total, "Card", "CYCLOPS", "{\\pos(640,340)}")
    say(total - CARD_S, total, "Foot", "recorded on the bench", "{\\pos(640,430)}")
    return "\n".join([*head, *events]) + "\n"


def _spread(start: float, end: float, pieces: list[str]) -> list[tuple[float, float]]:
    """One turn's span, shared between its caption chunks in proportion to their length.

    In proportion, and not evenly. A turn that splits into a full line and the one word that
    would not fit gave that word an equal share of the time, so the line flicked past and
    "calipers." sat there for four seconds. Reading time is about how much there is to read.
    """
    if len(pieces) <= 1:
        return [(start, end)]
    total = sum(len(piece) for piece in pieces) or 1
    out, at = [], start
    for piece in pieces:
        share = max(MIN_SUB_S, (end - start) * len(piece) / total)
        out.append((at, at + share))
        at += share
    return out


# ------------------------------------------------------------------ the render


@dataclass(frozen=True)
class Cut:
    """What one render came to. ``why`` is ffmpeg's last line, and empty when it worked."""

    ok: bool
    why: str = ""


def cut_command(plan: Plan, out_name: str) -> list[str]:
    """The one ffmpeg call that makes the video. Run with ``cwd`` set to the session folder.

    **Relative names throughout, and that is a safety property rather than a convenience.**
    ``subtitles=`` is the only argument here whose value ffmpeg parses as filter grammar, where
    ``:`` ``'`` ``\\`` ``,`` ``[`` ``]`` all mean something - and a session folder's name comes
    partly from a model. A bare ``cut.ass`` in the process's own working directory has none of
    those characters and cannot acquire one. That is slugify's argument in a different costume:
    restrict rather than escape.

    Lifted out of :func:`render` for ``record.mux_command``'s reason - so the same command can be
    pasted into ssh and run against the same three files, rather than a second copy of it that
    quietly rots out of step.

    ``tpad`` puts the two black cards on rather than concatenating ``color=`` sources, and that is
    measured rather than stylistic: ``concat`` requires every input link to agree on width,
    height, SAR, pixel format, sample rate and layout, so black cards that way need two more
    inputs and eight filters keeping them in step, any one of which being wrong is an "Input link
    parameters do not match" from a filtergraph you are reading over ssh. ``tpad`` pads the stream
    that already exists and inherits all of it by construction.
    """
    graph: list[str] = []
    for n, (start, end) in enumerate(plan.ranges):
        fade_out = max(0.0, (end - start) - EDGE_FADE_S)
        graph.append(f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS[v{n}]")
        graph.append(
            f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS,"
            # Cutting mid-waveform clicks. Twenty milliseconds either side of every join is
            # inaudible as a fade and is the difference between "edited" and "chopped".
            f"afade=t=in:st=0:d={EDGE_FADE_S},afade=t=out:st={fade_out:.3f}:d={EDGE_FADE_S}[a{n}]"
        )
    chain = "".join(f"[v{n}][a{n}]" for n in range(len(plan.ranges)))
    graph.append(f"{chain}concat=n={len(plan.ranges)}:v=1:a=1[vb][ab]")
    graph.append(
        f"[vb]scale={OUT_W - 80}:{OUT_H}:force_original_aspect_ratio=decrease,"
        f"pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,"
        f"tpad=start_duration={CARD_S}:stop_duration={CARD_S}:color=black,"
        f"subtitles={card.CUT_SUBS},"
        f"fade=t=in:st=0:d={FADE_S},fade=t=out:st={plan.total_s - FADE_S:.3f}:d={FADE_S},"
        "format=yuv420p[v]"
    )
    graph.append(
        # The microphone is on the left and Cyclops on the right, which is right for an archive
        # and unlistenable on headphones. This flattens one derived copy; record.py's note about
        # not flattening every recording ever made still stands.
        "[ab]pan=stereo|c0=0.5*c0+0.5*c1|c1=0.5*c0+0.5*c1,"
        f"adelay={int(CARD_S * 1000)}|{int(CARD_S * 1000)},apad=pad_dur={CARD_S},"
        "loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000[a]"
    )
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", card.VIDEO,
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


def render(folder: Path, plan: Plan, sessions_dir: Path) -> Cut:
    """Make the video, landing the file only if ffmpeg said it worked.

    ``record.mux``'s discipline and for its incident: the encode happens under a scratch name and
    is renamed into place, so ``cut.mp4`` existing means *a render returned zero* - which is the
    only reading under which the retry above can be trusted.

    ``Popen`` rather than ``subprocess.run`` is a deliberate deviation from the house pattern, for
    one reason: a two-minute render will routinely overlap a conversation that starts thirty
    seconds into it, so being polite only at the start is not being polite. This watches, and
    stands down. A stand-down leaves the request where it is - it is a pause, not a failure.
    """
    out = folder / card.CUT
    tmp = card.tmp_for(out)
    try:
        proc = subprocess.Popen(  # noqa: S603 - the command is ours, from cut_command
            cut_command(plan, tmp.name),
            cwd=folder, stdin=subprocess.DEVNULL,
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
        # request is still there under the new name, so the next sweep picks it up - the same
        # best-effort answer captions.fill gives to the same race.
        tmp.unlink(missing_ok=True)
        return Cut(False, f"{type(exc).__name__}: {exc}")
    return Cut(True)


# ------------------------------------------------------------------ the sweep


def one(settings: Settings) -> str:
    """Make at most one video, and say what happened. What the index service calls.

    One per wake rather than all of them: the finished file lands in a folder the index service
    is watching, which rings its bell, which schedules the next sweep, which makes the next one.
    The queue drains itself and no single wake ever burns two renders back to back.
    """
    if not settings.cut:
        return ""
    sessions_dir = settings.sessions_dir.expanduser()
    queue = waiting(sessions_dir)
    if not queue:
        return ""
    held_up = busy(sessions_dir)
    if held_up:
        return f"not cutting yet - {held_up}"

    with _held() as mine:
        if not mine:
            return ""  # somebody else is rendering; the next sweep is soon enough
        folder = queue[0]
        if not card.written(folder / card.VIDEO):
            (folder / card.CUT_REQUEST).unlink(missing_ok=True)
            return f"cannot cut {folder.name}: it has no recording"
        return _make(folder, sessions_dir, settings)


def _make(folder: Path, sessions_dir: Path, settings: Settings) -> str:
    """Decide if it has not been decided, then render. The two stages, in one place."""
    from . import tasks

    records, _ = card.read_log(folder / card.LOG_NAME)
    plan = read_plan(folder)
    asked = _asked(folder)
    seconds, has_video = probe(folder, records)
    if not has_video:
        # Sound and no picture. Nothing here can make a video out of that, and it will still be
        # true on the next sweep, so say so plainly and take it off the queue.
        write_plan(folder, Plan(asked=asked, seconds=seconds,
                                why="that recording has sound but no picture in it"))
        (folder / card.CUT_REQUEST).unlink(missing_ok=True)
        return f"cannot cut {folder.name}: it has no picture in it"
    if plan and plan.seconds:
        seconds = plan.seconds

    if plan is None or not plan.ranges:
        # Stage one. Skipped entirely on a retry, which is what makes a deploy that kills a
        # render cost the encode and never the model call.
        task = tasks.start(f"Choosing what to keep of {folder.name}…")
        captions = _captions(folder)
        title, desc, raw, asked = decide(timeline(records, captions), seconds, settings)
        ranges = sanitize(raw, seconds)
        by = "model"
        if not ranges:
            ranges, by = rules(records, seconds), "rules"
        if not ranges:
            tasks.fail(task, "nothing worth keeping")
            (folder / card.CUT_REQUEST).unlink(missing_ok=True)
            write_plan(folder, Plan(seconds=seconds, why="there is nothing in this one to keep"))
            return f"nothing to keep in {folder.name}"
        body_s = sum(end - start for start, end in ranges)
        title, desc = naming(folder, title if asked else "", desc if asked else "", body_s)
        plan = Plan(
            asked=asked,
            decided=f"{datetime.now().astimezone():%Y-%m-%dT%H:%M:%S%z}",
            by=by, seconds=seconds, ranges=ranges, title=title, desc=desc,
            source=_source(folder),
        )
        write_plan(folder, plan)
        tasks.finish(task, f"{len(ranges)} pieces, {_span(plan.body_s)}")
    else:
        plan = replace(plan, ranges=sanitize([list(r) for r in plan.ranges], seconds), why="")
        if not plan.ranges:
            (folder / card.CUT_REQUEST).unlink(missing_ok=True)
            return f"nothing to keep in {folder.name}"

    task = tasks.start(f"Cutting the video of {plan.title}…")
    with suppress(OSError, ValueError):
        card.write_text(folder / card.CUT_SUBS, script(records, plan, library._started(folder)))  # noqa: SLF001
    done = render(folder, plan, sessions_dir)
    if not done.ok:
        tasks.fail(task, done.why)
        if done.why in {"paused for a conversation", "stopped to let the board cool"}:
            return f"{folder.name}: {done.why}"  # the request stays; this is a pause
        write_plan(folder, replace(plan, why=done.why))
        (folder / card.CUT_REQUEST).unlink(missing_ok=True)
        return f"could not cut {folder.name}: {done.why}"
    (folder / card.CUT_REQUEST).unlink(missing_ok=True)
    tasks.finish(task, _span(plan.total_s))
    return f"cut {folder.name}: {len(plan.ranges)} pieces, {_span(plan.total_s)}"


def _asked(folder: Path) -> str:
    """When the button was pressed, off the request the view left. "" if it cannot be read."""
    try:
        raw = json.loads((folder / card.CUT_REQUEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    return str(raw.get("at", "")) if isinstance(raw, dict) else ""


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
class Video:
    """One made video, as much of it as the list needs to show."""

    name: str  # the session folder, which is its id in every URL
    title: str
    desc: str
    started: str
    seconds: float  # the cut's length, not the session's
    bytes: int
    by: str  # "model" | "rules"
    state: str


def videos(sessions_dir: Path, limit: int = 200) -> list[Video]:
    """Every session with a video, or working on one, newest first.

    The unfinished ones are in the list on purpose: "cutting…" is exactly what somebody who just
    pressed the button has come to this screen to see. Folder names start with the stamp, so the
    string sort ``library._folders`` already does is chronological and there is nothing to parse.
    """
    out: list[Video] = []
    for folder in library._folders(sessions_dir):  # noqa: SLF001
        if len(out) >= limit:
            break
        how = state(folder)
        if not how:
            continue
        plan = read_plan(folder)
        when = library._started(folder)  # noqa: SLF001
        title, desc = naming(folder, plan.title if plan else "", plan.desc if plan else "", 0.0)
        out.append(
            Video(
                name=folder.name,
                title=title,
                desc=desc,
                started=when.isoformat() if when else "",
                seconds=round(plan.total_s, 2) if plan else 0.0,
                bytes=_size(folder / card.CUT),
                by=plan.by if plan else "",
                state=how,
            )
        )
    return out


def bytes_of(folder: Path) -> int:
    """How big the finished video is, or 0 where there is not one yet."""
    return _size(folder / card.CUT)


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0
