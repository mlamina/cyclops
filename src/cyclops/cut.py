"""One recorded session as one video: the whole of it, with the silence played fast.

``video.mp4`` is the panel as it was on the glass, for as long as the session ran - and most of
that is somebody working with nobody talking. This makes ``clips/1.mp4`` out of it: everything
from the first word to the last, with every stretch where neither channel is audible played at
``SPEED``. Nothing is chosen, nothing is dropped and no model is asked anything, so the same
recording always cuts the same way and a cut needs no key and no network. (The design before this
asked four models which thirty seconds told a story; the answer was a taste judgement, and it read
as random.)

The queue is the filesystem::

    clips/plan.json   present means this session has been considered. That is the whole gate
    clips/1.mp4       the video. Existing means one ffmpeg run returned zero

A render killed by a deploy leaves the plan, so the next sweep renders again. A bad cut is fixed
by hand: edit the segments in ``plan.json``, delete the ``.mp4``, and the next sweep renders what
you wrote. Stdlib plus card, library and stats only - ``admin.views`` imports this in a request.
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

# The whole recording is decoded now. A synthetic 957 s one - the card's longest - rendered in
# 16.5 s on a Mac (009 job page), so this is many times what the Pi should need, and it stays a
# minute under indexer.CUT_BUDGET_S, which must never fire first.
RENDER_TIMEOUT_S = 360.0

SPEED = 8.0        # what a quiet stretch plays at. A pick, not a measurement: 4x and 16x lost
GAP_MIN_S = 1.2    # shorter than this is a breath, and a breath sped up reads as a dropped frame
LEAD_IN_S = 0.15   # measured audio has no onset in it; this keeps the first consonant
LEAD_OUT_S = 0.25
FPS = 30.0         # what the screen records at; boundaries snap to it, and so does the output

# Silence detection - see listen(). Two floors because they are two instruments: the mic is an
# open mic in a workshop, the right channel is the speaker's own zero-filled output.
MIC_FLOOR_DB = -38
CYC_FLOOR_DB = -50
SILENCE_MIN_S = 0.20
SILENCE_TIMEOUT_S = 120.0
BRIDGE_S = 0.30    # a hole this small between two audible spans is a breath, not a pause

# What a session needs before anything is spent on it - see worth_asking. Measured, not chosen.
WORTH_YOU = 1
WORTH_CYCLOPS = 2
WORTH_SECONDS = 30.0

SETTLE_S = 600.0   # a folder with no summary is one cyclops.after may be about to rename
MAX_TITLE_CHARS = 70
# The panel's resolution, as scale/pad rather than literal numbers because a camera recording is
# the sensor's shape and not 5:3.
OUT_W, OUT_H = 800, 480
CUTTING = "Cutting the video of"  # our rows in the background ledger - see _drop_stale_rows
PAUSES = {"paused for a conversation", "stopped to let the board cool"}

_SIL = re.compile(r"silence_(start|end):\s*(-?\d+(?:\.\d+)?)")

Segment = tuple[float, float, float]  # source start, source end, how fast it plays


# ------------------------------------------------------------------ the plan


@dataclass(frozen=True)
class Clip:
    """The video, as the pieces of the recording it plays, in order."""

    title: str = ""
    ranges: tuple[Segment, ...] = ()
    why: str = ""  # why this will never be rendered. Empty means it is still owed

    @property
    def seconds(self) -> float:
        """How long the finished video runs."""
        return sum((end - start) / speed for start, end, speed in self.ranges)


@dataclass(frozen=True)
class Plan:
    """The whole of ``clips/plan.json``. ``clips: []`` takes a session off the queue for good."""

    decided: str = ""      # ISO-8601, when it was looked at
    seconds: float = 0.0   # the source recording's length, from ffprobe
    why: str = ""          # why nothing could be decided. A failure, and shown
    note: str = ""         # why nothing was looked for. Not a failure - see worth_asking
    speech: tuple[tuple[float, float], ...] = ()  # what listen() heard, kept for a hand repair
    clips: tuple[Clip, ...] = ()
    source: dict[str, Any] = field(default_factory=dict)  # (size, mtime_ns) of video.mp4


def plan_path(folder: Path) -> Path:
    return folder / card.CLIPS / card.CLIP_PLAN


def clip_path(folder: Path, n: int) -> Path:
    return folder / card.CLIPS / f"{n}.mp4"


def read_plan(folder: Path) -> Plan | None:
    """The plan on the card, or None - bad JSON included, since only a hand edit makes that."""
    try:
        raw = json.loads(plan_path(folder).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    clips = [
        Clip(str(one.get("title", "")), _numbers(one.get("ranges"), 3), str(one.get("why", "")))
        for one in (raw.get("clips") if isinstance(raw.get("clips"), list) else [])
        if isinstance(one, dict)
    ]
    return Plan(
        decided=str(raw.get("decided", "")),
        seconds=float(raw.get("seconds", 0.0) or 0.0),
        why=str(raw.get("why", "")),
        note=str(raw.get("note", "")),
        speech=_numbers(raw.get("speech"), 2),
        clips=tuple(clips),
        source=raw.get("source") if isinstance(raw.get("source"), dict) else {},
    )


def _numbers(got: Any, width: int) -> tuple:
    """Rows of finite numbers, ``width`` wide. A pair where a triple was wanted plays at 1x."""
    out = []
    for one in got if isinstance(got, list | tuple) else ():
        if not isinstance(one, list | tuple) or len(one) not in {2, width}:
            continue
        if any(isinstance(x, bool) for x in one):
            continue
        with suppress(TypeError, ValueError):
            row = tuple(float(x) for x in one) + (1.0,) * (width - len(one))
            if all(math.isfinite(x) for x in row):
                out.append(row)
    return tuple(out)


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
    state: str = ""  # "" | "asked" | "clipping" | "done" | "none" | "failed"
    made: int = 0
    total: int = 0
    failed: int = 0


def progress(folder: Path) -> Progress:
    """What has happened to this session's video, from the folder and never from a ledger."""
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
    # The scratch file exists only while ffmpeg writes it; one a SIGKILL left is swept by recover.
    if any(card.tmp_for(clip_path(folder, n)).exists() for n in range(1, total + 1)):
        return Progress("clipping", made, total, failed)
    if made + failed >= total:
        return Progress("done" if made else "failed", made, total, failed)
    return Progress("asked", made, total, failed)


def state(folder: Path) -> str:
    return progress(folder).state


@dataclass(frozen=True)
class Work:
    """The single next unit. ``index`` -1 means decide; otherwise render that clip."""

    folder: Path
    index: int = -1


def work(sessions_dir: Path) -> Work | None:
    """The next unit, newest session first, or None.

    One unit per sweep is the whole rate limit on a backfill, and every way a render can decline
    writes a ``why`` (see :func:`_render_one`), so nothing sits at the head of this walk forever.
    """
    try:
        folders = library._folders(sessions_dir)  # noqa: SLF001 - newest first, already sorted
    except OSError:
        return None
    for folder in folders:
        if not card.written(folder / card.VIDEO) or not card.written(folder / card.PAGE_NAME):
            continue
        if card.locked(folder) or not settled(folder):
            continue
        plan = read_plan(folder)
        if plan is None:
            if plan_path(folder).exists():  # broken by hand: replaced, not decided every bell
                write_plan(folder, Plan(why="the plan on the card could not be read"))
                continue
            return Work(folder)
        if plan.why:
            continue
        for n, clip in enumerate(plan.clips, start=1):
            if not clip.why and not card.written(clip_path(folder, n)):
                return Work(folder, n)
    return None


def settled(folder: Path) -> bool:
    """Has the session stopped moving? cyclops.after renames the folder as it writes summary.md.

    ``SETTLE_S`` against the recording's own mtime is the way out for one that is never named.
    """
    if card.written(folder / card.SUMMARY_NAME):
        return True
    try:
        return time.time() - (folder / card.VIDEO).stat().st_mtime > SETTLE_S
    except OSError:
        return False


def forget(folder: Path) -> None:
    """What the Find clips button does: throw the decision away so the next sweep makes it again."""
    clips = folder / card.CLIPS
    if clips.is_dir():
        for made in clips.iterdir():
            made.unlink(missing_ok=True)
    _sweep_old(folder)


@contextmanager
def _held():
    """One render at a time on this box. Non-blocking: busy means come back next sweep."""
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
    """Why this session is not worth a video, or "" when it is. Reads the log and nothing else.

    A button pressed to check the microphone costs one read and gets an empty plan, never an
    ffprobe. ``WORTH_SECONDS`` is what turns away thirty "you" turns inside four seconds.
    """
    you = [r for r in records if r.get("type") == "you"]
    cyclops = [r for r in records if r.get("type") == "cyclops"]
    if len(you) < WORTH_YOU or len(cyclops) < WORTH_CYCLOPS:
        return "not enough was said in that one"
    spoken = max((float(r.get("t", 0.0) or 0.0) for r in records), default=0.0)
    said = _logged_seconds(records)
    if spoken < WORTH_SECONDS and said < WORTH_SECONDS:
        return "that one was over too quickly to hold a moment"
    return ""


def _logged_seconds(records: list[dict]) -> float:
    tail = next((r for r in reversed(records) if r.get("type") == "end"), {})
    said = tail.get("seconds")
    return float(said) if isinstance(said, int | float) and said > 0 else 0.0


# ------------------------------------------------------------------ measuring the recording


def probe(folder: Path, records: list[dict]) -> tuple[float, bool]:
    """How long the recording runs, and whether it has a picture in it at all.

    ffprobe's number, not the log's: ``-shortest`` trims the mux to the stream that ended first.
    A camera that never produced a frame still muxes a file, and ffmpeg's answer to one is half a
    page of filtergraph - so it is asked here instead, once, and answered in a sentence.
    """
    seconds, has_video = 0.0, True
    with suppress(OSError, subprocess.SubprocessError):
        done = subprocess.run(  # noqa: S603 - the command is ours
            ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type",
             "-of", "default=noprint_wrappers=1:nokey=1", str(folder / card.VIDEO)],
            capture_output=True, timeout=30.0, check=False,
        )
        if done.returncode == 0:
            out = done.stdout.decode(errors="replace").split()
            has_video = "video" in out
            for word in out:
                with suppress(ValueError):
                    if math.isfinite(found := float(word)) and found > 0:
                        seconds = found
    if seconds:
        return seconds, has_video
    said = _logged_seconds(records)  # slightly wrong beats refusing, on a box with no ffprobe
    return said or max((float(r.get("t", 0.0) or 0.0) for r in records), default=0.0), has_video


def silence_command(channel: int) -> list[str]:
    """One channel, as ffmpeg's list of the holes in it. ``-v info``: that is where it reports."""
    floor = MIC_FLOOR_DB if channel == 0 else CYC_FLOOR_DB
    return [
        "ffmpeg", "-hide_banner", "-nostdin", "-v", "info", "-i", card.VIDEO, "-vn",
        "-af", f"pan=mono|c0=c{channel},silencedetect=n={floor}dB:d={SILENCE_MIN_S}",
        "-f", "null", "-",
    ]


def listen(folder: Path, seconds: float) -> tuple[tuple[float, float], ...]:
    """When anybody was audible, on either channel. ``()`` when ffmpeg cannot say.

    Measured on the Pi at 0.55 s for a six-minute recording: ``-vn`` never decodes the video.
    """
    if seconds <= 0:
        return ()
    spans: list[tuple[float, float]] = []
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
        spans += read_silence(done.stderr.decode(errors="replace"), seconds)
    return tuple((a, b) for a, b in _bridge(spans, BRIDGE_S))


def read_silence(text: str, seconds: float) -> tuple[tuple[float, float], ...]:
    """silencedetect's report, inverted: the spans where something was audible."""
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


def _bridge(pairs, gap: float) -> list[list[float]]:
    """Spans in, spans out, with anything closer together than ``gap`` joined."""
    merged: list[list[float]] = []
    for start, end in sorted(pairs):
        if merged and start - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


# ------------------------------------------------------------------ what plays how fast


def tighten(speech, seconds: float) -> list[list[float]]:
    """Every audible span padded so no word onset is clipped, and what then touches, merged."""
    padded = [[max(0.0, a - LEAD_IN_S), min(seconds, b + LEAD_OUT_S)] for a, b in speech if b > a]
    return _bridge(padded, 0.0)


def segments(speech, seconds: float) -> tuple[Segment, ...]:
    """The recording from the first audible moment to the last, as back-to-back segments.

    Speech plays at 1x. A quiet stretch of ``GAP_MIN_S`` or more plays at ``SPEED``; a shorter one
    stays in the 1x segment around it. Boundaries are on frames and shared, so nothing falls out.
    """
    out: list[list[float]] = []
    for a, b in tighten(speech, seconds):
        a, b = _snap(a), _snap(b)
        if out and a - out[-1][1] >= GAP_MIN_S:
            out.append([out[-1][1], a, SPEED])
        elif out:
            a = out[-1][1]
        if out and out[-1][2] == 1.0:
            out[-1][1] = b
        elif b > a:
            out.append([a, b, 1.0])
    return tuple((a, b, s) for a, b, s in out)


def _valid(ranges, seconds: float) -> tuple[Segment, ...]:
    """A plan's segments, held to what renders: in order, on frames, inside the file, never slow."""
    out: list[Segment] = []
    for a, b, speed in sorted(ranges):
        a = _snap(max(a, out[-1][1] if out else 0.0))
        b = _snap(min(b, seconds) if seconds > 0 else b)
        if b > a:
            out.append((a, b, max(1.0, speed)))
    return tuple(out)


def _snap(t: float) -> float:
    """To the nearest frame. An unsnapped boundary makes concat's arithmetic unreproducible."""
    return round(t * FPS) / FPS


def naming(folder: Path) -> str:
    """The summary's first line; where there is none yet, what the session list would show."""
    title, _ = library._summary(folder)  # noqa: SLF001 - the list's own chain, reused
    if not title:
        slug, when = library._slug(folder), library._started(folder)  # noqa: SLF001
        if slug:
            title = slug.replace("-", " ").capitalize()
        else:
            title = f"Session {when:%Y-%m-%d %H:%M}" if when else folder.name
    flat = " ".join(title.replace(" ", " ").split())
    if len(flat) <= MAX_TITLE_CHARS:
        return flat
    return flat[:MAX_TITLE_CHARS].rsplit(" ", 1)[0].rstrip(",.;:") + "…"


def _span(seconds: float) -> str:
    minutes, rest = divmod(int(max(0.0, seconds)), 60)
    return f"{minutes}m {rest:02d}s" if minutes else f"{rest}s"


# ------------------------------------------------------------------ the render


@dataclass(frozen=True)
class Cut:
    """What one render came to. ``why`` is ffmpeg's last line, and empty when it worked."""

    ok: bool
    why: str = ""


def _tempo(speed: float) -> str:
    """``atempo`` for any speed. It caps at 2.0, so 8x is three of them in a row."""
    steps = []
    while speed > 2.0 + 1e-9:
        steps.append(",atempo=2")
        speed /= 2.0
    if speed > 1.0 + 1e-9:
        steps.append(f",atempo={speed:.6g}")
    return "".join(steps)


def clip_command(clip: Clip, out_name: str) -> list[str]:
    """The one ffmpeg call that makes the video. Run with ``cwd`` set to ``<session>/clips``.

    Every value in the graph is a number computed here; no title ever reaches the parser. A fast
    segment is sped up in both streams - ``setpts`` and ``atempo`` - rather than muted, which keeps
    picture and sound locked together by construction. ``fps`` after the concat stops the fast
    parts carrying eight times the frames. No fades: nothing jumps, so nothing needs explaining.

    ``segment`` splits the recording once, rather than a ``trim`` per segment: every trim sees
    every frame, and at 400 segments that was 150 s against 11 s - see the 009 job page.
    """
    first = clip.ranges[0][0]
    ends = ([first] if first > 0 else []) + [b for _, b, _ in clip.ranges]
    stamps = "|".join(f"{t:.3f}" for t in ends)
    spare = (["h"] if first > 0 else []) + ["t"]  # the trimmed head and tail
    labels = spare[:-1] + [str(n) for n in range(len(clip.ranges))] + spare[-1:]
    graph = [f"[0:v]segment=timestamps={stamps}" + "".join(f"[p{x}]" for x in labels),
             f"[0:a]asegment=timestamps={stamps}" + "".join(f"[q{x}]" for x in labels)]
    graph += [f"[p{x}]nullsink" for x in spare] + [f"[q{x}]anullsink" for x in spare]
    for n, (_, _, speed) in enumerate(clip.ranges):
        pts = "PTS-STARTPTS" if speed == 1.0 else f"(PTS-STARTPTS)/{speed:g}"
        graph.append(f"[p{n}]setpts={pts}[v{n}]")
        graph.append(f"[q{n}]asetpts=PTS-STARTPTS{_tempo(speed)}[a{n}]")
    chain = "".join(f"[v{n}][a{n}]" for n in range(len(clip.ranges)))
    graph.append(f"{chain}concat=n={len(clip.ranges)}:v=1:a=1[vb][ab]")
    graph.append(
        f"[vb]fps={FPS:g},scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease,"
        f"pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,format=yuv420p[v]"
    )
    # Mic on the left and Cyclops on the right is right for an archive and unlistenable in one
    # earbud. This folds one derived copy, never the recording.
    graph.append("[ab]pan=stereo|c0=0.5*c0+0.5*c1|c1=0.5*c0+0.5*c1,aresample=48000[a]")
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", f"../{card.VIDEO}", "-filter_complex", ";".join(graph),
        "-map", "[v]", "-map", "[a]",
        # Two threads, not four: this must never be able to take the whole box.
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
        "-threads", "2", "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart",
        "-f", "mp4",  # the output is a scratch name; ffmpeg cannot infer the container from it
        out_name,
    ]


def busy(sessions_dir: Path) -> str:
    """Why this is not the moment to render, or "". x264 beside a live conversation drops audio."""
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
    """Make the video under a scratch name, and land it only if ffmpeg returned zero.

    ``Popen`` so it can keep watching: a conversation starting or the board getting hot stands it
    down, and a stand-down is a pause that leaves the unit owed rather than a failure.
    """
    out = clip_path(folder, n)
    tmp = card.tmp_for(out)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(  # noqa: S603 - the command is ours, from clip_command
            clip_command(clip, tmp.name), cwd=out.parent, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # notably systemd's PATH
        tmp.unlink(missing_ok=True)
        return Cut(False, f"{type(exc).__name__}: {exc}")

    deadline, stood_down = time.monotonic() + RENDER_TIMEOUT_S, ""
    while proc.poll() is None:
        temp = stats.cpu_temp_c()
        if time.monotonic() > deadline:
            stood_down = f"gave up after {RENDER_TIMEOUT_S:.0f}s"
        elif temp is not None and temp >= stats.THROTTLE_C:
            stood_down = "stopped to let the board cool"
        elif busy(sessions_dir):
            stood_down = "paused for a conversation"
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
        with suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=2.0)  # back the moment ffmpeg exits

    detail = (proc.stderr.read() if proc.stderr else b"").decode(errors="replace").strip()
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        return Cut(False, (detail.splitlines() or [f"exit {proc.returncode}"])[-1])
    try:
        card.land(tmp, out)
    except OSError as exc:  # the folder was renamed mid-encode; the plan moved with it
        tmp.unlink(missing_ok=True)
        return Cut(False, f"{type(exc).__name__}: {exc}")
    return Cut(True)


# ------------------------------------------------------------------ the sweep


def one(settings: Settings) -> str:
    """At most one unit of work, and a line saying what happened - the index service's journal.

    The lock comes before the queue, so a killed render's ledger row is closed on the next sweep
    whether or not anything is left to do.
    """
    if not settings.cut:
        return ""
    sessions_dir = settings.sessions_dir.expanduser()
    with _held() as mine:
        if not mine:
            return ""  # somebody else is rendering; the next sweep is soon enough
        _drop_stale_rows()
        todo = work(sessions_dir)
        if todo is None:
            return ""
        if held_up := busy(sessions_dir):
            return f"not clipping yet - {held_up}"
        if todo.index < 0:
            return _decide_one(todo.folder)
        return _render_one(todo.folder, todo.index, sessions_dir)


def _drop_stale_rows() -> None:
    """Close our ledger rows a killed render left open. Safe only because we hold CUT_LOCK."""
    from . import tasks

    for task in tasks.running():
        if task.what.startswith(CUTTING):
            tasks.fail(task.id, "interrupted")


def _decide_one(folder: Path) -> str:
    """Look at one session once: is it worth a video, and where is the speech in it."""
    records, _ = card.read_log(folder / card.LOG_NAME)
    if thin := worth_asking(records):
        # In ``note``, not ``why``: not looked at is not a failure, and the page must not offer
        # to try again.
        write_plan(folder, Plan(decided=_now(), note=thin))
        _sweep_old(folder)
        return f"not clipping {folder.name} - {thin}"
    seconds, has_video = probe(folder, records)
    if not has_video or seconds <= 0:
        write_plan(folder, Plan(decided=_now(), seconds=seconds,
                                why="that recording has sound but no picture in it"))
        return f"cannot clip {folder.name}: it has no picture in it"
    speech = listen(folder, seconds)
    if not (cut := segments(speech, seconds)):
        write_plan(folder, Plan(decided=_now(), seconds=seconds,
                                why="nobody could be heard anywhere in that recording"))
        return f"cannot clip {folder.name}: nobody can be heard in it"
    clip = Clip(title=naming(folder), ranges=cut)
    write_plan(folder, Plan(decided=_now(), seconds=seconds, speech=speech, clips=(clip,),
                            source=_source(folder)))
    _sweep_old(folder)
    return f"planned {folder.name}: {_span(seconds)} of recording plays in {_span(clip.seconds)}"


def _sweep_old(folder: Path) -> None:
    """What the August design left in the folder; the names stay in card.KNOWN for good."""
    for old in (card.CUT, card.CUT_PLAN, card.CUT_SUBS, card.CUT_REQUEST):
        (folder / old).unlink(missing_ok=True)


def _render_one(folder: Path, n: int, sessions_dir: Path) -> str:
    """Render one clip. Every path out of here lands a file, writes a why, or is a pause."""
    from . import tasks

    plan = read_plan(folder)
    if plan is None or n > len(plan.clips):
        return ""
    clip = replace(plan.clips[n - 1], ranges=_valid(plan.clips[n - 1].ranges, plan.seconds))
    if not clip.ranges:
        _blame(folder, plan, n, "there is nothing left in that plan to render")
        return f"{folder.name} clip {n}: nothing left to render"
    task = tasks.start(f"{CUTTING} {clip.title}…")
    began = time.monotonic()
    done = render(folder, clip, n, sessions_dir)
    if not done.ok:
        tasks.fail(task, done.why)
        if done.why in PAUSES:
            return f"{folder.name}: {done.why}"
        _blame(folder, plan, n, done.why)
        return f"could not clip {folder.name} ({n}): {done.why}"
    tasks.finish(task, _span(clip.seconds))
    return (f"clipped {folder.name}: {_span(plan.seconds)} of recording into "
            f"{_span(clip.seconds)}, rendered in {time.monotonic() - began:.0f}s")


def _blame(folder: Path, plan: Plan, n: int, why: str) -> None:
    """Write a reason into one clip, which takes it off the queue for good."""
    clips = list(plan.clips)
    clips[n - 1] = replace(clips[n - 1], why=why)
    write_plan(folder, replace(plan, clips=tuple(clips)))


def _now() -> str:
    return f"{datetime.now().astimezone():%Y-%m-%dT%H:%M:%S%z}"


def _source(folder: Path) -> dict[str, Any]:
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
    """Every finished clip, newest session first, and counts that explain an empty reel."""
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
        found += bool(plan.clips)
        when = library._started(folder)  # noqa: SLF001
        for n, clip in enumerate(plan.clips, start=1):
            path = clip_path(folder, n)
            if len(out) >= limit or not card.written(path):
                continue
            out.append(Made(
                id=f"{folder.name}/{n}", name=folder.name, n=n,
                title=clip.title or naming(folder),
                started=when.isoformat() if when else "",
                seconds=round(clip.seconds, 2), bytes=_size(path),
                src=f"/media/{folder.name}/{card.CLIPS}/{n}.mp4",
            ))
    return out, {"looked": looked, "found": found, "waiting": waiting}


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:  # removed between the listing and the stat, by Find clips
        return 0
