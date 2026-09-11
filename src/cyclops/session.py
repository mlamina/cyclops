"""One session on the card: its folder, its log, and the two files that describe it.

Every session gets a folder of its own, named for the clock and - once it ends and a small
model has read the transcript - for what it was about::

    sessions/2026-08-26_14-32-05_lego-falcon/
        session.md        the conversation, with the photos in it, for a person
        session.jsonl     the same events, one JSON object per line, for a program
        video.mp4         the recording, when there was a camera to record (kiosk only)
        photos/14-32-40_cyclops.jpg

Hyphens in the time and never colons: FAT32 and exFAT forbid ``:``, and this card is meant to
be pulled out and read on something else. The UUID lives inside the two files and never in a
name - it identifies the session, but nobody can read it.

Two rules shape everything here:

* **Logging must never be able to break a conversation.** Same contract as :mod:`cyclops.record`:
  anything that goes wrong sets :attr:`SessionLog.failed`, says so once, and every method after
  it is a no-op. A full card, a read-only mount or a bad record costs you the log, never the
  session. This is why ``__init__`` only assigns and ``__enter__`` does everything fallible.
* **Records are flushed as they land, and fsynced once, at the end.** An fsync per line would
  grind an SD card for a guarantee we do not need, and an fsync on a timer would do it from the
  session's own event loop - the one thread that must never block, because on ext4 ``data=ordered``
  a small fsync waits on the whole transaction's ordered buffers, which on a kiosk means whatever
  x264 has in flight. Do not add one. The single place worth paying is :meth:`SessionLog._sync_log`
  at the close, because that is the moment ``session.md`` is written *from these same records*,
  and a page that outlived its own log is the one inconsistency this format cannot repair.
  Everything before that is still best-effort, which is exactly why the format is
  JSON Lines rather than JSON: no shared state between lines, so :func:`read_log` drops a
  half-written last line and everything before it is still a session.
* **Everything else lands whole or not at all.** Every file this module writes goes through
  :mod:`cyclops.card`, which writes to a scratch name, fsyncs it and renames it into place. A
  power cut used to leave a zero-byte ``session.md`` wearing the name that means "this session
  finished"; now it leaves the previous state, or nothing, and recovery can tell.
* **Nothing that needs a network happens here.** Naming the folder, writing its ``summary.md``,
  noting what was said about the person and filing the session under a project are all model
  calls, and all of them used to sit between the tap that ends a session and the panel going
  dark. They happen in a detached process now - see :mod:`cyclops.after` - which is why what is
  left in this teardown is arithmetic and file writes and nothing else.

Writes arrive from three threads - the session's event loop (:meth:`SessionLog.observe`), the
kiosk's shutter thread (via :func:`note`), and the teardown - so the handle is lock-guarded.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import uuid as uuid_module
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from html import unescape
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import card
from .config import ConfigError, Settings, load_settings
from .record import SessionRecorder, mux

if TYPE_CHECKING:  # importing these for real would be a cycle - agent.py imports this module
    from openai.types.realtime import RealtimeServerEvent

    from .agent import VoiceAgent
    from .audio import Microphone, Speaker
    from .record import FrameSource
    from .slug import Description

# Re-exported rather than redefined: cyclops.card owns what a session folder is called, so that
# stats.py - which must not import this module, because it would drag OpenCV into a status page -
# can ask the same questions of the same names. Everything that imported these from here still
# works.
LOG_NAME = card.LOG_NAME
PAGE_NAME = card.PAGE_NAME
SUMMARY_NAME = card.SUMMARY_NAME
RECEIPT_NAME = card.RECEIPT_NAME
PHOTOS = card.PHOTOS
PARTS = card.PARTS
VIDEO = card.VIDEO
STAMP = card.STAMP
STAMPED = card.STAMPED
read_log = card.read_log  # moved there so triage() can use it; the name still lives here
# How many sessions one boot-time recovery hands to the projects sweep. A backlog after a
# long outage drains a few at a time rather than in one unattended burst - and it drains
# anyway, because every session end spawns a full sweep of its own.
RECOVER_FILE_LIMIT = 10

# What a new session is told about the ones before it. One paragraph of the last one, and a
# sentence each for the last few - enough to pick up a thread, small enough that it cannot
# crowd out the standing rules it is appended to.
RECAP_SESSIONS = 5
RECAP_PARAGRAPH_CHARS = 1400  # slug.MAX_SUMMARY_CHARS; a summary is never trimmed in practice
RECAP_TITLE_CHARS = 120
RECAP_MAX_CHARS = 2400  # a backstop on the block as a whole, not a budget anything plans for

# The one session being logged in this process. Set by ``__enter__`` and cleared first thing in
# ``__exit__``, exactly as ``webcam.set_live_source`` does - and for the same reason: there is
# only ever one (``SessionController._running`` blocks re-entry), and the two writers that need
# it, the agent's tool coroutine and the kiosk's shutter thread, both sit far from wherever the
# log was built. Threading a reference to either would mean a new constructor argument on
# VoiceAgent and a new attribute on Kiosk, for something there is only ever one of.
_live: SessionLog | None = None
_live_lock = threading.Lock()


def current() -> SessionLog | None:
    """The session being logged right now, or None if nothing is running."""
    return _live


def photo_target(settings: Settings, *, by: str) -> tuple[Path, str]:
    """Where the next photo goes, and what to keep it as - see :func:`cyclops.webcam._save`.

    Outside a session there is no role worth recording and nothing worth keeping forever, so
    the role comes back empty and webcam falls back to its pruning ``captures/`` archive with
    ``latest.jpg`` - which is what ``cyclops-smoke`` still asserts on.
    """
    live = current()
    if live is None:
        return settings.captures_dir, ""
    return live.photos_dir, by


def note(kind: str, **fields: Any) -> None:
    """Add one record to the live session. A no-op when there is none, so callers need no check."""
    live = current()
    if live is not None:
        live.event(kind, **fields)


class SessionLog:
    """One session's folder, and the log of what happened in it.

    Built around whatever the entry point already has - the agent for its event stream, the mic
    and speaker for the recording's two audio tracks, and a camera if one is being held open.
    Used as a context manager, so the folder is finished exactly once and on every path out: a
    normal stop, a Ctrl+C, or a session that fell over.
    """

    def __init__(
        self,
        settings: Settings,
        agent: VoiceAgent,
        *,
        entrypoint: str,  # "cli" | "ui" | "kiosk" - recorded, not inferred
        mic: Microphone | None = None,
        speaker: Speaker | None = None,
        frames: FrameSource | None = None,  # only the kiosk has one; only it gets a video.mp4
        # Somewhere to say which step of the teardown is running. Optional because only the
        # kiosk has a panel to say it on - the CLI passes nothing and is unchanged. All that is
        # left to narrate is the mux, which gets up to record.MUX_TIMEOUT_S; it used to share one
        # fixed word with the model calls that now happen in cyclops.after.
        on_phase: Callable[[str], None] | None = None,
    ) -> None:
        self.settings = settings
        self.uuid = str(uuid_module.uuid4())
        self.entrypoint = entrypoint
        self.started = datetime.now().astimezone()
        self.dir = settings.sessions_dir / self.started.strftime(STAMP)
        self.failed = ""  # non-empty once logging gave up; the session carries on regardless
        self._agent = agent
        self._mic = mic
        self._speaker = speaker
        self._frames = frames
        self._on_phase = on_phase
        self._t0 = time.monotonic()
        self._lock = threading.Lock()
        self._handle = None
        self._records: list[dict] = []
        self._recorder: SessionRecorder | None = None
        self._reported = False
        # observe() state
        self._speech: deque[tuple[float, float]] = deque(maxlen=16)
        self._speech_at = 0.0
        self._item_at: dict[str, float] = {}

    def _say(self, phase: str) -> None:
        """Tell whoever is holding a screen what this teardown is up to. Never raises."""
        if self._on_phase is not None:
            self._on_phase(phase)

    @property
    def photos_dir(self) -> Path:
        return self.dir / PHOTOS

    # ---------------------------------------------------------------- lifecycle

    def __enter__(self) -> SessionLog:
        global _live
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            self._handle = (self.dir / LOG_NAME).open("a", encoding="utf-8")
            # This folder's claim on itself, held for the whole session and dropped by the
            # kernel if we die. It is what stops `cyclops-sessions --recover` renaming a folder
            # out from under a running conversation - see cyclops.card.claim.
            card.claim(self._handle)
            self.event(
                "session",
                uuid=self.uuid,
                started=self.started.isoformat(timespec="seconds"),
                entrypoint=self.entrypoint,
                model=self.settings.model,
                voice=self.settings.voice,
                lang=self.settings.transcribe_lang,
                record=bool(self.settings.record and self._frames is not None),
            )
            self._start_recorder()
            self._agent.on_event = self.observe
        except Exception as exc:  # a dead log, never an exception into the session thread
            self._give_up(f"{type(exc).__name__}: {exc}")
        if not self.failed:
            with _live_lock:
                _live = self
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        global _live
        # Unregister first: a shutter tap from here on falls back to captures/ rather than
        # writing into a folder that is about to be renamed out from under it.
        with _live_lock:
            if _live is self:
                _live = None
        self._agent.on_event = None
        self._stop_recorder()  # says "saving the video" for itself, when there is one
        reason = _reason(exc_type)
        fields: dict[str, Any] = {
            "reason": reason,
            "seconds": self._elapsed(),
            # Only the ones that landed, so this agrees with card.triage counting files in
            # photos/: an edit that failed leaves a record saying so and no picture.
            "photos": sum(
                1 for r in self._records if r.get("type") == "photo" and r.get("file")
            ),
        }
        if reason == "error" and exc is not None:
            fields["error"] = f"{type(exc).__name__}: {exc}"
        # One phrase over the three steps that follow rather than one each: the fsync is the only
        # one of them that takes measurable time, and the other two are a file write apiece.
        self._say("writing the notes…")
        self.event("end", **fields)
        self._sync_log()  # the records are on the card before anything is derived from them
        self._write_page()  # written last, which is what makes its presence mean "finished"
        self._report()
        # Closing drops this folder's flock, and it comes *before* the hand-off rather than after
        # everything else: the child's first act is to rename this folder, and card.triage reads
        # a folder that still claims itself as live and leaves it alone. Nothing writes after the
        # end record - `_live` was cleared at the top, so note() is already a no-op - which is
        # what makes it safe to have kept the handle open this long.
        self._close_handle()
        # Nothing happened in it, so there is nothing for the child to name, remember or file.
        if not self._discard():
            self._after()  # named, remembered and filed in another process, on its own time

    def _start_recorder(self) -> None:
        """Tap the mic and the speaker for this session's video, if there is a camera for one."""
        if self._frames is None or not self.settings.record:
            return
        if self._mic is None or self._speaker is None:
            return
        recorder = SessionRecorder(
            self._frames,
            self.dir,
            fps=self.settings.record_fps,
            width=self.settings.record_width,
        )
        if not recorder.start():  # it has said why; a session is never blocked on recording
            return
        self._mic.on_block = recorder.on_mic_block
        self._speaker.on_block = recorder.on_speaker_block
        self._recorder = recorder

    def _stop_recorder(self) -> None:
        recorder, self._recorder = self._recorder, None
        if recorder is None:
            return
        # Named on the panel because this is the long pole of a teardown - ffmpeg gets up to
        # record.MUX_TIMEOUT_S, and even the usual couple of seconds is a couple of seconds the
        # caption used to spend saying nothing more useful than "closing the link".
        self._say("saving the video…")
        finished = recorder.stop()
        if finished is not None:
            self.event("video", file=finished.name, seconds=self._elapsed())
            print(f"· recorded to {finished}", flush=True)
        elif recorder.failed:
            self.event("video", error=recorder.failed)

    def _write_page(self) -> None:
        if self.failed:
            return
        try:
            card.write_text(self.dir / PAGE_NAME, render_markdown(self._records))
        except OSError as exc:
            self._give_up(f"{type(exc).__name__}: {exc}")

    def _discard(self) -> bool:
        """Remove this folder when nothing happened in it. Whether it went.

        The cheap gate, at the only moment it is free. ``session``, ``video`` and ``end`` are
        records *about* a session; ``video.mp4`` and ``session.md`` are made *from* one. A folder
        holding nothing but those held a button press - there is nothing in it for a model to be
        asked about, for a sweep to file, or for anybody to come back to - so it goes here rather
        than being listed forever as a session that happened.

        Asked through :func:`cyclops.card.triage` rather than off ``self._records``, though both
        are to hand: two answers to "what is in this folder" is how they start to disagree. Which
        is also why this runs after :meth:`_close_handle` - a folder still holding the flock on
        its own log reads as ``live``, and triage refuses to judge one of those.

        Here and not in :mod:`cyclops.after` because that child is behind
        :func:`cyclops.after.wanted`: a box with no key never starts one, and this costs nothing
        and must not be something a missing key switches off. Nor inside :meth:`_after`, whose
        blanket ``except`` would swallow a deletion that failed.

        One race, not worth a lock but worth knowing: :func:`photo_target` hands the shutter
        thread ``photos_dir`` before ``_live`` is cleared, and a write lands its parent
        directories. A tap resolving its target just before that and landing its bytes just after
        the ``rmdir`` here would leave a folder with one picture and no log. It would be reported
        as unfinished forever - as any photos-only folder already is - rather than losing
        anything, which is the right direction for the accident to fall in.
        """
        if self.failed:
            return False  # a log we could not write is not one to judge a folder by
        try:
            if card.triage(self.dir).verdict != "empty":
                return False
            said = _remove(self.dir)
        except OSError as exc:  # noqa: BLE001 - a folder we could not read is a folder we keep
            said = f"could not tell whether anything survived ({exc})"
        print(f"· {self.dir.name}: {said}", flush=True)
        return said.endswith("removed")

    def _after(self) -> None:
        """Hand the folder to a detached child and let go of it - see :mod:`cyclops.after`.

        Everything a finished session still wants doing needs a model: a name, the paragraph in
        ``summary.md``, whatever it said about the person, a place in ``projects/``. None of it
        needs doing *now*. A person who taps to end a session has said they are done, and made
        the whole of it - three round trips - into somebody's wait.

        Spawned after :meth:`_close_handle`, which is the one ordering constraint left in this
        teardown: a folder still holding the flock on its own log reads as live to
        :func:`cyclops.card.triage`, and renaming this folder is the first thing the child does.

        Nothing is waited on and nothing can raise. The folder is complete and readable without
        any of it, and a child that never started is not a failed session - it leaves a folder
        with no name and no ``project.md``, which is exactly how the next one knows to pick this
        session up. The absence of the files is the retry queue.
        """
        if self.failed:
            return
        try:
            from . import after  # imported here, like slug, so a keyless box still runs

            if after.wanted(self.settings):
                after.spawn(self.dir, self.settings)
        except Exception:  # noqa: BLE001 - a hand-off that did not start never reaches teardown
            pass

    def _sync_log(self) -> None:
        """Put the records on the card. Once, at the end - see the module docstring.

        This is the one fsync a session pays, and it is here because the next line derives
        ``session.md`` from these very records. It is written atomically and durably; a page that
        outlived its own log would be the one inconsistency this format cannot repair.
        ``summary.md`` is derived from them too, in another process and out of the file this
        fsync has just made durable - the same guarantee, one step further along.
        """
        with self._lock:
            handle = self._handle
            if handle is None or handle.closed:
                return
            try:
                handle.flush()
                os.fsync(handle.fileno())
            except (OSError, ValueError):
                pass  # the log is what it is by now; never fail a teardown over it

    def _close_handle(self) -> None:
        with self._lock:
            handle, self._handle = self._handle, None
        if handle is not None:
            try:
                handle.close()  # and with it, the flock this folder held on itself
            except OSError:
                pass  # nothing useful to do about it while tearing down

    # ---------------------------------------------------------------- writing

    def _elapsed(self) -> float:
        return round(time.monotonic() - self._t0, 2)

    def event(self, kind: str, **fields: Any) -> None:
        """Append one record, from whatever thread you are on. Never raises."""
        self._write(self._elapsed(), kind, fields)

    def _write(self, at: float, kind: str, fields: dict[str, Any]) -> None:
        if self.failed:
            return
        record = {"t": at, "type": kind, **fields}
        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return  # a record we cannot serialise is dropped, not a reason to stop logging
        with self._lock:
            self._records.append(record)
            handle = self._handle
            if handle is None or handle.closed:
                return
            try:
                handle.write(line + "\n")
                handle.flush()  # flushed, deliberately not fsynced - see the module docstring
            except OSError as exc:
                self._give_up(f"{type(exc).__name__}: {exc}")

    def _give_up(self, message: str) -> None:
        self.failed = self.failed or message
        self._report()

    def _report(self) -> None:
        if self.failed and not self._reported:
            self._reported = True
            print(f"· [session] {self.failed}", file=sys.stderr, flush=True)

    # ---------------------------------------------------------------- the agent's events

    def observe(self, event: RealtimeServerEvent) -> None:
        """The :attr:`cyclops.agent.VoiceAgent.on_event` hook."""
        if self.failed:
            return
        try:
            self._observe(event)
        except Exception as exc:  # a malformed event must not reach the session's event loop
            self._give_up(f"observe: {type(exc).__name__}: {exc}")

    def _observe(self, event: RealtimeServerEvent) -> None:
        match event.type:
            case "input_audio_buffer.speech_started":
                self._speech_at = self._elapsed()
            case "input_audio_buffer.speech_stopped":
                # Transcription runs asynchronously and can land *after* the reply it caused, so
                # a user turn is stamped when it was spoken rather than when its text arrived -
                # otherwise the page shows Cyclops answering before you asked.
                self._speech.append((self._speech_at, self._elapsed()))
            case "conversation.item.input_audio_transcription.completed":
                text = (event.transcript or "").strip()
                if not text:
                    return
                now = self._elapsed()
                started, stopped = self._speech.popleft() if self._speech else (now, now)
                self._write(
                    started,
                    "you",
                    {
                        "text": text,
                        "item": getattr(event, "item_id", None),
                        "dur": round(max(0.0, stopped - started), 2),
                    },
                )
            case "conversation.item.input_audio_transcription.failed":
                self.event(
                    "transcript_failed",
                    item=getattr(event, "item_id", None),
                    error=event.error.message,
                )
            case "response.output_audio.delta":
                # Generation time, not playback time - the speaker buffers, so this runs a few
                # hundred ms early. Close enough to scrub the video by; not worth correcting.
                self._item_at.setdefault(event.item_id, self._elapsed())
            case "response.output_audio_transcript.done":
                text = (event.transcript or "").strip()
                if not text:
                    return
                fields: dict[str, Any] = {"text": text, "item": event.item_id}
                if event.item_id in self._agent.interrupted_item_ids:
                    fields["interrupted"] = True  # this is more than was ever spoken
                self._write(self._item_at.get(event.item_id, self._elapsed()), "cyclops", fields)
            case "response.done":
                self._item_at.clear()
            case "error":
                self.event(
                    "error",
                    code=event.error.code,
                    etype=event.error.type,
                    message=event.error.message,
                )


def _reason(exc_type: type[BaseException] | None) -> str:
    """Why the session ended, from however it left the ``with`` block."""
    if exc_type is None:
        return "done"
    name = exc_type.__name__
    if name == "CancelledError":
        return "stopped"  # SessionController.stop() cancels the task; this is the normal path
    if name == "KeyboardInterrupt":
        return "quit"
    return "error"


# ------------------------------------------------------------------ reading it back


def transcript_text(records: list[dict], limit: int | None = None) -> str:
    """Just the dialogue, in order - the whole of it, which is what describing one needs.

    Uncapped by default. It used to be trimmed to its first few thousand characters, which was
    right when all that came of it was a folder name: a conversation says what it is about in
    its opening minute. A summary is mostly about the other end - what was settled last, what
    was left open - so the trimming now happens in :func:`cyclops.slug.describe_session`, which
    takes it out of the middle and keeps both.
    """
    lines = []
    for record in records:
        kind = record.get("type")
        if kind == "you":
            lines.append(f"User: {record.get('text', '')}")
        elif kind == "cyclops":
            lines.append(f"Cyclops: {record.get('text', '')}")
    text = "\n".join(lines).strip()
    return text if limit is None else text[:limit].strip()


# ------------------------------------------------------------------ what came before


@dataclass(frozen=True)
class Recap:
    """What a new session is handed about the ones before it."""

    text: str = ""  # appended to the model's instructions; "" means the section is left out
    note: str = ""  # one line for the console, so what was handed over is never a mystery

    def __bool__(self) -> bool:
        return bool(self.text)


def read_summary(folder: Path) -> tuple[str, str]:
    """``(title, paragraph)`` out of a folder's ``summary.md``, or two empty strings.

    A folder without one is a session that is unfinished, was never described, or - the case
    that matters most here - is the session running *right now*: its directory exists from the
    moment it starts and its summary only when it ends. So this doubles as the filter that
    keeps a session out of its own recap, with no clock or path comparison to get wrong.
    """
    try:
        text = (folder / SUMMARY_NAME).read_text(encoding="utf-8")
    except OSError:
        return "", ""
    title, body = "", []
    for line in text.splitlines():
        stripped = line.strip()
        if not title:
            if stripped.startswith("#"):
                title = " ".join(stripped.lstrip("#").split())
            continue
        body.append(stripped)
    return title, " ".join(" ".join(body).split())


def _trim(text: str, limit: int) -> str:
    """One line, collapsed and cut at a word boundary."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rsplit(" ", 1)[0] + "…"


def _started(folder: Path) -> datetime | None:
    """When a session began, read off the timestamp its folder name still starts with."""
    try:
        return datetime.strptime(folder.name[: len("0000-00-00_00-00-00")], STAMP)
    except ValueError:
        return None


def _ago(when: datetime | None, now: datetime) -> str:
    """How long ago, in the words someone would actually say out loud."""
    if when is None:
        return "at some point"
    days = (now.date() - when.date()).days
    if days <= 0:
        return "earlier today"
    if days == 1:
        return "yesterday"
    if days < 7:
        return f"{days} days ago"
    weeks = days // 7
    return "a week ago" if weeks == 1 else f"{weeks} weeks ago"


def _ran_for(folder: Path) -> str:
    """How long a finished session lasted, or ``""`` if its log will not say."""
    records, _ = read_log(folder / LOG_NAME)
    tail = next((r for r in reversed(records) if r.get("type") == "end"), {})
    span = _span(tail.get("seconds"))
    return "" if span == "—" else span


def recent_context(settings: Settings) -> Recap:
    """The last session in a paragraph, and the last few in a sentence each.

    Read off the card at the start of every session, so a conversation begins knowing where the
    previous one got to. Only folders that have a ``summary.md`` count, which is what makes this
    cheap: a handful of small reads, and only the newest session's log is opened at all.
    """
    picked: list[tuple[Path, str, str]] = []
    for folder in reversed(_folders(settings.sessions_dir)):
        title, paragraph = read_summary(folder)
        if not title:
            continue
        picked.append((folder, title, paragraph))
        if len(picked) >= RECAP_SESSIONS:
            break
    if not picked:
        return Recap(note="nothing yet - this is the first session")

    now = datetime.now()
    folder, title, paragraph = picked[0]
    when = _started(folder)
    ran = _ran_for(folder)
    said = [_ago(when, now)]
    if when is not None:
        said.append(f"{when:%A %-d %B}")
    if ran:
        said.append(ran)
    # The title stands in when a summary somehow has no paragraph under it - a sentence about
    # last time is still worth more to the next conversation than silence about it.
    lines = [f"Last session ({', '.join(said)}):", _trim(paragraph or title, RECAP_PARAGRAPH_CHARS)]

    if len(picked) > 1:
        lines += ["", f"The last {len(picked)} sessions, oldest first:"]
        for earlier, headline, _ in reversed(picked):
            began = _started(earlier)
            date = f"{began:%a %-d %b}: " if began is not None else ""
            lines.append(f"- {date}{_trim(headline, RECAP_TITLE_CHARS)}")

    note = f"last session {said[0]}" + (f" ({ran})" if ran else "")
    if (others := len(picked) - 1) > 0:
        note += f" + {others} earlier headline{'' if others == 1 else 's'}"
    return Recap(text="\n".join(lines).strip()[:RECAP_MAX_CHARS], note=note)


def _clock(seconds: object) -> str:
    """``m:ss`` from the start of the session - which is the video's own timeline."""
    if not isinstance(seconds, int | float):
        return "?:??"
    total = max(0, int(seconds))
    return f"{total // 60}:{total % 60:02d}"


def _span(seconds: object) -> str:
    if not isinstance(seconds, int | float) or seconds <= 0:
        return "—"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m {secs:02d}s"


def _shot_time(file: str) -> str:
    """``14-32-40_you.jpg`` -> ``14:32:40``, so a photo's caption ties it to a file on the card."""
    stem = Path(file).stem.split("_")[0]
    return stem.replace("-", ":") if re.fullmatch(r"\d{2}-\d{2}-\d{2}", stem) else stem


def render_markdown(records: list[dict], *, dropped: int = 0, slug: str = "") -> str:
    """The whole page, from records alone.

    Pure on purpose: the live close path and ``cyclops-sessions --fix`` both call this, which is
    the only way the two can be guaranteed to agree about what a session looked like.

    The one thing the records cannot carry is the name, because a session is named after it has
    written its last one - so :func:`describe` passes the slug it just chose and renders the page
    again. Older logs wrote the name into their ``end`` record, back when naming happened in the
    teardown, and those still title themselves from it.
    """
    head = next((r for r in records if r.get("type") == "session"), {})
    tail = next((r for r in reversed(records) if r.get("type") == "end"), {})
    video = next((r for r in records if r.get("type") == "video" and r.get("file")), {})

    started = head.get("started", "")
    try:
        when = datetime.fromisoformat(str(started))
    except ValueError:
        when = None
    seconds = tail.get("seconds", records[-1].get("t") if records else None)
    slug = slug or str(tail.get("slug") or "")
    title = slug.replace("-", " ").title() if slug else (
        f"Session {when:%Y-%m-%d %H:%M}" if when else "Session"
    )

    front = [
        "---",
        f"uuid: {head.get('uuid', '')}",
        f"slug: {slug}".rstrip(),
        f"started: {when:%Y-%m-%d %H:%M:%S %z}" if when else f"started: {started}",
        f"duration: {_span(seconds)}",
        f"entrypoint: {head.get('entrypoint', '')}",
        f"model: {head.get('model', '')}",
        f"photos: {tail.get('photos', 0)}",
    ]
    if video:
        front.append(f"video: {video['file']}")
    if tail.get("reason") and tail["reason"] != "done":
        front.append(f"ended: {tail['reason']}")
    front.append("---")

    blocks = ["\n".join(front), f"# {title}"]
    if when:
        blocks.append(
            f"**{when:%A %-d %B %Y, %H:%M}** · {_span(seconds)} · {head.get('entrypoint', '')}"
        )

    # Sorted by ``t``, not by arrival. The jsonl stays in the order things actually landed,
    # which is the honest record; the page is for reading, and a transcript that arrived after
    # the reply it caused would otherwise show you answering before you spoke. Python's sort is
    # stable, so anything sharing a timestamp keeps the order it was written in.
    for record in sorted(records, key=lambda r: r.get("t") or 0.0):
        block = _render_record(record)
        if block:
            blocks.append(block)

    footer = []
    if video:
        footer.append(
            f"*Recorded to [{video['file']}]({video['file']}) — "
            "you on the left channel, Cyclops on the right.*"
        )
    if not tail:
        footer.append("*This session has no end record — it was interrupted.*")
    if dropped:
        plural = "" if dropped == 1 else "s"
        footer.append(
            f"*The log ends mid-line — {dropped} record{plural} dropped. The power probably went.*"
        )
    if footer:
        blocks.append("---")
        blocks.extend(footer)
    return "\n\n".join(blocks) + "\n"


def _render_record(record: dict) -> str:
    kind = record.get("type")
    at = _clock(record.get("t"))
    if kind == "you":
        return f"**You** ({at})\n{record.get('text', '')}"
    if kind == "cyclops":
        mark = " *(interrupted)*" if record.get("interrupted") else ""
        return f"**Cyclops** ({at}){mark}\n{record.get('text', '')}"
    if kind == "photo":
        return _render_photo(record, at)
    if kind == "search":
        query = record.get("query", "")
        if record.get("error"):
            return f'*Searched the web* ({at}) — "{query}" → failed: {record["error"]}'
        stale = " (you had moved on by the time it landed)" if record.get("stale") else ""
        chars = record.get("chars", 0)
        return f'*Searched the web* ({at}) — "{query}" → {chars} characters back{stale}'
    if kind == "recall":
        query = record.get("query", "")
        title = record.get("title", "")
        if not title:
            return f'*Looked for* ({at}) — "{query}" → nothing on the card matched'
        where = " and put it on the panel" if record.get("shown") else ""
        return f'*Looked for* ({at}) — "{query}" → **{title}**{where}'
    if kind == "project":
        name = record.get("name", "")
        if record.get("action") == "tracked":
            return f"*Started keeping notes on* ({at}) — **{name}**"
        return f"*Looked up its notes on* ({at}) — **{name}**"
    if kind == "screen":
        return f"*Wrote on the scratchpad* ({at}) — {_scratchpad_gist(record.get('html', ''))}"
    if kind == "data":
        return _render_data(record, at)
    if kind == "transcript_failed":
        return f"*You said something that could not be transcribed* ({at})"
    if kind == "error":
        return f"*Something went wrong* ({at}) — {record.get('message', '')}"
    return ""


SCRATCHPAD_GIST_CHARS = 70  # a line in a transcript, not the markup


def _scratchpad_gist(html: str) -> str:
    """What a scratchpad said, in a few words - the words out of the markup he wrote.

    Read out of the HTML rather than asked for as a title alongside it. A title would be a second
    argument the model has to finish writing before anything can appear on the glass, and the one
    thing this tool sells is that it appears while he is still talking; the words are already in
    there, and this is the only reader of them.

    Tags out, whitespace collapsed. Nothing here needs to be a parser: what it is reading is one
    screenful of headings and list items, and the answer only has to be recognisable a week later.
    """
    text = re.sub(r"<[^>]*>", " ", html)
    text = unescape(text)
    text = " ".join(text.split())
    if not text:
        return f"{len(html)} characters of markup, and no words in it"
    if len(text) <= SCRATCHPAD_GIST_CHARS:
        return text
    return text[:SCRATCHPAD_GIST_CHARS].rstrip() + "…"


def _render_data(record: dict, at: str) -> str:
    """One line for a trip to ``Project Data.xlsx``. The values themselves stay in the workbook.

    Named rather than quoted on purpose: the page is the conversation, and the sheet beside it is
    where a number lives. Printing both would give a reader two copies to disagree with each
    other, and only one of them is the one Cyclops will read back next week.
    """
    project = record.get("project", "")
    keys = ", ".join(str(key) for key in record.get("keys") or [])
    action = record.get("action")
    if action == "saved":
        again = f", {record['replaced']} replacing a value" if record.get("replaced") else ""
        tab = record.get("tab", "")
        return f"*Wrote down* ({at}) — **{project}** / {tab}: {keys}{again}"
    if action == "forgot":
        return f"*Deleted a value* ({at}) — **{project}** / {record.get('tab', '')}: {keys}"
    query = record.get("query", "")
    hits = record.get("hits", 0)
    found = f"{hits} found" if hits else "nothing written down"
    return f'*Looked up a value* ({at}) — "{query}" in **{project}** → {found}'


def _render_photo(record: dict, at: str) -> str:
    file = str(record.get("file", ""))
    if record.get("by") == "drawn":
        # A diagram, which is a photo record because it is a jpg in photos/ like any other - see
        # cyclops.imagine.draw.
        request = record.get("request", "")
        if record.get("error"):
            return f'*Tried to draw* ({at}) — "{request}" → failed: {record["error"]}'
        line = f'*Drew a diagram* ({at}) — "{request}"'
        label = "Drew"
    elif record.get("by") == "edit":
        # A picture cyclops.imagine made from an earlier one. It is a photo record because it is
        # a jpg in photos/ like any other, and it says what was asked for rather than who shot
        # it, because nobody shot it. A failed edit has no file and renders as the line alone.
        request = record.get("request", "")
        if record.get("error"):
            return f'*Tried to imagine a change* ({at}) — "{request}" → failed: {record["error"]}'
        line = f'*Imagined a change* ({at}) — "{request}"'
        label = "Imagined"
    elif record.get("by") == "cyclops":
        # Historical: Cyclops used to hold its own shutter. Kept because --fix re-renders old
        # logs, and a card full of them would otherwise lose half its pictures' captions.
        line = f"*Cyclops took a photo* ({at})"
        if record.get("focus"):
            line += f" — asked to focus on: {record['focus']}"
        label = "Cyclops"
    elif record.get("shown"):
        line = f"*You took a photo* ({at}) — the shutter button. Cyclops looked at it."
        label = "You"
    else:
        # No `shown` key at all means a record from before the shutter fed the conversation,
        # for which this sentence is simply true. A False one is a photo taken with nothing
        # live to show it to. Either way the picture stopped at the disk, so say so.
        line = f"*You took a photo* ({at}) — the shutter button. Cyclops never saw this one."
        label = "You"
    if not file:
        return line
    return f"{line}\n\n![{label}, {_shot_time(file)}]({file})"


# ------------------------------------------------------------------ cyclops-sessions


def _folders(sessions_dir: Path) -> list[Path]:
    if not sessions_dir.is_dir():
        return []
    return sorted(p for p in sessions_dir.iterdir() if p.is_dir())


def _summarise(folder: Path, state: card.State | None = None) -> str:
    state = state or card.triage(folder)
    records, _ = read_log(folder / LOG_NAME)
    tail = next((r for r in reversed(records) if r.get("type") == "end"), {})
    flags = []
    if state.verdict == "live":
        flags.append("LIVE")
    if state.video:
        flags.append("video")
    if state.parts:
        flags.append("parts/")
    if state.verdict == "empty":
        flags.append("NOTHING IN IT")  # not "EMPTY": it may well hold a video of an empty room
    elif not state.page:
        flags.append("UNFINISHED")
    else:
        if not state.named:
            flags.append("unnamed")
        if not state.filed:
            flags.append("unfiled")
    if state.dropped:
        flags.append(f"{state.dropped} bad line(s)")
    # Diagrams are rarer than photos, so they earn a column only when there are any - a card of
    # sessions that never drew anything reads exactly as it did before.
    return (
        f"{folder.name:<44}{_span(tail.get('seconds')):>8}"
        f"{state.photos:>4} photo{'' if state.photos == 1 else 's'}   {'  '.join(flags)}"
    ).rstrip()


def _append(log: Path, record: dict) -> None:
    """Add one record to a finished log, stamped after everything already in it."""
    records, _ = read_log(log)
    at = max((r.get("t") or 0.0 for r in records), default=0.0)
    try:
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"t": at, **record}, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())  # once per repair, not per line - see the module docstring
    except OSError:
        pass  # the repair is still worth reporting even if we cannot write it down


def _fix(folder: Path, state: card.State | None = None) -> list[str]:
    """Finish a session the power cut off. Offline, idempotent, never destroys anything.

    Both gates here used to be existence checks, and both were wrong in the same way: a
    truncated ``video.mp4`` and a zero-byte ``session.md`` each satisfied ``is_file()``, so the
    wreckage of an interrupted teardown was what stopped the repair from running. They now ask
    what is actually in the folder - see :func:`cyclops.card.triage`.
    """
    did = []
    state = state or card.triage(folder)
    log = folder / LOG_NAME
    parts, video = folder / PARTS, folder / VIDEO
    if state.parts:
        # No "and the video is missing" here any more. `parts/` is removed only when a mux
        # returns zero (record.mux), so its survival *is* the statement that one did not - and
        # re-muxing is idempotent, so this repairs an absent video and a truncated one alike.
        done = mux(parts, video)
        if done.ok:
            # Record it, so the page that gets written below links the video the same way a
            # session that finished normally would. The mux really did happen; the log should
            # say so rather than leave the one artifact it cannot see for itself unmentioned.
            _append(log, {"type": "video", "file": video.name, "repaired": True})
            did.append(f"muxed {video.name}")
        else:
            _append(log, {"type": "video", "error": done.why})
            did.append(f"could not mux ({done.why})")
    if state.records and not state.page:
        # `state.records`, not `log.is_file()`: a zero-byte log used to earn a fabricated
        # session.md, which then read as a finished session forever after.
        records, dropped = read_log(log)
        try:
            card.write_text(folder / PAGE_NAME, render_markdown(records, dropped=dropped))
        except OSError as exc:
            did.append(f"could not write {PAGE_NAME} ({exc})")  # never abort the rest of the card
        else:
            did.append(f"wrote {PAGE_NAME}")
    return did


def describe(
    folder: Path, settings: Settings, state: card.State | None = None
) -> tuple[Path, list[str]]:
    """Give a finished session whatever it is missing: a name, a summary, or both.

    Where the folder ends up comes back with what was done to it, because naming moves it. A
    folder that already has both costs nothing - it is never sent to the model at all, so this
    can be run over a whole card repeatedly without paying for it twice.

    Public because this is where naming lives now: :mod:`cyclops.after` calls it moments after a
    session ends, boot recovery calls it over the whole card, and ``cyclops-sessions --name``
    calls it by hand. All three are the same call on the same folder in different weather.
    """
    from .slug import describe_session

    state = state or card.triage(folder)
    needs_name = not state.named
    needs_summary = not state.summary  # written(), so a zero-byte summary.md asks again
    if not needs_name and not needs_summary:
        return folder, []
    if not needs_summary:
        # It has a summary but no name, which is not a job half done - it is the finished job.
        # Both halves come out of one call, so a summary is proof the model saw this session,
        # and an empty slug is its answer: `slugify` turns the "chat" it is told to reply with
        # for a conversation about nothing into "", meaning leave the folder dated. Asking
        # again buys the same answer at the same price, on every boot, forever.
        return folder, []
    records, dropped = read_log(folder / LOG_NAME)
    text = transcript_text(records)
    if not text:
        # Checked before spending a round trip: `--name` used to send every silent session on
        # the card to the model to be told there was nothing in it.
        return folder, []
    described = describe_session(text, settings)
    if _nothing_in_it(state, described):
        return folder, [_remove(folder)]
    did = []
    # Summary first, then the rename: the folder moves with its contents either way, and this
    # ordering means a rename that fails still leaves the summary where it belongs.
    if needs_summary and described.page:
        try:
            card.write_text(folder / SUMMARY_NAME, described.page)
            did.append(f"wrote {SUMMARY_NAME}")
        except OSError as exc:
            did.append(f"could not write {SUMMARY_NAME} ({exc})")
    if needs_name and described.slug and state.page:
        # The page was rendered before this session had a name. Give it the one it just got -
        # its heading and its `slug:` line are the only things in there that could not be known
        # at the time. Atomic like every other write here, so a failure leaves the page that is
        # already on the card, and a title is never worth failing a rename over.
        try:
            card.write_text(
                folder / PAGE_NAME, render_markdown(records, dropped=dropped, slug=described.slug)
            )
        except OSError as exc:
            did.append(f"could not retitle {PAGE_NAME} ({exc})")
    if needs_name and described.slug:
        target = folder.with_name(f"{folder.name}_{described.slug}")
        if not target.exists():
            try:
                folder.rename(target)
            except OSError as exc:
                did.append(f"could not rename ({exc})")
            else:
                card.sync_dir(target.parent)
                did.append(f"named {target.name}")
                folder = target
    return folder, did


def describe_pending(settings: Settings) -> int:
    """Name and summarise every session on the card that has neither. How many got a name.

    The sweep behind :func:`describe`, and the shape every other job in :mod:`cyclops.after`
    has: it does the session that just ended *and* whatever a power cut or a night with no wifi
    left behind, oldest first. A folder that is already described costs a :func:`cyclops.card.
    triage` and no model request, so running this after every conversation is close to free.

    A live folder is stepped over, which matters more here than it does at boot: this now runs
    while the kiosk is awake, and naming a folder *renames* it - out from under a session that
    is writing into it, if we got this wrong.
    """
    named = 0
    for folder in _folders(settings.sessions_dir):
        if card.locked(folder):
            continue  # a session is writing here; it will be named after it ends, like this one
        folder, did = describe(folder, settings)
        for one in did:
            print(f"· {folder.name}: {one}", flush=True)
            named += one.startswith("named")
    return named


def _nothing_in_it(state: card.State, described: Description) -> bool:
    """The model read the transcript, found no session in it, and nothing was made. Gate two.

    The only thing in the program that can tell a mic check from a conversation, because by then
    they look identical: both are dialogue, so :attr:`cyclops.card.State.salvage` keeps them both.
    A judgement is all that is left, and this is where it is allowed to act.

    ``state.made`` outranks it, always. A picture, a drawing, a project, a number - anything that
    survives the conversation is worth more than an opinion about the conversation, and a photo in
    particular is reachable from another session's transcript and cannot be rebuilt from anything.

    Never reached for a folder the model was not actually sent: :func:`describe` returns above
    this on an empty transcript, and :attr:`cyclops.slug.Description.nothing` is False on every
    failure inside the call, so silence can never be mistaken for a verdict.
    """
    return described.nothing and not state.made


def _remove(folder: Path, *, dry_run: bool = False) -> str:
    """Delete a folder nothing survived in. The one destructive thing in the program.

    So it asks twice, from two directions. :func:`cyclops.card.triage` has already said there is
    nothing of value here; :func:`cyclops.card.surprises` says whether there is anything here we
    did not put here, and a folder holding someone's note keeps the note and therefore keeps
    itself. Then the files go one at a time and the folder goes with ``rmdir``, not ``rmtree``:
    ``rmdir`` fails on anything unexpected still being there, which is a safety property worth
    more than the convenience.
    """
    odd = card.surprises(folder)
    if odd:
        return f"nothing survived, but {', '.join(odd)} is not ours - left alone"
    if dry_run:
        return "nothing survived - would remove"
    try:
        for name in (
            LOG_NAME, PAGE_NAME, SUMMARY_NAME, RECEIPT_NAME, VIDEO,
            # Everything cyclops.cut may have left. This list and card.KNOWN move together:
            # a name in one and not the other is a folder that can never be removed, because
            # surprises() would report the leftover as somebody else's file.
            card.CUT_REQUEST, card.CUT_PLAN, card.CUT_SUBS, card.CUT,
        ):
            (folder / name).unlink(missing_ok=True)
        # clips/ and parts/ hold generated names rather than a fixed set, so they empty
        # themselves first. Reached only after surprises() cleared the folder, so everything in
        # here is ours.
        #
        # parts/ joined clips/ here when "empty" stopped meaning "no files on disk": a session
        # nobody spoke in whose mux died leaves the raw video and two WAVs in there, and it now
        # reaches this function instead of being repaired. An rmdir on that raises, which left
        # the folder stuck forever - never deleted, and never muxed either, because _recover
        # takes the removal branch and steps over _fix.
        for generated in (card.CLIPS, PARTS):
            made = folder / generated
            if made.is_dir():
                for one in made.iterdir():
                    one.unlink(missing_ok=True)
                made.rmdir()
        if (folder / PHOTOS).is_dir():
            # No such list for photos/, deliberately: a folder with a picture in it is never
            # empty, so this is empty by definition and refusing is how we hear that triage was
            # wrong about the one thing in here that cannot be regenerated. Our own scratch
            # files are the exception - a photo killed mid-write leaves one, triage does not
            # count it as a picture, and it would otherwise wedge the folder the way parts/ did.
            for stray in card.strays(folder / PHOTOS):
                stray.unlink(missing_ok=True)
            (folder / PHOTOS).rmdir()
        folder.rmdir()
    except OSError as exc:
        return f"nothing survived, but could not remove it ({exc})"
    card.sync_dir(folder.parent)
    return "nothing survived - removed"


def _recover(settings: Settings, *, offline: bool = False, dry_run: bool = False) -> int:
    """Bring every session on the card to a finished state, or say why it could not be.

    What ``--fix`` and ``--name`` do, over the whole card, plus the two things nobody should get
    by accident: deleting what nothing survived in, and handing the results to the projects
    sweep. This is what runs at boot, and the order is the whole design:

    1. **A live session is never touched.** It holds a flock on its own log, so this can tell.
       Getting that wrong renames a folder out from under a running conversation.
    2. **Scratch files first**, so what is printed next is the truth rather than a half-write.
    3. **Delete the husks before repairing anything** - no ffmpeg run and no model request is
       ever spent on a folder that is about to go.
    4. **Repair offline, then describe.** A box with no network still leaves every folder as
       finished as it can be made without one.
    5. **File last**, because ``summary.md`` is what the filing agents are handed as the brief;
       filing a session before it has one files it worse.

    Returns the process exit code: non-zero when something is still unfinished, so
    ``Restart=on-failure`` means what it says and a boot that raced DHCP tries again.
    """
    folders = _folders(settings.sessions_dir)
    if not folders:
        print(f"· no sessions in {settings.sessions_dir.expanduser().resolve()}")
        return 0

    repaired = removed = named = skipped = 0
    for folder in folders:
        state = card.triage(folder)
        if state.verdict == "live":
            print(f"· {folder.name}: a session is writing here; left alone", flush=True)
            skipped += 1
            continue
        # ... and into clips/, because a render killed by a deploy leaves its scratch file one
        # level down, where card.strays' non-recursive glob would never find it - and a stray
        # there makes cut.progress say "clipping" forever.
        for stray in [*card.strays(folder), *card.strays(folder / card.CLIPS)]:
            if not dry_run:
                stray.unlink(missing_ok=True)
            print(f"· {folder.name}: swept {stray.name}", flush=True)
        if state.verdict == "empty":
            said = _remove(folder, dry_run=dry_run)
            print(f"· {folder.name}: {said}", flush=True)
            removed += said.endswith("removed") or said.endswith("would remove")
            continue
        if dry_run:
            for did in _would_fix(folder, state, offline=offline or not settings.api_key):
                print(f"· {folder.name}: would {did}", flush=True)
            print(_summarise(folder, state))
            continue
        for did in _fix(folder, state):
            print(f"· {folder.name}: {did}", flush=True)
            repaired += 1
        if not offline and settings.api_key:
            folder, did = describe(folder, settings)
            for one in did:
                print(f"· {folder.name}: {one}", flush=True)
                named += one.startswith("named")
                removed += one.endswith("removed")
        if folder.is_dir():  # describe may have just taken it away - see _nothing_in_it
            print(_summarise(folder))

    print(
        f"· {repaired} repaired, {removed} removed, {named} named, "
        f"{len(folders) - skipped} session(s) looked at",
        flush=True,
    )

    if offline or dry_run:
        print("· not filing: " + ("--offline" if offline else "--dry-run"), flush=True)
    elif not settings.api_key:
        print("· no OPENAI_API_KEY, so nothing could be named or filed", flush=True)
    elif settings.projects:
        _file_the_card(settings)

    # What is still wrong, if anything. This is the exit code, so say it out loud rather than
    # leaving whoever reads the journal to work out why systemd is coming back.
    #
    # Only ever non-zero when we had what it takes to finish and did not: a box with no key, or
    # one told to stay offline, is not failing at something it could retry - it is doing what it
    # was asked. `Restart=on-failure` would otherwise spin on every boot of a keyless box.
    if dry_run or offline or not settings.api_key:
        return 0
    unfinished = [f.name for f in _folders(settings.sessions_dir) if _wanting(f)]
    if unfinished:
        print(f"· still unfinished: {', '.join(unfinished)}", file=sys.stderr, flush=True)
        return 1
    return 0


def _tidy(settings: Settings, *, dry_run: bool = False) -> int:
    """Ask the model about every session that made nothing, and remove the ones that were nothing.

    The second gate applied to a card written before there was one. A verb you run, never one
    that runs at boot: a folder that already has a ``summary.md`` is one :func:`describe` will
    never ask about again, so the only way to re-judge it is to pay for the answer a second time,
    and doing that on every boot is the spend :func:`_wanting` exists to prevent.

    What gate one can see is free and is taken here too, so that one verb is the whole policy and
    ``--tidy`` never leaves behind something ``--recover`` would have removed anyway.

    Composes with ``--dry-run``, which is not optional here in the way it is elsewhere: this is
    the only path in the program that can remove a folder somebody spoke in.
    """
    from .slug import describe_session

    looked = removed = 0
    for folder in _folders(settings.sessions_dir):
        if card.locked(folder):
            continue  # a session is writing here
        state = card.triage(folder)
        if state.verdict == "empty":
            print(f"· {folder.name}: {_remove(folder, dry_run=dry_run)}", flush=True)
            removed += 1
            continue
        if state.made:
            continue  # something in here outlives any opinion about the conversation
        records, _ = read_log(folder / LOG_NAME)
        text = transcript_text(records)
        if not text:
            continue  # nothing to send, and gate one already had its say above
        looked += 1
        if not _nothing_in_it(state, describe_session(text, settings)):
            continue
        # The size, because a dry run is the only review this gets and a video is most of it.
        heft = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) // 1_000_000
        said = "would remove" if dry_run else _remove(folder)
        print(f"· {folder.name}: nothing in it ({heft} MB) - {said}", flush=True)
        removed += 1
    print(f"· {removed} removed, {looked} asked about", flush=True)
    return 0


def _wanting(folder: Path) -> bool:
    """Is there anything left a later run could finish? What the exit code is built from.

    The distinction that matters here is between work that did not happen and work that
    happened and came back with nothing to do - because this drives ``Restart=on-failure``, and
    a folder that reports itself unfinished forever is a unit that retries, and spends, on every
    boot forever.

    A session is "described" once it has a name **or** a summary: both come out of the single
    call in :func:`cyclops.slug.describe_session`, so either one means the model saw it. A
    session with a summary and no name is the normal, correct outcome for a conversation that
    never quite settled on anything: the model answered with a slug that did not survive
    :func:`cyclops.slug.slugify`, which *means* "leave this one dated". That is a decision, not a
    failure, and asking again next boot would only buy the same answer at the same price.

    A session the model found nothing in at all does not reach here, because it is no longer on
    the card - see :func:`_nothing_in_it`. Which closes a hole rather than opening one: a folder
    that took the escape hatch got neither a name nor a summary, so this called it unfinished and
    the boot unit came back for it on every boot, forever.
    """
    state = card.triage(folder)
    if state.verdict in {"live", "empty"}:
        return False  # one is not ours to finish, the other is not there any more
    if not state.page or state.parts:
        return True  # these are free to fix and there is no reason they should still be so
    records, _ = read_log(folder / LOG_NAME)
    if not transcript_text(records):
        return False  # nothing was said, so it will never earn a name or a summary
    return not state.named and not state.summary


def _would_fix(folder: Path, state: card.State, *, offline: bool = False) -> list[str]:
    """What :func:`_fix` and :func:`describe` would do, for ``--dry-run``. Touches nothing.

    Has to make exactly the decisions those two make, including the one that is easy to forget:
    a session nobody spoke in is never sent to the model, so predicting a name for one would be
    a dry run that promises work the real run will not do.
    """
    would = []
    if state.parts:
        would.append(f"mux {VIDEO}")
    if state.records and not state.page:
        would.append(f"write {PAGE_NAME}")
    if offline:
        return would
    records, _ = read_log(folder / LOG_NAME)
    if not transcript_text(records):
        return would  # the guard `describe` makes: nothing was said, so nothing to describe
    if not state.summary:
        would.append(f"write {SUMMARY_NAME}")
    if not state.named:
        would.append("name it")
    return would


def _file_the_card(settings: Settings) -> None:
    """Hand what we repaired to the projects sweep, and wait for it.

    In-process rather than :func:`cyclops.after.spawn`, which is what a session end uses. A
    detached child would outlive a ``Type=oneshot`` unit, take its output to
    ``~/.cache/cyclops/projects.log`` instead of the journal, and put its own failure outside
    systemd's reach. Imported here, as ``_file`` does, so a box with no key still runs.
    """
    import asyncio

    from . import projects

    try:
        asyncio.run(projects.sweep(settings, limit=RECOVER_FILE_LIMIT))
    except Exception as exc:  # noqa: BLE001 - the repairs above stand whatever filing did
        print(f"· could not file: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)


USAGE = """\
usage: cyclops-sessions [--fix] [--name]
       cyclops-sessions --recover [--offline] [--dry-run]
       cyclops-sessions --tidy [--dry-run]

  (no flags)  list what is on the card
  --fix       finish anything left half-done: mux an interrupted recording, rebuild a
              missing session.md. Offline; needs no key and no network.
  --name      name anything still unnamed and write any missing summary.md. Needs a key.
  --recover   --fix and --name over the whole card, and then two things you should not get
              by accident: folders nothing survived in are deleted, and what was repaired is
              handed to the projects sweep. This is what runs at boot.
  --tidy      remove the sessions nothing happened in - a mic check, a greeting, a wake that
              ended before anybody spoke. Reads each one back and asks; anything that made a
              picture, a drawing, a project or a number is never asked about. Needs a key, and
              costs one small call per session it asks about. Run it with --dry-run first.
  --offline   with --recover: stop after the free half. No key, no network, no spend.
  --dry-run   with --recover or --tidy: say what it would do and touch nothing."""


def main() -> None:
    """``cyclops-sessions`` - list what is on the card, and finish anything left half-done."""
    args = sys.argv[1:]
    known = {"--fix", "--name", "--recover", "--tidy", "--offline", "--dry-run", "--help", "-h"}
    if unknown := [a for a in args if a not in known]:
        raise SystemExit(f"error: unknown argument {unknown[0]!r}\n{USAGE}")
    if "--help" in args or "-h" in args:
        print(USAGE)
        return
    fix, rename, recover = "--fix" in args, "--name" in args, "--recover" in args
    tidy, offline, dry_run = "--tidy" in args, "--offline" in args, "--dry-run" in args
    # Silently doing nothing is how `cyclops-projects --again` came to be documented as something
    # it does not do. Say so instead.
    if offline and not recover:
        raise SystemExit(f"error: --offline only means something with --recover\n{USAGE}")
    if dry_run and not (recover or tidy):
        raise SystemExit(
            f"error: --dry-run only means something with --recover or --tidy\n{USAGE}"
        )
    try:
        settings = load_settings(require_api_key=False)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    if recover:
        raise SystemExit(_recover(settings, offline=offline, dry_run=dry_run))
    if tidy:
        if not settings.api_key:
            raise SystemExit("error: --tidy needs an OPENAI_API_KEY: it reads each session back")
        raise SystemExit(_tidy(settings, dry_run=dry_run))

    folders = _folders(settings.sessions_dir)
    if not folders:
        print(f"· no sessions in {settings.sessions_dir.expanduser().resolve()}")
        return
    for folder in folders:
        if fix:
            for did in _fix(folder):
                print(f"· {folder.name}: {did}", flush=True)
    if rename:
        describe_pending(settings)
    for folder in _folders(settings.sessions_dir):
        print(_summarise(folder))


if __name__ == "__main__":
    main()
