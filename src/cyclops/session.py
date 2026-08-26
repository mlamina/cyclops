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
* **Records are flushed, never fsynced.** An fsync per line would grind an SD card for a
  guarantee we do not need. The one corruption this can suffer is a half-written last line
  after a power cut, and that is exactly why the format is JSON Lines rather than JSON: no
  shared state between lines, so :func:`read_log` drops the bad one and everything before it is
  still a session.

Writes arrive from three threads - the session's event loop (:meth:`SessionLog.observe`), the
kiosk's shutter thread (via :func:`note`), and the teardown - so the handle is lock-guarded.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid as uuid_module
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import ConfigError, Settings, load_settings
from .record import MUX_TIMEOUT_S, SessionRecorder, mux_command

if TYPE_CHECKING:  # importing these for real would be a cycle - agent.py imports this module
    from openai.types.realtime import RealtimeServerEvent

    from .agent import VoiceAgent
    from .audio import Microphone, Speaker
    from .record import FrameSource

LOG_NAME = "session.jsonl"
PAGE_NAME = "session.md"
PHOTOS = "photos"
PARTS = "parts"
VIDEO = "video.mp4"
STAMP = "%Y-%m-%d_%H-%M-%S"
STAMPED = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")  # a folder nobody has named yet
SLUG_JOIN_S = 8.0  # a hair over slug.SLUG_TIMEOUT_S; it runs alongside the mux, not after it
MAX_TRANSCRIPT_CHARS = 3000

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
    ) -> None:
        self.settings = settings
        self.uuid = str(uuid_module.uuid4())
        self.entrypoint = entrypoint
        self.started = datetime.now().astimezone()
        self.dir = settings.sessions_dir / self.started.strftime(STAMP)
        self.slug = ""  # empty until __exit__ has named it
        self.failed = ""  # non-empty once logging gave up; the session carries on regardless
        self._agent = agent
        self._mic = mic
        self._speaker = speaker
        self._frames = frames
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

    @property
    def photos_dir(self) -> Path:
        return self.dir / PHOTOS

    # ---------------------------------------------------------------- lifecycle

    def __enter__(self) -> SessionLog:
        global _live
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            self._handle = (self.dir / LOG_NAME).open("a", encoding="utf-8")
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
        namer = self._start_naming()  # its round trip hides behind the mux below
        self._stop_recorder()
        if namer is not None:
            namer.join(SLUG_JOIN_S)
        reason = _reason(exc_type)
        fields: dict[str, Any] = {
            "reason": reason,
            "seconds": self._elapsed(),
            "slug": self.slug,  # in the file too, so the folder name is always reconstructible
            "photos": sum(1 for r in self._records if r.get("type") == "photo"),
        }
        if reason == "error" and exc is not None:
            fields["error"] = f"{type(exc).__name__}: {exc}"
        self.event("end", **fields)
        self._close_handle()
        self._write_page()  # written last, which is what makes its presence mean "finished"
        self._rename()
        self._report()

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
        finished = recorder.stop()
        if finished is not None:
            self.event("video", file=finished.name, seconds=self._elapsed())
            print(f"· recorded to {finished}", flush=True)
        elif recorder.failed:
            self.event("video", error=recorder.failed)

    def _start_naming(self) -> threading.Thread | None:
        """Ask a small model what this was about, off-thread so the mux hides the wait."""
        if self.failed or not self.settings.slug:
            return None
        text = transcript_text(self._records)
        if not text:
            return None
        thread = threading.Thread(
            target=self._name, name="session-slug", args=(text,), daemon=True
        )
        thread.start()
        return thread

    def _name(self, text: str) -> None:
        # Imported here rather than at module scope so this module - and, more to the point,
        # `cyclops-sessions --fix` - keeps working on a box with no key and no network.
        from .slug import name_session

        self.slug = name_session(text, self.settings)

    def _rename(self) -> None:
        """Append the slug. Atomic within a filesystem, so nothing sees a half-named folder."""
        if self.failed or not self.slug:
            return
        target = self.dir.with_name(f"{self.dir.name}_{self.slug}")
        if target.exists():
            return
        try:
            self.dir.rename(target)
        except OSError:
            return  # the end record already says what it should have been called
        self.dir = target

    def _write_page(self) -> None:
        if self.failed:
            return
        try:
            (self.dir / PAGE_NAME).write_text(render_markdown(self._records), encoding="utf-8")
        except OSError as exc:
            self._give_up(f"{type(exc).__name__}: {exc}")

    def _close_handle(self) -> None:
        with self._lock:
            handle, self._handle = self._handle, None
        if handle is not None:
            try:
                handle.close()
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


def read_log(path: Path) -> tuple[list[dict], int]:
    """Every record in a ``session.jsonl``, and how many lines were not one.

    Lines are flushed but never fsynced (see the module docstring), so a power cut can leave the
    last one half-written. That is the only corruption this format can suffer, and nothing is
    swallowed silently: the count comes back and :func:`render_markdown` says so on the page.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return [], 0
    records: list[dict] = []
    dropped = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:  # JSONDecodeError is a subclass
            dropped += 1
            continue
        if isinstance(record, dict):
            records.append(record)
        else:
            dropped += 1
    return records, dropped


def transcript_text(records: list[dict], limit: int = MAX_TRANSCRIPT_CHARS) -> str:
    """Just the dialogue, in order, capped from the front - for naming the session."""
    lines = []
    for record in records:
        kind = record.get("type")
        if kind == "you":
            lines.append(f"User: {record.get('text', '')}")
        elif kind == "cyclops":
            lines.append(f"Cyclops: {record.get('text', '')}")
    return "\n".join(lines)[:limit].strip()


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


def render_markdown(records: list[dict], *, dropped: int = 0) -> str:
    """The whole page, from records alone.

    Pure on purpose: the live close path and ``cyclops-sessions --fix`` both call this, which is
    the only way the two can be guaranteed to agree about what a session looked like.
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
    slug = str(tail.get("slug") or "")
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
    if kind == "transcript_failed":
        return f"*You said something that could not be transcribed* ({at})"
    if kind == "error":
        return f"*Something went wrong* ({at}) — {record.get('message', '')}"
    return ""


def _render_photo(record: dict, at: str) -> str:
    file = str(record.get("file", ""))
    if record.get("by") == "cyclops":
        line = f"*Cyclops took a photo* ({at})"
        if record.get("focus"):
            line += f" — asked to focus on: {record['focus']}"
        label = "Cyclops"
    else:
        # The shutter photo is written to disk and stops there - it never enters the
        # conversation - and a page that showed both images identically would be lying.
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


def _summarise(folder: Path) -> str:
    records, dropped = read_log(folder / LOG_NAME)
    tail = next((r for r in reversed(records) if r.get("type") == "end"), {})
    photos = len(list((folder / PHOTOS).glob("*.jpg"))) if (folder / PHOTOS).is_dir() else 0
    flags = []
    if (folder / VIDEO).is_file():
        flags.append("video")
    if (folder / PARTS).is_dir():
        flags.append("parts/")
    if not (folder / PAGE_NAME).is_file():
        flags.append("UNFINISHED")
    elif STAMPED.match(folder.name):
        flags.append("unnamed")
    if dropped:
        flags.append(f"{dropped} bad line(s)")
    return (
        f"{folder.name:<44}{_span(tail.get('seconds')):>8}"
        f"{photos:>4} photo{'' if photos == 1 else 's'}   {'  '.join(flags)}"
    ).rstrip()


def _append(log: Path, record: dict) -> None:
    """Add one record to a finished log, stamped after everything already in it."""
    records, _ = read_log(log)
    at = max((r.get("t") or 0.0 for r in records), default=0.0)
    try:
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"t": at, **record}, ensure_ascii=False) + "\n")
    except OSError:
        pass  # the repair is still worth reporting even if we cannot write it down


def _fix(folder: Path) -> list[str]:
    """Finish a session the power cut off. Offline, idempotent, never overwrites."""
    did = []
    log = folder / LOG_NAME
    parts, video = folder / PARTS, folder / VIDEO
    if parts.is_dir() and not video.is_file():
        done = subprocess.run(  # noqa: S603 - the command is ours, from record.mux_command
            mux_command(parts, video), capture_output=True, timeout=MUX_TIMEOUT_S, check=False
        )
        if done.returncode == 0:
            shutil.rmtree(parts, ignore_errors=True)
            # Record it, so the page that gets written below links the video the same way a
            # session that finished normally would. The mux really did happen; the log should
            # say so rather than leave the one artifact it cannot see for itself unmentioned.
            _append(log, {"type": "video", "file": video.name, "repaired": True})
            did.append(f"muxed {video.name}")
        else:
            tail = done.stderr.decode(errors="replace").strip().splitlines()
            did.append(f"could not mux ({tail[-1] if tail else done.returncode})")
    page = folder / PAGE_NAME
    if log.is_file() and not page.is_file():
        records, dropped = read_log(log)
        page.write_text(render_markdown(records, dropped=dropped), encoding="utf-8")
        did.append(f"wrote {PAGE_NAME}")
    return did


def _name(folder: Path, settings: Settings) -> str:
    from .slug import name_session

    if not STAMPED.match(folder.name):
        return ""
    records, _ = read_log(folder / LOG_NAME)
    slug = name_session(transcript_text(records), settings)
    if not slug:
        return ""
    target = folder.with_name(f"{folder.name}_{slug}")
    if target.exists():
        return ""
    folder.rename(target)
    return target.name


def main() -> None:
    """``cyclops-sessions`` - list what is on the card, and finish anything left half-done."""
    args = sys.argv[1:]
    fix, rename = "--fix" in args, "--name" in args
    if unknown := [a for a in args if a not in {"--fix", "--name"}]:
        raise SystemExit(f"error: unknown argument {unknown[0]!r} (use --fix and/or --name)")
    try:
        settings = load_settings(require_api_key=False)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    folders = _folders(settings.sessions_dir)
    if not folders:
        print(f"· no sessions in {settings.sessions_dir.expanduser().resolve()}")
        return
    for folder in folders:
        if fix:
            for did in _fix(folder):
                print(f"· {folder.name}: {did}", flush=True)
        if rename and (named := _name(folder, settings)):
            print(f"· {folder.name}: named {named}", flush=True)
            folder = folder.with_name(named)
        print(_summarise(folder))


if __name__ == "__main__":
    main()
