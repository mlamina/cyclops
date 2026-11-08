"""A time-lapse: one camera frame a second, played at 30, for as long as he wants (arc-eng/hq#669).

It owns one folder beside the sessions and the ffmpeg writing into it, and nothing else. It is not
part of a conversation: the kiosk holds it, so a run outlives the conversation that started it and
any later one can stop it. It never starts itself - only :meth:`TimeLapse.start`, which only the
voice tool calls (arc-eng/hq#602).

The folder is a session folder in every way the card cares about, so it lists and plays where his
videos do: a ``session.jsonl`` flocked for the whole run (recovery reads that as live and leaves it
alone), ``video.mp4`` once it stops, and a ``summary.md`` and a name of its own, so naming never
spends a model call on a video with nothing said in it. ``lapse.json`` is what marks it: filing and
the highlight cut skip any folder that holds one.

What it must never do is lose a night to an unplug. The video is written as fragmented MP4, a
fragment per keyframe and a keyframe every ``KEYFRAME_EVERY`` frames, and synced to the card once a
minute, so a power cut costs the last minute at most. :func:`finish` turns whatever survived into
a plain ``video.mp4``, here on a stop and in boot recovery after an unplug.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import cv2

from . import card

CAPTURE_EVERY_S = 1.0  # one frame a second of real time...
PLAYBACK_FPS = 30  # ...played at 30: an hour becomes two minutes
KEYFRAME_EVERY = 60  # frames per fragment: a minute of recording, two seconds of video
SYNC_EVERY_S = 60.0  # what an unplug can cost, at most
SPACE_EVERY_S = 10.0
MIN_FREE_BYTES = 1024**3  # the same line a normal recording is refused at
FINISH_TIMEOUT_S = 600
SLUG = "time-lapse"
MARK = card.LAPSE_MARK
RAW = "lapse.mp4"  # inside parts/, until finish() lands video.mp4
CRF = "23"

ASKED, NO_SPACE, BROKE = "asked", "space", "broke"

_current: TimeLapse | None = None  # the kiosk's, so any conversation's tool reaches the same run


def install(recorder: TimeLapse | None) -> None:
    global _current
    _current = recorder


def current() -> TimeLapse | None:
    return _current


def running_now() -> bool:
    """Is a time-lapse recording right now? What a normal video asks before it starts."""
    return _current is not None and _current.running()


is_lapse = card.is_lapse


def video_seconds(frames: int) -> float:
    """How long the finished video is after ``frames`` frames: what the panel shows."""
    return frames / PLAYBACK_FPS


def encoder_argv(width: int, height: int, out: Path) -> list[str]:
    """BGR frames on stdin to fragmented H.264, so a file cut off mid-run is still readable."""
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-framerate", str(PLAYBACK_FPS), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", CRF, "-pix_fmt", "yuv420p",
        "-g", str(KEYFRAME_EVERY), "-keyint_min", str(KEYFRAME_EVERY), "-sc_threshold", "0",
        "-movflags", "+frag_keyframe+empty_moov+default_base_moof", "-flush_packets", "1",
        "-f", "mp4", str(out),
    ]  # fmt: skip


def remux_argv(raw: Path, out: Path) -> list[str]:
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw),
            "-c", "copy", "-movflags", "+faststart", "-f", "mp4", str(out)]  # fmt: skip


def probe_seconds(video: Path, run: Callable | None = None) -> float | None:
    """The finished file's real length, as ffprobe reads it, or None if it can't be read."""
    try:
        done = (run or subprocess.run)(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(video)],
            capture_output=True, text=True, timeout=30, check=False,
        )  # fmt: skip
        return float(done.stdout.strip())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def finish(folder: Path, run: Callable | None = None) -> tuple[bool, str]:
    """Land ``parts/lapse.mp4`` as ``video.mp4``, and write the page. Idempotent, keeps footage.

    ``parts/`` goes only once a remux returned zero, as with a normal recording (``record.mux``),
    so its survival means the job isn't done and the next boot tries again.
    """
    run = run or subprocess.run
    parts, video = folder / card.PARTS, folder / card.VIDEO
    raw = parts / RAW
    if card.written(raw):
        tmp = card.tmp_for(video)
        try:
            done = run(
                remux_argv(raw, tmp), capture_output=True, timeout=FINISH_TIMEOUT_S, check=False
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            tmp.unlink(missing_ok=True)
            return False, f"{type(exc).__name__}: {exc}"
        if done.returncode != 0 or not card.written(tmp):
            tmp.unlink(missing_ok=True)
            detail = (done.stderr or b"").decode(errors="replace").strip().splitlines()
            return False, detail[-1] if detail else f"exit {done.returncode}"
        card.land(tmp, video)
    if parts.is_dir() and card.written(video):
        shutil.rmtree(parts, ignore_errors=True)
    seconds = probe_seconds(video, run) if card.written(video) else None
    _write_page(folder, seconds)
    return card.written(video), "finished"


def _write_page(folder: Path, seconds: float | None) -> None:
    mark = _read_mark(folder)
    ran = float(mark.get("ran_s") or 0.0)
    length = f"{seconds:.0f} s of video" if seconds is not None else "a video"
    why = {
        NO_SPACE: " It stopped because the card was nearly full.",
        BROKE: " It stopped because recording broke.",
    }
    stopped = why.get(mark.get("stopped", ""), "")
    summary = f"# Time-lapse\n\n{length}, from {_clock(ran)} of recording.{stopped}\n"
    card.write_text(folder / card.SUMMARY_NAME, summary)
    card.write_text(folder / card.PAGE_NAME, summary)


def _read_mark(folder: Path) -> dict:
    try:
        return json.loads((folder / MARK).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _clock(seconds: float) -> str:
    minutes = round(seconds / 60)
    return f"{minutes // 60} h {minutes % 60} min" if minutes >= 60 else f"{minutes} min"


class TimeLapse:
    """At most one run at a time, held by the kiosk. Thread-safe; every method returns, none raise.

    ``frames`` hands back the newest camera frame (BGR, full size, nothing drawn on it) or None.
    The frame size is fixed by the first one, as the encoder is opened with it.
    """

    def __init__(
        self,
        sessions_dir: Path,
        frames: Callable[[], object | None],
        *,
        popen: Callable = subprocess.Popen,
        run: Callable = subprocess.run,
        free_bytes: Callable[[Path], int] = lambda p: shutil.disk_usage(p).free,
        clock: Callable[[], float] = time.monotonic,
        on_stop: Callable[[str], None] = lambda why: None,
    ) -> None:
        self._dir = sessions_dir
        self._frames = frames
        self._popen, self._run, self._free, self._clock, self._on_stop = (
            popen,
            run,
            free_bytes,
            clock,
            on_stop,
        )
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ffmpeg = None
        self._log = None
        self.folder: Path | None = None
        self.frames = 0
        self._t0 = 0.0
        self._why = ""
        self.last: dict = {}  # the run that ended, for the tool and the next conversation
        self._untold: dict | None = None  # a run that stopped by itself, until he's told

    # ------------------------------------------------------------------ asked of it

    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def seconds(self) -> float:
        """The finished video's length so far: what the panel shows instead of recording time."""
        return video_seconds(self.frames)

    def take_untold(self) -> dict | None:
        """A run that stopped by itself, handed out once, for the next pause in any conversation."""
        with self._lock:
            untold, self._untold = self._untold, None
            return untold

    def start(self) -> str:
        """Start a run, or say plainly why not. "" means it started."""
        with self._lock:
            if self.running():
                return "A time-lapse is already running. Stop it first."
            try:
                if self._free(self._dir) <= MIN_FREE_BYTES:
                    return "There isn't enough space left for a time-lapse."
            except OSError as exc:
                return f"It can't record, because the card can't be read ({exc})."
            first = self._frames()
            if first is None:
                return "There's no camera picture to record."
            try:
                self._open(first)
            except Exception as exc:  # noqa: BLE001 - a refusal, never a crash in the voice thread
                self._close_quietly()
                return f"It couldn't start: {type(exc).__name__}: {exc}"
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._loop, args=(first,), name="lapse", daemon=True
            )
            self._thread.start()
            return ""

    def stop(self, why: str = ASKED) -> dict:
        """Stop and save. Returns what the run left: its length, its folder, and why it stopped."""
        thread = self._thread
        if thread is None:
            return dict(self.last)
        self._why = self._why or why
        self._stop.set()
        if thread is not threading.current_thread():
            thread.join()
        return dict(self.last)

    # ------------------------------------------------------------------ the run

    def _open(self, first) -> None:
        stamp = datetime.now().strftime(card.STAMP)
        folder = self._dir / f"{stamp}_{SLUG}"
        (folder / card.PARTS).mkdir(parents=True, exist_ok=True)
        self._log = (folder / card.LOG_NAME).open("a", encoding="utf-8")
        card.claim(self._log)  # live to recovery for the whole run, dropped by the kernel if we die
        height, width = first.shape[:2]
        self._size = (width - width % 2, height - height % 2)
        card.write_text(folder / MARK, json.dumps({"started": stamp, "size": list(self._size)}))
        self._record(
            "session", started=datetime.now().isoformat(timespec="seconds"), entrypoint="timelapse"
        )
        self._record("timelapse", every_s=CAPTURE_EVERY_S, fps=PLAYBACK_FPS, size=list(self._size))
        self._ffmpeg = self._popen(
            encoder_argv(*self._size, folder / card.PARTS / RAW),
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )  # fmt: skip
        self.folder, self.frames, self._why, self._t0 = folder, 0, "", self._clock()

    def _loop(self, frame) -> None:
        synced = spaced = self._t0
        try:
            while not self._stop.is_set():
                latest = self._frames()
                frame = (
                    latest if latest is not None else frame
                )  # a stalled camera repeats, never gaps
                self._write(frame)
                self.frames += 1
                now = self._clock()
                if now - spaced >= SPACE_EVERY_S:
                    spaced = now
                    if self._free(self._dir) <= MIN_FREE_BYTES:
                        self._why = NO_SPACE
                        break
                if now - synced >= SYNC_EVERY_S:
                    synced = now
                    self._sync()
                delay = self._t0 + self.frames * CAPTURE_EVERY_S - self._clock()
                if delay > 0:
                    self._stop.wait(delay)
        except Exception:  # noqa: BLE001 - the camera or ffmpeg broke: save what there is
            self._why = self._why or BROKE
        self._end()

    def _write(self, frame) -> None:
        if frame.shape[1::-1] != self._size:
            frame = cv2.resize(frame, self._size, interpolation=cv2.INTER_AREA)
        self._ffmpeg.stdin.write(frame.tobytes())

    def _sync(self) -> None:
        raw = self.folder / card.PARTS / RAW
        try:
            fd = os.open(raw, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _end(self) -> None:
        why, folder = self._why or ASKED, self.folder
        ran = self._clock() - self._t0
        try:
            self._ffmpeg.stdin.close()
            self._ffmpeg.wait(timeout=FINISH_TIMEOUT_S)
        except Exception:  # noqa: BLE001 - finish() reads whatever it left
            pass
        mark = {**_read_mark(folder), "frames": self.frames, "ran_s": round(ran, 1), "stopped": why}
        card.write_text(folder / MARK, json.dumps(mark))
        ok, said = finish(folder, self._run)
        seconds = probe_seconds(folder / card.VIDEO, self._run) if ok else None
        self._record(
            "end", seconds=round(seconds if seconds is not None else self.seconds(), 2), stopped=why
        )
        self._close_quietly()
        self.last = {
            "seconds": seconds, "frames": self.frames, "folder": folder.name,
            "stopped": why, "saved": ok, "problem": "" if ok else said,
        }  # fmt: skip
        self._thread = None
        if why != ASKED:
            self._untold = dict(self.last)
            self._on_stop(why)

    def _record(self, kind: str, **fields) -> None:
        if self._log is None:
            return
        t = round(self._clock() - self._t0, 2) if self._t0 else 0.0
        self._log.write(json.dumps({"t": t, "type": kind, **fields}, ensure_ascii=False) + "\n")
        self._log.flush()
        os.fsync(self._log.fileno())

    def _close_quietly(self) -> None:
        if self._log is not None:
            self._log.close()  # and with it, the flock
            self._log = None
