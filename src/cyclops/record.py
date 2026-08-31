"""Recording a whole kiosk session: what was on the panel, plus the two sides of the conversation.

A session leaves one ``video.mp4`` in its own folder behind - H.264 video of either the panel or
the camera, whichever the settings screen asked for (:mod:`cyclops.filming`), with a stereo audio
track carrying the user on the left channel and Cyclops on the right. Neither choice reaches this
module: it is handed a :class:`FrameSource` and samples it, and the two answers are two objects
that satisfy that one method.

Keeping the two voices apart means neither is mixed into the other (listen to one side alone, or
mix them down later), and it sidesteps the double-counting you would get from recording an open
microphone that is also hearing the speaker.

Three streams have to line up, and each arrives on its own clock:

* video is sampled here on the wall clock at a fixed rate - re-writing the previous frame when the
  source has not produced a new one, so video time never drifts away from wall time;
* the user track is written one block per microphone callback, so it stays 1:1 with the input clock;
* the agent track is taken from the speaker callback *after* its zero-fill, which makes it a
  continuous record of what actually came out of the speaker, the silence between utterances
  included - no timeline has to be reconstructed from response deltas.

Everything is aligned against a single ``t0`` taken at start: a stream that produces its first
block late gets exactly that much silence prepended, so the parts need no offsets when muxed.

PortAudio callbacks must not do I/O (see :mod:`cyclops.audio`), so the audio hooks here only
append to a lock-protected buffer and a writer thread puts it on disk. For the same reason a
failure on an audio thread only sets a flag; the writer thread is what reports it.

Recording must never be able to break a session, so every hook and thread is wrapped: anything
that goes wrong disables the recording and leaves the conversation running.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2

from . import card
from .audio import BYTES_PER_FRAME, SAMPLE_RATE, EchoGuard

DEFAULT_FPS = 15
# 0 means "whatever the source hands us", which is the only width that is right for both of the
# things this records: the panel is 800x480 and wants to be kept 1:1, and a camera has its own
# native size worth keeping. It is a ceiling when set, never an upscale - see _output_size.
DEFAULT_WIDTH = 0
CRF = "26"
# The guard releases up to PREROLL_BLOCKS at once on a barge-in; the delay line has to be able to
# reach back over all of them, so it is exactly that deep.
DELAY_BLOCKS = EchoGuard.PREROLL_BLOCKS
DRAIN_INTERVAL_S = 0.25
FIRST_FRAME_TIMEOUT_S = 1.0
JOIN_TIMEOUT_S = 5.0
MUX_TIMEOUT_S = 120.0
# The encoder's intermediate. Not "video.mp4": that is the finished file one level up, and a
# folder holding two of them would be a puzzle for whoever pulls the card.
RAW_VIDEO = "video-raw.mp4"


def mux_command(work_dir: Path, out_path: Path) -> list[str]:
    """The ffmpeg call that joins a session's parts into its mp4.

    Lifted out of :meth:`SessionRecorder._mux` so ``cyclops-sessions --fix`` can finish an
    interrupted session with exactly the command the recorder would have run, rather than a
    second copy of it that quietly rots out of step.

    ``-f mp4`` is not decoration. ffmpeg picks its muxer from the output's extension, and
    :func:`mux` deliberately writes to a scratch name ending ``.tmp`` - which ffmpeg cannot
    guess a container from, so it refuses the job before reading a single frame ("Unable to
    find a suitable output format"). The scratch name is what makes ``video.mp4`` existing mean
    *a mux returned zero*, so it is the extension that has to give way, not the safety.
    """
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(work_dir / RAW_VIDEO),
        "-i", str(work_dir / "user.wav"),
        "-i", str(work_dir / "agent.wav"),
        "-filter_complex", "[1:a][2:a]join=inputs=2:channel_layout=stereo[a]",
        "-map", "0:v", "-map", "[a]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "96k",
        "-shortest", "-movflags", "+faststart",
        "-f", "mp4",  # the output is a scratch name; ffmpeg cannot infer the container from it
        str(out_path),
    ]


@dataclass(frozen=True)
class Mux:
    """What one mux attempt came to. ``why`` is ffmpeg's last line, and empty when it worked."""

    ok: bool
    why: str = ""


def mux(work_dir: Path, out_path: Path) -> Mux:
    """Join a session's parts into its mp4, landing the file only if ffmpeg said it worked.

    ffmpeg used to be pointed straight at ``video.mp4``, and a power cut mid-encode left a
    truncated file sitting on the name. That was worse than losing it: ``cyclops-sessions --fix``
    asked ``not video.is_file()`` before retrying, so the wreckage of the interrupted mux
    permanently blocked its own repair. Now the encode happens under a scratch name and is
    renamed into place afterwards, which makes ``video.mp4`` existing mean *a mux returned zero* -
    the only reading under which the retry can be trusted.

    The parts are removed only on success, which is what makes ``parts/`` surviving the honest
    signal that this never finished. Both callers - the live teardown and the repair path - come
    through here rather than through :func:`mux_command` alone, so there is one implementation of
    "finish a recording" and not two that drift.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = card.tmp_for(out_path)
    try:
        done = subprocess.run(  # noqa: S603 - the command is ours, from mux_command
            mux_command(work_dir, tmp), capture_output=True, timeout=MUX_TIMEOUT_S, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        # Notably FileNotFoundError: systemd's PATH is not a login shell's, and the recovery
        # unit would otherwise die on the first folder rather than repair the other twelve.
        tmp.unlink(missing_ok=True)
        return Mux(False, f"{type(exc).__name__}: {exc}")
    if done.returncode != 0:
        tmp.unlink(missing_ok=True)
        detail = done.stderr.decode(errors="replace").strip().splitlines()
        return Mux(False, detail[-1] if detail else f"exit {done.returncode}")
    try:
        card.land(tmp, out_path)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        return Mux(False, f"{type(exc).__name__}: {exc}")
    shutil.rmtree(work_dir, ignore_errors=True)
    return Mux(True)


class FrameSource(Protocol):
    """What the recorder needs from whatever it is recording: the newest frame, or None.

    Two things satisfy it, and which one a session gets is the settings screen's answer (see
    :mod:`cyclops.filming`): :class:`~cyclops.camera.CameraSource`, which is the sensor, and
    :class:`PanelSource` below, which is the glass.
    """

    def frame(self): ...


class PanelSource:
    """The screen as a frame source: whatever the kiosk last put on the panel.

    :class:`~cyclops.camera.CameraSource` is polled - it holds a device open and hands out the
    newest frame it read. This is the same contract from the other side. Nothing here reads
    anything: the render loop hands over each finished frame as it paints it, and the recorder
    samples that on its own clock exactly as it sampled the camera. What lands in ``video.mp4``
    is then the panel - mirrored preview, halo, timer, caption, tab row - rather than the raw
    frames the chrome was drawn over.

    One rebinding of one name, so no lock: see :meth:`publish`.
    """

    def __init__(self) -> None:
        self._frame = None

    def publish(self, frame) -> None:
        """Hand over the frame that has just gone on the panel. The render thread's to call.

        A store and a load of an attribute are each one bytecode, so the recorder's thread sees
        the previous frame or the new one and never half of one - the same bargain
        :attr:`cyclops.ui.SessionController._phase` and the flags on the agent already make.
        What makes it *enough* rather than merely atomic is that these frames are never written
        to again: :func:`cyclops.overlay.composite` allocates a fresh array every call, so a
        frame handed over here can be encoded at leisure while the loop builds the next one.

        Only window-sized composites belong here. The first frame decides the encoder's size
        for the whole session (see :meth:`SessionRecorder._output_size`), so a raw camera frame
        slipped in before the loop starts would squash every panel frame after it.
        """
        self._frame = frame

    def frame(self):
        """The last frame painted, or None before the first one. See :class:`FrameSource`."""
        return self._frame


class _Track:
    """One mono PCM16 channel: appended to from an audio thread, written by the writer thread."""

    def __init__(self, path: Path, t0: float) -> None:
        self._t0 = t0
        self._lock = threading.Lock()
        self._pending = bytearray()
        self._started = False
        self._wave = wave.open(str(path), "wb")
        self._wave.setnchannels(1)
        self._wave.setsampwidth(BYTES_PER_FRAME)
        self._wave.setframerate(SAMPLE_RATE)

    def append(self, pcm: bytes, at: float) -> None:
        """Audio-thread safe: buffer only, no I/O. ``at`` is when this audio was captured."""
        with self._lock:
            if not self._started:
                self._started = True
                lead = max(0.0, at - self._t0)  # silence for however late this stream started
                self._pending.extend(bytes(int(lead * SAMPLE_RATE) * BYTES_PER_FRAME))
            self._pending.extend(pcm)

    def drain(self) -> None:
        with self._lock:
            chunk = bytes(self._pending)
            self._pending.clear()
        if chunk:
            self._wave.writeframes(chunk)

    def close(self) -> None:
        self.drain()
        self._wave.close()


class SessionRecorder:
    """Records one session. Construct, :meth:`start`, attach the hooks, :meth:`stop`."""

    def __init__(
        self,
        frames: FrameSource,
        session_dir: Path,
        *,
        fps: int = DEFAULT_FPS,
        width: int = DEFAULT_WIDTH,
    ) -> None:
        self.out_path = session_dir / "video.mp4"
        # The parts sit inside the session's own folder rather than beside it: an interrupted
        # session is simply the folder that still has a parts/ in it, and nothing has to be
        # matched up by timestamp with anything else.
        self.work_dir = session_dir / "parts"
        self.failed = ""  # non-empty once recording has given up; the session carries on
        self._frames = frames
        self._fps = max(1, fps)
        self._width = width if width <= 0 else max(2, width)  # <=0: the source's own width
        self._t0 = 0.0
        self._size: tuple[int, int] = (0, 0)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._ffmpeg: subprocess.Popen[bytes] | None = None
        self._log_file = None
        self._user: _Track | None = None
        self._agent: _Track | None = None
        self._delay: deque[list] = deque()  # [raw_block, admitted] slots, newest last
        self._delay_lock = threading.Lock()
        self._mic_at = 0.0  # capture time of the very first mic block
        self._reported = False

    # ---------------------------------------------------------------- lifecycle

    def start(self) -> bool:
        """Open the encoder and the tracks. Returns False if recording is unavailable."""
        try:
            if shutil.which("ffmpeg") is None:
                self._give_up("ffmpeg is not installed; this session will not be recorded")
                self._report()
                return False
            first = self._await_frame()
            if first is None:
                self._give_up("no frame to record; this session will not be recorded")
                self._report()
                return False

            self.work_dir.mkdir(parents=True, exist_ok=True)
            self._size = self._output_size(first)
            self._t0 = time.monotonic()
            self._user = _Track(self.work_dir / "user.wav", self._t0)
            self._agent = _Track(self.work_dir / "agent.wav", self._t0)
            self._start_encoder()
            self._spawn(self._video_loop, "record-video", first)
            self._spawn(self._writer_loop, "record-writer")
        except Exception as exc:  # never let a recording problem reach the session
            self._give_up(f"{type(exc).__name__}: {exc}")
            self._report()
            self._close_parts()
            return False
        return True

    def stop(self) -> Path | None:
        """Finish the recording and mux. Returns the finished mp4, or None if there is none."""
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=JOIN_TIMEOUT_S)
        self._threads.clear()
        try:
            self._flush_delay()
        except Exception as exc:
            self._give_up(f"{type(exc).__name__}: {exc}")
        self._close_parts()
        self._report()
        if self.failed:
            return None
        return self._mux()

    def _close_parts(self) -> None:
        for track in (self._user, self._agent):
            if track is not None:
                try:
                    track.close()
                except Exception:
                    pass  # nothing useful to do about it while tearing down
        self._user = self._agent = None
        proc, self._ffmpeg = self._ffmpeg, None
        if proc is not None:
            try:
                if proc.stdin is not None and not proc.stdin.closed:
                    proc.stdin.close()
                proc.wait(timeout=JOIN_TIMEOUT_S)
            except Exception:
                proc.kill()
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def _spawn(self, target, name: str, *args) -> None:
        thread = threading.Thread(target=target, name=name, args=args, daemon=True)
        self._threads.append(thread)
        thread.start()

    # ---------------------------------------------------------------- audio hooks

    def on_mic_block(self, block: bytes, admitted: int) -> None:
        """Hook for :meth:`cyclops.audio.Microphone._callback`. Runs on the audio thread.

        ``admitted`` is how many blocks the echo guard let through for this one input block: 0
        while Cyclops is talking, 1 normally, and a whole run of them when a barge-in releases
        its pre-roll. Exactly one slot is recorded per callback either way, so the track stays
        locked to the input clock - a release only goes back and un-mutes slots already queued.
        """
        if self.failed or self._user is None:
            return
        try:
            now = time.monotonic()
            if self._mic_at == 0.0:
                self._mic_at = now
            drained: list[tuple[bytes, bool]] = []
            with self._delay_lock:
                self._delay.append([block, admitted > 0])
                for i in range(1, min(admitted, len(self._delay))):
                    self._delay[-1 - i][1] = True  # the pre-roll did reach the server after all
                while len(self._delay) > DELAY_BLOCKS:
                    raw, ok = self._delay.popleft()
                    drained.append((raw, ok))
            for raw, ok in drained:
                self._user.append(raw if ok else bytes(len(raw)), self._mic_at)
        except Exception as exc:
            self.failed = self.failed or f"{type(exc).__name__}: {exc}"

    def on_speaker_block(self, block: bytes) -> None:
        """Hook for :meth:`cyclops.audio.Speaker._callback`, after its zero-fill."""
        if self.failed or self._agent is None:
            return
        try:
            self._agent.append(block, time.monotonic())
        except Exception as exc:
            self.failed = self.failed or f"{type(exc).__name__}: {exc}"

    def _flush_delay(self) -> None:
        """Push the last few hundred ms still sitting in the delay line."""
        if self._user is None:
            return
        with self._delay_lock:
            remaining = list(self._delay)
            self._delay.clear()
        for raw, ok in remaining:
            self._user.append(raw if ok else bytes(len(raw)), self._mic_at)

    # ---------------------------------------------------------------- video

    def _await_frame(self):
        deadline = time.monotonic() + FIRST_FRAME_TIMEOUT_S
        while time.monotonic() < deadline:
            frame = self._frames.frame()
            if frame is not None:
                return frame
            time.sleep(0.02)
        return None

    def _output_size(self, frame) -> tuple[int, int]:
        """Fit to the record width without upscaling; both dimensions even, for yuv420p.

        A width of 0 - the default - keeps the source's own, which is what makes recording the
        800x480 panel cost no resample at all: :meth:`_write_frame` only resizes a frame whose
        shape disagrees with this, and at 800 wide none of them do.

        Whatever comes back is fixed for the whole session, because it is the frame size the
        encoder was opened with. A window that changes size mid-session is therefore stretched
        back to this rather than breaking the stream - distorted, never fatal, and not reachable
        on the Pi, where the panel is one size from boot to shutdown.
        """
        height, width = frame.shape[:2]
        out_w = width if self._width <= 0 else min(self._width, width)
        out_h = max(2, round(height * out_w / width))
        return out_w - out_w % 2, out_h - out_h % 2

    def _start_encoder(self) -> None:
        width, height = self._size
        self._log_file = open(self.work_dir / "ffmpeg.log", "wb")
        self._ffmpeg = subprocess.Popen(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}", "-framerate", str(self._fps),
                "-i", "-",
                "-an",
                "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
                "-crf", CRF, "-pix_fmt", "yuv420p",
                str(self.work_dir / RAW_VIDEO),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._log_file,
        )

    def _video_loop(self, first) -> None:
        """Sample the camera on a fixed cadence, so video time tracks wall time exactly."""
        period = 1.0 / self._fps
        frame = first
        written = 0
        try:
            while not self._stop.is_set() and not self.failed:
                delay = (self._t0 + written * period) - time.monotonic()
                if delay > 0 and self._stop.wait(delay):
                    break
                latest = self._frames.frame()
                if latest is not None:
                    frame = latest  # otherwise the previous frame is written again, on purpose
                self._write_frame(frame)
                written += 1
        except Exception as exc:
            self.failed = self.failed or f"video: {type(exc).__name__}: {exc}"

    def _write_frame(self, frame) -> None:
        if frame.shape[1::-1] != self._size:
            frame = cv2.resize(frame, self._size, interpolation=cv2.INTER_AREA)
        proc = self._ffmpeg
        if proc is not None and proc.stdin is not None:
            proc.stdin.write(frame.tobytes())

    # ---------------------------------------------------------------- writer

    def _writer_loop(self) -> None:
        while not self._stop.wait(DRAIN_INTERVAL_S):
            if self.failed:
                self._report()
                return
            try:
                for track in (self._user, self._agent):
                    if track is not None:
                        track.drain()
            except Exception as exc:
                self.failed = f"writer: {type(exc).__name__}: {exc}"
                self._report()
                return

    # ---------------------------------------------------------------- finishing

    def _mux(self) -> Path | None:
        """Join the parts: copy the video, encode user|agent into one stereo track."""
        done = mux(self.work_dir, self.out_path)
        if not done.ok:
            self._give_up(f"mux failed ({done.why}); the parts are in {self.work_dir}")
            self._report()
            return None
        return self.out_path

    def _give_up(self, message: str) -> None:
        self.failed = self.failed or message

    def _report(self) -> None:
        """Print the failure once, from a thread that is allowed to do I/O."""
        if self.failed and not self._reported:
            self._reported = True
            print(f"· [record] {self.failed}", file=sys.stderr, flush=True)
