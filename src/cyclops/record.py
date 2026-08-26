"""Recording a whole kiosk session: the raw camera, plus the two sides of the conversation.

A session leaves one ``recordings/<timestamp>.mp4`` behind - H.264 video of what the camera saw,
with a stereo audio track carrying the user on the left channel and Cyclops on the right. Keeping
the two voices apart means neither is mixed into the other (listen to one side alone, or mix them
down later), and it sidesteps the double-counting you would get from recording an open microphone
that is also hearing the speaker.

Three streams have to line up, and each arrives on its own clock:

* video is sampled here on the wall clock at a fixed rate - re-writing the previous frame when the
  camera has not produced a new one, so video time never drifts away from wall time;
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
from pathlib import Path
from typing import Protocol

import cv2

from .audio import BYTES_PER_FRAME, SAMPLE_RATE, EchoGuard

DEFAULT_FPS = 15
DEFAULT_WIDTH = 640  # 640 wide at 15 fps measures ~4% of one Pi 5 core with libx264 ultrafast
CRF = "26"
# The guard releases up to PREROLL_BLOCKS at once on a barge-in; the delay line has to be able to
# reach back over all of them, so it is exactly that deep.
DELAY_BLOCKS = EchoGuard.PREROLL_BLOCKS
DRAIN_INTERVAL_S = 0.25
FIRST_FRAME_TIMEOUT_S = 1.0
JOIN_TIMEOUT_S = 5.0
MUX_TIMEOUT_S = 120.0


class FrameSource(Protocol):
    """What the recorder needs from a camera: the newest frame, or None."""

    def frame(self): ...


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
        recordings_dir: Path,
        *,
        fps: int = DEFAULT_FPS,
        width: int = DEFAULT_WIDTH,
    ) -> None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.out_path = recordings_dir / f"{stamp}.mp4"
        # A dotted work directory: an interrupted session is obvious, and its parts survive.
        self.work_dir = recordings_dir / f".{stamp}"
        self.failed = ""  # non-empty once recording has given up; the session carries on
        self._frames = frames
        self._fps = max(1, fps)
        self._width = max(2, width)
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
                self._give_up("camera delivered no frame; this session will not be recorded")
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
        """Fit to the record width without upscaling; both dimensions even, for yuv420p."""
        height, width = frame.shape[:2]
        out_w = min(self._width, width)
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
                str(self.work_dir / "video.mp4"),
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
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(self.work_dir / "video.mp4"),
            "-i", str(self.work_dir / "user.wav"),
            "-i", str(self.work_dir / "agent.wav"),
            "-filter_complex", "[1:a][2:a]join=inputs=2:channel_layout=stereo[a]",
            "-map", "0:v", "-map", "[a]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "96k",
            "-shortest", "-movflags", "+faststart",
            str(self.out_path),
        ]
        try:
            done = subprocess.run(command, capture_output=True, timeout=MUX_TIMEOUT_S, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            self._give_up(f"mux failed ({exc}); the parts are in {self.work_dir}")
            self._report()
            return None
        if done.returncode != 0:
            detail = done.stderr.decode(errors="replace").strip().splitlines()
            tail = detail[-1] if detail else f"exit {done.returncode}"
            self._give_up(f"mux failed ({tail}); the parts are in {self.work_dir}")
            self._report()
            return None
        shutil.rmtree(self.work_dir, ignore_errors=True)
        return self.out_path

    def _give_up(self, message: str) -> None:
        self.failed = self.failed or message

    def _report(self) -> None:
        """Print the failure once, from a thread that is allowed to do I/O."""
        if self.failed and not self._reported:
            self._reported = True
            print(f"· [record] {self.failed}", file=sys.stderr, flush=True)
