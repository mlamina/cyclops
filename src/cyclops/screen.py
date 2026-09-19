"""The whole screen as a frame source: exactly what the compositor put on the glass.

A session's video is of the panel - the camera and its chrome, and then the diagram, the photo,
the scratchpad, the manual page scrolling under a finger - and all of it arrives here as one
picture, because it is taken off the compositor rather than rebuilt. ``wf-recorder`` runs for the
whole session, writing raw frames into a pipe, and a reader thread keeps the newest one. The
recorder samples it on its own clock exactly as it samples the camera (:mod:`cyclops.record`),
which is the whole seam: this is one more thing with a ``frame()``.

The command is the one measured on the Pi before this was built (2026-09-18): ``rawvideo`` in
``bgr0``, which is the compositor's own layout, so ``wf-recorder`` converts nothing and encodes
nothing - about 15% of a core at the ~29 fps the screen repaints at. Whatever the name says, the
bytes arrive red first (R, G, B, X - checked against a ``grim`` shot of the same screen), so a
frame is read as RGBA; read as BGRA, every red on the panel came out blue. A frame carries no header, so
its size is the panel's and has to be told; the kiosk knows it from the DRM mode.

The pipe is a FIFO in a scratch directory rather than stdout, because ``wf-recorder`` talks on
stdout. It also asks before writing to a file that already exists - a FIFO does - so it is
answered ``Y`` on stdin; the Pi's 0.3 has no flag to skip the question.

What this must never do is outlive its session, or take one down. :meth:`ScreenSource.stop` is
called from the session log's teardown, which runs however the session ended, and every failure
to start comes back as a sentence rather than an exception - the session records the camera
instead and writes that sentence down (:meth:`cyclops.session.SessionLog._filmed`).
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np

PROGRAM = "wf-recorder"
FIRST_FRAME_S = 1.0  # past this with nothing on the pipe, the session records the camera
POLL_S = 0.02
STOP_S = 2.0  # how long SIGINT gets before the capture is killed
BYTES_PER_PIXEL = 4  # R, G, B, X - see the module docstring


def command(program: str, fifo: Path) -> list[str]:
    """The capture, as measured on the Pi: raw frames in the compositor's own pixel layout."""
    return [program, "-c", "rawvideo", "-m", "rawvideo", "-x", "bgr0", "-f", str(fifo)]


class ScreenSource:
    """``wf-recorder`` for one session at a time: :meth:`start`, sample :meth:`frame`, :meth:`stop`.

    Built once by the kiosk and reused, so ``size`` is the panel's and ``program`` is only ever
    something else in a test.
    """

    def __init__(self, size: tuple[int, int], program: str = PROGRAM) -> None:
        self._size = size
        self._program = program
        self._proc: subprocess.Popen[bytes] | None = None
        self._reader: threading.Thread | None = None
        self._hold: int | None = None  # our own write end of the FIFO; see start()
        self._dir: Path | None = None
        self._frame = None
        self._first = threading.Event()

    def start(self, timeout: float = FIRST_FRAME_S) -> str:
        """Start capturing and wait for the first frame. ``""`` once one is in, else why not.

        Returns early when the capture dies, rather than sitting out the timeout. Whatever this
        says, :meth:`stop` is still the caller's to call - it is what tidies a failed start too.
        """
        self.stop()
        if shutil.which(self._program) is None:
            return f"{self._program} is not installed"
        try:
            fifo = self._open_pipe()
            self._spawn(fifo)
        except (OSError, subprocess.SubprocessError) as exc:
            return f"{self._program} did not start ({type(exc).__name__}: {exc})"
        return self._await_first(timeout)

    def frame(self):
        """The newest frame of the screen, BGR, or None before the first. See FrameSource."""
        return self._frame

    def stop(self) -> None:
        """End the capture and tidy up after it. Safe to call at any time, any number of times."""
        proc, self._proc = self._proc, None
        if proc is not None:
            _end(proc)
        hold, self._hold = self._hold, None
        if hold is not None:
            os.close(hold)  # the last writer gone: the reader sees EOF and returns
        reader, self._reader = self._reader, None
        if reader is not None:
            reader.join(timeout=STOP_S)
        scratch, self._dir = self._dir, None
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)
        self._frame = None
        self._first = threading.Event()

    # ---------------------------------------------------------------- starting

    def _open_pipe(self) -> Path:
        """The FIFO, with the reader already on it.

        Opened from this side first, both ways round. The read end without blocking, because
        nobody is writing yet; then a write end of our own, held until :meth:`stop`, because a
        FIFO read with no writer at all is an immediate EOF - and ``wf-recorder`` only opens the
        file once it has a frame to put in it.
        """
        self._dir = Path(tempfile.mkdtemp(prefix="cyclops-screen-"))
        fifo = self._dir / "frames"
        os.mkfifo(fifo)
        read_end = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
        os.set_blocking(read_end, True)
        self._hold = os.open(fifo, os.O_WRONLY)
        self._first = threading.Event()
        self._reader = threading.Thread(
            target=self._read, args=(read_end, self._first), name="screen-reader", daemon=True
        )
        self._reader.start()
        return fifo

    def _spawn(self, fifo: Path) -> None:
        assert self._dir is not None
        with open(self._dir / "log", "wb") as log:
            self._proc = subprocess.Popen(  # noqa: S603 - the command is ours, from command()
                command(self._program, fifo),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=log,
            )
        try:
            assert self._proc.stdin is not None
            self._proc.stdin.write(b"Y\n")  # "Output file exists. Overwrite?" - it is our FIFO
            self._proc.stdin.close()
        except OSError:
            pass  # it has already gone; _await_first says so

    def _await_first(self, timeout: float) -> str:
        deadline = time.monotonic() + timeout
        while not self._first.wait(POLL_S):
            proc = self._proc
            if proc is not None and proc.poll() is not None:
                return f"{self._program} exited ({proc.returncode}) before its first frame" + (
                    f": {said}" if (said := self._last_word()) else ""
                )
            if time.monotonic() > deadline:
                return f"no frame from {self._program} within {timeout:g} s"
        return ""

    def _last_word(self) -> str:
        """The last line the capture wrote to stderr, which is usually the reason it stopped."""
        if self._dir is None:
            return ""
        try:
            lines = (self._dir / "log").read_text(errors="replace").strip().splitlines()
        except OSError:
            return ""
        return lines[-1].strip() if lines else ""

    # ---------------------------------------------------------------- reading

    def _read(self, read_end: int, first: threading.Event) -> None:
        """Keep the newest whole frame, until the pipe ends. The reader thread's whole life.

        Each frame is a new array - ``cvtColor`` allocates - so one handed to the recorder is
        never written into again while it is being encoded. The buffer it came out of is reused.
        """
        width, height = self._size
        raw = bytearray(width * height * BYTES_PER_PIXEL)
        pixels = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, BYTES_PER_PIXEL)
        with open(read_end, "rb", buffering=0) as pipe:
            while _fill(pipe, raw):
                self._frame = cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGR)
                first.set()


def _fill(pipe, buf: bytearray) -> bool:
    """Read exactly one frame into ``buf``. False at the end of the pipe, or a torn last frame."""
    view = memoryview(buf)
    got = 0
    while got < len(buf):
        try:
            n = pipe.readinto(view[got:])
        except OSError:
            return False
        if not n:
            return False
        got += n
    return True


def _end(proc: subprocess.Popen[bytes]) -> None:
    """Ask the capture to finish the way Ctrl+C would, and make sure it has."""
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=STOP_S)
    except (OSError, subprocess.TimeoutExpired):
        proc.kill()
        proc.wait()
