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

**Two owners want this alive, on different clocks**, so it is *held* rather than started: a
session for as long as it records, and a companion viewer (:mod:`cyclops.companion`) for as long
as its socket is open. :meth:`ScreenSource.acquire` and :meth:`ScreenSource.release` are the
whole of the public lifecycle, and the capture ends on the last release and at no other moment.
That is what stops either owner pulling it out from under the other - a session starting mid-view
must not restart the capture a phone is watching, and a session ending must not end it. There is
never more than one ``wf-recorder``, whoever is holding it.

What this must never do is outlive its last owner, or take one down. :meth:`_release_screen` in
the session log's teardown runs however the session ended, the encoder thread releases in a
``finally``, and every failure to start comes back as a sentence rather than an exception - the
session records the camera instead and writes that sentence down
(:meth:`cyclops.session.SessionLog._filmed`), and the companion streams the camera too.
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
    """One ``wf-recorder``, shared: :meth:`acquire`, sample :meth:`frame`, :meth:`release`.

    Built once by the kiosk and handed to everything that wants the glass, so ``size`` is the
    panel's and ``program`` is only ever something else in a test.
    """

    def __init__(self, size: tuple[int, int], program: str = PROGRAM) -> None:
        self._size = size
        self._program = program
        self._proc: subprocess.Popen[bytes] | None = None
        self._reader: threading.Thread | None = None
        self._hold: int | None = None  # our own write end of the FIFO; see _start()
        self._dir: Path | None = None
        self._latest: tuple | None = None
        self._first = threading.Event()
        # Guards the lease and nothing else - see acquire(). Deliberately not held over frame(),
        # latest() or connected: those are read by the recorder's sampling thread and by the
        # encoder thread, and parking a frame consumer behind a lifecycle event that can take
        # STOP_S to finish would stutter a recording to tidy up after somebody else.
        self._lock = threading.Lock()
        self._leases = 0

    # ---------------------------------------------------------------- the lease

    def acquire(self, timeout: float = FIRST_FRAME_S) -> str:
        """Hold the capture up while you need it: ``""`` once it is running, else why it is not.

        The first owner through starts it and waits for a first frame; everyone after joins the
        one already running and pays nothing at all. Refusal grants no hold, and tidies its own
        failed start - which is a change from the old :meth:`_start`, where cleaning up was the
        caller's job. It has to be: a caller that was refused must not be able to reach a capture
        somebody else is holding.

        A lease is a promise that something is delivering, so a held capture that has since died
        is refused rather than handed over. Otherwise a session would be told it had the screen
        and would file a recording of one frozen frame under ``source: "screen"``.
        """
        with self._lock:
            if self._leases and self.connected:
                self._leases += 1
                return ""
            if self._leases:
                # Held, but the process underneath has gone. Not restarted here: the owner still
                # holding it would have the capture swapped under them mid-frame.
                return f"{self._program} is no longer running"
            why = self._start(timeout)
            if why:
                self._stop()
                return why
            self._leases = 1
            return ""

    def release(self) -> None:
        """Give up one hold; the capture ends with the last of them. Unheld, this does nothing.

        It counts holds and cannot tell whose it is dropping, so an owner that may not have taken
        one keeps its own flag rather than calling this hopefully - see
        :meth:`cyclops.session.SessionLog._release_screen`, where getting that wrong would end a
        companion's stream.
        """
        with self._lock:
            if not self._leases:
                return
            self._leases -= 1
            if not self._leases:
                self._stop()

    @property
    def connected(self) -> bool:
        """Whether a capture is running - the question ``CameraSource`` answers about a device.

        The process and not the reader thread: :meth:`_open_pipe` holds its own write end of the
        FIFO open until :meth:`_stop`, so the reader never sees EOF even once ``wf-recorder``
        has died, and an alive reader would say yes about a capture that is gone.
        """
        proc = self._proc
        return proc is not None and proc.poll() is None

    @property
    def size(self) -> tuple[int, int]:
        """The panel's, as the kiosk read it off the DRM mode."""
        return self._size

    # ---------------------------------------------------------------- the capture

    def _start(self, timeout: float = FIRST_FRAME_S) -> str:
        """Start capturing and wait for the first frame. ``""`` once one is in, else why not.

        Returns early when the capture dies, rather than sitting out the timeout. Private because
        the lease is the only safe way in: a bare start would restart a capture another owner is
        watching, which is the whole thing :meth:`acquire` exists to prevent.
        """
        self._stop()
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
        got = self._latest
        return None if got is None else got[0]

    def latest(self) -> tuple | None:
        """The newest frame and when it arrived, or None before the first.

        Exactly ``CameraSource.latest()``'s shape, so anything that samples one can sample the
        other without knowing which it has - which is how :mod:`cyclops.companion` shows the
        glass and falls back to the sensor through a single call.
        """
        return self._latest

    def _stop(self) -> None:
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
        self._latest = None
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

        The frame and its stamp are one rebinding, not two, so a reader can never be handed this
        frame with the last one's clock.
        """
        width, height = self._size
        raw = bytearray(width * height * BYTES_PER_PIXEL)
        pixels = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, BYTES_PER_PIXEL)
        with open(read_end, "rb", buffering=0) as pipe:
            while _fill(pipe, raw):
                self._latest = (cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGR), time.monotonic())
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
