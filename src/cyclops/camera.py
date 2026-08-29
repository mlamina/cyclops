"""A camera held open for the whole session, shared by the live preview and the webcam tool.

:mod:`cyclops.webcam` opens the device, warms it up and releases it for every single photo -
right for a one-shot capture, impossible once something is drawing a live preview, because a
second open of the same ``/dev/video0`` fails. :class:`CameraSource` owns the device instead:
one reader thread keeps the newest frame, the preview draws it, and the tool borrows it. That
also removes the per-photo warm-up, since auto-exposure settled long ago.

Holding the camera open also means there is a short history to be *picky* about. Handheld shots
of a small object are blurry a good fraction of the time, and the model never says "that photo
was too blurry to read" - it just answers worse. So the reader scores every frame for focus and
:meth:`CameraSource.snapshot` hands out the sharpest of the last fraction of a second rather
than whatever happened to arrive most recently. The preview still draws the newest frame, so
what you see stays live.

The device is also allowed to come and go. A USB camera can be unplugged mid-run and plugged
back in, so opening it is a supervisor thread's standing job rather than something ``start()``
does once: it retries until something delivers frames, and returns to retrying the moment the
device stops. Nothing above here has to restart anything - :attr:`CameraSource.connected` says
whether there is a camera right now, and the frames simply resume.
"""

from __future__ import annotations

import threading
import time
from collections import deque

import cv2

from .webcam import WebcamError, open_camera

STALE_AFTER_S = 2.0  # a frame older than this means the camera stopped delivering
READ_ERROR_GRACE = 30  # consecutive failed reads tolerated before calling the device gone
RECONNECT_EVERY_S = 2.0  # how often to look for a camera that is absent, or has come back
HISTORY = 12  # frames kept for the sharpest-of-recent pick (~0.5 s at 25 fps)
SHARP_WINDOW_S = 0.7  # only frames this fresh compete; older ones may show a different scene
FOCUS_WIDTH, FOCUS_HEIGHT = 320, 180  # score on a downscale: same ranking, ~1.5 ms on a Pi 5


def focus_score(frame) -> float:
    """How sharp a frame is: variance of its Laplacian, the standard focus measure.

    Blur suppresses high spatial frequencies, so the second derivative goes flat and its
    variance collapses. Absolute values mean nothing across scenes - only the ranking within
    one burst of frames matters, which is all we use it for.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (FOCUS_WIDTH, FOCUS_HEIGHT), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(small, cv2.CV_64F).var())


class CameraSource:
    """Opens one camera and keeps its most recent frame available to any thread."""

    def __init__(self, camera_index: int | None = None) -> None:
        self._preferred = camera_index
        self._lock = threading.Lock()
        self._frame = None  # the newest decoded frame, BGR
        self._stamp = 0.0  # time.monotonic() when it arrived
        self._recent: deque[tuple[object, float, float]] = deque(maxlen=HISTORY)
        self.last_pick: tuple[float, float, float] | None = None  # (score, newest_score, age_s)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._cap: cv2.VideoCapture | None = None
        self._index: int | str | None = None
        self._error = ""
        self._generation = 0  # bumped by start(); a supervisor with a stale one retires itself

    @property
    def index(self) -> int | str | None:
        """What delivered frames: a ``/dev/video`` number, ``webcam.USEEPLUS``, or None."""
        return self._index

    @property
    def error(self) -> str:
        """Why there is no camera at the moment, or '' while one is delivering frames."""
        return self._error

    @property
    def connected(self) -> bool:
        """Whether a device is open right now. False between unplugging and the next one."""
        with self._lock:
            return self._cap is not None

    @property
    def live(self) -> bool:
        """Whether a fresh frame is available - open *and* actually delivering."""
        got = self.latest()
        return got is not None and time.monotonic() - got[1] <= STALE_AFTER_S

    def start(self) -> None:
        """Begin looking for a camera, and keep looking. Never raises.

        The device may be absent now and plugged in a minute from now, so this only starts the
        supervisor; it is that thread which opens, reads, and reopens. Callers that need a frame
        before they can continue follow this with :meth:`wait_for_frame`.

        Each supervisor carries the generation it was started under. :meth:`stop` cannot promise
        the old one has exited - it may be several seconds inside a blocking open of a device
        that is warming up - and clearing the stop event for a new run would otherwise revive it,
        leaving two threads reading one camera. A supervisor whose generation is no longer the
        current one stands down instead, whatever the stop event says.
        """
        if self._thread is not None and self._thread.is_alive() and not self._stop.is_set():
            return  # already supervising
        self._generation += 1
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._supervise, args=(self._generation,), name="camera-source", daemon=True
        )
        self._thread.start()

    def latest(self) -> tuple[object, float] | None:
        """The newest frame and its monotonic timestamp, or None if none has arrived yet."""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame, self._stamp

    def frame(self):
        """Just the newest frame, or None. Convenience for the render loop."""
        got = self.latest()
        return None if got is None else got[0]

    def snapshot(self):
        """The sharpest frame from the last moment, or None if the camera has gone stale.

        Falls back to the newest frame whenever there is no history to choose from, so this is
        never worse than taking the latest.
        """
        with self._lock:
            if self._frame is None:
                return None
            newest, newest_stamp = self._frame, self._stamp
            recent = list(self._recent)
        if time.monotonic() - newest_stamp > STALE_AFTER_S:
            return None

        cutoff = newest_stamp - SHARP_WINDOW_S
        candidates = [(f, s, sc) for f, s, sc in recent if s >= cutoff]
        if not candidates:
            self.last_pick = None
            return newest
        best_frame, best_stamp, best_score = max(candidates, key=lambda c: c[2])
        newest_score = recent[-1][2]  # the reader appends in order, so the last is the newest
        self.last_pick = (best_score, newest_score, newest_stamp - best_stamp)
        return best_frame

    def stop(self) -> None:
        """Stop the reader and release the device so a later run can open it again."""
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._forget()

    def __enter__(self) -> CameraSource:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def _supervise(self, token: int) -> None:
        """Hold a camera open for as long as the source is running, through unplugs.

        Every attempt probes again rather than reusing the index it left on: a camera plugged
        back in can land on a different ``/dev/video*`` than the one it came up as, so the
        index has to be rediscovered along with the device. Failing to find one is not an error
        to stop on - it is a state to sit in, and to leave again when a camera turns up.
        """
        while not self._stop.is_set() and self._generation == token:
            try:
                cap, index = open_camera(self._preferred)
            except WebcamError as exc:
                self._error = str(exc)
                self._forget()  # whatever it was showing is now last minute's room
                self._stop.wait(RECONNECT_EVERY_S)
                continue
            if self._stop.is_set() or self._generation != token:
                cap.release()  # retired while that open was blocking; never adopt the device
                return
            if self._error:  # we had been looking, so this is news; a first open is not
                print(f"· camera {index} found", flush=True)
            with self._lock:
                self._cap, self._index, self._error = cap, index, ""
            try:
                self._read_loop(cap, token)
            finally:
                with self._lock:
                    self._cap = None
                cap.release()
                self._forget()

    def _forget(self) -> None:
        """Drop the frames from before the device went away, so none is drawn or photographed."""
        with self._lock:
            self._frame = None
            self._stamp = 0.0
            self._recent.clear()
        self.last_pick = None

    def _read_loop(self, cap, token: int) -> None:
        failures = 0
        while not self._stop.is_set() and self._generation == token:
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                failures += 1
                if failures > READ_ERROR_GRACE:
                    self._error = f"camera {self._index} stopped delivering frames"
                    print(f"· camera {self._index} unplugged - looking for it again", flush=True)
                    return
                time.sleep(0.02)
                continue
            failures = 0
            score = focus_score(frame)
            now = time.monotonic()
            with self._lock:
                self._frame, self._stamp = frame, now
                self._recent.append((frame, now, score))

    def wait_for_frame(self, timeout: float = 5.0) -> None:
        """Block until the first frame arrives, so the UI never opens on a black window."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.latest() is not None:
                return
            if self._error:
                raise WebcamError(self._error)
            time.sleep(0.02)
        raise WebcamError(f"camera {self._index} delivered no frame within {timeout:g}s")
