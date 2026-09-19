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

The newest frame is also steadied (:mod:`cyclops.steady`): the panel, the admin page and a
CAMERA recording get a window of it that follows the hand's smoothed path rather than its shake.
Photos never do - :meth:`CameraSource.snapshot` hands out raw frames, whole and unshifted, because
a mark the model makes is a fraction of the picture it was shown.

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

from . import steady
from .webcam import DEFAULT_FRAMING, FRAMINGS, RPICAM, WebcamError, open_camera

STALE_AFTER_S = 2.0  # a frame older than this means the camera stopped delivering
RECONNECT_EVERY_S = 2.0  # how often to look for a camera that is absent, or has come back
# How long stop() waits for the reader to come back before giving up on it. Comfortably
# more than a healthy read (33 ms at 30 fps) and deliberately less than a stalled one,
# which _read_loop's docstring measures at ten seconds: waiting that out would push the
# whole teardown past the ten seconds start-kiosk.sh allows before it sends SIGKILL, and
# a SIGKILL is the unclean exit this is all trying to avoid.
CLOSE_WAIT_S = 3.0
HISTORY = 24  # frames kept for the sharpest-of-recent pick - 0.8 s at the 30 fps we ask for
# HISTORY is a count and SHARP_WINDOW_S is a duration, so the two are tied to webcam.FRAME_RATE:
# twenty-four frames just covers the window at 30 fps, and at 15 fps would cover twice it.
SHARP_WINDOW_S = 0.7  # only frames this fresh compete; older ones may show a different scene
FOCUS_WIDTH, FOCUS_HEIGHT = 320, 180  # score on a downscale: same ranking, ~1.5 ms on a Pi 5


def thumbnail(frame):
    """The frame in grey at 320x180 - what the focus score and the stabilizer both work on."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (FOCUS_WIDTH, FOCUS_HEIGHT), interpolation=cv2.INTER_AREA)


def focus_score(small) -> float:
    """How sharp a frame is, from its :func:`thumbnail`: variance of its Laplacian.

    Blur suppresses high spatial frequencies, so the second derivative goes flat and its
    variance collapses. Absolute values mean nothing across scenes - only the ranking within
    one burst of frames matters, which is all we use it for.
    """
    return float(cv2.Laplacian(small, cv2.CV_64F).var())


class Film:
    """What a CAMERA-mode session records: the steadied picture in a fixed 1120x630 window.

    A :class:`cyclops.record.FrameSource` of its own rather than the camera itself, because the
    panel's window reaches into the edge fill (under the metal, where nobody sees it) and a
    recording has no metal to hide anything behind.
    """

    def __init__(self, source: CameraSource) -> None:
        self._source = source

    def frame(self):
        """The newest frame's recording window, or None."""
        return self._source.film_frame()


class CameraSource:
    """Opens one camera and keeps its most recent frame available to any thread."""

    def __init__(self, camera_index: int | None = None) -> None:
        self._preferred = camera_index
        self._lock = threading.Lock()
        self._frame = None  # the newest decoded frame, BGR, raw
        self._live = None  # ...and the steadied window of it the panel is shown
        self._shift = (0.0, 0.0)  # where that window is centred off the frame's; None: raw
        self.steady_note = steady.STEADY_FILE  # the settings screen's STABILIZE switch
        self._stamp = 0.0  # time.monotonic() when it arrived
        self._recent: deque[tuple[object, float, float]] = deque(maxlen=HISTORY)
        self.last_pick: tuple[float, float, float] | None = None  # (score, newest_score, age_s)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._cap: cv2.VideoCapture | None = None
        self._index: int | str | None = None
        self._framing = DEFAULT_FRAMING  # read by the reader thread, which stands down for it
        self._error = ""
        self._generation = 0  # bumped by start(); a supervisor with a stale one retires itself
        self.film = Film(self)

    @property
    def index(self) -> int | str | None:
        """What delivered frames: a ``/dev/video`` number, ``webcam.USEEPLUS``,
        ``webcam.RPICAM``, or None."""
        return self._index

    @property
    def framing(self) -> str:
        """Which of :data:`webcam.FRAMINGS` the module is running - always ``wide`` elsewhere."""
        return self._framing

    def cycle_framing(self) -> str | None:
        """Move to the next framing, or None if this camera has only the one.

        The change is a *request*, not an action: a sensor mode is fixed when the pipeline is
        configured, so what actually happens is that the reader sees a framing it was not started
        under, stands down, and the supervisor opens the module again with the new flags. About a
        second, all of it in :meth:`_supervise`, and the panel keeps drawing the last frame
        throughout - see the ``finally`` there, which is the only reason this does not flash the
        "no camera" card at every tap.

        None for a webcam or an endoscope, because they have one lens at one angle and there is
        nothing to cycle. Answering None rather than restarting them for no change matters: a
        C920 taken down and put back up is the exact move that wedges it (see :meth:`stop`).
        """
        if self._index != RPICAM:
            return None
        order = list(FRAMINGS)
        self._framing = order[(order.index(self._framing) + 1) % len(order)]
        return self._framing

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
        """The newest frame, steadied, and its monotonic timestamp, or None if none has arrived."""
        with self._lock:
            if self._live is None:
                return None
            return self._live, self._stamp

    def frame(self):
        """Just the newest frame, steadied, or None. Convenience for the render loop."""
        got = self.latest()
        return None if got is None else got[0]

    def film_frame(self):
        """The newest frame as a CAMERA recording frames it - see :class:`Film`."""
        with self._lock:
            raw, shift = self._frame, self._shift
        if raw is None or shift is None:
            return raw  # switched off: the whole frame, as a recording was before stabilizing
        return steady.film(raw, shift)

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
        """Stop the reader and release the device so a later run can open it again.

        **The device is released by the thread that owns it, and by nothing else.** That is the
        whole of this method, and it used to do the opposite.

        It used to join the reader for two seconds and then call ``self._cap.release()`` itself.
        That release looks like belt and braces and is the opposite: ``_supervise`` clears
        ``_cap`` in its own ``finally`` before releasing, so on a clean exit there is nothing left
        here to release. The only way that line ever ran was with the reader thread still alive -
        which means still inside ``cap.read()``, on the same ``VideoCapture`` we were then
        releasing from this thread.

        Two threads in one ``VideoCapture`` is how a UVC device gets left mid-stream, and a C920
        left mid-stream does not come back: it stays on the bus, answers ``lsusb``, and gives
        ``uvcvideo ... Failed to set UVC probe control : -110`` for ever with no ``/dev/video``
        node at all. Only a USB re-enumerate clears it. That happened on 2026-09-05, after three
        kiosk restarts in a row, and it was not a rare race: ``_read_loop``'s own docstring
        measures a stalled read at ten seconds against this two-second join, so any camera that
        was already unhappy hit it every single time.

        So if the reader will not come back, do nothing at all. The process is on its way out and
        the kernel closes the fd on exit, which stops the stream from the one place that cannot
        race a reader. A device released late beats a device wedged now.
        """
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=CLOSE_WAIT_S)
            if thread.is_alive():
                print(
                    "· camera reader is still in a read; leaving the device to the process exit",
                    flush=True,
                )
                self._forget()
                return
        if self._cap is not None:  # no reader ever ran; nobody else can be holding this
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
            framing = self._framing
            try:
                cap, index = open_camera(self._preferred, framing)
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
                self._read_loop(cap, token, framing)
            finally:
                with self._lock:
                    self._cap = None
                cap.release()
                if self._framing == framing:
                    self._forget()  # the device went; what it was showing is now last minute's
                # Otherwise this is a framing change we asked for, and the frames are kept on
                # purpose: the reopen below takes about a second, and dropping them would put
                # the "no camera" card up in the middle of a deliberate gesture. A second is
                # comfortably inside STALE_AFTER_S, so nothing downstream believes it is live.

    def _forget(self) -> None:
        """Drop the frames from before the device went away, so none is drawn or photographed."""
        with self._lock:
            self._frame = self._live = None
            self._shift = (0.0, 0.0)
            self._stamp = 0.0
            self._recent.clear()
        self.last_pick = None

    def _read_loop(self, cap, token: int, framing: str = DEFAULT_FRAMING) -> None:
        """Read frames until the device stops giving them, then return so it is opened again.

        The grace is a *duration*, and deliberately the same one the panel calls a frame stale
        after: the moment what is on screen stops being a picture of the room is the moment this
        stops believing the handle it is holding. It was a count of thirty failed reads once,
        which sounds equivalent and is not - a V4L2 read of a camera that has stalled blocks for
        ten seconds before it fails, so thirty of them is five minutes of a frozen panel with
        nothing to say it was frozen. That is not a hypothetical: a C920 stalled mid-session on
        2026-08-31 and the preview showed one frame for the rest of the session.

        Timed from *before* the read for the same reason. The ten seconds a stalled read spends
        waiting are ten seconds with no frame, and starting the clock when it returns would cost
        a second timeout to notice the first.

        Every entry is a fresh open - a framing change, a reconnect, a wake - so the stabilizer
        starts from zero here too: the last device's shake says nothing about this one's. So does
        switching it back on: the frames it missed while off say nothing about where the hand is.
        """
        steadier = steady.Steady()
        steadying, looked = steady.enabled(self.steady_note), time.monotonic()
        failing_since = 0.0
        while not self._stop.is_set() and self._generation == token and self._framing == framing:
            attempted = time.monotonic()
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                failing_since = failing_since or attempted
                if time.monotonic() - failing_since > STALE_AFTER_S:
                    self._error = f"camera {self._index} stopped delivering frames"
                    print(
                        f"· camera {self._index} stopped delivering - looking for it again",
                        flush=True,
                    )
                    return
                time.sleep(0.02)
                continue
            failing_since = 0.0
            if attempted - looked >= steady.NOTE_POLL_S:
                looked, wanted = attempted, steady.enabled(self.steady_note)
                if wanted != steadying:
                    steadier, steadying = steady.Steady(), wanted
                    print(f"· stabilize {'on' if wanted else 'off'}", flush=True)
            small = thumbnail(frame)
            score = focus_score(small)
            if steadying:
                shift = steadier.update(small, frame.shape[1], frame.shape[0])
                shown = steady.live(frame, shift)
            else:
                shift, shown = None, frame
            now = time.monotonic()
            with self._lock:
                self._frame, self._live, self._shift, self._stamp = frame, shown, shift, now
                self._recent.append((frame, now, score))  # raw: photos are never steadied

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
