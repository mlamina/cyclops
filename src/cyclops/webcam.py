"""Webcam capture: grab one frame, shrink it, and return it as a JPEG data URL."""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import contextlib
import os
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2

from .overlay import sharpen

WARMUP_FRAMES = 10  # let auto-exposure/white-balance settle before the real shot
WARMUP_SECONDS = 0.4
MAX_EDGE = 1024
JPEG_QUALITY = 85
PROBE_INDICES = 3
USEEPLUS = "useeplus"  # the index reported for an endoscope, which has no /dev/video number
USEEPLUS_READ_TIMEOUT_S = 1.0  # generous at 20 fps, and bounds the retry on a dead device
KEEP_CAPTURES = 20  # timestamped archive files to keep besides latest.jpg

_camera_lock = threading.Lock()  # one capture at a time, even if a timed-out one is still running
_last_good_index: int | None = None  # remembered across captures when probing automatically
_live_source: object | None = None  # a cyclops.camera.CameraSource while the kiosk is running


def set_live_source(source: object | None) -> None:
    """Register an already-open camera (a :class:`~cyclops.camera.CameraSource`) to shoot from.

    The kiosk holds the device open for its live preview, so the tool cannot open it again.
    While a source is registered, captures borrow the sharpest of its recent frames - no open,
    no warm-up. Passing ``None`` restores the standalone open-warm-shoot-release path.
    """
    global _live_source
    _live_source = source


class WebcamError(RuntimeError):
    """Raised when the camera cannot be opened or produces no usable frame."""


@dataclass(frozen=True)
class Capture:
    data_url: str
    path: Path  # the timestamped file, never latest.jpg - the one still there tomorrow
    width: int
    height: int
    jpeg_bytes: int
    camera_index: int | str  # a /dev/video number, or USEEPLUS


def _candidate_indices(preferred: int | None) -> list[int]:
    if preferred is not None:
        return [preferred]  # an explicit choice is never silently overridden
    order = [] if _last_good_index is None else [_last_good_index]
    return order + [i for i in range(PROBE_INDICES) if i not in order]


@contextlib.contextmanager
def _quiet_probe():
    """Silence OpenCV's own chatter while we try indices that may well be empty.

    Looking for a camera that is not plugged in is a normal, repeating state - the live source
    retries every couple of seconds - and each failed open prints three lines from the C++ layer
    about a device it was asked to try. Left alone that buries the kiosk log in identical
    warnings and hides anything that actually matters. The level is restored afterwards, so a
    real failure somewhere else still says so.
    """
    api = getattr(cv2.utils, "logging", None)
    if api is None:  # older wheels have no logging control; the noise is the lesser problem
        yield
        return
    previous = api.getLogLevel()
    api.setLogLevel(api.LOG_LEVEL_SILENT)
    try:
        yield
    finally:
        api.setLogLevel(previous)


class _UseeplusCapture:
    """A useeplus endoscope wearing :class:`cv2.VideoCapture`'s clothes.

    The cheap endoscopes sold as "supercamera" are not UVC devices. Both their USB interfaces
    are vendor-specific, so no kernel driver binds them and no ``/dev/video*`` node is ever
    created - the device sits on the bus repeating a heartbeat at a host that never answers.
    They stream perfectly well once something speaks their protocol, which the ``supercamera``
    package does over libusb. All this class adds is the shape the rest of the module already
    expects - ``read()`` returning ``(ok, frame)``, and ``release()`` - so neither
    :class:`~cyclops.camera.CameraSource` nor the capture path has to know which kind of camera
    it was handed.

    The one thing worth doing here is failing *fast*. The underlying read retries internally
    until its own timeout before admitting defeat, and on an unplugged device that retry is a
    tight loop over an error that can never clear: a pegged core for a second, then again for
    every read the supervisor's grace count allows. So a failed read asks whether the device is
    still on the bus at all, and once it isn't, every later read says so immediately - which
    spends the remaining grace in microseconds and gets us back to looking for the camera.
    """

    def __init__(self, camera: object, still_present: Callable[[], bool]) -> None:
        self._camera = camera
        self._still_present = still_present
        self._gone = False

    def read(self) -> tuple[bool, object]:
        if self._gone:
            return False, None
        ok, frame = self._camera.read()
        if not ok and not self._still_present():
            self._gone = True  # unplugged: stop paying the retry timeout on every later read
        return ok, frame

    def release(self) -> None:
        self._camera.release()


def _open_useeplus() -> tuple[_UseeplusCapture, str]:
    """Open the first endoscope on the bus, or raise :class:`WebcamError` if there is none.

    Every failure here is the ordinary "no camera of this kind" answer, including an import
    that fails because the driver was never installed: this is one of two places a camera might
    be found, and neither is allowed to take the kiosk down by being absent.
    """
    try:
        from supercamera import Camera, list_devices
    except ImportError as exc:  # a venv without the driver simply has no endoscope to offer
        raise WebcamError(f"no useeplus driver installed ({exc})") from exc
    try:
        camera = Camera(timeout=USEEPLUS_READ_TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001 - RuntimeError when absent, USBError when unusable
        raise WebcamError(f"no useeplus endoscope ({exc})") from exc
    return _UseeplusCapture(camera, lambda: bool(list_devices())), USEEPLUS


def open_camera(preferred: int | None) -> tuple[cv2.VideoCapture | _UseeplusCapture, int | str]:
    """Open the preferred camera, or probe for one that actually delivers frames.

    On macOS the index order follows AVFoundation's uniqueID sort, so an idle iPhone
    (Continuity Camera) can sit at index 0 and "open" without ever returning a frame.

    A useeplus endoscope is looked for only once no ``/dev/video*`` has answered, because it
    cannot be probed the same way - it has no node to probe - and because a real webcam, when
    one is plugged in, should stay the camera you get.
    """
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    params = [cv2.CAP_PROP_FRAME_WIDTH, 1280, cv2.CAP_PROP_FRAME_HEIGHT, 720]
    candidates = _candidate_indices(preferred)
    with _quiet_probe():
        for index in candidates:
            cap = cv2.VideoCapture(index, backend, params)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    return cap, index
            cap.release()
    with contextlib.suppress(WebcamError):
        return _open_useeplus()
    hint = (
        "Check System Settings → Privacy & Security → Camera for your terminal app, or set "
        "CYCLOPS_CAMERA_INDEX."
        if sys.platform == "darwin"
        else "Check that it shows up in `lsusb`, and is a UVC camera or a useeplus endoscope."
    )
    raise WebcamError(
        f"No camera delivered a frame (tried indices {candidates}, and the USB bus for a "
        f"useeplus endoscope). {hint}"
    )


def _resize_to_max_edge(frame, max_edge: int):
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_edge:
        return frame
    scale = max_edge / longest
    return cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def _write_private(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        os.fchmod(fd, 0o600)  # photos of your room stay private, even if the file pre-existed
        f.write(data)


def _unique(path: Path) -> Path:
    """Two photos in the same second must not become one file."""
    if not path.exists():
        return path
    for n in range(2, 100):
        candidate = path.with_name(f"{path.stem}-{n}{path.suffix}")
        if not candidate.exists():
            return candidate
    return path  # a hundred shots inside one second is not a thing anyone does


def _age(path: Path) -> float:
    """Sort key for the prune. Deliberately mtime, not the name: the name format has changed
    once already, and two generations of it do not sort against each other the way you would
    guess (``2026-08-26_...`` sorts *before* ``20260826-...``, because ``-`` < ``0``). mtime
    cannot lie about which photo is oldest."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _save(save_dir: Path, jpeg: bytes, keep_as: str) -> Path:
    """Write the photo and return the path to the file that stays put.

    Two shapes, one function. With a ``keep_as`` role this is a session's ``photos/``:
    ``14-32-40_you.jpg``, kept forever, no ``latest.jpg`` - the folder *is* the archive, and
    pruning it would throw away half of what the session log points at. Without one it is the
    standalone ``captures/`` dir the smoke test and any session-less capture still use: a full
    date-time name, a ``latest.jpg`` beside it, and everything past KEEP_CAPTURES swept up so an
    unattended box cannot fill its card.

    Either way it returns the *timestamped* file, never ``latest.jpg`` - what the caller wants
    to log is the one that will still be there tomorrow. (It used to return ``latest.jpg``,
    which made every logged photo path identical.)
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    if keep_as:
        path = _unique(save_dir / f"{time.strftime('%H-%M-%S')}_{keep_as}.jpg")
        _write_private(path, jpeg)
        return path

    path = _unique(save_dir / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}.jpg")
    _write_private(path, jpeg)
    latest = save_dir / "latest.jpg"
    _write_private(latest, jpeg)
    archive = sorted((p for p in save_dir.glob("*.jpg") if p != latest), key=_age)
    for stale in archive[:-KEEP_CAPTURES]:
        stale.unlink(missing_ok=True)
    return path


def capture_image(
    camera_index: int | None,
    *,
    save_dir: Path,
    abort: threading.Event | None = None,
    keep_as: str = "",
) -> Capture:
    """Grab a photo, either from the live preview's camera or by opening one just for this shot.

    With a live source registered (the kiosk), this borrows the sharpest of its recent frames
    and returns in milliseconds. Otherwise it is the standalone path: open, warm up, shoot,
    release, which costs ~0.5-1.5 s on a MacBook.

    ``keep_as`` is who took the photo ("cyclops"/"you") when it belongs to a session and is
    being filed in that session's ``photos/``. Empty means the pruning ``captures/`` archive -
    see :func:`_save`. Callers get it from :func:`cyclops.session.photo_target` rather than
    deciding for themselves.
    """
    source = _live_source
    if source is not None:
        frame = source.snapshot()
        if frame is None:
            raise WebcamError(
                "The live preview's camera has no recent frame to photograph."
            )  # never fall back to opening it - the preview still holds the device
        return _encode_and_save(frame, save_dir, source.index, keep_as)
    with _camera_lock:
        return _capture_locked(camera_index, save_dir, abort, keep_as)


def _capture_locked(
    camera_index: int | None, save_dir: Path, abort: threading.Event | None, keep_as: str
) -> Capture:
    global _last_good_index
    cap, index = open_camera(camera_index)
    try:
        deadline = time.monotonic() + WARMUP_SECONDS
        frames_read = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames_read += 1
            if frames_read > WARMUP_FRAMES and time.monotonic() >= deadline:
                break
    finally:
        cap.release()

    if abort is not None and abort.is_set():
        raise WebcamError("capture abandoned (caller timed out)")
    if not ok or frame is None or frame.size == 0:
        raise WebcamError(f"Camera {index} opened but returned no frame.")
    if isinstance(index, int):  # USEEPLUS is not a number to hand cv2.VideoCapture next time
        _last_good_index = index

    return _encode_and_save(frame, save_dir, index, keep_as)


def _encode_and_save(frame, save_dir: Path, index: int | str, keep_as: str) -> Capture:
    """Sharpen, shrink, JPEG-encode and archive a frame, whoever grabbed it.

    The sharpen is the same one the preview gets and is here for the same reason: the endoscope
    is a 640x480 sensor behind a quality-44 encoder, and this is the only lever left. It matters
    more here than on the panel, because this frame is what the model is asked to read a part
    number off. It happens before the resize so that a frame big enough to be shrunk - a real
    webcam - is sharpened at full size and then cleanly reduced, rather than the other way about.
    """
    frame = _resize_to_max_edge(sharpen(frame), MAX_EDGE)
    encoded, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not encoded:
        raise WebcamError("JPEG encoding failed.")
    jpeg = buf.tobytes()
    saved = _save(save_dir, jpeg, keep_as)

    h, w = frame.shape[:2]
    data_url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
    return Capture(
        data_url=data_url,
        path=saved,
        width=w,
        height=h,
        jpeg_bytes=len(jpeg),
        camera_index=index,
    )


async def capture_image_async(
    camera_index: int | None, *, save_dir: Path, keep_as: str = ""
) -> Capture:
    """Run :func:`capture_image` on a daemon thread so the event loop keeps streaming audio.

    A daemon thread (rather than ``asyncio.to_thread``) means a camera stuck on a macOS
    permission prompt can't block interpreter exit. If the caller stops waiting (e.g.
    ``asyncio.timeout``), the abort flag makes the late capture discard its result instead
    of overwriting ``latest.jpg`` afterwards.
    """
    result: concurrent.futures.Future[Capture] = concurrent.futures.Future()
    abort = threading.Event()

    def work() -> None:
        if not result.set_running_or_notify_cancel():
            return
        try:
            result.set_result(
                capture_image(camera_index, save_dir=save_dir, abort=abort, keep_as=keep_as)
            )
        except BaseException as exc:  # noqa: BLE001 - forwarded to the awaiting task
            result.set_exception(exc)

    threading.Thread(target=work, name="webcam-capture", daemon=True).start()
    try:
        return await asyncio.wrap_future(result)
    finally:
        abort.set()  # no-op if the capture already finished
