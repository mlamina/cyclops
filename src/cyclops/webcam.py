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
from dataclasses import dataclass
from pathlib import Path

import cv2

WARMUP_FRAMES = 10  # let auto-exposure/white-balance settle before the real shot
WARMUP_SECONDS = 0.4
MAX_EDGE = 1024
JPEG_QUALITY = 85
PROBE_INDICES = 3
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
    camera_index: int


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


def open_camera(preferred: int | None) -> tuple[cv2.VideoCapture, int]:
    """Open the preferred camera, or probe for one that actually delivers frames.

    On macOS the index order follows AVFoundation's uniqueID sort, so an idle iPhone
    (Continuity Camera) can sit at index 0 and "open" without ever returning a frame.
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
    raise WebcamError(
        f"No camera delivered a frame (tried indices {candidates}). Check System Settings → "
        "Privacy & Security → Camera for your terminal app, or set CYCLOPS_CAMERA_INDEX."
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
        raise WebcamError(f"Camera index {index} opened but returned no frame.")
    _last_good_index = index

    return _encode_and_save(frame, save_dir, index, keep_as)


def _encode_and_save(frame, save_dir: Path, index: int, keep_as: str) -> Capture:
    """Shrink, JPEG-encode and archive a frame, whoever grabbed it."""
    frame = _resize_to_max_edge(frame, MAX_EDGE)
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
