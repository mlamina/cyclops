"""Webcam capture: grab one frame, shrink it, and return it as a JPEG data URL."""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
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


class WebcamError(RuntimeError):
    """Raised when the camera cannot be opened or produces no usable frame."""


@dataclass(frozen=True)
class Capture:
    data_url: str
    path: Path
    width: int
    height: int
    jpeg_bytes: int
    camera_index: int


def _candidate_indices(preferred: int | None) -> list[int]:
    if preferred is not None:
        return [preferred]  # an explicit choice is never silently overridden
    order = [] if _last_good_index is None else [_last_good_index]
    return order + [i for i in range(PROBE_INDICES) if i not in order]


def _open_camera(preferred: int | None) -> tuple[cv2.VideoCapture, int]:
    """Open the preferred camera, or probe for one that actually delivers frames.

    On macOS the index order follows AVFoundation's uniqueID sort, so an idle iPhone
    (Continuity Camera) can sit at index 0 and "open" without ever returning a frame.
    """
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    params = [cv2.CAP_PROP_FRAME_WIDTH, 1280, cv2.CAP_PROP_FRAME_HEIGHT, 720]
    candidates = _candidate_indices(preferred)
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


def _save(save_dir: Path, jpeg: bytes) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
    _write_private(save_dir / f"{stamp}.jpg", jpeg)
    latest = save_dir / "latest.jpg"
    _write_private(latest, jpeg)
    archive = sorted(p for p in save_dir.glob("*.jpg") if p != latest)
    for old in archive[:-KEEP_CAPTURES]:
        old.unlink(missing_ok=True)
    return latest


def capture_image(
    camera_index: int | None, *, save_dir: Path, abort: threading.Event | None = None
) -> Capture:
    """Blocking capture (~0.5-1.5 s on a MacBook): open, warm up, grab a frame, encode JPEG."""
    with _camera_lock:
        return _capture_locked(camera_index, save_dir, abort)


def _capture_locked(
    camera_index: int | None, save_dir: Path, abort: threading.Event | None
) -> Capture:
    global _last_good_index
    cap, index = _open_camera(camera_index)
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

    frame = _resize_to_max_edge(frame, MAX_EDGE)
    encoded, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not encoded:
        raise WebcamError("JPEG encoding failed.")
    jpeg = buf.tobytes()
    latest = _save(save_dir, jpeg)

    h, w = frame.shape[:2]
    data_url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
    return Capture(
        data_url=data_url,
        path=latest,
        width=w,
        height=h,
        jpeg_bytes=len(jpeg),
        camera_index=index,
    )


async def capture_image_async(camera_index: int | None, *, save_dir: Path) -> Capture:
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
            result.set_result(capture_image(camera_index, save_dir=save_dir, abort=abort))
        except BaseException as exc:  # noqa: BLE001 - forwarded to the awaiting task
            result.set_exception(exc)

    threading.Thread(target=work, name="webcam-capture", daemon=True).start()
    try:
        return await asyncio.wrap_future(result)
    finally:
        abort.set()  # no-op if the capture already finished
