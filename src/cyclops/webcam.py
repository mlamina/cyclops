"""Webcam capture: grab one frame, shrink it, and return it as a JPEG data URL."""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import contextlib
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import devices, flip
from .overlay import sharpen

WARMUP_FRAMES = 10  # let auto-exposure/white-balance settle before the real shot
WARMUP_SECONDS = 0.4
MAX_EDGE = 1024
JPEG_QUALITY = 85
PROBE_INDICES = 3
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
FRAME_RATE = 30  # asked of the camera, not the reader; see the format note in open_camera
USEEPLUS = "useeplus"  # the index reported for an endoscope, which has no /dev/video number
USEEPLUS_READ_TIMEOUT_S = 1.0  # generous at 20 fps, and bounds the retry on a dead device
RPICAM = "rpicam"  # the index reported for a CSI module, which has no capture node either
# The endoscopes cyclops drives over libusb - the same two IDs deploy/99-useeplus-camera.rules
# grants access to, and the same two cyclops/extensions/endoscope.py declares. Here and not read
# out of the extension, because this is how the camera is opened and a broken extension must not
# be a dead camera; a test keeps the three lists in step. Here so that "is one plugged in?" can be
# answered off sysfs without opening the device; see outranked().
USEEPLUS_IDS = ("2ce3:3828", "0329:2022")
RPICAM_ROTATION = 180  # the module is mounted upside down in the case; see _open_rpicam
RPICAM_AF = "continuous"  # a fixed camera watching a changing bench, not a shutter to half-press
# The three framings the panel cycles between, as the rpicam-vid flags each one takes. Only the
# module has them: a webcam is one lens at one angle, and there is nothing here to choose from.
# Every step is a smaller piece of the *sensor* rather than a bigger piece of the same frame, so
# each one is real detail arriving and not a magnified version of the last - see _open_rpicam.
WIDE, NARROW, ZOOMED = "wide", "narrow", "zoomed"
FRAMINGS: dict[str, list[str]] = {
    WIDE: ["--mode", "2304:1296"],  # the whole sensor, binned 2x2. 1x, and what the lens is for
    NARROW: ["--mode", "1536:864"],  # the sensor's own centre crop, 1.5x, still every pixel of it
    # 1:1 sensor pixels: 1280x720 of the full 4608x2592 frame, which is 0.278 of each edge and
    # centred, so 3.6x with nothing interpolated. The mode tops out at 14.35 fps against the 30
    # we ask for, which is the whole of what this framing costs.
    ZOOMED: ["--mode", "4608:2592", "--roi", "0.361,0.361,0.278,0.278"],
}
DEFAULT_FRAMING = WIDE
V4L_NODES = Path("/sys/class/video4linux")  # absent off Linux, which is what picks the fallback
PIPE_CHUNK = 1 << 16  # a pipe hands back its buffer, not your frame; see _RpicamCapture._fill
KEEP_CAPTURES = 20  # timestamped archive files to keep besides latest.jpg
CAPTURE_TIMEOUT_S = 12.0  # generous: a cold open can sit behind a macOS permission prompt

_camera_lock = threading.Lock()  # one capture at a time, even if a timed-out one is still running
_last_good_index: int | None = None  # remembered across captures when probing automatically
_live_source: object | None = None  # a cyclops.camera.CameraSource while the kiosk is running


def set_live_source(source: object | None) -> None:
    """Register an already-open camera (a :class:`~cyclops.camera.CameraSource`) to shoot from.

    The kiosk holds the device open for its live preview, so a capture cannot open it again.
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
    camera_index: int | str  # a /dev/video number, or USEEPLUS, or RPICAM


def _usb_video_nodes() -> list[int]:
    """The ``/dev/video*`` numbers that belong to something plugged into USB, lowest first.

    Probing a fixed 0, 1, 2 was right while a USB webcam was the only camera the box could
    have. A CSI module ends that: ``rp1-cfe`` and ``pispbe`` between them claim video0 through
    video7 before anything is plugged in at all, so a C920 now comes up as video8 and a fixed
    range of three never reaches it. Fitting the module would have quietly cost us the webcam.

    So the bus decides rather than the number. Every node the Pi's own silicon owns is a
    platform device; only something plugged in sits on the USB bus, and ``device/subsystem``
    says which in one link, for any node, without opening it.
    """
    nodes = []
    for entry in V4L_NODES.glob("video*"):
        with contextlib.suppress(OSError, ValueError):
            if (entry / "device" / "subsystem").resolve().name == "usb":
                nodes.append(int(entry.name.removeprefix("video")))
    return sorted(nodes)


def outranked(index: int | str, preferred: int | None = None) -> bool:
    """Is there a camera on the box right now that we would rather be using than this one?

    A camera used to be given up only when it stopped delivering, which meant the first one to
    open kept the panel for the rest of the run. That is wrong the moment a camera is a thing you
    reach for: plug the endoscope in to look down a bore and the panel carries on showing the
    room, and the only way to get the picture you plugged in for is to restart the kiosk. Marco
    hit that twice on 2026-09-20, which is what this is for.

    The order is :func:`open_camera`'s own probe order, because that order already *is* the
    preference: a webcam you plugged into USB, then an endoscope, then the module screwed to the
    case. Nothing outranks a USB webcam, and the CSI module is outranked by anything at all - it
    is the camera that is always there, so it is the one to fall back to and never the one to
    hold on to while something better is waiting.

    An explicit ``CYCLOPS_CAMERA_INDEX`` is never second-guessed, here as everywhere else.

    Read off sysfs, not off libusb: this is asked on a timer by a thread holding a camera open,
    and enumerating the bus underneath a live handle is how the last fault in this area started.
    """
    if preferred is not None:
        return False
    if index == RPICAM:
        return bool(_usb_video_nodes()) or _endoscope_on_bus()
    if index == USEEPLUS:
        return bool(_usb_video_nodes())
    return False  # already on a USB webcam; nothing beats it


def _endoscope_on_bus() -> bool:
    """Is one of the vendor-specific endoscopes plugged in? Sysfs only, no libusb."""
    return any(one.ident in USEEPLUS_IDS for one in devices.connected())


def _candidate_indices(preferred: int | None) -> list[int]:
    if preferred is not None:
        return [preferred]  # an explicit choice is never silently overridden
    found = _usb_video_nodes() if V4L_NODES.is_dir() else list(range(PROBE_INDICES))
    order = [] if _last_good_index is None else [_last_good_index]
    return order + [i for i in found if i not in order]


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


# Endoscopes whose device was pulled out from under them, kept alive deliberately for the rest of
# the process. Nothing ever reads this list; holding the reference IS the whole point.
#
# Declining to *call* release() on a yanked device is not enough, which is what the first attempt
# at this got wrong and what Marco found by pulling the cable a second time. `supercamera.Camera`
# has a `__del__` that calls `release()` itself, so the abort just moved house: the supervisor
# drops the dead capture, CPython's refcount hits zero, the finalizer runs, and libusb walks into
# the same assertion. A handle we have decided to abandon has to be unreachable by the *collector*,
# not merely unreferenced by us - so it is parked here, where it outlives everything.
_STRANDED: list[object] = []


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
        if not ok and not self._present():
            # Unplugged. Two things at once, and the order matters: later reads stop paying the
            # retry timeout, and the driver object is stranded here and now rather than at
            # release() - because nothing guarantees release() is ever called. The supervisor
            # may simply drop this capture and go looking for a camera, and a drop is all the
            # collector needs to run the finalizer that aborts the process.
            self._gone = True
            self._strand()
        return ok, frame

    def _strand(self) -> None:
        """Abandon the driver object somewhere the garbage collector cannot reach it."""
        if self._camera is not None:
            _STRANDED.append(self._camera)
            self._camera = None

    def _present(self) -> bool:
        """Is it still on the bus? A probe that cannot answer is answering no.

        The probe walks the bus over libusb, which is the same library that has just had a
        device pulled out from under it, so it is entitled to raise. Letting that propagate
        would take the reader thread down mid-loop; and a bus we cannot enumerate is not a bus
        we are going to find this endoscope on either way.
        """
        try:
            return bool(self._still_present())
        except Exception:  # noqa: BLE001 - libusb raises its own, and none of them mean "yes"
            return False

    def release(self) -> None:
        """Hand the device back - unless it is not there any more, in which case do nothing.

        This is the one call on this class that can take the whole kiosk down, and it does not
        do it by raising. Releasing a libusb handle whose device has been yanked walks into

            usbi_mutex_destroy: Assertion `pthread_mutex_destroy(mutex) == 0' failed.

        which is an ``assert()`` in C: it raises SIGABRT and the process is gone - panel, face,
        session and all. There is no ``except`` that catches it and no ``finally`` that runs
        after it. Marco found it on 2026-09-20 by pulling the endoscope out while it was the
        camera on screen, which is the ordinary way to finish looking at something.

        So the rule is that the handle is only given back while there is something to give it
        back to. Otherwise it is stranded in :data:`_STRANDED` and the process keeps the few
        file descriptors libusb had open until it exits - which is the same trade
        :meth:`CameraSource.stop` already makes for a reader stuck inside a read, and for the
        same reason: a leaked handle is cheap and a dead kiosk is not.

        The presence probe is here as well as in :meth:`read` because the two arrive by
        different roads. ``read`` catches the cable being pulled mid-stream; this catches the
        kiosk being shut down, or the framing being changed, in the window after the device went
        and before anything tried to read from it. Both end at the same place.

        After a *clean* release the driver object is dropped normally: its own ``release`` has
        already set its device to ``None``, so the ``__del__`` that follows returns immediately
        and there is nothing left to abort on.
        """
        if self._camera is None:
            return
        if self._gone or not self._present():
            self._strand()
            return
        self._camera.release()
        self._camera = None


def _open_useeplus() -> tuple[_UseeplusCapture, str]:
    """Open the first endoscope on the bus, or raise :class:`WebcamError` if there is none.

    Every failure here is the ordinary "no camera of this kind" answer, including an import
    that fails because the driver was never installed: this is one of three places a camera
    might be found, and none of them is allowed to take the kiosk down by being absent.
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


class _RpicamCapture:
    """The CSI camera module wearing :class:`cv2.VideoCapture`'s clothes.

    A camera module is not a webcam with a shorter cable. It hangs off the CSI bus, and the
    nodes it brings up - ``rp1-cfe-csi2_ch0`` and friends - carry raw Bayer straight off the
    sensor, not frames: V4L2 will tell you outright that they are "not a video capture device",
    which is exactly what :func:`open_camera`'s probe was being told. Everything that makes the
    sensor into a picture - debayer, black level, lens shading, AWB, the whole ISP - lives in
    libcamera, above those nodes. There is no index to open. Something has to run the pipeline.

    The obvious something is picamera2, and it cannot be used here: it imports ``libcamera``,
    whose Python binding is a compiled extension built against the system's 3.11, while this
    venv is 3.14. No ``PYTHONPATH`` bridges an ABI. ``rpicam-vid`` is that same libcamera stack
    reached over a pipe instead of an import, so it does not care what Python we are - which is
    what makes this a subprocess rather than a library, and not a preference.

    Raw YUV420 out of it rather than MJPEG, deliberately. MJPEG would buy a smaller pipe by
    spending a software encode there and a decode here - the Pi 5 has no JPEG encoder in
    hardware - and the pipe is local and not the bottleneck. Measured at 15 fps, this cost 4%
    of one core and delivered every frame asked for.
    """

    def __init__(self, proc: subprocess.Popen, width: int, height: int) -> None:
        self._proc = proc
        self._width, self._height = width, height
        self._frame_bytes = width * height * 3 // 2  # YUV420: Y, then two quarter-size planes
        self._first = None  # the frame prime() read, handed to the first read() that asks

    def read(self) -> tuple[bool, object]:
        if self._first is not None:
            frame, self._first = self._first, None
            return True, frame
        return self._read_frame()

    def prime(self) -> bool:
        """Read one frame now, so that opening either proves the camera or fails.

        This is what earns the CSI path the same guarantee the ``/dev/video`` path gets from its
        probe read: :func:`open_camera` never hands back a capture that will turn out to be a
        camera which was not there. Without it, a Pi with no module fitted answers every open
        with a healthy-looking object, and :class:`~cyclops.camera.CameraSource` spawns a fresh
        ``rpicam-vid`` every two seconds, for ever, each one dying on its own. It costs the
        second or so libcamera spends configuring the sensor, once per open.
        """
        ok, self._first = self._read_frame()
        return ok

    def _read_frame(self) -> tuple[bool, object]:
        buf = self._fill(self._frame_bytes)
        if buf is None:
            return False, None
        plane = np.frombuffer(buf, np.uint8).reshape(self._height * 3 // 2, self._width)
        return True, cv2.cvtColor(plane, cv2.COLOR_YUV2BGR_I420)

    def _fill(self, want: int) -> bytes | None:
        """Read exactly one frame, or None once the stream has ended.

        A pipe read returns what is in the pipe - 64 KB - and not what you asked for, so a
        1.4 MB frame arrives as twenty-odd of them. That makes the obvious ``read(n)`` wrong
        twice over: taken as a frame it is a torn image, and taken as end-of-stream (which is
        what a bare ``len(buf) != n`` check makes it) it tears down a camera that is working
        perfectly and closes the pipe under ``rpicam-vid``, which then dies of SIGPIPE. Only an
        empty read means the stream is over.
        """
        chunks, got = [], 0
        while got < want:
            chunk = self._proc.stdout.read(min(want - got, PIPE_CHUNK))
            if not chunk:
                return None
            chunks.append(chunk)
            got += len(chunk)
        return b"".join(chunks)

    def release(self) -> None:
        """Stop the encoder and reap it, from the thread that owns it and no other.

        Closing the pipe alone would be enough eventually - ``rpicam-vid`` takes a SIGPIPE on
        its next write - but only at its next write, which is why the signal comes first: this
        has to be synchronous, so that the next open finds the sensor free rather than racing
        the previous holder for it.
        """
        self._proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            self._proc.wait(timeout=2.0)
        if self._proc.poll() is None:
            self._proc.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._proc.wait(timeout=1.0)
        with contextlib.suppress(OSError):
            self._proc.stdout.close()


def _open_rpicam(framing: str = DEFAULT_FRAMING) -> tuple[_RpicamCapture, str]:
    """Start ``rpicam-vid`` on the camera module, or raise if there is no module to start it on.

    Absence is the ordinary answer here as it is for the endoscope, and takes two shapes: a box
    with no ``rpicam-vid`` at all (any Mac), and a Pi that has it but no module on the ribbon.
    The second only announces itself when the frame does not come, which is what :meth:`prime`
    is for.

    Two of these flags are about the module being *screwed into a case* rather than sitting on
    a desk, and both are free here and expensive anywhere else - the ISP does them on the way
    past, where doing either in numpy would cost a copy of every frame on a Pi that runs warm.

    ``--rotation 180`` because the module went in upside down. A whole rotation and not
    ``--vflip``, which is the tempting one-word answer and is wrong: a camera bolted to a case
    can only ever be *rotated*, never mirrored, so undoing an upside-down mount with a bare
    vertical flip fixes the sky and leaves the world reflected left-to-right. Nobody notices
    that on a row of houses. Everybody notices it the first time they hold up a part and ask
    what the number on it says.

    Autofocus because nothing was asking for it. The module has PDAF and the default mode
    leaves the lens parked, which looks fine on a wall two metres away and is soft on
    everything else - that is the whole of why the first picture out of it was blurry.
    Continuous rather than a one-off ``auto`` pass: there is no shutter button to half-press
    here, the camera simply watches whatever has been put in front of it, and a session where
    focus is a thing you have to ask for is a session spent thinking about the tool. Hunting
    costs a few soft frames when the scene changes, which is exactly what
    :meth:`~cyclops.camera.CameraSource.snapshot` already picks around.

    ``--mode`` because asking for 1280x720 does not ask for the whole lens. Mode selection takes
    the smallest sensor mode that covers the size requested, and on the IMX708 that is 1536x864 -
    which is not the sensor binned down but ``crop (768,432)/3072x1728``, two thirds of it,
    centred. A *Wide* module was therefore framing like the standard one, at about 1.5x: shot
    against 2304x1296 on 2026-09-09, the whole window and both sides of the sill were outside the
    picture. 2304x1296 is the full frame binned 2x2, and the PiSP scales it to 1280x720 on the way
    past - the one place a downscale costs no CPU at all.

    Which mode, though, is now the *framing*'s to say, and that is why this takes an argument at
    all. A sensor mode is chosen when the camera is configured and cannot be changed under a
    running pipeline, so the panel's three framings are three processes and switching between
    them is a stop and a start - 0.9 s to the first frame, measured on the Pi 2026-09-09.
    :meth:`~cyclops.camera.CameraSource.cycle_framing` is what spends it, and what keeps the last
    picture on the panel while it does.
    """
    if shutil.which("rpicam-vid") is None:
        raise WebcamError("no rpicam-vid installed")
    proc = subprocess.Popen(
        # -n because the kiosk owns the screen: rpicam-vid's own preview would open a second
        # window on top of it. -t 0 because the supervisor decides when this ends, not a timeout.
        [
            "rpicam-vid",
            "-n",
            "-t",
            "0",
            "--codec",
            "yuv420",
            "--width",
            str(FRAME_WIDTH),
            "--height",
            str(FRAME_HEIGHT),
            "--framerate",
            str(FRAME_RATE),
            "--rotation",
            # Turned over, the module is the right way up and its correction comes off.
            str(0 if flip.enabled() else RPICAM_ROTATION),
            "--autofocus-mode",
            RPICAM_AF,
            *FRAMINGS[framing],
            "-o",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,  # libcamera greets every start with a dozen INFO lines
        bufsize=0,
    )
    cap = _RpicamCapture(proc, FRAME_WIDTH, FRAME_HEIGHT)
    if not cap.prime():
        cap.release()
        raise WebcamError("no CSI camera module (rpicam-vid produced no frame)")
    return cap, RPICAM


def open_camera(
    preferred: int | None, framing: str = DEFAULT_FRAMING
) -> tuple[cv2.VideoCapture | _UseeplusCapture | _RpicamCapture, int | str]:
    """Open the preferred camera, or probe for one that actually delivers frames.

    On macOS the index order follows AVFoundation's uniqueID sort, so an idle iPhone
    (Continuity Camera) can sit at index 0 and "open" without ever returning a frame.

    A useeplus endoscope, and then the CSI camera module, are looked for only once no
    ``/dev/video*`` has answered. Neither can be probed the same way, neither having a capture
    node to probe, and a real webcam - when one is plugged in - should stay the camera you get.
    The module comes last for that same reason turned around: it is screwed to the case and so
    it is always there, and anything always there put first is a camera you can never override
    by plugging one in. Last, it is what you get whenever you have not chosen something else.

    The format is asked for explicitly because the default is expensive. Left alone, V4L2 hands
    out the driver's first format - uncompressed YUYV - and 720p of that is 1.8 MB a frame,
    which is 10 fps and nothing more on a USB 2.0 bus. That is the whole reason the preview used
    to step: the panel redraws about 25 times a second and only had ten new frames to draw. MJPG
    frames are a tenth the size, so the wire stops deciding, and the rate we ask for is the rate
    we get. 30 is one new frame for every redraw of the panel, so the preview - and a recording
    of it - moves as smoothly as the panel can show it. It was 15 while the C920 was the camera
    and every frame was a JPEG to decode; the module sends raw frames, and that cost is gone.
    """
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    params = [
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG"),  # before the size: the format decides what sizes fit
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH,
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT,
        cv2.CAP_PROP_FPS,
        FRAME_RATE,
    ]
    candidates = _candidate_indices(preferred)
    with _quiet_probe():
        for index in candidates:
            cap = cv2.VideoCapture(index, backend, params)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    return cap, index
            cap.release()
    for fallback in (_open_useeplus, lambda: _open_rpicam(framing)):
        with contextlib.suppress(WebcamError):
            return fallback()
    hint = (
        "Check System Settings → Privacy & Security → Camera for your terminal app, or set "
        "CYCLOPS_CAMERA_INDEX."
        if sys.platform == "darwin"
        else "Check that a webcam shows up in `lsusb` and is UVC or a useeplus endoscope, or "
        "that `rpicam-hello --list-cameras` sees the module on the ribbon."
    )
    raise WebcamError(
        f"No camera delivered a frame (tried indices {candidates}, the USB bus for a useeplus "
        f"endoscope, and rpicam-vid for a CSI module). {hint}"
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
