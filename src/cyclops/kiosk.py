"""Fullscreen OpenCV kiosk: the live camera *is* the UI, with tappable controls drawn on it.

``cyclops-kiosk`` is an alternative to ``cyclops-ui`` that needs no browser. It owns the camera
(via :class:`~cyclops.camera.CameraSource`), draws each frame into a fullscreen OpenCV window
with a PIL-rendered overlay on top, and turns taps into session control. The agent itself runs
on a background thread inside the same :class:`~cyclops.ui.SessionController` the web UI uses,
and its webcam tool borrows frames from the very camera you are watching.

highgui must own the main thread, so the render loop lives here and everything else is off-thread.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import BinaryIO

# The opencv-python wheel bundles Qt but no fonts, so Qt prints a five-line QFontDatabase
# complaint on every window. Only the blanket rule silences it - the narrower
# `qt.qpa.fonts.warning=false` and `QT_QPA_FONTDIR` were both measured to have no effect.
# This gags Qt's own logging only; Python warnings and exceptions are untouched. Must be set
# before cv2 imports, because Qt reads it once at load.
os.environ.setdefault("QT_LOGGING_RULES", "*.warning=false")

import cv2  # noqa: E402 - must follow the QT_LOGGING_RULES default above
import numpy as np  # noqa: E402 - kept with cv2, which pulls it in anyway

from . import webcam  # noqa: E402
from .backlight import Backlight  # noqa: E402
from .camera import CameraSource  # noqa: E402
from .config import BROWSER_CLOSE_FLAG, ConfigError, load_settings  # noqa: E402
from .overlay import (  # noqa: E402
    IDLE,
    LOOKING,
    STARTING,
    STOPPING,
    Overlay,
    composite,
    fit_to_window,
    mirror,
    platform_font_note,
)
from .ui import ERROR, SessionController  # noqa: E402
from .webcam import WebcamError  # noqa: E402


def detect_screen_size() -> tuple[int, int] | None:
    """The panel's resolution, straight from the kernel's DRM state on Linux.

    Nothing in highgui will tell us this reliably, so on a Pi we read the first mode of the
    connected output (the 7" DSI panel reports ``800x480``). Returns None elsewhere, where the
    kiosk is normally run windowed anyway; ``--size=WxH`` overrides in either case.
    """
    if sys.platform != "linux":
        return None
    for modes in sorted(Path("/sys/class/drm").glob("*/modes")):
        status = modes.with_name("status")
        try:
            if status.read_text().strip() != "connected":
                continue
            first = modes.read_text().splitlines()[0].strip()
            w, h = (int(v) for v in first.split("x", 1))
        except (OSError, ValueError, IndexError):
            continue
        if w > 1 and h > 1:
            return w, h
    return None


def _parse_size(argv: list[str]) -> tuple[int, int] | None:
    for arg in argv:
        if arg.startswith("--size="):
            try:
                w, h = (int(v) for v in arg.split("=", 1)[1].split("x", 1))
            except ValueError:
                raise SystemExit(f"error: bad --size, expected WxH, got {arg!r}") from None
            return w, h
    return None


WINDOW = "cyclops"
TARGET_FPS = 25
FLASH_SECONDS = 0.45
PRESS_SECONDS = 0.18  # how long the button stays visibly depressed after a tap
PENDING_TIMEOUT_S = 8.0  # give up on an optimistic state if the session never corroborates
SLEEP_AFTER_S = 60.0  # untouched for this long, the panel goes dark until it is tapped again
SLEEP_FPS = 4  # render rate while it is dark - there is nothing on screen but black
SHUTDOWN_JOIN_S = 15.0  # on exit, a stopping session may still be muxing its recording
BROWSERS = ("chromium-browser", "chromium")  # same probe order as cyclops-ui
ADMIN_POLL_S = 0.2  # how often the watcher looks for the page asking to be closed
ADMIN_FPS = 5  # render rate while the browser covers the panel - nobody can see us anyway
ADMIN_MAX_S = 15 * 60  # a window nobody closed comes down rather than stranding the panel
ADMIN_PROBE_S = 1.0  # how long the admin service gets to answer before we refuse the tap
BROWSER_GRACE_S = 5.0  # how long Chromium gets to go quietly before it is killed
# Its own profile, distinct from the one cyclops-ui's kiosk uses, and under ~/.cache rather than
# /tmp so the second open is a warm start instead of a first-run.
CHROME_PROFILE = Path.home() / ".cache" / "cyclops" / "admin-profile"
CHROME_LOG = Path.home() / ".cache" / "cyclops" / "chromium.log"
CHROME_FLAGS = (
    "--kiosk",  # fullscreen, no omnibox, no tab strip - the page carries its own way out
    "--noerrdialogs",
    "--disable-infobars",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-session-crashed-bubble",  # a terminated browser must not nag on the next open
    "--disable-features=Translate",
    "--disable-component-update",
    "--password-store=basic",  # never block waiting on a keyring
    "--force-device-scale-factor=1",  # 1 CSS px == 1 panel px, so the page's layout maths holds
)


def _black(width: int, height: int) -> np.ndarray:
    """A blank frame - for the dark panel, and for the moment the camera is still coming back."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def _admin_reachable(url: str) -> bool:
    """Is the admin service actually answering?

    A kiosk-mode browser has no address bar and no back button, so a fullscreen "this site can't
    be reached" would leave the panel with nothing to tap. If the service is down, do nothing.
    """
    try:
        with urllib.request.urlopen(url, timeout=ADMIN_PROBE_S):
            return True
    except (OSError, ValueError):
        return False


def _spawn_browser(url: str, log: BinaryIO) -> subprocess.Popen | None:
    """Chromium, fullscreen and chrome-less, over the kiosk window. None if none is installed.

    Its output goes to a file rather than /dev/null: the two usual reasons the button appears to
    do nothing - no Wayland socket because the kiosk was started over ssh, or a corrupt profile -
    both announce themselves there and nowhere else.
    """
    flags = list(CHROME_FLAGS)
    if os.environ.get("WAYLAND_DISPLAY"):  # the Pi's labwc session; omitted elsewhere
        flags.insert(0, "--ozone-platform=wayland")
    flags.append(f"--user-data-dir={CHROME_PROFILE}")
    for browser in BROWSERS:
        try:
            return subprocess.Popen([browser, *flags, url], stdout=log, stderr=log)
        except FileNotFoundError:
            continue
    print(f"· no chromium found; the admin page is at {url}", file=sys.stderr, flush=True)
    return None


def _stop_browser(proc: subprocess.Popen) -> None:
    """Ask Chromium to go, then insist."""
    try:
        proc.terminate()
        proc.wait(BROWSER_GRACE_S)
    except subprocess.TimeoutExpired:
        proc.kill()
    except OSError:
        pass


class Kiosk:
    """The render loop, the window, and the tap handling."""

    def __init__(
        self,
        controller: SessionController,
        camera: CameraSource,
        fullscreen: bool,
        screen: tuple[int, int] | None = None,
    ):
        self.controller = controller
        self.camera = camera
        self.fullscreen = fullscreen
        self.overlay: Overlay | None = None
        self.backlight = Backlight()  # the panel's light, off while it sleeps
        self.running = True
        self._flash_until = 0.0
        self._press_until = 0.0
        self._pressed: str | None = None  # which button is still showing its tap
        self._pending: str | None = None  # "start"/"stop" until the session catches up
        self._pending_at = 0.0
        self._prev_state = IDLE
        self._touched_at = time.monotonic()  # last tap, for the idle blank
        self._asleep = False  # dark panel: the camera is released until it is touched
        self._camera_on_at = 0.0  # when the camera was last (re)started, to date its frames
        self._browser: subprocess.Popen | None = None  # the admin page, when it is open
        self._admin_busy = threading.Event()  # set from the tap until the browser is gone
        self._refocus = threading.Event()  # asks the render loop to re-assert fullscreen
        self._size = (0, 0)
        self.screen = screen

    # ---- window ----

    def open_window(self, first_frame, width: int, height: int) -> None:
        """Create the window, give it a real surface, and only then go fullscreen.

        highgui has no drawable until the first ``imshow``, and a fullscreen request made
        before that is quietly dropped under XWayland - the window comes up as a band in the
        middle of the panel instead of filling it. Show a frame first, then set the property.
        """
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, width, height)
        cv2.imshow(WINDOW, first_frame)
        cv2.waitKey(1)
        self._apply_fullscreen()
        cv2.setMouseCallback(WINDOW, self._on_mouse)

    def _apply_fullscreen(self) -> None:
        cv2.setWindowProperty(
            WINDOW,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL,
        )
        cv2.waitKey(1)  # let the compositor finish the resize before we draw again

    def _window_size(self, fallback: tuple[int, int]) -> tuple[int, int]:
        """The size to render at: the panel when fullscreen, else whatever the window is.

        ``getWindowImageRect`` is not usable to discover the panel size - it reports where the
        *last* image landed, so it echoes back whatever aspect we last fed it and stays wedged
        there. Fullscreen therefore takes its size from the display itself.
        """
        if self.fullscreen and self.screen is not None:
            return self.screen
        try:
            _, _, w, h = cv2.getWindowImageRect(WINDOW)
            if w > 1 and h > 1:
                return w, h
        except cv2.error:
            pass
        return fallback

    # ---- input ----

    def _on_mouse(self, event: int, x: int, y: int, flags: int, _param: object) -> None:
        """Touchscreen taps arrive here as ordinary mouse events via XWayland."""
        if self.overlay is None:
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        self._touched_at = time.monotonic()
        if self._asleep:
            # The tap that wakes the panel is spent waking it. With the preview dark you cannot
            # see what you are aiming at, so it must not also fire whatever sits underneath.
            self._wake()
            return
        boxes = self.overlay.hitboxes
        if boxes.shutter.contains(x, y):
            self._press("shutter")
            self._snap()
        elif boxes.eye.contains(x, y):
            self._press("eye")
            self._toggle_session()
        elif boxes.admin.contains(x, y):
            self._press("admin")
            self._open_admin()

    def _press(self, button: str) -> None:
        self._pressed = button
        self._press_until = time.monotonic() + PRESS_SECONDS

    def _pressed_now(self) -> str | None:
        """Which button to draw as held. The admin disc stays lit while its browser is up.

        Chromium takes two to four seconds to appear on a Pi. Without this the disc goes dark
        180 ms after the tap and the panel looks like it ignored you.
        """
        if self._admin_busy.is_set():
            return "admin"
        return self._pressed if time.monotonic() < self._press_until else None

    def _snap(self) -> None:
        """Take a photo straight away, off-thread, and flash the screen as the shutter.

        The capture borrows a frame from the camera we are already previewing (see
        :func:`cyclops.webcam.set_live_source`), so it costs milliseconds - but it still writes
        a file, and the render loop must not stall behind a disk that is busy.
        """
        self._flash_until = time.monotonic() + FLASH_SECONDS
        threading.Thread(target=self._capture, name="kiosk-snap", daemon=True).start()

    def _capture(self) -> None:
        settings = self.controller.settings
        try:
            shot = webcam.capture_image(settings.camera_index, save_dir=settings.captures_dir)
        except WebcamError as exc:
            print(f"· snapshot failed: {exc}", file=sys.stderr, flush=True)
        else:
            print(f"· snapped {shot.path}", flush=True)

    # ---- admin page ----

    def _open_admin(self) -> None:
        """Hand the panel to Chromium, off-thread - highgui owns the main one.

        Same shape as :meth:`_snap`. The busy flag means a second tap during the cold start is
        ignored rather than starting a second browser: sharing a profile directory, the second
        invocation would hand its URL to the first and exit at once, leaving us holding a dead
        pid and a fullscreen window nothing can close.
        """
        if self._admin_busy.is_set():
            return
        self._admin_busy.set()
        threading.Thread(target=self._admin_session, name="kiosk-admin", daemon=True).start()

    def _admin_session(self) -> None:
        """The whole life of the admin window: probe, launch, wait, take it back down."""
        url = f"http://127.0.0.1:{self.controller.settings.admin_port}/"
        try:
            if not _admin_reachable(url):
                print(
                    f"· admin page not answering on {url} (systemctl status cyclops-admin)",
                    file=sys.stderr,
                    flush=True,
                )
                return
            BROWSER_CLOSE_FLAG.parent.mkdir(parents=True, exist_ok=True)
            BROWSER_CLOSE_FLAG.unlink(missing_ok=True)  # a stale note must not close this one
            CHROME_LOG.parent.mkdir(parents=True, exist_ok=True)
            launched_at = time.time()
            with CHROME_LOG.open("wb") as log:
                proc = _spawn_browser(url, log)
                if proc is None:
                    return
                self._browser = proc
                print(f"· admin page open ({url})", flush=True)
                self._watch_browser(proc, launched_at)
        finally:
            self._browser = None
            BROWSER_CLOSE_FLAG.unlink(missing_ok=True)
            self._admin_busy.clear()
            self._refocus.set()
            print("· admin page closed", flush=True)

    def _watch_browser(self, proc: subprocess.Popen, launched_at: float) -> None:
        """Wait for the page to ask to close, for the browser to die, or for the hard cap.

        The page cannot close a window it did not open, so its Close button drops
        :data:`~cyclops.config.BROWSER_CLOSE_FLAG` and the decision is taken here, where the
        process handle lives. The note counts only if it was written after we launched, so one
        left behind by an earlier round can never shut this window the moment it opens.
        """
        deadline = time.monotonic() + ADMIN_MAX_S
        while proc.poll() is None:
            try:
                asked = BROWSER_CLOSE_FLAG.stat().st_mtime >= launched_at
            except OSError:
                asked = False
            if asked or time.monotonic() > deadline:
                break
            time.sleep(ADMIN_POLL_S)
        _stop_browser(proc)

    def close_browser(self) -> None:
        """Take the admin page down with the kiosk, so nothing is left covering the panel."""
        proc = self._browser
        if proc is not None and proc.poll() is None:
            _stop_browser(proc)

    # ---- session ----

    def _toggle_session(self) -> None:
        """Act on the tap and record what we asked for, so the UI can show it at once."""
        state = self.controller.status()["state"]
        starting = state in (IDLE, ERROR)
        self._pending = "start" if starting else "stop"
        self._pending_at = time.monotonic()
        if starting:
            self.controller.start()
        else:
            self.controller.stop()

    def _effective(self, raw: str) -> str:
        """Overlay the tap's intent on the session's own state until the two agree.

        Starting is honest on its own - the controller reports ``connecting`` within a
        millisecond. Stopping is not: the session keeps reporting ``listening`` for the ~2.3 s
        it takes to cancel the task, close the socket and stop the audio streams, so the halo
        would look ignored. Until it settles we show what was asked for, not what still is.
        """
        if self._pending is None:
            return raw
        if self._pending == "stop":
            done = raw in (IDLE, ERROR)  # both mean the session is no longer up
        else:
            done = raw != IDLE  # anything else is an outcome, ERROR included - show it
        if done or time.monotonic() - self._pending_at > PENDING_TIMEOUT_S:
            self._pending = None
            return raw
        return STOPPING if self._pending == "stop" else STARTING

    # ---- sleep ----

    def _sleeping(self, state: str) -> bool:
        """Should the panel be dark? True once nothing has touched it for a while.

        A live session counts as company even when nobody is touching the glass - the halo and
        the timer are a conversation's only feedback, and blanking them mid-sentence would read
        as a crash - and so does an open admin page, which is covering the panel itself. Only an
        idle kiosk goes dark, and only a tap brings it back (see :meth:`_on_mouse`).
        """
        now = time.monotonic()
        if state not in (IDLE, ERROR) or self._admin_busy.is_set():
            self._touched_at = now
        elif not self._asleep and now - self._touched_at > SLEEP_AFTER_S:
            self._sleep()
        return self._asleep

    def _sleep(self) -> None:
        """Put the light out and hand the camera back.

        Between them they are most of what an idle kiosk costs: the panel's backlight, and a
        camera streaming 25 fps at a room nobody is in. Both cost a reopen on the way back.
        Only ever called with the session down, so nothing is mid-recording and the agent's
        tool has no shot to take.
        """
        self._asleep = True
        self.backlight.off()  # the light first: on a battery it is the expensive half
        self.camera.stop()
        print("· idle: light off, camera released - tap to wake", flush=True)

    def _wake(self) -> None:
        """Light the panel, reopen the camera, and start drawing again.

        highgui dispatches mouse callbacks from inside ``waitKey``, so this runs on the render
        thread and the reopen simply stalls one frame that was black anyway. If the device does
        not come back we stay dark rather than half-woken, and the next tap tries again.
        """
        self.backlight.on()  # light first, so the panel answers the tap before the camera can
        try:
            self.camera.start()
        except WebcamError as exc:
            print(f"· camera did not come back: {exc}", file=sys.stderr, flush=True)
            self.backlight.off()  # lit with nothing to show is worse than dark; stay asleep
            return
        self._camera_on_at = time.monotonic()
        self._asleep = False

    # ---- loop ----

    def run(self) -> None:
        frame_budget = 1.0 / TARGET_FPS
        first = self.camera.frame()
        h, w = first.shape[:2]
        self.open_window(first, min(w, 1280), min(h, 720))

        while self.running:
            started = time.monotonic()
            if self._refocus.is_set():
                # highgui is main-thread only, so the browser watcher cannot do this itself.
                self._refocus.clear()
                self._apply_fullscreen()

            status = self.controller.status()
            state = self._effective(str(status["state"]))
            asleep = self._sleeping(state)

            frame = None  # nothing to draw: the camera is either off or still coming back
            if not asleep:
                got = self.camera.latest()
                if got is None:
                    break  # it has never delivered a frame; there is nothing to wait for
                if got[1] >= self._camera_on_at:
                    # A reopened device keeps handing back the frame it stopped on, and that
                    # is last minute's room. Anything older than the reopen is not shown.
                    frame = got[0]

            fallback = self._size if frame is None else (frame.shape[1], frame.shape[0])
            width, height = self._window_size(fallback)
            if (width, height) != self._size or self.overlay is None:
                self.overlay = Overlay(width, height)
                self._size = (width, height)

            if state == LOOKING and self._prev_state != LOOKING:
                self._flash_until = time.monotonic() + FLASH_SECONDS
            self._prev_state = state

            if asleep:
                cv2.imshow(WINDOW, _black(width, height))  # no picture, and no chrome either
            else:
                # The chrome is drawn through the wake-up: the buttons must answer the tap even
                # while the camera is still opening behind them.
                canvas = (
                    _black(width, height)
                    if frame is None
                    else fit_to_window(mirror(frame), width, height)
                )
                flash = max(0.0, (self._flash_until - time.monotonic()) / FLASH_SECONDS)
                elapsed = status["elapsed"]
                chrome = self.overlay.render(
                    state=state,
                    level=float(status["level"]),
                    elapsed=None if elapsed is None else float(elapsed),
                    recording=self.controller.settings.record,
                    flash=flash,
                    pressed=self._pressed_now(),
                )
                cv2.imshow(WINDOW, composite(canvas, chrome))

            spent = time.monotonic() - started
            # While the browser covers the panel we are compositing frames nobody can see,
            # against the very cold start we are waiting on. Give the core back. A dark panel
            # is cheaper still - one black frame, redrawn only to keep taps and keys answered.
            if self._asleep:
                budget = 1.0 / SLEEP_FPS
            elif self._admin_busy.is_set():
                budget = 1.0 / ADMIN_FPS
            else:
                budget = frame_budget
            wait_ms = max(1, int((budget - spent) * 1000))
            key = cv2.waitKey(wait_ms) & 0xFF
            if key in (27, ord("q")):  # ESC or q
                self.running = False
            elif key == ord("f"):
                self.fullscreen = not self.fullscreen
                self._apply_fullscreen()
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                self.running = False  # the user closed the window


def main() -> None:
    args = sys.argv[1:]
    fullscreen = "--windowed" not in args
    screen = _parse_size(args) or (detect_screen_size() if fullscreen else None)
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)

    camera = CameraSource(settings.camera_index)
    controller = SessionController(settings, frames=camera)  # sessions record from this camera
    try:
        camera.start()
        camera.wait_for_frame()
    except WebcamError as exc:
        print(f"error: {exc}", file=sys.stderr)
        camera.stop()
        sys.exit(1)

    webcam.set_live_source(camera)  # the agent's tool now shoots from this same camera
    kiosk = Kiosk(controller, camera, fullscreen, screen)
    kiosk.backlight.on()  # a previous run may have been killed while the panel was dark
    # SIGTERM (start_kiosk.sh's pkill, systemd) otherwise skips the finally below and would
    # leave a panel that looks like a dead Pi. Exit properly instead, and the light comes back.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    print(
        f"· cyclops kiosk on camera {camera.index} · font: {platform_font_note()}\n"
        f"  screen: {'x'.join(map(str, screen)) if screen else 'window-sized'}"
        f" · backlight: {kiosk.backlight.note}\n"
        "  tap the shutter (bottom left) to snap · the eye (bottom right) to start/stop\n"
        "  the gear (top left) opens the admin page\n"
        f"  after {SLEEP_AFTER_S:g}s untouched the light goes off and the camera is released\n"
        "  any tap wakes it\n"
        "  q or ESC to quit · f toggles fullscreen",
        flush=True,
    )
    try:
        kiosk.run()
    except KeyboardInterrupt:
        print("\n· bye", flush=True)
    finally:
        kiosk.close_browser()
        kiosk.backlight.on()  # never leave the panel dark behind us
        webcam.set_live_source(None)
        controller.stop()
        controller.join(SHUTDOWN_JOIN_S)  # let it finish writing before the camera goes away
        camera.stop()
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # let highgui actually tear the window down


if __name__ == "__main__":
    main()
