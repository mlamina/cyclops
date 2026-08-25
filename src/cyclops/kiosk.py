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
import sys
import time
from pathlib import Path

# The opencv-python wheel bundles Qt but no fonts, so Qt prints a five-line QFontDatabase
# complaint on every window. Only the blanket rule silences it - the narrower
# `qt.qpa.fonts.warning=false` and `QT_QPA_FONTDIR` were both measured to have no effect.
# This gags Qt's own logging only; Python warnings and exceptions are untouched. Must be set
# before cv2 imports, because Qt reads it once at load.
os.environ.setdefault("QT_LOGGING_RULES", "*.warning=false")

import cv2  # noqa: E402 - must follow the QT_LOGGING_RULES default above

from . import webcam  # noqa: E402
from .camera import CameraSource  # noqa: E402
from .config import ConfigError, load_settings  # noqa: E402
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
        self.running = True
        self._flash_until = 0.0
        self._press_until = 0.0
        self._pending: str | None = None  # "start"/"stop" until the session catches up
        self._pending_at = 0.0
        self._prev_state = IDLE
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
        boxes = self.overlay.hitboxes
        if boxes.quit.contains(x, y):
            self._press_until = time.monotonic() + PRESS_SECONDS
            self.running = False
        elif boxes.eye.contains(x, y):
            self._press_until = time.monotonic() + PRESS_SECONDS
            self._toggle_session()

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

    def _effective(self, raw: str, detail: str) -> tuple[str, str]:
        """Overlay the tap's intent on the session's own state until the two agree.

        Starting is honest on its own - the controller reports ``connecting`` within a
        millisecond. Stopping is not: the session keeps reporting ``listening`` for the ~2.3 s
        it takes to cancel the task, close the socket and stop the audio streams, so the button
        would look ignored. Until it settles we show what was asked for, not what still is.
        """
        if self._pending is None:
            return raw, detail
        if self._pending == "stop":
            done = raw in (IDLE, ERROR)  # both mean the session is no longer up
        else:
            done = raw != IDLE  # anything else is an outcome, ERROR included - show it
        if done or time.monotonic() - self._pending_at > PENDING_TIMEOUT_S:
            self._pending = None
            return raw, detail
        if self._pending == "stop":
            return STOPPING, "Stopping…"
        return STARTING, "Starting…"

    # ---- loop ----

    def run(self) -> None:
        frame_budget = 1.0 / TARGET_FPS
        first = self.camera.frame()
        h, w = first.shape[:2]
        self.open_window(first, min(w, 1280), min(h, 720))

        while self.running:
            started = time.monotonic()
            frame = self.camera.frame()
            if frame is None:
                break

            width, height = self._window_size((frame.shape[1], frame.shape[0]))
            if (width, height) != self._size or self.overlay is None:
                self.overlay = Overlay(width, height)
                self._size = (width, height)

            status = self.controller.status()
            state, detail = self._effective(str(status["state"]), str(status["detail"]))
            if state == LOOKING and self._prev_state != LOOKING:
                self._flash_until = time.monotonic() + FLASH_SECONDS
            self._prev_state = state

            canvas = fit_to_window(mirror(frame), width, height)
            flash = max(0.0, (self._flash_until - time.monotonic()) / FLASH_SECONDS)
            chrome = self.overlay.render(
                state=state,
                detail=detail,
                level=float(status["level"]),
                flash=flash,
                pressed=time.monotonic() < self._press_until,
            )
            cv2.imshow(WINDOW, composite(canvas, chrome))

            spent = time.monotonic() - started
            wait_ms = max(1, int((frame_budget - spent) * 1000))
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
    controller = SessionController(settings)
    try:
        camera.start()
        camera.wait_for_frame()
    except WebcamError as exc:
        print(f"error: {exc}", file=sys.stderr)
        camera.stop()
        sys.exit(1)

    webcam.set_live_source(camera)  # the agent's tool now shoots from this same camera
    print(
        f"· cyclops kiosk on camera {camera.index} · font: {platform_font_note()}\n"
        f"  screen: {'x'.join(map(str, screen)) if screen else 'window-sized'}\n"
        "  tap the eye to start/stop · q or ESC to quit · f toggles fullscreen",
        flush=True,
    )
    kiosk = Kiosk(controller, camera, fullscreen, screen)
    try:
        kiosk.run()
    except KeyboardInterrupt:
        print("\n· bye", flush=True)
    finally:
        webcam.set_live_source(None)
        controller.stop()
        camera.stop()
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # let highgui actually tear the window down


if __name__ == "__main__":
    main()
