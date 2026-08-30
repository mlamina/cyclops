"""Fullscreen OpenCV kiosk: the live camera *is* the UI, with tappable controls drawn on it.

``cyclops-kiosk`` is the whole front-end and needs no browser of its own. It owns the camera
(via :class:`~cyclops.camera.CameraSource`), draws each frame into a fullscreen OpenCV window
with a PIL-rendered overlay on top, and turns taps into session control. The agent itself runs
on a background thread inside a :class:`~cyclops.ui.SessionController`, and the photos the
SNAP button feeds it are borrowed from the very camera you are watching.

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

from . import mixer, session, sfx, webcam  # noqa: E402
from .audio import SAMPLE_RATE, resolve_device  # noqa: E402
from .backlight import Backlight  # noqa: E402
from .camera import CameraSource  # noqa: E402
from .config import (  # noqa: E402
    BROWSER_CLOSE_FLAG,
    PAGE_SERVED_FLAG,
    ConfigError,
    load_settings,
)
from .overlay import (  # noqa: E402
    IDLE,
    STARTING,
    STOPPING,
    Overlay,
    composite,
    fit_to_window,
    message,
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
# What the picture area says when there is no camera, and how big the window comes up without
# one to take a size from. The panel is 800x480; this fits it and looks deliberate on anything
# larger, which is the point - a black window with no explanation reads as a crashed Pi.
NO_CAMERA = "No camera found"
NO_CAMERA_SIZE = (800, 480)
FLASH_SECONDS = 0.45
PRESS_SECONDS = 0.18  # how long the button stays visibly depressed after a tap
PENDING_TIMEOUT_S = 8.0  # give up on an optimistic state if the session never corroborates
SLEEP_FPS = 4  # render rate while it is dark - there is nothing on screen but black
SHUTDOWN_JOIN_S = 20.0  # on exit, a stopping session may still be muxing and naming itself
BROWSERS = ("chromium-browser", "chromium")  # whichever of the two names this Pi installed
ADMIN_POLL_S = 0.2  # how often the watcher looks for the page asking to be closed
ADMIN_FPS = 5  # render rate while the browser covers the panel - nobody can see us anyway
ADMIN_MAX_S = 15 * 60  # a page nobody closed gives the panel back rather than stranding it
ADMIN_PROBE_S = 1.0  # how long the admin service gets to answer before we refuse the tap
# The browser is started once, at boot, and afterwards only uncovered, because starting one is
# not something a button press can wait for: measured on this Pi, spawn to first pixels is 1.4 s
# with Chromium's 254 MB of binary warm in the page cache and 8.8 s with it cold, against 2 ms
# for the page itself. Below is what the warm-up needs to get there and stay out of the way.
PREWARM_TRIES = 30  # the admin service is a systemd unit and may still be coming up at boot
PREWARM_RETRY_S = 2.0  # gap between those tries - a minute of patience, then the slow path
PAGE_WAIT_S = 30.0  # how long the browser gets to fetch the page; a cold start eats 9 s of it
# When to take the panel back after the warm-up's window maps on top of ours. Twice, because
# the stacking order cannot be read back: labwc raises whatever mapped last and no always-on-top
# hint survives that (measured), so the first retake covers the usual case and the second, later
# one covers a map slow enough to have landed after it - the failure it prevents is a panel left
# showing the dashboard with nobody having asked for it.
PANEL_RETAKE_S = (0.6, 2.4)
VOLUME_POLL_S = 0.4  # how often we look for a volume the admin page left for us
BROWSER_GRACE_S = 5.0  # how long Chromium gets to go quietly before it is killed
# Its own profile, under ~/.cache rather than /tmp so the second open is a warm start rather
# than a first-run.
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
    """Chromium, fullscreen and chrome-less, over the kiosk window. None if none is installed."""
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


def _wait_for_page(proc: subprocess.Popen, since: float, timeout: float) -> bool:
    """Wait for the admin service to say it has handed the page to a browser here.

    Chromium announces nothing when it is ready, and nothing can be asked what the panel is
    showing, so the service leaves a note instead (:data:`~cyclops.config.PAGE_SERVED_FLAG`).
    The note has to be newer than the launch, or the one left by our own reachability probe a
    moment earlier would answer for a browser that is still faulting itself in off the card.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            if PAGE_SERVED_FLAG.stat().st_mtime >= since:
                return True
        except OSError:
            pass
        time.sleep(ADMIN_POLL_S)
    return False


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
        self._snap_busy = threading.Event()  # one shutter at a time; see _snap
        # The shutter sound is the kiosk's own: a photo can be taken with no session running,
        # so it cannot wait for an agent to exist to own the noise.
        self._cues = sfx.Cues(
            rate=SAMPLE_RATE,
            device=resolve_device(controller.settings.output_device),
            enabled=controller.settings.sounds,
        )
        self._touched_at = time.monotonic()  # last tap, for the idle blank
        self._asleep = False  # dark panel: the camera is released until it is touched
        self._camera_on_at = 0.0  # when the camera was last (re)started, to date its frames
        self._volume: int | None = None  # the level we last put on the sink
        self._volume_at = 0.0  # when we last looked for a new one
        self._browser: subprocess.Popen | None = None  # the admin browser, kept warm from boot
        self._admin_busy = threading.Event()  # set from the tap until the page is done with
        self._reveal = threading.Event()  # asks the render loop to uncover the admin page
        self._retake = threading.Event()  # ... and to take the panel back off it
        self._hidden = False  # the admin page has the panel; nothing we draw can be seen
        self._window_up = False  # whether highgui currently has a window for us
        self._size = (0, 0)
        self.screen = screen

    # ---- window ----

    def open_window(self, first_frame, width: int, height: int) -> None:
        """Create the window, give it a real surface, and only then go fullscreen.

        highgui has no drawable until the first ``imshow``, and a fullscreen request made
        before that is quietly dropped under XWayland - the window comes up as a band in the
        middle of the panel instead of filling it. Show a frame first, then set the property.

        Called again for every retake of the panel, which is why it takes the frame to come up
        with: a window built around the picture it is about to show never flashes black.
        """
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, width, height)
        cv2.imshow(WINDOW, first_frame)
        cv2.waitKey(1)
        self._window_up = True
        self._apply_fullscreen()
        cv2.setMouseCallback(WINDOW, self._on_mouse)

    def _drop_window(self) -> None:
        """Take our window down, putting whatever is behind it - the admin page - on the panel."""
        if not self._window_up:
            return
        self._window_up = False
        cv2.destroyWindow(WINDOW)
        cv2.waitKey(1)  # let highgui actually unmap it before anything else is drawn

    def _paint(self, image, width: int, height: int) -> None:
        """Put a frame on the panel, building the window first if we have just taken it back.

        Rebuilding is the only way back on top of the browser: a window that is merely redrawn
        keeps the place in the stack it already had, which is underneath.
        """
        if self._window_up:
            cv2.imshow(WINDOW, image)
        else:
            self.open_window(image, width, height)

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
        """Which tab to draw as held. The SYSTEM tab stays lit while its page is up.

        Uncovering a warm browser is immediate, so this is normally seen for a frame or two.
        It still earns its place on the one tap that has to start a browser: without it the
        tab goes dark 180 ms in and a panel that is busy looks like a panel that ignored you.
        """
        if self._admin_busy.is_set():
            return "admin"
        return self._pressed if time.monotonic() < self._press_until else None

    def _snap(self) -> None:
        """Take a photo straight away, off-thread, and flash the screen as the shutter.

        The capture borrows a frame from the camera we are already previewing (see
        :func:`cyclops.webcam.set_live_source`), so it costs milliseconds - but it still writes
        a file, and the render loop must not stall behind a disk that is busy.

        One at a time. Two taps inside the same second raced in webcam._unique, which checks a
        name is free and then writes it, so both shots landed on one path and one was lost.
        They would now also be two photos handed to the model for one thing held up.
        """
        if self._snap_busy.is_set():
            return
        self._snap_busy.set()
        self._flash_until = time.monotonic() + FLASH_SECONDS
        self._cues.play("shutter")  # the flash, said out loud - same moment, same event
        threading.Thread(target=self._capture, name="kiosk-snap", daemon=True).start()

    def _capture(self) -> None:
        settings = self.controller.settings
        # Where it lands depends on whether a session is running: into that session's photos/ if
        # one is, and into the plain captures/ archive if not. The log is told either way -
        # session.note is a no-op when there is nothing live to tell.
        save_dir, keep_as = session.photo_target(settings, by="you")
        try:
            shot = webcam.capture_image(
                settings.camera_index, save_dir=save_dir, keep_as=keep_as
            )
        except WebcamError as exc:
            print(f"· snapshot failed: {exc}", file=sys.stderr, flush=True)
        else:
            # The whole point of the button: the photo goes to Cyclops, which answers out loud.
            # False means there was no live session to show it to - it is still on the card.
            shown = self.controller.show_photo(shot)
            session.note(
                "photo",
                by="you",
                file=f"{session.PHOTOS}/{shot.path.name}",
                width=shot.width,
                height=shot.height,
                bytes=shot.jpeg_bytes,
                shown=shown,
            )
            print(f"· snapped {shot.path}{'' if shown else ' (nothing live to show it to)'}",
                  flush=True)
        finally:
            self._snap_busy.clear()

    # ---- admin page ----

    def _admin_url(self) -> str:
        return f"http://127.0.0.1:{self.controller.settings.admin_port}/"

    def prewarm(self) -> None:
        """Start the admin browser now, in the background, so the gear only has to uncover it."""
        threading.Thread(target=self._prewarm, name="kiosk-prewarm", daemon=True).start()

    def _prewarm(self) -> None:
        """Wait for the admin service, start the browser on it, then take the panel back.

        The browser then sits behind our window for the life of the kiosk with the page loaded
        and polling, which is what makes the gear instant. It costs a couple of hundred MB of a
        box that has 8 GB, and the alternative is a button that answers in seconds.
        """
        url = self._admin_url()
        for _ in range(PREWARM_TRIES):
            if not self.running:
                return
            if _admin_reachable(url):
                break
            time.sleep(PREWARM_RETRY_S)
        else:
            print(
                f"· admin service never answered on {url}; the gear will start its own browser",
                file=sys.stderr,
                flush=True,
            )
            return
        if not self._start_browser(url):
            return
        print("· admin page warmed up behind the panel", flush=True)
        for delay in PANEL_RETAKE_S:  # its window mapped on top of ours; see PANEL_RETAKE_S
            time.sleep(delay)
            if self._admin_busy.is_set():
                return  # someone tapped the gear while we were starting: the page is theirs now
            self._retake.set()

    def _start_browser(self, url: str) -> bool:
        """Launch Chromium on the admin page and wait for it to have it. True if it is up.

        Its output goes to a file rather than /dev/null: the two usual reasons the panel ends up
        with no admin page - no Wayland socket because the kiosk was started over ssh, and a
        corrupt profile - both announce themselves there and nowhere else.
        """
        CHROME_LOG.parent.mkdir(parents=True, exist_ok=True)
        launched_at = time.time()
        with CHROME_LOG.open("wb") as log:
            proc = _spawn_browser(url, log)  # the child keeps the fd; closing ours is fine
        if proc is None:
            return False
        self._browser = proc
        if not _wait_for_page(proc, launched_at, PAGE_WAIT_S):
            if proc.poll() is not None:
                print(
                    f"· the admin browser died starting up; see {CHROME_LOG}",
                    file=sys.stderr,
                    flush=True,
                )
                return False
            # It is alive but never fetched the page - an admin service that went away between
            # the probe and now, most likely. Show it anyway: a browser on an error page can at
            # least be looked at, and refusing the tap outright explains nothing.
            print(
                "· the admin browser never reported the page; showing it anyway",
                file=sys.stderr,
                flush=True,
            )
        if not self.running:  # we are on our way out; do not leave a window covering the panel
            _stop_browser(proc)
            return False
        return True

    def _ensure_browser(self, url: str) -> bool:
        """The warm browser, or a fresh one if it died. False when there is nothing to show."""
        proc = self._browser
        if proc is not None and proc.poll() is None:
            return True
        # The warm-up never got one up, or it crashed since. This is the slow path the warm-up
        # exists to avoid, and it costs this one tap: the new window maps on top of ours by
        # itself, which is exactly where the uncovering below wants it.
        return self._start_browser(url)

    def _open_admin(self) -> None:
        """Uncover the admin page, off-thread - highgui owns the main one.

        Same shape as :meth:`_snap`. The busy flag means a second tap while the page is up is
        ignored rather than starting a second browser: sharing a profile directory, the second
        invocation would hand its URL to the first and exit at once, leaving us holding a dead
        pid and a fullscreen window nothing can close.
        """
        if self._admin_busy.is_set():
            return
        self._admin_busy.set()
        threading.Thread(target=self._admin_session, name="kiosk-admin", daemon=True).start()

    def _admin_session(self) -> None:
        """The whole life of the visible page: probe, uncover, wait, take the panel back."""
        url = self._admin_url()
        shown = False
        try:
            if not _admin_reachable(url):
                # A kiosk-mode browser has no address bar and no back button, so a page that
                # cannot reach its own service would leave the panel with nothing to tap.
                print(
                    f"· admin page not answering on {url} (systemctl status cyclops-admin)",
                    file=sys.stderr,
                    flush=True,
                )
                return
            if not self._ensure_browser(url):
                return
            BROWSER_CLOSE_FLAG.parent.mkdir(parents=True, exist_ok=True)
            BROWSER_CLOSE_FLAG.unlink(missing_ok=True)  # a stale note must not close this one
            shown_at = time.time()
            shown = True
            self._reveal.set()
            print(f"· admin page open ({url})", flush=True)
            self._watch_page(shown_at)
        finally:
            if shown:
                self._retake.set()
                print("· admin page closed", flush=True)
            BROWSER_CLOSE_FLAG.unlink(missing_ok=True)
            self._admin_busy.clear()

    def _watch_page(self, shown_at: float) -> None:
        """Wait for the page to ask to close, for the browser to die, or for the hard cap.

        The page cannot uncover or cover anything itself, so its Close button drops
        :data:`~cyclops.config.BROWSER_CLOSE_FLAG` and the decision is taken here, on the thread
        that put it up. The note counts only if it was written after we uncovered the page, so
        one left behind by an earlier round can never take it away the moment it appears.
        """
        proc = self._browser
        deadline = time.monotonic() + ADMIN_MAX_S
        while proc is not None and proc.poll() is None:
            try:
                asked = BROWSER_CLOSE_FLAG.stat().st_mtime >= shown_at
            except OSError:
                asked = False
            if asked or time.monotonic() > deadline:
                break
            time.sleep(ADMIN_POLL_S)

    def close_browser(self) -> None:
        """Take the warm browser down with the kiosk, so nothing is left covering the panel."""
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
            # On the tap, not on the teardown. Everything after this point waits on a task
            # that has to notice it was cancelled, and the panel already shows CLOSING from
            # here - the sound belongs to the press, the same way the shutter does.
            self._cues.play("closing")
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

    # ---- volume ----

    def adopt_volume(self) -> None:
        """Make the speaker and the page agree at startup.

        The note is the desired state, not a record of what happened: a level chosen from the
        page while the kiosk was down still lands when it comes back. If there is no note yet,
        seed it from the sink so the page's slider opens where the speaker actually is.
        """
        wanted = mixer.requested()
        if wanted is None:
            self._volume = mixer.level()
            if self._volume is not None:
                mixer.request(self._volume)
        elif mixer.set_level(wanted):
            self._volume = wanted

    @property
    def volume(self) -> int | None:
        """The level currently on the sink, as far as we know."""
        return self._volume

    def _sync_volume(self) -> None:
        """Follow the level the page left for us. A few bytes, a couple of times a second."""
        now = time.monotonic()
        if now - self._volume_at < VOLUME_POLL_S:
            return
        self._volume_at = now
        wanted = mixer.requested()
        if wanted is None or wanted == self._volume:
            return
        if mixer.set_level(wanted):
            self._volume = wanted
            print(f"· volume {wanted}%", flush=True)

    # ---- sleep ----

    def _sleeping(self, state: str) -> bool:
        """Should the panel be dark? True once nothing has touched it for a while.

        A live session counts as company even when nobody is touching the glass - the halo and
        the timer are a conversation's only feedback, and blanking them mid-sentence would read
        as a crash - and so does an open admin page, which is covering the panel itself. Only an
        idle kiosk goes dark, and only a tap brings it back (see :meth:`_on_mouse`).

        ``CYCLOPS_SLEEP_AFTER_S=0`` turns the blanking off altogether and the panel simply stays
        lit. Watching the camera come and go is the obvious case: the thing you are trying to
        observe is also the thing a dark panel has just released.
        """
        now = time.monotonic()
        after = self.controller.settings.sleep_after_s
        if state not in (IDLE, ERROR) or self._admin_busy.is_set():
            self._touched_at = now
        elif after and not self._asleep and now - self._touched_at > after:
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
        thread, and starting the source no longer blocks on the device: it hands the reopen to
        the supervisor and returns. A camera that is gone means a panel that says so, not a
        panel that refuses to wake.
        """
        self.backlight.on()  # light first, so the panel answers the tap before the camera can
        self.camera.start()  # never raises; if the device is absent it keeps looking for it
        self._camera_on_at = time.monotonic()
        self._asleep = False

    # ---- loop ----

    def run(self) -> None:
        frame_budget = 1.0 / TARGET_FPS
        first = self.camera.frame()
        if first is None:  # nothing plugged in yet; the window still opens, and says why
            first = message(*NO_CAMERA_SIZE, NO_CAMERA)
        h, w = first.shape[:2]
        self.open_window(first, min(w, 1280), min(h, 720))

        while self.running:
            started = time.monotonic()
            # highgui is main-thread only, so the admin thread asks for both of these rather
            # than touching the window itself.
            if self._reveal.is_set():
                self._reveal.clear()
                self._drop_window()  # the warm browser has been behind us all along
                self._hidden = True
            if self._retake.is_set():
                self._retake.clear()
                self._drop_window()  # ... and the next frame builds a window on top of it again
                self._hidden = False
                self._touched_at = time.monotonic()  # closing the page is a touch like any other
            self._sync_volume()  # the page sets the volume, so keep reading it while it is up
            if self._hidden:
                time.sleep(1.0 / ADMIN_FPS)  # nothing we draw now can be seen by anyone
                continue

            status = self.controller.status()
            state = self._effective(str(status["state"]))
            asleep = self._sleeping(state)

            frame = None  # nothing to draw: the camera is off, absent, or still coming back
            if not asleep:
                got = self.camera.latest()
                # A reopened device keeps handing back the frame it stopped on, and that is
                # last minute's room. Anything older than the reopen is not shown.
                if got is not None and got[1] >= self._camera_on_at:
                    frame = got[0]

            fallback = self._size if frame is None else (frame.shape[1], frame.shape[0])
            width, height = self._window_size(fallback)
            if (width, height) != self._size or self.overlay is None:
                self.overlay = Overlay(width, height)
                self._size = (width, height)

            if asleep:
                self._paint(_black(width, height), width, height)  # no picture, no chrome
            else:
                # The chrome is drawn through the wake-up: the buttons must answer the tap even
                # while the camera is still opening behind them.
                if frame is not None:
                    canvas = fit_to_window(mirror(frame), width, height)
                elif self.camera.connected:
                    canvas = _black(width, height)  # open; the first frame is along shortly
                else:
                    canvas = message(width, height, NO_CAMERA)
                flash = max(0.0, (self._flash_until - time.monotonic()) / FLASH_SECONDS)
                elapsed = status["elapsed"]
                chrome = self.overlay.render(
                    state=state,
                    level=float(status["level"]),
                    elapsed=None if elapsed is None else float(elapsed),
                    recording=self.controller.settings.record,
                    flash=flash,
                    pressed=self._pressed_now(),
                    # The controller's own sentence, which is the only place a session that
                    # fell over says what went wrong. The strip beneath the picture is now
                    # somewhere to put it; the old chrome could only render it as a red rim.
                    detail=str(status["detail"]),
                )
                self._paint(composite(canvas, chrome), width, height)

            spent = time.monotonic() - started
            # A tap that has to start a browser is compositing frames against a cold start it
            # is waiting on; give the core back. (Once the page is up we are not here at all -
            # the loop is asleep above.) A dark panel is cheaper still: one black frame, redrawn
            # only to keep taps and keys answered.
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
            # Never our own uncovering: that path sleeps above rather than arriving here
            # with no window to ask about.
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
    camera.start()  # returns at once; the device may only be plugged in a minute from now
    try:
        camera.wait_for_frame()
    except WebcamError as exc:
        # A missing camera is a degraded panel, not a dead one. Everything else still works -
        # SESSION starts a session, SYSTEM opens the admin page, the light and the volume
        # behave - and a Pi showing nothing at all reads as broken hardware, which sends
        # someone looking for a keyboard. Say so on the screen and carry on looking.
        print(f"· no camera yet: {exc}", file=sys.stderr, flush=True)

    webcam.set_live_source(camera)  # the shutter shoots from this same camera
    # The recorder gets the source even when nothing is plugged in: it waits its own moment for
    # a first frame and says so in the session log if none comes, and a camera present by the
    # time the next session starts is then recorded without anything being rewired.
    controller = SessionController(settings, frames=camera, entrypoint="kiosk")
    kiosk = Kiosk(controller, camera, fullscreen, screen)
    kiosk.backlight.on()  # a previous run may have been killed while the panel was dark
    kiosk.adopt_volume()
    # Get the admin browser up now rather than on the tap that wants it. It comes up in front
    # of the window opened below and is covered again a moment later - a flicker of the
    # dashboard a second or two into startup is this, and is the price of an instant gear.
    kiosk.prewarm()
    # SIGTERM (start_kiosk.sh's pkill, systemd) otherwise skips the finally below and would
    # leave a panel that looks like a dead Pi. Exit properly instead, and the light comes back.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    found = f"on camera {camera.index}" if camera.connected else "still looking for a camera"
    idle_note = (
        f"  after {settings.sleep_after_s:g}s untouched the light goes off and the camera is"
        " released\n  any tap wakes it\n"
        if settings.sleep_after_s
        else "  idle blanking is OFF (CYCLOPS_SLEEP_AFTER_S=0) - the panel stays lit\n"
    )
    print(
        f"· cyclops kiosk {found}"
        f" · font: {platform_font_note()}\n"
        f"  screen: {'x'.join(map(str, screen)) if screen else 'window-sized'}"
        f" · backlight: {kiosk.backlight.note}"
        f" · volume: {'—' if kiosk.volume is None else f'{kiosk.volume}%'}\n"
        "  the tab row along the bottom: SNAP shows Cyclops a photo · SESSION starts and stops\n"
        "  SYSTEM opens the admin page, which is where the volume lives\n"
        f"{idle_note}"
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
