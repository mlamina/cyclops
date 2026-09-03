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

from . import barge, diagram, filming, mixer, power, session, sfx, stats, webcam  # noqa: E402
from .audio import SAMPLE_RATE, resolve_device  # noqa: E402
from .backlight import Backlight  # noqa: E402
from .camera import STALE_AFTER_S, CameraSource  # noqa: E402
from .config import (  # noqa: E402
    BROWSER_CLOSE_FLAG,
    DIAGRAM_SHOWN_FLAG,
    PAGE_SERVED_FLAG,
    ConfigError,
    load_settings,
)
from .overlay import (  # noqa: E402
    CANCEL,
    IDLE,
    POWER_OFF,
    STARTING,
    STOPPING,
    Overlay,
    composite,
    fit_to_window,
    message,
    mirror,
    platform_font_note,
    session_up,
)
from .record import PanelSource  # noqa: E402
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
# 30 rather than 25, which the box has the room for: a frame costs 13.3 ms to draw on this Pi
# (render 6.4, composite 4.5, fit_to_window 2.3, mirror 0.1), so the loop still finishes one in
# well under the 33 ms it now has. The camera is slower than this and the picture is no smoother
# for it - what gets smoother is everything drawn rather than photographed, which is the eye, the
# ring, the caption and the rim, and which is most of what anyone watches on an idle panel.
TARGET_FPS = 30
# What the picture area says when there is no camera, and how big the window comes up without
# one to take a size from. The panel is 800x480; this fits it and looks deliberate on anything
# larger, which is the point - a black window with no explanation reads as a crashed Pi.
NO_CAMERA = "No camera found"
# ...and what it says when there *is* one and it has stopped talking. Worth its own words rather
# than falling back to NO_CAMERA: the device is still open and still enumerated, so "no camera
# found" would send someone off checking cables that are fine. This is the state a stalled USB
# camera leaves the panel in, and until it had a message the panel simply went on showing the
# last frame it got - a picture of a minute ago, presented as the room.
CAMERA_STALLED = "Camera stopped responding"
NO_CAMERA_SIZE = (800, 480)
TEMP_POLL_S = 5.0  # how often the heat lamp re-reads sysfs; a board warms up over minutes
NOTICE_S = 4.0  # how long one of the kiosk's own lines holds the caption
FLASH_SECONDS = 0.45
PRESS_SECONDS = 0.18  # how long the button stays visibly depressed after a tap
# How long his face has to be held to open the power menu. Long enough that no tap can arrive
# there by accident - this is the one control on the panel that ends the session you are in the
# middle of - and short enough that a finger held on a button that is not answering gets there
# before anybody concludes it is broken. Phones use half a second; this asks for a little more
# because there is no way back from one of the two things it offers.
LONG_PRESS_S = 0.7
MENU_TIMEOUT_S = 20.0  # a menu nobody chose from gives the panel back rather than holding it
# How young the box has to be for this process starting to count as the box starting. deploy's
# push.sh restarts the kiosk several times an hour, and a fourteen-second fanfare per push is a
# fanfare nobody hears as one. 0 turns it off; CYCLOPS_SOUNDS=0 turns off every cue there is.
BOOT_FANFARE_S = 180.0
# What the panel says while it finishes the session and goes. Not a caption: this is the last
# thing the screen does, and everything else on it has stopped being true.
POWER_SAYS = {power.POWEROFF: "Shutting down…", power.REBOOT: "Restarting…"}

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
# When to take the panel back after the warm-up's window maps on top of ours: gaps between
# retakes, so the last one lands about 38 s after the page was served. The stacking order cannot
# be read back - labwc raises whatever mapped last and no always-on-top hint survives that
# (measured) - so this is a schedule rather than a check, and it has to outlast Chromium rather
# than guess at it. Chromium does not map once: measured on a warm restart, its window appears
# 0.5 s after the page is served and then raises itself *again* at 3.1 s, and at boot - cold
# binary, contended box - that whole sequence stretches by an order of magnitude. The old
# schedule was two shots, at 0.6 s and 3.0 s, and a boot whose second raise landed after 3.0 s
# left the panel showing the dashboard until somebody sshed in: every tap from then on went to
# a page nothing was watching, Close included. A retake nothing was covering costs one window
# rebuild carrying the frame it is about to show, which is invisible; the failure is total.
PANEL_RETAKE_S = (0.6, 1.2, 2.4, 4.8, 9.6, 19.2)
# How long the page gets to lay a drawing out before we uncover it anyway. Generously over the
# ~400 ms poll plus a JointJS layout, because the cost of being wrong is asymmetric: uncovering
# early shows the dashboard for a moment, and never uncovering loses the diagram entirely.
DIAGRAM_WAIT_S = 8.0
VOLUME_POLL_S = 0.4  # how often we look for a volume, or a barge-in switch, the page left us
# ...and how often we check that the window still fills the panel. See _keep_fullscreen: this is
# a compositor's answer being verified rather than a value being read, so it can be lazy.
FULLSCREEN_POLL_S = 0.5
FULLSCREEN_COMPLAINTS = 3  # say it a few times and then stop; the fix is silent after that
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


def lost_fullscreen(shown: tuple[int, int], screen: tuple[int, int] | None) -> bool:
    """Did the last image land somewhere other than the whole panel?

    A fullscreen window on an 800x480 panel shows an 800x480 picture at 800x480, because that is
    what :meth:`Kiosk._window_size` renders for it. Anything else means the window is not
    actually fullscreen, whatever we last asked for - see :meth:`Kiosk._keep_fullscreen`. The
    measured shape of the fault this was written for is 696x418 on an 800x480 panel: the picture
    fitted, aspect intact, into the client area of a window that came back decorated.
    """
    if screen is None:
        return False
    width, height = shown
    return width > 0 and height > 0 and (width, height) != screen


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


def _noted_since(flag: Path, since: float) -> bool:
    """Has this note been left, and left *after* ``since``?

    The freshness half is the whole point: a note from an earlier round - or, for the page flag,
    one left by our own reachability probe a moment before - would otherwise answer for something
    that has not happened yet.
    """
    try:
        return flag.stat().st_mtime >= since
    except OSError:
        return False


def _wait_for_flag(flag: Path, since: float, timeout: float) -> bool:
    """Wait until :func:`_noted_since` says so, or give up. True if the note arrived."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _noted_since(flag, since):
            return True
        time.sleep(ADMIN_POLL_S)
    return False


def _wait_for_page(proc: subprocess.Popen, since: float, timeout: float) -> bool:
    """Wait for the admin service to say it has handed the page to a browser here.

    Chromium announces nothing when it is ready, and nothing can be asked what the panel is
    showing, so the service leaves a note instead (:data:`~cyclops.config.PAGE_SERVED_FLAG`).
    Unlike :func:`_wait_for_flag` this also gives up the moment the browser dies, which is the
    difference between a slow start and one that is not coming.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        if _noted_since(PAGE_SERVED_FLAG, since):
            return True
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
        # Where every painted frame is published, for a session that is recording the screen
        # rather than the camera. Its own object rather than a handle back to this one, because
        # the recorder is given it directly and must not be able to reach anything else here.
        # Empty until the first paint, and that cannot matter: the only thing that starts a
        # session is a tap, taps are dispatched from inside waitKey, and the callback that
        # catches them is not installed until the window's first frame is already up.
        self.panel = PanelSource()
        self.fullscreen = fullscreen
        self.overlay: Overlay | None = None
        self.backlight = Backlight()  # the panel's light, off while it sleeps
        self.running = True
        self._flash_until = 0.0
        self._press_until = 0.0
        self._pressed: str | None = None  # which button is still showing its tap
        # His face is the one control that carries two things - a tap for what the box has kept,
        # a hold for the power menu - so it is the one that has to wait for your finger to come
        # off before it knows which you meant. None when nothing is being held down.
        self._eye_down_at: float | None = None
        self._menu = False  # the power menu has the panel; nothing behind it is live
        self._menu_until = 0.0
        # What was chosen, honoured after the whole teardown below has run: the session's video
        # is still being muxed while the panel says goodbye, and systemd starts killing units the
        # moment the command underneath this returns. See main().
        self.power: str | None = None
        self._power_at = 0.0  # ...but not before the row you pressed has been seen to invert
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
        self._heat = ""  # stats.heat_alarm(), re-read on TEMP_POLL_S; drives the lamp on the strip
        self._heat_at = 0.0
        # The kiosk's own voice in the caption, for the things that happen to it rather than to
        # the session - a shutter that could not take a photo being the one that matters. The
        # controller owns that line the rest of the time and knows nothing about any of this.
        self._notice = ""
        self._notice_until = 0.0
        self._volume: int | None = None  # the level we last put on the sink
        self._volume_at = 0.0  # when we last looked for a new one
        # The switch's position as we last read it, so a session in progress is only told when
        # it actually changes. Seeded from the note rather than from nothing: at boot the page
        # and the session already agree, and there is nothing to say.
        self._barge_margin = barge.margin_db(controller.settings)
        self._barge_at = 0.0
        self._browser: subprocess.Popen | None = None  # the admin browser, kept warm from boot
        self._warm_at = time.time()  # when it was launched, so a note older than it means nothing
        self._close_at = 0.0  # when we last looked for a Close from a page nobody here put up
        self._admin_busy = threading.Event()  # set from the tap until the page is done with
        # A second latch rather than reusing _admin_busy, which the tab row reads to decide
        # whether the eye is lit (see _pressed_now). A diagram is not that page, and a
        # panel that lights the eye whenever Cyclops draws would be telling the truth about the
        # browser and a lie about what you are looking at. Both still gate _open_admin, so the
        # two can never be up at once.
        self._page_busy = threading.Event()  # any page has the panel: the admin one or a diagram
        self._reveal = threading.Event()  # asks the render loop to uncover the admin page
        self._retake = threading.Event()  # ... and to take the panel back off it
        self._hidden = False  # the admin page has the panel; nothing we draw can be seen
        self._window_up = False  # whether highgui currently has a window for us
        # Deliberately "now" rather than zero, so the first check happens one interval in - by
        # which time run() has painted a panel-sized frame over open_window's camera-sized one,
        # and the rect being read is one worth reading.
        self._fullscreen_at = time.monotonic()
        self._fullscreen_fixes = 0
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

        Everything that reaches the glass comes through here, which is why this is where a
        session recording the screen takes its frames from - the dark panel and the "No camera
        found" card included, since those are as much what you were looking at as the picture is.
        """
        self.panel.publish(image)  # a recording of the screen samples this; see PanelSource
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

    def _keep_fullscreen(self, now: float) -> None:
        """Make sure the window still fills the panel, and ask again when it does not.

        Fullscreen is a request to the compositor, not a fact. Under XWayland one made before
        the surface is mapped is dropped in silence - which is why :meth:`open_window` shows a
        frame before asking - and the window is left decorated instead: an 800x480 outer frame
        with an 800x418 client area, into which highgui fits the 800x480 picture at 87% with
        black around it. That is what somebody sees as "the panel got smaller and grew a black
        bar along the bottom".

        Nothing recovers from that on its own, which is the reason this exists rather than a
        longer wait in ``open_window``. The render size comes from ``self.screen``, so it stays
        right while only the window is wrong, and every later frame is fitted into the same
        too-small client area. It is wrong until the kiosk is restarted.

        Taking the panel back from a page is when it happens: we destroy and rebuild our window
        in the same instant the browser behind us is tearing down whatever it was showing, and
        the busier that is - a megapixel photo rather than a diagram - the likelier the request
        lands too early. So it is checked rather than assumed. ``getWindowImageRect`` reports
        where the last image actually landed, which is exactly the question being asked; it is
        useless for *discovering* the panel size, for the reason :meth:`_window_size` gives, and
        ideal for confirming one we already know.
        """
        if not (self.fullscreen and self._window_up and self.screen is not None):
            return
        if now - self._fullscreen_at < FULLSCREEN_POLL_S:
            return
        self._fullscreen_at = now
        try:
            _, _, width, height = cv2.getWindowImageRect(WINDOW)
        except cv2.error:
            return  # no window to ask about; the next paint builds one
        if not lost_fullscreen((width, height), self.screen):
            return
        self._fullscreen_fixes += 1
        if self._fullscreen_fixes <= FULLSCREEN_COMPLAINTS:
            print(
                f"· the window came back {width}x{height} on a"
                f" {self.screen[0]}x{self.screen[1]} panel; asking for fullscreen again",
                flush=True,
            )
        self._apply_fullscreen()

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
        """Touchscreen taps arrive here as ordinary mouse events via XWayland.

        Two of the three tabs act on the press, which is what a screen with no travel should do:
        the flash and the shutter belong to the moment your finger lands. His face is the one
        exception, because it now carries two things - a tap for what the box has kept, a hold
        for the power menu - and the release is the only event that can tell them apart. That is
        the bargain every phone makes, and it costs the eye nothing anybody can feel: what the
        tap does is uncover a browser that has been warm since boot.
        """
        if self.overlay is None:
            return
        if event == cv2.EVENT_LBUTTONUP:
            self._lifted(x, y)
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        self._touched_at = time.monotonic()
        # Whatever was being held, this is not it any more. A press whose release never arrived
        # would otherwise sit there and turn the next tap into a hold that was already half done.
        self._eye_down_at = None
        if self._asleep:
            # The tap that wakes the panel is spent waking it. With the preview dark you cannot
            # see what you are aiming at, so it must not also fire whatever sits underneath.
            self._wake()
            return
        if self._menu:
            # Modal, and it has to be: two of its three rows end the box. Nothing behind the
            # card can be reached while it is up, and anywhere off the card is a way out.
            self._choose(self.overlay.menu_hit(x, y))
            return
        boxes = self.overlay.hitboxes
        if boxes.shutter.contains(x, y):
            self._press("shutter")
            self._snap()
        elif boxes.eye.contains(x, y):
            # His face, and what it opens: everything the box has kept. The eye moved into the
            # middle of the row and took the gear's job with it, which is the right way round -
            # you tap him to ask what he remembers. Held rather than tapped, it opens the power
            # menu instead; both are decided in _lifted and _holding.
            self._press("eye")
            # It sounds for exactly as long as he is held, so the cue is his face answering
            # under the finger rather than a notification about it: _lifted ends it. Nothing
            # else on this panel works that way because nothing else is held - the shutter is
            # over before you have finished pressing it.
            self._cues.play("pressed")
            self._eye_down_at = self._touched_at
        elif boxes.wake.contains(x, y):
            self._press("wake")
            self._toggle_session()

    def _lifted(self, x: int, y: int) -> None:
        """A finger coming off the glass. Only his face has anything left to do here."""
        down_at, self._eye_down_at = self._eye_down_at, None
        if down_at is None or self.overlay is None:
            return  # nothing was being held, or the hold already landed and opened the menu
        # The finger is off him, so his answer stops with it - wherever it came off, because
        # what ended is the press and not the tap. Safe to be unconditional only because of the
        # guard above: a hold that already opened the menu left `_eye_down_at` at None and
        # returned up there, so this can never cut "menu" off half a beat after sounding it.
        self._cues.stop()
        # Lifted somewhere else: the tap was taken back, which is what sliding off a button has
        # meant since the first one.
        if self.overlay.hitboxes.eye.contains(x, y):
            self._open_admin()

    def _holding(self) -> float:
        """How far a press on his face has got towards the power menu, 0 to 1.

        0 whenever nothing is being held, which is what the panel draws no collar for.
        """
        down_at = self._eye_down_at
        if down_at is None:
            return 0.0
        return min(1.0, (time.monotonic() - down_at) / LONG_PRESS_S)

    # ---- the power menu ----

    def _open_menu(self) -> None:
        """The hold has landed: put the menu up."""
        self._eye_down_at = None
        self._pressed = None  # he stops being held: the menu itself is the acknowledgement
        self._menu = True
        self._menu_until = time.monotonic() + MENU_TIMEOUT_S
        # A sound, because the menu opens under the very finger that is covering the eye: the
        # panel's answer to the hold is the one piece of feedback a hand can be in the way of.
        self._cues.play("menu")
        print("· power menu", flush=True)

    def _close_menu(self) -> None:
        self._menu = False
        self._touched_at = time.monotonic()  # putting it away is a touch like any other

    def _choose(self, key: str | None) -> None:
        """Act on a tap while the menu is up. Its rows act on the press, as every tab does.

        Pointedly *not* on the release: the menu opens under a finger that is still down on his
        face, and a row that acted on a lift would be chosen by the very press that asked for the
        menu - most likely SHUT DOWN, which is the row his face is behind.
        """
        if key is None or self.power is not None:
            return  # a tap on the card but on no row, or a choice already made and under way
        self._menu_until = time.monotonic() + MENU_TIMEOUT_S
        if key == CANCEL:
            self._close_menu()
            return
        self._press(key)
        self.power = power.POWEROFF if key == POWER_OFF else power.REBOOT
        self._power_at = time.monotonic() + PRESS_SECONDS  # let the row be seen to invert
        print(f"· {self.power} asked for from the panel", flush=True)

    def _farewell(self) -> None:
        """Say what is happening and stop drawing. The rest is main()'s teardown, then the box.

        The message stays on the panel for the whole of that teardown - a session being muxed
        and named can take a good few seconds - so what a box being shut down looks like is a
        screen saying so, rather than a picture that froze.
        """
        width, height = self._size if all(self._size) else NO_CAMERA_SIZE
        self._menu = False
        self._paint(message(width, height, POWER_SAYS[self.power]), width, height)
        cv2.waitKey(1)  # highgui only puts a frame up from inside one of these
        self.running = False

    def _press(self, button: str) -> None:
        self._pressed = button
        self._press_until = time.monotonic() + PRESS_SECONDS

    def _pressed_now(self) -> str | None:
        """Which control to draw as held - a tab, or a row of the power menu.

        His cell stays lit while its page is up, and for as long as a finger is on him: a hold
        that is going somewhere should look held for all of the second it takes, not for the
        180 ms a tap gets.

        Uncovering a warm browser is immediate, so the page half of this is normally seen for a
        frame or two. It still earns its place on the one tap that has to start a browser:
        without it the tab goes dark 180 ms in and a panel that is busy looks like a panel that
        ignored you.
        """
        if self._admin_busy.is_set() or self._eye_down_at is not None:
            return "eye"
        return self._pressed if time.monotonic() < self._press_until else None

    def _say(self, message: str) -> None:
        """Put one of the kiosk's own lines in the caption for a few seconds. Any thread."""
        self._notice = message
        self._notice_until = time.monotonic() + NOTICE_S

    def _saying(self) -> str:
        """That line while it is still current, else "" - and then the controller's own wins."""
        return self._notice if time.monotonic() < self._notice_until else ""

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
            # On the panel, not only on a log nobody is reading. This is the failure the shutter
            # can actually have in normal use - the camera stalled, so there is no recent frame
            # to photograph - and it used to be completely silent: the screen flashed, the
            # shutter clicked, and no photo existed. A flash that means "taken" has to be able
            # to mean "not taken" too.
            self._say(CAMERA_STALLED if self.camera.connected else NO_CAMERA)
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
        """The page, opened on the sessions list rather than on the four numbers.

        Tapping his face asks what he remembers, so it had better land on some. The hash is what
        the page routes on and it never navigates, so this only decides which screen the warm
        browser is holding when it is uncovered - the numbers are one tap away inside it.
        """
        return f"http://127.0.0.1:{self.controller.settings.admin_port}/#/sessions"

    def prewarm(self) -> None:
        """Start the admin browser now, in the background, so a tap only has to uncover it."""
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
            if self._page_busy.is_set():
                return  # something took the panel while we were starting: the page is theirs now
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
        self._warm_at = launched_at
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
        if self._page_busy.is_set():
            return
        self._page_busy.set()
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
            # The eye promises the dashboard, so make sure that is what is behind our window.
            # The warm browser shows whatever was last offered to the panel, and an offer that
            # was never shown - a crashed session, a diagram whose show() lost the race with
            # this tap - would otherwise be what the tap uncovers. The two can never legitimately
            # be up at once: _page_busy gates both.
            diagram.withdraw()
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
            self._page_busy.clear()

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
            if _noted_since(BROWSER_CLOSE_FLAG, shown_at) or time.monotonic() > deadline:
                break
            time.sleep(ADMIN_POLL_S)

    # ---- diagrams ----

    def show_diagram(self) -> bool:
        """Put the diagram waiting in ``DIAGRAM_FILE`` on the panel. False if the panel is busy.

        Called from the agent's thread, so it does nothing here but set a flag and start a
        thread: highgui belongs to the render loop and this is not it.
        """
        if self._page_busy.is_set():
            return False  # the admin page is up, or a diagram already is; do not stack them
        self._page_busy.set()
        threading.Thread(target=self._diagram_session, name="kiosk-diagram", daemon=True).start()
        return True

    def _diagram_session(self) -> None:
        """The whole life of one diagram: wait for it to be drawn, show it, wait, take it back.

        The same shape as :meth:`_admin_session` and for the same reasons, with one difference:
        the admin page is already loaded in the warm browser, and a diagram is not. So this waits
        for the page to say it has actually painted before uncovering, rather than uncovering onto
        a dashboard that turns into a diagram half a second later while somebody is watching.
        """
        url = self._admin_url()
        shown = False
        try:
            if not _admin_reachable(url) or not self._ensure_browser(url):
                print("· diagram: no page to draw it on", file=sys.stderr, flush=True)
                return
            DIAGRAM_SHOWN_FLAG.parent.mkdir(parents=True, exist_ok=True)
            DIAGRAM_SHOWN_FLAG.unlink(missing_ok=True)  # a stale note must not answer for this one
            BROWSER_CLOSE_FLAG.unlink(missing_ok=True)
            asked_at = time.time()
            if not _wait_for_flag(DIAGRAM_SHOWN_FLAG, asked_at, DIAGRAM_WAIT_S):
                # Uncover anyway. The page polls, so it is probably a slow layout rather than a
                # dead browser, and a diagram arriving a moment late beats one that never comes.
                print("· diagram: the page was slow to draw; showing anyway", flush=True)
            shown_at = time.time()
            shown = True
            self._reveal.set()
            # Something was made and it is on the panel now: look up. The one place that is true
            # for both kinds, since a diagram drawn or found and a photo he imagined all arrive
            # here through diagram.show(). Pointedly not _admin_session's reveal, which is his
            # eye opening the dashboard and already has a sound of its own.
            self._cues.play("shown")
            print("· diagram on the panel", flush=True)
            self._watch_page(shown_at)
        finally:
            # The drawing goes before the panel comes back, so the page has already switched
            # itself off the diagram by the time it is visible again behind the window.
            diagram.withdraw()
            if shown:
                self._retake.set()
                print("· diagram closed", flush=True)
            BROWSER_CLOSE_FLAG.unlink(missing_ok=True)
            self._page_busy.clear()

    def close_browser(self) -> None:
        """Take the warm browser down with the kiosk, so nothing is left covering the panel."""
        proc = self._browser
        if proc is not None and proc.poll() is None:
            _stop_browser(proc)

    # ---- session ----

    def _toggle_session(self) -> None:
        """Act on the tap and record what we asked for, so the UI can show it at once."""
        state = self.controller.status()["state"]
        # The same question the tab's own label asks, so what the button says and what the button
        # does cannot drift: WAKE UP is an imperative, and one that stays put once it is no longer
        # what the tap will do is a lie.
        starting = not session_up(str(state))
        self._pending = "start" if starting else "stop"
        self._pending_at = time.monotonic()
        if starting:
            # What this session's video will be of, settled here because here is the last moment
            # it is free: the encoder is opened at a fixed frame size a second or two from now.
            # A switch flipped after this lands on the next session, which is what the settings
            # screen says it does. See cyclops.filming.
            wanted = filming.chosen(self.controller.settings)
            self.controller.set_record_source(
                self.panel if wanted == filming.SCREEN else self.camera
            )
            print(f"· recording the {wanted}", flush=True)
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
        # The timeout guards against a session that never corroborates - but a teardown that
        # is still visibly working *is* corroboration, and it can legitimately take minutes: the
        # mux alone gets record.MUX_TIMEOUT_S. Giving up on it at eight seconds would drop a
        # green LISTENING strip over a caption saying, correctly, that the video is still being
        # written, which is the one moment this panel most needs to be believed.
        stale = time.monotonic() - self._pending_at > PENDING_TIMEOUT_S
        if done or (stale and not self.controller.closing):
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

    def _sync_stranded(self) -> None:
        """Answer the page's Close button even when nothing here uncovered it.

        The warm browser is meant to sit behind the panel, and :data:`PANEL_RETAKE_S` is a
        schedule rather than a certainty. When a retake loses the race, the panel is covered by a
        page no thread of ours is watching - :meth:`_watch_page` only runs for a page this kiosk
        put up - so every tap goes to the dashboard and its Close button writes a note nobody
        reads. That is the state this exists for: honour the note here too, and the one gesture
        somebody standing at the panel would try is also the one that works.
        """
        now = time.monotonic()
        if now - self._close_at < ADMIN_POLL_S:
            return
        self._close_at = now
        if self._page_busy.is_set():
            return  # a page we put up: _watch_page owns the note, and the retake after it
        if _noted_since(BROWSER_CLOSE_FLAG, self._warm_at):
            BROWSER_CLOSE_FLAG.unlink(missing_ok=True)
            self._retake.set()
            print("· panel taken back from the warm browser", flush=True)

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

    # ---- barge-in ----

    def _sync_barge_in(self) -> None:
        """Follow the settings screen's answer to "may I talk over you?" into a live session.

        Read here rather than in the session because a session is the wrong length of thing to
        hold this: it can run for an hour, and the whole point of the switch is that you reach
        for it in the middle of one, having just been cut off by your own voice.
        """
        now = time.monotonic()
        if now - self._barge_at < VOLUME_POLL_S:
            return
        self._barge_at = now
        margin = barge.margin_db(self.controller.settings)
        if margin == self._barge_margin:
            return
        self._barge_margin = margin
        self.controller.set_barge_in(margin)
        print(f"· barge-in {'off' if margin is None else f'on at {margin:g} dB'}", flush=True)

    # ---- sleep ----

    def _sleeping(self, state: str) -> bool:
        """Should the panel be dark? True once nothing has touched it for a while.

        A live session counts as company even when nobody is touching the glass - the halo and
        the timer are a conversation's only feedback, and blanking them mid-sentence would read
        as a crash - and so does an open admin page, which is covering the panel itself, and so
        does the power menu, which is a question waiting for an answer. Only an idle kiosk goes
        dark, and only a tap brings it back (see :meth:`_on_mouse`).

        ``CYCLOPS_SLEEP_AFTER_S=0`` turns the blanking off altogether and the panel simply stays
        lit. Watching the camera come and go is the obvious case: the thing you are trying to
        observe is also the thing a dark panel has just released.
        """
        now = time.monotonic()
        after = self.controller.settings.sleep_after_s
        if session_up(state) or self._page_busy.is_set() or self._menu:
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

        This wakes the *panel*; the WAKE UP tab wakes Cyclops. The two senses never contradict
        each other on screen because they nest: :meth:`_sleeping` only ever blanks the glass
        while the session is down, so the panel can only be dark when he is already asleep, and
        the tap that lights it is spent doing that and fires no button underneath.

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
        # The first frame is on the glass and the tab row can be pressed, which is the only
        # thing anybody was waiting for. Here rather than inside open_window, which is called
        # again for every retake of the panel from the browser. One cue at a time means this
        # also *ends* the fanfare main() started: that one is a bed under however long getting
        # here took, and this is the note that resolves it.
        self._cues.play("started")

        while self.running:
            started = time.monotonic()
            # highgui is main-thread only, so the admin thread asks for both of these rather
            # than touching the window itself.
            if self._reveal.is_set():
                self._reveal.clear()
                self._drop_window()  # the warm browser has been behind us all along
                self._hidden = True
                # A session recording the screen is still sampling, and the screen is no longer
                # ours to hand it. Black rather than the frame we happened to stop on: a diagram
                # can hold the panel for a quarter of an hour mid-session, and a frozen halo over
                # a running timer watches back as a hung encoder rather than as what happened.
                # `or` would not do: (0, 0) before the first frame is a truthy tuple, and
                # a 0x0 frame is one the recorder cannot resize and gives up over.
                self.panel.publish(_black(*(self._size if all(self._size) else NO_CAMERA_SIZE)))
            if self._retake.is_set():
                self._retake.clear()
                self._drop_window()  # ... and the next frame builds a window on top of it again
                self._hidden = False
                self._touched_at = time.monotonic()  # closing the page is a touch like any other
            self._sync_volume()  # the page sets the volume, so keep reading it while it is up
            self._sync_barge_in()  # ...and whether it may be interrupted, on the same beat
            self._sync_stranded()  # ...and whether the warm browser has ended up in front of us
            if self._hidden:
                time.sleep(1.0 / ADMIN_FPS)  # nothing we draw now can be seen by anyone
                continue
            # Only once the panel is ours again: while a page has it there is no window of ours
            # to measure, and the retake is the very thing this is here to catch.
            self._keep_fullscreen(started)

            # The one gesture on this panel that is not a tap, resolved here rather than in the
            # callback: a finger held still sends no events at all, so the moment a hold becomes
            # a hold can only be noticed by a clock that is already running.
            hold = self._holding()
            if hold >= 1.0:
                self._open_menu()
            if self._menu and started > self._menu_until:
                self._close_menu()  # nobody chose; the panel is not the menu's to keep

            status = self.controller.status()
            state = self._effective(str(status["state"]))
            asleep = self._sleeping(state)

            # Two sysfs reads, five seconds apart, on a board that takes minutes to change
            # temperature. Cheap enough to do while the panel is dark, which is worth it: the
            # lamp is then already right on the first frame after a tap rather than five
            # seconds into it.
            if started - self._heat_at >= TEMP_POLL_S:
                self._heat_at = started
                self._heat = stats.heat_alarm(stats.cpu_temp_c(), self._heat)

            frame = None  # nothing to draw: the camera is off, absent, or still coming back
            stalled = False  # ...or open, enumerated, and no longer delivering anything
            if not asleep:
                got = self.camera.latest()
                # A reopened device keeps handing back the frame it stopped on, and that is
                # last minute's room. Anything older than the reopen is not shown.
                if got is not None and got[1] >= self._camera_on_at:
                    # Nor is anything older than STALE_AFTER_S, whoever produced it. A USB
                    # camera that stalls stays open and stays enumerated while delivering
                    # nothing at all, so the last frame it managed goes on being the newest one
                    # for as long as that lasts - and drawing it is the panel telling a lie it
                    # has no way to catch itself in. Past that age, say so instead.
                    if started - got[1] <= STALE_AFTER_S:
                        frame = got[0]
                    else:
                        stalled = True

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
                elif stalled:
                    canvas = message(width, height, CAMERA_STALLED)
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
                    # The controller's own sentence: what is being searched for, which
                    # project is being opened, how far the teardown has got, and - the one it
                    # was added for - what went wrong, which the old chrome could only render
                    # as a red rim. Empty whenever the state alone says it all.
                    # The kiosk's own line wins while it is current, because the things it has
                    # to say are about the panel in front of you rather than about the session -
                    # a shutter that took no photo is worth interrupting "listening — talk to
                    # me" for, and it has four seconds to do it in.
                    detail=self._saying() or str(status["detail"]),
                    # One instant for the whole frame, taken at the top of the loop. The caption
                    # breathes and counts its dots off this rather than off a clock of its own,
                    # so the animation cannot drift between elements or with the frame rate.
                    phase=started,
                    heat=self._heat,
                    hold=hold,
                    menu=self._menu,
                )
                self._paint(composite(canvas, chrome), width, height)

            # Last of all, so the row that was pressed has had its frame on the glass: the panel
            # says what it is doing and the loop ends. Everything after this is main()'s
            # teardown, which finishes the session before the box is taken down.
            if self.power is not None and time.monotonic() >= self._power_at:
                self._farewell()
                break

            spent = time.monotonic() - started
            # A tap that has to start a browser is compositing frames against a cold start it
            # is waiting on; give the core back. (Once the page is up we are not here at all -
            # the loop is asleep above.) A dark panel is cheaper still: one black frame, redrawn
            # only to keep taps and keys answered.
            if self._asleep:
                budget = 1.0 / SLEEP_FPS
            elif self._page_busy.is_set():
                budget = 1.0 / ADMIN_FPS
            else:
                budget = frame_budget
            # Handing waitKey the whole rest of the slot looks like a place to save a core and
            # is not: it blocks. Measured in the loop itself at 30 fps, a frame is 16.6 ms of
            # work in a 33 ms slot - 13.3 ms drawing it, 3.3 ms in here presenting it - and the
            # render thread sits at 44% of a core. Pumping with waitKey(1) and sleeping the
            # remainder was tried and measured identical, for the price of a tap waiting a frame
            # for the next pump, so it was taken back out. The number that makes this look like
            # a spin is the *process* at 74%, which is not this thread: the camera's four MJPEG
            # decode threads are the other 30%, and they are where to go looking next.
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

    # The box is up. This process starting *is* the desktop session being up, because labwc
    # autostarts it, so this is the first moment anything on the Pi can make a noise - and on a
    # power-on that is exactly what it is. It plays under everything below until run() gets a
    # window onto the glass and cuts it off with "started". Only on a real power-on, though:
    # see BOOT_FANFARE_S. /proc/uptime is the whole test, and it is None off Linux, so a Mac
    # never hears this. sfx.play rather than a Cues because this makes one noise, once, before
    # there is a Kiosk to own it.
    up = stats.uptime_s()
    if settings.sounds and up is not None and up < BOOT_FANFARE_S:
        sfx.play("booted", rate=SAMPLE_RATE, device=resolve_device(settings.output_device))

    camera = CameraSource(settings.camera_index)
    camera.start()  # returns at once; the device may only be plugged in a minute from now
    try:
        camera.wait_for_frame()
    except WebcamError as exc:
        # A missing camera is a degraded panel, not a dead one. Everything else still works -
        # WAKE UP starts a session, the eye opens the admin page, the light and the volume
        # behave - and a Pi showing nothing at all reads as broken hardware, which sends
        # someone looking for a keyboard. Say so on the screen and carry on looking.
        print(f"· no camera yet: {exc}", file=sys.stderr, flush=True)

    webcam.set_live_source(camera)  # the shutter shoots from this same camera, always
    # Whichever of the two the tap picks, the recorder gets a source even when nothing is plugged
    # in: it waits its own moment for a first frame and says so in the session log if none comes,
    # and a camera present by the time the next session starts is then recorded without anything
    # being rewired. The camera is the constructor's answer only so a session started by anything
    # but _toggle_session still has one; _toggle_session always says which of the two it wants.
    controller = SessionController(settings, frames=camera, entrypoint="kiosk")
    kiosk = Kiosk(controller, camera, fullscreen, screen)
    diagram.set_panel(kiosk)  # so a finished diagram can find a panel to appear on
    # Nothing can be waiting for a panel that has only just come up, so anything here is a
    # leftover from a process that is gone - and the browser prewarmed below would paint it.
    diagram.withdraw()
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
        f"  after {settings.sleep_after_s:g}s untouched the panel's light goes off and the"
        " camera is released\n  any tap lights it again\n"
        if settings.sleep_after_s
        else "  idle blanking is OFF (CYCLOPS_SLEEP_AFTER_S=0) - the panel stays lit\n"
    )
    print(
        f"· cyclops kiosk {found}"
        f" · font: {platform_font_note()}\n"
        f"  screen: {'x'.join(map(str, screen)) if screen else 'window-sized'}"
        f" · backlight: {kiosk.backlight.note}"
        f" · volume: {'—' if kiosk.volume is None else f'{kiosk.volume}%'}\n"
        "  the tab row along the bottom: SNAP shows Cyclops a photo · tap his eye in the\n"
        "  middle for the recordings and pictures, and for the volume · WAKE UP wakes him,\n"
        "  and says GO TO SLEEP while he is up\n"
        f"  hold his eye for {LONG_PRESS_S:g}s to shut the box down or restart it\n"
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
        diagram.set_panel(None)
        diagram.withdraw()  # nothing should be waiting for a panel that is gone
        webcam.set_live_source(None)
        controller.stop()
        controller.join(SHUTDOWN_JOIN_S)  # let it finish writing before the camera goes away
        camera.stop()
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # let highgui actually tear the window down

    # Last of everything, and only if the panel asked for it: the session is on the card by now,
    # the video is muxed and the camera is back. systemd starts killing units the moment this
    # returns, which is exactly why it is here and not in the tap that chose it.
    if kiosk.power is not None:
        print(f"· {kiosk.power}", flush=True)
        if not power.take_down(kiosk.power):
            print(f"· {kiosk.power} was refused; the box is still up", file=sys.stderr,
                  flush=True)


if __name__ == "__main__":
    main()
