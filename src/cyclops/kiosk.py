"""Fullscreen OpenCV kiosk: the live camera *is* the UI, with tappable controls drawn on it.

``cyclops-kiosk`` is the whole front-end and needs no browser of its own. It owns the camera
(via :class:`~cyclops.camera.CameraSource`), draws each frame into a fullscreen OpenCV window
with a PIL-rendered overlay on top, and turns taps into session control. The agent itself runs
on a background thread inside a :class:`~cyclops.ui.SessionController`, and the photos the
button beside the panel feeds it are borrowed from the very camera you are watching.

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

# Before cv2, deliberately: importing it and the OpenAI client is most of what a cold start on
# this box costs, and a clock started after them would report a boot that took forty seconds as
# having taken four. See _phase.
_BEGAN = time.monotonic()

# The opencv-python wheel bundles Qt but no fonts, so Qt prints a five-line QFontDatabase
# complaint on every window. Only the blanket rule silences it - the narrower
# `qt.qpa.fonts.warning=false` and `QT_QPA_FONTDIR` were both measured to have no effect.
# This gags Qt's own logging only; Python warnings and exceptions are untouched. Must be set
# before cv2 imports, because Qt reads it once at load.
os.environ.setdefault("QT_LOGGING_RULES", "*.warning=false")

import cv2  # noqa: E402 - must follow the QT_LOGGING_RULES default above
import numpy as np  # noqa: E402 - kept with cv2, which pulls it in anyway

from . import (  # noqa: E402
    barge,
    filming,
    imagine,
    mixer,
    panel,
    power,
    session,
    sfx,
    stats,
    still,
    voice,
    webcam,
)
from .audio import SAMPLE_RATE, resolve_device  # noqa: E402
from .backlight import Backlight  # noqa: E402
from .button import RING_ACTIVE, RING_ERROR, RING_IDLE, ShutterButton  # noqa: E402
from .camera import STALE_AFTER_S, CameraSource  # noqa: E402
from .config import (  # noqa: E402
    BROWSER_CLOSE_FLAG,
    PAGE_ALIVE_FLAG,
    PAGE_SCREEN_FILE,
    PAGE_SERVED_FLAG,
    PANEL_PAINTED_FLAG,
    SAY_VOICE_FILE,
    ConfigError,
    load_settings,
)
from .overlay import (  # noqa: E402
    CANCEL,
    HEAT,
    IDLE,
    POWER_OFF,
    STARTING,
    STOPPING,
    VOLUME,
    VOLUME_STEP,
    Overlay,
    composite,
    fit_to_window,
    message,
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
# 30 rather than 25, which the box has the room for: a frame costs 13.2 ms to draw on this Pi
# (render 6.4, composite 4.5, fit_to_window 2.3), so the loop still finishes one in
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
# What the panel says while it finishes the session and goes. Not a caption: this is the last
# thing the screen does, and everything else on it has stopped being true.
POWER_SAYS = {power.POWEROFF: "Shutting down…", power.REBOOT: "Restarting…"}

# What has to be true before a grab on the knob is a drag. The column is up either way - it comes
# up on the touch - but until both of these hold, nothing is asked of the speaker.
#
# Two floors, guarding two different things. The travel floor is about the glass: a touchscreen
# delivers a pixel or two on every press, and a tap that counted would be a tap that moved the
# level. The track is about the geometry, and it is the one that matters now that the sink
# follows the finger: the column reads *absolutely* and the knob sits below the foot of it, so
# the first thing a drag off the disc reports is silence. Left ungated that would cut him off
# mid-sentence every time you reached for the volume, and you would climb back out of nothing.
# Waiting for the finger to arrive on the ladder means the column, the speaker and the beep agree
# at every instant. Silence is still somewhere you can drag to, by going back down off the foot.
SLIDE_GRAB_PX = 6
# One rung of the column: the tick's own length and a beat. It is what makes a sweep read as
# twenty detents rather than one buzz, and it is also the whole governor on what this gesture
# costs - a pactl and a stream open per rung, capped near fourteen a second.
SLIDE_RUNG_S = 0.045
# Which screen the panel asks the page for, by what opened it. His face promises the recordings,
# the gauge promises the rest of the numbers behind itself.
SESSIONS_SCREEN, SYSTEM_SCREEN = "/sessions", "/"
PAGE_ROUTE_S = 0.5  # one poll of /api/panel, plus a little: how long the page has to route itself

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
# for the page itself.
#
# It is started *before this kiosk has a window at all*, with the panel's light off, and that
# ordering is the whole trick. labwc raises whatever mapped last and no always-on-top hint
# survives that, and the stacking order cannot be read back (both measured) - so the only way to
# be reliably on top of Chromium is to map after it has finished mapping. Chromium does not map
# once: on a warm restart its window appears 0.5 s after the page is served and raises itself
# again at 3.1 s, and at boot that stretches by an order of magnitude.
#
# The previous arrangement started it *behind* an existing window and then destroyed and rebuilt
# that window six times on a blind schedule to claw the panel back, on the theory that a retake
# covering nothing is invisible. On a cold boot it is not: it is six unmap/map round-trips with
# the dashboard showing through each gap, which is what "the UI flickers for a while" was.
# Gap between asks while the admin service is still coming up. Short, because this runs inside
# the dark stretch: the service is up from boot on this box and a probe only misses when
# gunicorn is busy answering its first request, which is over in well under a second. A two
# second gap spent three of them waiting for a service that was already listening.
ADMIN_RETRY_S = 0.5
PAGE_WAIT_S = 30.0  # how long the browser gets to fetch the page; a cold start eats 9 s of it
# The whole dark stretch, capped. Everything the warm-up waits on can fail to arrive - no admin
# service, no Chromium installed, a page that never runs - and none of those may leave the panel
# black: past this we light up and carry on without a warm browser, which costs the first tap on
# his eye a browser start and costs the boot nothing.
WARM_UP_S = 25.0
# ...and a beat after the page starts polling, for Chromium's second raise. The poll is the last
# thing the page does - it sits below half a megabyte of JointJS that blocks the parser - so by
# the time it arrives the browser has finished its own startup, and this is only the margin.
BROWSER_SETTLE_S = 2.0
# How long the page gets to lay a drawing out before we uncover it anyway. Generously over the
# ~400 ms poll plus a JointJS layout, because the cost of being wrong is asymmetric: uncovering
# early shows the dashboard for a moment, and never uncovering loses the picture entirely.
PAINT_WAIT_S = 8.0
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
    # What a mapped-but-unpainted window is filled with. Chromium's own default is white, and
    # the page is #000, so without this every browser start is a white flash. It costs nothing
    # at boot, where the panel is dark anyway, and everything on the one start somebody watches:
    # the slow path in _ensure_browser, after the warm one has died.
    "--default-background-color=FF000000",
)


def _phase(what: str) -> None:
    """One timed line per startup phase, timed from the top of this module\'s imports.

    So a slow boot can be read off the kiosk\'s own log.

    The panel is dark for most of what this records and a dark panel explains nothing by itself.
    Without these the only way to find out where a boot went was to reconstruct it from the
    journal, the desktop's stderr and a Chromium log with no timestamps in it.
    """
    print(f"· +{time.monotonic() - _BEGAN:5.1f}s  {what}", flush=True)


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


# Where the browser's real binary lives on a Pi - /usr/bin/chromium is a shell wrapper and only
# a few kB of it. Both names, for the same reason BROWSERS has both.
CHROME_BINARIES = (
    Path("/usr/lib/chromium/chromium"),
    Path("/usr/lib/chromium-browser/chromium-browser"),
)


def preload_browser() -> None:
    """Pull Chromium's binary into the page cache, off-thread, before anything wants it.

    254 MB, which this card reads in 1.2 s - against the seven seconds it adds to a browser
    started cold, every one of which is a second of black panel later on. So it is done here,
    early, where it overlaps the camera opening and costs the boot nothing anybody can see.

    Nothing to fail over: a box with no Chromium, or one that has it somewhere else, simply
    starts its browser cold as before.
    """

    def read() -> None:
        for path in CHROME_BINARIES:
            try:
                with path.open("rb") as binary:
                    while binary.read(4 << 20):
                        pass
            except OSError:
                continue
            return

    threading.Thread(target=read, name="kiosk-preload", daemon=True).start()


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
        # The reading the lamp is derived from, kept now that the panel has a gauge to put it on.
        # One sysfs read still answers both: the lamp wants a step with hysteresis on it and the
        # gauge wants the degrees, and throwing the degrees away was only ever because nothing
        # was showing them.
        self._temp_c: float | None = None
        # The kiosk's own voice in the caption, for the things that happen to it rather than to
        # the session - a shutter that could not take a photo being the one that matters. The
        # controller owns that line the rest of the time and knows nothing about any of this.
        self._notice = ""
        self._notice_until = 0.0
        self._volume: int | None = None  # the level we last put on the sink
        self._volume_at = 0.0  # when we last looked for a new one
        self._turning = False  # a finger is on the knob: the column is up, and follows it
        self._sliding = False  # ...and has reached the track, so the speaker follows it too
        self._slide_from = 0.0  # where that finger landed, which is what the floor is measured on
        # What the column is asking for. Written by the finger, read and answered by _walk, and
        # cleared by _walk on its way out rather than by _let_go - which is what lets the last
        # pass of a gesture see the reading the lift took.
        self._wanted: int | None = None
        self._knob_busy = threading.Event()  # a thread is walking the sink to where the finger is
        # Which screen the page was last asked for. The warm browser holds whatever it was told
        # last, so a reveal only has to wait for a route when it is asking for a different one -
        # which is never the common case, his face being the button that gets pressed.
        self._screen: str | None = None
        # The switch's position as we last read it, so a session in progress is only told when
        # it actually changes. Seeded from the note rather than from nothing: at boot the page
        # and the session already agree, and there is nothing to say.
        self._barge_margin = barge.margin_db(controller.settings)
        self._barge_at = 0.0
        self._say_at = 0.0  # when we last looked for a voice the page wants sounded
        self._browser: subprocess.Popen | None = None  # the admin browser, kept warm from boot
        self._warm_at = time.time()  # when it was launched, so a note older than it means nothing
        self._close_at = 0.0  # when we last looked for a Close from a page nobody here put up
        self._admin_busy = threading.Event()  # set from the tap until the page is done with
        # A second latch rather than reusing _admin_busy, which the chrome reads to decide
        # whether the eye is lit (see _pressed_now). A picture is not that page, and a
        # panel that lights the eye whenever Cyclops draws would be telling the truth about the
        # browser and a lie about what you are looking at. Both still gate _open_admin, so the
        # two can never be up at once.
        self._page_busy = threading.Event()  # any page has the panel: the admin one or a picture
        # ...and a third, narrower than either: a picture of ours is not merely on its way to the
        # panel but actually visible on it. _page_busy goes up when the thread is *spawned*, and
        # _picture_session then waits up to PAINT_WAIT_S for the page to paint and may give up
        # without ever uncovering - so it cannot answer "is something of ours on the glass now?".
        # show_picture needs that exact question: the page polls, so a second picture offered
        # while one is up swaps itself in, and refusing it would tell the model nobody can see a
        # picture that everybody can. See show_picture.
        self._panel_showing = threading.Event()
        self._reveal = threading.Event()  # asks the render loop to uncover the admin page
        self._retake = threading.Event()  # ... and to take the panel back off it
        self._hidden = False  # the admin page has the panel; nothing we draw can be seen
        # What is on that panel while it is not ours, for a session recording the screen to
        # sample - see cyclops.still. None whenever there is nothing to say, which is the
        # dashboard and every failure; the render loop publishes black for it as it always did.
        self._page_still: np.ndarray | None = None
        self._window_up = False  # whether highgui currently has a window for us
        # Whether the panel's light has been turned on for this run. It comes on with the first
        # fully drawn frame rather than at startup, because everything before that - the browser
        # warming up over a panel we have no window on yet - is deliberately not shown.
        self._lit = False
        # Deliberately "now" rather than zero, so the first check happens one interval in - by
        # which time run() has painted a panel-sized frame over open_window's camera-sized one,
        # and the rect being read is one worth reading.
        self._fullscreen_at = time.monotonic()
        self._fullscreen_fixes = 0
        self._size = (0, 0)
        self.screen = screen
        # While this is in the future the ring says a photo did not happen; see _ring_state.
        self._shutter_error_until = 0.0
        # The physical shutter, built here for the same reason Backlight is: it is hardware that
        # may simply not be there, it settles that question itself, and everything downstream
        # can then stop asking. Last of all in this constructor deliberately - it arms a handler
        # on a thread of its own, and a press landing on a half-built kiosk would find half the
        # attributes above missing.
        self.button = ShutterButton(
            pin=controller.settings.button_pin,
            led_pin=controller.settings.button_led_pin,
            # The same threshold his eye is held for, so the two long presses on this box feel
            # like one gesture rather than two conventions. Passed down rather than declared
            # again over there: a second constant is a second thing to keep in step.
            hold_s=LONG_PRESS_S,
            on_tap=self.shutter_pressed,
            on_hold=self.button_held,
        )

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

    def _panel_size(self) -> tuple[int, int]:
        """The size a frame has to be to stand in for the panel, before or after the first one.

        (0, 0) is what ``self._size`` says until the loop has drawn once, and a 0x0 frame is one
        the recorder cannot resize and gives up over - so it is answered with the panel's own
        size rather than passed on. ``all`` and not ``or``: a truthy tuple of zeroes would sail
        straight through.
        """
        return self._size if all(self._size) else NO_CAMERA_SIZE

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
        the busier that is - a megapixel photo rather than a few numbers - the likelier the
        request lands too early. So it is checked rather than assumed. ``getWindowImageRect``
        reports where the last image actually landed, which is exactly the question being asked;
        it is useless for *discovering* the panel size, for the reason :meth:`_window_size`
        gives, and ideal for confirming one we already know.
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

        Everything but his face acts on the press, which is what a screen with no travel should
        do: the pointer belongs under the finger that landed. His face is the exception, because
        it carries two things - a tap for what the box has kept, a hold for the power menu - and
        the release is the only event that can tell them apart. That is the bargain every phone
        makes, and it costs the eye nothing anybody can feel: what the tap does is uncover a
        browser that has been warm since boot.

        The knob is the one control here that also cares what happens *between* the two events.
        Moves were thrown away for the life of this panel, because until there was something to
        turn nothing on it was worth a second event; they are now read while - and only while -
        a finger that landed on the knob is still down, so the level follows it and you hear
        where you are rather than having to look.

        The fourth target is everything that is not the other three, and it only exists while
        INTERRUPT is off: a press on the bare picture stops him talking. It is deliberately the
        fall-through rather than a box of its own, so it can never take a tap away from a
        control - it is only ever reached by one that missed all of them.
        """
        if self.overlay is None:
            return
        if event == cv2.EVENT_MOUSEMOVE:
            # No wake, no touch clock, nothing else: this is a finger already on the glass, and
            # everything that a press means was decided when it landed.
            if self._turning and flags & cv2.EVENT_FLAG_LBUTTON:
                self._slide(y)
            return
        if event == cv2.EVENT_LBUTTONUP:
            self._lifted(x, y)
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        self._touched_at = time.monotonic()
        # Whatever was being held, this is not it any more. A press whose release never arrived
        # would otherwise sit there and turn the next tap into a hold that was already half done -
        # or, for the knob, leave a drag that began on his face turning the volume, because the
        # moves that follow a press do not carry where the press was.
        self._eye_down_at = None
        if self._turning:
            self._let_go()
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
        if boxes.volume.contains(x, y):
            # Nothing is set yet. The disc lights and the column comes up beside it showing
            # where the speaker already is; what the grab turns out to *mean* is decided by
            # whether the finger walks up onto the track - see _slide. Both held rather than
            # flashed: they stay up for as long as the finger is on the glass, and _lifted
            # puts them away.
            self._pressed = VOLUME
            self._press_until = float("inf")
            self._turning = True
            self._slide_from = y
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
        elif boxes.heat.contains(x, y):
            # A gauge is a thing you read, and this one is the corner of a screen with three more
            # numbers on it. The tap opens that screen rather than the recordings his face opens.
            self._press(HEAT)
            self._open_admin(SYSTEM_SCREEN)
        elif self._barge_margin is None:
            # None of the three, which is most of the panel: "stop, my turn". With INTERRUPT off
            # the mic is held shut for as long as he is audible, so a finger on the picture is
            # the only way back into a sentence you have heard enough of. Gated on the switch
            # because with INTERRUPT on you simply talk over him, and a second answer to a
            # settled question is one more rule to carry. Nothing sounds and nothing lights:
            # what this does is make the room quiet, which no cue could say more plainly.
            self.controller.interrupt()

    def _lifted(self, x: int, y: int) -> None:
        """A finger coming off the glass. His face, and the knob, have something left to do here.

        For the knob this is the end of the gesture rather than the answer to it: the level has
        been following the finger the whole way up the track. One last reading, in case highgui
        never sent the move that got there, and then the column comes down.
        """
        if self._turning:
            self._slide(y)  # wherever it ended up, including a last move highgui never sent
            self._let_go()
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
        """Act on a tap while the menu is up. Its rows act on the press, as the switches do.

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
        """Which control to draw as held - the knob, the gauge, his face, or a row of the menu.

        He stays lit while his page is up, and for as long as a finger is on him: a hold
        that is going somewhere should look held for all of the second it takes, not for the
        180 ms a tap gets.

        Uncovering a warm browser is immediate, so the page half of this is normally seen for a
        frame or two. It still earns its place on the one tap that has to start a browser:
        without it he goes dark 180 ms in and a panel that is busy looks like a panel that
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

    def shutter_pressed(self) -> None:
        """A *tap* on the button beside the panel: the aperture's path, coordinates taken out.

        On the release rather than the press, because a tap and a hold are the same event until
        somebody lets go and the hold now means something (see :meth:`button_held`). The photo
        is therefore late by however long you lean on the button, which is the bargain his eye
        has always made for the same reason.

        Neither this nor the hold arrives on the render thread - highgui dispatches taps from
        inside ``waitKey``, gpiozero has threads of its own. Everything they touch is either
        lock-guarded (:meth:`cyclops.camera.CameraSource.start`) or a plain attribute the loop
        only reads, and :meth:`_snap` already hands the work to a thread anyway.

        A tap while the panel is dark is spent waking it, exactly as a tap on the glass is,
        though for the opposite reason: you *can* find this button in the dark. Sleeping
        released the camera, so the photo it took would be of nothing.
        """
        self._touched_at = time.monotonic()
        if self._asleep:
            self._wake()
            return
        if self._menu:
            return  # modal, and two of its three rows end the box: this is no answer to it
        self._snap()

    def button_held(self) -> None:
        """The button held down: the microphone switch, without having to find the glass.

        The switch that starts and ends a conversation is a 12 mm target you have to look at,
        and the moment you want to start talking to him is the moment both hands are full. This
        is the same :meth:`_toggle_session` the switch itself calls - the record source, the
        printed line, the closing cue and the optimistic ``_pending`` all come with it - so
        there is no second way for a session to begin.

        Unlike the tap, a hold on a dark panel is not spent waking it: the tap is ambiguous
        under a black screen and this cannot be, so it lands. It has to light the glass itself,
        though. :meth:`_sleeping` only keeps the idle clock reset while a session is up; it
        never clears :attr:`_asleep`, so a session started on a dark panel would otherwise run
        its whole length behind one.

        No cue of its own, and no ring state of its own, and nothing on the glass to invert:
        the two switches that used to answer this button are a knob and a gauge now. ``_pending``
        makes :meth:`_effective` report STARTING at once, which is a session as far as
        :meth:`_ring_state` is concerned, so the ring lights as the hold lands - and after it the
        connecting ping and the closing pair say the rest, which is the half of this a head under
        a bench can hear.
        """
        self._touched_at = time.monotonic()
        if self._menu:
            return  # modal, exactly as it is for the tap above
        if self._asleep:
            self._wake()
        self._toggle_session()

    def _ring_state(self, state: str) -> str:
        """What the ring in the button should be saying: idle, listening, or a photo that failed.

        Tied to what the box is doing rather than to the finger on it, which the press already
        answers for itself with a flash and a click. The failure wins for as long as the caption
        it matches, because it is the half of that message a head under a bench can see.
        """
        if time.monotonic() < self._shutter_error_until:
            return RING_ERROR
        return RING_ACTIVE if session_up(state) else RING_IDLE

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
            # ...and on the ring for those same four seconds, which is the half of this you can
            # read without looking up at the panel. See _ring_state.
            self._shutter_error_until = time.monotonic() + NOTICE_S
            print(f"· snapshot failed: {exc}", file=sys.stderr, flush=True)
        else:
            # Up on the glass first, then off to Cyclops. The panel is a file write away and the
            # model is a network round trip, and the picture belongs on the screen either way:
            # with no session running there is nobody to show it to and it should still go up.
            # The same two lines every other picture reaches the panel through.
            panel = self._show_snapped(shot.path)
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
                # Whether it reached the *panel*, which is the other question - `shown` answers
                # whether Cyclops was shown it. The edit record already splits the two this way.
                panel=panel,
            )
            print(f"· snapped {shot.path}{'' if shown else ' (nothing live to show it to)'}"
                  f"{'' if panel else ' (no panel free)'}", flush=True)
        finally:
            self._snap_busy.clear()

    def _show_snapped(self, path: Path) -> bool:
        """Put the photo just taken on the panel. False if there is no panel free for it.

        The one picture somebody was deliberate about used to be the one they could not see: the
        shutter flashed, the photo went to Cyclops, and the panel carried on showing the live
        camera. It now goes up exactly the way a recalled picture does, through the same two
        lines and the same offer file - which is also what makes it editable, because
        ``edit_photo`` works on whatever is on the panel.

        ``for_panel`` is close to a no-op here, ``webcam._save`` having already capped the long
        edge at the same 1024, but it is what the other callers use and it is what keeps this
        honest whatever the camera hands over. Read from the card rather than reusing
        ``Capture.data_url``: that one is the model's copy, and ``offer_image`` wants raw bytes
        and builds its own.

        Never raises. A photo that will not go on the panel is still on the card and still went
        to Cyclops, and the shutter is not the place to find that out the hard way.
        """
        try:
            small = imagine.for_panel(path.read_bytes())
        except OSError as exc:
            print(f"· could not read {path} for the panel ({exc})", file=sys.stderr, flush=True)
            return False
        return bool(panel.offer_image(small, path.stem) and panel.show())

    # ---- admin page ----

    def _admin_url(self) -> str:
        """The page as the browser is *started* on it: the sessions list, not the four numbers.

        Tapping his face asks what he remembers, so it had better land on some. The hash is only
        read once, when Chromium is launched - it never navigates afterwards - so anything that
        wants a different screen out of the same warm browser asks for it through
        :meth:`_ask_for_screen` instead of through here.
        """
        return f"http://127.0.0.1:{self.controller.settings.admin_port}/#{SESSIONS_SCREEN}"

    def _ask_for_screen(self, screen: str) -> None:
        """Leave word for the page to route itself to *screen*, and give it time to.

        The panel has two things on it that uncover one browser and they promise different
        screens - his face the recordings, the gauge the numbers behind itself - and the browser
        is warm precisely because nobody ever navigates it. So the page is told, on the poll it
        already runs two and a half times a second, and routes itself before it is uncovered.

        The wait is skipped whenever the page is already holding the screen being asked for,
        which is nearly always: his face is the button that gets pressed, and it asks for the
        screen Chromium was launched on. Only the odd tap that changes screens pays for a poll,
        and paying for it here is the whole point - uncovering first and letting the page swap
        underneath somebody is exactly the flicker :meth:`_picture_session` waits to avoid.
        """
        if screen == self._screen:
            return
        try:
            PAGE_SCREEN_FILE.parent.mkdir(parents=True, exist_ok=True)
            PAGE_SCREEN_FILE.write_text(screen, encoding="utf-8")
        except OSError as exc:  # the page stays where it is; a wrong screen beats no page
            print(f"· could not ask the page for {screen} ({exc})", file=sys.stderr, flush=True)
            return
        self._screen = screen
        time.sleep(PAGE_ROUTE_S)

    def warm_browser(self) -> bool:
        """Get the admin browser up and painting, before this kiosk has a window at all.

        Blocking, and called with the panel's light off - see :func:`main`. Everything ugly about
        a browser starting happens in here: the white fill of a mapped window with nothing in it
        yet, the dashboard painting, and Chromium raising its own window a second time a few
        seconds later. None of it is seen, and by the time we return it is over, so the window
        opened after this maps last and stays on top for the life of the kiosk. That is what
        replaced a schedule of six blind window rebuilds.

        The browser then sits behind our window with the page loaded and polling, which is what
        makes his eye instant. It costs a couple of hundred MB of a box that has 8 GB, and the
        alternative is a button that answers in seconds.

        False if the panel is about to come up without a warm browser, which is not a failure
        worth stopping for: the eye then starts one on the tap, exactly as it does when the warm
        one has died. Everything in here is bounded by one deadline, because the cost of waiting
        is a black panel and nobody watching a black panel knows what it is waiting for.
        """
        deadline = time.monotonic() + WARM_UP_S
        url = self._admin_url()
        while not _admin_reachable(url):
            if not self.running or time.monotonic() >= deadline:
                print(
                    f"· admin service never answered on {url}; his eye will start its own"
                    " browser",
                    file=sys.stderr,
                    flush=True,
                )
                return False
            time.sleep(ADMIN_RETRY_S)
        _phase("admin service answering; starting the browser")
        # Cleared before the spawn, so what comes back can only have been written by the browser
        # started below - a note from a previous run would otherwise answer for this one instantly.
        PAGE_ALIVE_FLAG.parent.mkdir(parents=True, exist_ok=True)
        PAGE_ALIVE_FLAG.unlink(missing_ok=True)
        asked_at = time.time()
        if not self._start_browser(url, timeout=max(0.0, deadline - time.monotonic())):
            return False
        _phase("browser has the page; waiting for it to run")
        if not _wait_for_flag(PAGE_ALIVE_FLAG, asked_at, max(0.0, deadline - time.monotonic())):
            # Served but never polled. A page that is merely slow is still worth being behind us,
            # so this is a note rather than a refusal - and the deadline has already spent
            # whatever patience the panel could afford.
            print(
                "· the admin page never started polling; covering it anyway",
                file=sys.stderr,
                flush=True,
            )
        time.sleep(max(0.0, min(BROWSER_SETTLE_S, deadline - time.monotonic())))
        return True

    def _start_browser(self, url: str, timeout: float = PAGE_WAIT_S) -> bool:
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
        if not _wait_for_page(proc, launched_at, timeout):
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
        # A browser that has just been launched is holding the hash it was launched with, whatever
        # the last one had been asked for. Said here rather than by the caller because this is the
        # only place a URL is ever handed to Chromium.
        self._screen = SESSIONS_SCREEN
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

    def _open_admin(self, screen: str = SESSIONS_SCREEN) -> None:
        """Uncover the admin page on *screen*, off-thread - highgui owns the main one.

        Same shape as :meth:`_snap`. The busy flag means a second tap while the page is up is
        ignored rather than starting a second browser: sharing a profile directory, the second
        invocation would hand its URL to the first and exit at once, leaving us holding a dead
        pid and a fullscreen window nothing can close.
        """
        if self._page_busy.is_set():
            return
        self._page_busy.set()
        self._admin_busy.set()
        threading.Thread(target=self._admin_session, args=(screen,),
                         name="kiosk-admin", daemon=True).start()

    def _admin_session(self, screen: str = SESSIONS_SCREEN) -> None:
        """The whole life of the visible page: probe, route, uncover, wait, take the panel back."""
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
            # was never shown - a crashed session, a picture whose show() lost the race with
            # this tap - would otherwise be what the tap uncovers. The two can never legitimately
            # be up at once: _page_busy gates both.
            panel.withdraw()
            BROWSER_CLOSE_FLAG.parent.mkdir(parents=True, exist_ok=True)
            BROWSER_CLOSE_FLAG.unlink(missing_ok=True)  # a stale note must not close this one
            self._ask_for_screen(screen)
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
            # The note is for the half-second either side of a reveal and no longer: left lying
            # about, it would be a stat with an answer in it on every poll for the rest of the day,
            # and the page would refuse to route the next time it was asked for the same screen.
            PAGE_SCREEN_FILE.unlink(missing_ok=True)
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

    # ---- pictures on the panel ----

    def show_picture(self) -> bool:
        """Put the picture waiting in ``PANEL_FILE`` on the panel. False if the panel is busy.

        Called from the agent's thread, so it does nothing here but set a flag and start a
        thread: highgui belongs to the render loop and this is not it.

        A picture of ours already being up is not busy - it is the ordinary case. Snapping a
        photo puts one on the panel, and the edit asked for a minute later arrives while it is
        still there. The page polls ``/api/panel`` several times a second and repaints on any new
        payload id, and the caller has already written one, so the swap has effectively happened
        by the time this returns: saying False would have the model announce out loud that nobody
        can see a picture that is about to be on the glass in 400 ms. Only the admin page really
        has no room, because that is a different page and not ours to paint over.
        """
        if self._panel_showing.is_set():
            if panel.announces():
                self._cues.play("shown")  # a drawing arriving is news; see _picture_session
            print("· picture swapped on the panel", flush=True)
            # A recording is being handed the picture on the panel rather than the black the
            # kiosk is painting behind the browser (see cyclops.still), and that hand-off happens
            # once, at the reveal. A swap never goes past the reveal, so without this the video
            # would hold the first picture for as long as the panel kept showing others. Off on
            # a thread of its own because rasterising a drawing shells out to ffmpeg, and this
            # method is called from the agent's and promises to do nothing slow.
            threading.Thread(target=self._restill, name="kiosk-restill", daemon=True).start()
            return True
        if self._page_busy.is_set():
            return False  # the admin page is up, or a picture is on its way; do not stack them
        self._page_busy.set()
        threading.Thread(target=self._picture_session, name="kiosk-picture", daemon=True).start()
        return True

    def _restill(self) -> None:
        """Rebuild the panel's stand-in frame after a picture was swapped for another.

        Reads the offer file, which the caller has already written, so it does not have to wait
        for the page to repaint to know what the page is about to show.

        Checks the latch again on the way out: a picture that came down while this was rendering
        would otherwise republish itself over the black that replaced it.

        Nothing to rebuild means black, and not the frame we already had. ``still.of_panel``
        answers None for whatever it cannot turn back into pixels - today that is a scratchpad of
        HTML, which nothing on this side of the glass can rasterise - and keeping the last frame
        there would put the *previous* picture in the recording for as long as the new thing is
        up. That is a worse answer than black for the reason the reveal already gives: a frozen
        picture over a running timer watches back as a hung encoder rather than as what happened.
        """
        width, height = self._panel_size()
        frame = still.of_panel(width, height)
        if frame is None:
            frame = _black(width, height)
        if self._panel_showing.is_set():
            self._page_still = frame
            self.panel.publish(frame)

    def _picture_session(self) -> None:
        """The whole life of one picture: wait for the page to paint it, show it, then take it
        back.

        The same shape as :meth:`_admin_session` and for the same reasons, with one difference:
        the admin page is already loaded in the warm browser, and a picture is not. So this waits
        for the page to say it has actually painted before uncovering, rather than uncovering onto
        a dashboard that turns into a picture half a second later while somebody is watching.
        """
        url = self._admin_url()
        shown = False
        try:
            if not _admin_reachable(url) or not self._ensure_browser(url):
                print("· panel: no page to paint it on", file=sys.stderr, flush=True)
                return
            PANEL_PAINTED_FLAG.parent.mkdir(parents=True, exist_ok=True)
            PANEL_PAINTED_FLAG.unlink(missing_ok=True)  # a stale note must not answer for this one
            BROWSER_CLOSE_FLAG.unlink(missing_ok=True)
            asked_at = time.time()
            if not _wait_for_flag(PANEL_PAINTED_FLAG, asked_at, PAINT_WAIT_S):
                # Uncover anyway. The page polls, so it is probably a slow layout rather than a
                # dead browser, and a picture arriving a moment late beats one that never comes.
                print("· panel: the page was slow to paint; showing anyway", flush=True)
            # Before the reveal and on this thread, not the render loop's: rasterising a drawing
            # costs a third of a second on this box, which is a frame the panel would drop and a
            # third of a second nobody notices at the end of the ten this drawing already took.
            # By now the page has posted what it painted, which is the whole reason we waited.
            self._page_still = still.of_panel(*self._panel_size())
            shown_at = time.time()
            shown = True
            self._panel_showing.set()  # from here a second picture swaps rather than being refused
            self._reveal.set()
            # A drawing was made and it is on the panel now: look up. Where one that takes an
            # empty panel says so - a picture drawn or found arrives here through panel.show(),
            # and one replacing another says it in show_picture instead, which is the path that
            # does not come back through here. Pointedly not _admin_session's reveal, which is
            # his eye opening the dashboard and already has a sound of its own.
            #
            # A photograph says nothing. Every snap already sounds the shutter a moment before
            # this, and a second cue on top of it is one too many - which is only obvious now
            # that the shutter puts its photo on the panel rather than only handing it over.
            # A drawing does sound, because nothing else announced it and it took a minute.
            if panel.announces():
                self._cues.play("shown")
            print("· picture on the panel", flush=True)
            self._watch_page(shown_at)
        finally:
            # Cleared before the withdraw, so nothing can offer a picture into the gap between
            # this one coming down and the panel coming back.
            self._panel_showing.clear()
            # The drawing goes before the panel comes back, so the page has already switched
            # itself off the picture by the time it is visible again behind the window.
            panel.withdraw()
            if shown:
                self._retake.set()
                print("· picture closed", flush=True)
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
        # The same question the halo, the strip and the ring in the button all answer, so what
        # the panel shows and what the button does cannot drift. There is nothing on the glass
        # left saying which way this goes: the switch that used to went the way the words that
        # used to say it went before it.
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

        The warm browser is meant to sit behind the panel, and :meth:`warm_browser` makes that
        as certain as it can be by finishing before our window exists at all. It is still not a
        fact that can be read back: if Chromium ever raises itself again afterwards, the panel is
        covered by a page no thread of ours is watching - :meth:`_watch_page` only runs for a
        page this kiosk put up - so every tap goes to the dashboard and its Close button writes a
        note nobody reads. That is the state this exists for, and it is now the only recovery
        from it: honour the note here too, and the one gesture somebody standing at the panel
        would try is also the one that works.
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

    def _slide(self, y: int) -> None:
        """Follow a finger that grabbed the knob, and set a thread walking the speaker after it.

        The drag begins when the finger has gone somewhere *and* has arrived on the track - both
        floors, for the two reasons in :data:`SLIDE_GRAB_PX`. Until then a grab on the knob is
        free: the disc lights, the column is up showing where the speaker already is, and nothing
        has been asked of anything. Past the floors the reading is where the finger is on the
        track, snapped to the same 5 the page's slider steps in so the two controls cannot
        disagree about what a level is.

        Nothing is applied *here* - this runs on the render thread, out of highgui's callback,
        and pactl is a subprocess. What this sets is the number the column is showing and the
        number :meth:`_walk` is chasing.

        Nothing at all with no mixer under us - a Mac, or a Pi with no sink - because a column
        that fills and changes nothing is worse than one that never comes up.
        """
        if self.overlay is None or self._volume is None:
            return
        if not self._sliding:
            if abs(y - self._slide_from) < SLIDE_GRAB_PX or y > self.overlay.slider.bottom:
                return
            self._sliding = True
        value = self.overlay.slider_value(y) * 100
        self._wanted = max(0, min(100, round(value / VOLUME_STEP) * VOLUME_STEP))
        # Spawned on the first rung rather than on the press, which is what keeps a tap on the
        # knob costing nothing at all. A second grab arriving within a rung of the last one can
        # find the latch still up under a thread on its way out; that drag is silent until its
        # next move event, sixteen milliseconds it is not worth a handshake to save.
        if not self._knob_busy.is_set():
            self._knob_busy.set()
            threading.Thread(target=self._walk, name="kiosk-knob", daemon=True).start()

    def _walk(self) -> None:
        """Walk the speaker up the ladder after the finger, one rung at a time, until it lifts.

        Off the render thread because both halves of a rung are slow there: pactl is a subprocess
        and ``sd.play`` opens a PortAudio stream, and twenty a second of either through highgui's
        callback would be a preview that stutters whenever you touch the volume.

        It only ever looks at where the finger is *now*. Everything it crossed on the way is
        never asked for, so a flick up the track is *cheaper* than a slow deliberate slide rather
        than more expensive - which is the right way round, and is the whole defence of doing
        this at all on a board that heats in its case.

        The read order is load-bearing. ``_turning`` is taken before the level, and
        :meth:`_let_go` clears it last and never touches ``_wanted``, so a pass that finds the
        gesture over is guaranteed to be looking at the reading the lift took. Taken the other
        way round it drops the last rung of every drag. That rests on the GIL making these stores
        visible in order, which is true of CPython and is a better bargain than a lock the render
        thread would have to take on every mouse event.
        """
        landed: int | None = None
        try:
            while True:
                turning = self._turning  # first: see the note above
                if self._rung():
                    landed = self._volume
                if not turning:
                    return
                time.sleep(SLIDE_RUNG_S)
        finally:
            # Inside the latch, so the poll that reads this note can never see it before the sink
            # it describes - see _sync_volume. One line per gesture rather than per rung, which
            # is also the quick check from the log that this thread is exiting cleanly.
            self._wanted = None
            if landed is not None:
                mixer.request(landed)
                print(f"· volume {landed}% from the panel", flush=True)
            self._knob_busy.clear()

    def _rung(self) -> bool:
        """One rung: put the speaker where the column is, and click. False if it was already there.

        That short circuit is what keeps a finger resting on one rung silent rather than a
        machine gun, and it is why the walk can run on a clock instead of waiting on an event.

        The click sounds *after* the level lands, and at a fixed amplitude, because the sink it
        is going out through is the very thing being set: a rung near the foot of the ladder is
        meant to be faint. See the cue in :mod:`cyclops.sfx`.
        """
        wanted = self._wanted
        if wanted is None or wanted == self._volume or not mixer.set_level(wanted):
            return False
        self._volume = wanted
        self._cues.play("rung")
        return True

    def _let_go(self) -> None:
        """The finger is off the knob. Put the column away and let the walk finish.

        Nothing is set here, and nothing is taken back. :meth:`_walk` is the only thing that
        writes the sink for the length of a gesture, which is what keeps two pactls from ever
        being in flight at once - the loser of that race is the one that started first and lands
        last, and what it leaves behind is a speaker at one level and a panel certain of another.

        This is also why there is no longer anything to undo when the release was lost and a
        fresh press arrives instead. The levels were set rung by rung, out loud, as the finger
        crossed them; putting the volume back after you heard it climb to a hundred would be the
        surprise. What a lost release costs you is the column coming down, and nothing else.
        """
        self._sliding = False
        if self._pressed == VOLUME:
            self._pressed, self._press_until = None, 0.0
        self._turning = False  # last, and _walk reads it first: see the note there

    def _sync_volume(self) -> None:
        """Follow the level the page left for us. A few bytes, a couple of times a second.

        Not while somebody is turning the knob. The note is the page's opinion and it is stale
        for as long as a finger is setting the sink by hand, so without this every drag would be
        fought back down four hundred milliseconds at a time. The latch rather than ``_turning``
        because :meth:`_walk` outlives the lift by up to a rung and writes its note before
        clearing the latch - so the first poll after any gesture, lift or abort, finds the note
        and the sink already agreeing. Checked before the clock, so that poll is immediate.
        """
        if self._knob_busy.is_set():
            return
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

    # ---- the voice ----

    def _sync_voice(self) -> None:
        """Sound the voice the settings screen is asking to hear.

        The other three notes are settings, read over and over; this one is a request, taken and
        thrown away. It is here rather than on the page because the page is a browser and this is
        the only process on the box with a /dev/snd - the same reason the volume cannot set
        itself (see cyclops.mixer).

        Which voice a session actually opens with is not decided here at all: it is the ordinary
        note beside this one, read by cyclops.ui when the session opens. This is only the sound.
        """
        now = time.monotonic()
        if now - self._say_at < VOLUME_POLL_S:
            return
        self._say_at = now
        try:
            wanted = SAY_VOICE_FILE.read_text().strip().lower()
        except OSError:
            return  # no request, which is almost always the answer
        # Taken before it is played, and unlinked either way: a name that is not one of the ten
        # left lying here would be asked for again every half second for the rest of the day.
        SAY_VOICE_FILE.unlink(missing_ok=True)
        if wanted in voice.NAMES:
            self._cues.play(voice.cue(wanted))

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

        This wakes the *panel*; a hold on the button wakes Cyclops. The two senses never contradict
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
        # Come up at the size the panel actually is. Sized from the camera's own frame, this
        # opened an 800x450 window on an 800x480 panel and _keep_fullscreen corrected it a beat
        # later, which is a visible jump - and one nobody could explain, because the window was
        # never *wrong*, it was only briefly honest about a 16:9 camera.
        if self.fullscreen and self.screen is not None:
            width, height = self.screen
            first = fit_to_window(first, width, height)
        else:
            h, w = first.shape[:2]
            width, height = min(w, 1280), min(h, 720)
        self.open_window(first, width, height)

        while self.running:
            started = time.monotonic()
            # highgui is main-thread only, so the admin thread asks for both of these rather
            # than touching the window itself.
            if self._reveal.is_set():
                self._reveal.clear()
                self._drop_window()  # the warm browser has been behind us all along
                self._hidden = True
                # A session recording the screen is still sampling, and the screen is no longer
                # ours to hand it. What is on it, when the page was handed a picture and we could
                # rebuild it (see cyclops.still), and black otherwise. Black rather than the frame
                # we happened to stop on: a picture can hold the panel for a quarter of an hour
                # mid-session, and a frozen halo over a running timer watches back as a hung
                # encoder rather than as what happened.
                self.panel.publish(
                    self._page_still
                    if self._page_still is not None
                    else _black(*self._panel_size())
                )
            if self._retake.is_set():
                self._retake.clear()
                self._drop_window()  # ... and the next frame builds a window on top of it again
                self._hidden = False
                self._page_still = None  # the panel is ours again; it speaks for itself
                self._touched_at = time.monotonic()  # closing the page is a touch like any other
            self._sync_volume()  # the page sets the volume, so keep reading it while it is up
            self._sync_barge_in()  # ...and whether it may be interrupted, on the same beat
            self._sync_voice()  # ...and plays a voice the settings screen asks to hear
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

            # What the ring is saying. Handed over every frame because what it reflects is a
            # state rather than an event; it only reaches the pin when the answer changes.
            self.button.show(self._ring_state(state))

            # Two sysfs reads, five seconds apart, on a board that takes minutes to change
            # temperature. Cheap enough to do while the panel is dark, which is worth it: the
            # lamp is then already right on the first frame after a tap rather than five
            # seconds into it.
            if started - self._heat_at >= TEMP_POLL_S:
                self._heat_at = started
                self._temp_c = stats.cpu_temp_c()
                self._heat = stats.heat_alarm(self._temp_c, self._heat)

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
                    canvas = fit_to_window(frame, width, height)
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
                    # The two instruments in the corner. Both may be None - no sink, no thermal
                    # zone - and both draw that as a dial that is not reading.
                    #
                    # The knob's two flags, doing their two different jobs. `_turning` is a
                    # finger on it, and puts the column up the instant you touch it, showing
                    # where the speaker actually is. `_sliding` is that finger having reached the
                    # track, which is when the reading becomes the one under it - knob and column
                    # together, because the two are one control and a pointer left behind on the
                    # old level says they are not.
                    volume=self._wanted if self._sliding else self._volume,
                    temp_c=self._temp_c,
                    turning=self._turning,
                )
                self._paint(composite(canvas, chrome), width, height)

            # The panel is showing what it will go on showing - picture, chrome and all - so
            # light it and say so. This is the moment "ready" is about, and it is deliberately
            # not open_window's: that one is called again for every retake of the panel from a
            # page, and it puts a bare frame up a beat before the brackets are drawn over it.
            # Everything that used to flicker after this point happened before it instead, in
            # warm_browser, with the light off. The other half of the pair - "booted", for Linux
            # itself - is sounded much earlier and from another process (cyclops.boot).
            if not self._lit:
                self._lit = True
                self.backlight.on()
                self._cues.play("started")
                _phase("panel up")

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

    _phase("settings loaded; opening the camera")
    preload_browser()  # 254 MB of Chromium, read while the camera is opening; see its docstring
    camera = CameraSource(settings.camera_index)
    camera.start()  # returns at once; the device may only be plugged in a minute from now
    try:
        camera.wait_for_frame()
    except WebcamError as exc:
        # A missing camera is a degraded panel, not a dead one. Everything else still works -
        # The button starts a session, the eye opens the admin page, the knob and the gauge
        # read - and a Pi showing nothing at all reads as broken hardware, which sends
        # someone looking for a keyboard. Say so on the screen and carry on looking.
        print(f"· no camera yet: {exc}", file=sys.stderr, flush=True)

    _phase("camera settled" if camera.connected else "camera absent; carrying on")
    webcam.set_live_source(camera)  # the shutter shoots from this same camera, always
    # Whichever of the two the tap picks, the recorder gets a source even when nothing is plugged
    # in: it waits its own moment for a first frame and says so in the session log if none comes,
    # and a camera present by the time the next session starts is then recorded without anything
    # being rewired. The camera is the constructor's answer only so a session started by anything
    # but _toggle_session still has one; _toggle_session always says which of the two it wants.
    controller = SessionController(settings, frames=camera, entrypoint="kiosk")
    kiosk = Kiosk(controller, camera, fullscreen, screen)
    panel.set_kiosk(kiosk)  # so a finished picture can find a panel to appear on
    # Nothing can be waiting for a panel that has only just come up, so anything here is a
    # leftover from a process that is gone - and the browser warmed below would paint it.
    panel.withdraw()
    # Same argument, and the one that matters more: _sync_stranded reads this note every 200 ms
    # and answers it by taking the panel back. Left behind by a kiosk that was killed with the
    # page up, it would fire a window rebuild seconds into a boot that has nothing to take back.
    BROWSER_CLOSE_FLAG.parent.mkdir(parents=True, exist_ok=True)
    BROWSER_CLOSE_FLAG.unlink(missing_ok=True)
    kiosk.adopt_volume()
    # SIGTERM (start_kiosk.sh's pkill, systemd) otherwise skips the finally below and would
    # leave a panel that looks like a dead Pi. Exit properly instead, and the light comes back.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    found = f"on camera {camera.index}" if camera.connected else "still looking for a camera"
    # Only when there is one. A line about a button nobody wired is a line that sends somebody
    # looking for it - the same argument the idle note below makes about blanking that is off.
    button_note = (
        "  the button beside the panel is both switches: tap it for a photo,\n"
        f"  hold it {LONG_PRESS_S:g}s to wake him or put him back to sleep\n"
        if kiosk.button.available
        else ""
    )
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
        f" · button: {kiosk.button.note}"
        f" · volume: {'—' if kiosk.volume is None else f'{kiosk.volume}%'}\n"
        "  the two bottom corners: his eye on the left - tap it for the recordings and\n"
        "  pictures · on the right, a knob for the volume - drag it and a column comes up -\n"
        "  and a gauge reading the board, which opens the rest of the numbers when tapped\n"
        f"  hold his eye for {LONG_PRESS_S:g}s to shut the box down or restart it\n"
        f"{button_note}"
        f"{idle_note}"
        "  q or ESC to quit · f toggles fullscreen",
        flush=True,
    )
    try:
        # The curtain, and the only reason any of this is invisible. Between here and the first
        # drawn frame the panel is dark, and behind it Chromium starts, paints white, loads the
        # dashboard and raises itself as many times as it likes. Inside the try, because the
        # finally below is what guarantees the light comes back from anything that goes wrong in
        # here - a panel left dark is indistinguishable from a dead Pi.
        kiosk.backlight.off()
        _phase("panel dark; warming the browser behind it")
        kiosk.warm_browser()
        _phase("browser warm; taking the panel")
        kiosk.run()
    except KeyboardInterrupt:
        print("\n· bye", flush=True)
    finally:
        kiosk.close_browser()
        kiosk.button.close()  # the ring out while we still own the pin, then hand it back
        kiosk.backlight.on()  # never leave the panel dark behind us
        panel.set_kiosk(None)
        panel.withdraw()  # nothing should be waiting for a panel that is gone
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
