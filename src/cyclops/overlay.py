"""The kiosk's chrome: a Pip-Boy-style terminal bezel drawn over the live camera frame.

OpenCV can only draw Hershey stroke fonts, which look like a 1980s oscilloscope. Everything
here is rendered into one RGBA layer with PIL instead - real TrueType text, anti-aliased
shapes, a numpy-built halo and scanline field - and handed to :mod:`cyclops.kiosk` as a numpy
array to alpha-blend onto the frame. Geometry doubles as the hit-test map: every interactive
element returns its rectangle, so a tap can be resolved without a second layout.

The layout is a terminal screen: a rounded border hard against the panel's edges, a readout
strip along the top, three tabs along the bottom with Cyclops himself standing in the middle of
them, and the camera behind all of it. The picture itself is left alone - no wash, no
scanlines, nothing between you and the lens - because the panel's job is to let you see the room
and the endoscope has no detail to spare. The filter is
what the strip and the tab row are made of instead: they used to be opaque bars, and are now
that same phosphor wash, corner shading and scanline field laid over the live picture, so the
camera shows through the chrome as well as between it. The border carries the session state in
its colour and glows inwards from it, which is the one thing that has to be readable across a
workshop without reading any words - and it says so in *hue* rather than in brightness, because
dim green and bright green are the same colour to anyone more than a pace away.

His eye is the one thing here that is a face rather than a readout, and it is what lets the rest
stay this terse: a glance at the middle of the row answers "is he there, and what is he up to",
so the strip is left free to spell it out only for whoever is close enough to read it. He is the
boot mark brought to life - :mod:`cyclops.eye` draws the splash's iris-inside-HUD-rings with the
rings turning and the iris breathing, and :data:`MOODS` says how, one row per state. Tapping him
opens what the box has kept, because what you ask a face is what it remembers.

The row's top rule runs in from both sides, lifts over the top of his head and comes down the
other side. That shoulder is the join: it is what makes him part of the bar rather than a badge
sitting on it. And nothing whatsoever moves while he is asleep, which is what makes any of the
rest of it read as awake. The single exception is the WAKE UP cell, which breathes: it is the
only control left to press, and a control nobody finds is worse than one that beckons.

Everything that holds still while the state does - the halo, the scanlines, the vignette, the
frame and its glow, the mode word, the tab row and its two glyphs, the shoulder - is built once
and cached, keyed on the state. A Pi rendering this at 25 fps has 40 ms for the whole loop and
the camera wants most of them; what is left for a frame here is a signal meter, a clock, a
caption, one ring, the border line and the eye. Every part of the eye moves, so none of it is
cached at all; measured, it is a tenth of a millisecond.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .eye import EyeEngine, Mood, breath, mix

IDLE, CONNECTING, LISTENING, SPEAKING, LOOKING, SEARCHING, DRAWING, ERROR = (
    "idle",
    "connecting",
    "listening",
    "speaking",
    "looking",
    "searching",
    "drawing",
    "error",
)
# Optimistic states the kiosk shows the instant you tap, before the session agrees. Tearing a
# session down takes ~2.3 s, during which the controller still honestly reports "listening" -
# so without these the button looks dead for over two seconds after you press stop.
STARTING, STOPPING = "starting", "stopping"


def session_up(state: str) -> bool:
    """Is there a session at all - from the tap that starts one to the last file it writes?

    The panel's coarsest question, and the one the *words* answer to: the tab says SLEEP rather
    than WAKE UP, the caption breathes, the border breathes. The kiosk asks it too, so that what
    the button says and what the button does can never drift apart - see ``_toggle_session``.
    """
    return state not in (IDLE, ERROR)


def awake(state: str) -> bool:
    """Is Cyclops himself up - eye open, listening, able to answer?

    Stricter than :func:`session_up` by exactly the teardown, which can run for a minute while
    the video is muxed. Both are right about their own half: the eye shuts on the tap because
    that is what you asked for, and the cell stays lit with the caption saying "saving the
    video…" because that is what is still true.
    """
    return state not in (IDLE, ERROR, STOPPING)


# Phosphor palette. The chrome is one hue - the brand, the rules, the ticks, the two tabs that do
# not change, the words of the caption - because a panel where everything is a colour has none.
# What steps off it is the handful of things that only mean anything while a session is up, and
# they all wear the state's own accent. Asleep, none of them do, and the whole screen is green.
GREEN = (86, 255, 140)  # phosphor at full brightness: text, live chrome
GREEN_MID = (46, 176, 100)  # rules, dividers, glyphs at rest
GREEN_DIM = (32, 118, 70)  # the faintest thing still legible on the panel
AMBER = (255, 184, 60)
RED = (255, 86, 70)
# The one the tube gets while he is up. Asleep, the panel is a single hue and he is the same
# green as the furniture he sits in; awake, the live parts step off it. That is the whole job of
# it: dim green and bright green are the same colour to anyone more than a pace away, so a panel
# that said "awake" only by getting brighter did not say it at all.
#
# Thirty degrees round the wheel from the phosphor and no further. A blue-cyan at +50 was tried
# and fought the green - two colours near enough to be compared and far enough to argue. This one
# is plainly not the phosphor and still belongs to the same tube. Every state he is up in wears
# it, searching included: what tells those apart is the eye, which by then is spinning at 155
# degrees a second with a white-hot arc going round its rim, and the word in the strip.
AQUA = (80, 245, 215)
SCREEN = (5, 15, 10)  # the green-black the readout strip and the tab row are made of
INK = (3, 11, 7)  # text on a filled tab

# The halo answers one question only from across a room - is the agent up? - and the state's
# accent answers it in colour rather than in brightness, which is the only half of it that
# survives the distance. Everything wearing this moves together: the border and its inward glow,
# the mode word, the signal meter, the session clock, the caption's marker and the lit edge of
# the tab that is holding the session open. The rest of the chrome stays green, because a panel
# where everything is an accent has none.
HALOS = {
    IDLE: GREEN_DIM,
    STARTING: AMBER,
    STOPPING: AMBER,  # disconnecting is the same transition, run backwards
    CONNECTING: AMBER,
    LISTENING: AQUA,
    SPEAKING: AQUA,
    LOOKING: AQUA,
    SEARCHING: AQUA,
    DRAWING: AQUA,
    ERROR: RED,
}
# What the strip calls each state. Kept here rather than taken from the controller's ``detail``
# because two of these states are the kiosk's own invention and the controller has never heard
# of them; the controller's sentence goes in the caption underneath instead.
#
# Asleep, waking, sleeping - because the button underneath says WAKE UP, and a box that is asked
# to wake up does not answer STANDBY. The five states in the middle already read as a creature
# doing something and are left alone, and so is FAULT: the metaphor does not get to swallow the
# one word that has to be believed.
LABELS = {
    IDLE: "ASLEEP",
    STARTING: "WAKING",
    STOPPING: "SLEEPING",
    CONNECTING: "WAKING",
    LISTENING: "LISTENING",
    SPEAKING: "SPEAKING",
    LOOKING: "OPTICS",
    SEARCHING: "SEARCH",
    DRAWING: "DRAWING",
    ERROR: "FAULT",
}
# ... and the resting line underneath: what is true about a state when nothing finer is known.
# The controller sends a better sentence whenever it has one - what is being searched for, which
# project is being opened, which step of the teardown is running - and that wins; this is what the
# panel falls back on. Every state has one, so the line is never blank while a session is up.
CAPTIONS = {
    IDLE: "asleep — tap WAKE UP",
    STARTING: "waking up…",
    STOPPING: "going to sleep…",
    CONNECTING: "waking up…",
    LISTENING: "listening — talk to me",
    SPEAKING: "speaking…",
    LOOKING: "looking…",
    SEARCHING: "searching the web…",
    DRAWING: "drawing…",
}

# ...and how the eye behaves in each - the fourth table, and the one that says what he is *like*
# rather than what he is doing. :mod:`cyclops.eye` is only the mechanism; every state here is
# nine numbers and a colour, so a new one costs a line and tuning one costs a keystroke. Render
# `tools/eye_sheet.py` after touching any of them: these are meant to be looked at, not reasoned
# about.
#
# The colours are not the border's. The border answers "is the agent up?" in three colours and
# has to be readable across a workshop; the eye is a face, and it may say something finer - so a
# state that looks the same on the rim can still look different in the middle of the bar.
MOODS = {
    # Shut, dark, and utterly still. Everything else on this panel holds still with him, and that
    # stillness is what makes any of the rest read as being awake.
    IDLE: Mood(tint=GREEN_MID, aperture=0.0, swell=0.0, breath_s=0.0, spin=0.0),
    # Coming round: the iris only half up, the rings running fast, and a highlight sweeping the
    # rim - a thing spinning itself up rather than a thing paying attention.
    STARTING: Mood(tint=AMBER, aperture=0.34, swell=0.10, breath_s=1.5, spin=54.0, scan=88.0),
    CONNECTING: Mood(tint=AMBER, aperture=0.34, swell=0.10, breath_s=1.5, spin=54.0, scan=88.0),
    # Winding down. The same transition run backwards, which is what the rings do.
    STOPPING: Mood(tint=AMBER, aperture=0.10, swell=0.04, breath_s=3.0, spin=-22.0),
    # Awake and attending. A resting breath, a barely-moving ring set, and the one mood whose
    # iris opens to your voice - which is the panel saying it can hear you.
    LISTENING: Mood(
        tint=AQUA, aperture=0.52, swell=0.07, breath_s=4.0, voice=0.30, spin=7.0, blink_s=4.4
    ),
    # Talking: a faster breath and a wider iris, because he is doing the thing rather than
    # waiting to. Barely opens to level here - the level *is* his own voice coming back.
    SPEAKING: Mood(
        tint=AQUA, aperture=0.70, swell=0.17, breath_s=1.1, voice=0.10, spin=13.0, blink_s=5.5
    ),
    # Looking at a photo. Wide, still, and it does not blink: this is a stare.
    LOOKING: Mood(tint=AQUA, aperture=0.88, swell=0.02, breath_s=6.0, spin=3.0),
    # Hunting. Narrowed to a point, breathing fast, rings tearing round with a scanning arc.
    SEARCHING: Mood(tint=AQUA, aperture=0.36, swell=0.06, breath_s=0.9, spin=155.0, scan=118.0),
    # Drawing. Deliberate, and turning the other way, because it is making rather than looking.
    DRAWING: Mood(tint=AQUA, aperture=0.46, swell=0.05, breath_s=2.2, spin=-34.0),
    # A fault. Still and red, and pointedly not pulsing: a thing that throbs is asking to be
    # watched, and this one is asking to be read - the caption underneath says what broke.
    ERROR: Mood(tint=RED, aperture=0.20, swell=0.0, breath_s=0.0, spin=0.0),
}

# A caption that ends in an ellipsis is a caption about work in flight, and that is the whole test
# the line uses to decide whether to move: "searching the web…" walks its dots and breathes,
# "listening — talk to me" holds still. Every phrase the controller publishes obeys the same rule,
# which is why none of them has to say twice whether it is a job or a state.
BUSY_MARK = "…"
MARKER = "› "  # what every caption opens with, and the smallest thing that wears the accent
CAPTION_ALPHA = 245
CAPTION_DOTS = 3
DOT_PERIOD_S = 1.2  # one sweep of the three dots...
BREATH_PERIOD_S = 2.4  # ...and one breath of the phosphor, at half that rate so the two never lock
BREATH_DEPTH = 0.30  # how far the text sinks towards the slab at the bottom of a breath

HALO_CORE = 0.004  # fraction of the height held at full brightness, hard against the edge
HALO_FALLOFF = 0.024  # and how far the light reaches inwards before it is gone
HALO_PEAK = 0.45  # alpha at the border, falling away to nothing before it reaches any text
TINT_ALPHA = 0.05  # green wash under the chrome - phosphor cast, not a colour filter
SCANLINE_EVERY = 3  # every third row of the chrome is darkened...
SCANLINE_ALPHA = 0.17  # ...by this much, which is a CRT at arm's length and not a zebra
VIGNETTE_FROM = 0.46  # where the corner shading starts, as a fraction of the half-diagonal
VIGNETTE_ALPHA = 0.42
GLOW_RADIUS = 4.0  # blur, in reference pixels, of the bloom baked in under the frame lines
GLOW_ALPHA = 0.42
# The strip, the tab row and the four corner scraps used to be opaque near-black, which bought
# contrast at the price of a third of the panel: HEADER_H and FOOTER_H together cover 29% of the
# screen, and that is 29% of a feed you are holding down a pipe to see what is at the bottom of
# it. They now carry the filter and nothing else, so the picture runs edge to edge behind them
# and the chrome earns its contrast from its own opaque glyphs rather than from a bar.
PLATE_ALPHA = 210  # the one dark backing left: the caption slab, which sits on the picture
TAB_LIVE_ALPHA = 165  # ...and the selected tab's cell, tinted just enough to read as selected

# Layout, all as fractions of the height - the official 7" panel is 800x480 and is the
# reference. The tab row is deliberately the tallest thing here: three cells across the full
# width is the largest target the screen can offer, which is the point of putting the controls
# in a row rather than on discs in the corners.
#
# The border runs along the very edge of the panel: there is no margin outside it, because a
# glow sitting outside a border reads as light leaking off the device rather than as a screen
# lit from within. The state light therefore falls *inwards* from the border instead.
FRAME_RADIUS = 0.034
LINE = 0.0042  # stroke of the border and the rules
PAD = 0.036  # inner padding - wide enough that the inward glow never reaches any text
HEADER_H = 0.118
FOOTER_H = 0.170
# The eye, in the middle of the tab row. Radius as a fraction of the panel's height, and a
# centre a little above the row's top edge so the disc sits *into* the bar rather than on it -
# the rule runs in from both sides, lifts over the top of him and comes down the other side.
# That shoulder is the whole reason he is here and not in a box in a corner: the bar is the one
# piece of chrome you already look at to do anything, so his face belongs in it.
EYE_R = 0.125  # 60 px at 800x480
EYE_RISE = 0.052  # how far the centre sits above the tab row's top rule - 25 px,
# which is what leaves the word under him room to be a word and not a chin strap
EYE_SHOULDER = 0.013  # ...and the gap between his rim and the rule that arcs over it
EYE_PLATE_ALPHA = 205  # the disc behind him. Lighter than the caption's slab on purpose: this
# one sits over the middle of the picture, and a porthole you cannot see through is a hole
# The one thing that moves on a sleeping panel. Everything else holds still - that stillness is
# what makes awake read as awake - but the button that ends it may say so, because a control that
# does nothing until you find it is worth pointing at. A slow swell, not a flash: this is an
# invitation, and a panel blinking at you across a workshop is an alarm.
WAKE_PERIOD_S = 2.9  # seconds a breath takes, and not a multiple of any other on this panel
WAKE_GLOW = 1.0  # the glyph and the word travel the whole way from GREEN_MID to GREEN...

RIM_PERIOD_S = 3.7  # one breath of the border, slower than the caption's and not a multiple of it
RIM_DEPTH = 0.14  # how far it sinks towards SCREEN - a mix, not an alpha, and a quarter of what
# the caption may do, because this is the one thing readable across a workshop

METER_SEGMENTS = 8  # steps in the signal bar
# How much colour is stirred into the chrome for the parts that are meant to look faded. These
# are mixes rather than alphas on purpose - see _mix: drawing them translucently would not dim
# them, it would open a window onto whatever the camera is pointed at.
TAB_LIVE = 0.20  # the selected tab's cell, which is then drawn at TAB_LIVE_ALPHA
METER_OFF = 0.45  # an unlit signal segment
RING_MIX = 0.55  # the ring around the open eye
TABS = ("shutter", "eye", "wake")  # left to right
# The keys used to be ("shutter", "admin", "eye"), from a layout where the eye was the button
# that started a session and a gear opened the page. Both moved, and the names were left naming
# the wrong cells - `hitboxes.eye` was the microphone. Renaming them does reach into kiosk.py's
# press handling, which the old comment here said would buy nothing anybody can see; what it
# buys now is a file that is not lying about which one is the eye.
TAB_LABELS = {"shutter": "SNAP", "eye": "HISTORY"}
WAKE_LABEL, SLEEP_LABEL = "WAKE UP", "SLEEP"


def tab_label(name: str, state: str) -> str:
    """What a tab is called right now. Only the last one changes, because only it is a toggle."""
    if name != "wake":
        return TAB_LABELS[name]
    return SLEEP_LABEL if session_up(state) else WAKE_LABEL


_FONT_CANDIDATES = (
    # A terminal is monospaced, so this list is monospaced faces first and only.
    # macOS
    "/System/Library/Fonts/SFNSMono.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Supplemental/Andale Mono.ttf",
    # Raspberry Pi OS / Debian
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # last resort: not mono, but real
)


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """First real TrueType face that exists on this machine; PIL's bitmap default if none do."""
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default(size)


def caption_pulse(phase: float) -> tuple[float, int]:
    """How far the caption has sunk, and how many dots trail it, at monotonic time *phase*.

    A raised cosine rather than a square wave: this is a phosphor tube, and a line snapping on
    and off reads as a fault light rather than as work being done. What comes back is a *mix*
    and not an alpha, which is not a detail - see :func:`_mix`. PIL writes into the chrome layer
    rather than compositing onto it, so a translucent letter is not a dimmer letter, it is a
    window onto whatever the camera is pointed at, punched through the slab that was put there
    to stop exactly that.

    Time rather than frames, because the loop does not run at one rate - 25 fps with the camera
    up, 5 while the admin page covers the panel, 4 asleep - and a dot per frame would gallop and
    stall along with it.
    """
    step = DOT_PERIOD_S / (CAPTION_DOTS + 1)
    return BREATH_DEPTH * breath(phase, BREATH_PERIOD_S), int((phase % DOT_PERIOD_S) / step)


def rim_breath(phase: float) -> float:
    """How far the border has sunk towards SCREEN. A *mix*, not an alpha - see :func:`_mix`."""
    return RIM_DEPTH * breath(phase, RIM_PERIOD_S)


@dataclass(frozen=True)
class Rect:
    """A hit-testable box in window pixels."""

    x: int
    y: int
    w: int
    h: int

    def contains(self, px: int, py: int) -> bool:
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h


@dataclass(frozen=True)
class Hitboxes:
    """Where the interactive elements ended up, for the mouse callback to test against."""

    shutter: Rect
    eye: Rect  # the middle cell, which is his face and opens what the box has kept
    wake: Rect


def halo_alpha(width: int, height: int) -> np.ndarray:
    """A 0..1 mask that is bright along every edge and gone a little way inside it.

    Built once per window size: the distance to the nearest edge, run through a squared ramp so
    the light drops off fast enough to stay a rim rather than a fog over the picture.
    """
    core = max(2, round(HALO_CORE * height))
    fall = max(6, round(HALO_FALLOFF * height))
    xs = np.minimum(np.arange(width), width - 1 - np.arange(width))
    ys = np.minimum(np.arange(height), height - 1 - np.arange(height))
    dist = np.minimum(ys[:, None], xs[None, :]).astype(np.float32)
    ramp = np.clip((core + fall - dist) / fall, 0.0, 1.0)
    return ramp * ramp * HALO_PEAK


def scanline_alpha(width: int, height: int) -> np.ndarray:
    """Every *SCANLINE_EVERY*-th row darkened - the cheapest half of looking like a tube."""
    mask = np.zeros((height, width), dtype=np.float32)
    mask[::SCANLINE_EVERY] = SCANLINE_ALPHA
    return mask


def vignette_alpha(width: int, height: int) -> np.ndarray:
    """Corner shading, zero in the middle and squared up towards the corners."""
    ys = (np.arange(height, dtype=np.float32) - (height - 1) / 2) / max(1.0, height / 2)
    xs = (np.arange(width, dtype=np.float32) - (width - 1) / 2) / max(1.0, width / 2)
    radius = np.sqrt(ys[:, None] ** 2 + xs[None, :] ** 2) / math.sqrt(2.0)
    ramp = np.clip((radius - VIGNETTE_FROM) / (1.0 - VIGNETTE_FROM), 0.0, 1.0)
    return ramp * ramp * VIGNETTE_ALPHA


def _over(
    dst: tuple[np.ndarray, np.ndarray], colour: tuple[int, int, int], alpha: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Source-over one flat-coloured layer onto an accumulating (rgb, alpha) pair."""
    dst_rgb, dst_a = dst
    src = np.asarray(colour, dtype=np.float32)
    a = alpha[..., None]
    out_a = alpha + dst_a * (1.0 - alpha)
    carried = dst_rgb * dst_a[..., None] * (1.0 - a)
    out_rgb = (src * a + carried) / np.maximum(out_a[..., None], 1e-6)
    return out_rgb, out_a


def _to_image(rgb: np.ndarray, alpha: np.ndarray) -> Image.Image:
    rgba = np.empty((*alpha.shape, 4), dtype=np.uint8)
    rgba[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    rgba[:, :, 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


class Overlay:
    """Renders the kiosk chrome for a given window size, and remembers where it put things."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        scale = height / 480.0  # the official 7" panel is the reference layout
        self.scale = scale
        self.line = max(1, round(LINE * height))
        self.pad = max(4, round(PAD * height))
        self.radius = max(2, round(FRAME_RADIUS * height))

        self.frame = Rect(0, 0, width, height)
        header_h = max(18, round(HEADER_H * height))
        footer_h = max(28, round(FOOTER_H * height))
        self.header = Rect(self.frame.x, self.frame.y, self.frame.w, header_h)
        self.footer = Rect(self.frame.x, self.frame.bottom - footer_h, self.frame.w, footer_h)
        self.viewport = Rect(
            self.frame.x,
            self.header.bottom,
            self.frame.w,
            self.footer.y - self.header.bottom,
        )

        # Cyclops himself: a disc centred a little above the tab row's top rule, in the middle
        # cell. Half of him is in the tab row and half is in the picture, which is what makes the
        # rule lift over him rather than run past him.
        self.eye_r = max(10, round(EYE_R * height))
        self.eye = (self.frame.x + width // 2, self.footer.y - max(2, round(EYE_RISE * height)))
        self.shoulder = self.eye_r + max(2, round(EYE_SHOULDER * height))

        self.font_mode = _load_font(max(11, round(27 * scale)))
        self.font_read = _load_font(max(9, round(21 * scale)))
        self.font_tab = _load_font(max(8, round(15 * scale)))
        self.font_brand = _load_font(max(7, round(14 * scale)))
        self.font_micro = _load_font(max(7, round(12 * scale)))
        self.font_caption = _load_font(max(8, round(14 * scale)))

        self._halo = halo_alpha(width, height)
        # The tube filter - wash, corner shading, scanlines - flattened once. It is the whole
        # substance of the strip and the tab row now, rather than something laid under opaque
        # bars, so the picture band is cut straight out of it: the camera reaches the middle of
        # the panel with nothing whatsoever in front of it, and only the chrome wears the tube.
        base: tuple[np.ndarray, np.ndarray] = (
            np.zeros((height, width, 3), dtype=np.float32),
            np.zeros((height, width), dtype=np.float32),
        )
        base = _over(base, GREEN, np.full((height, width), TINT_ALPHA, dtype=np.float32))
        base = _over(base, (0, 0, 0), vignette_alpha(width, height))
        base = _over(base, (0, 0, 0), scanline_alpha(width, height))
        rgb, alpha = base
        alpha = alpha.copy()
        v = self.viewport
        alpha[v.y : v.bottom, v.x : v.right] = 0.0
        # ...and his disc with it. Half of him is in the washed tab row and half is over the raw
        # picture, so without this the filter's own edge runs across his face as a tide line.
        # His plate is what backs him instead, and it is the same all the way round.
        cx, cy = self.eye
        ys, xs = np.ogrid[:height, :width]
        alpha[(ys - cy) ** 2 + (xs - cx) ** 2 <= self.eye_r**2] = 0.0
        self._backdrop = (rgb, alpha)
        self._plate = self._build_plate()
        self._chrome = self._build_chrome()
        # One engine per window size: it owns the geometry, and it remembers which mood it is
        # easing out of, which is why it is built here and not per frame.
        self.engine = EyeEngine(self.eye_r, self.line, SCREEN, MOODS[IDLE])
        self._bases: dict[tuple[str, bool], Image.Image] = {}
        # The readout strip's right-hand group is laid out from the frame edge inwards, and in a
        # monospaced face every width in it is a constant, so it is worked out here rather than
        # per frame - and, more to the point, the baked half and the drawn half then agree.
        self._clock_w = self.font_read.getlength("00:00")
        self._rec_w = self.font_micro.getlength("REC") + round(7 * scale) * 2
        # The space the dots will need, reserved whether any of them are showing or not. The slab
        # is sized to its text, so without this it would breathe in and out with them - and a dark
        # rectangle changing width four times a second is far more distracting than the dots.
        self._dots_w = self.font_caption.getlength("." * CAPTION_DOTS)
        self._gap = max(4, round(18 * scale))
        self._seg = (max(3, round(9 * scale)), max(6, round(18 * scale)), max(2, round(5 * scale)))
        self.hitboxes = self._layout()

    # ---- layout ----

    def _tab(self, index: int) -> Rect:
        """One of the three cells of the tab row, laid out to cover the full frame width."""
        x0 = self.footer.x + round(index * self.footer.w / len(TABS))
        x1 = self.footer.x + round((index + 1) * self.footer.w / len(TABS))
        return Rect(x0, self.footer.y, x1 - x0, self.footer.h)

    def _glyph_at(self, cell: Rect) -> tuple[int, int, int]:
        """Centre and radius of a tab's glyph - shared by the baked tab and the live ring."""
        return cell.center[0], cell.y + round(29 * self.scale), round(16 * self.scale)

    def _layout(self) -> Hitboxes:
        """Three tabs across the bottom. The rest of the frame is picture.

        Shutter left and session right are where the two discs used to be, so the thumb that
        learnt them keeps being right; the admin tab takes the middle, where a disc could never
        have gone. Each cell is roughly a third of the panel wide - about six times the area of
        the disc it replaces, which is what a tab row buys over a corner control.
        """
        self._cells = {name: self._tab(index) for index, name in enumerate(TABS)}
        # The eye's cell reaches up to take in the half of him that stands proud of the row.
        # People tap the face, not the strip of bar underneath it, and there is nothing else up
        # there to hit - only picture, which has never done anything on a tap.
        cell = self._cells["eye"]
        top = min(cell.y, self.eye[1] - self.eye_r)
        return Hitboxes(
            shutter=self._cells["shutter"],
            eye=Rect(cell.x, top, cell.w, cell.bottom - top),
            wake=self._cells["wake"],
        )

    # ---- the cached backdrop ----

    def _build_plate(self) -> Image.Image:
        """The disc behind the eye, on its own layer under the chrome.

        The same argument the caption's slab makes: he is drawn in thin rings over a live camera,
        and a green hairline over whatever the lens is pointed at is a coin toss. Lighter than
        the caption's slab, though - that one backs a line of text you have to read, and this one
        backs a face you only have to recognise, so the room can still ghost through behind him.

        A separate layer rather than part of :meth:`_build_chrome`, because that one blurs its own
        alpha to make the bloom, and a filled disc this size through a Gaussian blur is not a
        glow, it is a lamp.
        """
        layer = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        cx, cy, r = *self.eye, self.eye_r
        ImageDraw.Draw(layer).ellipse(
            [cx - r, cy - r, cx + r, cy + r], fill=(*SCREEN, EYE_PLATE_ALPHA)
        )
        return layer

    def _build_chrome(self) -> Image.Image:
        """The rules, the tab dividers and the viewport ticks, on transparency.

        Drawn once and kept: nothing in here depends on the state, only on the window size. The
        border around the outside is not in here - it carries the state colour, so it belongs to
        the per-state base and goes on last of all.
        """
        layer = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        f, pad, w = self.frame, self.pad, self.line
        gap = max(2, round(4 * self.scale))
        # A rule with a thinner companion a few pixels off it, top and bottom - the doubled
        # divider is most of what makes a green terminal read as a terminal rather than a form.
        # Along the top it crosses the whole panel; along the bottom it lifts into the eye, which
        # is the one place on this screen where the chrome gets out of something's way.
        d.line([f.x + pad, self.header.bottom, f.right - pad, self.header.bottom],
               fill=(*GREEN_MID, 230), width=w)
        d.line([f.x + pad * 3, self.header.bottom + gap, f.right - pad * 3,
                self.header.bottom + gap], fill=(*GREEN_DIM, 220), width=max(1, w // 2))
        self._draw_eye_shoulder(d, gap)
        for index in range(1, len(TABS)):  # the two dividers between the three tabs
            x = self._tab(index).x
            d.line(
                [x, self.footer.y + pad, x, f.bottom - pad],
                fill=(*GREEN_DIM, 210),
                width=max(1, w // 2),
            )
        self._draw_ticks(d)

        # Bloom, from the alpha of what was just drawn. Blurring the RGBA directly would drag
        # the colour towards black wherever it is transparent, so only the mask is blurred and
        # the glow is a flat green wearing it.
        mask = layer.getchannel("A").filter(
            ImageFilter.GaussianBlur(max(1.0, GLOW_RADIUS * self.scale))
        )
        glow = Image.new("RGBA", layer.size, (*GREEN, 0))
        glow.putalpha(mask.point(lambda v: int(v * GLOW_ALPHA)))
        return Image.alpha_composite(glow, layer)

    def _draw_eye_shoulder(self, d: ImageDraw.ImageDraw, gap: int) -> None:
        """The tab row's top rule, which runs in from both sides and arcs over the eye.

        The doubled divider everywhere else on this panel is two straight lines a few pixels
        apart. Here it is two straight lines and two arcs, concentric on him: the bar's edge
        lifts, goes over the top of his head and comes down again. That shoulder is the join -
        it is what makes him part of the row rather than a badge sitting on it.

        Chrome, not his own colour. His rim ring is drawn per frame just inside this and is free
        to go as dim as the mood wants; the line that has to stay unbroken is this one.
        """
        f, pad, w = self.frame, self.pad, self.line
        cx, cy = self.eye
        thin = max(1, w // 2)
        for radius, y, colour, stroke, inset in (
            (self.shoulder, self.footer.y, (*GREEN_MID, 230), w, pad),
            (self.shoulder + gap, self.footer.y - gap, (*GREEN_DIM, 220), thin, pad * 3),
        ):
            # Where the arc meets the straight run. His centre is above the rule, so the two
            # touch off to either side rather than at the widest point of the circle - and the
            # arc has to start and end on exactly those points or the join shows as a step.
            sin = (y - cy) / radius
            if abs(sin) >= 1.0:  # a window too short for him to stand proud of the row at all
                d.line([f.x + inset, y, f.right - inset, y], fill=colour, width=stroke)
                continue
            a = math.degrees(math.asin(sin))
            reach = radius * math.cos(math.radians(a))
            d.line([f.x + inset, y, cx - reach, y], fill=colour, width=stroke)
            d.line([cx + reach, y, f.right - inset, y], fill=colour, width=stroke)
            # 180 - a to 360 + a, which is the way round that goes over the top of his head.
            d.arc([cx - radius, cy - radius, cx + radius, cy + radius],
                  start=180 - a, end=360 + a, fill=colour, width=stroke)

    def _draw_ticks(self, d: ImageDraw.ImageDraw) -> None:
        """Corner ticks around the picture, so the live area reads as a framed feed."""
        v, pad = self.viewport, self.pad
        arm = max(6, round(26 * self.scale))
        box = (v.x + pad, v.y + pad, v.right - pad, v.bottom - pad)
        for x, dx in ((box[0], 1), (box[2], -1)):
            for y, dy in ((box[1], 1), (box[3], -1)):
                d.line([x, y, x + arm * dx, y], fill=(*GREEN_MID, 200), width=self.line)
                d.line([x, y, x, y + arm * dy], fill=(*GREEN_MID, 200), width=self.line)

    def _base(self, state: str, recording: bool) -> Image.Image:
        """Everything that holds still while the state does, built once and copied per frame.

        The state light goes on last, after the words, and falls inwards from the border. It
        was tried the other way - a margin of dark bezel with the glow spreading outwards into
        it - and it read as light leaking off the edge of the device rather than as a screen lit
        from inside. A tube blooms in front of what it is showing, so this one does too.

        Keyed on the state rather than on the halo colour, because the words change with it too.
        Baking the brand, the mode, the REC tag and the whole tab row in here is what keeps a
        frame down to a meter, a clock, a caption and a ring: drawing all of it every time cost
        10 ms of the Pi's 40 ms budget, against 2.8 ms for the chrome this design replaced.
        """
        cached = self._bases.get((state, recording))
        if cached is not None:
            return cached
        halo = HALOS.get(state, GREEN_DIM)
        # The filter, and then the chrome drawn on it. There is nothing opaque underneath either
        # of them any more: the strip, the tab row and the four scraps outside the border's
        # rounded corners are all just the wash and the scanlines over the live picture, and the
        # picture between them is not covered at all - not even by the border's own corners.
        image = Image.alpha_composite(_to_image(*self._backdrop), self._plate)
        image = Image.alpha_composite(image, self._chrome)

        d = ImageDraw.Draw(image)
        self._bake_header(d, state, halo, recording)
        for index, name in enumerate(TABS):
            self._draw_tab(d, self._tab(index), name, state, halo, pressed=False)

        # The state light, falling inwards from the border over everything drawn so far - a tube
        # blooms in front of what it is showing, not behind it. It reaches about 13 px at 480,
        # and PAD keeps every word further in than that, so nothing legible sits in it.
        rim = Image.new("RGBA", image.size, (*halo, 0))
        rim.putalpha(Image.fromarray((self._halo * 255.0).astype(np.uint8), "L"))
        image = Image.alpha_composite(image, rim)

        # ...and the border itself, crisp on top of its own glow and in the state's colour.
        ImageDraw.Draw(image).rounded_rectangle(
            [self.frame.x, self.frame.y, self.frame.right - 1, self.frame.bottom - 1],
            radius=self.radius,
            outline=(*halo, 255),
            width=self.line,
        )
        self._bases[(state, recording)] = image
        return image

    # ---- where the readouts sit ----

    @staticmethod
    def _taping(state: str, recording: bool) -> bool:
        """Is a recording actually being made? Configured to record is not the same thing.

        ``recording`` only says the setting is on. The tag has to mean "tape is running", or a
        panel sitting at ASLEEP claims to be filming the room.
        """
        return recording and session_up(state)

    def _readouts(self, taping: bool) -> tuple[float, float, float]:
        """Right edges of the clock, the REC tag and the signal meter, laid out edge inwards."""
        clock = self.frame.right - self.pad
        rec = clock - self._clock_w - self._gap
        meter = rec - (self._rec_w + self._gap) if taping else rec
        return clock, rec, meter

    def _meter_x(self, right: float) -> float:
        seg_w, _, gap = self._seg
        return right - (METER_SEGMENTS * seg_w + (METER_SEGMENTS - 1) * gap)

    # ---- text ----

    def _width(self, text: str, font: ImageFont.FreeTypeFont, tracking: float) -> float:
        return font.getlength(text) + tracking * max(0, len(text) - 1)

    def _text(
        self,
        d: ImageDraw.ImageDraw,
        x: float,
        y: float,
        text: str,
        font: ImageFont.FreeTypeFont,
        fill: tuple,
        *,
        align: str = "l",
        tracking: float = 0.0,
    ) -> float:
        """Draw *text* centred vertically on *y*, letter-spaced by *tracking*. Returns its width.

        Tracking is what separates a terminal readout from a caption, and PIL has no such
        thing, so spaced text is stepped through a character at a time.
        """
        width = self._width(text, font, tracking)
        start = x if align == "l" else x - width if align == "r" else x - width / 2
        if not tracking:
            d.text((start, y), text, font=font, fill=fill, anchor="lm")
            return width
        for char in text:
            d.text((start, y), char, font=font, fill=fill, anchor="lm")
            start += font.getlength(char) + tracking
        return width

    def _elide(self, text: str, font: ImageFont.FreeTypeFont, limit: float) -> str:
        """Trim to fit, with an ellipsis - an API error can be a paragraph long."""
        if font.getlength(text) <= limit:
            return text
        while text and font.getlength(text + "…") > limit:
            text = text[:-1]
        return text.rstrip() + "…"

    # ---- the frame ----

    def render(
        self,
        state: str,
        level: float,
        elapsed: float | None = None,
        recording: bool = False,
        flash: float = 0.0,
        pressed: str | None = None,
        detail: str = "",
        phase: float = 0.0,
    ) -> np.ndarray:
        """Draw the whole chrome for this frame and return it as an RGBA numpy array.

        ``phase`` is a monotonic clock in seconds, and the only argument here that is not about
        what the panel is showing but about *when*. It is passed in rather than read here so a
        frame is a pure function of its arguments and the caption's animation can be tested
        without a clock - the same shape as ``flash``, which the kiosk has always computed.

        Nothing here moves while he is asleep. That is deliberate and it is half the design: the
        eye is shut, the border holds still, the caption stops breathing, and two frames of an
        idle panel are byte-identical. Against a panel that was quietly pulsing whatever it was
        doing, an awake one that pulses says nothing.
        """
        halo = HALOS.get(state, GREEN_DIM)
        layer = self._base(state, recording).copy()
        d = ImageDraw.Draw(layer)

        self._draw_readouts(d, halo, level, elapsed, self._taping(state, recording))
        self._draw_caption(d, state, halo, detail, phase)
        held = pressed == "eye"
        if pressed is not None and not held:
            # Redrawn over the tab the base has at rest: an inverted cell is the only feedback
            # a screen with no travel can give, and it lasts a handful of frames.
            cell = self._cells.get(pressed)
            if cell is not None:
                self._draw_tab(d, cell, pressed, state, halo, pressed=True)
        # After the pressed cell, not before it: the ring used to be drawn, painted over by an
        # inverted tab and then drawn again in ink. One order, one draw, one colour.
        if awake(state):
            inverted = pressed == "wake"
            ring = (
                mix(halo, INK, RING_MIX)
                if inverted
                else mix(mix(SCREEN, halo, TAB_LIVE), halo, RING_MIX)
            )
            self._draw_ring(d, ring, level)
        # Him, last of everything in the row. His cell is the one that never inverts under a
        # thumb: a face in photographic negative is not the same face, and half of him is over
        # the picture anyway, where there is no cell to invert. He acknowledges a tap by coming
        # up to full instead - which also holds for as long as the page behind him is loading,
        # so a slow browser looks like a box that heard you rather than one that ignored you.
        mood = self.engine.look(state, MOODS.get(state, MOODS[IDLE]), phase)
        if held:
            # Wide, bright and steady. Startled open is the right shape for an acknowledgement,
            # and it is the one gesture that reads the same from every mood - including asleep,
            # where you have just tapped a shut eye and it has opened to look at you.
            mood = replace(mood, tint=GREEN, rings=1.0, aperture=1.0, swell=0.0, voice=0.0)
        self.engine.draw(d, *self.eye, mood, phase, level)
        if session_up(state):
            # The teardown breathes too. He is not listening any more - the eye is already shut -
            # but the box is still working, and a panel that went stone still the moment you
            # pressed stop would look like it had stopped rather than like it was finishing.
            self._draw_rim(d, halo, phase)
        elif state == IDLE and pressed != "wake":
            self._draw_invite(d, phase)
        if flash > 0.0:
            # Green-white rather than white: a photo taken through a phosphor screen.
            d.rectangle([0, 0, self.width, self.height], fill=(214, 255, 228, int(190 * flash)))
        return np.asarray(layer)

    def _bake_header(
        self, d: ImageDraw.ImageDraw, state: str, halo: tuple[int, int, int], recording: bool
    ) -> None:
        """The half of the readout strip that only moves when the state does.

        Brand, divider and mode on the left; the SIG caption and the REC tag on the right. All
        of it is letter-spaced, which PIL can only do a character at a time, which is precisely
        why it is baked rather than redrawn 25 times a second.
        """
        _, cy = self.header.center
        x = self.frame.x + self.pad
        track = max(1.0, 2.0 * self.scale)

        x += self._text(d, x, cy, "CYCLOPS", self.font_brand, (*GREEN_DIM, 255), tracking=track)
        x += round(11 * self.scale)
        rule = round(9 * self.scale)
        d.line([x, cy - rule, x, cy + rule], fill=(*GREEN_DIM, 255), width=max(1, self.line // 2))
        x += round(11 * self.scale)
        self._text(
            d, x, cy, LABELS.get(state, "—"), self.font_mode, (*halo, 255), tracking=track * 0.7
        )

        taping = self._taping(state, recording)
        _, rec_right, meter_right = self._readouts(taping)
        if taping:
            # A filled tag rather than a red dot: the tube only has the one hue, so the way to
            # shout on it is to invert.
            half = round(11 * self.scale)
            d.rounded_rectangle(
                [rec_right - self._rec_w, cy - half, rec_right, cy + half],
                radius=max(1, round(3 * self.scale)),
                fill=(*GREEN, 255),
            )
            self._text(
                d, rec_right - round(7 * self.scale), cy, "REC", self.font_micro, (*INK, 255),
                align="r",
            )
        self._text(
            d,
            self._meter_x(meter_right) - round(9 * self.scale),
            cy,
            "SIG",
            self.font_micro,
            (*GREEN_DIM, 255),
            align="r",
        )

    def _draw_readouts(
        self,
        d: ImageDraw.ImageDraw,
        halo: tuple[int, int, int],
        level: float,
        elapsed: float | None,
        taping: bool,
    ) -> None:
        """The half that moves: the signal bar, and the clock counting the session up."""
        _, cy = self.header.center
        clock_right, _, meter_right = self._readouts(taping)
        whole = 0 if elapsed is None else int(elapsed)
        clock = "--:--" if elapsed is None else f"{whole // 60:02d}:{whole % 60:02d}"
        # Dim green with nothing to count, the state's accent the moment there is - the numbers
        # that only mean something during a session are the right place for the colour that only
        # appears during one.
        colour = (*GREEN_DIM, 255) if elapsed is None else (*halo, 255)
        self._text(d, clock_right, cy, clock, self.font_read, colour, align="r")

        seg_w, seg_h, gap = self._seg
        lit = round(max(0.0, min(1.0, level)) * METER_SEGMENTS)
        x = self._meter_x(meter_right)
        for index in range(METER_SEGMENTS):
            x0 = x + index * (seg_w + gap)
            d.rectangle(
                [x0, cy - seg_h / 2, x0 + seg_w, cy + seg_h / 2],
                fill=(*halo, 255) if index < lit else (*mix(SCREEN, GREEN_DIM, METER_OFF), 255),
            )

    def _draw_caption(
        self, d: ImageDraw.ImageDraw, state: str, halo: tuple, detail: str, phase: float
    ) -> None:
        """One line of plain English along the bottom of the picture, on its own dark slab.

        The slab is not decoration: this text sits on the live camera, and white-on-anything is
        a coin toss. It is also where an error actually says what went wrong, which the old
        chrome could only render as a red rim.

        The controller's sentence wins over this module's own table, and not the other way round
        as it used to: it is the half that knows what is being searched for, which project is
        being opened and how far the teardown has got, and CAPTIONS is what is left to say when
        it knows nothing finer. Reversed, every one of those sentences would be swallowed by a
        state word during exactly the states worth narrating.
        """
        text = detail or CAPTIONS.get(state, "")
        if not text:
            return  # the strip already says the mode; saying it twice is not a caption
        busy = text.endswith(BUSY_MARK)
        if busy:
            text = text[: -len(BUSY_MARK)]  # the dots take the ellipsis's place, and move
        font = self.font_caption
        pad = self.pad
        # Clear of the bottom-left corner tick, which would otherwise run under the slab.
        x = self.viewport.x + pad + round(34 * self.scale)
        limit = self.viewport.right - pad - x - round(30 * self.scale)
        # Off the limit before the trim and back onto the width after it, so a sentence long
        # enough to be elided cannot push its own dots off the edge of the panel.
        dots_w = self._dots_w if busy else 0.0
        text = self._elide(text, font, limit - dots_w - font.getlength(MARKER))
        width = font.getlength(MARKER + text) + dots_w
        height = round(24 * self.scale)
        y = self.footer.y - round(12 * self.scale) - height / 2
        inset = round(8 * self.scale)
        d.rectangle(
            [x, y - height / 2, x + width + inset * 2, y + height / 2], fill=(*SCREEN, PLATE_ALPHA)
        )
        # The breath runs under every caption of a session that is up - it is what makes the line
        # read as a live tube rather than a printed label - and the dots only under one about work
        # in flight, where they mean the thing everybody already reads them to mean.
        #
        # With nothing running it stops, and that is the point rather than an economy: a sleeping
        # creature's line does not breathe. It used to breathe unconditionally, which meant a
        # panel with nothing on it was quietly pulsing 809 pixels of caption - and against that
        # background an awake panel that pulses says nothing at all.
        sunk, lit = caption_pulse(phase) if session_up(state) else (0.0, 0)
        colour = mix(halo if state == ERROR else GREEN, SCREEN, sunk)
        # The marker takes the accent and the sentence does not. A whole line of running text in
        # aqua over a live camera is harder to read than the same line in phosphor, and the
        # marker is the part that is decoration anyway - so it is the part that gets to be a
        # colour, and it breathes with the words it introduces.
        used = self._text(d, x + inset, y, MARKER, font, (*mix(halo, SCREEN, sunk), CAPTION_ALPHA))
        used += self._text(d, x + inset + used, y, text, font, (*colour, CAPTION_ALPHA))
        if busy and lit:
            # Hard against the last letter, where an ellipsis belongs - these are standing in
            # for the one the phrase arrived with, not sitting beside it as a separate mark.
            self._text(d, x + inset + used, y, "." * lit, font, (*colour, CAPTION_ALPHA))

    # ---- the tab row ----

    def _draw_tab(
        self,
        d: ImageDraw.ImageDraw,
        cell: Rect,
        name: str,
        state: str,
        halo: tuple,
        pressed: bool,
    ) -> None:
        """One cell of the tab row: a glyph, a tracked label, and the fill that says what it is.

        Three appearances, and they have to stay distinguishable: at rest it is chrome on the
        dark strip; while a session is up the eye tab carries a lit edge and a tinted cell,
        the way a selected tab does; and under a thumb any tab inverts completely, which is the
        only feedback a touchscreen with no travel can give.
        """
        live = name == "wake" and session_up(state)
        if pressed:
            fill, glyph, label = (*halo, 255), INK, INK
        elif live:
            # Translucent, unlike a press: the row is see-through now, and a selected tab that
            # blacked out a third of it would put back the bar this layout just took away.
            fill, glyph, label = (*mix(SCREEN, halo, TAB_LIVE), TAB_LIVE_ALPHA), halo, GREEN
        else:
            fill, glyph, label = None, GREEN_MID, GREEN_MID

        inset = max(1, round(3 * self.scale))
        if fill is not None:
            # The outermost tabs reach the panel's own bottom corners, so their fill has to be
            # rounded to match or it squeezes out past the border, into the corner the border
            # cuts off. Only the one outer corner each: the rest of the cell is square.
            d.rounded_rectangle(
                [cell.x + inset, cell.y + inset, cell.right - inset, cell.bottom - inset],
                radius=max(0, self.radius - inset),
                corners=(False, False, name == "wake", name == "shutter"),
                fill=fill,
            )
        if live and not pressed:  # the lit edge along the top of the selected tab
            edge = max(2, round(4 * self.scale))
            d.rectangle(
                [cell.x + inset, cell.y + inset, cell.right - inset, cell.y + inset + edge],
                fill=(*halo, 255),
            )

        cx, gy, radius = self._glyph_at(cell)
        if name == "shutter":
            self._glyph_aperture(d, cx, gy, radius, glyph)
        elif name == "wake":
            self._glyph_mic(d, cx, gy, radius, glyph, state)
        # ...and the middle cell has no glyph baked at all: what stands there is the eye, and
        # every part of him moves. See :meth:`Overlay.render`.
        self._text(
            d,
            cx,
            cell.bottom - round(23 * self.scale),
            tab_label(name, state),
            self.font_tab,
            (*label, 255),
            align="c",
            tracking=max(1.0, 2.4 * self.scale),
        )

    def _glyph_aperture(self, d: ImageDraw.ImageDraw, cx: int, cy: int, r: int, c: tuple) -> None:
        """Take a photo now. A six-bladed aperture, swept the one way round."""
        stroke = max(2, round(3 * self.scale))
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(*c, 255), width=stroke)
        for blade in range(6):
            outer = math.radians(blade * 60)
            inner = outer + math.radians(60)  # the sweep is what makes it read as an iris
            d.line(
                [
                    cx + r * math.cos(outer),
                    cy + r * math.sin(outer),
                    cx + r * 0.34 * math.cos(inner),
                    cy + r * 0.34 * math.sin(inner),
                ],
                fill=(*c, 255),
                width=stroke,
            )

    def _draw_invite(self, d: ImageDraw.ImageDraw, phase: float) -> None:
        """Breathe the WAKE UP cell, because it is the only thing left to do.

        Drawn per frame over the cell the base baked at rest, which is why the wash goes down
        first and the glyph and the word on top of it: the base's own copies of those are
        underneath, and this covers them.

        Only while he is asleep proper. Not on a fault - a red panel with a green button
        beckoning at you is a machine asking to be prodded rather than read, and the line under
        the picture is where a fault has something to say.
        """
        cell = self._cells["wake"]
        swell = breath(phase, WAKE_PERIOD_S)
        inset = max(1, round(3 * self.scale))
        d.rounded_rectangle(
            [cell.x + inset, cell.y + inset, cell.right - inset, cell.bottom - inset],
            radius=max(0, self.radius - inset),
            corners=(False, False, True, False),
            fill=(*mix(SCREEN, GREEN, TAB_LIVE), round(TAB_LIVE_ALPHA * swell)),
        )
        # ...and a lit edge along the top, swelling with it. Both of those are the vocabulary
        # this panel already uses for a selected tab, borrowed rather than invented: at the top
        # of the breath the cell looks exactly the way the SLEEP tab does while a session is up,
        # and then it fades back to nothing. Nothing is selected while he is asleep, so there is
        # nothing for it to be confused with.
        edge = max(2, round(4 * self.scale))
        d.rectangle(
            [cell.x + inset, cell.y + inset, cell.right - inset, cell.y + inset + edge],
            fill=(*GREEN, round(255 * swell)),
        )
        colour = mix(GREEN_MID, GREEN, WAKE_GLOW * swell)
        cx, gy, radius = self._glyph_at(cell)
        self._glyph_mic(d, cx, gy, radius, colour, IDLE)
        self._text(
            d,
            cx,
            cell.bottom - round(23 * self.scale),
            tab_label("wake", IDLE),
            self.font_tab,
            (*colour, 255),
            align="c",
            tracking=max(1.0, 2.4 * self.scale),
        )

    def _draw_rim(self, d: ImageDraw.ImageDraw, halo: tuple, phase: float) -> None:
        """Re-stroke the border, sunk by one breath. Only ever called while a session is up.

        The state light *around* it is baked - it is a full-screen composite and would cost
        milliseconds a frame - so what breathes is the crisp line on top of it, which is the
        brightest edge on the panel and the one a workshop reads from across the room. At the top
        of the breath the mix is zero and this puts back exactly the pixels ``_base`` drew, so the
        rim only ever dims from where it is now and never brightens past it.

        Redrawing over the same geometry is safe because ImageDraw does not anti-alias: the sunk
        stroke covers precisely the pixels the bright one did, with no fringe left showing.
        """
        d.rounded_rectangle(
            [self.frame.x, self.frame.y, self.frame.right - 1, self.frame.bottom - 1],
            radius=self.radius,
            outline=(*mix(halo, SCREEN, rim_breath(phase)), 255),
            width=self.line,
        )

    def _draw_ring(self, d: ImageDraw.ImageDraw, colour: tuple, level: float) -> None:
        """The ring around the microphone, which swells with your voice as the eye's pupil does.

        Drawn per frame rather than baked with the rest of the tab, for the obvious reason that
        it is the only part of the tab row with anything to say between one frame and the next.
        """
        cx, cy, r = self._glyph_at(self.hitboxes.wake)
        ring = round(r * (1.25 + 0.30 * max(0.0, min(1.0, level))))
        d.ellipse(
            [cx - ring, cy - ring, cx + ring, cy + ring],
            outline=(*colour, 255),
            width=max(1, self.line // 2),
        )

    def _glyph_mic(
        self, d: ImageDraw.ImageDraw, cx: int, cy: int, r: int, c: tuple, state: str
    ) -> None:
        """Start or stop the agent. Hollow while it is down, filled in while it is listening.

        A microphone rather than the eye that was here: what this tab starts is a conversation,
        and the camera it also opens is the *other* two tabs' business - SNAP shows Cyclops a
        picture, and the panel behind all three is already a viewfinder. An eye over the word
        SESSION said the wrong one of the two things this box does.

        The word says WAKE UP now, and the eye is back - but in the corner of the panel where it
        can be a face rather than a control (see :meth:`_draw_eye`). This tab is still about the
        conversation, so it is still a microphone.

        Filled is the whole state indicator, and it has to survive being 32 px on a panel seen
        from across a bench: an outline that gained a detail when live would read as neither.
        """
        stroke = max(2, round(3 * self.scale))
        live = awake(state)

        # Four parts, and they must not overlap. The first cut of this made the head 1.84 r tall,
        # which left the cradle and the stem drawn straight through it - at 32 px that reads as a
        # bullet with fins rather than as a microphone. The head gets the top half and stops; the
        # cradle's arms rise past its waist without reaching its crown; the stand has the rest.
        half = round(r * 0.44)
        head = [cx - half, cy - round(r * 0.94), cx + half, cy + round(r * 0.19)]
        if live:
            d.rounded_rectangle(head, radius=half, fill=(*c, 255))
        else:
            d.rounded_rectangle(head, radius=half, outline=(*c, 255), width=stroke)

        # An arc over an ellipse wider than the head, so its arms end beside the head and not
        # under it. Drawn 0 to 180, which is the lower half.
        arms = round(r * 0.69)
        d.arc(
            [cx - arms, cy - round(r * 0.69), cx + arms, cy + round(r * 0.56)],
            start=0, end=180, fill=(*c, 255), width=stroke,
        )
        d.line([cx, cy + round(r * 0.56), cx, cy + round(r * 0.88)], fill=(*c, 255), width=stroke)
        # The foot is what stops it reading as a pill hung on a hook.
        foot, base = round(r * 0.38), cy + round(r * 0.88)
        d.line([cx - foot, base, cx + foot, base], fill=(*c, 255), width=stroke)


def composite(frame_bgr: np.ndarray, rgba: np.ndarray) -> np.ndarray:
    """Alpha-blend an RGBA overlay onto a BGR frame, in place-ish, without touching PIL again."""
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    rgb = rgba[:, :, :3][:, :, ::-1].astype(np.float32)  # RGB -> BGR to match the frame
    blended = frame_bgr.astype(np.float32) * (1.0 - alpha) + rgb * alpha
    return blended.astype(np.uint8)


# Unsharp masking, for a camera that cannot be asked to do better. The endoscope streams
# 640x480 JPEG at quality ~44 and ignores the protocol's resolution command, so this is the only
# remaining lever on how much of the room you can actually make out - see the README.
#
# The floor and the ceiling are what separate this from a sharpen slider. Detail below the floor
# is the sensor's noise and the encoder's blocking, and amplifying that is how a sharpened cheap
# camera comes to look like a cheap camera someone has sharpened. The ceiling caps how far any
# one pixel may travel, which is what kills the white halo an unsharp mask otherwise draws down
# every high-contrast edge - a face against a bright window grew one immediately without it.
# Measured on a Pi frame: 2.9x the Laplacian variance for +7% in the flat-area noise floor.
SHARPEN_RADIUS = 1.1  # gaussian sigma of the blur that defines "detail", in source pixels
SHARPEN_KERNEL = (5, 5)  # ...over a kernel stated rather than derived - see sharpen()
SHARPEN_AMOUNT = 0.8  # how much of the detail layer goes back on top
SHARPEN_FLOOR = 4  # ...but only where the detail is at least this strong
SHARPEN_CEILING = 12  # ...and no pixel may move further than this


def sharpen(frame_bgr: np.ndarray) -> np.ndarray:
    """Unsharp-mask a frame, gated so it does not amplify what the JPEG encoder invented.

    Built from OpenCV primitives rather than the obvious numpy, because this runs on every
    preview frame inside the kiosk's 40 ms budget: the numpy version of the same arithmetic is
    several times slower for a result that differs by at most 3 levels in 1% of pixels.

    Two things here are performance, not taste, and both were measured on the Pi. The kernel is
    stated because leaving it to OpenCV derives a 9x9 from the sigma and spends 3.6 ms on the
    blur where 5x5 spends 1.7 ms - and at sigma 1.1 everything outside 5x5 is past 2.3 sigma and
    weighs nothing. And *frame_bgr must be contiguous*: this makes eight passes over it, and
    OpenCV copies a non-contiguous input on every one of them, which took the same function from
    8 ms to 30 ms. :func:`mirror` and :func:`fit_to_window` are what guarantee that.
    """
    import cv2  # local import keeps this module importable without a camera stack

    blur = cv2.GaussianBlur(frame_bgr, SHARPEN_KERNEL, SHARPEN_RADIUS)
    edges = cv2.threshold(
        cv2.absdiff(frame_bgr, blur), SHARPEN_FLOOR - 1, 255, cv2.THRESH_BINARY
    )[1]
    boosted = cv2.addWeighted(frame_bgr, 1.0 + SHARPEN_AMOUNT, blur, -SHARPEN_AMOUNT, 0)
    ceiling = (SHARPEN_CEILING,) * 4  # a scalar here would only reach the blue channel
    boosted = cv2.min(
        cv2.max(boosted, cv2.subtract(frame_bgr, ceiling)), cv2.add(frame_bgr, ceiling)
    )
    out = frame_bgr.copy()
    cv2.copyTo(boosted, edges, out)  # leave the flat areas exactly as they arrived
    return out


def fit_to_window(frame_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    """Centre-crop to the window's aspect, then scale - so faces keep their proportions.

    The webcam is 16:9 and the official Pi panel is 5:3; stretching one to the other makes
    everyone look wrong, so the sides get trimmed instead.

    A frame that has to be *enlarged* to fill the panel is sharpened first and then enlarged
    cubically. Both halves of that matter and both are cheap: the endoscope's 640x480 is smaller
    than the panel in both axes, so every preview pixel is invented by the interpolator, and
    INTER_LINEAR invents blurry ones. Sharpening happens before the resize because it is a third
    of the pixels there and therefore a third of the cost, and because sharpening after an
    upscale sharpens the interpolation's own softness rather than the picture's detail. A frame
    being *reduced* - a real webcam at 1280x720 - skips both: INTER_AREA is already sharp, and
    there is nothing there to rescue.
    """
    import cv2  # local import keeps this module importable without a camera stack

    h, w = frame_bgr.shape[:2]
    want = width / height
    have = w / h
    if have > want:  # too wide - trim left and right
        new_w = int(h * want)
        x0 = (w - new_w) // 2
        frame_bgr = frame_bgr[:, x0 : x0 + new_w]
    elif have < want:  # too tall - trim top and bottom
        new_h = int(w / want)
        y0 = (h - new_h) // 2
        frame_bgr = frame_bgr[y0 : y0 + new_h, :]
    if frame_bgr.shape[0] > height:
        return cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
    # Trimming rows leaves the buffer contiguous and this costs nothing; trimming columns does
    # not, and then paying for one copy here is far cheaper than letting OpenCV make its own
    # inside every call that follows. See sharpen() for what that was worth measuring.
    return cv2.resize(
        sharpen(np.ascontiguousarray(frame_bgr)), (width, height), interpolation=cv2.INTER_CUBIC
    )


def mirror(frame_bgr: np.ndarray) -> np.ndarray:
    """Flip horizontally so the preview behaves like a mirror, which is what people expect.

    ``frame[:, ::-1]`` is the obvious way to write this and was how it was written. It is also
    free only where it is written: a reversed slice is a view with a negative stride, and every
    OpenCV call downstream of it silently copies the frame into a contiguous buffer before it can
    do anything. That was costing the render loop 4.6 ms a frame on the resize alone, long before
    anything else wanted the pixels. Flipping properly, once, hands the rest of the chain a
    buffer it can work on directly.
    """
    import cv2  # local import keeps this module importable without a camera stack

    return cv2.flip(frame_bgr, 1)


def platform_font_note() -> str:
    """Which font face got picked - useful when the Pi and the Mac disagree."""
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            return path
    return f"PIL default bitmap font (no TrueType found on {sys.platform})"


def message(width: int, height: int, text: str) -> np.ndarray:
    """A black frame with one line centred on it, as BGR - the panel with nothing to show.

    Drawn here rather than in :mod:`cyclops.kiosk` because this is where the fonts already live.
    Dim phosphor rather than white: it is a caption explaining a black rectangle, not a thing to
    read across the room, and it has to sit under the same green chrome as everything else.
    """
    image = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    font = _load_font(max(16, height // 26))
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    draw.text(
        ((width - (right - left)) // 2 - left, (height - (bottom - top)) // 2 - top),
        text,
        font=font,
        fill=GREEN_MID,
    )
    return np.array(image)[:, :, ::-1].copy()  # RGB -> BGR, as everything downstream expects
