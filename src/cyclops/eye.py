"""Cyclops' eye: the one thing on the panel that is a face rather than a readout.

The boot mark brought to life. ``assets/splash.png`` is an iris inside concentric HUD rings, and
this is that drawing with the rings turning, the iris breathing and the whole thing tinted by
whatever the box is doing - so a glance at the middle of the tab row answers "is he there, and
what is he up to" without reading a word.

Everything the eye does is a :class:`Mood`: ten numbers and a colour. The engine is only the
mechanism that draws them, which is what makes the eye tunable - a state is not code here, it is
a row in a table (``overlay.MOODS``), and a new one costs a line. The parameters are meant to be
pushed around: ``tools/eye_sheet.py`` renders every mood over a strip of time, so a change can be
looked at rather than argued about.

Nothing in here knows about the panel, the palette or the camera. It takes a radius, a colour and
a clock, and draws. That is what lets the preview harness import it on its own, and it is why the
state table lives in :mod:`cyclops.overlay` where the phosphor colours are.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from PIL import Image, ImageDraw

SUPERSAMPLE = 4  # how many times over-size he is drawn before being averaged onto the panel.
# PIL will not anti-alias anything, and he is nothing but curves: every arc, tick and blade in
# here is a hard-edged run of pixels, which on a circle 120 px across reads as a staircase rather
# than as a ring. Drawing him four times over and shrinking is the whole fix. Four is where the
# stairs stop showing at this size; eight costs four times as much and looks the same from a
# bench, and one puts back exactly the pixels this drew before any of it existed.

GAMMA = 2.2  # the panel's, and the reason the shrink happens in light rather than in numbers.
# Averaging two 8-bit values does not average two brightnesses. A pixel half covered by a bright
# line, averaged as numbers, comes out at half its value - which is under a quarter of its light
# - so a ring made mostly of half-covered pixels goes grey and the smoothing reads as a fade. The
# tile is drawn in linear light and converted back on the way out instead, which is what keeps a
# smoothed hairline as bright as the jagged one it replaces.
LINEAR = tuple(round(255 * (v / 255) ** GAMMA) for v in range(256))
SRGB = np.array([round(255 * (v / 255) ** (1 / GAMMA)) for v in range(256)], dtype=np.uint8)

# The ring set, as fractions of the eye's radius, outside in. Four rings and an iris is as much
# structure as survives being 110 px across on a panel seen from a bench; the splash mark has
# seven, and at this size the inner three turn to porridge.
RIM = 1.00  # the outer ring, which is also what the tab row's rule runs into
TICKS = 0.86  # a ring of radial ticks, turning with the mood's spin
TICK_N = 16
TICK_LEN = 0.07
BRACKETS = 0.74  # four arcs with gaps, turning back the other way
BRACKET_N = 4
BRACKET_ARC = 52  # degrees each
DOTS = 0.62  # a dotted ring, turning slowly with the ticks
DOT_N = 12
IRIS = 0.50  # the iris rim, and the blades that sweep in from it
BLADE_N = 10
BLADE_SWEEP = 46  # degrees each blade leans round - the swirl that makes it an iris
HOLE_MIN = 0.06  # the hole's radius as a fraction of the iris, shut...
HOLE_MAX = 0.62  # ...and wide open. Not larger: past this the pupil swallows the
# blades and the iris stops reading as an iris at all

# Counter-rotation, as multiples of the mood's spin. Rings that all turn together read as one
# disc; rings that disagree read as a mechanism.
TICK_SPIN = 1.0
BRACKET_SPIN = -0.62
DOT_SPIN = 0.31
SCAN_SPIN = 2.3  # the sweeping highlight, when a mood asks for one...
SCAN_DIM = 0.42  # ...and how far the rim is turned *down* underneath it while it sweeps

# How present each ring is against the rim, before the mood's own `rings` scales all of them.
RIM_LIT = 1.0
TICK_LIT = 0.72
BRACKET_LIT = 0.85
DOT_LIT = 0.55
IRIS_LIT = 0.80
BLADE_LIT = 0.62
PUPIL_LIT = 0.95

BLINK_S = 0.22  # one blink, down and back up: five or six frames at 25 fps, and anything
# shorter is indistinguishable from a dropped frame
BLINK_DRIFT_S = 2.6  # how far a blink may wander inside its window: a blink on a fixed period
# is a status LED, not a creature
BLINK_DRIFT = 0.6180339887  # golden ratio, so no two consecutive gaps come out the same length

MOOD_EASE_S = 0.45  # how long the eye takes to become a different mood. Colour that snaps reads
# as a warning lamp; colour that travels reads as the same creature changing its mind


def mix(base: tuple[int, int, int], other: tuple[int, int, int], amount: float) -> tuple:
    """*amount* of *other* stirred into *base*, so a tinted fill can be one opaque colour.

    The rule everything small on this panel obeys. PIL's ImageDraw *writes* into an RGBA image
    rather than compositing onto it, so a fill at 40% alpha does not come out 40% dimmer - it
    stamps a 40%-opaque patch and the picture behind the panel shows through it. That is fine for
    a whole cell of see-through chrome and no good at all for a ring that has to stay legible
    over whatever the camera happens to be pointed at. Those are mixed towards the screen's own
    black and drawn opaque instead.
    """
    a = max(0.0, min(1.0, amount))
    return tuple(round(b + (o - b) * a) for b, o in zip(base, other, strict=True))


def breath(phase: float, period: float) -> float:
    """0 at the top of a cycle, 1 at the bottom, on a raised cosine.

    The one shape everything alive on this panel moves with. A square wave here reads as a fault
    light rather than as something breathing.
    """
    if period <= 0.0:
        return 0.0
    return 0.5 - 0.5 * math.cos(2.0 * math.pi * (phase % period) / period)


def blink(phase: float, every: float) -> float:
    """1.0 between blinks, 0.0 at the bottom of one. ``every`` of 0 means a mood that never does.

    One blink per window of *every* seconds, but not at the top of it: each window's is offset by
    a different fraction of :data:`BLINK_DRIFT_S`, so consecutive gaps differ and the thing reads
    as a face rather than as an indicator lamp. The offsets come off the golden ratio because it
    is the least rational number there is - the sequence never settles into a period anybody can
    anticipate - and because a multiply and a mod is the whole cost. Not a PRNG, and emphatically
    not ``hash()`` of anything, which is salted per process and would give the box a different
    blink on every boot.
    """
    if every <= 0.0:
        return 1.0
    window = math.floor(phase / every)
    started = window * every + min(BLINK_DRIFT_S, every - BLINK_S) * ((window * BLINK_DRIFT) % 1.0)
    since = phase - started
    if not 0.0 <= since < BLINK_S:
        return 1.0
    return 1.0 - breath(since, BLINK_S)


@dataclass(frozen=True)
class Mood:
    """How the eye looks and moves. One per state, and the only thing that differs between them.

    Ten numbers and a colour. Every one of them is meant to be pushed around - see the module
    docstring - so none of them is allowed to be load-bearing on its own: an eye with every
    parameter at zero is a dim ring, not a crash.
    """

    tint: tuple[int, int, int]  # the eye's own colour, whatever the border happens to be doing
    aperture: float = 0.5  # how far the iris stands open at rest, 0 shut .. 1 wide
    swell: float = 0.06  # ...and how much of that the breath gives and takes back
    breath_s: float = 4.0  # seconds per breath. A resting creature is twelve to fifteen a minute
    voice: float = 0.0  # how much further your voice opens it, on top of the breath
    spin: float = 6.0  # degrees a second the ring set turns; the sign is a direction
    rings: float = 1.0  # how present the rings are at all, 0 .. 1
    sink: float = 0.0  # ...and how much of that the breath takes away again at the bottom
    # of it. The other half of a breath, and the half a shut eye is left with: `swell` moves
    # the iris, which a sleeping face has none of to move, and this dims the whole of him
    # instead. Named for the two things on this panel that already do it - see
    # `overlay.rim_breath` and `overlay.caption_pulse`, both of which hand back a *sunk*.
    scan: float = 0.0  # length in degrees of a bright arc sweeping the rim, 0 for none
    blink_s: float = 0.0  # mean seconds between blinks; 0 for a mood that does not blink

    def lerp(self, other: Mood, t: float) -> Mood:
        """Part-way from this mood to *other*, colour included."""
        k = max(0.0, min(1.0, t))
        if k <= 0.0:
            return self
        if k >= 1.0:
            return other
        return replace(
            other,
            tint=mix(self.tint, other.tint, k),
            **{
                name: getattr(self, name) + (getattr(other, name) - getattr(self, name)) * k
                for name in ("aperture", "swell", "breath_s", "voice", "spin", "scan", "sink")
            },
            rings=self.rings + (other.rings - self.rings) * k,
        )


def at(p: float) -> float:
    """A panel coordinate - or a radius, which maps the same way - in the oversampled tile.

    A pixel at *p* covers the run from ``p * SUPERSAMPLE`` to ``p * SUPERSAMPLE + SUPERSAMPLE-1``
    over there, and PIL's boxes are inclusive of both ends, so the middle of that run is half a
    sample short of the far edge. Getting this wrong by that half-sample is not a rounding error
    you can ignore: it moves every ring off the centre the tile was cut around.

    Centres and radii, then, and not corners - a bounding box is built out of the two, because a
    corner put through here comes out a third of a pixel wide of where it belongs.
    """
    return p * SUPERSAMPLE + (SUPERSAMPLE - 1) / 2.0


def wide(stroke: float) -> float:
    """A stroke width over there, which is the plain multiple - a width has no ends to be off."""
    return max(1.0, stroke * SUPERSAMPLE)


def linear(colour: tuple[int, int, int], alpha: int = 255) -> tuple:
    """A colour in the form the tile wants it: linear light, multiplied by its own coverage.

    Both halves are undone by :func:`_straighten` on the way out, and both are there for the same
    reason - an average is only meaningful over numbers that mean something added together.
    Brightness in the panel's gamma does not (see :data:`GAMMA`), and neither does a colour
    holding out on how much of the pixel it actually covers.
    """
    return (*(LINEAR[c] * alpha // 255 for c in colour), alpha)


def smoothed(size: int, paint: Callable[[ImageDraw.ImageDraw], None]) -> Image.Image:
    """*paint* drawn into a tile ``SUPERSAMPLE`` times *size*, handed back at *size*, unstepped.

    The whole of the anti-aliasing on this panel, in four lines. What *paint* draws must be laid
    out through :func:`at` and :func:`wide` and coloured through :func:`linear`; everything else
    about it is ordinary PIL.

    The ground is black at nothing, and that is load-bearing rather than tidy: the shrink averages
    the empty pixels in with the drawn ones, so what comes out is already multiplied by its own
    coverage - the form a composite wants, and the only one that does not leave a dark fringe down
    the outside of every curve.
    """
    tile = Image.new("RGBA", (size * SUPERSAMPLE, size * SUPERSAMPLE), (0, 0, 0, 0))
    paint(ImageDraw.Draw(tile))
    # reduce() rather than resize(): the same box average, told up front that the ratio is a whole
    # number, and half the time for it on the Pi.
    return _straighten(tile.reduce(SUPERSAMPLE))


def _straighten(tile: Image.Image) -> Image.Image:
    """Divide the shrunken tile back out by its own coverage, and return it to the panel's gamma.

    The second half of the trick in :meth:`EyeEngine.paint`, and the reason it runs *after* the
    shrink rather than before it: fifty-odd thousand numbers on the tile as it lands instead of a
    million on the tile as it was drawn, which is the difference between a rounding error in the
    frame budget and a dropped frame.

    Where nothing was drawn there is nothing to divide by, and those pixels stay at zero - fully
    transparent, so what they carry never reaches the panel anyway.
    """
    px = np.asarray(tile)
    alpha = px[..., 3]
    scale = np.divide(255.0, alpha, out=np.zeros(alpha.shape, np.float32), where=alpha > 0)
    rgb = (px[..., :3] * scale[..., None]).clip(0.0, 255.0).astype(np.uint8)
    # The alpha band rides through untouched: it is a coverage, not a brightness, and has no
    # gamma to be in.
    return Image.fromarray(np.dstack((SRGB[rgb], alpha)), "RGBA")


class EyeEngine:
    """Draws the eye at one size, and remembers which mood it is on its way to.

    Geometry is worked out once per size and the mood per frame, for the reason the whole overlay
    is built that way: a Pi has 40 ms for the loop and the camera wants most of them. What is
    left here is four rings, ten blades and a pupil - all of them arcs, lines and ellipses at the
    perimeter of a 110 px circle, which is nothing, and none of it cached, because every single
    part of it moves.
    """

    def __init__(self, radius: int, line: int, screen: tuple[int, int, int], resting: Mood) -> None:
        self.r = radius
        self.screen = screen
        self.stroke = max(1, line)
        self.thin = max(1, line // 2)
        self._mood = resting  # what it is now...
        self._from = resting  # ...what it was before that, and when it changed
        self._at = 0.0
        self._key = ""
        # His tile is his bounding box and not a pixel more - 121 px square as it lands on the
        # 800x480 panel - which is what makes drawing him four times over affordable inside a
        # 40 ms frame. His centre in it is also the rim's radius, the rim being the tile drawn
        # edge to edge, and the strokes are worked out here because none of them ever changes.
        self._size = 2 * radius + 1
        self._c = at(radius)
        self._stroke = round(wide(self.stroke))
        self._thin = round(wide(self.thin))

    # ---- the mood ----

    def look(self, key: str, mood: Mood, phase: float) -> Mood:
        """The mood to draw right now, eased out of whatever the last one was.

        The eye is the only thing on the panel that crossfades. The border snaps between its
        three colours because it is a signal and wants to be read the instant it changes; the
        eye is a face, and a face that jumped from green to amber between two frames would read
        as a different creature rather than as this one waking up.
        """
        if key != self._key:
            self._from = self._mood
            self._at = phase
            self._key = key
        elapsed = phase - self._at
        t = 1.0 if MOOD_EASE_S <= 0 else max(0.0, min(1.0, elapsed / MOOD_EASE_S))
        self._mood = self._from.lerp(mood, t)
        return self._mood

    # ---- the drawing ----

    def aperture(self, mood: Mood, phase: float, level: float) -> float:
        """How far the iris actually stands open this frame: rest, breath, voice, then the blink.

        The blink multiplies rather than subtracts, so it closes the eye all the way from
        wherever it happened to be - a mood that sits half open still blinks shut, not to a
        quarter.
        """
        open_ = mood.aperture + mood.swell * breath(phase, mood.breath_s)
        open_ += mood.voice * max(0.0, min(1.0, level))
        return max(0.0, min(1.0, open_)) * blink(phase, mood.blink_s)

    def paint(
        self, img: Image.Image, cx: int, cy: int, mood: Mood, phase: float, level: float
    ) -> None:
        """The whole eye, onto *img* with its centre at *cx, cy*.

        Drawn into a tile of its own and shrunk onto the panel rather than stroked straight onto
        it, because PIL will not anti-alias and he is nothing but curves. Everything about the
        drawing is the same either way - :data:`SUPERSAMPLE` of 1 puts back the pixels this used
        to put down - so the smoothing is a property of how he lands, not of how he is drawn.

        The tile is opaque only where he is, so what shows between the rings is still his plate
        and the room through it.
        """
        tile = smoothed(self._size, lambda d: self._draw(d, mood, phase, level))
        img.alpha_composite(tile, (cx - self.r, cy - self.r))

    def _draw(self, d: ImageDraw.ImageDraw, mood: Mood, phase: float, level: float) -> None:
        """The whole eye, outside in, at the centre of its own oversampled tile."""
        cx = cy = r = self._c
        tint = mood.tint
        turn = mood.spin * phase
        # His presence, with the breath taken out of it. A mood that does not sink comes through
        # untouched, and so does one with no breath at all: breath() is 0 at a period of 0.
        lit = max(0.0, min(1.0, mood.rings * (1.0 - mood.sink * breath(phase, mood.breath_s))))
        stroke, thin = self._stroke, self._thin

        def shade(strength: float) -> tuple:
            # Mixed in the panel's own gamma, because that is where the palette was chosen and
            # where every other cell on the screen mixes, then handed over in the tile's terms.
            return linear(mix(self.screen, tint, strength * lit))

        # The rim. Also the thing the tab row's rule runs into, so it is normally the brightest
        # ring: it is doing structural work as well as saying how he feels. The exception is a
        # mood that sweeps - see below.
        scanning = mood.scan > 0.0
        self._circle(d, cx, cy, r, shade(RIM_LIT * (SCAN_DIM if scanning else 1.0)), stroke)
        if scanning:
            # A radar sweep, and built the way one is: the trace is bright and the ring under it
            # is faint. The highlight used to be mixed towards white over a rim at full, which
            # worked while the accent was a colour and stopped working the day the accent became
            # the tube's own white - there is nowhere paler than pale to go. Turning the rim down
            # instead needs no headroom above the tint and cannot be defeated by any of it.
            start = (SCAN_SPIN * turn) % 360.0
            d.arc(self._box(cx, cy, r), start=start, end=start + mood.scan,
                  fill=shade(RIM_LIT), width=stroke + thin)

        tick = shade(TICK_LIT)
        inner, outer = r * (TICKS - TICK_LEN), r * TICKS
        for i in range(TICK_N):
            a = math.radians(turn * TICK_SPIN + i * 360.0 / TICK_N)
            d.line(
                [cx + inner * math.cos(a), cy + inner * math.sin(a),
                 cx + outer * math.cos(a), cy + outer * math.sin(a)],
                fill=tick, width=thin,
            )

        bracket, box = shade(BRACKET_LIT), self._box(cx, cy, r * BRACKETS)
        for i in range(BRACKET_N):
            start = turn * BRACKET_SPIN + i * 360.0 / BRACKET_N
            d.arc(box, start=start, end=start + BRACKET_ARC, fill=bracket, width=thin)

        dot, ring = shade(DOT_LIT), r * DOTS
        size = max(1, thin)
        for i in range(DOT_N):
            a = math.radians(turn * DOT_SPIN + i * 360.0 / DOT_N)
            x, y = cx + ring * math.cos(a), cy + ring * math.sin(a)
            d.ellipse([x - size, y - size, x + size, y + size], fill=dot)

        # The iris, and the hole it makes. The blades lean round rather than pointing at the
        # centre, which is the whole difference between an aperture and a wagon wheel.
        open_ = self.aperture(mood, phase, level)
        iris = r * IRIS
        hole = iris * (HOLE_MIN + (HOLE_MAX - HOLE_MIN) * open_)
        self._circle(d, cx, cy, iris, shade(IRIS_LIT), thin)
        blade = shade(BLADE_LIT)
        for i in range(BLADE_N):
            a = math.radians(turn * TICK_SPIN + i * 360.0 / BLADE_N)
            b = a + math.radians(BLADE_SWEEP)
            d.line(
                [cx + iris * math.cos(a), cy + iris * math.sin(a),
                 cx + hole * math.cos(b), cy + hole * math.sin(b)],
                fill=blade, width=thin,
            )
        if hole >= 2.0 * SUPERSAMPLE and open_ > 0.08:
            d.ellipse(self._box(cx, cy, hole), fill=shade(PUPIL_LIT))

    # ---- primitives ----

    @staticmethod
    def _box(cx: float, cy: float, r: float) -> list[float]:
        return [cx - r, cy - r, cx + r, cy + r]

    def _circle(
        self, d: ImageDraw.ImageDraw, cx: float, cy: float, r: float, c: tuple, w: int
    ) -> None:
        d.ellipse(self._box(cx, cy, r), outline=c, width=w)
