"""The kiosk's chrome: a Pip-Boy-style terminal bezel drawn over the live camera frame.

OpenCV can only draw Hershey stroke fonts, which look like a 1980s oscilloscope. Everything
here is rendered into one RGBA layer with PIL instead - real TrueType text, anti-aliased
shapes, a numpy-built halo and scanline field - and handed to :mod:`cyclops.kiosk` as a numpy
array to alpha-blend onto the frame. Geometry doubles as the hit-test map: every interactive
element returns its rectangle, so a tap can be resolved without a second layout.

The layout is a terminal screen: a rounded border hard against the panel's edges, a readout
strip along the top, three tabs along the bottom, and the camera behind all of it. The picture
itself is left alone - no wash, no scanlines, nothing between you and the lens - because the
panel's job is to let you see the room and the endoscope has no detail to spare. The filter is
what the strip and the tab row are made of instead: they used to be opaque bars, and are now
that same phosphor wash, corner shading and scanline field laid over the live picture, so the
camera shows through the chrome as well as between it. The border carries the session state in
its colour and glows inwards from it, which is the one thing that has to be readable across a
workshop without reading any words.

Everything that holds still while the state does - the halo, the scanlines, the vignette, the
frame and its glow, the mode word, the tab row - is built once and cached, keyed on the state.
A Pi rendering this at 25 fps has 40 ms for the whole loop and the camera wants most of them;
what is left for a frame here is a signal meter, a clock, a caption and one ring.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

IDLE, CONNECTING, LISTENING, SPEAKING, LOOKING, SEARCHING, ERROR = (
    "idle",
    "connecting",
    "listening",
    "speaking",
    "looking",
    "searching",
    "error",
)
# Optimistic states the kiosk shows the instant you tap, before the session agrees. Tearing a
# session down takes ~2.3 s, during which the controller still honestly reports "listening" -
# so without these the button looks dead for over two seconds after you press stop.
STARTING, STOPPING = "starting", "stopping"

# Phosphor palette. A Pip-Boy screen is one hue, so the chrome is one hue: the only colour that
# changes is the halo, because it is the only part answering "is the agent up?" from across a
# room. Amber and red are the two the tube is allowed - a terminal warning and a terminal fault.
GREEN = (86, 255, 140)  # phosphor at full brightness: text, live chrome
GREEN_MID = (46, 176, 100)  # rules, dividers, glyphs at rest
GREEN_DIM = (32, 118, 70)  # the faintest thing still legible on the panel
AMBER = (255, 184, 60)
RED = (255, 86, 70)
SCREEN = (5, 15, 10)  # the green-black the readout strip and the tab row are made of
INK = (3, 11, 7)  # text on a filled tab

# The halo answers one question only - is the agent up? - so the states collapse onto three
# colours, plus red for a session that fell over. Anything finer is on the readout strip, which
# spells the state out in words for anyone close enough to read them.
HALOS = {
    IDLE: GREEN_DIM,
    STARTING: AMBER,
    STOPPING: AMBER,  # disconnecting is the same transition, run backwards
    CONNECTING: AMBER,
    LISTENING: GREEN,
    SPEAKING: GREEN,
    LOOKING: GREEN,
    SEARCHING: GREEN,
    ERROR: RED,
}
# What the strip calls each state. Kept here rather than taken from the controller's ``detail``
# because two of these states are the kiosk's own invention and the controller has never heard
# of them; the controller's sentence goes in the caption underneath instead.
LABELS = {
    IDLE: "STANDBY",
    STARTING: "LINKING",
    STOPPING: "CLOSING",
    CONNECTING: "LINKING",
    LISTENING: "LISTENING",
    SPEAKING: "SPEAKING",
    LOOKING: "OPTICS",
    SEARCHING: "SEARCH",
    ERROR: "FAULT",
}
CAPTIONS = {  # ... and what it says underneath before a session exists to say anything
    IDLE: "ready — tap SESSION to begin",
    STARTING: "opening the link…",
    STOPPING: "closing the link…",
}

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

METER_SEGMENTS = 8  # steps in the signal bar
# How much colour is stirred into the chrome for the parts that are meant to look faded. These
# are mixes rather than alphas on purpose - see _mix: drawing them translucently would not dim
# them, it would open a window onto whatever the camera is pointed at.
TAB_LIVE = 0.20  # the selected tab's cell, which is then drawn at TAB_LIVE_ALPHA
METER_OFF = 0.45  # an unlit signal segment
RING_MIX = 0.55  # the ring around the open eye
TABS = ("shutter", "admin", "eye")  # left to right; shutter and eye keep the corners they had
TAB_LABELS = {"shutter": "SNAP", "admin": "SYSTEM", "eye": "SESSION"}

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


def _mix(base: tuple[int, int, int], other: tuple[int, int, int], amount: float) -> tuple:
    """*amount* of *other* stirred into *base*, so a tinted fill can be one opaque colour.

    The rule everything small on the chrome obeys. PIL's ImageDraw *writes* into an RGBA image
    rather than compositing onto it, so a fill at 40% alpha does not come out 40% dimmer - it
    stamps a 40%-opaque patch and the picture behind the panel shows through it. Now that the
    strip and the tab row are see-through that is fine for a whole cell, and deliberate for the
    selected one, but it is no good for the small things that have to stay legible against
    whatever the camera happens to be pointed at: an unlit signal segment has to read as unlit
    over a lamp. Those are mixed towards :data:`SCREEN` and drawn opaque instead.
    """
    return tuple(round(b + (o - b) * amount) for b, o in zip(base, other, strict=True))


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
    eye: Rect
    admin: Rect


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
        self._backdrop = (rgb, alpha)
        self._chrome = self._build_chrome()
        self._bases: dict[tuple[str, bool], Image.Image] = {}
        # The readout strip's right-hand group is laid out from the frame edge inwards, and in a
        # monospaced face every width in it is a constant, so it is worked out here rather than
        # per frame - and, more to the point, the baked half and the drawn half then agree.
        self._clock_w = self.font_read.getlength("00:00")
        self._rec_w = self.font_micro.getlength("REC") + round(7 * scale) * 2
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
        return Hitboxes(
            shutter=self._cells["shutter"], eye=self._cells["eye"], admin=self._cells["admin"]
        )

    # ---- the cached backdrop ----

    def _build_chrome(self) -> Image.Image:
        """The rules, the tab dividers and the viewport ticks, on transparency.

        Drawn once and kept: nothing in here depends on the state, only on the window size. The
        border around the outside is not in here - it carries the state colour, so it belongs to
        the per-state base and goes on last of all.
        """
        layer = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        f, pad, w = self.frame, self.pad, self.line
        # A rule with a thinner companion a few pixels off it, top and bottom - the doubled
        # divider is most of what makes a green terminal read as a terminal rather than a form.
        gap = max(2, round(4 * self.scale))
        for y, offset in ((self.header.bottom, gap), (self.footer.y, -gap)):
            d.line([f.x + pad, y, f.right - pad, y], fill=(*GREEN_MID, 230), width=w)
            d.line(
                [f.x + pad * 3, y + offset, f.right - pad * 3, y + offset],
                fill=(*GREEN_DIM, 220),
                width=max(1, w // 2),
            )
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
        image = Image.alpha_composite(_to_image(*self._backdrop), self._chrome)

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
        panel sitting at STANDBY claims to be filming the room.
        """
        return recording and state not in (IDLE, ERROR)

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
    ) -> np.ndarray:
        """Draw the whole chrome for this frame and return it as an RGBA numpy array."""
        halo = HALOS.get(state, GREEN_DIM)
        layer = self._base(state, recording).copy()
        d = ImageDraw.Draw(layer)

        self._draw_readouts(d, halo, level, elapsed, self._taping(state, recording))
        self._draw_caption(d, state, halo, detail)
        if state not in (IDLE, ERROR, STOPPING):  # the eye is open and the ring is breathing
            self._draw_ring(d, _mix(_mix(SCREEN, halo, TAB_LIVE), halo, RING_MIX), level)
        if pressed is not None:
            # Redrawn over the tab the base has at rest: an inverted cell is the only feedback
            # a screen with no travel can give, and it lasts a handful of frames.
            cell = self._cells.get(pressed)
            if cell is not None:
                self._draw_tab(d, cell, pressed, state, halo, pressed=True)
                if pressed == "eye" and state not in (IDLE, ERROR, STOPPING):
                    self._draw_ring(d, _mix(halo, INK, RING_MIX), level)
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
        colour = (*GREEN_DIM, 255) if elapsed is None else (*GREEN, 255)
        self._text(d, clock_right, cy, clock, self.font_read, colour, align="r")

        seg_w, seg_h, gap = self._seg
        lit = round(max(0.0, min(1.0, level)) * METER_SEGMENTS)
        x = self._meter_x(meter_right)
        for index in range(METER_SEGMENTS):
            x0 = x + index * (seg_w + gap)
            d.rectangle(
                [x0, cy - seg_h / 2, x0 + seg_w, cy + seg_h / 2],
                fill=(*halo, 255) if index < lit else (*_mix(SCREEN, GREEN_DIM, METER_OFF), 255),
            )

    def _draw_caption(
        self, d: ImageDraw.ImageDraw, state: str, halo: tuple, detail: str
    ) -> None:
        """One line of plain English along the bottom of the picture, on its own dark slab.

        The slab is not decoration: this text sits on the live camera, and white-on-anything is
        a coin toss. It is also where an error actually says what went wrong, which the old
        chrome could only render as a red rim.
        """
        text = CAPTIONS.get(state) or detail
        if not text:
            return  # the strip already says the mode; saying it twice is not a caption
        font = self.font_caption
        pad = self.pad
        # Clear of the bottom-left corner tick, which would otherwise run under the slab.
        x = self.viewport.x + pad + round(34 * self.scale)
        limit = self.viewport.right - pad - x - round(30 * self.scale)
        text = self._elide(f"› {text}", font, limit)
        width = font.getlength(text)
        height = round(24 * self.scale)
        y = self.footer.y - round(12 * self.scale) - height / 2
        inset = round(8 * self.scale)
        d.rectangle(
            [x, y - height / 2, x + width + inset * 2, y + height / 2], fill=(*SCREEN, PLATE_ALPHA)
        )
        colour = halo if state == ERROR else GREEN
        self._text(d, x + inset, y, text, font, (*colour, 245))

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
        dark strip; while a session is up the SESSION tab carries a lit edge and a tinted cell,
        the way a selected tab does; and under a thumb any tab inverts completely, which is the
        only feedback a touchscreen with no travel can give.
        """
        live = name == "eye" and state not in (IDLE, ERROR)
        if pressed:
            fill, glyph, label = (*halo, 255), INK, INK
        elif live:
            # Translucent, unlike a press: the row is see-through now, and a selected tab that
            # blacked out a third of it would put back the bar this layout just took away.
            fill, glyph, label = (*_mix(SCREEN, halo, TAB_LIVE), TAB_LIVE_ALPHA), halo, GREEN
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
                corners=(False, False, name == "eye", name == "shutter"),
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
        elif name == "admin":
            self._glyph_gear(d, cx, gy, radius, glyph)
        else:
            self._glyph_eye(d, cx, gy, radius, glyph, state)
        self._text(
            d,
            cx,
            cell.bottom - round(23 * self.scale),
            TAB_LABELS[name],
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

    def _glyph_gear(self, d: ImageDraw.ImageDraw, cx: int, cy: int, r: int, c: tuple) -> None:
        """Open the admin page."""
        stroke = max(2, round(3 * self.scale))
        ring = round(r * 0.72)
        d.ellipse([cx - ring, cy - ring, cx + ring, cy + ring], outline=(*c, 255), width=stroke)
        hub = max(2, round(ring * 0.34))
        d.ellipse([cx - hub, cy - hub, cx + hub, cy + hub], fill=(*c, 255))
        tooth = ring * 0.42
        for index in range(8):  # eight teeth read as a gear even at this size
            angle = math.radians(index * 45)
            d.line(
                [
                    cx + ring * math.cos(angle),
                    cy + ring * math.sin(angle),
                    cx + (ring + tooth) * math.cos(angle),
                    cy + (ring + tooth) * math.sin(angle),
                ],
                fill=(*c, 255),
                width=stroke,
            )

    def _draw_ring(self, d: ImageDraw.ImageDraw, colour: tuple, level: float) -> None:
        """The ring around the open eye - the one thing on the panel that moves with your voice.

        Drawn per frame rather than baked with the rest of the tab, for the obvious reason that
        it is the only part of the tab row with anything to say between one frame and the next.
        """
        cx, cy, r = self._glyph_at(self.hitboxes.eye)
        ring = round(r * (1.25 + 0.30 * max(0.0, min(1.0, level))))
        d.ellipse(
            [cx - ring, cy - ring, cx + ring, cy + ring],
            outline=(*colour, 255),
            width=max(1, self.line // 2),
        )

    def _glyph_eye(
        self, d: ImageDraw.ImageDraw, cx: int, cy: int, r: int, c: tuple, state: str
    ) -> None:
        """Start or stop the agent. Closed while it is down, open and watching while it is up."""
        stroke = max(2, round(3 * self.scale))
        if state in (IDLE, ERROR, STOPPING):
            d.arc(
                [cx - r, cy - round(r * 0.8), cx + r, cy + round(r * 0.8)],
                start=15,
                end=165,
                fill=(*c, 255),
                width=stroke,
            )
            return
        ry = round(r * 0.68)
        d.ellipse([cx - r, cy - ry, cx + r, cy + ry], outline=(*c, 255), width=stroke)
        ir = round(r * 0.40)
        d.ellipse([cx - ir, cy - ir, cx + ir, cy + ir], fill=(*c, 255))


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
