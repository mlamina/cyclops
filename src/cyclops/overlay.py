"""The kiosk's chrome, drawn with PIL and composited over the live camera frame.

OpenCV can only draw Hershey stroke fonts, which look like a 1980s oscilloscope. Everything
here is rendered into one RGBA layer with PIL instead - real TrueType text, anti-aliased
circles, a numpy-built edge halo - and handed to :mod:`cyclops.kiosk` as a numpy array to
alpha-blend onto the frame. Geometry doubles as the hit-test map: every interactive element
returns its rectangle, so a tap can be resolved without a second layout.

The layout is deliberately tiny: the picture runs full-bleed to all four edges, a rim of light
around those edges carries the session state, and there are exactly two controls, one in each
bottom corner - shutter on the left, session on the right - both sized for a thumb in a glove.
A third, smaller disc sits in the free top-left corner and opens the admin page; it is drawn
half the size of the others so it reads as secondary and never competes for the thumb.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Same state vocabulary and accent colours as ui.html, so both front-ends read alike.
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
ACCENTS = {
    IDLE: (70, 80, 92),
    STARTING: (90, 160, 255),
    STOPPING: (150, 160, 176),
    CONNECTING: (90, 160, 255),
    LISTENING: (55, 224, 196),
    SPEAKING: (139, 157, 255),
    LOOKING: (255, 184, 77),
    SEARCHING: (120, 210, 255),
    ERROR: (255, 93, 93),
}
# The halo answers one question only - is the agent up? - so the seven states collapse onto
# three colours, plus red for a session that fell over. Anything finer stays on the eye button,
# which keeps its own per-state accent above.
OFF, CONNECTING_HALO, ON = ACCENTS[IDLE], ACCENTS[CONNECTING], ACCENTS[LISTENING]
HALOS = {
    IDLE: OFF,
    STARTING: CONNECTING_HALO,
    STOPPING: CONNECTING_HALO,  # disconnecting is the same transition, run backwards
    CONNECTING: CONNECTING_HALO,
    LISTENING: ON,
    SPEAKING: ON,
    LOOKING: ON,
    SEARCHING: ON,
    ERROR: ACCENTS[ERROR],
}
TEXT = (231, 237, 244)
PLATE = (18, 24, 32, 175)  # smoked glass behind both buttons
RECORD_DOT = (255, 93, 93)

HALO_CORE = 0.008  # fraction of the height held at full brightness, hard against the edge
HALO_FALLOFF = 0.038  # and how far the light reaches inwards before it is gone
HALO_PEAK = 0.85  # alpha at the very edge; the picture keeps the other 95% of itself
BUTTON = 0.30  # button diameter as a fraction of the screen height
ADMIN_BUTTON = 0.18  # the admin disc is smaller: still a comfortable tap, clearly secondary
MARGIN = 0.046  # and how far they sit off the corner

_FONT_CANDIDATES = (
    # macOS
    "/System/Library/Fonts/SFNSDisplay.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    # Raspberry Pi OS / Debian
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/piboto/PibotoCondensed-Regular.ttf",
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


class Overlay:
    """Renders the kiosk chrome for a given window size, and remembers where it put things."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        scale = height / 480.0  # the official 7" panel is the reference layout
        self.scale = scale
        self._button_d = int(BUTTON * height)
        self._admin_d = int(ADMIN_BUTTON * height)
        self.font_timer = _load_font(int(34 * scale))
        self._alpha = halo_alpha(width, height)
        self._halos: dict[tuple[int, int, int], Image.Image] = {}
        self.hitboxes = self._layout()

    def _layout(self) -> Hitboxes:
        """Two thumb discs along the bottom, a small one top left. The rest is picture.

        Top *right* is not free - the session timer lives there - so the admin disc takes the
        opposite corner, where nothing else has ever been drawn.
        """
        side = self._button_d
        margin = int(MARGIN * self.height)
        y = self.height - side - margin
        return Hitboxes(
            shutter=Rect(margin, y, side, side),
            eye=Rect(self.width - side - margin, y, side, side),
            admin=Rect(margin, margin, self._admin_d, self._admin_d),
        )

    def _halo(self, colour: tuple[int, int, int]) -> Image.Image:
        """The edge light in one colour, built once and copied per frame.

        Rebuilding the ramp every frame costs a few milliseconds of a Pi's render budget for a
        picture that only changes when the session does, so each colour is kept.
        """
        cached = self._halos.get(colour)
        if cached is None:
            rgba = np.zeros((self.height, self.width, 4), dtype=np.uint8)
            rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2] = colour
            rgba[:, :, 3] = (self._alpha * 255.0).astype(np.uint8)
            cached = self._halos[colour] = Image.fromarray(rgba, "RGBA")
        return cached

    def render(
        self,
        state: str,
        level: float,
        elapsed: float | None = None,
        recording: bool = False,
        flash: float = 0.0,
        pressed: str | None = None,
    ) -> np.ndarray:
        """Draw the whole chrome for this frame and return it as an RGBA numpy array."""
        layer = self._halo(HALOS.get(state, OFF)).copy()  # the halo is the backmost chrome
        d = ImageDraw.Draw(layer)
        accent = ACCENTS.get(state, ACCENTS[IDLE])

        self._draw_admin(d, pressed == "admin")
        self._draw_shutter(d, pressed == "shutter")
        self._draw_eye(d, state, accent, level, pressed == "eye")
        if elapsed is not None:
            self._draw_timer(d, elapsed, recording)
        if flash > 0.0:
            d.rectangle([0, 0, self.width, self.height], fill=(255, 255, 255, int(190 * flash)))
        return np.asarray(layer)

    # ---- pieces ----

    def _plate(
        self, d: ImageDraw.ImageDraw, box: Rect, accent: tuple, pressed: bool
    ) -> None:
        """The glass disc both controls sit on, lit up while your thumb is still on it."""
        d.ellipse(
            [box.x, box.y, box.x + box.w, box.y + box.h],
            fill=(*accent, 150) if pressed else PLATE,
            outline=(*accent, 255 if pressed else 130),
            width=max(2, int(3 * self.scale)) if pressed else max(1, int(2 * self.scale)),
        )

    def _draw_shutter(self, d: ImageDraw.ImageDraw, pressed: bool) -> None:
        """Bottom left: take a photo now. A six-bladed aperture, swept the one way round."""
        box = self.hitboxes.shutter
        cx, cy = box.center
        self._plate(d, box, TEXT, pressed)

        ring = int(box.w * 0.28)
        stroke = max(2, int(4 * self.scale))
        d.ellipse([cx - ring, cy - ring, cx + ring, cy + ring], outline=(*TEXT, 235), width=stroke)
        for blade in range(6):
            outer = math.radians(blade * 60)
            inner = outer + math.radians(60)  # the sweep is what makes it read as an iris
            d.line(
                [
                    cx + ring * math.cos(outer),
                    cy + ring * math.sin(outer),
                    cx + ring * 0.34 * math.cos(inner),
                    cy + ring * 0.34 * math.sin(inner),
                ],
                fill=(*TEXT, 235),
                width=stroke,
            )

    def _draw_admin(self, d: ImageDraw.ImageDraw, pressed: bool) -> None:
        """Top left: open the admin page. A gear, on the same glass disc as the others."""
        box = self.hitboxes.admin
        cx, cy = box.center
        self._plate(d, box, TEXT, pressed)

        ring = int(box.w * 0.24)
        stroke = max(2, int(3 * self.scale))
        d.ellipse([cx - ring, cy - ring, cx + ring, cy + ring], outline=(*TEXT, 235), width=stroke)
        hub = max(2, int(ring * 0.34))
        d.ellipse([cx - hub, cy - hub, cx + hub, cy + hub], fill=(*TEXT, 235))
        tooth = ring * 0.42
        for index in range(8):  # eight teeth read as a gear even at 86 px
            angle = math.radians(index * 45)
            d.line(
                [
                    cx + ring * math.cos(angle),
                    cy + ring * math.sin(angle),
                    cx + (ring + tooth) * math.cos(angle),
                    cy + (ring + tooth) * math.sin(angle),
                ],
                fill=(*TEXT, 235),
                width=stroke,
            )

    def _draw_eye(
        self,
        d: ImageDraw.ImageDraw,
        state: str,
        accent: tuple,
        level: float,
        pressed: bool,
    ) -> None:
        """Bottom right: start or stop the agent. Closed while it is down, open while it is up."""
        box = self.hitboxes.eye
        cx, cy = box.center
        self._plate(d, box, accent, pressed)
        r = box.w // 2 - int(11 * self.scale)

        # audio-reactive ring, the same idea as ui.html's --level scaling
        ring = int(r * (1.0 + 0.34 * max(0.0, min(1.0, level))))
        d.ellipse([cx - ring, cy - ring, cx + ring, cy + ring], outline=(*accent, 120), width=2)

        if state in (IDLE, ERROR, STOPPING):  # closed eye: a simple arc, like the .closed path
            d.arc(
                [cx - r, cy - int(r * 0.72), cx + r, cy + int(r * 0.72)],
                start=15,
                end=165,
                fill=(*accent, 210),
                width=max(3, int(6 * self.scale)),
            )
            return

        ry = int(r * 0.70)  # open eye: almond sclera, iris, pupil, glint
        d.ellipse(
            [cx - r, cy - ry, cx + r, cy + ry], fill=(15, 22, 32, 205), outline=(*accent, 150)
        )
        ir = int(r * 0.44)
        d.ellipse([cx - ir, cy - ir, cx + ir, cy + ir], fill=(*accent, 255))
        pr = int(r * 0.19)
        d.ellipse([cx - pr, cy - pr, cx + pr, cy + pr], fill=(5, 8, 12, 255))
        gr = max(2, int(r * 0.09))
        gx, gy = cx - int(r * 0.14), cy - int(r * 0.16)
        d.ellipse([gx - gr, gy - gr, gx + gr, gy + gr], fill=(255, 255, 255, 220))

    def _draw_timer(self, d: ImageDraw.ImageDraw, elapsed: float, recording: bool) -> None:
        """How long this session has been up, top right - and whether it is being recorded."""
        whole = int(elapsed)
        text = f"{whole // 60:02d}:{whole % 60:02d}"
        margin = int(MARGIN * self.height)
        x, y = self.width - margin, margin
        d.text((x, y), text, font=self.font_timer, fill=(*TEXT, 245), anchor="ra")
        if not recording:
            return
        dot = max(3, int(6 * self.scale))
        dx = x - int(self.font_timer.getlength(text)) - int(16 * self.scale)
        dy = y + int(self.font_timer.size * 0.52)
        d.ellipse([dx - dot, dy - dot, dx + dot, dy + dot], fill=(*RECORD_DOT, 255))


def composite(frame_bgr: np.ndarray, rgba: np.ndarray) -> np.ndarray:
    """Alpha-blend an RGBA overlay onto a BGR frame, in place-ish, without touching PIL again."""
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    rgb = rgba[:, :, :3][:, :, ::-1].astype(np.float32)  # RGB -> BGR to match the frame
    blended = frame_bgr.astype(np.float32) * (1.0 - alpha) + rgb * alpha
    return blended.astype(np.uint8)


def fit_to_window(frame_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    """Centre-crop to the window's aspect, then scale - so faces keep their proportions.

    The webcam is 16:9 and the official Pi panel is 5:3; stretching one to the other makes
    everyone look wrong, so the sides get trimmed instead.
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
    interp = cv2.INTER_AREA if frame_bgr.shape[0] > height else cv2.INTER_LINEAR
    return cv2.resize(frame_bgr, (width, height), interpolation=interp)


def mirror(frame_bgr: np.ndarray) -> np.ndarray:
    """Flip horizontally so the preview behaves like a mirror, which is what people expect."""
    return frame_bgr[:, ::-1]


def platform_font_note() -> str:
    """Which font face got picked - useful when the Pi and the Mac disagree."""
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            return path
    return f"PIL default bitmap font (no TrueType found on {sys.platform})"
