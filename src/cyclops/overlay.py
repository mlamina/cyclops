"""The kiosk's chrome, drawn with PIL and composited over the live camera frame.

OpenCV can only draw Hershey stroke fonts, which look like a 1980s oscilloscope. Everything
here is rendered into one RGBA layer with PIL instead - real TrueType text, anti-aliased
circles, rounded rectangles, translucent scrims - and handed to :mod:`cyclops.kiosk` as a
numpy array to alpha-blend onto the frame. Geometry doubles as the hit-test map: every
interactive element returns its rectangle, so a tap can be resolved without a second layout.
"""

from __future__ import annotations

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
TEXT = (231, 237, 244)
MUTED = (139, 151, 166)
SCRIM = (12, 15, 20)

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

    eye: Rect
    quit: Rect


class Overlay:
    """Renders the kiosk chrome for a given window size, and remembers where it put things."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        scale = height / 480.0  # the official 7" panel is the reference layout
        self._eye_r = int(52 * scale)
        self.font_status = _load_font(int(26 * scale))
        self.font_hint = _load_font(int(14 * scale))
        self.font_chip = _load_font(int(12 * scale))
        self.hitboxes = self._layout(scale)

    def _layout(self, scale: float) -> Hitboxes:
        """Stack the controls in a rail down the right edge, leaving the picture clear."""
        w, h = self.width, self.height
        self.rail_w = int(132 * scale)
        self.rail_x = w - self.rail_w
        eye_d = self._eye_r * 2
        eye = Rect(
            self.rail_x + (self.rail_w - eye_d) // 2, int(h * 0.30) - eye_d // 2, eye_d, eye_d
        )
        quit_side = int(46 * scale)
        quit = Rect(
            self.rail_x + (self.rail_w - quit_side) // 2,
            h - quit_side - int(20 * scale),
            quit_side,
            quit_side,
        )
        return Hitboxes(eye=eye, quit=quit)

    def render(
        self,
        state: str,
        detail: str,
        level: float,
        flash: float = 0.0,
        pressed: bool = False,
    ) -> np.ndarray:
        """Draw the whole chrome for this frame and return it as an RGBA numpy array."""
        layer = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        accent = ACCENTS.get(state, ACCENTS[IDLE])
        scale = self.height / 480.0

        self._draw_rail(d, scale)
        self._draw_eye(d, state, accent, level, scale, pressed)
        self._draw_status(d, detail, accent, scale)
        self._draw_quit(d, scale)
        if flash > 0.0:
            d.rectangle([0, 0, self.width, self.height], fill=(255, 255, 255, int(190 * flash)))
        return np.asarray(layer)

    # ---- pieces ----

    def _draw_rail(self, d: ImageDraw.ImageDraw, scale: float) -> None:
        """A soft vertical scrim behind the control rail, so only the edge is dimmed."""
        feather = int(46 * scale)
        for i in range(feather):  # fade in from the picture towards the rail
            x = self.rail_x - feather + i
            d.line([(x, 0), (x, self.height)], fill=(*SCRIM, int(150 * (i / feather))))
        d.rectangle(
            [self.rail_x, 0, self.width, self.height], fill=(*SCRIM, 150)
        )
        d.text(
            (self.rail_x + self.rail_w // 2, int(24 * scale)),
            "CYCLOPS",
            font=self.font_chip,
            fill=(*MUTED, 230),
            anchor="mm",
        )

    def _draw_eye(
        self,
        d: ImageDraw.ImageDraw,
        state: str,
        accent: tuple,
        level: float,
        scale: float,
        pressed: bool = False,
    ) -> None:
        box = self.hitboxes.eye
        cx, cy = box.center
        r = self._eye_r

        pad = int(9 * scale)  # button plate, so it reads as something you press
        d.ellipse(
            [box.x - pad, box.y - pad, box.x + box.w + pad, box.y + box.h + pad],
            fill=(*accent, 150) if pressed else (18, 24, 32, 175),
            outline=(*accent, 255 if pressed else 110),
            width=3 if pressed else 1,
        )

        # audio-reactive ring, the same idea as ui.html's --level scaling
        ring = int(r * (1.0 + 0.34 * max(0.0, min(1.0, level))))
        d.ellipse([cx - ring, cy - ring, cx + ring, cy + ring], outline=(*accent, 120), width=2)

        if state in (IDLE, ERROR, STOPPING):  # closed eye: a simple arc, like the .closed path
            d.arc(
                [cx - r, cy - int(r * 0.72), cx + r, cy + int(r * 0.72)],
                start=15,
                end=165,
                fill=(*accent, 210),
                width=max(3, int(6 * scale)),
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

    def _draw_status(
        self, d: ImageDraw.ImageDraw, detail: str, accent: tuple, scale: float
    ) -> None:
        """The state line, wrapped to the rail so it never spills over the picture."""
        cx = self.rail_x + self.rail_w // 2
        y = self.hitboxes.eye.y + self.hitboxes.eye.h + int(16 * scale)
        for line in self._wrap(detail, self.font_hint, self.rail_w - int(16 * scale)):
            d.text((cx, y), line, font=self.font_hint, fill=(*TEXT, 245), anchor="ma")
            y += int(self.font_hint.size * 1.35)
        dot = max(3, int(5 * scale))
        dy = int(24 * scale)
        d.ellipse(
            [cx + int(38 * scale) - dot, dy - dot, cx + int(38 * scale) + dot, dy + dot],
            fill=(*accent, 255),
        )

    def _wrap(self, text: str, font: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
        """Greedy word wrap - PIL has no layout engine, and the rail is narrow."""
        lines: list[str] = []
        current = ""
        for word in text.split():
            trial = f"{current} {word}".strip()
            if current and font.getlength(trial) > max_w:
                lines.append(current)
                current = word
            else:
                current = trial
        if current:
            lines.append(current)
        return lines

    def _draw_quit(self, d: ImageDraw.ImageDraw, scale: float) -> None:
        box = self.hitboxes.quit
        cx, cy = box.center
        arm = int(box.w * 0.22)
        d.ellipse(
            [box.x, box.y, box.x + box.w, box.y + box.h], fill=(20, 26, 34, 140)
        )
        for dx, dy in (((-arm, -arm), (arm, arm)), ((-arm, arm), (arm, -arm))):
            d.line(
                [cx + dx[0], cy + dx[1], cx + dy[0], cy + dy[1]],
                fill=(*MUTED, 220),
                width=max(2, int(2.5 * scale)),
            )


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
