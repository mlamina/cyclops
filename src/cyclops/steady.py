"""Keeps the live picture still while the hand holding Cyclops is not.

Cyclops is held, so every picture it shows shakes with the hand. This measures how far the
picture moved between two frames, keeps a smoothed path of where it has been going, and cuts
the window of the raw frame that follows the smoothed path instead of the shaking one. What is
cut out is the shake.

The room to move that window in is the picture the panel was already throwing away - the
strips :func:`cyclops.overlay.fit_to_window` trims to get from 16:9 to the panel's 5:3, and the
band under the metal frame's fully opaque edge - plus a 4 % crop, which is the only thing this
costs that anybody can see. REACH is how far that goes, and it was set by the gap test in
tests/test_steady.py rather than by arithmetic: at full reach the edge fill lands in the discarded
strip or under opaque metal and nowhere else.

It owns the measurement, the smoothing and the cutting. It never writes into a raw frame: the
photos the model is shown are the raw frames, whole and unshifted, and the windows handed out
here are views of them (or, at the very edge of the reach, copies).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .webcam import FRAME_RATE, FRAME_WIDTH

# The one knob: how long the smoothed path takes to catch up with the hand. Longer is steadier
# and more rubbery.
TIME_CONSTANT_S = 0.5
LEAK = math.exp(-1.0 / (TIME_CONSTANT_S * FRAME_RATE))  # the shift kept from one frame to the next

LIVE_CROP = 0.96  # the panel's window: 4 % of each edge given up to make up/down reach
FILM_CROP = 0.875  # a CAMERA recording's window, 1120x630: tight enough that no fill is ever in it
# How far the window may move, in camera pixels of a 1280x720 frame. Sideways it is the strip
# fit_to_window discards, the 4 % crop and the opaque band; up and down only the last two.
REACH_X, REACH_Y = 79, 29
DRAIN_PX = 5.0  # the most a frame nobody could track may move the picture back towards centre

CORNERS = 100  # tracked per frame, on the 320x180 grey the focus score already made
CORNER_QUALITY = 0.01
CORNER_GAP = 8
MIN_TRACKED = 10  # fewer agreeing corners than this and the frame is not believed
# A step past this share of the width in one frame is a whip pan, a hand over the lens or a new
# scene - not shake, and not something to hold the picture against.
MAX_STEP = 0.1


def steadiable(width: int, height: int) -> bool:
    """16:9 and big enough to track: the Camera Module and the C920. Not the 4:3 endoscope."""
    return width * 9 == height * 16 and width >= FRAME_WIDTH // 4


def reach(width: int) -> np.ndarray:
    """How far the window may move for a frame this wide, in its own pixels."""
    return np.array([REACH_X, REACH_Y], float) * width / FRAME_WIDTH


def window(frame: np.ndarray, shift, crop: float, fill=None) -> np.ndarray:
    """A crop-sized window of *frame*, its centre moved by *shift*, edges repeated past the frame.

    A view whenever it stays inside the frame, which is nearly always. *fill* paints what lies
    past the edge one colour instead of repeating it - the gap tests' sentinel, never the panel's.
    """
    h, w = frame.shape[:2]
    ww, wh = round(w * crop), round(h * crop)
    x0 = (w - ww) // 2 + round(shift[0])
    y0 = (h - wh) // 2 + round(shift[1])
    x1, y1 = x0 + ww, y0 + wh
    if x0 >= 0 and y0 >= 0 and x1 <= w and y1 <= h:
        return frame[y0:y1, x0:x1]
    inner = frame[max(y0, 0) : min(y1, h), max(x0, 0) : min(x1, w)]
    border = (max(0, -y0), max(0, y1 - h), max(0, -x0), max(0, x1 - w))
    if fill is None:
        return cv2.copyMakeBorder(inner, *border, cv2.BORDER_REPLICATE)
    return cv2.copyMakeBorder(inner, *border, cv2.BORDER_CONSTANT, value=fill)


def live(frame: np.ndarray, shift) -> np.ndarray:
    """What the panel, the SCREEN recording and the admin page are shown."""
    h, w = frame.shape[:2]
    return window(frame, shift, LIVE_CROP) if steadiable(w, h) else frame


def film(frame: np.ndarray, shift) -> np.ndarray:
    """What a CAMERA recording is of: a fixed-size window, so the encoder's size holds."""
    h, w = frame.shape[:2]
    return window(frame, shift, FILM_CROP) if steadiable(w, h) else frame


class Steady:
    """The shift to cut each frame's window at. One per open of the camera: a reopen starts at zero.

    The state is the shift alone. Keeping a smoothed path S of the picture's path P and cutting
    at P - S comes to the same thing as a leaky sum of the motion - each frame the shift gains
    that frame's motion and keeps LEAK of the total - and a leaky sum cannot grow without bound
    over a long pan the way two cumulative paths do.
    """

    def __init__(self) -> None:
        self._prev: np.ndarray | None = None
        self._shift = np.zeros(2)

    def update(self, small: np.ndarray, width: int, height: int) -> tuple[float, float]:
        """Take the next frame's 320x180 grey and return where to centre its window, in frame px."""
        if not steadiable(width, height):
            return 0.0, 0.0
        step = self._measure(small)
        self._prev = small
        limit = reach(width)
        if step is not None:
            step = step * width / small.shape[1]
            if np.abs(step).max() > MAX_STEP * width:
                step = None
        if step is None:
            # Counted as no motion, so the shift leaks away rather than snapping back - and at
            # no more than DRAIN_PX a frame, or losing track at full reach would be a lurch.
            drained = np.clip(self._shift * LEAK, -limit, limit)
            self._shift = self._shift + np.clip(drained - self._shift, -DRAIN_PX, DRAIN_PX)
        else:
            # Past the reach the smoothed path is dragged along, which is the clamp: the picture
            # follows the hand rather than showing an edge or falling further behind.
            self._shift = np.clip((self._shift + step) * LEAK, -limit, limit)
        return float(self._shift[0]), float(self._shift[1])

    def _measure(self, small: np.ndarray) -> np.ndarray | None:
        """How far the middle of the picture moved since the last frame, or None if unknowable.

        Corners tracked, then a similarity fitted to them with RANSAC so a hand or a tool moving
        through the picture is outvoted by the bench behind it. Only the centre's displacement is
        used - no roll correction.
        """
        prev = self._prev
        if prev is None or prev.shape != small.shape:
            return None
        corners = cv2.goodFeaturesToTrack(prev, CORNERS, CORNER_QUALITY, CORNER_GAP)
        if corners is None or len(corners) < MIN_TRACKED:
            return None
        moved, found, _ = cv2.calcOpticalFlowPyrLK(prev, small, corners, None)
        ok = found.ravel() == 1
        if ok.sum() < MIN_TRACKED:
            return None
        fit, inliers = cv2.estimateAffinePartial2D(corners[ok], moved[ok])
        if fit is None or inliers is None or inliers.sum() < MIN_TRACKED:
            return None
        centre = np.array([(small.shape[1] - 1) / 2, (small.shape[0] - 1) / 2, 1.0])
        return fit @ centre - centre[:2]

