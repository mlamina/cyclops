"""The stabilizer: how far it may move the picture, and what it hands to whom.

The reach is the claim everything else leans on - fill past the raw frame's edge must only ever
land where the panel does not show it - so it is checked the way the panel would show it: through
fit_to_window and composite over the real chrome, with the fill painted a sentinel colour.
"""

from __future__ import annotations

import functools

import cv2
import numpy as np
import pytest

from cyclops import camera, overlay, steady
from cyclops.webcam import FRAME_RATE

W, H = 1280, 720
SMALL = (camera.FOCUS_WIDTH, camera.FOCUS_HEIGHT)
TO_SMALL = W / camera.FOCUS_WIDTH  # camera px per thumbnail px
GREY, SENTINEL = 128, (255, 0, 255)
REACH = steady.reach(W)
DIRECTIONS = [(sx, sy) for sx in (-1, 0, 1) for sy in (-1, 0, 1) if (sx, sy) != (0, 0)]


@functools.cache
def texture() -> np.ndarray:
    """A bench with plenty of corners on it, bigger than any window cut from it."""
    noise = np.random.default_rng(0).integers(0, 255, (400, 560), np.uint8)
    return cv2.GaussianBlur(noise, (0, 0), 2)


def thumb(dx: float, dy: float) -> np.ndarray:
    """The 320x180 grey of a camera that has moved (dx, dy) camera px from the start."""
    t = texture()
    centre = (t.shape[1] / 2 + dx / TO_SMALL, t.shape[0] / 2 + dy / TO_SMALL)
    return cv2.getRectSubPix(t, SMALL, centre)


def run(path) -> np.ndarray:
    """Feed a camera path (None: a frame nothing can be tracked in) and collect the shifts."""
    held = steady.Steady()
    flat = np.full(SMALL[::-1], GREY, np.uint8)
    return np.array([held.update(flat if p is None else thumb(*p), W, H) for p in path])


@pytest.fixture(scope="module")
def chrome() -> np.ndarray:
    return overlay.Overlay(800, 480).render(phase=12.0, state=overlay.LISTENING, level=0.3,
                                            elapsed=12.0)


def panel(frame: np.ndarray, chrome: np.ndarray) -> np.ndarray:
    return overlay.composite(overlay.fit_to_window(frame, 800, 480), chrome)


def test_a_jolt_never_moves_the_picture_past_the_reach() -> None:
    path = [(0, 0)] * 3 + [(100, 100)] * 6 + [(0, 0)] * 6
    shifts = run(path)
    assert (np.abs(shifts) <= REACH + 1e-9).all()
    assert np.allclose(np.abs(shifts).max(0), REACH), "the jolt should have used the whole reach"


def test_at_full_reach_no_fill_shows_through_the_frame(chrome) -> None:
    grey = np.full((H, W, 3), GREY, np.uint8)
    clean = panel(steady.live(grey, (0, 0)), chrome)
    shown = []
    for step in (0, 1):  # at the reach nothing, and one pixel past it something, or it is slack
        seen = 0
        for sx, sy in DIRECTIONS:
            shift = (sx * (REACH[0] + step), sy * (REACH[1] + step))
            filled = steady.window(grey, shift, steady.LIVE_CROP, fill=SENTINEL)
            seen += int((panel(filled, chrome) != clean).any(axis=2).sum())
        shown.append(seen)
    assert shown[0] == 0, f"{shown[0]} panel pixels of edge fill visible at full reach"
    assert shown[1] > 0, "a pixel past the reach still hid the fill: the reach is short"


def test_a_camera_recording_frame_never_contains_fill() -> None:
    grey = np.full((H, W, 3), GREY, np.uint8)
    for sx, sy in DIRECTIONS:
        shot = steady.window(grey, (sx * REACH[0], sy * REACH[1]), steady.FILM_CROP, fill=SENTINEL)
        assert shot.shape == (630, 1120, 3)
        assert (shot == GREY).all(), f"fill in the recording at {(sx, sy)}"


def test_a_steady_pan_is_followed_once_it_stops() -> None:
    pan = np.linspace(-300, 300, 2 * FRAME_RATE)  # 300 camera px/s on both axes, for two seconds
    hold = round(1.5 * FRAME_RATE)
    shifts = run([(p, p) for p in pan] + [(300, 300)] * hold)
    assert np.allclose(np.abs(shifts[len(pan) - 1]), REACH), "the pan should have used the reach"
    assert (np.abs(shifts[-1]) < 5).all(), f"still {shifts[-1]} off centre 1.5 s after the pan"


def test_losing_track_drains_the_shift_without_a_lurch() -> None:
    path = [(0, 0)] * 3 + [(100, 60)] * 4 + [None] * (3 * FRAME_RATE)
    shifts = run(path)
    assert np.abs(shifts[6]).max() > 20, "the jolt should have left something to drain"
    assert np.abs(np.diff(shifts[6:], axis=0)).max() <= steady.DRAIN_PX + 1e-9
    assert np.abs(shifts[-1]).max() < 1.0, "and it drains away rather than sticking"


class _Frames:
    """A camera that delivers a jolted bench once and then stops the reader.

    *before* maps a frame's index to something to do just before that frame is read.
    """

    def __init__(self, source: camera.CameraSource, path, before=None) -> None:
        self._source, self._frames = source, [self._frame(*p) for p in path]
        self._before = before or {}
        self.sent = []

    @staticmethod
    def _frame(dx: float, dy: float) -> np.ndarray:
        grey = cv2.resize(thumb(dx, dy), (W, H), interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

    def read(self):
        if len(self.sent) in self._before:
            self._before[len(self.sent)]()
        if not self._frames:
            self._source._stop.set()
            return False, None
        self.sent.append(self._frames.pop(0))
        return True, self.sent[-1]


def test_photos_are_raw_while_the_live_picture_is_shifted(tmp_path) -> None:
    source = camera.CameraSource()
    source.steady_note = tmp_path / "steady"  # never switched: on
    cap = _Frames(source, [(0, 0), (0, 0), (40, 20)])
    source._read_loop(cap, source._generation)

    live, photo = source.frame(), source.snapshot()
    centred = steady.window(cap.sent[-1], (0, 0), steady.LIVE_CROP)
    assert live.shape == centred.shape and (live != centred).any(), "the live picture is not moved"
    assert any(photo is sent for sent in cap.sent), "the photo must be a raw frame, untouched"


def test_the_switch_lands_mid_stream_and_outlives_a_restart(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(steady, "NOTE_POLL_S", 0.0)  # look every frame, not twice a second
    note = tmp_path / "steady"
    steady.request(False, note)  # switched off before this process was started
    source = camera.CameraSource()
    source.steady_note = note
    off = {}

    def switch_on() -> None:
        off["live"], off["film"] = source.frame(), source.film.frame()
        steady.request(True, note)

    jolt = [(0, 0), (0, 0), (40, 20)]
    cap = _Frames(source, jolt + jolt, before={len(jolt): switch_on})
    source._read_loop(cap, source._generation)

    raw = cap.sent[len(jolt) - 1]
    assert off["live"] is raw and off["film"] is raw, "off, everything gets the raw frame, whole"
    live, centred = source.frame(), steady.window(cap.sent[-1], (0, 0), steady.LIVE_CROP)
    assert live.shape == centred.shape and (live != centred).any(), "on again, it is steadied"
