"""How far in the module looks, and the tap on the reticle that changes it.

Three framings, and each one is a different *sensor mode* rather than a crop of the same frame -
so switching is a camera stopped and started, about a second of it. Most of what is pinned here
is that second: it is long enough for the wrong thing to be visible (the "no camera" card, in the
middle of a deliberate gesture), long enough that the panel has to say something while it passes,
and long enough for a device that hates being restarted to be restarted for no reason at all (a
webcam, which has one lens and one angle, and which answers the tap by doing nothing).

No camera anywhere in here. The capture is a stand-in, the same trick ``test_camera_stop.py``
plays, and the panel is the one ``test_dials.py`` builds by hand.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from cyclops import camera as camera_module
from cyclops import kiosk as kiosk_module
from cyclops import overlay, webcam
from cyclops.camera import CameraSource

DOWN = 1  # cv2.EVENT_LBUTTONDOWN


class _Cues:
    def __init__(self) -> None:
        self.played: list[str] = []

    def play(self, name: str, *, loop: bool = False) -> None:
        self.played.append(name)


class _Cap:
    """A capture that always has a frame, and never blocks."""

    def __init__(self) -> None:
        self.frame = np.zeros((16, 16, 3), np.uint8)

    def read(self) -> tuple[bool, np.ndarray]:
        return True, self.frame

    def release(self) -> None:
        pass


def _module_camera() -> CameraSource:
    """A source that believes it has the CSI module open, without opening anything."""
    source = CameraSource()
    source._index = webcam.RPICAM
    return source


def _panel(camera: CameraSource) -> kiosk_module.Kiosk:
    kiosk = object.__new__(kiosk_module.Kiosk)
    kiosk.overlay = overlay.Overlay(800, 480)
    kiosk.camera = camera
    kiosk._cues = _Cues()
    kiosk._eye_down_at = None
    kiosk._menu = False
    kiosk._asleep = False
    kiosk._turning = False
    kiosk._touched_at = 0.0
    kiosk._barge_margin = None  # INTERRUPT off: the picture around the reticle is the barge
    return kiosk


def test_tapping_the_reticle_walks_the_framings_and_comes_back() -> None:
    """The mark that says where the frame is, pressed, asks for the next one - and the third tap
    returns to where it started, because a control you can only walk forwards through is one you
    have to walk all the way round anyway."""
    source = _module_camera()
    kiosk = _panel(source)
    x, y = kiosk.overlay.hitboxes.framing.center
    seen = []
    for _ in range(len(webcam.FRAMINGS) + 1):
        kiosk._on_mouse(DOWN, x, y, 0, None)
        seen.append(source.framing)
    assert seen == [*list(webcam.FRAMINGS)[1:], webcam.WIDE, list(webcam.FRAMINGS)[1]]


def test_a_webcam_is_left_alone() -> None:
    """Only the module has framings. Anything else is one lens at one angle, and answering the
    tap by restarting it would be the exact move that wedges a C920 - for no change at all."""
    source = CameraSource()
    source._index = 0
    assert source.cycle_framing() is None
    assert source.framing == webcam.DEFAULT_FRAMING


def test_the_picture_stays_up_while_the_framing_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    """A framing change is a camera stopped and started, and the frames from before it are kept
    on purpose: dropping them is how the panel comes to say "No camera found" for the second it
    takes a gesture to land."""
    reopening = threading.Event()
    opened: list[str] = []

    def fake_open(preferred: int | None, framing: str = webcam.DEFAULT_FRAMING):
        opened.append(framing)
        if len(opened) > 1:
            reopening.set()  # held open until the test has looked at the panel
            assert released.wait(2.0), "the test never let the second open finish"
        return _Cap(), webcam.RPICAM

    released = threading.Event()
    monkeypatch.setattr(camera_module, "open_camera", fake_open)
    source = CameraSource()
    source.start()
    try:
        source.wait_for_frame(2.0)
        source.cycle_framing()
        assert reopening.wait(2.0), "the reader never stood down for the new framing"
        assert source.frame() is not None, "the panel was blanked mid-gesture"
    finally:
        released.set()
        source.stop()
    assert opened[:2] == [webcam.WIDE, list(webcam.FRAMINGS)[1]]


def test_the_name_is_on_the_panel_and_then_is_not() -> None:
    """A tap changes a picture that takes a second to arrive, so the panel says what was asked
    for while it waits - in the middle, where the reticle keeps the glass clear - and then gives
    the middle back. What it says is not this test's business; that it is there, and gone, is."""
    ov = overlay.Overlay(800, 480)
    shown = dict(state=overlay.LISTENING, level=0.3, phase=4.0)
    bare = ov.render(**shown)
    up = ov.render(framing=webcam.NARROW, framing_fade=1.0, **shown)
    faded = ov.render(framing=webcam.NARROW, framing_fade=0.0, **shown)
    middle = (slice(200, 280), slice(340, 460))  # the reticle's own clear span
    assert not np.array_equal(bare[middle], up[middle]), "nothing was said in the middle"
    assert np.array_equal(bare[middle], faded[middle]), "the name outstayed its second"
