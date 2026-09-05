"""The two instruments in the bottom-right corner: the one you set and the one you read.

They replaced a shutter and a microphone, which the button beside the panel now does both of.
What is tested here is the half that is genuinely new - a finger on the glass becoming a level on
the sink - plus the one thing the gauge does besides sit there.

The dials' own drawing is tested in ``test_eye.py``, off rendered frames. This is about where a
touch goes, and about the column that comes up under it.
"""

from __future__ import annotations

import numpy as np
import pytest

from cyclops import kiosk as kiosk_module
from cyclops import overlay

DOWN, UP, MOVE = 1, 4, 0  # cv2.EVENT_LBUTTONDOWN / _LBUTTONUP / _MOUSEMOVE
HELD = 1  # cv2.EVENT_FLAG_LBUTTON


# ------------------------------------------------------------------ where a level lives


def test_the_level_is_where_your_finger_is() -> None:
    """Absolute, not relative. The foot of the column is silence and its head is everything, and
    a finger put halfway up it is asking for half - whatever it was showing when you grabbed."""
    ov = overlay.Overlay(800, 480)
    track = ov.slider
    assert ov.slider_value(track.bottom) == pytest.approx(0.0, abs=0.01)
    assert ov.slider_value(track.y) == pytest.approx(1.0, abs=0.01)
    assert ov.slider_value(track.y + track.h / 2) == pytest.approx(0.5, abs=0.02)


def test_a_finger_off_either_end_is_an_end_and_not_a_wrap() -> None:
    """The knob sits below the foot of its own column, so *every* grab starts off the end of it.
    Clamping is what makes that harmless - it is silence, which is somewhere you drag away from."""
    ov = overlay.Overlay(800, 480)
    knob_y = ov.switches[overlay.VOLUME][1]
    assert knob_y > ov.slider.bottom, "the column runs into the disc it comes out of"
    assert ov.slider_value(knob_y) == 0.0
    assert ov.slider_value(-200) == 1.0


def test_the_column_stands_clear_of_everything_it_would_cover() -> None:
    """It is up only while a finger is on it, but while it is up it is over the picture, and the
    two things on that side of the panel worth not covering are the gauge and his sentence."""
    ov = overlay.Overlay(800, 480)
    gauge_x, _ = ov.switches[overlay.HEAT]
    assert ov.slider.right < gauge_x - ov.btn_r, "the column crosses the gauge"
    assert ov.slider.x > ov.caption_right, "the column lands on the caption"


# ------------------------------------------------------------------ the drag


class _Cues:
    def play(self, name: str) -> None:
        pass

    def stop(self) -> None:
        pass


class _Mixer:
    """A sink that remembers what it was asked for, in place of pactl."""

    def __init__(self) -> None:
        self.levels: list[int] = []
        self.noted: list[int] = []

    def set_level(self, percent: int) -> bool:
        self.levels.append(percent)
        return True

    def request(self, percent: int) -> None:
        self.noted.append(percent)


def _panel(monkeypatch: pytest.MonkeyPatch) -> tuple[kiosk_module.Kiosk, _Mixer]:
    kiosk = object.__new__(kiosk_module.Kiosk)
    kiosk.overlay = overlay.Overlay(800, 480)
    kiosk._eye_down_at = None
    kiosk._menu = False
    kiosk._asleep = False
    kiosk._touched_at = 0.0
    kiosk._pressed = None
    kiosk._press_until = 0.0
    kiosk._turning = False
    kiosk._sliding = False
    kiosk._slide_from = 0.0
    kiosk._wanted = None
    kiosk._volume = 50
    kiosk._cues = _Cues()
    kiosk.opened: list[str] = []
    monkeypatch.setattr(kiosk, "_open_admin", lambda screen: kiosk.opened.append(screen))
    mixer = _Mixer()
    monkeypatch.setattr(kiosk_module, "mixer", mixer)
    return kiosk, mixer


def _knob(kiosk: kiosk_module.Kiosk) -> tuple[int, int]:
    return kiosk.overlay.hitboxes.volume.center


def _at(kiosk: kiosk_module.Kiosk, level: int) -> int:
    """The height on the track that asks for *level*."""
    track = kiosk.overlay.slider
    return round(track.bottom - track.h * level / 100.0)


def test_a_tap_on_the_knob_asks_for_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The column reads absolutely and the knob sits below the foot of it, so a tap that counted
    would be a tap that muted him. The disc lights, and that is the whole of what a tap does."""
    kiosk, mixer = _panel(monkeypatch)
    x, y = _knob(kiosk)
    kiosk._on_mouse(DOWN, x, y, 0, None)
    assert kiosk._pressed == overlay.VOLUME and not kiosk._sliding, "the column came up on a tap"
    kiosk._on_mouse(UP, x, y, 0, None)
    assert mixer.levels == [] and kiosk._volume == 50
    assert kiosk._pressed is None, "the knob stayed lit after the finger came off"


def test_a_press_that_barely_moves_is_still_a_tap(monkeypatch: pytest.MonkeyPatch) -> None:
    """A touchscreen delivers a pixel or two of travel on every press. That is not a drag."""
    kiosk, mixer = _panel(monkeypatch)
    x, y = _knob(kiosk)
    kiosk._on_mouse(DOWN, x, y, 0, None)
    kiosk._on_mouse(MOVE, x + 2, y - (kiosk_module.SLIDE_GRAB_PX - 1), HELD, None)
    assert not kiosk._sliding
    kiosk._on_mouse(UP, x, y, 0, None)
    assert mixer.levels == []


def test_the_column_follows_the_finger_and_lands_on_the_lift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole gesture: grab the knob, drag up the track to the level you want, let go."""
    kiosk, mixer = _panel(monkeypatch)
    x, y = _knob(kiosk)
    kiosk._on_mouse(DOWN, x, y, 0, None)
    seen = []
    for level in (20, 55, 80):
        kiosk._on_mouse(MOVE, x, _at(kiosk, level), HELD, None)
        seen.append(kiosk._wanted)
    assert seen == [20, 55, 80], f"the column did not follow the finger: {seen}"
    assert kiosk._sliding and mixer.levels == [], "the speaker was asked before the finger lifted"

    kiosk._on_mouse(UP, x, _at(kiosk, 80), 0, None)
    assert mixer.levels == [80], "the level you let go on is the level you meant"
    assert mixer.noted == [80], "the page's slider was left where the column was not"
    assert kiosk._volume == 80
    assert not kiosk._sliding and kiosk._wanted is None, "the column stayed up"


def test_the_level_is_snapped_to_the_step_the_page_uses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Five, like the slider on the admin page, so the two cannot disagree about what a level is
    - and so one rung of the column is one setting rather than a place between two."""
    kiosk, _ = _panel(monkeypatch)
    x, y = _knob(kiosk)
    kiosk._on_mouse(DOWN, x, y, 0, None)
    for level in range(0, 101, 3):
        kiosk._on_mouse(MOVE, x, _at(kiosk, level), HELD, None)
        assert kiosk._wanted % overlay.VOLUME_STEP == 0, f"{kiosk._wanted} is off the step"


def test_pactl_is_asked_once_for_a_whole_drag(monkeypatch: pytest.MonkeyPatch) -> None:
    """A subprocess a frame, on a board this layout already spends its corners keeping cool. It
    is also why the grab is free: nothing you drag across on the way is ever heard."""
    kiosk, mixer = _panel(monkeypatch)
    x, y = _knob(kiosk)
    kiosk._on_mouse(DOWN, x, y, 0, None)
    for level in range(0, 101, 5):
        kiosk._on_mouse(MOVE, x, _at(kiosk, level), HELD, None)
    assert mixer.levels == []
    kiosk._on_mouse(UP, x, _at(kiosk, 100), 0, None)
    assert mixer.levels == [100]


def test_a_drag_that_lands_on_the_level_it_started_at_asks_for_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kiosk, mixer = _panel(monkeypatch)
    x, y = _knob(kiosk)
    kiosk._on_mouse(DOWN, x, y, 0, None)
    kiosk._on_mouse(MOVE, x, _at(kiosk, 20), HELD, None)
    kiosk._on_mouse(UP, x, _at(kiosk, 50), 0, None)
    assert mixer.levels == [], "it was already at 50"


def test_a_finger_that_landed_somewhere_else_does_not_open_the_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A drag that started on his face and crossed the corner is not a volume change."""
    kiosk, mixer = _panel(monkeypatch)
    kiosk._on_mouse(DOWN, *kiosk.overlay.hitboxes.eye.center, 0, None)
    kiosk._on_mouse(MOVE, 660, _at(kiosk, 100), HELD, None)
    assert not kiosk._sliding
    kiosk._on_mouse(UP, 660, _at(kiosk, 100), 0, None)
    assert mixer.levels == []


def test_a_release_that_never_arrived_sets_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A press with no release behind it means the release was lost, so nothing was ever let go
    of. Acting on the column then would set a level from a gesture that is over."""
    kiosk, mixer = _panel(monkeypatch)
    x, y = _knob(kiosk)
    kiosk._on_mouse(DOWN, x, y, 0, None)
    kiosk._on_mouse(MOVE, x, _at(kiosk, 100), HELD, None)
    kiosk._on_mouse(DOWN, *kiosk.overlay.hitboxes.eye.center, 0, None)  # no UP in between
    assert not kiosk._sliding and mixer.levels == []
    kiosk._on_mouse(MOVE, x, _at(kiosk, 0), HELD, None)
    assert mixer.levels == [], "a drag off his face set the volume"


def test_a_panel_with_no_sink_under_it_sets_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Off the Pi there is no pactl, so there is no level. A column that fills and changes
    nothing is worse than one that never comes up."""
    kiosk, mixer = _panel(monkeypatch)
    kiosk._volume = None
    x, y = _knob(kiosk)
    kiosk._on_mouse(DOWN, x, y, 0, None)
    kiosk._on_mouse(MOVE, x, _at(kiosk, 100), HELD, None)
    assert not kiosk._sliding
    kiosk._on_mouse(UP, x, _at(kiosk, 100), 0, None)
    assert mixer.levels == []


# ------------------------------------------------------------------ what the column shows


def _rungs(ov: overlay.Overlay, level: int) -> int:
    """How many rungs of the column are lit, counted off a rendered frame."""
    frame = ov.render(state=overlay.IDLE, level=0.0, elapsed=None, phase=10.0, volume=level,
                      temp_c=58.0, pressed=overlay.VOLUME, sliding=True).astype(int)
    track = ov.slider
    column = frame[track.y : track.bottom, track.center[0], :3]
    # Phosphor only. The thumb is white and sits across the top of the stack, so counting by
    # brightness alone would find one lit rung on a column showing silence.
    bright = (column.sum(axis=1) > sum(overlay.GREEN) * 0.9) & (column[:, 0] < 150)
    # One run of lit pixels per rung, so the gaps between them are what makes this a count.
    return int(np.sum(bright[1:] & ~bright[:-1]) + bright[0])


def test_the_column_is_only_up_while_a_finger_is_on_it() -> None:
    """It is over the picture, and the picture is what somebody is holding a camera down a pipe
    to see. It has no business being there a frame longer than the gesture."""
    ov = overlay.Overlay(800, 480)
    shown = dict(state=overlay.IDLE, level=0.0, elapsed=None, phase=10.0, volume=60, temp_c=58.0)
    quiet = ov.render(**shown)  # type: ignore[arg-type]
    track = ov.slider
    band = (slice(track.y, track.bottom), slice(track.x, track.right))
    assert not np.array_equal(quiet[band], ov.render(sliding=True, **shown)[band])  # type: ignore[arg-type]
    assert ov.render(**shown)[band].tobytes() == quiet[band].tobytes()  # type: ignore[arg-type]


@pytest.mark.parametrize("level", [0, 25, 50, 75, 100])
def test_one_rung_lit_per_setting_of_the_level(level: int) -> None:
    """The column is the pod's signal meter stood on end, and it steps in fives because the level
    does: drawn continuously it would show a level the knob cannot actually be left at."""
    ov = overlay.Overlay(800, 480)
    assert _rungs(ov, level) == level // overlay.VOLUME_STEP


# ------------------------------------------------------------------ the gauge


def test_the_gauge_opens_the_screen_the_rest_of_its_numbers_are_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It sits where a switch used to, so it will be tapped. What it opens is the one screen it
    is already the corner of - not the recordings, which is what his face promises."""
    kiosk, _ = _panel(monkeypatch)
    kiosk._on_mouse(DOWN, *kiosk.overlay.hitboxes.heat.center, 0, None)
    assert kiosk.opened == [kiosk_module.SYSTEM_SCREEN]
    assert kiosk._pressed == overlay.HEAT, "nothing acknowledged the tap"
