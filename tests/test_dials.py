"""The two instruments in the bottom-right corner: the one you set and the one you read.

They replaced a shutter and a microphone, which the button beside the panel now does both of.
What is tested here is the half that is genuinely new - a finger on the glass becoming a level on
the sink - plus the one thing the gauge does besides sit there.

The dials' own drawing is tested in ``test_eye.py``, off rendered frames. This is about where a
touch goes, and about the column that comes up under it.
"""

from __future__ import annotations

import threading

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
    def __init__(self) -> None:
        self.played: list[str] = []

    def play(self, name: str, *, loop: bool = False) -> None:
        self.played.append(name)

    def stop(self) -> None:
        pass


class _Controller:
    """The session, in place of one: only the switch the handover actually reaches."""

    def __init__(self) -> None:
        self.on_air: list[bool] = []

    def set_on_air(self, live: bool) -> None:
        self.on_air.append(live)


class _Mixer:
    """A sink that remembers what it was asked for, in place of pactl."""

    def __init__(self) -> None:
        self.levels: list[int] = []
        self.noted: list[int] = []
        self.note: int | None = None  # what the page is supposed to have left us

    def set_level(self, percent: int) -> bool:
        self.levels.append(percent)
        return True

    def request(self, percent: int) -> None:
        self.noted.append(percent)
        self.note = percent

    def requested(self) -> int | None:
        return self.note


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
    kiosk._handed_over = False  # his voice is on this box's amp, so the knob is a control
    kiosk._slide_from = 0.0
    kiosk._wanted = None
    kiosk._volume = 50
    kiosk._volume_at = 0.0
    kiosk._cues = _Cues()
    # Up before a single test runs, which is the whole testing strategy here: with the latch
    # already held, _slide never spawns a thread, so nothing in this file forks a pactl or opens
    # PortAudio. The gesture is turned by hand instead, one _rung() at a time - which is why
    # _rung is a method rather than three lines inside _walk.
    kiosk._knob_busy = threading.Event()
    kiosk._knob_busy.set()
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
    assert kiosk._turning, "the column did not come up under the finger"
    assert not kiosk._sliding and kiosk._wanted is None, "a tap asked for a level"
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


def test_the_speaker_follows_the_finger_rung_by_rung(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole gesture: grab the knob, drag up the track, hear every rung on the way.

    The speaker is asked *during* the drag now rather than on the lift, which is the only way
    the beep can be at the level it is announcing - what you hear is the sink's own gain on a
    tone of fixed amplitude, so a rung that has not been set yet cannot be heard at its level.
    """
    kiosk, mixer = _panel(monkeypatch)
    x, y = _knob(kiosk)
    kiosk._on_mouse(DOWN, x, y, 0, None)
    seen = []
    for level in (20, 55, 80):
        kiosk._on_mouse(MOVE, x, _at(kiosk, level), HELD, None)
        seen.append(kiosk._wanted)
        kiosk._rung()
    assert seen == [20, 55, 80], f"the column did not follow the finger: {seen}"
    assert kiosk._sliding and mixer.levels == [20, 55, 80], "the speaker lagged the column"
    assert kiosk._cues.played == ["rung"] * 3, "a rung went by without a sound"

    kiosk._on_mouse(UP, x, _at(kiosk, 80), 0, None)
    assert kiosk._volume == 80, "the level you let go on is the level you meant"
    assert not kiosk._sliding and not kiosk._turning, "the column stayed up"


def test_the_level_is_snapped_to_the_step_the_page_uses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Five, like the slider on the admin page, so the two cannot disagree about what a level is
    - and so one rung of the column is one setting rather than a place between two."""
    kiosk, _ = _panel(monkeypatch)
    x, y = _knob(kiosk)
    kiosk._on_mouse(DOWN, x, y, 0, None)
    for level in range(0, 101, 3):
        kiosk._on_mouse(MOVE, x, _at(kiosk, level), HELD, None)
        assert kiosk._wanted % overlay.VOLUME_STEP == 0, f"{kiosk._wanted} is off the step"


def test_every_rung_is_a_pactl_and_a_beep(monkeypatch: pytest.MonkeyPatch) -> None:
    """A subprocess a rung, which is what hearing where you are costs.

    It used to be one for a whole drag, on the argument that a board which spends its corners
    keeping cool should not fork a process a frame. The ladder is what makes the new bargain
    affordable: there are only twenty rungs, so a sweep of the entire column is bounded at
    twenty however long you take over it. And _walk only ever looks at where the finger is
    *now*, so a flick spends fewer of them than a slow deliberate slide rather than more.

    The grab is still free. This drag is a finger walking every rung on purpose, which is the
    most expensive gesture the panel has and the one nobody makes twice.
    """
    kiosk, mixer = _panel(monkeypatch)
    x, y = _knob(kiosk)
    kiosk._on_mouse(DOWN, x, y, 0, None)
    for level in range(0, 101, 5):
        kiosk._on_mouse(MOVE, x, _at(kiosk, level), HELD, None)
        kiosk._rung()
    assert mixer.levels == list(range(0, 101, 5))
    assert kiosk._cues.played == ["rung"] * 21


def test_a_rung_the_sink_is_already_on_is_not_spent(monkeypatch: pytest.MonkeyPatch) -> None:
    """What keeps a finger resting on one rung silent rather than a machine gun - and what lets
    the walk run on a clock instead of waiting to be told the finger moved."""
    kiosk, mixer = _panel(monkeypatch)
    x, y = _knob(kiosk)
    kiosk._on_mouse(DOWN, x, y, 0, None)
    kiosk._on_mouse(MOVE, x, _at(kiosk, 50), HELD, None)
    assert kiosk._wanted == 50
    for _ in range(4):
        assert not kiosk._rung(), "it was already at 50"
    assert mixer.levels == [] and kiosk._cues.played == []


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


def test_a_release_that_never_arrived_leaves_the_level_where_the_drag_put_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A press with no release behind it means the release was lost. What that costs you is the
    column coming down, and nothing else: the level was set rung by rung, out loud, as the
    finger crossed them, so putting it back where it started after you have heard it climb to a
    hundred would be the surprise. What must not survive is the *drag* - the moves that follow
    the new press do not carry where that press was."""
    kiosk, mixer = _panel(monkeypatch)
    x, y = _knob(kiosk)
    kiosk._on_mouse(DOWN, x, y, 0, None)
    kiosk._on_mouse(MOVE, x, _at(kiosk, 100), HELD, None)
    kiosk._rung()
    assert mixer.levels == [100] and kiosk._volume == 100
    kiosk._on_mouse(DOWN, *kiosk.overlay.hitboxes.eye.center, 0, None)  # no UP in between
    assert not kiosk._sliding and not kiosk._turning
    kiosk._on_mouse(MOVE, x, _at(kiosk, 0), HELD, None)
    kiosk._rung()
    assert mixer.levels == [100], "a drag off his face set the volume"
    assert kiosk._volume == 100, "the level was taken back from a gesture that had landed it"


def test_a_panel_with_no_sink_under_it_sets_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Off the Pi there is no pactl, so there is no level. A column that fills and changes
    nothing is worse than one that never comes up."""
    kiosk, mixer = _panel(monkeypatch)
    kiosk._volume = None
    x, y = _knob(kiosk)
    kiosk._on_mouse(DOWN, x, y, 0, None)
    kiosk._on_mouse(MOVE, x, _at(kiosk, 100), HELD, None)
    assert not kiosk._sliding
    assert not kiosk._rung() and kiosk._wanted is None
    kiosk._on_mouse(UP, x, _at(kiosk, 100), 0, None)
    assert mixer.levels == []


# ------------------------------------------------------------------ the column, and the note


def test_the_column_is_up_the_moment_the_knob_is_touched(monkeypatch: pytest.MonkeyPatch) -> None:
    """It used to wait for six pixels of travel, which is a control you have to start using
    blind. Up on the touch, showing the level the speaker is already at - so what you reach for
    is a column standing where you left it rather than one that materialises under your thumb."""
    kiosk, _ = _panel(monkeypatch)
    kiosk._on_mouse(DOWN, *_knob(kiosk), 0, None)
    assert kiosk._turning, "nothing put the column up"
    assert not kiosk._sliding and kiosk._wanted is None, "it opened on a level nobody asked for"


def test_the_speaker_waits_until_the_finger_is_on_the_ladder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The knob sits below the foot of its own column and the column reads absolutely, so the
    first thing a drag off the disc reports is silence. Now that the sink follows the finger,
    acting on that would cut him off mid-sentence every time you reached for the volume - and
    you would climb back out of nothing. So the drag begins where the ladder does."""
    kiosk, mixer = _panel(monkeypatch)
    x, y = _knob(kiosk)
    kiosk._on_mouse(DOWN, x, y, 0, None)
    below = kiosk.overlay.slider.bottom + 1
    assert y - below > kiosk_module.SLIDE_GRAB_PX, "the gap is too small to travel the floor in"
    kiosk._on_mouse(MOVE, x, below, HELD, None)
    assert not kiosk._sliding, "the drag began in the gap under the track"
    assert not kiosk._rung() and mixer.levels == [], "he was cut off on the way to the ladder"
    kiosk._on_mouse(MOVE, x, _at(kiosk, 40), HELD, None)
    assert kiosk._sliding and kiosk._wanted == 40, "the ladder itself did not take the finger"


def test_the_lift_lands_where_the_thread_left_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """The walk's last pass, run here on the main thread. The gesture is already over, so it
    sets the level one final time, leaves the note for the page's own slider, and goes - and it
    is the note being written *inside* the latch that keeps _sync_volume from ever seeing it
    before the sink it describes."""
    kiosk, mixer = _panel(monkeypatch)
    kiosk._turning = False
    kiosk._wanted = 80
    kiosk._walk()  # returns on the first pass: no clock, no sleep
    assert mixer.levels == [80] and kiosk._volume == 80
    assert mixer.noted == [80], "the page's slider was left where the column was not"
    assert kiosk._wanted is None, "the walk left a level behind for the next grab to inherit"
    assert not kiosk._knob_busy.is_set(), "the latch outlived the thread that held it"


def test_a_finger_on_the_knob_outranks_the_note_the_page_left(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The note is the page's opinion, and it is stale for as long as somebody is setting the
    sink by hand. Without this every drag is fought back down four hundred milliseconds at a
    time, by a poll that reads the disagreement it is itself half of."""
    kiosk, mixer = _panel(monkeypatch)
    mixer.note = 50
    kiosk._volume = 80  # where the finger has walked it, three rungs into a drag
    kiosk._sync_volume()
    assert mixer.levels == [] and kiosk._volume == 80, "the page took the knob back mid-drag"
    kiosk._knob_busy.clear()  # the walk is done and has left its own note
    mixer.note = 50
    kiosk._sync_volume()
    assert mixer.levels == [50], "the page stopped being able to set the volume at all"


# ------------------------------------------------------------------ what the column shows


def _rungs(ov: overlay.Overlay, level: int) -> int:
    """How many rungs of the column are lit, counted off a rendered frame."""
    frame = ov.render(state=overlay.IDLE, level=0.0, elapsed=None, phase=10.0, volume=level,
                      temp_c=58.0, pressed=overlay.VOLUME, turning=True).astype(int)
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
    assert not np.array_equal(quiet[band], ov.render(turning=True, **shown)[band])  # type: ignore[arg-type]
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


# ------------------------------------------------------------------ whose speaker he is on


def test_the_knob_says_when_a_companion_has_his_voice() -> None:
    """The knob goes over to the handover hue whenever the claim is held - asleep or mid-session
    - and nothing else in that corner moves with it.

    The gauge beside it is in the assertion for the same reason its hitbox is: what the knob
    turns into has to stay inside the knob's own hole in the panel, and a gauge that changed
    with it would mean a board temperature had started depending on who was listening.
    """
    ov = overlay.Overlay(800, 480)
    for state, elapsed in ((overlay.IDLE, None), (overlay.LISTENING, 12.0)):
        shown = dict(state=state, level=0.0, elapsed=elapsed, phase=10.0, volume=60, temp_c=58.0)
        here = ov.render(**shown)  # type: ignore[arg-type]
        away = ov.render(handed_over=True, **shown)  # type: ignore[arg-type]
        for name, box, same in ((overlay.VOLUME, ov.hitboxes.volume, False),
                                (overlay.HEAT, ov.hitboxes.heat, True)):
            a, b = (f[box.y : box.bottom, box.x : box.right].tobytes() for f in (here, away))
            assert (a == b) is same, f"the {name} dial was the wrong kind of unchanged"


def test_the_whole_knob_goes_over_and_not_one_mark_on_it() -> None:
    """Every lit thing on the instrument, not a badge in the gap. That is the difference between
    this and the round before it, which was looked at on the Pi and called too easy to miss.

    Counted as pixels that changed hue rather than as a colour anywhere: a mark swapped for
    another mark of the same size moves a couple of hundred pixels in the gap under the hub, and
    a dial relit moves the arc, the pointer, the hub, the graduations and the seat's reveal with
    it. The floor is a tenth of the face, which no mark that fits in :meth:`_mark_box` can reach.
    """
    ov = overlay.Overlay(800, 480)
    shown = dict(state=overlay.LISTENING, level=0.0, elapsed=12.0, phase=10.0,
                 volume=60, temp_c=58.0)
    box = ov.hitboxes.volume
    here, away = (ov.render(handed_over=h, **shown)[  # type: ignore[arg-type]
        box.y : box.bottom, box.x : box.right].astype(int) for h in (False, True))
    # Bluer than it was green, which is the one direction that cannot be reached by dimming.
    bluer = (away[..., 2] - away[..., 1]) - (here[..., 2] - here[..., 1])
    face = np.pi * ov.btn_r ** 2
    assert (bluer > 30).sum() > face / 10, "the knob changed in one place instead of all over"


def test_a_knob_that_is_an_icon_takes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """No column, no rung, no level - and no barge-in either. A press that lands on a control
    has not missed every control, whatever the control has stopped doing."""
    kiosk, mixer = _panel(monkeypatch)
    kiosk._handed_over = True
    x, y = _knob(kiosk)
    kiosk._on_mouse(DOWN, x, y, 0, None)
    kiosk._on_mouse(MOVE, x, _at(kiosk, 90), HELD, None)
    kiosk._on_mouse(UP, x, _at(kiosk, 90), 0, None)
    assert not kiosk._turning and not kiosk._sliding, "the column came up on a dead knob"
    assert mixer.levels == [] and kiosk._volume == 50, "an icon set the sink"
    assert kiosk._cues.played == [], "an icon clicked"


def test_a_claim_that_lands_mid_drag_ends_the_drag(monkeypatch: pytest.MonkeyPatch) -> None:
    """The finger is already on the track when a phone picks his voice up. The knob becomes an
    icon under it, so the gesture goes with it rather than carrying on out of sight."""
    kiosk, mixer = _panel(monkeypatch)
    kiosk._handed_over = False
    kiosk._handover_at = 0.0  # the clock is monotonic, so any poll is overdue
    kiosk.controller = _Controller()
    x, y = _knob(kiosk)
    kiosk._on_mouse(DOWN, x, y, 0, None)
    kiosk._on_mouse(MOVE, x, _at(kiosk, 90), HELD, None)
    assert kiosk._sliding, "set the drag up first, or this test is about nothing"
    monkeypatch.setattr(kiosk_module.companion, "listening", lambda: True)
    kiosk._sync_handover()
    assert not kiosk._turning and not kiosk._sliding, "the drag outlived the control"
