"""The long press on his face, and the menu it opens: shut the box down, or restart it.

Two things here can go wrong quietly, and each has a test that would rather they went loudly.

The gesture opens a menu *under the finger that asked for it*, so a row acting on the release
would be chosen by that same press - and the row his face is behind is SHUT DOWN. And a tap and
a hold now share one control, so the tap has to survive: a panel where his eye stopped opening
the sessions list would be a real loss for a feature nobody has pressed yet.

All of it is pure: no window, no camera, no subprocess.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from cyclops import kiosk as kiosk_module
from cyclops import overlay, power

SIZES = ((800, 480), (480, 320), (1280, 720))
DOWN, UP = cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP


# ---------------------------------------------------------------- the card on the panel


@pytest.mark.parametrize(("width", "height"), SIZES)
def test_the_card_leaves_his_face_showing(width: int, height: int) -> None:
    """He is what you pressed to get here. A card over him would orphan the gesture.

    It used to have to dodge him upwards, because he stood in the middle of the tab row and the
    honest centre of the panel was his face. With him in a corner the card can simply be centred
    and still clear him - sideways now rather than above, which is why this measures against his
    swell on the x axis and not against the top of his head.
    """
    ov = overlay.Overlay(width, height)
    card = ov.menu_card
    assert card.x > ov.eye[0] + ov.shoulder, "the menu covers the face that opened it"
    assert ov.frame.x <= card.x and card.right <= ov.frame.right
    assert ov.frame.y <= card.y and card.bottom <= ov.frame.bottom


@pytest.mark.parametrize(("width", "height"), SIZES)
def test_every_row_is_a_target_a_thumb_can_find(width: int, height: int) -> None:
    ov = overlay.Overlay(width, height)
    rows = [ov.menu_cells[key] for key, _ in overlay.MENU_ROWS]
    for row in rows:
        assert row.h >= 0.11 * height, "a row too shallow to hit without looking"
        assert ov.menu_card.y <= row.y and row.bottom <= ov.menu_card.bottom
    for one, other in zip(rows, rows[1:], strict=False):
        assert one.bottom <= other.y, "two rows of a power menu overlap"


@pytest.mark.parametrize(("width", "height"), SIZES)
def test_every_row_label_fits_beside_its_mark(width: int, height: int) -> None:
    """The Mac and the Pi pick different faces, so this is a test and not a measurement."""
    ov = overlay.Overlay(width, height)
    tracking = max(1.0, 2.0 * ov.scale)
    r = max(6, round(overlay.MENU_GLYPH_R * height))
    gap = max(6, round(14 * ov.scale))
    for key, label in overlay.MENU_ROWS:
        used = 0 if key == overlay.CANCEL else 2 * r + 2 * gap
        room = ov.menu_cells[key].w - used
        assert ov._width(label, ov.font_read, tracking) <= room, f"{label} at {width}x{height}"


def test_the_header_says_which_of_the_two_sleeps_this_is() -> None:
    """GO TO SLEEP ends the session and SHUT DOWN ends the box, and the card says so."""
    ov = overlay.Overlay(800, 480)
    tracking = max(1.0, 2.4 * ov.scale)
    room = ov.menu_card.w - 4 * max(3, round(overlay.MENU_PAD * 480))
    title = ov._width(overlay.MENU_TITLE, ov.font_tab, tracking)
    assert title + ov.font_micro.getlength(overlay.MENU_NOTE) <= room


# ---------------------------------------------------------------- where a tap lands


def test_a_tap_off_the_card_is_the_way_out() -> None:
    ov = overlay.Overlay(800, 480)
    assert ov.menu_hit(5, 5) == overlay.CANCEL
    assert ov.menu_hit(*ov.hitboxes.shutter.center) == overlay.CANCEL, "a tab reached through"
    assert ov.menu_hit(*ov.hitboxes.wake.center) == overlay.CANCEL


def test_every_row_answers_to_its_own_middle() -> None:
    ov = overlay.Overlay(800, 480)
    for key, _ in overlay.MENU_ROWS:
        assert ov.menu_hit(*ov.menu_cells[key].center) == key


def test_the_card_is_not_a_way_out_by_accident() -> None:
    """Missing a row is not a dismissal: a card that answers for you is worse than no card."""
    ov = overlay.Overlay(800, 480)
    head = (ov.menu_card.y + ov.menu_cells[overlay.POWER_OFF].y) // 2
    assert ov.menu_hit(ov.menu_card.center[0], head) is None


def test_the_choices_are_opaque() -> None:
    """They sit over a live camera. A choice you read the room through is a coin toss."""
    ov = overlay.Overlay(800, 480)
    chrome = ov.render(state=overlay.LISTENING, level=0.0, menu=True)
    cell = ov.menu_cells[overlay.POWER_OFF]
    band = chrome[cell.y + 4 : cell.bottom - 4, cell.x + 4 : cell.right - 4, 3]
    assert band.min() == 255


# ---------------------------------------------------------------- the collar filling in


def _collar(chrome: np.ndarray, ov: overlay.Overlay) -> int:
    """How much of the rail that goes round him is lit in full phosphor, in pixels.

    The whole square he stands in, not the top half of it: the rail leaves the straight below his
    equator on one side and rejoins it below on the other, so a band cut at his centre would miss
    both ends of the sweep and see the fill only in the middle of the press.
    """
    cx, cy = ov.eye
    reach = ov.shoulder + ov.rail_w
    band = chrome[max(0, cy - reach) : cy + reach, max(0, cx - reach) : cx + reach, :3]
    # Full phosphor, not merely bright: the rail he is mounted in has a lit lip of its own a
    # couple of pixels outside this arc, and a threshold loose enough to include that is a
    # threshold measuring a constant. He sits in a corner now, so the band is clamped to the
    # panel as well - a negative slice would quietly measure the wrong side of the screen.
    return int((band[:, :, 1] > 240).sum())


def test_the_collar_fills_as_the_press_goes_on() -> None:
    """Without this a long press is a second of nothing followed by a menu, which reads as a
    fault that fixed itself."""
    ov = overlay.Overlay(800, 480)
    shown = dict(state=overlay.LISTENING, level=0.0, pressed="eye")
    lit = [_collar(ov.render(hold=hold, **shown), ov) for hold in (0.0, 0.5, 1.0)]
    assert lit[0] < lit[1] < lit[2], f"the collar did not fill: {lit}"


# ---------------------------------------------------------------- the tap and the hold


class Cues:
    def __init__(self) -> None:
        self.played: list[str] = []
        self.stopped = 0

    def play(self, name: str, *, loop: bool = False) -> float:
        self.played.append(name)
        return 0.0

    def stop(self) -> None:
        self.stopped += 1


def _panel(monkeypatch: pytest.MonkeyPatch) -> kiosk_module.Kiosk:
    """A kiosk with a panel and nothing else - no camera, no controller, no window."""
    kiosk = object.__new__(kiosk_module.Kiosk)
    kiosk.overlay = overlay.Overlay(800, 480)
    kiosk._eye_down_at = None
    kiosk._menu = False
    kiosk._menu_until = 0.0
    kiosk.power = None
    kiosk._power_at = 0.0
    kiosk._pressed = None
    kiosk._press_until = 0.0
    kiosk._touched_at = 0.0
    kiosk._asleep = False
    kiosk._notice = ""
    kiosk._notice_until = 0.0
    kiosk._cues = Cues()
    kiosk.opened: list[str] = []
    monkeypatch.setattr(kiosk, "_open_admin", lambda: kiosk.opened.append("admin"))
    monkeypatch.setattr(kiosk, "_snap", lambda: kiosk.opened.append("snap"))
    monkeypatch.setattr(kiosk, "_toggle_session", lambda: kiosk.opened.append("session"))
    return kiosk


def test_a_tap_on_his_face_still_opens_what_the_box_has_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tap is the thing people already do. It survives the hold being added to it."""
    kiosk = _panel(monkeypatch)
    x, y = kiosk.overlay.hitboxes.eye.center
    kiosk._on_mouse(DOWN, x, y, 0, None)
    assert kiosk.opened == [], "the page opened before the finger came off"
    kiosk._on_mouse(UP, x, y, 0, None)
    assert kiosk.opened == ["admin"]
    assert not kiosk._menu


def test_a_finger_that_slides_off_takes_the_tap_back(monkeypatch: pytest.MonkeyPatch) -> None:
    kiosk = _panel(monkeypatch)
    kiosk._on_mouse(DOWN, *kiosk.overlay.hitboxes.eye.center, 0, None)
    kiosk._on_mouse(UP, *kiosk.overlay.hitboxes.shutter.center, 0, None)
    assert kiosk.opened == []
    assert kiosk._cues.stopped == 1, "the sound is the press, so it ends wherever the finger did"


def test_his_face_sounds_for_as_long_as_it_is_held(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cue tracks the finger: down starts it, up ends it. Every other control on this panel
    fires and forgets, because every other one is over before you have finished pressing it."""
    kiosk = _panel(monkeypatch)
    x, y = kiosk.overlay.hitboxes.eye.center
    kiosk._on_mouse(DOWN, x, y, 0, None)
    assert kiosk._cues.played == ["pressed"] and kiosk._cues.stopped == 0
    kiosk._on_mouse(UP, x, y, 0, None)
    assert kiosk._cues.stopped == 1


def test_lifting_off_a_menu_that_already_opened_does_not_cut_its_cue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one way "stop on the lift" could go wrong: the hold has already sounded "menu" and
    the finger is still down, so an unconditional stop on the way up would cut the menu off a
    beat after it opened - which is the one moment the panel is behind a palm and the sound is
    all there is. _open_menu clears the hold, so _lifted returns before it can."""
    kiosk = _panel(monkeypatch)
    x, y = kiosk.overlay.hitboxes.eye.center
    kiosk._on_mouse(DOWN, x, y, 0, None)
    kiosk._eye_down_at -= kiosk_module.LONG_PRESS_S
    kiosk._open_menu()
    kiosk._on_mouse(UP, x, y, 0, None)  # the finger that asked for it, coming off
    assert kiosk._cues.played == ["pressed", "menu"]
    assert kiosk._cues.stopped == 0


def test_holding_his_face_opens_the_power_menu(monkeypatch: pytest.MonkeyPatch) -> None:
    kiosk = _panel(monkeypatch)
    kiosk._on_mouse(DOWN, *kiosk.overlay.hitboxes.eye.center, 0, None)
    kiosk._eye_down_at -= kiosk_module.LONG_PRESS_S  # a finger that stayed put; see _holding
    assert kiosk._holding() >= 1.0
    kiosk._open_menu()
    assert kiosk._menu
    # Both, in that order: his face answers the finger the moment it lands, and the menu cue
    # cuts that off at LONG_PRESS_S. The panel is behind a palm by then, so the sound is the
    # only thing left that can say what just opened.
    assert kiosk._cues.played == ["pressed", "menu"]
    assert kiosk.opened == []


def test_the_press_that_opened_the_menu_cannot_choose_from_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The nastiest failure this gesture has: the menu opens under a finger that is still down
    on his face, and SHUT DOWN is the row his face is behind."""
    kiosk = _panel(monkeypatch)
    x, y = kiosk.overlay.hitboxes.eye.center
    kiosk._on_mouse(DOWN, x, y, 0, None)
    kiosk._eye_down_at -= kiosk_module.LONG_PRESS_S
    kiosk._open_menu()
    kiosk._on_mouse(UP, *kiosk.overlay.menu_cells[overlay.POWER_OFF].center, 0, None)
    assert kiosk.power is None, "the hold chose for you"
    assert kiosk._menu, "and then put the menu away"
    assert kiosk.opened == []


def test_the_menu_is_modal(monkeypatch: pytest.MonkeyPatch) -> None:
    kiosk = _panel(monkeypatch)
    kiosk._menu = True
    kiosk._on_mouse(DOWN, *kiosk.overlay.hitboxes.shutter.center, 0, None)
    assert kiosk.opened == [], "the shutter fired through the menu"
    assert not kiosk._menu, "a tap off the card is a way out"


def test_choosing_shuts_the_box_down_after_the_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tap records what to do and nothing else. main() honours it once the session is on
    the card - the video is still being muxed while the panel says goodbye."""
    kiosk = _panel(monkeypatch)
    monkeypatch.setattr(
        power, "take_down", lambda verb: pytest.fail(f"{verb} ran from the tap itself")
    )
    kiosk._menu = True
    kiosk._on_mouse(DOWN, *kiosk.overlay.menu_cells[overlay.POWER_OFF].center, 0, None)
    assert kiosk.power == power.POWEROFF
    kiosk2 = _panel(monkeypatch)
    kiosk2._menu = True
    kiosk2._on_mouse(DOWN, *kiosk2.overlay.menu_cells[overlay.RESTART].center, 0, None)
    assert kiosk2.power == power.REBOOT


def test_cancel_leaves_the_box_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    kiosk = _panel(monkeypatch)
    kiosk._menu = True
    kiosk._on_mouse(DOWN, *kiosk.overlay.menu_cells[overlay.CANCEL].center, 0, None)
    assert not kiosk._menu
    assert kiosk.power is None
