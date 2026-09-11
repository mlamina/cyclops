"""The mark language, the gesture slot, and the mapping onto the panel."""

from __future__ import annotations

import numpy as np
import pytest

from cyclops import overlay, point


@pytest.fixture(autouse=True)
def _no_gesture():
    """The slot is a module global; a test that leaves one up would poison the next."""
    point.clear()
    yield
    point.clear()


def test_every_kind_reads_and_rubbish_is_skipped() -> None:
    """One mark per line, four kinds, and anything illegible dropped rather than raised - the
    model writes this mid-sentence and cannot be sent back to fix it."""
    marks = point.parse(
        "ring 0.42 0.31 cold joint\n"
        "n 2 0.55 0.62\n"
        "tag 0.1 0.9 earth\n"
        "arrow 0.2 0.8 0.35 0.55 here\n"
        "ring\n"
        "wiggle 0.5 0.5\n"
        "ring one two\n"
        "\n"
    )

    assert [mark.kind for mark in marks] == ["ring", "n", "tag", "arrow"]
    assert marks[0].label == "cold joint"
    assert marks[1].n == 2
    assert (marks[3].x2, marks[3].y2) == (0.35, 0.55)


def test_coordinates_are_clamped_and_the_list_is_capped() -> None:
    """Just off the edge means the edge, and a model that writes a diagram gets a gesture."""
    marks = point.parse("ring 1.7 -0.4\n" + "ring 0.5 0.5\n" * 12)

    assert (marks[0].x, marks[0].y) == (1.0, 0.0)
    assert len(marks) == point.MAX_MARKS


def test_the_middle_of_any_picture_is_the_middle_of_the_panel() -> None:
    """The camera is 16:9, the endoscope 4:3 and the panel 5:3; a fraction has to survive all
    three, which is why marks travel as proportions rather than pixels."""
    assert point.to_panel(0.5, 0.5, 1024, 576, 800, 480) == (400, 240)
    assert point.to_panel(0.5, 0.5, 640, 480, 800, 480) == (400, 240)


@pytest.mark.parametrize("src_w,src_h", [(1024, 576), (1280, 720), (640, 480)])
def test_a_mark_lands_where_the_frame_it_was_cropped_from_put_it(src_w: int, src_h: int) -> None:
    """The one piece of arithmetic here that has to be right. Checked against the crop the panel
    actually performs rather than against a second copy of the sums: a white dot painted on the
    source frame has to come out under the mapped coordinate once fit_to_window has had it."""
    frame = np.zeros((src_h, src_w, 3), np.uint8)
    fx, fy = 0.25, 0.7
    frame[round(fy * src_h) - 2 : round(fy * src_h) + 2,
          round(fx * src_w) - 2 : round(fx * src_w) + 2] = 255

    shown = overlay.fit_to_window(frame, 800, 480)
    x, y = point.to_panel(fx, fy, src_w, src_h, 800, 480)

    assert shown[y, x].mean() > 128, "the mark missed the thing it was pointing at"


def test_a_gesture_holds_then_fades_then_is_over() -> None:
    gesture = point.Gesture((point.Mark("ring", 0.5, 0.5),), None, born=100.0, hold_s=3.0)

    assert point.fade(gesture, 102.9) == 1.0
    assert 0.0 < point.fade(gesture, 103.3) < 1.0
    assert point.fade(gesture, 104.0) == 0.0
    assert not point.expired(gesture, 103.3) and point.expired(gesture, 103.6)


def test_the_reticle_leaves_the_lens_axis_and_comes_back() -> None:
    """It is pulled back with the gesture rather than snapping home when the gesture ends."""
    gesture = point.Gesture((point.Mark("ring", 0.5, 0.5),), None, born=0.0, hold_s=3.0)
    home, target = (400, 240), (700, 100)

    assert point.reticle_at(gesture, 0.0, home, target) == home
    assert point.reticle_at(gesture, point.TRAVEL_S, home, target) == target
    assert point.reticle_at(gesture, 3.0 + point.FADE_S, home, target) == home


def test_a_half_written_call_draws_only_its_finished_lines() -> None:
    """What the line language buys: no closing anything, so the last line is simply dropped
    until it ends. A JSON parser cannot help with a string that stops mid-word."""
    started = '{"marks": "ring 0.4 0.3 plant\\nn 1 0.5'
    assert len(point.partial_marks(started)) == 1

    assert len(point.partial_marks(started + ' 0.62\\ntag 0.9 0.9 x')) == 2
    assert len(point.partial_marks(started + ' 0.62"}')) == 2
    assert point.partial_marks('{"picture": "a') == []
