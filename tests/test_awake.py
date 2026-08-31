"""The panel as a creature: what it says, whether its eye is open, and whether it holds still.

Four failures live here that nothing else would catch. A button whose word stops matching what
the tap does is a lie nobody notices until they press it. A blink on a fixed period is a status
LED and reads as a fault rather than as a face, and one shorter than a few frames is
indistinguishable from a dropped frame - neither shows up in any assertion about *whether* it
blinks. A resting caption that keeps breathing turns "asleep" into "asleep, sort of", which is
the whole design lost quietly. And a corner square that grows with the window can walk into the
readouts on a screen nobody tested on.

Everything here is pure: no camera, no key, no window. `Overlay` is PIL and numpy only.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cyclops import overlay

STATES = (
    overlay.IDLE,
    overlay.STARTING,
    overlay.STOPPING,
    overlay.CONNECTING,
    overlay.LISTENING,
    overlay.SPEAKING,
    overlay.LOOKING,
    overlay.SEARCHING,
    overlay.DRAWING,
    overlay.ERROR,
)
SIZES = ((800, 480), (480, 320), (1280, 720))


# ---------------------------------------------------------------- the words


@pytest.mark.parametrize("state", STATES)
def test_the_button_says_what_the_tap_will_do(state: str) -> None:
    want = overlay.SLEEP_LABEL if overlay.session_up(state) else overlay.WAKE_LABEL
    assert overlay.tab_label("eye", state) == want
    # ...and the same question the kiosk asks before deciding to start or stop, so the word on
    # the button and the action behind it cannot drift apart.
    assert overlay.session_up(state) == (state not in (overlay.IDLE, overlay.ERROR))


@pytest.mark.parametrize("name", ("shutter", "admin"))
def test_the_other_two_tabs_do_not_change_under_you(name: str) -> None:
    assert len({overlay.tab_label(name, state) for state in STATES}) == 1


def test_the_resting_caption_names_a_button_that_is_actually_there() -> None:
    # One line, and exactly the bug this change invited: the caption said "tap SESSION to begin"
    # for as long as the tab said SESSION, and would have gone on saying it afterwards.
    assert overlay.tab_label("eye", overlay.IDLE) in overlay.CAPTIONS[overlay.IDLE]


@pytest.mark.parametrize("state", STATES)
def test_every_state_still_has_a_word_for_the_strip(state: str) -> None:
    assert overlay.LABELS[state]


# ---------------------------------------------------------------- the blink


def _openness(seconds: float, step: float = 1 / 200) -> list[float]:
    return [overlay.eye_open(i * step) for i in range(int(seconds / step))]


def test_the_eye_is_wide_open_between_blinks_and_shuts_completely_in_one() -> None:
    sweep = _openness(60.0)
    assert max(sweep) == 1.0
    assert min(sweep) == pytest.approx(0.0, abs=1e-9)


def test_the_lid_travels_rather_than_snapping() -> None:
    # A raised cosine, not a square wave - the same argument caption_pulse makes. An eye that
    # switched between two pictures would read as a dropped frame, not as a blink.
    assert any(0.05 < v < 0.95 for v in _openness(20.0))


def test_he_blinks_often_enough_to_be_alive_and_seldom_enough_not_to_nag() -> None:
    shut = [v < 0.5 for v in _openness(60.0)]
    blinks = sum(1 for a, b in zip(shut, shut[1:], strict=False) if b and not a)
    assert 8 <= blinks <= 20, f"{blinks} blinks a minute is not a face"


def test_a_blink_survives_a_panel_running_at_25_fps() -> None:
    # The loop's real sampling rate. A blink briefer than a few frames is a dropped frame.
    frames = [overlay.eye_open(i / 25.0) for i in range(25 * 30)]
    runs, run = [], 0
    for v in frames:
        run = run + 1 if v < 0.5 else 0
        runs.append(run)
    assert max(runs) >= 3, "a blink that lands on fewer than three frames will not be seen"


def test_the_blinks_are_not_a_metronome() -> None:
    # A fixed period is a status LED. The drift is what makes it a creature, so assert the drift.
    shut = [v < 0.5 for v in _openness(300.0)]
    at = [i / 200 for i, (a, b) in enumerate(zip(shut, shut[1:], strict=False)) if b and not a]
    gaps = [round(b - a, 2) for a, b in zip(at, at[1:], strict=False)]
    assert len(set(gaps)) > 1, "every gap is the same length"
    assert min(gaps) > 1.5, f"two blinks {min(gaps)}s apart is a twitch"
    assert len(set(gaps[:8])) > 2, "the gaps merely alternate between two values"


def test_no_blink_is_clipped_by_the_edge_of_its_own_window() -> None:
    # BLINK_DRIFT_S + BLINK_S must stay under BLINK_EVERY_S, asserted through the function
    # rather than against the constants: a blink that ran past its window would be cut off
    # half-shut, and the eye would jump back open.
    assert overlay.BLINK_DRIFT_S + overlay.BLINK_S < overlay.BLINK_EVERY_S
    for window in range(200):
        start = window * overlay.BLINK_EVERY_S
        n = int(overlay.BLINK_EVERY_S * 500)
        inside = [overlay.eye_open(start + i / 500) for i in range(n)]
        assert min(inside) == pytest.approx(0.0, abs=1e-3), f"window {window} lost its blink"
        assert inside[0] == 1.0 and inside[-1] == 1.0, f"window {window} starts or ends mid-blink"


def test_the_eye_never_stalls_however_long_the_panel_has_been_up() -> None:
    # A Pi's monotonic clock is its uptime and this panel is left running for weeks - which is
    # long enough that a float64 losing its last digits would quietly freeze the lid open. What
    # has to hold at a million seconds is not a particular count - the drift makes that vary by
    # one, deliberately - but that he is still blinking at a living rate, and still all the way
    # shut. Same argument as the caption's dots.
    for base in (0.0, 86_400.0, 1_000_000.0, 5_000_000.0):
        sweep = [overlay.eye_open(base + i / 200) for i in range(60 * 200)]
        shut = [v < 0.5 for v in sweep]
        blinks = sum(1 for a, b in zip(shut, shut[1:], strict=False) if b and not a)
        assert 8 <= blinks <= 20, f"{blinks} blinks a minute at {base:,.0f}s of uptime"
        assert min(sweep) == pytest.approx(0.0, abs=1e-3), f"the lid stopped closing at {base}"


# ---------------------------------------------------------------- the eye's shape


def test_shut_is_the_same_shape_as_open_and_not_a_second_symbol() -> None:
    top, bottom = overlay.eye_lids(0.0)
    assert top == bottom, "a shut eye is one curve, so a blink is a lerp and not a swap"
    assert top > 0, "and it curves downwards - a straight line is a minus sign"
    top, bottom = overlay.eye_lids(1.0)
    assert top < 0 < bottom
    assert abs(top) > bottom, "fuller on top, or it is a lens rather than an eye"


def test_the_lids_move_monotonically_so_a_blink_never_reverses() -> None:
    lids = [overlay.eye_lids(i / 40) for i in range(41)]
    assert all(a[0] > b[0] for a, b in zip(lids, lids[1:], strict=False))
    assert all(a[1] < b[1] for a, b in zip(lids, lids[1:], strict=False))


def test_a_fully_dilated_pupil_exactly_fills_the_eye() -> None:
    # The relation the geometry is built on: your voice can open the pupil to the lower lid and
    # no further, so it can never spill out of the eye it is in.
    assert overlay.EYE_PUPIL + overlay.EYE_DILATE == pytest.approx(overlay.EYE_BOTTOM)


@pytest.mark.parametrize("state", STATES)
def test_the_eye_is_only_open_when_he_is(state: str) -> None:
    openness = overlay.eye_openness(state, 2.0)
    assert (openness > 0) == overlay.awake(state)
    if state in (overlay.STARTING, overlay.CONNECTING):
        assert openness < 1.0, "coming round is not the same as being up"


# ---------------------------------------------------------------- the breath


def test_the_border_breath_stays_inside_its_depth_and_comes_back_round() -> None:
    sunk = [overlay.rim_breath(i * overlay.RIM_PERIOD_S / 64) for i in range(64)]
    assert min(sunk) == 0.0, "the brightest frame must be exactly the one _base already drew"
    assert max(sunk) == pytest.approx(overlay.RIM_DEPTH, rel=1e-3)
    for t in (0.15, 1.9, 86_400.15):
        assert overlay.rim_breath(t + overlay.RIM_PERIOD_S) == pytest.approx(
            overlay.rim_breath(t), abs=1e-6
        )


def test_no_two_rhythms_on_the_panel_lock_together() -> None:
    # Two periods that lock read as one mechanism. The caption's dots and its breath are already
    # 2:1 and that is old news; nothing added since may lock to either, or to the other.
    periods = (overlay.RIM_PERIOD_S, overlay.BLINK_EVERY_S, overlay.BREATH_PERIOD_S)
    for i, a in enumerate(periods):
        for b in periods[i + 1 :]:
            ratio = max(a, b) / min(a, b)
            assert abs(ratio - round(ratio)) > 0.1, f"{a}s and {b}s beat together"


# ---------------------------------------------------------------- on the panel


def _eye(ov: overlay.Overlay, **kwargs: object) -> np.ndarray:
    """Just the eye, cropped off a rendered frame.

    Inside the housing's rings rather than the whole square: the box's own rules, the two arcs
    and the panel border all live in that square and hold still, so a crop that included them
    would let a frozen eye pass every test below.
    """
    a = ov.avatar
    cx, cy = a.center
    half = int(a.w * overlay.EYE_RINGS / math.sqrt(2)) - 1  # clear of the arcs, corners included
    return ov.render(**kwargs)[cy - half : cy + half, cx - half : cx + half]  # type: ignore[arg-type]


def test_nothing_moves_while_he_is_asleep() -> None:
    # The headline. Every animation on this panel has to be gated on there being a session, and
    # this is the one assertion that notices when a new one is not - including the caption's
    # breath, which used to run at IDLE and made a resting panel quietly pulse.
    ov = overlay.Overlay(800, 480)
    for state, detail in ((overlay.IDLE, ""), (overlay.ERROR, "OpenAI rejected the API key")):
        frames = [
            ov.render(state=state, level=0.0, detail=detail, elapsed=None, phase=i * 0.73)
            for i in range(40)
        ]
        moved = [i for i, f in enumerate(frames) if not np.array_equal(f, frames[0])]
        assert not moved, f"{state} moved at phases {moved[:5]}"


def test_the_eye_opens_when_he_wakes() -> None:
    ov = overlay.Overlay(800, 480)
    shut = _eye(ov, state=overlay.IDLE, level=0.0, phase=0.0)
    open_ = _eye(ov, state=overlay.LISTENING, level=0.0, phase=2.0)

    def height(crop: np.ndarray) -> int:
        rows = np.where((crop[:, :, :3].astype(int).sum(axis=2) > 300).any(axis=1))[0]
        return int(rows.max() - rows.min()) if len(rows) else 0

    assert height(open_) > 2 * height(shut), "an open eye is not twice a shut one"


def test_he_blinks_while_he_is_awake() -> None:
    ov = overlay.Overlay(800, 480)
    at = min(
        (i / 200 for i in range(200 * 20) if overlay.eye_open(i / 200) < 0.2), default=None
    )
    assert at is not None
    shown = dict(state=overlay.LISTENING, level=0.0)
    assert not np.array_equal(_eye(ov, phase=at, **shown), _eye(ov, phase=2.0, **shown))


def test_the_pupil_widens_with_your_voice() -> None:
    ov = overlay.Overlay(800, 480)
    quiet = _eye(ov, state=overlay.LISTENING, level=0.0, phase=2.0)
    loud = _eye(ov, state=overlay.LISTENING, level=1.0, phase=2.0)
    lit = lambda c: int((c[:, :, :3].astype(int).sum(axis=2) > 300).sum())  # noqa: E731
    assert lit(loud) > lit(quiet)


def test_the_border_breathes_while_he_is_up() -> None:
    ov = overlay.Overlay(800, 480)
    shown = dict(state=overlay.LISTENING, level=0.0)
    top = lambda p: ov.render(phase=p, **shown)[0]  # noqa: E731
    bright, sunk = top(0.0), top(overlay.RIM_PERIOD_S / 2)
    assert not np.array_equal(bright, sunk), "the rim held still"
    # It dims and never goes dark, and it never brightens past what _base already drew.
    lit = bright[:, :3].astype(int).sum(axis=1)
    dim = sunk[:, :3].astype(int).sum(axis=1)
    assert (dim <= lit).all(), "the breath brightened the border past its own state colour"
    assert dim[bright[:, 3] > 200].min() > 0


# ---------------------------------------------------------------- the layout it displaced


@pytest.mark.parametrize(("width", "height"), SIZES)
def test_the_readouts_clear_his_corner(width: int, height: int) -> None:
    ov = overlay.Overlay(width, height)
    for taping in (False, True):
        clock_right, _, _ = ov._readouts(taping)
        assert clock_right <= ov.avatar.x, "the clock has walked under the avatar"


@pytest.mark.parametrize(("width", "height"), SIZES)
def test_the_mode_word_clears_the_readouts(width: int, height: int) -> None:
    ov = overlay.Overlay(width, height)
    track = max(1.0, 2.0 * ov.scale)
    start = ov.pad + ov._width("CYCLOPS", ov.font_brand, track) + round(11 * ov.scale) * 2
    for taping in (False, True):
        _, _, meter_right = ov._readouts(taping)
        sig = ov._meter_x(meter_right) - round(9 * ov.scale) - ov._width("SIG", ov.font_micro, 0)
        for word in set(overlay.LABELS.values()):
            end = start + ov._width(word, ov.font_mode, track * 0.7)
            assert end < sig, f"{word} runs into SIG at {width}x{height} (taping={taping})"


@pytest.mark.parametrize(("width", "height"), SIZES)
def test_every_tab_label_fits_its_cell(width: int, height: int) -> None:
    # The Mac and the Pi pick different faces, so this is a test and not a measurement.
    ov = overlay.Overlay(width, height)
    tracking = max(1.0, 2.4 * ov.scale)
    for name in overlay.TABS:
        for state in STATES:
            text = overlay.tab_label(name, state)
            room = ov._cells[name].w - 2 * max(1, round(3 * ov.scale))
            assert ov._width(text, ov.font_tab, tracking) <= room, f"{text} at {width}x{height}"
