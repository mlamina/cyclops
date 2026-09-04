"""Cyclops' eye: how it moves, how it changes its mind, and where it sits in his bracket.

Five failures live here, and none of them raises. A microphone that stops filling in while he is
up is a lie nobody notices until they press it - and it is the *only* thing left saying which way
that switch will go, now that no control on this panel carries a word. A blink on a fixed
period is a status LED and
reads as a fault rather than as a face, and one shorter than a few frames is indistinguishable
from a dropped frame - neither shows up in any assertion about *whether* it blinks. A mood table
that gains a state without a row falls back to the sleeping face, so a whole state of the machine
quietly stops being on the panel. A resting caption that keeps breathing turns "asleep" into
"asleep, sort of", which is the design lost quietly. And an eye grown a few pixels too big walks
into the word underneath it on a screen nobody tested on.

Everything here is pure: no camera, no key, no window. `Overlay` is PIL and numpy only.
"""

from __future__ import annotations

import math
import re
from dataclasses import replace

import numpy as np
import pytest

from cyclops import eye, overlay

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
AWAKE = dict(state=overlay.LISTENING, level=0.0, elapsed=12.0)


def _panel(width: int = 800, height: int = 480) -> overlay.Overlay:
    return overlay.Overlay(width, height)


def _settle(ov: overlay.Overlay, **shown: object) -> None:
    """Run the crossfade into this mood out, so what follows is the mood and not the journey.

    Two renders at *different* phases, which is the whole trick: the engine starts its clock on
    the frame the state changes and moves on the ones after it, so settling with the same phase
    twice settles nothing.
    """
    ov.render(phase=0.0, **shown)  # type: ignore[arg-type]
    ov.render(phase=eye.MOOD_EASE_S * 4, **shown)  # type: ignore[arg-type]


# ---------------------------------------------------------------- the words


def _mic_core(ov: overlay.Overlay, state: str) -> float:
    """How lit the middle of the microphone's head is - the whole of what "filled in" means.

    The head is a rounded rectangle: an outline while he is down, a solid while he is up. Its
    strokes are in the same place either way, so the only honest place to ask is inside it.
    """
    shown = dict(state=state, level=0.0, elapsed=None if state == overlay.IDLE else 12.0)
    _settle(ov, **shown)
    frame = ov.render(phase=10.0, **shown).astype(float)
    cx, cy = (round(v) for v in ov.switches["wake"])
    r = round(ov.btn_r * 0.5) + 2
    box = frame[cy - round(r * 0.55) : cy - round(r * 0.15), cx - 3 : cx + 3]
    return float((box[:, :, :3].sum(axis=2) * box[:, :, 3] / 255).mean())


@pytest.mark.parametrize("state", STATES)
def test_the_microphone_says_what_the_tap_will_do(state: str) -> None:
    """Filled while he is up, hollow while he is down, and that is now the whole of the message.

    The control this replaces had GO TO SLEEP or WAKE UP written across a third of the panel, and
    the fill was a detail on top of the word. There is no word any more, so a fill that stopped
    tracking `awake` would leave the panel with nothing at all saying which way the switch goes.
    """
    ov = _panel()
    hollow, filled = _mic_core(ov, overlay.IDLE), _mic_core(ov, overlay.LISTENING)
    assert filled > hollow * 2, "the microphone looks the same up as it does down"
    assert (_mic_core(ov, state) > (hollow + filled) / 2) == overlay.awake(state)
    # ...and the same question the kiosk asks before deciding to start or stop, so what the
    # switch shows and the action behind it cannot drift apart.
    assert overlay.session_up(state) == (state not in (overlay.IDLE, overlay.ERROR))


def test_the_shutter_does_not_change_under_you() -> None:
    # The one control that means the same thing in every state, so it has to look the same in
    # every state. Byte-for-byte: it sits far enough inside the panel that the border's glow
    # never reaches it, so there is nothing legitimate to differ.
    ov = _panel()
    box = ov.hitboxes.shutter
    seen = set()
    for state in STATES:
        shown = dict(state=state, level=0.0, elapsed=None if state == overlay.IDLE else 12.0)
        _settle(ov, **shown)
        crop = ov.render(phase=10.0, **shown)[box.y : box.bottom, box.x : box.right]
        seen.add(crop.tobytes())
    assert len(seen) == 1, "the shutter changed with a state it has nothing to do with"


def test_no_caption_names_a_button_that_is_not_there() -> None:
    """A caption may name a control, and then the control has to be on the panel.

    The bug this invites, once: the resting line said "tap SESSION to begin" for as long as the
    tab said SESSION, and would have gone on saying it afterwards. It said "tap WAKE UP" until
    the line became a snore.

    Nothing on this panel carries a word now - the three controls are a face and two glyphs - so
    the rule has got stricter rather than going away: a caption that shouts anything is naming
    something nobody can find.
    """
    for state, caption in overlay.CAPTIONS.items():
        shouts = re.findall(r"[A-Z][A-Z ]+[A-Z]", caption)
        assert not shouts, f"the {state} line names {shouts[0]!r}, and no control has a word"


@pytest.mark.parametrize("state", STATES)
def test_every_state_has_a_sentence_and_a_face(state: str) -> None:
    # There used to be a word for each of these as well, in a corner of its own, and dropping it
    # is what makes the other two load-bearing rather than decorative: the sentence is now the
    # only thing that *names* what he is doing, and the face is the only thing that shows it.
    #
    # MOODS is the one that fails silently: a state missing from it does not raise, it falls back
    # to the sleeping face, and the panel simply stops saying anything about that state at all.
    assert state in overlay.MOODS, "no mood for this state - the eye would go to sleep in it"
    assert state == overlay.ERROR or state in overlay.CAPTIONS


# ---------------------------------------------------------------- the moods


def test_only_the_broken_face_holds_still() -> None:
    # The invariant the whole design rests on, stated on the table rather than on the pixels:
    # every mood but the faulted one has something about it that moves. The sleeping face is in
    # here now too - it breathes - and its own, much narrower rule is the test below.
    for state, mood in overlay.MOODS.items():
        # The gaze is in here because it is a way of moving: a mood that only looked around
        # would otherwise pass this test as "still" while visibly not being.
        moves = bool(mood.spin or mood.swell or mood.scan or mood.blink_s or mood.voice
                     or mood.gaze or mood.look_x or mood.look_y)
        assert moves == (state != overlay.ERROR), state


def test_the_sleeping_face_moves_slowly_and_by_moving() -> None:
    """He turns and his iris breathes, and he does neither of them by changing brightness.

    Both halves are the requirement rather than a preference, and the second one is here because
    it was got wrong: the sleeping face first shipped as a still drawing behind a slow fade,
    which is a lamp on a dimmer and not a creature. Brightness is what this panel says *state*
    with - the border, the caption, the wake cell all breathe in light - so a face doing it too
    says nothing, and says it over the one thing that was supposed to be a face.

    Slow is the other half. Every number here is the gentlest of its kind in the table, which is
    what separates a creature asleep from one at work.
    """
    mood = overlay.MOODS[overlay.IDLE]
    assert mood.spin, "the rings hold still - the ticks are in the same place in every frame"
    assert mood.swell, "the iris holds still"
    assert not mood.blink_s, "a shut-and-open lid is not what a sleeping face does"
    assert not (mood.scan or mood.voice), "he is asleep, not hunting or listening"
    still = (overlay.IDLE, overlay.ERROR)
    working = [m for state, m in overlay.MOODS.items() if state not in still]
    assert mood.breath_s > max(m.breath_s for m in working), "he breathes faster than a working eye"
    # His *steady* rate, which is not what you see - the wander runs well past it, and is meant
    # to. What this pins is that the thing it wanders around is the gentlest in the table.
    assert abs(mood.spin) < min(abs(m.spin) for m in working), "he turns faster than a working eye"
    # ...and, like every other period on this panel, out of step with all of them - see
    # WAKE_PERIOD_S. His is the longest, so it can only lock by being a multiple of one of them.
    for period in (overlay.WAKE_PERIOD_S, overlay.RIM_PERIOD_S, overlay.BREATH_PERIOD_S):
        assert mood.breath_s % period > 1e-6, f"his breath locks to the {period}s one"


def _rates(mood: eye.Mood, share: float, ring: int, seconds: float = 300.0) -> list[float]:
    """How fast one ring is turning, degree per second, sampled across *seconds* of its wander."""
    step = 0.05
    n = int(seconds / step)
    at = [eye.wander(i * step, mood.spin, share, mood.sway, ring) for i in range(n)]
    return [(b - a) / step for a, b in zip(at, at[1:], strict=False)]


RING_SET = (
    ("knurl", eye.KNURL_SPIN, 0),
    ("castellation", eye.CASTLE_SPIN, 1),
    ("dots", eye.DOT_SPIN, 2),
)


def test_the_sleeping_rings_change_their_minds() -> None:
    """Every ring speeds up, falls back and turns over, and no two of them do it together.

    What a gear train cannot do, and the complaint this answers: rings on fixed multiples of one
    rate do turn, but the set only ever reaches arrangements it has reached before, and a few
    seconds in the eye stops seeing it as movement at all. Every assertion here is false of the
    version that shipped before it, which is the whole reason they are worth writing down.
    """
    mood = overlay.MOODS[overlay.IDLE]
    assert mood.sway > 1.0, "under 1 a ring's wander never outruns its own rate, so it never turns"
    for name, share, ring in RING_SET:
        rates = _rates(mood, share, ring)
        assert min(rates) < 0 < max(rates), f"the {name} only ever turn one way"
        own = abs(mood.spin * share)
        assert max(rates) > 2 * own, f"the {name} never get away from their own rate"
        # ...and it is a wander and not a judder: no ring ever runs faster than a face that is
        # actually paying attention, or this stops reading as sleep.
        assert max(abs(r) for r in rates) < abs(overlay.MOODS[overlay.LISTENING].spin) * 2.0


def test_the_sleeping_rings_are_not_a_gear_train() -> None:
    # Two rings whose rates keep a fixed ratio are one mechanism however oddly it is geared, and
    # the arrangement they hold comes round again. These do not: the ratio between them wanders
    # over a range, which is what "the movement never looks the same" actually amounts to.
    mood = overlay.MOODS[overlay.IDLE]
    ticks = _rates(mood, eye.KNURL_SPIN, 0)
    dots = _rates(mood, eye.DOT_SPIN, 2)
    ratios = [t / d for t, d in zip(ticks, dots, strict=True) if abs(d) > 0.5]
    assert len(ratios) > 100, "not enough of the run to say anything"
    assert max(ratios) - min(ratios) > 1.0, "the rings hold a fixed ratio - this is a gear train"
    # The dotted ring is the slowest of the three on paper and must not be the slowest in fact,
    # or the wander is decoration on top of a fixed order rather than a rearrangement of it.
    assert max(dots) > max(ticks) * 0.5, "the slow ring never gets to overtake anything"


def test_a_mood_that_asks_for_no_wander_gets_the_turn_it_always_had() -> None:
    # sway defaults to 0 and every other mood leaves it there, so this is what keeps the change
    # to the sleeping face from quietly rewriting the other nine.
    for phase in (0.0, 3.3, 91.7):
        for share in (eye.KNURL_SPIN, eye.CASTLE_SPIN, eye.DOT_SPIN, eye.SCAN_SPIN):
            assert eye.wander(phase, 7.0, share, 0.0, 0) == pytest.approx(7.0 * share * phase)
    assert all(m.sway == 0.0 for state, m in overlay.MOODS.items() if state != overlay.IDLE)


def test_the_states_do_not_all_look_the_same() -> None:
    # A table of nine identical rows would pass every other test in this file.
    assert len({tuple(vars(m).values()) for m in overlay.MOODS.values()}) >= 6
    assert len({m.tint for m in overlay.MOODS.values()}) >= 3, "the eye should change colour"


def test_only_the_listening_face_opens_to_your_voice() -> None:
    # SPEAKING hears its own output on the meter, so it may lean on level a little; nothing that
    # is not listening to a room should react to one at all.
    for state, mood in overlay.MOODS.items():
        if state not in (overlay.LISTENING, overlay.SPEAKING):
            assert mood.voice == 0.0, f"{state} opens its iris at a noise it is not listening to"
    assert overlay.MOODS[overlay.LISTENING].voice > overlay.MOODS[overlay.SPEAKING].voice


def test_a_mood_travels_to_the_next_one_rather_than_snapping() -> None:
    a, b = overlay.MOODS[overlay.IDLE], overlay.MOODS[overlay.LISTENING]
    assert a.lerp(b, 0.0) == a
    assert a.lerp(b, 1.0) == b
    half = a.lerp(b, 0.5)
    assert a.aperture < half.aperture < b.aperture
    for i in range(3):  # the colour travels too, or waking reads as a different creature
        assert min(a.tint[i], b.tint[i]) <= half.tint[i] <= max(a.tint[i], b.tint[i])
    assert half.tint not in (a.tint, b.tint)


def test_the_eye_eases_between_moods_over_the_clock_it_is_given() -> None:
    engine = eye.EyeEngine(50, 2, overlay.SCREEN, overlay.MOODS[overlay.IDLE])
    engine.look(overlay.IDLE, overlay.MOODS[overlay.IDLE], 0.0)
    listening = overlay.MOODS[overlay.LISTENING]
    assert engine.look(overlay.LISTENING, listening, 0.0) == overlay.MOODS[overlay.IDLE], (
        "the frame the state changes on is still the old mood - the clock starts there"
    )
    part = engine.look(overlay.LISTENING, listening, eye.MOOD_EASE_S / 2)
    assert part != listening, "it arrived in one frame"
    assert part.aperture > overlay.MOODS[overlay.IDLE].aperture, "...and it did not set off"
    assert engine.look(overlay.LISTENING, listening, eye.MOOD_EASE_S * 2) == listening


# ---------------------------------------------------------------- the blink


def _blinks(every: float, seconds: float, step: float = 1 / 200) -> list[float]:
    return [i * step for i in range(int(seconds / step)) if eye.blink(i * step, every) < 0.5]


def _starts(every: float, seconds: float, step: float = 1 / 200) -> list[float]:
    shut = [eye.blink(i * step, every) < 0.5 for i in range(int(seconds / step))]
    return [i * step for i, (a, b) in enumerate(zip(shut, shut[1:], strict=False)) if b and not a]


def test_a_mood_with_no_blink_never_blinks() -> None:
    assert all(eye.blink(i / 50, 0.0) == 1.0 for i in range(5000))


def test_a_blink_shuts_the_eye_and_travels_rather_than_snapping() -> None:
    sweep = [eye.blink(i / 500, 4.4) for i in range(500 * 30)]
    assert min(sweep) == pytest.approx(0.0, abs=1e-3)
    assert max(sweep) == 1.0
    # A raised cosine, not a square wave - a lid that switched between two pictures would read
    # as a dropped frame rather than as a blink.
    assert any(0.05 < v < 0.95 for v in sweep)


def test_he_blinks_often_enough_to_be_alive_and_seldom_enough_not_to_nag() -> None:
    blinks = len(_starts(4.4, 60.0))
    assert 8 <= blinks <= 20, f"{blinks} blinks a minute is not a face"


def test_a_blink_survives_a_panel_running_at_25_fps() -> None:
    frames = [eye.blink(i / 25.0, 4.4) for i in range(25 * 30)]
    runs, run = [], 0
    for v in frames:
        run = run + 1 if v < 0.5 else 0
        runs.append(run)
    assert max(runs) >= 3, "a blink that lands on fewer than three frames will not be seen"


def test_the_blinks_are_not_a_metronome() -> None:
    at = _starts(4.4, 300.0)
    gaps = [round(b - a, 2) for a, b in zip(at, at[1:], strict=False)]
    assert len(set(gaps)) > 1, "every gap is the same length"
    assert min(gaps) > 1.5, f"two blinks {min(gaps)}s apart is a twitch"
    assert len(set(gaps[:8])) > 2, "the gaps merely alternate between two values"


def test_no_blink_is_clipped_by_the_edge_of_its_own_window() -> None:
    # A blink that ran past its window would be cut off half shut and the eye would jump back
    # open. Asserted through the function rather than against the constants, and for a mood far
    # faster than any in the table, because the clamp inside blink() is what has to hold.
    for every in (0.6, 1.4, 4.4, 9.0):
        for window in range(60):
            start = window * every
            n = max(4, int(every * 500))
            inside = [eye.blink(start + i / 500, every) for i in range(n)]
            assert min(inside) == pytest.approx(0.0, abs=2e-3), f"{every}s window {window}"
            assert inside[0] == 1.0, f"{every}s window {window} starts mid-blink"


def test_the_eye_never_stalls_however_long_the_panel_has_been_up() -> None:
    # A Pi's monotonic clock is its uptime and this panel is left running for weeks - long enough
    # that a float64 losing its last digits would quietly freeze the lid open. What has to hold
    # is not a particular count, which the drift varies by one deliberately, but that he is still
    # blinking at a living rate and still all the way shut.
    for base in (0.0, 86_400.0, 1_000_000.0, 5_000_000.0):
        sweep = [eye.blink(base + i / 200, 4.4) for i in range(60 * 200)]
        shut = [v < 0.5 for v in sweep]
        blinks = sum(1 for a, b in zip(shut, shut[1:], strict=False) if b and not a)
        assert 8 <= blinks <= 20, f"{blinks} blinks a minute at {base:,.0f}s of uptime"
        assert min(sweep) == pytest.approx(0.0, abs=1e-3), f"it stopped closing at {base}"


# ---------------------------------------------------------------- the iris


def test_the_iris_never_leaves_its_range_whatever_the_mood_asks_for() -> None:
    engine = eye.EyeEngine(50, 2, overlay.SCREEN, overlay.MOODS[overlay.IDLE])
    absurd = eye.Mood(tint=overlay.GREEN, aperture=0.9, swell=0.9, breath_s=0.7, voice=0.9)
    for mood in (*overlay.MOODS.values(), absurd):
        for i in range(200):
            for level in (0.0, 0.5, 1.0, 4.0, -1.0):
                assert 0.0 <= engine.aperture(mood, i / 7, level) <= 1.0


def test_your_voice_opens_the_listening_iris() -> None:
    engine = eye.EyeEngine(50, 2, overlay.SCREEN, overlay.MOODS[overlay.IDLE])
    mood = overlay.MOODS[overlay.LISTENING]
    quiet = engine.aperture(mood, 2.0, 0.0)
    assert engine.aperture(mood, 2.0, 1.0) > quiet + 0.1


def test_the_breath_comes_back_round_and_stays_inside_its_swell() -> None:
    # No blink in this one: a blink would shut the iris mid-sweep and swamp the swell it is
    # measuring. What the blink does is asserted on its own, above.
    mood = replace(overlay.MOODS[overlay.SPEAKING], blink_s=0.0)
    engine = eye.EyeEngine(50, 2, overlay.SCREEN, mood)
    sweep = [engine.aperture(mood, i * mood.breath_s / 64, 0.0) for i in range(64)]
    assert max(sweep) - min(sweep) == pytest.approx(mood.swell, abs=1e-2)
    for t in (0.15, 1.9, 86_400.15):
        assert eye.breath(t + mood.breath_s, mood.breath_s) == pytest.approx(
            eye.breath(t, mood.breath_s), abs=1e-6
        )


def test_a_mood_with_no_breath_does_not_divide_by_it() -> None:
    assert eye.breath(3.7, 0.0) == 0.0


# ---------------------------------------------------------------- on the panel


def _face(ov: overlay.Overlay, frame: np.ndarray) -> np.ndarray:
    """Just him, cropped square out of a rendered frame."""
    cx, cy = ov.eye
    r = ov.eye_r
    return frame[cy - r : cy + r, cx - r : cx + r]


def _lit(crop: np.ndarray) -> int:
    """Opaque phosphor in a crop. The alpha test is not optional: the chrome layer carries a
    green RGB under fully transparent pixels, so counting colour alone counts the whole sky."""
    return int(((crop[:, :, :3].astype(int).sum(axis=2) > 300) & (crop[:, :, 3] > 150)).sum())


def _asleep(ov: overlay.Overlay, state: str, detail: str = "") -> list[np.ndarray]:
    shown = dict(state=state, level=0.0, detail=detail, elapsed=None)
    _settle(ov, **shown)  # the eye is *meant* to travel between moods, and then hold
    return [ov.render(phase=100.0 + i * 0.31, **shown) for i in range(40)]


def _glow(crop: np.ndarray) -> float:
    """How much light a crop is putting out. Brightness, not coverage.

    What the invite changes is the *colour* of the glyph and its bezel, not how many pixels they
    cover - so a count of lit pixels says nothing at all about it: both ends of the breath are
    opaque phosphor and both pass any threshold loose enough to include the dim one. That is not
    a hypothetical; counting is what this measured first, and it reported a control that was
    plainly breathing as holding still.
    """
    return round(float((crop[:, :, :3].astype(float).sum(axis=2) * crop[:, :, 3] / 255).mean()), 3)


def _line_box(ov: overlay.Overlay) -> tuple[slice, slice]:
    """Everywhere the caption bubble can reach: its tallest, plus the tail, out to its far edge.

    Both ends are the bubble's own and neither is the panel's, because it now has a neighbour at
    each: his swell is a few pixels left of the tail's tip, and the microphone is a few pixels
    right of where the longest sentence stops. Either one dragged into this band is something
    that moves for its own reasons, being measured as though it were the line.
    """
    top = int(ov.caption_bottom - overlay.CAPTION_LINES * ov.caption_h)
    bottom = int(ov.caption_bottom) + ov.caption_tail + 1
    left = max(int(ov.caption_left - ov.caption_lean), ov.eye[0] + ov.shoulder + 1)
    return slice(top, bottom), slice(left, int(ov.caption_right) + 2)


def test_only_three_things_move_while_he_is_asleep() -> None:
    # The headline. Every animation on this panel is gated, and this is the one assertion that
    # notices when a new one is not - it is how the caption's breath was caught running at IDLE
    # and quietly pulsing a resting panel. Three exceptions and no more: the WAKE UP cell, which
    # may beckon because it is the only thing left to press; his own face, which may move because
    # he is asleep rather than off; and his line, which is a snore and has dots that walk.
    #
    # A fault gets none of the three, and that is what keeps this honest - blanking the same
    # regions in both states would leave nobody watching the pixels they are drawn on.
    ov = _panel()
    keep = ov.hitboxes.wake
    cx, cy, r = *ov.eye, ov.eye_r
    rows, cols = _line_box(ov)
    for state, detail in ((overlay.IDLE, ""), (overlay.ERROR, "OpenAI rejected the API key")):
        frames = [f.copy() for f in _asleep(ov, state, detail)]
        for f in frames:
            f[keep.y : keep.bottom, keep.x : keep.right] = 0
            if state == overlay.IDLE:
                f[cy - r : cy + r + 1, cx - r : cx + r + 1] = 0
                f[rows, cols] = 0
        moved = [i for i, f in enumerate(frames) if not np.array_equal(f, frames[0])]
        assert not moved, f"{state} moved outside those at frames {moved[:5]}"


def test_his_line_snores_while_he_is_asleep() -> None:
    # The dots were gated on there being a session, so a sleeping panel's line was a printed
    # label. It says a snore now, and a snore that holds still is not one.
    ov = _panel()
    assert overlay.CAPTIONS[overlay.IDLE].endswith(overlay.BUSY_MARK), "it would never animate"
    rows, cols = _line_box(ov)
    shown = dict(state=overlay.IDLE, level=0.0, elapsed=None)
    _settle(ov, **shown)
    step = overlay.DOT_PERIOD_S / (overlay.CAPTION_DOTS + 1)
    bare = ov.render(phase=step / 2, **shown)[rows, cols]
    full = ov.render(phase=3 * step + step / 2, **shown)[rows, cols]
    assert not np.array_equal(bare, full), "the dots do not walk while he is asleep"
    # ...and the line still does not brighten and dim doing it. The eye gave that up - brightness
    # is how this panel says which state it is in - and a line that pulses under a face that has
    # stopped is the same mistake one row further down.
    def ink(phase: float) -> int:
        """The brightest thing in the line. Its own text, in practice - nothing else out here is
        anywhere near phosphor at full - so it is exactly GREEN unless the breath is sinking it.

        Not `alpha == CAPTION_ALPHA`: PIL's glyph blending lands a pixel or two short of the ink
        it was asked for, so that matches nothing at all and quietly measures an empty array.
        """
        crop = ov.render(phase=phase, **shown)[rows, cols]
        return int(crop[:, :, :3][crop[:, :, 3] >= 200].astype(int).sum(axis=1).max())

    # Whole dot periods apart, so the same dots are lit in every sample and the breath is the
    # only thing left that could differ - and it runs at half the dot period, which puts these
    # alternately at the top and the bottom of it.
    assert len({ink(i * overlay.DOT_PERIOD_S) for i in range(4)}) == 1, "the line breathes"


RING_BAND = 0.75  # ...as a fraction of his radius: outside anything the gaze can reach


def _rings(crop: np.ndarray) -> np.ndarray:
    """Just his ring set - the band outside anything his gaze can move.

    Load-bearing, and not obviously so. Both "the rings turn" tests below work by rendering two
    frames a whole breath apart and asserting they differ, having first pinned the aperture equal
    so the iris cannot explain it. Since he gained a gaze that no longer leaves rotation as the
    only explanation: the optic slides even with every ring frozen, so a whole-face comparison
    passes on the gaze alone and says nothing about whether anything turns. Verified - freeze
    spin and scan and the face crops still differ, while this band does not.

    The optic reaches eye.IRIS + eye.GAZE_SHIFT = 0.705 of him at full lean, so 0.75 clears it.
    """
    r = crop.shape[0] // 2
    ys, xs = np.ogrid[-r:r, -r:r]
    return crop[np.hypot(ys, xs) > RING_BAND * r]


def _peak(crop: np.ndarray) -> int:
    """The brightest opaque pixel inside his rim. His own rim, in practice: it is the one ring
    drawn at the mood's tint undiluted, so it is exactly the tint in every frame unless something
    is dimming him.

    Inside the *disc*, not the square :func:`_face` cuts. The corners of that square reach past
    him into the tab row's chrome, which is drawn in full phosphor and never moves - brighter
    than anything of his, so a peak taken over the square is a constant that would sit there
    unchanged while he faded to nothing. This has already caught one measurement out.
    """
    r = crop.shape[0] // 2
    ys, xs = np.ogrid[-r:r, -r:r]
    inside = (ys**2 + xs**2 <= r**2) & (crop[:, :, 3] > 200)
    return int(crop[:, :, :3][inside].astype(int).sum(axis=1).max())


def test_he_turns_and_breathes_while_he_is_asleep() -> None:
    # What "asleep and breathing" was asked to look like, on the pixels rather than on the table.
    ov = _panel()
    asleep = dict(state=overlay.IDLE, level=0.0, elapsed=None)
    _settle(ov, **asleep)
    mood = overlay.MOODS[overlay.IDLE]
    # The rings, measured a whole breath apart so the iris is in exactly the same place in both
    # frames and rotation is the only thing left that can differ. Two arbitrary phases would pass
    # on the breath alone and say nothing at all about whether anything turns.
    first, second = 10.0, 10.0 + mood.breath_s
    assert ov.engine.aperture(mood, first, 0.0) == pytest.approx(
        ov.engine.aperture(mood, second, 0.0)
    ), "pick two phases a whole breath apart, or this test is about the iris"
    a = _face(ov, ov.render(phase=first, **asleep))
    b = _face(ov, ov.render(phase=second, **asleep))
    assert not np.array_equal(_rings(a), _rings(b)), "the ring set held still while he slept"
    # The iris, which is what actually breathes: it swells and shrinks, and stays the whole time
    # on the open side of the threshold the pupil is drawn on, so it grows rather than blinking
    # into existence once a breath.
    sweep = [ov.engine.aperture(mood, i * mood.breath_s / 24, 0.0) for i in range(24)]
    assert max(sweep) - min(sweep) == pytest.approx(mood.swell, abs=1e-2)
    assert min(sweep) > 0.08, "the pupil pops in and out rather than breathing"
    # ...and he does none of it by getting brighter and darker. Brightness is how this panel says
    # what state it is in; a face that pulses is competing with its own border, and the first
    # attempt at this was nothing but that pulse.
    peaks = {
        _peak(_face(ov, ov.render(phase=i * mood.breath_s / 60, **asleep))) for i in range(60)
    }
    assert len(peaks) == 1, f"his brightness pulses over a breath: {sorted(peaks)}"


def test_the_spark_always_covers_a_whole_pixel() -> None:
    """The rule that keeps his brightest pixel from flickering as his gaze drags it about.

    Stated on the arithmetic rather than sampled off the pixels, because sampling for it is what
    let it through: the spark fails only at a narrow band of sub-pixel alignments - 4 of 576
    positions swept at eye_r 40 with a floor of 1.2 - so a render test would have to be enormous
    to catch it reliably and would still only be evidence, not the rule.

    The rule: a disc covers a whole panel pixel only if all four of that pixel's corners are
    inside it, and the worst case puts the disc's centre on a corner, sqrt(2) from the nearest
    pixel's far corner. Anything below that is luck - measured, a floor of 1.0 passes every one
    of those 576 positions and 1.2 fails four of them, so "smaller" does not even mean "worse".

    What this catches: lowering SPARK_FLOOR, or making the spark a fraction of something small
    enough that the floor stops binding. Either turns a sleeping face into one that dims as it
    looks around, which is the single thing the sleeping face is not allowed to do.
    """
    assert eye.SPARK_FLOOR >= math.sqrt(2), "the spark can land without covering a whole pixel"
    # ...and the floor has to actually bind, in the units it is written in. The pupil threshold
    # this module used to carry was in oversampled units and read as if it were in panel ones,
    # which made it fire four times too early; this is the same mistake waiting to be made again.
    for radius in (28, 40, 60, 90):
        iris = eye.at(radius) * eye.IRIS
        for aperture in (0.0, 0.1, 0.2, 0.5, 1.0):
            hole = iris * (eye.HOLE_MIN + (eye.HOLE_MAX - eye.HOLE_MIN) * aperture)
            spark = max(eye.SPARK_FLOOR * eye.SUPERSAMPLE, hole * eye.SPARK) / eye.SUPERSAMPLE
            assert spark >= math.sqrt(2), (
                f"at eye_r {radius}, aperture {aperture}, the spark is {spark:.2f} panel px"
            )


def _glyph_box(ov: overlay.Overlay, name: str) -> tuple[slice, slice]:
    """Just the glyph inside a switch, which is the part that wears a colour."""
    cx, cy = (round(v) for v in ov.switches[name])
    r = round(ov.btn_r * 0.5) + 2
    return slice(cy - r, cy + r), slice(cx - r, cx + r)


def _invite(ov: overlay.Overlay, phase: float) -> float:
    """How bright the microphone glyph's strokes are, off a rendered frame.

    The brightest pixel in the box rather than its mean: ImageDraw does not anti-alias, so the
    strokes are exactly the colour they were drawn in, and a mean would be dragged around by the
    bezel swelling behind them - which moves the other way.
    """
    rows, cols = _glyph_box(ov, "wake")
    frame = ov.render(state=overlay.IDLE, level=0.0, elapsed=None, phase=phase)
    crop = frame[rows, cols]
    return float(crop[crop[:, :, 3] > 150][:, 1].max())


def test_the_way_out_glows_while_he_is_asleep() -> None:
    # It matters more than it did. The control this replaces had WAKE UP written across a third
    # of the panel and the breath was a flourish on top of it; this one is a microphone in a
    # corner with nothing written anywhere, so the breath is now most of how anybody finds it.
    ov = _panel()
    box = ov.hitboxes.wake
    lit = [_glow(f[box.y : box.bottom, box.x : box.right]) for f in _asleep(ov, overlay.IDLE)]
    assert max(lit) > min(lit), "the microphone held still - nothing invites the tap"


def test_the_glow_swells_rather_than_flashing() -> None:
    # A swell, not a blink: a panel flashing at you across a workshop is an alarm, and a control
    # that switches between two brightnesses reads as a fault light rather than as an invitation.
    # Same raised-cosine argument the caption's breath makes.
    ov = _panel()
    sweep = [_invite(ov, i * overlay.WAKE_PERIOD_S / 24) for i in range(24)]
    steps = {round(v) for v in sweep}
    assert len(steps) > 8, f"it steps rather than swelling: {sorted(steps)}"
    assert min(sweep) > 0.55 * max(sweep), "it goes dark at the bottom of the breath"
    assert max(sweep) > 1.15 * min(sweep), "the swell is too slight to notice"
    for t in (0.0, 1.3, 86_400.7):  # ...and it comes back round
        assert eye.breath(t + overlay.WAKE_PERIOD_S, overlay.WAKE_PERIOD_S) == pytest.approx(
            eye.breath(t, overlay.WAKE_PERIOD_S), abs=1e-6
        )


def _glyph_hue(ov: overlay.Overlay, phase: float) -> str:
    """Whether the microphone glyph's strokes are nearer the phosphor or nearer the accent."""
    rows, cols = _glyph_box(ov, "wake")
    frame = ov.render(state=overlay.IDLE, level=0.0, elapsed=None, phase=phase)
    crop = frame[rows, cols].astype(float)
    px = crop[(crop[:, :, 3] > 200) & (crop[:, :, :3].sum(axis=2) > 250)][:, :3]
    seen = px.mean(axis=0) / px.mean(axis=0).sum()
    near = {
        name: float(np.abs(seen - np.array(c) / sum(c)).sum())
        for name, c in (("phosphor", overlay.GREEN_MID), ("accent", overlay.WHITE))
    }
    return min(near, key=near.get)  # type: ignore[arg-type]


def _word_colour(ov: overlay.Overlay, state: str, phase: float) -> np.ndarray:
    """The microphone glyph, as a colour, normalised so brightness is out of the question.

    It used to be the word underneath it. There is no word, so the glyph carries the state on its
    own - which is a stronger claim than the one this made before, not a weaker one.
    """
    rows, cols = _glyph_box(ov, "wake")
    shown = dict(state=state, level=0.0, elapsed=None if state == overlay.IDLE else 12.0)
    _settle(ov, **shown)
    crop = ov.render(phase=phase, **shown)[rows, cols].astype(float)
    px = crop[(crop[:, :, 3] > 200) & (crop[:, :, :3].sum(axis=2) > 250)][:, :3]
    assert len(px), f"nothing lit on the microphone in {state}"
    return px.mean(axis=0) / px.mean(axis=0).sum()


def _nearest(seen: np.ndarray, *options: tuple[str, tuple[int, int, int]]) -> str:
    near = {n: float(np.abs(seen - np.array(c) / sum(c)).sum()) for n, c in options}
    return min(near, key=near.get)  # type: ignore[arg-type]


@pytest.mark.parametrize("state", STATES)
def test_the_button_wears_the_state_it_is_in(state: str) -> None:
    # It is the only cell on the row carrying a state, so it is the only one that takes a colour
    # - and it has to do it in every state, not only the ones somebody happened to look at.
    # Asleep is the exception and the reason for it: green is the floor the glow lifts off.
    ov = _panel()
    seen = _word_colour(ov, state, phase=overlay.WAKE_PERIOD_S)
    if state == overlay.IDLE:
        assert _nearest(seen, ("resting", overlay.GREEN_MID), ("state", overlay.WHITE)) == "resting"
    else:
        assert _nearest(
            seen, ("resting", overlay.GREEN_MID), ("state", overlay.HALOS[state])
        ) == "state", f"the word stayed green in {state}"


def test_the_glow_changes_colour_and_not_only_brightness() -> None:
    # It breathes towards the accent, which is a promise as well as a signal: the button wears
    # the colour the whole screen turns when you press it. Brightness alone was the complaint
    # the accent was introduced to answer, and it would be the same complaint here.
    ov = _panel()
    assert _glyph_hue(ov, overlay.WAKE_PERIOD_S) == "phosphor", "it starts somewhere else"
    assert _glyph_hue(ov, overlay.WAKE_PERIOD_S * 1.5) == "accent", "it only got brighter"


def test_a_fault_does_not_beckon() -> None:
    # A red panel with a green button pulsing at you is a machine asking to be prodded rather
    # than read, and a fault has something to say on the line under the picture.
    ov = _panel()
    box = ov.hitboxes.wake
    lit = [_glow(f[box.y : box.bottom, box.x : box.right])
           for f in _asleep(ov, overlay.ERROR, "OpenAI rejected the API key")]
    assert len(set(lit)) == 1


def test_the_rings_turn_while_he_is_awake() -> None:
    # A whole breath apart, so the iris is in exactly the same place in both frames and the only
    # thing left that can differ is the rotation. Comparing two arbitrary phases would pass on
    # the breath alone and say nothing at all about the rings.
    ov = _panel()
    mood = overlay.MOODS[overlay.LISTENING]
    first, second = 10.0, 10.0 + mood.breath_s
    assert ov.engine.aperture(mood, first, 0.0) == pytest.approx(
        ov.engine.aperture(mood, second, 0.0)
    ), "pick two phases a whole breath apart, or this test is about the iris"
    _settle(ov, **AWAKE)
    a = _face(ov, ov.render(phase=first, **AWAKE))
    b = _face(ov, ov.render(phase=second, **AWAKE))
    assert not np.array_equal(_rings(a), _rings(b)), "the ring set held still"


def test_the_rings_do_not_all_turn_together() -> None:
    # Rings that agree read as one printed disc. The counter-rotations are the whole reason it
    # reads as a mechanism, so they are stated here rather than left to whoever tunes them.
    rates = (eye.KNURL_SPIN, eye.CASTLE_SPIN, eye.DOT_SPIN)
    assert len(set(rates)) == len(rates)
    assert min(rates) < 0 < max(rates), "nothing counter-rotates"


def _rim_profile(ov: overlay.Overlay, phase: float) -> np.ndarray:
    """Brightness all the way round his rim, one sample a degree."""
    hunting = dict(state=overlay.SEARCHING, level=0.0, elapsed=12.0)
    frame = ov.render(phase=phase, **hunting).astype(float)
    cx, cy = ov.eye
    a = np.radians(np.arange(360))
    xs = np.round(cx + ov.eye_r * np.cos(a)).astype(int)
    ys = np.round(cy + ov.eye_r * np.sin(a)).astype(int)
    px = frame[ys, xs]
    return px[:, :3].sum(axis=1) * px[:, 3] / 255


def test_the_scan_arc_sweeps_the_rim_while_he_is_hunting() -> None:
    # A radar sweep: a bright trace on a faint ring. Both halves are asserted, because either one
    # alone passes for the wrong reason - a ring that is bright all the way round still "moves" a
    # little as the thicker trace goes by, and a trace no brighter than the ring is not a trace.
    ov = _panel()
    _settle(ov, state=overlay.SEARCHING, level=0.0, elapsed=12.0)

    def trace(phase: float) -> tuple[float, int]:
        ring = _rim_profile(ov, phase)
        lit = ring > (ring.max() + ring.min()) / 2
        assert lit.any(), "nothing bright anywhere on the rim of a searching eye"
        # Angular centre of the bright band, taken as a vector so it survives wrapping past 360.
        a = np.radians(np.flatnonzero(lit))
        centre = np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360
        return float(centre), int(lit.sum())

    where, span = trace(10.0)
    scan = overlay.MOODS[overlay.SEARCHING].scan
    assert span < 300, f"the whole rim is lit - there is no trace, just a bright ring ({span} deg)"
    assert 0.4 * scan < span < 2.0 * scan, f"the trace is {span} deg against a scan of {scan:.0f}"
    # A quarter of a second, not a whole one: at this mood's spin the trace goes round very nearly
    # once a second, so sampling a second apart would find it back where it started.
    moved = (trace(10.25)[0] - where) % 360
    assert 15 < moved < 345, f"the trace sat still ({moved:.0f} degrees in a quarter second)"


def test_the_iris_is_narrower_asleep_than_awake() -> None:
    ov = _panel()
    asleep = dict(state=overlay.IDLE, level=0.0, elapsed=None)
    _settle(ov, **asleep)
    dozing = _face(ov, ov.render(phase=10.0, **asleep))
    _settle(ov, **AWAKE)
    open_ = _face(ov, ov.render(phase=10.0, **AWAKE))
    assert _lit(open_) > _lit(dozing), "an attending iris is not wider than a dozing one"
    # ...and on the table, where the pupil's own size comes from.
    assert overlay.MOODS[overlay.IDLE].aperture < overlay.MOODS[overlay.LISTENING].aperture


def test_the_pupil_widens_with_your_voice() -> None:
    ov = _panel()
    _settle(ov, **AWAKE)
    shown = dict(state=overlay.LISTENING, elapsed=12.0, phase=10.0)
    quiet = _lit(_face(ov, ov.render(level=0.0, **shown)))
    loud = _lit(_face(ov, ov.render(level=1.0, **shown)))
    assert loud > quiet


def test_awake_is_a_different_colour_and_not_just_a_brighter_green() -> None:
    # The whole point of the accent. Dim green and bright green are the same colour to anyone
    # more than a pace away, so a panel that said "awake" by getting brighter did not say it.
    # Measured on the border, which is the one thing meant to be read from across a workshop.
    ov = _panel()

    def rim(state: str) -> np.ndarray:
        shown = dict(state=state, level=0.0, elapsed=None if state == overlay.IDLE else 12.0)
        _settle(ov, **shown)
        row = ov.render(phase=0.0, **shown)[0].astype(float)
        px = row[row[:, 3] > 200][:, :3]
        return px.mean(axis=0) / max(1.0, px.mean(axis=0).sum())  # hue, with brightness divided out

    asleep = rim(overlay.IDLE)
    for state in (overlay.LISTENING, overlay.SEARCHING, overlay.CONNECTING, overlay.ERROR):
        apart = float(np.abs(rim(state) - asleep).sum())
        assert apart > 0.1, f"{state} is the same hue as asleep, only brighter ({apart:.3f})"


def test_the_live_readouts_wear_the_accent_and_the_furniture_does_not() -> None:
    # A panel where everything is an accent has none. The brand, the rules and the two tabs that
    # do not change stay phosphor whatever he is doing.
    ov = _panel()
    shown = dict(state=overlay.LISTENING, level=0.8, elapsed=73.0)
    _settle(ov, **shown)
    frame = ov.render(phase=10.0, **shown)

    def wears(box: tuple[int, int, int, int]) -> str:
        """Whether the brightest thing in a box is nearer the phosphor or nearer the accent.

        Hue, with brightness divided out - which is the whole point of the accent. Asking "is it
        blue" would have been a question about one particular accent rather than about this one.
        """
        crop = frame[box[1] : box[3], box[0] : box[2]].astype(float)
        px = crop[(crop[:, :, 3] > 200) & (crop[:, :, :3].sum(axis=2) > 250)][:, :3]
        assert len(px), f"nothing lit in {box}"
        seen = px.mean(axis=0) / px.mean(axis=0).sum()
        near = {
            name: float(np.abs(seen - np.array(c) / sum(c)).sum())
            for name, c in (("phosphor", overlay.GREEN), ("accent", overlay.WHITE))
        }
        return min(near, key=near.get)  # type: ignore[arg-type]

    clock_right, _, meter_right = ov._readouts(False)
    meter, clock = ov.row_top, ov.row_bottom
    assert wears(
        (int(ov._meter_x(meter_right)), meter - 10, int(meter_right), meter + 10)
    ) == "accent"
    assert wears(
        (int(clock_right - ov._clock_w), clock - 12, int(clock_right), clock + 12)
    ) == "accent"
    # The caption's marker takes the accent and its sentence does not - a running line in aqua
    # over a live camera is harder to read than the same line in phosphor. The slab hangs off
    # ov.caption_right and grows leftwards, so the marker is found by measuring back from there.
    inset = round(8 * ov.scale)
    marker_w = int(ov.font_caption.getlength(overlay.MARKER))
    at = int(ov.caption_left) + inset  # the bubble grows rightwards from a fixed left edge
    top = int(ov.caption_y) - 6  # caption_y is the bottom line's centre, and this one fits on it
    assert wears((at, top, at + marker_w, top + 14)) == "accent"
    assert wears((at + marker_w, top, at + marker_w + 60, top + 14)) == "phosphor"
    # ...and the furniture: the shutter's aperture, which means the same thing in every state and
    # so wears the panel's own phosphor in all of them.
    rows, cols = _glyph_box(ov, "shutter")
    assert wears((cols.start, rows.start, cols.stop, rows.stop)) == "phosphor"


def test_the_accent_belongs_to_the_same_tube_as_the_phosphor() -> None:
    """The rule three accents were tried against, written down so a fourth need not repeat them.

    A blue-cyan fifty degrees round the wheel fought the green - near enough to be compared with
    it and far enough to argue. An aqua at thirty got on with it. The tube's own white is eleven
    degrees off and does not read as green at all, because what separates it is saturation.

    So the constraint is not a hue distance, which would have thrown out the answer. It is that
    the further round the wheel an accent goes the paler it has to get, and that it must be
    plainly distinguishable from the phosphor whichever way it got there.
    """
    import colorsys

    def hue_sat(c: tuple[int, int, int]) -> tuple[float, float]:
        h, sat, _ = colorsys.rgb_to_hsv(*[v / 255 for v in c])
        return h * 360, sat

    def chroma(c: tuple[int, int, int]) -> np.ndarray:
        return np.array(c, dtype=float) / sum(c)

    def budget(c: tuple[int, int, int]) -> float:
        h, sat = hue_sat(c)
        return abs((h - hue_sat(overlay.GREEN)[0] + 180) % 360 - 180) * sat

    apart = float(np.abs(chroma(overlay.WHITE) - chroma(overlay.GREEN)).sum())
    assert apart > 0.15, f"the accent does not read as different from the phosphor ({apart:.2f})"
    assert budget(overlay.WHITE) < 25, "too saturated to be that far round the wheel"
    # ...and the rule has to reject the one that was rejected, or it is a rubber stamp.
    assert budget((64, 226, 255)) > 25, "the rule would have let the blue-cyan through"


def test_the_rec_tag_is_red() -> None:
    # Red is what a record light is on every machine anybody has ever used, and that is worth
    # more than the panel's preference for its own green.
    ov = _panel()
    shown = dict(state=overlay.LISTENING, level=0.0, elapsed=12.0, recording=True)
    _settle(ov, **shown)
    frame = ov.render(phase=10.0, **shown)
    _, tags, _ = ov._readouts(True)
    cy = ov.row_bottom
    box = frame[cy - 8 : cy + 8, int(tags) : int(tags + ov._rec_w)].astype(float)
    px = box[box[:, :, 3] > 200][:, :3]
    seen = px.mean(axis=0) / px.mean(axis=0).sum()
    assert _nearest(seen, ("red", overlay.RED), ("green", overlay.GREEN)) == "red"


def test_he_changes_colour_with_what_he_is_doing() -> None:
    # Stated as what it claims rather than as "warmer": the accent used to be a colour and is now
    # the tube's own white, which is neutral, so a test that measured red-versus-green was really
    # a test about one particular accent. Chromaticity against the mood's own tint holds for any.
    ov = _panel()
    faces = {
        "asleep": overlay.IDLE,
        "waking": overlay.CONNECTING,
        "awake": overlay.LISTENING,
        "fault": overlay.ERROR,
    }
    tints = [(name, overlay.MOODS[state].tint) for name, state in faces.items()]
    assert len({c for _, c in tints}) == len(tints), "two of these faces are the same colour"
    for name, state in faces.items():
        shown = dict(state=state, level=0.0, elapsed=None if state == overlay.IDLE else 12.0)
        _settle(ov, **shown)
        crop = _face(ov, ov.render(phase=10.0, **shown)).astype(float)
        px = crop[(crop[:, :, 3] > 200) & (crop[:, :, :3].sum(axis=2) > 200)][:, :3]
        seen = px.mean(axis=0) / px.mean(axis=0).sum()
        assert _nearest(seen, *tints) == name, f"his {name} face is wearing another mood's colour"


def test_he_acknowledges_a_tap_without_going_photographic_negative() -> None:
    # The other two tabs invert under a thumb. His cell must not: half of him is over the picture
    # where there is no cell to invert, and a face in negative is not the same face.
    ov = _panel()
    _settle(ov, **AWAKE)
    rest = _face(ov, ov.render(phase=10.0, **AWAKE))
    held = _face(ov, ov.render(phase=10.0, pressed="eye", **AWAKE))
    assert not np.array_equal(rest, held), "a tap on his face changed nothing"
    assert _lit(held) > _lit(rest), "...and it should brighten him, not invert him"


def test_the_border_breathes_while_he_is_up() -> None:
    ov = _panel()
    top = lambda p: ov.render(phase=p, **AWAKE)[0]  # noqa: E731
    bright, sunk = top(0.0), top(overlay.RIM_PERIOD_S / 2)
    assert not np.array_equal(bright, sunk), "the rim held still"
    # It dims and never goes dark, and it never brightens past what _base already drew.
    lit = bright[:, :3].astype(int).sum(axis=1)
    dim = sunk[:, :3].astype(int).sum(axis=1)
    assert (dim <= lit).all(), "the breath brightened the border past its own state colour"
    assert dim[bright[:, 3] > 200].min() > 0


# ---------------------------------------------------------------- the room he takes up


@pytest.mark.parametrize(("width", "height"), SIZES)
def test_he_rides_the_ramp_and_his_rim_is_off_the_bezel(width: int, height: int) -> None:
    """Where he sits comes off the bracket's ramp and nothing else, at every window size.

    Sunk EYE_SEAT of his swell below the rail's centreline, which is what sets the angle the rail
    leaves the straight at. Move him off that line and the rail's swell stops being concentric
    with him: it would still be drawn round *something*, just not round his face.
    """
    ov = overlay.Overlay(width, height)
    bracket = ov.brackets["bl"]
    want = bracket.on_ramp(0.0, ov.shoulder * overlay.EYE_SEAT)
    assert ov.eye == (round(want[0]), round(want[1])), "he has come off his own ramp"
    # Against the inward glow's own reach rather than against PAD. PAD is what keeps *text* out
    # of the light; what has to hold for a face is that the border's bloom does not land on his
    # rim, which is a smaller number and the one this is actually about.
    reach = round((overlay.HALO_CORE + overlay.HALO_FALLOFF) * height)
    for axis, edge in ((0, width), (1, height)):
        assert ov.eye[axis] - ov.eye_r > reach, f"the rim glow is on his face at {width}x{height}"
        assert ov.eye[axis] + ov.eye_r < edge - reach, f"the rim glow is on his face at {width}"


@pytest.mark.parametrize(("width", "height"), SIZES)
def test_his_swell_stays_on_the_panel(width: int, height: int) -> None:
    # The rail goes round him at `shoulder`, so that is the circle that has to fit - not his rim.
    # A swell running off the edge is a bracket with a piece missing out of it.
    ov = overlay.Overlay(width, height)
    for axis, edge in ((0, width), (1, height)):
        assert 0 < ov.eye[axis] - ov.shoulder and ov.eye[axis] + ov.shoulder < edge


def _bubble_mask(band: np.ndarray) -> np.ndarray:
    """Where the caption bubble is, by its own two colours: the fill, and the outline round it.

    Alpha alone is not enough and never was. The shoulder's arcs are anti-aliased and a handful
    of their coverage values land on the fill's exactly; thin the fill and his plate, the corner
    brackets and half the chrome come in as well. The edge has to be in here too - it is the
    outermost pixel of the shape, so a mask of the fill alone reports the bubble a pixel further
    from his face than it is.
    """
    rgb, alpha = band[:, :, :3], band[:, :, 3]
    fill = (alpha == overlay.PLATE_ALPHA) & (rgb == np.array(overlay.SCREEN, np.uint8)).all(axis=2)
    edge = (alpha == overlay.BUBBLE_EDGE_ALPHA) & (
        rgb == np.array(overlay.GREEN_DIM, np.uint8)
    ).all(axis=2)
    return fill | edge


def test_the_bubble_comes_out_of_his_face_without_landing_on_it() -> None:
    """Both halves, because they pull against each other and only one of them is obvious.

    A bubble whose tail stops somewhere out over the picture is a bubble attached to nothing,
    which is the entire point of shaping it like one - so "well clear of him" is a failure here,
    not a safe default. The line used to hang off the right-hand edge and grow leftwards, which
    put the tail wherever the sentence happened to end.

    And he must still not be *under* it. Measured against his circle rather than a box round it,
    because the tail hangs below the bubble's bottom edge, which is exactly where his disc is
    widest - a bounding box would call that clear while the tail sat on his rim.
    """
    ov = _panel()
    cx, cy, r = *ov.eye, ov.shoulder
    top = int(ov.caption_bottom - overlay.CAPTION_LINES * ov.caption_h)
    bottom = int(ov.caption_bottom) + ov.caption_tail + 1
    for detail in (
        "listening — talk to me",
        "looking for the torque specification for an M8 stainless bolt into aluminium…",
        "x" * 200 + "…",  # nowhere to break, so this one runs to the far edge on both lines
    ):
        band = ov.render(
            state=overlay.SEARCHING, level=0.0, elapsed=12.0, detail=detail, phase=10.0
        )[top:bottom]
        mask = _bubble_mask(band)
        assert mask.any(), "no bubble on the panel at all - this would pass on a blank line"
        ys, xs = np.nonzero(mask)
        reach = np.hypot(xs - cx, ys + top - cy).min() - r
        # Measured against his *swell* and not his rim: the rail goes round him out there, and a
        # tail that stopped short of the rail would be coming out of the bracket's edge rather
        # than out of him. Both bounds are the gap itself, which is the one number here anybody
        # chose - the left edge is solved for so the tail's tip lands exactly on it.
        assert reach >= ov.caption_nose, (
            f"the bubble is {ov.caption_nose - reach:.1f}px inside its own gap: {detail[:30]!r}"
        )
        assert reach < 2 * ov.caption_nose, (
            f"its nearest point is {reach:.1f}px off his rim - the tail comes out of nothing"
        )


def test_a_long_caption_takes_a_second_line_rather_than_a_stub() -> None:
    # One line meant every sentence worth reading was cut to a stub ending in an ellipsis, in a
    # bubble with most of the picture still free beside it. Measured on the bubble's height,
    # which is the thing that has to grow, and upwards, because the bottom edge is where the
    # tail hangs from and the tail has his shoulder to clear.
    ov = _panel()
    shown = dict(state=overlay.SEARCHING, level=0.0, elapsed=12.0, phase=10.0)

    def box(detail: str) -> tuple[int, int]:
        rows = np.where(_bubble_mask(ov.render(detail=detail, **shown)).any(axis=1))[0]
        return int(rows.min()), int(rows.max())

    short = box("searching…")
    long_ = box("looking for the torque specification for an M8 stainless bolt into aluminium…")
    assert long_[1] == short[1], "the bubble grew downwards, onto the tab row"
    assert short[0] - long_[0] == pytest.approx(ov.caption_h, abs=2), "no second line appeared"
    # ...and the second line is used to say more, not to say the same amount twice as tall.
    lines = ov._wrap("a b c d e f g h i j k l m n o p q r s t u v w x y z", ov.font_caption, 60, 2)
    assert len(lines) == 2 and lines[-1].endswith("…"), "the wrap neither filled nor cut"
    assert ov._wrap("short", ov.font_caption, 200, 2) == ["short"], "a short line was padded out"


@pytest.mark.parametrize(("width", "height"), SIZES)
def test_the_face_is_tappable_where_the_face_is(width: int, height: int) -> None:
    # Half of him stands proud of the row. People tap what they can see.
    ov = overlay.Overlay(width, height)
    cx, cy = ov.eye
    assert ov.hitboxes.eye.contains(cx, cy)
    assert ov.hitboxes.eye.contains(cx, cy - ov.eye_r + 1)
    for other in (ov.hitboxes.shutter, ov.hitboxes.wake):
        assert not other.contains(cx, cy), "his face overlaps another tab's target"


@pytest.mark.parametrize(("width", "height"), SIZES)
def test_the_pod_is_centred_and_holds_its_worst_case(width: int, height: int) -> None:
    """The Mac and the Pi pick different faces, so this is a test and not a measurement.

    Its plate is cut once per window size and the tags come and go per state, so it has to be
    sized for the state that needs the most room - the tape running *and* the board throttled -
    or a warning that is not lit leaves a hole in the chrome the shape of itself.
    """
    ov = overlay.Overlay(width, height)
    clock_right, tags, meter_right = ov._readouts(False)
    assert ov.pod.center[0] in (width // 2, (width - 1) // 2), f"the pod is off centre at {width}"
    assert tags + ov._width("SIG", ov.font_micro, 0) < ov._meter_x(meter_right), "SIG hits the bar"
    widest = tags + ov._rec_w + ov._gap + ov._rec_w + ov._gap
    assert widest <= clock_right - ov._clock_w, f"both tags reach the clock at {width}x{height}"
    # ...and the whole of it is inside the flat the rail draws round it, at both rows.
    left, right = ov.spines["pod"][3][0], ov.spines["pod"][2][0]
    assert left + ov.rail_w / 2 < ov.pod.x and ov.pod.right < right - ov.rail_w / 2
    assert ov.row_bottom + ov.font_read.size / 2 < ov.pod.bottom - ov.rail_w / 2, "on the rail"
