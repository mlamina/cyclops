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
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from cyclops import eye, overlay, stats

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


@contextmanager
def _uncovered() -> Iterator[None]:
    """The sleeping face with the steel wound off it, so a measurement can reach him.

    Four of the tests here are about the face he wears while he is asleep - what colour it is,
    how open its iris is, whether its rings turn - and since the cover went across him none of
    that is on the panel at IDLE. That is the feature and not a regression, so they lift the lid
    rather than quietly become tests about a lump of steel. What the lid itself does is three
    tests of its own; this is what is behind it.

    The panel has to be built inside the block: the engine takes its resting mood at
    construction, and that is where the cover learns it starts open rather than shut.
    """
    was = overlay.MOODS[overlay.IDLE]
    overlay.MOODS[overlay.IDLE] = replace(was, cover=0.0)
    try:
        yield
    finally:
        overlay.MOODS[overlay.IDLE] = was


def _settle(ov: overlay.Overlay, **shown: object) -> None:
    """Run the crossfade into this mood out, so what follows is the mood and not the journey.

    Two renders at *different* phases, which is the whole trick: the engine starts its clock on
    the frame the state changes and moves on the ones after it, so settling with the same phase
    twice settles nothing.

    It settles two things now. The caption types itself onto the terminal from the frame its
    sentence turns up on, and MOOD_EASE_S * 4 clears TYPE_MAX_S four times over, so anything
    rendered after this has its line printed out as well as its mood arrived at.
    """
    ov.render(phase=0.0, **shown)  # type: ignore[arg-type]
    ov.render(phase=eye.MOOD_EASE_S * 4, **shown)  # type: ignore[arg-type]


# ---------------------------------------------------------------- the words


@pytest.mark.parametrize("state", STATES)
def test_the_state_still_has_somewhere_to_be_read(state: str) -> None:
    """The microphone used to carry it in the corner, filled while he was up. It is a heat gauge
    now, so the question this asks is the one that outlived it: the panel and the kiosk have to
    agree about what "a session is running" means, or the halo says one thing and the button
    beside the panel does another.
    """
    assert overlay.session_up(state) == (state not in (overlay.IDLE, overlay.ERROR))


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
        # would otherwise pass this test as "still" while visibly not being. `gaze` covers all of
        # it - it scales the anchor, every glance and the tremor, so a mood at zero is pinned dead
        # centre whatever places it names - and `drift` is the float a sleeping face moves by
        # while attending to nothing at all.
        moves = bool(mood.spin or mood.swell or mood.scan or mood.blink_s or mood.voice
                     or mood.gaze or mood.drift)
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
    # RIM_PERIOD_S. His is the longest, so it can only lock by being a multiple of one of them.
    for period in (overlay.RIM_PERIOD_S, overlay.BREATH_PERIOD_S, overlay.CURSOR_PERIOD_S):
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
    # sway defaults to 0, and this is what keeps a wander added to one mood from quietly
    # rewriting the rest. Two moods ask for one and they are named here rather than counted:
    # the sleeping face, whose rings turn over because a resting creature is not a gear train,
    # and the working one, whose rings turn over because a machine under load is not either.
    # A third arriving without a decision still trips this.
    wanders = {overlay.IDLE, overlay.WORKING}
    for phase in (0.0, 3.3, 91.7):
        for share in (eye.KNURL_SPIN, eye.CASTLE_SPIN, eye.DOT_SPIN, eye.SCAN_SPIN):
            assert eye.wander(phase, 7.0, share, 0.0, 0) == pytest.approx(7.0 * share * phase)
    assert all(m.sway == 0.0 for state, m in overlay.MOODS.items() if state not in wanders)
    assert all(overlay.MOODS[state].sway > 0.0 for state in wanders), "or the row says nothing"


def test_a_job_in_the_background_puts_the_working_face_up() -> None:
    """The bug this is here for: it shipped gated on IDLE alone, and a diagram is asked for in a
    conversation - so the state for the whole ninety seconds is DRAWING, and the one path anybody
    would actually take never showed the face."""
    assert overlay.working_over(overlay.DRAWING, True) == overlay.WORKING, "the commonest case"
    assert overlay.working_over(overlay.IDLE, True) == overlay.WORKING, "and the quiet one"


def test_nothing_running_leaves_every_state_alone() -> None:
    for state in (*STATES, overlay.WORKING):
        assert overlay.working_over(state, False) == state


def test_a_job_never_takes_the_face_off_something_he_is_doing_with_you() -> None:
    # Listening, speaking and looking outrank a job running behind them: those are things he is
    # doing *with* somebody, and a shut machine face in the middle of one is a lie about it.
    for state in (overlay.LISTENING, overlay.SPEAKING, overlay.LOOKING, overlay.SEARCHING,
                  overlay.CONNECTING, overlay.STARTING, overlay.STOPPING, overlay.ERROR):
        assert overlay.working_over(state, True) == state, state


def test_the_states_do_not_all_look_the_same() -> None:
    # A table of nine identical rows would pass every other test in this file.
    assert len({tuple(vars(m).values()) for m in overlay.MOODS.values()}) >= 6
    assert len({m.tint for m in overlay.MOODS.values()}) >= 3, "the eye should change colour"


def test_only_the_face_hearing_itself_opens_to_a_level() -> None:
    # SPEAKING hears its own output on the meter, and a face moving with what it is saying is a
    # face doing the right thing. Every other row, listening loudest among them, leaves it alone:
    # an iris that widens on your vowels reads as a mouth, and the level already has a meter.
    for state, mood in overlay.MOODS.items():
        if state != overlay.SPEAKING:
            assert mood.voice == 0.0, f"{state} opens its iris at a sound that is not its own"
    assert overlay.MOODS[overlay.SPEAKING].voice > 0.0, "his own voice should still show on him"


def _longest_still(seen: list[tuple[float, float]], reach: float = 0.2) -> int:
    """The longest unbroken run of frames he spends within *reach* of dead centre.

    A count of frames rather than a fraction of them, because a gaze crossing the middle on its
    way somewhere else is not a gaze resting in it - see the mood test below, where taking the
    fraction instead calls a hunt a rest.
    """
    best = run = 0
    for x, y in seen:
        run = run + 1 if math.hypot(x, y) < reach else 0
        best = max(best, run)
    return best


def test_every_place_a_mood_names_is_a_place_the_panel_knows() -> None:
    # The whole point of naming the places is that a typo becomes a thing that can be caught, and
    # this is where. An unknown name does not raise on the panel - gaze_at falls back to straight
    # ahead rather than dropping a frame at 25 fps - so nothing else would ever tell you.
    ov = _panel()
    for state, mood in overlay.MOODS.items():
        assert mood.look, f"{state} looks nowhere at all, not even ahead"
        for name in mood.look:
            assert name in ov.places, f"{state} looks at {name!r}, which is nowhere"


def test_the_eyes_own_landmarks_are_the_panels() -> None:
    # eye.LANDMARKS is what tools/eye_sheet.py renders against, and it is only honest while it
    # agrees with what the panel actually resolves off its own geometry. Move the bracket he rides
    # or the reticle he looks at and the sheet quietly stops being the eye that is shipping; this
    # is the thing that says so.
    ov = _panel()
    for name, where in eye.LANDMARKS.items():
        assert ov.places[name] == pytest.approx(where, abs=2e-3), name


def test_a_window_of_attention_outlasts_a_saccade_and_the_way_home() -> None:
    # A peek goes out and comes back inside its own window. Under eye.DART_FLOOR the way home is
    # shorter than one saccade, so he has not finished arriving before the next window starts and
    # the sequence stops being continuous. Asserted on the table rather than clamped in gaze_at,
    # because a mood that asks for a window that short is a mood that wants telling.
    for state, mood in overlay.MOODS.items():
        if mood.dart > 0.0:
            assert mood.dart_s >= eye.DART_FLOOR, f"{state} peeks faster than it can look back"


def test_he_looks_at_you_while_he_listens_and_comes_back_from_every_peek() -> None:
    """The row this whole model was written for, checked on the rendered vector rather than the

    table: he holds you, glances at the picture now and then, and *returns*. The old gaze could
    not have passed this - it wandered continuously and its targets had a reach floor, so looking
    straight at you was the one thing it never did.
    """
    mood = overlay.MOODS[overlay.LISTENING]
    ov = _panel()
    frame = ov.places[eye.FRAME]
    seen = [eye.gaze_at(i / 25.0, mood, ov.places) for i in range(25 * 240)]
    home = [g for g in seen if math.hypot(*g) < 0.2]
    assert len(home) / len(seen) > 0.75, "he is not looking at you for most of a conversation"
    # ...and when he does look away it is at the picture, not at some angle off a walk.
    out = max(seen, key=lambda g: math.hypot(*g))
    assert math.hypot(*out) > 0.4, "the peek never actually goes anywhere"
    cos = (out[0] * frame[0] + out[1] * frame[1]) / math.hypot(*out)
    assert cos > 0.97, "he peeks somewhere that is not the middle of the picture"


def test_the_hunting_face_never_settles_and_the_staring_one_never_leaves() -> None:
    # The two ends of the same three numbers, and the reason they are numbers and not branches.
    #
    # "Never settles" is a statement about *dwell* and not about proximity, and the difference is
    # the trap: a hunting eye crossing from one side of its travel to the other passes through the
    # middle, so a fifth of its frames are near centre while it is never once resting there. What
    # separates the two moods is how long a run near the middle lasts - 0.5 s of transit against
    # eight seconds of holding your eye.
    ov = _panel()
    hunt = [eye.gaze_at(i / 25.0, overlay.MOODS[overlay.SEARCHING], ov.places) for i in range(1500)]
    assert _longest_still(hunt) < 25, "a search that rests is not a search"
    stare = [eye.gaze_at(i / 25.0, overlay.MOODS[overlay.LOOKING], ov.places) for i in range(750)]
    frame = ov.places[eye.FRAME]
    assert all(math.hypot(g[0] - frame[0] * 0.85, g[1] - frame[1] * 0.85) < 0.15 for g in stare), (
        "a stare that wanders is not one"
    )


def test_the_gaze_never_jumps_further_than_a_saccade() -> None:
    # The whole model in one assertion: every move he makes is either a tremor or a ballistic
    # flick, and there is nothing sudden in between. It is here because the arithmetic that keeps
    # it true is the least obvious part of gaze_at - the way home starts from where he actually
    # *was* when he turned round, not from the place he was heading for, and taking the second
    # teleports him out and then eases him back.
    ov = _panel()
    for state, mood in overlay.MOODS.items():
        was, worst = None, 0.0
        for i in range(25 * 300):
            now = eye.gaze_at(i / 25.0, mood, ov.places)
            if was is not None:
                worst = max(worst, math.hypot(now[0] - was[0], now[1] - was[1]))
            was = now
        # the smoothstep's own slope, over the longest travel there is, plus room for the tremor
        bound = 1.5 / eye.SACCADE_S / 25.0 * 2.0 + 0.02
        assert worst <= bound, f"{state} teleports: {worst:.3f} against {bound:.3f}"


def test_the_faces_that_hold_still_hold_completely_still() -> None:
    # `gaze` at 0 has to pin him dead centre whatever else the row says, because three separate
    # things lean on it: the fault, the working face with its head down, and the tap
    # acknowledgement - which is the one gesture that must read the same from every mood.
    ov = _panel()
    for state in (overlay.ERROR, overlay.WORKING):
        mood = overlay.MOODS[state]
        assert {eye.gaze_at(p / 10.0, mood, ov.places) for p in range(600)} == {(0.0, 0.0)}, state
    held = replace(overlay.MOODS[overlay.LOOKING], gaze=0.0, dart=0.0, drift=0.0)
    assert {eye.gaze_at(p / 10.0, held, ov.places) for p in range(600)} == {(0.0, 0.0)}, (
        "a tap has to bring him back to you even from a mood anchored somewhere else"
    )


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


def test_the_listening_iris_does_not_move_with_the_room() -> None:
    # The level goes on arriving while he listens - the signal bar is drawn off the same number -
    # and the eye is the one thing that must ignore it. Somebody shouting at him should change
    # nothing about his face.
    engine = eye.EyeEngine(50, 2, overlay.SCREEN, overlay.MOODS[overlay.IDLE])
    mood = overlay.MOODS[overlay.LISTENING]
    quiet = engine.aperture(mood, 2.0, 0.0)
    assert engine.aperture(mood, 2.0, 1.0) == pytest.approx(quiet)


def test_his_own_voice_still_opens_the_talking_iris() -> None:
    # The knob is not dead, and this is the case it was right for all along: the level under
    # SPEAKING is his own output coming back, so the face moves with what the face is saying.
    engine = eye.EyeEngine(50, 2, overlay.SCREEN, overlay.MOODS[overlay.IDLE])
    mood = replace(overlay.MOODS[overlay.SPEAKING], blink_s=0.0)  # no blink to swamp the term
    quiet = engine.aperture(mood, 2.0, 0.0)
    assert engine.aperture(mood, 2.0, 1.0) > quiet


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
    """Just him: a square crop out of a rendered frame, blanked outside his own rim.

    The blanking is not tidiness. A square around a circle has four corners of something else in
    it, and since the collar went round him that something else is opaque steel wearing the
    chrome's green - so a mood's colour measured over the raw square is measured over a housing
    that never changes colour, and his red fault face reads as amber. His rim is the boundary
    every one of these measurements means by "him"; this is just it being said in the crop rather
    than assumed by the threshold.
    """
    cx, cy = ov.eye
    r = ov.eye_r
    crop = frame[cy - r : cy + r, cx - r : cx + r].copy()
    yy, xx = np.mgrid[0 : crop.shape[0], 0 : crop.shape[1]]
    crop[(xx - r + 0.5) ** 2 + (yy - r + 0.5) ** 2 > r * r] = 0
    return crop


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
    """The terminal's screen: everywhere the caption can put a lit pixel, and nowhere else.

    Both ends are the text's own and neither is the glass's, because the glass has a neighbour at
    each - its two side edges are buried under the mounts' bottom rails, and a rail dragged into
    this band is a piece of chrome being blanked as though it were the line. The rows are the
    screen's own two, which is what it is cut to.
    """
    return (
        slice(int(ov.caption_top), int(ov.caption_top) + overlay.CAPTION_LINES * ov.caption_h),
        slice(int(ov.caption_left) - 2, int(ov.caption_right) + 2),
    )


def test_only_two_things_move_while_he_is_asleep() -> None:
    # The headline. Every animation on this panel is gated, and this is the one assertion that
    # notices when a new one is not - it is how the caption's breath was caught running at IDLE
    # and quietly pulsing a resting panel. Two exceptions and no more: his own face, which may
    # move because he is asleep rather than off, and his line, which is a snore and has dots
    # that walk.
    #
    # It was three. The third was the microphone beckoning, which could because it was the only
    # thing left to press; the corner it was in holds a knob and a gauge now, and neither of them
    # has any business moving while nothing is happening to the thing it is reading.
    #
    # A fault gets neither of the two, and that is what keeps this honest - blanking the same
    # regions in both states would leave nobody watching the pixels they are drawn on.
    ov = _panel()
    cx, cy, r = *ov.eye, ov.eye_r
    rows, cols = _line_box(ov)
    for state, detail in ((overlay.IDLE, ""), (overlay.ERROR, "OpenAI rejected the API key")):
        frames = [f.copy() for f in _asleep(ov, state, detail)]
        for f in frames:
            if state == overlay.IDLE:
                f[cy - r : cy + r + 1, cx - r : cx + r + 1] = 0
                f[rows, cols] = 0
        moved = [i for i, f in enumerate(frames) if not np.array_equal(f, frames[0])]
        assert not moved, f"{state} moved outside those at frames {moved[:5]}"


def test_his_line_snores_while_he_is_asleep() -> None:
    # The cursor was gated on there being a session, so a sleeping panel's line was a printed
    # label. It says a snore now, and a snore that holds still is not one.
    ov = _panel()
    assert overlay.CAPTIONS[overlay.IDLE].endswith(overlay.BUSY_MARK), "it would never animate"
    rows, cols = _line_box(ov)
    shown = dict(state=overlay.IDLE, level=0.0, elapsed=None)
    _settle(ov, **shown)
    # Every phase down here is a breath past where it wants to be, and BREATH_PERIOD_S is one
    # breath and exactly two blinks - so both pulses are where they would have been anyway, and
    # the snore has finished being typed onto the screen rather than still arriving.
    half = overlay.CURSOR_PERIOD_S / 2
    bare = ov.render(phase=overlay.BREATH_PERIOD_S + half * 1.5, **shown)[rows, cols]
    full = ov.render(phase=overlay.BREATH_PERIOD_S + half / 2, **shown)[rows, cols]
    assert not np.array_equal(bare, full), "the cursor does not blink while he is asleep"
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

    # Whole cursor periods apart, so the cursor is showing in every sample and the breath is the
    # only thing left that could differ - and it runs at twice the cursor period, which puts these
    # alternately at the top and the bottom of it.
    assert len({ink(overlay.BREATH_PERIOD_S + i * overlay.CURSOR_PERIOD_S) for i in range(4)}) == (
        1
    ), "the line breathes"


RING_BAND = 0.75  # ...as a fraction of his radius: outside anything the gaze can reach


def _rings(crop: np.ndarray) -> np.ndarray:
    """Just his ring set - the band outside anything his gaze can move.

    Load-bearing, and not obviously so. Both "the rings turn" tests below work by rendering two
    frames a whole breath apart and asserting they differ, having first pinned the aperture equal
    so the iris cannot explain it. Since he gained a gaze that no longer leaves rotation as the
    only explanation: the optic slides even with every ring frozen, so a whole-face comparison
    passes on the gaze alone and says nothing about whether anything turns. Verified - freeze
    spin and scan and the face crops still differ, while this band does not.

    The optic reaches eye.IRIS + eye.GAZE_SHIFT = 0.730 of him at full lean, so 0.75 clears it.
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
    # Under the lid, which is where he does it now: the steel is across him at IDLE and holds
    # still, so this is a measurement of the face and not of what is in front of it. That the
    # cover hides all of this is its own test.
    with _uncovered():
        _he_turns_and_breathes()


def _he_turns_and_breathes() -> None:
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


def _nearest(seen: np.ndarray, *options: tuple[str, tuple[int, int, int]]) -> str:
    """Which of *options* a normalised colour is closest to. Hue, with brightness divided out."""
    near = {n: float(np.abs(seen - np.array(c) / sum(c)).sum()) for n, c in options}
    return min(near, key=near.get)  # type: ignore[arg-type]


# ------------------------------------------------------------ the two instruments


def _ring(ov: overlay.Overlay, frame: np.ndarray, name: str, lo: float, hi: float
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The lit pixels of dial *name* between *lo* and *hi* of its radius: dy, dx and colour.

    An annulus rather than a box, because a dial is a stack of concentric things and the only way
    to ask about one of them is to cut the ring it lives on: the bezel outside, the graduations
    under it, the scale at DIAL_TRACK, the pointer inside that and the reading below the hub. A
    box measures all five at once and can be satisfied by any of them.
    """
    cx, cy = (round(v) for v in ov.switches[name])
    span = ov.dial_span
    crop = frame[cy - span : cy + span + 1, cx - span : cx + span + 1].astype(float)
    ys, xs = np.mgrid[-span : span + 1, -span : span + 1]
    reach = np.hypot(ys, xs)
    lit = (
        (crop[:, :, 3] > 200)
        & (crop[:, :, :3].sum(axis=2) > 250)
        & (reach >= lo * ov.btn_r)
        & (reach <= hi * ov.btn_r)
    )
    return ys[lit], xs[lit], crop[:, :, :3][lit]


def _hand(ov: overlay.Overlay, frame: np.ndarray, name: str
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Just the pointer: everything lit between the hub and the tip, where nothing else is."""
    got = _ring(ov, frame, name, overlay.DIAL_HUB + 0.08, overlay.DIAL_HAND)
    assert len(got[0]), f"nothing lit between the hub and the tip of the {name} dial"
    return got


def _scale_lit(ov: overlay.Overlay, frame: np.ndarray, name: str) -> int:
    """How much of the dial's scale is lit, in pixels. The knob's whole reading, measured."""
    _, _, rgb = _ring(ov, frame, name, overlay.DIAL_TRACK - 0.08, overlay.DIAL_TRACK + 0.08)
    return int((rgb[:, 1] > 180).sum())


def _hand_angle(ov: overlay.Overlay, frame: np.ndarray, name: str) -> float:
    """Which way the pointer is pointing, in the degrees the overlay lays it out in.

    The mean position of the taper, which for a shape symmetrical about its own axis with nothing
    else in the annulus is a point on that axis.
    """
    ys, xs, _ = _hand(ov, frame, name)
    return math.degrees(math.atan2(ys.mean(), xs.mean())) % 360.0


def _hand_hue(ov: overlay.Overlay, frame: np.ndarray, name: str) -> str:
    """Which of the panel's colours the pointer is drawn in."""
    _, _, rgb = _hand(ov, frame, name)
    seen = rgb.mean(axis=0)
    return _nearest(
        seen / seen.sum(),
        ("ok", overlay.GREEN),
        ("warn", overlay.AMBER),
        ("hot", overlay.RED),
        ("held", overlay.WHITE),
    )


def _dialled(ov: overlay.Overlay, **kwargs: object) -> np.ndarray:
    shown = dict(state=overlay.IDLE, level=0.0, elapsed=None, phase=10.0)
    return ov.render(**{**shown, **kwargs})  # type: ignore[arg-type]


@pytest.mark.parametrize("level", [0, 25, 60, 100])
def test_the_knob_shows_where_it_has_been_turned_to(level: int) -> None:
    """A knob's whole job. The pointer says it once and the lit stretch of scale says it again,
    because on a 72 px disc across a bench the second one is what carries at all."""
    ov = _panel()
    frame = _dialled(ov, volume=level)
    want = overlay.DIAL_FROM + overlay.DIAL_SWEEP * level / 100.0
    assert _hand_angle(ov, frame, overlay.VOLUME) == pytest.approx(want % 360.0, abs=6.0)


def test_the_lit_scale_grows_with_the_level() -> None:
    ov = _panel()
    lit = [_scale_lit(ov, _dialled(ov, volume=v), overlay.VOLUME) for v in (0, 25, 60, 100)]
    assert lit == sorted(lit), f"the knob's scale did not follow the level: {lit}"
    assert lit[-1] > lit[0] * 3, "turning it all the way lit almost nothing"


def test_the_gauge_reads_the_board_and_not_the_session() -> None:
    """The needle is on the board's own scale: COOL_C empty, THROTTLE_C full, and the two colour
    breaks where cyclops.stats puts them - the same numbers the admin page's bar is given."""
    ov = _panel()
    for temp, band in ((stats.COOL_C, "ok"), (stats.WARN_C + 2, "warn"), (stats.HOT_C + 2, "hot")):
        frame = _dialled(ov, temp_c=temp)
        want = overlay.DIAL_FROM + overlay.DIAL_SWEEP * stats.temp_percent(temp) / 100.0
        assert _hand_angle(ov, frame, overlay.HEAT) == pytest.approx(want % 360.0, abs=6.0)
        assert _hand_hue(ov, frame, overlay.HEAT) == band, f"{temp} C did not read as {band}"


def test_a_dial_with_nothing_behind_it_is_not_a_dial_reading_zero() -> None:
    """No sink, no thermal zone: off the Pi both are None, and a pointer parked at the bottom of
    the scale would be a panel claiming silence and a cold board rather than admitting ignorance.
    """
    ov = _panel()
    frame = _dialled(ov, volume=None, temp_c=None)
    for name in (overlay.VOLUME, overlay.HEAT):
        ys, _, _ = _ring(ov, frame, name, overlay.DIAL_HUB + 0.08, overlay.DIAL_HAND)
        assert not len(ys), f"the {name} dial drew a pointer with no reading"


def test_neither_instrument_changes_with_the_session() -> None:
    """Byte for byte, in every state. A volume and a board temperature are true whether or not
    anybody is talking to him, and this is what keeps them out of the per-state bake: the two
    switches they replace *did* carry the state, and rebuilt a full-screen layer to say so.
    """
    ov = _panel()
    seen: dict[str, set[bytes]] = {overlay.VOLUME: set(), overlay.HEAT: set()}
    for one in STATES:
        shown = dict(state=one, level=0.0, elapsed=None if one == overlay.IDLE else 12.0,
                     volume=60, temp_c=58.0)
        _settle(ov, **shown)
        frame = ov.render(phase=10.0, **shown)  # type: ignore[arg-type]
        for name, box in ((overlay.VOLUME, ov.hitboxes.volume), (overlay.HEAT, ov.hitboxes.heat)):
            seen[name].add(frame[box.y : box.bottom, box.x : box.right].tobytes())
    assert [len(v) for v in seen.values()] == [1, 1], "a dial changed with the state"


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


# ------------------------------------------------------------ the steel across him


def _colour_of_him(crop: np.ndarray) -> float:
    """The share of the lit face inside his rim wearing his own colour rather than steel.

    Chromaticity and not brightness, for the reason :func:`_nearest` gives: a lid drawn dark
    would pass any threshold about how bright he is while still being his colour, and the
    question here is whether what is on the screen is him or the thing in front of him.
    """
    r = crop.shape[0] // 2
    ys, xs = np.ogrid[-r:r, -r:r]
    px = crop[:, :, :3][(ys**2 + xs**2 <= r**2) & (crop[:, :, 3] > 200)].astype(float)
    px = px[px.sum(axis=1) > 60]  # ...that is lit at all
    assert len(px), "nothing lit inside his rim"
    hue = px / px.sum(axis=1)[:, None]
    his = np.array(overlay.GREEN_MID) / sum(overlay.GREEN_MID)
    steel = np.array(eye.STEEL) / sum(eye.STEEL)
    return float(np.mean(np.abs(hue - his).sum(1) < np.abs(hue - steel).sum(1)))


def test_a_shut_cover_hides_the_whole_face() -> None:
    """Edge to edge, and nothing of him left but the light coming past the seams.

    The point of the whole thing, and the only version of it worth having: a cover that left his
    rings showing round the outside would be a lens cap drawn on top of a face rather than one in
    front of it. Measured as a share of his colour rather than as a count of pixels, because the
    steel is bright - a threshold on brightness passes a lid that is not covering anything.

    The second half is the price, and it is stated here so nobody has to rediscover it: with the
    steel down he holds *completely* still. He turns and breathes underneath it - that is
    `test_he_turns_and_breathes_while_he_is_asleep`, run with the lid lifted - and none of it is
    on the panel. Nothing in this drawing can move while it is shut without the movement being a
    change of brightness, which is the one thing the sleeping face has never been allowed to do.
    """
    ov = _panel()
    asleep = dict(state=overlay.IDLE, level=0.0, elapsed=None)
    _settle(ov, **asleep)
    lid = _face(ov, ov.render(phase=10.0, **asleep))
    # A twentieth, and most of what it does allow is the seam light rather than him: with the
    # seams drawn in steel instead of in his colour the same measurement reads a hundredth, and
    # that hundredth is the darkest metal on the disc, which is nearer the screen's green-black
    # than it is to steel. What this would catch is a ring of him left showing round the outside.
    assert _colour_of_him(lid) < 0.05, "his rings are showing round the outside of the cover"
    half = overlay.MOODS[overlay.IDLE].breath_s / 2
    assert np.array_equal(lid, _face(ov, ov.render(phase=10.0 + half, **asleep))), (
        "the shut cover moves - the steel is a lid, not a part of the face"
    )
    with _uncovered():
        bare = _panel()
        _settle(bare, **asleep)
        assert _colour_of_him(_face(bare, bare.render(phase=10.0, **asleep))) > 0.4, (
            "he is not there to be hidden - this test would pass over an empty bracket"
        )


def _luma(crop: np.ndarray) -> np.ndarray:
    """The lit face inside his rim as brightness, which is what "steel" is an argument about."""
    r = crop.shape[0] // 2
    ys, xs = np.ogrid[-r:r, -r:r]
    px = crop[:, :, :3][(ys**2 + xs**2 <= (r * RING_BAND) ** 2) & (crop[:, :, 3] > 200)]
    return px.astype(float) @ np.array([0.2126, 0.7152, 0.0722])


def test_the_shut_steel_is_lit_metal_and_not_grey_card() -> None:
    """It carries evidence of a light source: a shadowed end and a lit end, far apart.

    The whole of what separates a piece of metal in a room with a light in it from a grey disc
    with a spiral drawn on it. A face with nothing bright on it and nothing dark on it carries no
    evidence of a light at all, so the eye reads it as a paper cutout laid over a lit picture
    rather than as a lid in front of one.

    What is asserted is the SPREAD, not a near-black floor. That distinction is the whole history
    of this test: it demanded a second percentile under 22% for a round, on a critic's advice,
    and holding the metal to it is what made it look like glossy plastic - real polished steel is
    high-key, and the reference this is built against measures a median of 133 with a tenth
    percentile of 84, nowhere near black. The drawing sits at 131 and 104 against those.

    This is the assertion that fails if the shading is ever flattened back towards one mid grey,
    which is what it was for a round: 42% to 92% of full, a fifth of the range there is.
    """
    ov = _panel()
    asleep = dict(state=overlay.IDLE, level=0.0, elapsed=None)
    _settle(ov, **asleep)
    lit = _luma(_face(ov, ov.render(phase=10.0, **asleep)))
    dark, bright = (float(np.percentile(lit, p)) for p in (2, 98))
    assert bright > 0.82 * 255, "nothing on the steel catches the lamp"
    assert bright - dark > 0.45 * 255, "the steel is one grey; there is no light in the room"
    assert dark < 0.50 * 255, "nothing on the steel is turned away from the lamp"


def test_the_face_outshines_the_lid_while_there_is_still_a_face() -> None:
    """He is the brightest thing on the tile until the hole he is behind stops being one.

    The constraint the cover is only allowed to exist under. The eye is a display and not a
    machined object - metal inside his radius measurably killed the face the last time it was
    tried - so a near-white specular on the steel while the core is still in shot would make the
    lid the brightest part of a picture whose whole point is that the face is the lit part. See
    :meth:`cyclops.eye.Pen._arrived`, which holds both of the cover's whites back until it is
    home; this is what would notice if that were tuned away.

    Below a tenth of him the aperture has no face in it - the pupil is not even drawn - and past
    that the steel may be as bright as it needs to be.
    """
    r = 88  # his size on the panel, so this is the drawing that actually ships
    engine = eye.EyeEngine(r, 2, overlay.SCREEN, overlay.MOODS[overlay.LISTENING])
    engine.look(overlay.IDLE, overlay.MOODS[overlay.IDLE], 100.0)
    ys, xs = np.ogrid[-r : r + 1, -r : r + 1]
    far = ys**2 + xs**2
    for i in range(1, 33):
        phase = 100.04 + i * eye.COVER_SHUT_S / 32
        mood = engine.look(overlay.IDLE, overlay.MOODS[overlay.IDLE], phase)
        shut = engine.shut(mood, phase)
        hole = r * eye.buried(shut)
        under = (far <= r * r) & (far > (hole + 1) ** 2)
        if hole < 0.1 * r or not under.any():
            continue  # ...or there is no steel on the tile yet to out-shine anything
        tile = Image.new("RGBA", (2 * r + 1,) * 2, (*overlay.SCREEN, 255))
        engine.paint(tile, r, r, mood, phase, 0.0)
        lit = np.asarray(tile).astype(float)[:, :, :3] @ np.array([0.2126, 0.7152, 0.0722])
        him = lit[far <= (hole - 1) ** 2].max()
        steel = lit[under].max()
        assert him >= steel, f"the steel out-shines him at a hole {hole:.0f} px across"


def test_only_the_sleeping_face_is_behind_steel() -> None:
    # The non-negotiable: awake, there is no cover, and there is none at any uptime either. Not
    # "a thin one" - metal inside his radius is what killed the face the last time it was tried,
    # and the whole licence for this one is that it is gone while somebody is looking at him.
    assert {s for s, m in overlay.MOODS.items() if m.cover} == {overlay.IDLE}
    for state, mood in overlay.MOODS.items():
        if mood.cover:
            continue
        engine = eye.EyeEngine(50, 2, overlay.SCREEN, mood)
        for phase in (0.0, 0.37, 4.2, 9000.0):
            assert engine.shut(mood, phase) == 0.0, f"{state} has steel across it"


def test_the_cover_takes_its_time_and_gets_all_the_way_there() -> None:
    """Both ends reached exactly, nothing between them out of order, and neither one a wipe.

    What this catches is a curve rather than a look. An ease that misses its end leaves a
    hairline of him showing under a cover that is meant to be shut; one that steps arrives in a
    frame and reads as a cut; and a linear one is a wipe, which is the whole thing this design is
    not. The last two assertions are the ones with an opinion in them, and it is Marco's: both
    ends are slow and the travel is spent in the middle, so the set eases off the rim and settles
    into the seal instead of breaking hard and creeping home.
    """
    engine = eye.EyeEngine(50, 2, overlay.SCREEN, overlay.MOODS[overlay.LISTENING])
    steps = 40
    walk = [engine.shut(overlay.MOODS[overlay.IDLE], i * eye.COVER_SHUT_S / steps)
            for i in range(steps + 1)]
    assert (walk[0], walk[-1]) == (0.0, 1.0), "it starts part way across, or never arrives"
    assert all(b >= a for a, b in zip(walk, walk[1:], strict=False)), "it backs up"
    # The burst still has to be a movement rather than an arrival: at 25 fps this window is 40
    # frames, and a frame that crossed a fifth of the travel would read as a cut with a held set
    # on either side of it. The ends, meanwhile, have to be *held* and not merely slow - which is
    # the whole of what an exponential is here for, and what a smoothstep could not do.
    step = [b - a for a, b in zip(walk, walk[1:], strict=False)]
    assert max(step) < 0.15, "the middle is a cut, not a burst"
    assert sum(1 for v in step if v < 0.01) > steps // 3, "the ends are merely slow, not held"
    assert abs(walk[steps // 2] - 0.5) < 0.02, "it is faster at one end than the other"
    back = [engine.shut(overlay.MOODS[overlay.LISTENING],
                        eye.COVER_SHUT_S + i * eye.COVER_OPEN_S / steps)
            for i in range(steps + 1)]
    assert (back[0], back[-1]) == (1.0, 0.0), "it opens from somewhere else, or never clears him"
    assert all(b <= a for a, b in zip(back, back[1:], strict=False)), "it backs up on the way out"
    assert eye.COVER_OPEN_S < eye.COVER_SHUT_S, "waking is as much of a deliberation as sleeping"


def test_the_iris_is_narrower_asleep_than_awake() -> None:
    with _uncovered():  # the iris he has under the steel is still an iris
        ov = _panel()
        asleep = dict(state=overlay.IDLE, level=0.0, elapsed=None)
        _settle(ov, **asleep)
        dozing = _face(ov, ov.render(phase=10.0, **asleep))
        _settle(ov, **AWAKE)
        open_ = _face(ov, ov.render(phase=10.0, **AWAKE))
        assert _lit(open_) > _lit(dozing), "an attending iris is not wider than a dozing one"
    # ...and on the table, where the pupil's own size comes from.
    assert overlay.MOODS[overlay.IDLE].aperture < overlay.MOODS[overlay.LISTENING].aperture


def test_the_listening_pupil_holds_while_you_shout_at_it() -> None:
    # The whole face, not just the aperture number: nothing he is drawn out of - iris, bezel,
    # core - may move with the room while he is listening to it. The level is still arriving on
    # this frame; the signal bar is what is supposed to be reading it.
    ov = _panel()
    _settle(ov, **AWAKE)
    shown = dict(state=overlay.LISTENING, elapsed=12.0, phase=10.0)
    quiet = _face(ov, ov.render(level=0.0, **shown))
    loud = _face(ov, ov.render(level=1.0, **shown))
    assert np.array_equal(quiet, loud), "his face moved at a noise that is not his"


def test_the_talking_pupil_widens_with_his_own_voice() -> None:
    # And the same measurement where the gesture belongs: under SPEAKING the level is his own
    # output coming back off the speaker, so the face may move with what the face is saying.
    ov = _panel()
    talking = dict(state=overlay.SPEAKING, level=0.0, elapsed=12.0)
    _settle(ov, **talking)
    shown = dict(state=overlay.SPEAKING, elapsed=12.0, phase=10.0)
    quiet = _lit(_face(ov, ov.render(level=0.0, **shown)))
    loud = _lit(_face(ov, ov.render(level=1.0, **shown)))
    assert loud > quiet


def test_the_face_is_where_the_state_is_read() -> None:
    """Awake is a different COLOUR and not a brighter one, and the face is what says so.

    This rule used to be measured on the border - a green ring round the whole picture that took
    the session's colour and breathed. That was taken off on 2026-09-07: it was the last thing on
    the panel still drawn as a HUD rather than as a machine. The rule outlived it, because it was
    never about the border. Dim green and bright green are the same colour to anyone more than a
    pace away, so a panel that says "awake" by getting brighter has not said it - and the thing
    somebody actually looks at when they want to know is his face.

    Measured over his rim rather than the whole square, for the reason :func:`_face` gives: the
    housing around him never changes colour, and a mood measured over it is measured over steel.
    That is also why the sleeping face is measured with its cover lifted: shut, it IS steel, and
    the question here is what colour he goes when he wakes up rather than what the lid is made of.
    """
    with _uncovered():
        _hues_differ()


def _hues_differ() -> None:
    ov = _panel()

    def hue(state: str) -> np.ndarray:
        shown = dict(state=state, level=0.0, elapsed=None if state == overlay.IDLE else 12.0)
        _settle(ov, **shown)
        # Well past MOOD_EASE_S, and forwards: `look` eases from the phase the state arrived at,
        # so rendering at 0.0 after settling at 1.8 winds its clock back and every mood comes out
        # resting. That is what this test measured on its first draft, and it read four identical
        # faces without complaining once.
        crop = _face(ov, ov.render(phase=10.0, **shown)).astype(float)
        px = crop[(crop[:, :, 3] > 200) & (crop[:, :, :3].sum(axis=2) > 200)][:, :3]
        assert len(px), f"nothing lit on his face in {state}"
        return px.mean(axis=0) / max(1.0, px.mean(axis=0).sum())  # brightness divided out

    asleep = hue(overlay.IDLE)
    for state in (overlay.LISTENING, overlay.SEARCHING, overlay.CONNECTING, overlay.ERROR):
        apart = float(np.abs(hue(state) - asleep).sum())
        assert apart > 0.1, f"{state} is the same hue as asleep, only brighter ({apart:.3f})"


def test_the_live_readouts_wear_the_accent_and_the_furniture_does_not() -> None:
    # A panel where everything is an accent has none. The brand, the rules and the two tabs that
    # do not change stay phosphor whatever he is doing.
    ov = _panel()
    shown = dict(state=overlay.LISTENING, level=0.8, elapsed=73.0, volume=60, temp_c=58.0)
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
    meter = clock = ov.row
    assert wears(
        (int(ov._meter_x(meter_right)), meter - 10, int(meter_right), meter + 10)
    ) == "accent"
    assert wears(
        (int(clock_right - ov._clock_w), clock - 12, int(clock_right), clock + 12)
    ) == "accent"
    # The caption's marker takes the accent and its sentence does not - a running line in aqua
    # over a live camera is harder to read than the same line in phosphor. It is the first thing
    # on the terminal's first line, so it is found at the left edge of the screen's own text.
    marker_w = int(ov.font_caption.getlength(overlay.MARKER))
    at = int(ov.caption_left)  # the screen prints from its own left edge, always the same one
    top = int(ov.caption_y) - 6  # caption_y is the *first* line's centre, and this one fits on it
    assert wears((at, top, at + marker_w, top + 14)) == "accent"
    assert wears((at + marker_w, top, at + marker_w + 60, top + 14)) == "phosphor"
    # ...and the furniture: the volume knob, which reads the same number in every state and so
    # wears the panel's own phosphor in all of them.
    #
    # What is asked is the knob's *reading* - the face inside the bezel - and not the whole
    # hitbox. The hitbox used to be the same thing, because the bezel was a dull ring that never
    # crossed the brightness at which a pixel counts as lit; now that the instruments are turned
    # steel under the panel's lamp, a box round one averages the ring, and the rail in the box's
    # corners, in with the reading. Metal on this panel is grey by rule, so a box that includes
    # it can only ever come out grey; the question this test exists to ask is whether the
    # phosphor printed on the face went and took the accent, and the face is where to ask it.
    cx, cy = (round(v) for v in ov.switches[overlay.VOLUME])
    face = int(overlay.DIAL_BEZEL_IN * ov.btn_r / math.sqrt(2))  # the square inside the glass
    assert wears((cx - face, cy - face, cx + face, cy + face)) == "phosphor"


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
    cy = ov.row
    box = frame[cy - 8 : cy + 8, int(tags) : int(tags + ov._tag_w)].astype(float)
    px = box[box[:, :, 3] > 200][:, :3]
    seen = px.mean(axis=0) / px.mean(axis=0).sum()
    assert _nearest(seen, ("red", overlay.RED), ("green", overlay.GREEN)) == "red"


def test_he_changes_colour_with_what_he_is_doing() -> None:
    # Stated as what it claims rather than as "warmer": the accent used to be a colour and is now
    # the tube's own white, which is neutral, so a test that measured red-versus-green was really
    # a test about one particular accent. Chromaticity against the mood's own tint holds for any.
    with _uncovered():  # ...and the sleeping one from under its lid, which is not a colour of his
        _each_face_wears_its_own_colour()


def _each_face_wears_its_own_colour() -> None:
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


def _ink(band: np.ndarray) -> np.ndarray:
    """Lit text in a crop of the terminal's screen.

    The front itself is no use to measure. It is baked with the rest of the chrome and is the
    same object whatever is printed on it, which is the whole point of it - so what gets measured
    is the words.

    Two tests, and it needs both. Bright enough to be a letter, *and* the colour of one. Bright
    alone stopped working the day the case got a lamp in the top-left corner: the glare is put
    down in WHITE and a threshold cannot tell a lit letter from a lit corner. The colour can -
    phosphor is (86, 255, 140), which is green by a hundred and fifteen over its own red and
    blue, and the tube's white is green by fifteen. Nothing else on this panel sits between them.

    This replaces an upper bound on alpha, which used to be what separated a letter from chrome.
    That bound was a fact about the old flat well and died with it: the front's alpha across the
    caption band now runs 227 to 250 against a CAPTION_ALPHA of 245, so the rule it encoded threw
    away a third of the real text.
    """
    rgb = band[:, :, :3].astype(int)
    green = rgb[:, :, 1] - np.maximum(rgb[:, :, 0], rgb[:, :, 2])
    return (rgb.sum(axis=2) > 300) & (green > 60) & (band[:, :, 3] > 150)


def test_the_screen_stands_clear_of_both_mounts() -> None:
    """The whole of why this stopped being a slab: it is a monitor, and you can see all of it.

    It used to run from the middle of one mount's bottom rail to the middle of the other's, buried
    at both ends for the lower half of its depth, and that burial was what said "bolted in rather
    than laid on". It bought that sentence with the two ends of the shape. A monitor is a thing
    you can see the whole of - four corners, four edges - and ends that disappear under a bracket
    make it a slot again however round its corners are.

    So the case stands clear of both rails and a little ear bridges each gap instead. That is the
    same sentence said in something visible: a tab off the side of the chassis, landing on the
    mount's rail, with a bolt through where it lands.

    Both halves matter and pull against each other. No gap and the ends are buried again; a gap
    with nothing in it and the monitor is floating in the bay rather than mounted in it.
    """
    ov = _panel()
    half = ov.rail_w / 2.0
    for name, edge, sign in (("bl", ov.term.x, 1), ("br", ov.term.right, -1)):
        rail = ov._rail_at(name, ov.ear_y)
        gap = sign * (edge - rail) - half
        assert gap > 2, (
            f"the {name} mount's rail reaches within {gap:.1f}px of the case - the monitor is "
            "buried in it again rather than standing clear"
        )
        assert gap < ov.term.h / 2, (
            f"{gap:.1f}px between the case and the {name} mount is a monitor adrift in the bay, "
            "not one bracketed into it"
        )
    # ...and an ear across each of those gaps, touching the case at one end and the rail at the
    # other. An ear that reaches neither is a tab drawn beside the joint rather than making it.
    for ear, name, edge in zip(ov.ears, ("bl", "br"), (ov.term.x, ov.term.right), strict=True):
        near, far = min(ear.x, ear.right), max(ear.x, ear.right)
        assert near <= edge <= far, f"the {name} ear does not reach the case"
        assert near <= ov._rail_at(name, ov.ear_y) <= far, f"the {name} ear lands on nothing"


def test_the_caption_never_lands_on_the_moulding() -> None:
    """However long the sentence, and whatever it wraps to.

    The text is inset from the two rails rather than from the glass's own corners, which are half
    a rail further out and buried for the whole of the second line's height. Inset from the glass
    instead and a long caption starts underneath the left-hand bracket.
    """
    ov = _panel()
    rows, _ = _line_box(ov)
    # The whole of the glass, so that ink landing on the moulding is ink this can see. It used to
    # look thirty pixels outside the case, back when the case ran into both mounts and the thing
    # a long line could stand on was a bracket; the monitor stands clear of them now, so the only
    # thing left for a sentence to run onto is its own bezel.
    wide = slice(ov.tube.x, ov.tube.right)
    for detail in (
        "listening — talk to me",
        "looking for the torque specification for an M8 stainless bolt into aluminium…",
        "x" * 200 + "…",  # nowhere to break, so this one runs to the far edge on both lines
    ):
        shown = dict(state=overlay.SEARCHING, level=0.0, elapsed=12.0, detail=detail)
        _settle(ov, **shown)  # let the line finish printing; this is about where it lands
        band = ov.render(phase=10.0, **shown)[rows, wide]
        ink = _ink(band)
        assert ink.any(), "nothing on the screen at all - this would pass on a blank line"
        xs = np.nonzero(ink.any(axis=0))[0] + wide.start
        assert ov.caption_left <= xs.min() and xs.max() <= ov.caption_right, (
            f"{detail[:30]!r} put ink at {xs.min()}..{xs.max()}, outside "
            f"{ov.caption_left}..{ov.caption_right} - it is standing on the moulding"
        )


def test_a_long_caption_takes_a_second_line_rather_than_a_stub() -> None:
    # One line meant every sentence worth reading was cut to a stub ending in an ellipsis, with
    # most of the panel still free beside it. Measured on the ink now rather than on a slab: the
    # screen is two lines deep whatever is on it, so what has to grow is what is printed - and it
    # grows *downwards*, onto the second line, because a terminal prints from the top.
    ov = _panel()
    shown = dict(state=overlay.SEARCHING, level=0.0, elapsed=12.0)
    rows, cols = _line_box(ov)
    def box(detail: str) -> tuple[int, int]:
        _settle(ov, detail=detail, **shown)  # a line still arriving has no second line yet
        ink = _ink(ov.render(detail=detail, phase=10.0, **shown)[rows, cols])
        lines = np.where(ink.any(axis=1))[0]
        return int(lines.min()), int(lines.max())

    short = box("searching…")
    long_ = box("looking for the torque specification for an M8 stainless bolt into aluminium…")
    assert long_[0] == short[0], "the first line moved - the screen prints from its own top edge"
    assert long_[1] - short[1] == pytest.approx(ov.caption_h, abs=3), "no second line appeared"
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
    for other in (ov.hitboxes.volume, ov.hitboxes.heat):
        assert not other.contains(cx, cy), "his face overlaps an instrument's target"


@pytest.mark.parametrize(("width", "height"), SIZES)
def test_every_pod_is_centred_and_packed(width: int, height: int) -> None:
    """The Mac and the Pi pick different faces, so this is a test and not a measurement.

    There is one pod per number of tags and each is exactly as wide as what it shows, which is
    the whole of why the tags are laid out packed rather than into a reserved slot: a pod cut for
    its worst case had a tag-shaped hole in the middle of it whenever neither tag was lit, and
    that hole was wider than the meter.
    """
    ov = overlay.Overlay(width, height)
    for tags in range(3):
        box = ov.pod_boxes[tags]
        clock_right, tag_x, meter_right = ov._readouts(tags)
        assert box.center[0] in (width // 2, (width - 1) // 2), f"pod {tags} is off centre"
        assert ov._meter_x(meter_right) == box.x, f"pod {tags}: the meter is off its left edge"
        # Packed: what the tags need is exactly what is between the meter and the clock, and
        # with none lit the two stops either side of them collapse into one.
        room = clock_right - ov._clock_w - tag_x
        want = tags * ov._tag_w + (tags - 1) * ov._tag_gap + ov._stop if tags else 0
        assert round(room) == round(want), f"pod {tags} has {room - want:.0f}px going spare"
        # ...and the whole of it is inside the flat the rail draws round it.
        left, right = ov.pods[tags].spine[3][0], ov.pods[tags].spine[2][0]
        assert left + ov.rail_w / 2 < box.x and box.right < right - ov.rail_w / 2
    assert ov.row + ov.font_read.size / 2 < ov.pod.bottom - ov.rail_w / 2, "the clock hits the rail"
