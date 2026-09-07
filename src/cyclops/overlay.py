"""The kiosk's chrome: a Pip-Boy-style terminal bezel drawn over the live camera frame.

OpenCV can only draw Hershey stroke fonts, which look like a 1980s oscilloscope. Everything
here is rendered into one RGBA layer with PIL instead - real TrueType text, anti-aliased
shapes, a numpy-built halo and scanline field - and handed to :mod:`cyclops.kiosk` as a numpy
array to alpha-blend onto the frame. Geometry doubles as the hit-test map: every interactive
element returns its rectangle, so a tap can be resolved without a second layout.

The layout is two corner mounts, a status pod, a terminal and a picture. A mount's rail comes in
square to one panel edge, ramps across the corner at 45 degrees and lands square on the other,
and the two of them carry the three controls; the pod is that same shape turned inwards, hanging
off the middle of the top edge with the readouts in it; and the terminal is a monitor in the
middle of the bottom bay, bracketed to a mount at either end, with the caption on its glass. The
middle of the panel, which is what somebody holding a camera down a pipe is actually looking at,
is nothing but picture and a four-arc reticle on the lens axis.

The chrome covers 18% of the panel, against 29% for the strip and three-cell tab row this layout
replaced. That is a much thinner win than it was - it was 15% before the terminal arrived, which
took the caption off the middle of the picture and put eight per cent of chassis along the bottom
to do it - and it is only a win at all because of what *kind* of coverage each is. The tab row
was an opaque near-black bar. Everything here is the tube filter over the live picture, so the
room runs behind the metal as well as between it; the only parts you cannot see through are the
terminal's bezel, its glass and the wells the two dials are sunk into. Re-measure with
`Overlay(800, 480)._bracket_mask(0).mean()` rather than reasoning about it, which is how both of
the numbers on this line came to be wrong for a while.

What made that affordable was giving up the words. SNAP, WAKE UP and GO TO SLEEP were what
forced the controls into a row across the bottom: a cell has to be as wide as its label. Two
dials and a face need a disc each, and a disc can go in a corner. What the words used to say is
said by the ring in the button, by the border's colour, and by the terminal along the bottom -
which can say "searching the web…" where a word could only say SEARCH. The state word went the
same way and for the same reason: three things were already saying it better.

A bracket is drawn to look like one. The rail is a bar of steel rather than a stroke - a
round-edged section standing proud of the plate under the panel's one lamp
(:mod:`cyclops.material`), bright along the edge that faces it and dark along the edge that does
not, its shadow falling the other way - with hex-socket cap screws wherever it turns and
stiffeners across the deep corner. The plate itself is still see-through: the tube filter (a
phosphor wash over brushed gunmetal, corner shading and a scanline field) laid over the live
picture rather than an opaque bar, so the camera shows through the chrome as well as between it.
The border carries the session state in its colour and glows inwards from it, which is the one
thing that has to be readable across a workshop without reading any words - and it says so in
*hue* rather than in brightness, because dim green and bright green are the same colour to anyone
more than a pace away.

His eye is the one thing here that is a face rather than a readout, and it is what lets the rest
stay this terse: a glance at the bottom-left corner answers "is he there, and what is he up to",
so the readouts are left to spell it out only for whoever is close enough to read them. He is the
boot mark brought to life - :mod:`cyclops.eye` draws the splash's iris-inside-HUD-rings with the
rings turning and the iris breathing, and :data:`MOODS` says how, one row per state. Tapping him
opens what the box has kept, because what you ask a face is what it remembers.

His bracket's rail leaves the straight, goes round him and comes back. That shoulder is the
join: it is what makes him part of the bracket rather than a badge sitting on it, and it is why
his corner is the big one - a face wants room, and the two instruments opposite are bolted
straight through their rail and take a third of the space. Those two were a shutter and a
microphone until the box grew a button of its own, which does both without anybody having to
find a 12 mm target on glass; they are a volume knob and a heat gauge now, which is the pair of
things this panel could not otherwise be told or asked. While he is asleep the only thing on
this panel that moves is his own breath: no ring turns, nothing blinks, no readout changes, and
nothing beckons - there is nothing left in that corner to press. That is what makes any of the
rest of it read as awake: the *mechanism* stopping, rather than the creature holding its breath.

Everything that holds still while the state does - the halo, the brackets and their bolts, the
reticle, the pod's tags, the switches at rest - is built once and cached, keyed on the state. A
Pi rendering this at 25 fps has 40 ms for the whole loop and the camera wants most of them; what
is left for a frame here is a signal meter, a clock, a caption, one ring, the border line and the
eye. The terminal is baked with the rest of the chrome except for the line printed on it, whose
halation is 1.3 ms of that - measured on the Pi against the same frame drawn crisp-only, and
worth having at the price because it is what makes the glass read as lit rather than printed.
Every part of the eye moves, so none of it is cached at all; measured on the Pi, he is 10.5
ms of a 15.5 ms frame, which is the single largest thing this loop does and is meant to be - he is
the only part of the panel anybody looks at. He grew from r60 to r88 when he moved into the
corner and took about 3.5 ms with him, which is the whole of the difference between this and the
tab row; the board sat at 59 C and 0x0 throttled afterwards, so it is a price that is being paid
out of headroom rather than out of frames. The number is worth keeping honest, because it was
wrong here for a long time: this line once claimed a tenth of a millisecond, which was the figure
before he was ever supersampled. Re-measure with `deploy/push.sh && ssh cyclops@cyclops.local`
and a timing loop around `Overlay.render`, not by reasoning about it.
"""

from __future__ import annotations

import math
import sys
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from . import material
from .eye import (
    AHEAD,
    AWAY,
    BLINK_DRIFT,
    DIALS,
    FRAME,
    LANDMARKS,
    WORDS,
    WORK,
    EyeEngine,
    Mood,
    at,
    breath,
    linear,
    mix,
    smoothed,
    wide,
)
from .stats import HOT_C, WARN_C, temp_band, temp_percent

IDLE, CONNECTING, LISTENING, SPEAKING, LOOKING, SEARCHING, DRAWING, ERROR = (
    "idle",
    "connecting",
    "listening",
    "speaking",
    "looking",
    "searching",
    "drawing",
    "error",
)
# Optimistic states the kiosk shows the instant you tap, before the session agrees. Tearing a
# session down takes ~2.3 s, during which the controller still honestly reports "listening" -
# so without these the button looks dead for over two seconds after you press stop.
STARTING, STOPPING = "starting", "stopping"
# ...and the one state that is not about a session at all: something is running in the background
# with nobody in a conversation to narrate it - a drawing that outlived the tool call that asked
# for it, or the child naming and filing a session that has already ended (see cyclops.tasks).
# The panel used to show that as IDLE, so a box that was working for a minute wore a sleeping
# face and snored through it.
WORKING = "working"


def session_up(state: str) -> bool:
    """Is there a session at all - from the tap that starts one to the last file it writes?

    The panel's coarsest question: the caption breathes, the border breathes, the microphone
    fills in. The kiosk asks it too, so that what the switch shows and what the switch does can
    never drift apart - see ``_toggle_session``.
    """
    return state not in (IDLE, ERROR, WORKING)


def working_over(state: str, busy: bool) -> str:
    """*state*, or WORKING when something is running in the background that it does not cover.

    Two states get replaced and the second one is the whole reason this is a function rather than
    a comparison. IDLE is the obvious half: nobody is talking to him and the child is filing a
    session, so a panel that snores through it is lying. DRAWING is the half that was missed on
    the first attempt, and it is the commonest case by far - you ask for a diagram *in* a
    conversation, so the state during those ninety seconds is DRAWING and never IDLE, and gating
    on IDLE alone meant the one path anybody would actually take never showed the face.

    DRAWING is replaced rather than kept alongside because since the picture stopped blocking its
    tool call the two say the same thing: ``agent.drawing_active`` is true exactly while a picture
    is being made in the background, which is what a row in :mod:`cyclops.tasks` is. Given the
    same condition twice, the one with the face wins.

    Nothing else is touched. Listening, speaking and looking are things he is doing *with you*
    and outrank a job running behind them.
    """
    return WORKING if busy and state in (IDLE, DRAWING) else state


def awake(state: str) -> bool:
    """Is Cyclops himself up - eye open, listening, able to answer?

    Stricter than :func:`session_up` by exactly the teardown, which can run for a minute while
    the video is muxed. Both are right about their own half: the eye shuts on the tap because
    that is what you asked for, and the cell stays lit with the caption saying "saving the
    video…" because that is what is still true.
    """
    return state not in (IDLE, ERROR, STOPPING, WORKING)


# Phosphor palette. The chrome is one hue - the brand, the rules, the ticks, the two tabs that do
# not change, the words of the caption - because a panel where everything is a colour has none.
# What steps off it is the handful of things that only mean anything while a session is up, and
# they all wear the state's own accent. Asleep, none of them do, and the whole screen is green.
GREEN = (86, 255, 140)  # phosphor at full brightness: text, live chrome
GREEN_MID = (46, 176, 100)  # rules, dividers, glyphs at rest
GREEN_DIM = (32, 118, 70)  # the faintest thing still legible on the panel
AMBER = (255, 184, 60)
RED = (255, 86, 70)
# The one the tube gets while he is up. Asleep, the panel is a single hue and he is the same
# green as the furniture he sits in - bar the microphone, which breathes towards this to say what
# pressing it does; awake, the live parts step off the green.
#
# Not a second hue at all, in the end: the tube's own white. A single-phosphor screen driven hard
# blooms towards white with its own colour still in it, which is why this is 225,255,240 and not
# paper white, and it is why nothing here clashes - there is no second hue to argue with the
# first. A blue-cyan fifty degrees round the wheel was tried and fought the green; an aqua at
# thirty got on with it and still read as a decision. This reads as the same screen turned up.
#
# It is also the panel's existing way of shouting, promoted: the scan arc on a hunting eye and
# the REC tag both went pale long before this did, because on a one-hue tube pale is the only
# direction left.
WHITE = (225, 255, 240)
# And the one the machine gets while it works with nobody talking to it. The note above says a
# blue-cyan was tried as the *accent* and fought the green, and that still holds - which is why
# this is not one: it wears no chrome, tints no border and lights no readout. It is the eye's own
# colour and nothing else's, for the one state where he is not a face answering anybody but a
# mechanism running on its own, and a cold hue is what says that where the tube's white would
# only say "up". Kept dark and slightly green-shifted rather than a pure hue for the same reason
# WHITE is not paper white: this is one phosphor screen, and a colour with none of the screen in
# it reads as a sticker on the glass.
BLUE = (96, 168, 255)
SCREEN = (5, 15, 10)  # the green-black the brackets and their plates are made of
INK = (3, 11, 7)  # text on a filled tab

# The halo answers one question only from across a room - is the agent up? - and the state's
# accent answers it in colour rather than in brightness, which is the only half of it that
# survives the distance. Everything wearing this moves together: the border and its inward glow,
# the signal meter, the session clock, the caption's marker, and the microphone on the switch
# that is holding the session open. The rest of the chrome stays green, because a
# panel where everything is an accent has none.
HALOS = {
    IDLE: GREEN_DIM,
    # The border stays where a sleeping one is: it answers "is the agent up?", and he is not.
    # Working is a thing the box is doing, not a conversation - the eye says so, and the eye is
    # the one part of this panel whose job is to say what he is like rather than whether he is on.
    WORKING: GREEN_DIM,
    STARTING: AMBER,
    STOPPING: AMBER,  # disconnecting is the same transition, run backwards
    CONNECTING: AMBER,
    LISTENING: WHITE,
    SPEAKING: WHITE,
    LOOKING: WHITE,
    SEARCHING: WHITE,
    DRAWING: WHITE,
    ERROR: RED,
}
# There is no word for the state any more. There was one for a long time - ASLEEP, LISTENING,
# OPTICS - in a corner of its own, and what finally argued it off the panel is that three other
# things were already saying it better: the border's colour, which is readable across a workshop;
# the eye's mood, which is a face rather than a label; and the line under the picture, which says
# "searching the web…" where the word could only say SEARCH. A fourth voice saying the same thing
# in fewer letters is not redundancy, it is noise.
# The heat lamp, which is the one thing on the strip that is not about the session at all. It is
# the board's temperature, and it is here rather than only on the admin page because by the time
# it matters the panel is already misbehaving - a throttled Pi drops camera frames and misses
# audio deadlines - and the person watching that happen deserves to be told why rather than left
# to guess. Amber where the clock starts being capped, red where it is capped in earnest; the
# word does not change, because on this panel colour is what carries severity.
#
# Filled, like REC, because a tag with a word in it is how this tube shouts and an outline is how
# it murmurs. Unlike REC it is not red at the first step: red is the fault colour here, and a warm
# Pi is not yet a broken one.
HEAT_LAMP = {"hot": AMBER, "throttled": RED}
# ...and the gauge's own three, which are a band wider than the lamp's on purpose: the lamp is an
# interruption and only lights once the board is taking something away, and the gauge is a thing
# you went and looked at, so it may go amber where cyclops.stats.temp_band does.
HEAT_INK = {"ok": GREEN, "warn": AMBER, "hot": RED}
HEAT_WORD = "HOT"

# ... and the resting line underneath: what is true about a state when nothing finer is known.
# The controller sends a better sentence whenever it has one - what is being searched for, which
# project is being opened, which step of the teardown is running - and that wins; this is what the
# panel falls back on. Every state has one, so the line is never blank while a session is up.
CAPTIONS = {
    # Never actually read: this state exists only while cyclops.tasks has a sentence, and that
    # sentence is what the caption shows. Here because every state has a resting line and the one
    # that could get away without it is the one that would be blank on the day something changed.
    WORKING: "working…",
    IDLE: "zzZzzzZ…",  # he snores. The microphone opposite still breathes towards the colour it
    # will turn, which is the half of this that was ever load-bearing - so the line is free to
    # stop being an instruction and go back to being him.
    STARTING: "waking up…",
    STOPPING: "going to sleep…",
    CONNECTING: "waking up…",
    LISTENING: "listening — talk to me",
    SPEAKING: "speaking…",
    LOOKING: "looking…",
    SEARCHING: "searching the web…",
    DRAWING: "drawing…",
}

# ...and how the eye behaves in each - the fourth table, and the one that says what he is *like*
# rather than what he is doing. :mod:`cyclops.eye` is only the mechanism; every state here is
# fourteen numbers and a colour, so a new one costs a line and tuning one costs a keystroke.
# Render
# `tools/eye_sheet.py` after touching any of them: these are meant to be looked at, not reasoned
# about.
#
# The colours are not the border's. The border answers "is the agent up?" in three colours and
# has to be readable across a workshop; the eye is a face, and it may say something finer - so a
# state that looks the same on the rim can still look different in the middle of the bar.
MOODS = {
    # Asleep: turning slowly, and breathing with the iris. Two things move, and both of them
    # are movement rather than light - a resting creature is not a lamp on a dimmer. Brightness
    # was tried as the whole of it and is not animation at all: a face whose ticks sit in exactly
    # the same place in every frame reads as a still picture with something flickering behind it,
    # however deep the pulse is made.
    #
    # Slower than anything else in the table, which is what makes this sleep rather than work:
    # the steady rate is a sixtieth of a hunting eye's and a third of an attending one's, slower
    # even than the stare, which is the stillest he gets while awake.
    #
    # But the steady rate is not what you see. At that rate alone he was barely perceptible - a
    # gear train turning too slowly to notice, and one that only ever reaches arrangements it has
    # reached before. The sway is what makes it read: past 1 a ring's wander outruns its own rate,
    # so each of them speeds up, falls back and turns over on its own schedule, and the ticks,
    # brackets and dots swap places instead of holding formation. At 1.6 the tick ring runs
    # anywhere from five degrees a second backwards to ten forwards and spends about a quarter of
    # its time going the wrong way - never fast, never still, and never twice the same.
    #
    # The breath is the longest here, about nine a minute, and like every other period on this
    # panel it is not a multiple of any of the others.
    #
    # The iris carries that breath, and stays narrow doing it: it swells between a tenth and
    # under two fifths open, which never crosses the threshold the pupil is drawn on, so it grows
    # and shrinks rather than popping in and out - dozing, and nowhere near the 0.52 of a face
    # that is paying attention.
    #
    # The gaze is the set worth reading as a set rather than a row at a time. `look` is the list
    # of *places* he attends to and the first of them is his anchor - where he rests and what a
    # glance comes back to; `gaze` is how far he actually turns to reach one; `dart` is the share
    # of his attention that goes anywhere but the anchor, and `dart_s` how long one span of it
    # lasts; `drift` is a float on top of all of it. What separates a creature from a turret is
    # that the mix differs per state - an attending eye holds you and a hunting one never rests -
    # and it costs a list of names rather than a branch.
    #
    # That the places are *named* is the whole of it. They used to be angles off a walk, which is
    # a thing that looks around without ever looking *at* anything, and it had a reach floor - so
    # the one direction the old eye could never manage was straight at you.
    #
    # He floats in his sleep and never darts: a sleeping face that flicks about is a dreaming one,
    # and this panel is not claiming that. `drift` rather than a glance, because there is nothing
    # he is attending to - the float is the third thing that moves while he is asleep, after the
    # rings and the breath, and that scarcity is what makes awake read as awake.
    IDLE: Mood(tint=GREEN_MID, aperture=0.24, swell=0.14, breath_s=6.5, spin=2.5, sway=1.6,
               drift=0.30),
    # Coming round: the iris only half up, the rings running fast, and a highlight sweeping the
    # rim - a thing spinning itself up rather than a thing paying attention. It checks its own
    # instruments while it does it - the pod, the picture, then nothing in particular - which is
    # the difference between waking and booting, and is why the list is longer here than anywhere
    # except the hunt.
    STARTING: Mood(tint=AMBER, aperture=0.34, swell=0.10, breath_s=1.5, spin=54.0, scan=88.0,
                   look=(AHEAD, DIALS, FRAME, AWAY), gaze=0.70, dart=0.55, dart_s=1.1),
    CONNECTING: Mood(tint=AMBER, aperture=0.34, swell=0.10, breath_s=1.5, spin=54.0, scan=88.0,
                     look=(AHEAD, DIALS, FRAME, AWAY), gaze=0.70, dart=0.55, dart_s=1.1),
    # Winding down. The same transition run backwards, which is what the rings do - and he stops
    # attending to anything as he goes, because a thing finishing is not still looking for
    # something. What is left is a float settling out.
    STOPPING: Mood(tint=AMBER, aperture=0.10, swell=0.04, breath_s=3.0, spin=-22.0, drift=0.12),
    # Awake and attending, and the row the whole model was written for. A resting breath, a
    # barely-moving ring set, and an iris that holds where it is while you talk.
    #
    # It used to open to your voice, on the argument that an iris moving with the room is the
    # panel saying it can hear you. It says something else. A face that widens on your vowels and
    # settles in your pauses is a mouth, or a meter with a face painted on it - and it is the one
    # thing on here nobody has to be taught to read, so it was loud about the wrong idea. Nobody
    # listening to you does that; what they do is hold still and hold your eye, which is the rest
    # of this row. The level has a place on this panel already - the signal bar beside the clock,
    # where a number belongs and where moving with the room is the whole job. Leaving it there and
    # taking it off his face costs nothing: the bar is what proves he can hear you, and the eye
    # goes back to being what proves somebody is home.
    #
    # He looks at *you*: the anchor is AHEAD, which is the pupil dead centre and the eye looking
    # out of its own glass. Every few seconds he takes a peek at the picture he is sitting on and
    # comes straight back. That is the whole of it, and it is two names and three numbers.
    #
    # 0.26 over a 2.6 s window, measured rather than guessed: he holds you for 5, 8 or 13 seconds
    # - three lengths, because the walk's gaps come in three (see eye.gaze_at) - and the peek
    # itself lasts about a second. Nine in ten frames have him looking at you, which is roughly
    # what a person listening does and is nothing like the old row, which wandered continuously
    # and never came back anywhere.
    LISTENING: Mood(
        tint=WHITE, aperture=0.52, swell=0.07, breath_s=4.0, spin=7.0, blink_s=4.4,
        look=(AHEAD, FRAME), gaze=0.62, dart=0.26, dart_s=2.6, drift=0.05,
    ),
    # Talking: a faster breath and a wider iris, because he is doing the thing rather than waiting
    # to. The one row left that opens to level at all, and the reason the knob still exists: the
    # level here *is* his own voice coming back, so a face moving with it is a face moving with
    # what it is saying. That is the case the gesture was always right for. A tenth, because his
    # own words should show on him without him mouthing them.
    #
    # He holds your eye while he talks, and what he glances at when he does look away is his own
    # caption - the one place on the panel that is what he is saying. Fewer glances than listening
    # and a longer window between them, which is the opposite of a person (speakers avert more
    # than listeners do) and right for this one: he is a face on a panel, and a panel that looked
    # away while answering you would read as not answering.
    SPEAKING: Mood(
        tint=WHITE, aperture=0.70, swell=0.17, breath_s=1.1, voice=0.10, spin=13.0, blink_s=5.5,
        look=(AHEAD, WORDS), gaze=0.50, dart=0.18, dart_s=3.7, drift=0.04,
    ),
    # Looking at a photo, and now actually at it. Wide, still, and it does not blink: this is a
    # stare, and the thing it is aimed at is the middle of the picture rather than nowhere in
    # particular. One name in `look` and no `dart` is how "he never takes his eyes off it" is
    # said. The old row put him at 0.10 of the travel, which is a face examining a photograph by
    # staring at the person holding it.
    LOOKING: Mood(tint=WHITE, aperture=0.88, swell=0.02, breath_s=6.0, spin=3.0,
                  look=(FRAME,), gaze=0.85, drift=0.04),
    # Hunting. Narrowed to a point, breathing fast, rings tearing round with a scanning arc, and
    # the eye going place to place twice a second and hardly ever home. This is the one mood where
    # the gaze is the loudest thing about him, and it should be - he is looking *for* something
    # rather than *at* something, which is why the list is everything on the panel and why the
    # anchor is the picture rather than you.
    #
    # `dart` at 0.90 is what makes it a search rather than a metronome: consecutive glances chain
    # place to place instead of returning between them - see eye._hold - so what you get is runs
    # of four to twelve saccades broken by one glance home.
    SEARCHING: Mood(tint=WHITE, aperture=0.36, swell=0.06, breath_s=0.9, spin=155.0, scan=118.0,
                    look=(FRAME, WORDS, DIALS, AWAY, AHEAD), gaze=0.90, dart=0.90, dart_s=0.55),
    # Drawing. Deliberate, and turning the other way, because it is making rather than looking -
    # and head down at the work while it does. That used to be a lean it held and nothing else;
    # now it is a place, which costs the same and buys the glance up at you that anybody looks up
    # with every twenty seconds or so of doing something with their hands.
    DRAWING: Mood(tint=WHITE, aperture=0.46, swell=0.05, breath_s=2.2, spin=-34.0,
                  look=(WORK, AHEAD), gaze=0.60, dart=0.18, dart_s=4.5, drift=0.03),
    # Working, with nobody talking to him. The one mood in this table that is not a face at all,
    # and every number here is pushing away from the creature the rest of them describe.
    #
    # The pupil is shut, and `core` at 0 is what shuts it rather than a small `aperture`. Narrowing
    # the iris was tried first and leaves an ember: the blades stop at HOLE_MIN and Pen.core puts a
    # guaranteed spark in whatever hole is left, so the brightest mark on the face was still there,
    # still centred, still reading as something looking back. At 0 the blades meet and nothing is
    # drawn inside them. That is the whole of "he is not attending to you" - every other awake mood
    # opens to something, and this one has nothing to open to. `aperture` is left near zero anyway
    # so that the closing is a closing: he winds the iris down on his way into this mood rather
    # than snapping shut.
    #
    # `sway` is what stops it being a graphic. It was zero here at first, on the argument that a
    # locked gear train is what a machine is - and a set of rings turning at one rate that never
    # varies reads as a spinning logo, because every arrangement it reaches it has reached before.
    # A machine under changing load winds up, falls back and turns parts over, and that is the
    # half of "running" you cannot get from speed alone.
    #
    # `sway_s` is the half of that which says how *often*, and it is the reason this mood needed
    # a knob the table did not have. Amplitude and rate are separate in eye.wander - a term pushes
    # just as hard whatever its period - so `sway` alone cannot make a ring turn over more than
    # about twice a minute at sleep's sixteen-second surge. Raising it only makes the same slow
    # heave harder.
    #
    # 1.2 over 1.6 s, measured rather than guessed: each of the three visible rings changes
    # direction about once a second, and the knurl spends 22% of its time running backwards. What
    # was tried on either side of it - the surge lengths are what matters, not the amplitude.
    # At 2.5 s it turns over every two or three seconds, which is a machine changing its mind
    # rather than one under load. At 1.2 s and sway 1.6 the runs get shorter than the eye can
    # follow and it stops reading as turning at all - it judders. This keeps sustained runs in
    # both directions, which is what makes the reversals read as reversals.
    #
    # The scan arc rides the same wander at share 2.3 and is barely touched by it - 1% of its time
    # backwards - so the sweep goes on going one way underneath all of it, which is what a rotor
    # should do while the gear train around it argues.
    #
    # Briskly, and one way. 52 degrees a second is a turn every seven seconds - fast enough to be
    # plainly running from across a bench, slow enough not to read as the hunt SEARCHING does at
    # 155. The scan arc is the rotor: one bright sweep going round the rim, which is the panel's
    # existing way of saying a mechanism is turning rather than merely lit.
    #
    # And it does not look around, does not blink and does not breathe like a creature - `gaze`
    # and `blink_s` at zero, and the swell a small fast tick at 1.8 s rather than a lung. A face
    # that glanced about while it worked would be waiting for you; this one has its head down.
    # `rings` above his resting level, which is the one number here that is about being seen
    # rather than about being a machine. At 1.0 he was dimmer working than asleep - the sleeping
    # face is green, and green is the brightest thing this phosphor does, so the same presence in
    # a cold hue reads as less. 1.25 puts him back level with it; 1.5 was tried and reads as lit
    # rather than running, which is the opposite of the point.
    WORKING: Mood(tint=BLUE, aperture=0.04, swell=0.03, breath_s=1.8, spin=52.0, sway=1.2,
                  sway_s=1.6, scan=64.0, gaze=0.0, rings=1.25, core=0.0),
    # A fault. Still and red, and pointedly not pulsing - not even the sleeping breath: a thing
    # that throbs is asking to be watched, and this one is asking to be read. The caption
    # underneath says what broke, and it is the only face on the panel that never moves at all
    # - which now includes its gaze: `gaze` at 0 with no lean is a face staring dead ahead
    # forever, and that is exactly what a fault should do.
    ERROR: Mood(tint=RED, aperture=0.20, swell=0.0, breath_s=0.0, spin=0.0),
}

# A caption that ends in an ellipsis is a caption that moves, and that is the whole test the line
# uses: "searching the web…" gets a blinking cursor, "listening — talk to me" holds still. Every
# phrase
# the controller publishes obeys the same rule, which is why none of them has to say twice whether
# it is a job or a state. Nearly always that means work in flight; the exception is the snore,
# which is not work but is just as much a thing going on.
BUSY_MARK = "…"
MARKER = "› "  # what every caption opens with, and the smallest thing that wears the accent
CAPTION_LINES = 2  # how far a sentence may wrap before it is cut short instead, and now also
# how deep the terminal's screen is - the glass is cut to its text rather than the other way
# round. One line meant every phrase worth reading - a fault, a search, what a tool is doing -
# was trimmed to a stub ending in an ellipsis with most of the panel's width still free beside
# it. Two is where it stops: a third is a screen deep enough to start eating the picture, and a
# caption that big is a dialogue box rather than something said in passing.
CAPTION_ALPHA = 245
BLOOM_R = 2.6  # reference px of skirt round a lit glyph. 1.3 ms a frame on the Pi...
BLOOM_ALPHA = 0.62  # ...and how much of the letter's own alpha goes into it. Both are held well
# under what a photograph shows: a real tube blooms far harder than this, and at 14 px a line that
# blooms like a photograph is a line nobody can read. It is a depth cue here, not an effect.
CURSOR = "_"  # what a line about work in flight ends in: a cursor, blinking, hard against the
# last letter. It was three dots walking up and starting over for a long time, which is the same
# sentence in a language this panel does not speak any more - the line is printed on a terminal
# now, and a terminal that has not finished says so with a cursor. It costs a character where the
# dots cost three, and it means the thing everybody has read it to mean since a VT100.
CURSOR_PERIOD_S = 1.2  # one blink of it, half on and half off...
BREATH_PERIOD_S = 2.4  # ...and one breath of the phosphor, at half that rate so the two never lock
BREATH_DEPTH = 0.30  # how far the text sinks towards the glass at the bottom of a breath
TYPE_CHAR_S = 0.045  # the terminal's own rate, near enough a character every other frame at
# 30 fps. It was 0.028 and read as a wipe: fast enough to be over before you had looked down at
# it, which is a line that arrived whole with extra steps.
TYPE_JITTER = 0.45  # ...and how far ahead of or behind it any one character may land
TYPE_GAP = 2.0  # extra intervals the hand rests for after a word or the end of a clause
TYPE_REST = " ,.;:—-…\n"  # what it rests after
TYPE_MAX_S = 0.9  # the longest a whole line may take to arrive, however long it is. This used
# to be held under agent.ACTIVITY_HOLD_S - 0.6, the floor under a tool that is off the card and
# back in five milliseconds - so that no sentence could still be arriving when it was replaced.
# That bought a guarantee about the rarest case on the panel, two instant tools back to back, at
# the price of the only case anybody ever looks at, and it made the hand too quick to read. So a
# burst of instant tools can now clip a sentence. If that ever strobes, the number to change is
# ACTIVITY_HOLD_S, which is what decides how long a line is worth reading, not this one.

HALO_CORE = 0.004  # fraction of the height held at full brightness, hard against the edge
HALO_FALLOFF = 0.024  # and how far the light reaches inwards before it is gone
HALO_PEAK = 0.45  # alpha at the border, falling away to nothing before it reaches any text
TINT_ALPHA = 0.05  # green wash under the chrome - phosphor cast, not a colour filter
SCANLINE_EVERY = 3  # every third row of the chrome is darkened...
SCANLINE_ALPHA = 0.17  # ...by this much, which is a CRT at arm's length and not a zebra
VIGNETTE_FROM = 0.46  # where the corner shading starts, as a fraction of the half-diagonal
VIGNETTE_ALPHA = 0.42
# The strip and the tab row this layout replaced used to be opaque near-black, which bought
# contrast at the price of a third of the panel - 29% of a feed you are holding down a pipe to
# see what is at the bottom of. The brackets carry the filter and nothing else, so the picture
# runs edge to edge behind them and the chrome earns its contrast from its own opaque glyphs
# rather than from a bar.
TERM_ALPHA = 205  # the well the caption is printed in, and the one dark backing left. It used
# to be a slab under a speech bubble at 170, thin because that one floated on the picture and
# every point of alpha was a point of the room taken away from somebody holding a camera down a
# pipe. This one is a hole in a housing, and a hole you can see the wall through is not a screen -
# so it sits with the switch wells and the disc behind his face rather than with the slab.
TERM_SCAN = 34  # ...and how much more opaque every third row of it is. The tube's own scanline
# field comes through the glass at about a fifth of its strength once the well is over it, which
# turns a ten-point modulation into a two-point one: the one surface on this panel that is
# literally a CRT ends up the flattest thing on it. So the well draws its own, on the filter's
# pitch and in the filter's phase, and the two reinforce instead of beating.
SWITCH_ALPHA = 215  # ...and the well a switch is sunk into, which is dark enough to read a
# glyph off and no darker: it sits over the picture like everything else in a bracket does

# Layout. Fractions of the height where a thing is a proportion of the panel, reference pixels
# at 800x480 scaled by `scale` where a thing is a shape - the official 7" panel is the reference
# either way. The controls are in the corners now, in four brackets, and the middle of the screen
# is picture.
#
# The border runs along the very edge of the panel: there is no margin outside it, because a
# glow sitting outside a border reads as light leaking off the device rather than as a screen
# lit from within. The state light therefore falls *inwards* from the border instead.
FRAME_RADIUS = 0.034
LINE = 0.0042  # stroke of the border and the hairlines
PAD = 0.036  # inner padding - wide enough that the inward glow never reaches any text

# ---- the four brackets ----
#
# Each corner carries a bracket, and a bracket is a *spine*: the rail comes in square to one
# panel edge, ramps across the corner at 45 degrees, and lands square on the other. The square
# landings are the whole reason the shape is four points rather than two - a naked diagonal runs
# off the frame at an angle and eats the edges either side of the corner, and these give that
# room back to the picture.
#
# The rail on that spine is flat bar stock rather than a stroke: a mitred bar with a flat face,
# a chamfer down each long edge and square ends, standing a little proud of the plate and lit by
# the panel's one lamp (:mod:`cyclops.material`) so the chamfer turned towards it is one bright
# line and the one turned away gives back a duller glint off the plate below, with a hard line of
# contact shadow under it and a soft one falling the other way from the light. That is what makes
# a bracket read as something bolted to the panel rather than as a line drawn on it, and it is
# why the rail is this thick: at a hairline there is no width for a chamfer. It was a round-edged
# section for a while and read as a tube - a bright edge fading to dark is a pipe whatever it is
# made of, and flat stock is what a bracket is cut from.
RAIL = 17.0  # reference pixels, and the one number the whole bracket language rests on
RAIL_EDGE = 1.2  # reference px of the chamfer the lamp lights: one crisp line, no wider. The
# whole of what says "edge" at arm's length is that it is a line and not a band
RAIL_CHAMFER = 3.0  # reference px of the chamfer turned away from it, which is read as a ramp
# rather than a line and can afford the width; capped for a thin member so a clamp's nine-pixel
# strap keeps a flat between its two chamfers
RAIL_CHAMFER_TILT = 0.75  # sin of the lit chamfer's slope: a machinist's 45
RAIL_CROWN = 0.02  # sin of the tilt the flat face has reached by the chamfer. Nearly nothing:
# any more and the face is a gradient again, and a gradient across a bar is a tube
RAIL_RETURN = 0.85  # how far the far chamfer is put towards STEEL_LIT at its arris - the lit
# plate under the bar reflected in it. Duller than the lit edge on purpose: brighter and the bar
# is lit from two sides, which is the one thing the panel's single lamp must never look like
RAIL_LIFT = 2.0  # how proud the bar stands of what it is bolted to, which sets its cast shadow
RAIL_CONTACT = 0.72  # alpha of the hard contact line where the bar meets the plate...
RAIL_CONTACT_W = 1.6  # ...how far out from the edge it reaches before it is gone...
RAIL_CONTACT_LIT = 0.55  # ...and how much of it survives on the edge the lamp lights
RAIL_WEAR = 0.45  # how much the highlight comes and goes along a length - handled steel is
# polished where hands have been and dull between
RAIL_GRAIN = 0.8  # of the material's brushing. Under one and the bar is smoother than the
# sheet it is on; under half and at arm's length it is plastic
RAIL_SCRATCHES = 8  # hairlines drawn over each bar's box, of which two or three cross the bar
RAIL_SCRATCH_SPREAD = 4.0  # degrees either side of the bar's own run they wander - dragged
# along it, not across it
RAIL_SCRATCH_LEN = (12.0, 60.0)  # reference px, shortest to longest: long enough to read as a
# scratch and never the whole bar
RAIL_SCRATCH = 0.35  # how pale a hairline shows where it crosses a bar - its own and the
# sheet's, because a scratch that stops at the rail is a scratch on a drawing
RAIL_END_WEAR = 0.13  # how much darker the last few pixels of a bar are, where a cut end rusts
RAIL_END_W = 3.0  # ...and how many pixels that is
RAIL_LIP = 0.78  # fraction of a ring's width that is the lit chamfer, for the collar and the
RAIL_BODY = 0.28  # dial bezels, which still bend the old three-band profile - see _rail_colour
RAIL_SHADOW = 150  # alpha of the soft shadow the rail casts onto its own plate
BAR_SS = 4  # a bar's outline is filled this many times over and boxed down - its anti-aliasing
BOLT_R = 7.0  # a socket head, sunk through the rail wherever it turns
RIB_N = 3  # stiffeners across the deep corner of a bracket
RIB_W = 4.0  # ...their section, in reference px
RIB_ALPHA = 0.36  # how much steel a rib lays over the plate. Translucent like the plate it
# stiffens, so the room keeps running behind it.
PLATE_WASH = 0.34  # how far a bracket's plate is put towards SCREEN. Not opaque: a bracket you
# cannot see the room through is a bar, and this layout exists to stop having those.
PLATE_STEEL = 0.04  # ...and the grey stirred into that wash. A plate is gunmetal seen through
# the tube's glass rather than the glass alone, and a wash with no grey in it is a tint.
PLATE_GRAIN = 0.07  # how much the brushing shows through the wash, either way
PLATE_SCRATCHES = 70  # hairlines across the whole sheet, of which the plates keep about a sixth
PLATE_SCRATCH_ALPHA = 0.30  # ...and how pale the palest of them is
PLATE_LIGHT = 0.11  # how far towards STEEL_LIT the sheet is lifted where the lamp is nearest it.
# A wash is the same everywhere; a surface is lit, and falls off away from the light
PLATE_LAMP = (0.12, -0.30)  # where over the panel the lamp stands, in widths across and heights
# down - the same up-and-left the terminal's glare and every bar's bright edge agree on
PLATE_REACH = 1.7  # how far its light carries across the sheet, in panel heights. Long: the
# corners are still lit, only less
PLATE_WEAR = 0.035  # how much the lift comes and goes across the sheet - a plate handled for
# years is brighter where it is rubbed and duller between, and never the same shade twice

# The spines, at the reference. Anything on the right or the bottom is written as a distance in
# from that edge, so one table lays out all four and mirrors without a second.
#
# The two bottom brackets are deliberately unequal. The left one is a housing - it has a face in
# it, and a face wants room. The right one is a bracket with two switches bolted through it, and
# every pixel it does not take is a pixel of the room somebody is holding a camera down a pipe
# to look at.
# The status pod: a narrow module hanging off the middle of the top edge, and the one piece of
# chrome here that is not a corner. It went corner, then half a corner, then this - and the
# middle is where it belonged all along, because none of what it carries is *about* a corner.
# What it says is the box's own vital signs: how loud the room is, how long this has been going
# on, and whether anything is being recorded or running hot.
#
# The shape is the bracket language turned inwards: the rail drops square off the top edge,
# splays out at 45 to the flat that carries the readouts, and goes back up the same way. So it
# is a module rather than a mount - it has no corner to be bolted into - but it is plainly the
# same metal, and both its ends land square on the frame like everything else does.
# One row, not two, and no label on the meter: eight lit segments beside a running clock are
# not going to be mistaken for anything else, and the word cost a row of depth to say so. The
# pod is wider for it and much slimmer, which is the right trade for something hanging over the
# middle of the picture - depth is what it takes away from the room, width is not.
POD_H = 50.0  # its depth, which is one row of readout and the rail under it
POD_STEP = 16.0  # the square drop off the top edge before the splay starts
POD_PAD = 16.0  # inside the flat, either side of the readouts
POD_STOP = 12.0  # between the meter, the tags and the clock...
POD_TAG_GAP = 8.0  # ...and the tighter one between two tags, which read as one group
BOT_L = 250.0  # the bottom-left bracket's reach along both edges
BOT_L_STEP = 38.0  # its square landings
BOT_R_OUT = 170.0  # the bottom-right bracket's reach in from the right edge...
BOT_R_STEP = 38.0  # ...its landing on the bottom edge...
BOT_R_LAND = 26.0  # ...and the shorter one on the right, which is what makes it the small one

# ---- the terminal ----
#
# The line that says what he is doing was a speech bubble for a long time - a slab with a tail
# leaning down towards his face, sized to its sentence, floating over the middle of the picture.
# It is a machine now: a monitor standing in the one strip of this panel that had nothing in it,
# the bottom middle. Nothing about what the line *says* changed. What changed is that it is
# printed on something - and that the something is still there when there is nothing to say,
# which is the whole difference between the two. A bubble with no words in it is a bug; a
# terminal with a blank screen is a terminal.
#
# It is not a fourth bracket. A bracket is a spine with a corner to brace or an edge to land
# square on, and this has neither. It had a rail across its top for a while and both its ends
# buried in a mount, which is what a thing bolted *between* two other things looks like - and it
# cost the shape its two ends and its top edge, which is most of what there is to see of a
# monitor. So: a case standing clear in the middle of the bay, held at either end by a clamp -
# a strap standing on the case's edge and an arm back to the mount. The joint is a thing you can
# look at now instead of a thing hidden behind the bracket that makes it.
#
# The clamps are built out of :meth:`Overlay._draw_rail`, which is the whole reason they read as
# part of this machine: the same extrusion the mounts are made of, drawn by the same method, in a
# thinner section. They were drawn as filled rounded rectangles for an afternoon and no amount of
# shading rescued that - a bracket made of anything but the frame's own metal is a shape sitting
# next to a frame. Nor do they stand on a plate: a plate is what a *mount* is built out of, and
# giving one to a stay across open picture put a stripe of washed, scanlined chrome behind each
# clamp. A stay is a bar in the air with a shadow under it.
# What is in the housing is a monitor, and a monitor has a bezel with the tube's four corners
# inside it. That is nearly the whole of what makes this read as a screen rather than as a hole:
# a rectangle with square corners is a cut-out however it is shaded, and the same rectangle with
# its corners pulled in is a piece of glass sitting in a moulding. So the glass no longer runs to
# the edges of the chassis - it is a rounded face with metal all the way round it.
#
# That reverses two things this file argued for at length, and both were arguments about a
# *see-through* strip rather than about a bezel. There was no foot under the glass because a band
# of translucent chassis along the bottom "reads as a gap the module has not been pushed all the
# way into rather than as a bezel", and no bezel above it for the same reason. Both are right, and
# neither applies to opaque moulding with a lit lower lip: that is not a gap with the room showing
# through it, it is the thing the old strip was failing to be. The rail lands on the bezel now
# instead of on the glass, which is what a rail bolted across a monitor's housing does, and its
# cast shadow still falls - onto moulding rather than onto the picture.
TERM_FOOT = 6.0  # how far the case stands off the panel's own bottom edge. It sat on it while
# it was a slab bolted between the two mounts, where the edge was one of the things holding it;
# a monitor on its own brackets is a thing with air all the way round, and the border's glow
# running under it is what says so.
TERM_CLEAR = 9.0  # how far the monitor's case stands clear of each mount's rail, so that
# all four of its corners are its own. It used to run from the middle of one mount's bottom rail
# to the middle of the other's, buried at both ends for the lower half of its depth - which read
# as bolted in, and which cost the two ends of the shape. A monitor is a thing you can see the
# whole of; buried ends make it a slot again however round its corners are.
EAR_BOLT = 1.6  # how far a bracket's bolt head reaches past the member it goes through, in
# reference px. The head is sized off the strap rather than off the mounts' bolts, because what
# it has to do here is cover a square butt cap - see Overlay.ear_bolt.
EAR_RAIL = 9.0  # the section of a mounting bracket's members, against RAIL's seventeen for the
# mounts. Thinner because it is secondary metal - a stay carrying a screen, not a spine carrying
# the panel - and the same extrusion otherwise, drawn by the same method, so it is visibly the
# same alloy machined to a smaller size rather than a different thing that happens to be green.
TERM_EAR_H = 0.5  # how tall each mounting bracket is, as a fraction of the case's own depth - so
# it stays half of it at any window size rather than being a pixel count that drifts. Half is
# what makes it read as *mounting*: a thin tab reads as a wire or a seam, and the monitor above
# it looks like it is floating in the middle of the bay with two scratches beside it. Something
# with real depth is a bracket, and a bracket is the thing that says the screen is held. Not the
# full depth either - a bracket as deep as what it carries is a shelf, and the sentence is that
# this is held at two points rather than sitting on something. Its reach is
# not a number here: it is whatever the gap turns out to be. A rack ear: a tab off the side of
# the chassis reaching out
# to land on the mount's rail, with a bolt through where it lands. It is what carries the sentence
# the buried ends used to carry - that this is mounted rather than drawn on - and it says it in
# something you can see the whole of rather than by hiding the ends of the thing it holds.
TERM_BEZEL = 7.0  # the moulding round the tube - and it is a *transition* rather than a band.
# Nothing here draws an edge between the case and the glass. The front is one field: opaque and
# near-black hard against the outside, easing over this many pixels into glass you can see the
# room through. That is what a monitor actually looks like from a pace away - one smooth glassy
# face with the picture fading out into its surround - and it is what a bezel drawn as a ring of
# lit metal cannot look like however the profile is stepped. There was a bright lip round the
# aperture here for an afternoon and it read as a glowing pill, which is the opposite of glass.
TERM_PAD = 3.0  # inside the glass, above the first line and below the last. It was 6.0 while the
# glass ran to the chassis edges and the text had nothing but its own padding holding it off
# them; the moulding is that separation now.
TERM_RADIUS = 22.0  # the corner of the front. One radius and not two: the case and the glass are
# the same shape at different depths into the same field, so there is one fillet to turn and no
# pair of concentric ones to keep from drifting apart.
TUBE_GLOW = 15.0  # how far in from the moulding the phosphor takes to come up...
TUBE_GLOW_A = 0.085  # ...and how much of it there is at full. Blank at the edge and a little
# green towards the middle is the way round a tube does it; bright at the rim and dark in the
# centre is a hole with a lamp behind it.
TUBE_SHADOW = 0.24  # how far down the glass the case's own shadow falls, as a fraction of its
# depth, and...
TUBE_SHADOW_A = 22  # ...how much opacity it puts there at the top.
# The room, and where it is coming from. Everything below is one lamp: a white source up and to
# the left of the panel, which is where a bench light is and where anybody reads a highlight from
# without having to be told. Two terms make it - how much of its light reaches a point at all,
# which falls away with distance from the source, and the streak it draws down the face, which is
# what a long glossy surface does with a small bright thing.
GLARE_X = 0.02  # the source, in face widths across...
GLARE_Y = -0.16  # ...and in face heights down, so it sits just off the top-left corner
GLARE_REACH = 2.1  # how far its light carries, in face heights
GLARE_ALPHA = 0.30  # and how bright it is where it lands hardest
GLARE_AT = 0.18  # where down the left-hand edge the streak passes...
GLARE_DEPTH = 0.36  # ...and how broad it is either side of that
GLARE_TILT = 0.62  # how far down the face the streak's middle travels on its way across. A screen
# this wide cannot have a forty-five degree sheen - it would cross the whole depth inside fifty
# pixels and read as a scratch on the glass. What a letterbox catches is a shallow wipe.
GLARE_ON_CASE = 2.4  # how much more of it the moulding returns than the glass does. Gloss black
# gives back nearly all of what falls on it; a phosphor face is already lit, so the same
# reflection is a far smaller part of what it is doing. This is what keeps the brightest part of
# the highlight on the surround rather than across the first word of the sentence.
SHEEN_D = 3.5  # how far in from the case's edge its own rim light reaches...
SHEEN_A = 0.50  # ...and how strong it is where the moulding faces the lamp squarely...
SHEEN_AMBIENT = 0.22  # ...against how much it still catches where it faces away. Not zero: a room
# bounces light back into the far corner of anything in it, and an edge with none at all reads as
# a hole cut in the panel rather than as the dark side of an object.
GLARE_AMBIENT = 0.20  # the same floor under the wipe across the face, for the same reason. The
# corner furthest from the lamp was going dead black without it, which no glass does.
# The eye. He rides the left bracket's ramp, sunk halfway into it - `EYE_SEAT` is that depth as a
# fraction of the swell's radius, and acos(0.5) is a 60-degree shoulder, which is where the rail
# leaves the straight and goes round him. Half of him is in the bracket and half is over the
# picture, which is the same join the tab row used to make and the reason he reads as part of the
# machine rather than as a badge stuck on it.
EYE_R = 0.1833  # 88 px at 800x480, against 60 in the row this replaced
EYE_SHOULDER = 16.0  # reference px between his rim and the rail's centreline round him
EYE_SEAT = 0.5
EYE_PLATE_ALPHA = 255  # the body behind him, and the one thing on this panel that is not a hole
# in a housing. It sat at the terminal's own 205 for a while on the porthole argument - that a
# window you cannot see through is not a window - and the argument was about the wrong object.
# What is drawn inside this circle is a mechanism, not a view: an iris, a stator, a knurled ring,
# and between every one of them the tile he is painted from is empty. At 205 the room came
# through all of it, and the one part of the panel that is supposed to be a machine was the only
# part you could see the wall through. He is solid now, and the reticle in the middle of the
# picture is where the seeing-through belongs.

# The body runs out to the swell rather than to his rim. Between those two there used to be
# sixteen pixels of translucent plate with the rail's inner edge floating over it, and that strip
# is where the collar goes: the bezel he is seated in, a ring of worn brass standing proud of the
# plate with a polished steel lip on its inner edge holding the glass down, under the same lamp as
# every bar on the panel (:mod:`cyclops.material`). Brass and not steel, and only here and on the
# two dials: an instrument's bezel is the one part of a machine like this that was ever a
# different metal, and it is what says "instrument" about a disc of rings before any of them turn.
COLLAR_IN = 0.86  # the collar's inner flank, as a fraction of the swell
COLLAR_LIP = 2.5  # reference px of the steel lip on the bezel's inner edge, a half-round wire
COLLAR_ROLL = 3.0  # reference px of the brass's outer edge that roll down to the plate
COLLAR_STEP = 1.5  # ...and of its inner edge that turn down onto the lip
COLLAR_GROOVE = 0.45  # how dark the turned line just inside the roll is - the one mark a lathe
# leaves on every bezel, and what separates the face from the roll at a glance
COLLAR_LIFT = 3.0  # how proud the bezel stands of the plate, which is what sets its shadow
COLLAR_SHADOW = 0.62  # ...and how dark that shadow is where it is deepest
COLLAR_WEAR = 0.6  # how much the brass's highlight comes and goes round the ring: handled where
# a thumb lands on it, dull between
COLLAR_TARNISH = 0.6  # how much the same slow drift shows in the brass itself, as multiples of
# the brushing - old brass is not one colour, it is polished in patches
COLLAR_SCRATCH = 0.25  # how pale the sheet's hairlines show where they cross the brass
COLLAR_CROWN = 0.40  # sin of the tilt the brass face has reached where its roll begins. Far
# more than a bar's DOME, because a bezel is not a bar: it is turned with a rounded section, and
# the fall from its lit side to its dark side across a dozen pixels is most of what says so.
BRASS = (96, 86, 58)  # a flat face of it square to the viewer. Well short of the metal in a
# catalogue: this is brass seen by a phosphor tube, and it borrows what little colour it has.
# Never orange, and never brighter than STEEL_SPEC in any channel.
BRASS_SPEC = (196, 176, 128)  # where the lamp lands hardest on it. Warm where steel's is cool,
# because a highlight carries the metal's own colour, and no brighter than the steel's ceiling.
WELL_FLOOR = (24, 32, 27)  # the machined floor he is drawn on, at its middle...
WELL_WALL = (9, 15, 12)  # ...and where it meets the wall. Dark enough that no test counting
# phosphor ever sees it, light enough that a shadow falling on it has something to fall on -
# SCREEN is so near black that a shadow on SCREEN is nothing at all.
WELL_DEPTH = 5.0  # reference px the floor sits below the bezel's lip, which is how far the near
# wall's shadow reaches across it
WELL_SHADOW = 0.75  # how dark that shadow is under the wall
WELL_GRAIN = 0.5  # how much the floor's turning marks show, as a fraction of material.GRAIN
# The glass over him. A dome held down by the lip, and the one part of him that is not drawn
# every frame: the lamp's reflection on it and the light it gathers along its edge are a tile
# built once and laid over the eye after it is painted. It has to stay off everything that moves.
# The iris and everything inside it travel with his gaze, as far as IRIS + GAZE_SHIFT plus the
# width of a stroke, and a highlight over a moving spark makes the spark's brightness a function
# of where it is - which is exactly the flicker the sleeping face is not allowed. And it has to
# stay off the rim, which the scan sweep is measured on. What is left is the band between, the
# stator and the castellated ring, and that is where a dome's glare falls anyway.
GLASS_IN = 0.735  # of his radius: where the glass may start, past the optic's furthest reach
GLASS_OUT = 0.955  # ...and where it must have ended, short of the rim
GLASS_EASE = 0.05  # how far past each of those it fades in and out
GLASS_AT = 0.86  # how far up the dome towards the lamp its reflection sits, as a fraction of him
GLASS_REACH = 0.55  # ...and how far that reflection spreads, in the same units
GLASS_GLARE = 0.24  # its alpha where it is brightest, in the tube's own white
GLASS_RIM = 0.14  # the light the dome gathers along its edge on the side facing the lamp
GLASS_SHADE = 0.14  # ...and how much the far side of it darkens what is under it
# The loom: three runs of flexible steel conduit leaving the back of his housing through a gland,
# the same lamp on them as on everything else. Conduit and not cable, because everything on this
# panel is metal or glass; darker than the bars, because it is braid and not a machined face.
LOOM_STEEL = 0.45  # how far the conduit is put from STEEL towards STEEL_DARK
LOOM_LIFT = 2.0  # how proud a run stands of the plate, which sets its shadow
LOOM_RIB = 1.6  # how much the braid's ribbing shows, as multiples of a bar's brushing
GLAND_ROLL = 2.0  # reference px of the gland's edges that roll
# Where the loom leaves him, in PIL's degrees - straight at the panel's own corner, because that
# is the only direction with any run in it. His swell comes within three pixels of both the left
# edge and the bottom one, so the pocket between him and the corner is the whole cable budget:
# about 47 px at 800x480, which is enough for a gland and three cables and nothing else.
GLAND_AT = 135.0
LOOM_N = 3
LOOM_FAN = 11.0  # degrees between one cable and the next
LOOM_REACH = 70.0  # reference px, which is past the corner: they are meant to leave the panel
# The mount points, in the arc the bracket's rail does not already cover - it comes round him
# from -105 to +15, and these two are what say the rest of the collar is bolted down as well.
COLLAR_BOLTS = (72.0, 198.0)

# The two instruments, bolted straight through the small bracket's rail rather than sitting in a
# plate of their own. Neither has a word: a dial that needs a label is the wrong dial, and the
# words were what forced the controls into a row across the bottom in the first place.
#
# They used to be a shutter and a microphone. The button beside the panel now does both - a tap
# is the photo, a hold is the session - so two of the best-placed targets on the glass were a
# second copy of a control your hand can already find without looking. These are the two things
# the panel could not otherwise say or be told: how loud he is, and how hot the board is.
#
# The knob nearer the middle of the panel and the gauge nearer the corner, because the one you
# reach for should be the one a thumb gets to first and the one you only read can sit further out.
BTN_R = 36.0
BTN_SPACING = 52.0  # along the ramp, either side of its middle
VOLUME, HEAT = "volume", "heat"
SWITCHES = (VOLUME, HEAT)

# Both dials sweep 270 degrees with the gap at the bottom, which is where the gap is on every knob
# anybody has ever turned and every gauge anybody has ever read. PIL measures clockwise from three
# o'clock, so that is 135 (down-left, empty) through the top and round to 405 (down-right, full),
# and the bottom quarter is left free for the reading to sit in.
DIAL_FROM = 135.0
DIAL_SWEEP = 270.0
DIAL_TICKS = 7  # graduations across the sweep, both ends included
# The scale runs well inside the bezel, with the graduations between the two: a green arc drawn up
# against a green chamfer is a green chamfer, and that is what the first cut of this looked like.
DIAL_TRACK = 0.64
DIAL_TICK_IN = 0.75
DIAL_TICK_OUT = 0.87
DIAL_HAND = 0.50  # the pointer's tip, kept short of the reading printed in the gap below it
DIAL_HUB = 0.15
VOLUME_STEP = 5  # what one setting of the knob is worth, matching the page's own slider

# What a drag on the knob opens. A dial is the right shape to *read* a level off and the wrong one
# to set with a finger: the hand covering it covers the pointer, the number and the scale at once,
# which is the whole of what you came to look at. So the drag leaves the disc behind and becomes a
# column standing well clear of it - the value is where your finger is, and nothing is asked of
# the speaker until you let go, so a grab that landed somewhere you did not mean costs nothing.
SLIDER_W = 34.0  # the track, in reference pixels
SLIDER_TOP = 54.0  # its top edge, down from the panel's own
SLIDER_GAP = 12.0  # ...and between its foot and the knob the finger came off
SLIDER_RUNGS = 100 // VOLUME_STEP  # one rung per setting, so the step you get is the step you see
SLIDER_RUNG_GAP = 0.30  # of a rung, left dark between one and the next
SLIDER_THUMB = 5.0  # the bar across the top of what is lit: the indicator proper
SLIDER_EAR = 7.0  # ...and how far it stands out either side of the track, so it reads as a grip
DIAL_OFF = 0.95  # how far an unlit stretch of scale is stirred out of the screen's own black
DIAL_LABEL = 0.66  # where the reading sits in the gap under the hub, clear of the pointer's
# reach: a needle at either end of the sweep points down into that quarter, and a number it
# grazes on the way past is a number you read twice to be sure of

# The reticle: four corners on the lens axis and nothing else. It was a cross with graduations
# for about an hour, which is exactly as long as it took somebody to say it looked like a gun
# sight - and they were right. Corners say "the frame is here" and say nothing else.
RETICLE_R = 0.112  # 54 px at 480, from the middle out to any edge of the box
RETICLE_LEG = 0.5  # of that reach, per leg; the rest of the edge is the gap on the axis
RETICLE_W = 1.4  # of a hairline: marks this short need the weight back to read as drawn

RIM_PERIOD_S = 3.7  # one breath of the border, slower than the caption's and not a multiple of it
RIM_DEPTH = 0.14  # how far it sinks towards SCREEN - a mix, not an alpha, and a quarter of what
# the caption may do, because this is the one thing readable across a workshop

METER_SEGMENTS = 8  # steps in the signal bar
# How much colour is stirred into the chrome for the parts that are meant to look faded. These
# are mixes rather than alphas on purpose - see _mix: drawing them translucently would not dim
# them, it would open a window onto whatever the camera is pointed at.
METER_OFF = 0.45  # an unlit signal segment
STEEL = 0.58  # how far the rail's body is stirred towards SCREEN out of GREEN_MID

# ---- the power menu ----
#
# What a long press on his face opens: the two things a box with no keyboard and one button
# cannot otherwise be asked for. It is the shape of every power menu anybody has ever held a
# phone's side button down for - press and hold, choose, or tap away from it to think again -
# because nobody should have to be taught this one, and because the gesture is deliberate enough
# that it cannot be arrived at by a thumb landing on the wrong third of the panel.
#
# Why it is a menu drawn here rather than a screen on the admin page: the page is a browser that
# has to be uncovered, and the one moment you most want to shut a box down is the moment it is
# behaving badly enough that you would rather not ask Chromium for anything first.
POWER_OFF, RESTART, CANCEL = "poweroff", "restart", "cancel"
MENU_ROWS = ((POWER_OFF, "SHUT DOWN"), (RESTART, "RESTART"), (CANCEL, "CANCEL"))
MENU_TITLE = "POWER"
# The one line that keeps this apart from the tab a finger's width underneath it. GO TO SLEEP
# ends the *session*; these two end the *box*, and in English those are near enough the same
# sentence that the menu says which it means rather than trusting the words to.
MENU_NOTE = "the whole box, not the session"

MENU_W = 0.80  # of the panel height, like every other fraction here
MENU_ROW_H = 0.13  # 62 px at 480 - a target for a thumb that is not being looked at
MENU_HEAD_H = 0.09  # the title line above the rows
MENU_PAD = 0.021
MENU_SCRIM = 168  # how far the panel behind the card is put out. Not all the way: the picture
# and the border keep saying what the box is doing underneath a question about turning it off.
MENU_GLYPH_R = 0.030  # the power and restart marks, which are the half of a row that is read
# from further away than its word


def unit(ax: float, ay: float, bx: float, by: float) -> Point:
    """The unit vector from a to b, and (0, 0) never happens because no spine has a zero leg."""
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy) or 1.0
    return dx / length, dy / length


Point = tuple[float, float]


def offset_path(points: Sequence[Point], distance: float) -> list[Point]:
    """Miter-offset a polyline. Positive is the ``(dy, -dx)`` side of it.

    Every spine on this panel is wound so that side points away from the corner its bracket is
    bolted into, which is what lets one sign mean "outwards" for all four of them. The miter is
    clamped rather than limited: at the 90-degree turns a spine actually has, 1/cos(45) is 1.41
    and nothing here ever reaches the clamp, but a window size that rounded a knee into a spike
    would put a spike on the panel and not raise.
    """
    out: list[tuple[float, float]] = []
    for i, (px, py) in enumerate(points):
        normals = []
        if i:
            dx, dy = unit(*points[i - 1], px, py)
            normals.append((dy, -dx))
        if i < len(points) - 1:
            dx, dy = unit(px, py, *points[i + 1])
            normals.append((dy, -dx))
        nx, ny = sum(n[0] for n in normals), sum(n[1] for n in normals)
        length = math.hypot(nx, ny)
        if length < 1e-6:  # a doubled-back segment; there are none, but a spike is worse
            nx, ny, miter = normals[0][0], normals[0][1], 1.0
        else:
            nx, ny = nx / length, ny / length
            miter = 1.0 / max(0.45, nx * normals[0][0] + ny * normals[0][1])
        out.append((px + nx * distance * miter, py + ny * distance * miter))
    return out


def arc_points(
    cx: float, cy: float, r: float, a0: float, a1: float, step: float = 4.0
) -> list[Point]:
    """An arc as a polyline, so a swell round a control can be offset like any other segment."""
    n = max(2, int(abs(a1 - a0) / step))
    return [
        (cx + r * math.cos(math.radians(a0 + (a1 - a0) * i / n)),
         cy + r * math.sin(math.radians(a0 + (a1 - a0) * i / n)))
        for i in range(n + 1)
    ]


@dataclass(frozen=True)
class Seat:
    """A control riding a bracket's ramp: where it sits, how big it is, and its swell.

    ``depth`` is how far its centre is sunk below the rail's centreline. Zero puts the rail
    straight through it - which is what the two switches want, because a rail that parted round
    a 36 px disc twice in 170 px would be more swell than rail. The eye is sunk half its swell
    instead, so the rail leaves the straight at 60 degrees, goes round him and comes back.
    """

    centre: tuple[float, float]
    radius: float
    shoulder: float

    @property
    def swell(self) -> float:
        return self.radius + self.shoulder


class Bracket:
    """One corner mount: the stepped spine, and whatever is seated on its ramp.

    Geometry only. What is *drawn* on it - the rail, the bolts, the ribs - is
    :meth:`Overlay._draw_bracket`, because all of that is baked once per window size and this
    has to answer a hit test every time a finger lands.
    """

    def __init__(self, corner: tuple[int, int] | None, spine: Sequence[tuple[int, int]],
                 seats: Sequence[Seat] = ()) -> None:
        self.corner = corner
        self.spine = list(spine)
        self.seats = list(seats)
        self.dir = unit(*spine[1], *spine[2])          # along the ramp
        self.out = (self.dir[1], -self.dir[0])         # away from the corner, by construction
        self.mid = ((spine[1][0] + spine[2][0]) / 2.0, (spine[1][1] + spine[2][1]) / 2.0)

    def on_ramp(self, along: float, depth: float = 0.0) -> tuple[float, float]:
        """A point *along* the ramp from its middle and *depth* in from the rail's centreline."""
        return (self.mid[0] + self.dir[0] * along - self.out[0] * depth,
                self.mid[1] + self.dir[1] * along - self.out[1] * depth)

    def shoulder(self, seat: Seat) -> tuple[tuple[float, float], tuple[float, float], float, float]:
        """Where the rail leaves the ramp for a seat and where it rejoins, and the two angles.

        The angles come back in PIL's convention - degrees clockwise from three o'clock, with the
        second always greater than the first - because both the swell and the long press's fill
        are drawn as arcs and they have to be the same arc.
        """
        (cx, cy), R = seat.centre, seat.swell
        depth = abs((cx - self.mid[0]) * self.out[0] + (cy - self.mid[1]) * self.out[1])
        half = math.sqrt(max(1.0, R * R - depth * depth))
        foot = (cx + self.out[0] * depth, cy + self.out[1] * depth)
        p0 = (foot[0] - self.dir[0] * half, foot[1] - self.dir[1] * half)
        p1 = (foot[0] + self.dir[0] * half, foot[1] + self.dir[1] * half)
        a0 = math.degrees(math.atan2(p0[1] - cy, p0[0] - cx))
        a1 = math.degrees(math.atan2(p1[1] - cy, p1[0] - cx))
        while a1 < a0:
            a1 += 360.0
        return p0, p1, a0, a1

    def path(self) -> list[tuple[float, float]]:
        """The rail's centreline: in from one edge, round whatever is seated on it, out to the
        other.

        Anything seated rides the ramp, which is always the segment between the first two knees -
        the pod has four knees and nothing seated, the mounts have two and up to one.
        """
        if not self.seats:
            return list(self.spine)
        points: list[tuple[float, float]] = [self.spine[0], self.spine[1]]
        for seat in self.seats:
            p0, p1, a0, a1 = self.shoulder(seat)
            points.append(p0)
            points += arc_points(*seat.centre, seat.swell, a0, a1)[1:-1]
            points.append(p1)
        points += list(self.spine[2:])
        return points

    def plate(self, d: ImageDraw.ImageDraw) -> None:
        """Fill the footprint into a mask: the gusset or the pod, plus every seat's swell.

        A module's spine starts and finishes on the same panel edge, so its polygon closes along
        that edge on its own and there is no corner to close it against.
        """
        d.polygon([*([self.corner] if self.corner else []), *self.spine], fill=255)
        for seat in self.seats:
            (cx, cy), R = seat.centre, seat.swell
            d.ellipse([cx - R, cy - R, cx + R, cy + R], fill=255)


_FONT_CANDIDATES = (
    # A terminal is monospaced, so this list is monospaced faces first and only.
    # macOS
    "/System/Library/Fonts/SFNSMono.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Supplemental/Andale Mono.ttf",
    # Raspberry Pi OS / Debian
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # last resort: not mono, but real
)


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """First real TrueType face that exists on this machine; PIL's bitmap default if none do."""
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default(size)


def caption_pulse(phase: float) -> tuple[float, bool]:
    """How far the caption has sunk, and whether its cursor is showing, at monotonic *phase*.

    A raised cosine rather than a square wave: this is a phosphor tube, and a line snapping on
    and off reads as a fault light rather than as work being done. What comes back is a *mix*
    and not an alpha, which is not a detail - see :func:`_mix`. PIL writes into the chrome layer
    rather than compositing onto it, so a translucent letter is not a dimmer letter, it is a
    window onto whatever the camera is pointed at, punched through the slab that was put there
    to stop exactly that.

    Time rather than frames, because the loop does not run at one rate - 25 fps with the camera
    up, 5 while the admin page covers the panel, 4 asleep - and a blink per frame would gallop
    and stall along with it.

    The cursor is a square wave where the breath is not, and for the same reason the breath is
    not one: a cursor that faded in and out would be a thing being dimmed, and a cursor is a
    thing being *switched*. That is what a terminal has always done and it is the half of this
    line anybody reads without looking at it.
    """
    return (BREATH_DEPTH * breath(phase, BREATH_PERIOD_S),
            phase % CURSOR_PERIOD_S < CURSOR_PERIOD_S / 2)


def type_stops(text: str) -> list[float]:
    """When each character of *text* lands, in seconds after the line turned up.

    A terminal prints; it does not publish. The line arriving whole, in one frame, is the one
    thing on this panel that was never a machine doing something - and the cursor on the end of
    it has been claiming otherwise since it replaced the three walking dots.

    Not an even rate, because an even rate is a progress bar. Every character lands up to
    TYPE_JITTER either side of the terminal's own interval, and the hand rests TYPE_GAP longer
    after a space or the end of a clause - which is the half of this anybody actually sees, the
    per-character wobble being under a frame on any line long enough to be squeezed.

    The wobble comes off the golden ratio and off the character's own code point, for the reason
    :func:`cyclops.eye.blink` gives for walking its offsets the same way: the sequence never
    settles into a period anybody can anticipate, a multiply and a mod is the whole cost, and it
    is the same on every boot. Not a PRNG and emphatically not `hash()` - a box that typed
    differently each time it was switched on would be a box with a personality nobody chose.

    The marker is not typed. It is the prompt, and a prompt is on the screen before anything is
    printed at it, so it lands at zero and the first frame of any caption is `› _`.

    Whatever it all comes to is squeezed into TYPE_MAX_S - see that constant for whose number it
    really is.
    """
    lead = len(MARKER) if text.startswith(MARKER) else 0
    stops: list[float] = []
    when = 0.0
    for i, char in enumerate(text):
        if i >= lead:
            drift = 2.0 * (((i + ord(char)) * BLINK_DRIFT) % 1.0) - 1.0
            when += TYPE_CHAR_S * (1.0 + TYPE_JITTER * drift)
            if i and text[i - 1] in TYPE_REST:
                when += TYPE_CHAR_S * TYPE_GAP
        stops.append(when)
    return [stop * min(when, TYPE_MAX_S) / when for stop in stops] if when else stops


class Typist:
    """Which sentence the terminal is printing, and how much of it is on the screen.

    The overlay's one piece of remembered state outside the eye, and it is the same three fields
    for the same reason - see :meth:`cyclops.eye.EyeEngine.look`, which eases out of the mood it
    was in by remembering exactly this much. A frame is handed what the panel is showing and
    never what it was showing a moment ago, so the only thing that can notice a sentence has
    *changed* is the thing that saw the last one.
    """

    def __init__(self) -> None:
        self._key = ""
        self._at = 0.0
        self._stops: list[float] = []

    def printed(self, key: str, phase: float) -> int:
        """How many characters of *key* have landed by monotonic *phase*.

        Any change at all starts the line again at the prompt, the screen being cleared and the
        same sentence coming back included: that is new text appearing, and a terminal has no
        way to tell it from any other.
        """
        if key != self._key:
            self._key, self._at, self._stops = key, phase, type_stops(key)
        return bisect_right(self._stops, phase - self._at)


def rim_breath(phase: float) -> float:
    """How far the border has sunk towards SCREEN. A *mix*, not an alpha - see :func:`_mix`."""
    return RIM_DEPTH * breath(phase, RIM_PERIOD_S)


@dataclass(frozen=True)
class Rect:
    """A hit-testable box in window pixels."""

    x: int
    y: int
    w: int
    h: int

    def contains(self, px: int, py: int) -> bool:
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h


@dataclass(frozen=True)
class Hitboxes:
    """Where the interactive elements ended up, for the mouse callback to test against."""

    volume: Rect  # the knob, and the only control on this panel you turn rather than press
    eye: Rect  # his face, which opens what the box has kept
    heat: Rect  # the gauge, which is read - and, tapped, opens the screen the rest of it is on


def halo_alpha(width: int, height: int) -> np.ndarray:
    """A 0..1 mask that is bright along every edge and gone a little way inside it.

    Built once per window size: the distance to the nearest edge, run through a squared ramp so
    the light drops off fast enough to stay a rim rather than a fog over the picture.
    """
    core = max(2, round(HALO_CORE * height))
    fall = max(6, round(HALO_FALLOFF * height))
    xs = np.minimum(np.arange(width), width - 1 - np.arange(width))
    ys = np.minimum(np.arange(height), height - 1 - np.arange(height))
    dist = np.minimum(ys[:, None], xs[None, :]).astype(np.float32)
    ramp = np.clip((core + fall - dist) / fall, 0.0, 1.0)
    return ramp * ramp * HALO_PEAK


def scanline_alpha(width: int, height: int) -> np.ndarray:
    """Every *SCANLINE_EVERY*-th row darkened - the cheapest half of looking like a tube."""
    mask = np.zeros((height, width), dtype=np.float32)
    mask[::SCANLINE_EVERY] = SCANLINE_ALPHA
    return mask


def vignette_alpha(width: int, height: int) -> np.ndarray:
    """Corner shading, zero in the middle and squared up towards the corners."""
    ys = (np.arange(height, dtype=np.float32) - (height - 1) / 2) / max(1.0, height / 2)
    xs = (np.arange(width, dtype=np.float32) - (width - 1) / 2) / max(1.0, width / 2)
    radius = np.sqrt(ys[:, None] ** 2 + xs[None, :] ** 2) / math.sqrt(2.0)
    ramp = np.clip((radius - VIGNETTE_FROM) / (1.0 - VIGNETTE_FROM), 0.0, 1.0)
    return ramp * ramp * VIGNETTE_ALPHA


def tube_field(width: int, height: int, radius: int) -> np.ndarray:
    """Signed distance to the tube's edge: negative inside the glass, zero on it, positive out.

    One field, and everything the face is made of comes off it - the coverage that anti-aliases
    the corners, the dark band hugging the rim, and the phosphor wash that fades up out of it.
    That is why it is a distance and not a mask: a mask can only say in or out, and every one of
    those three wants to know *how far* in.

    Distance to a rounded rectangle, which is the standard fold: measure to the corner box, keep
    the outside part as a radius and the inside part as the larger of the two axes, then step in
    by the fillet. Cheaper than rasterising the shape at four times the size and it is exact at
    every pixel rather than averaged, which is what lets the rim band be a hard edge.
    """
    ys = (np.arange(height, dtype=np.float32) - (height - 1) / 2.0)[:, None]
    xs = (np.arange(width, dtype=np.float32) - (width - 1) / 2.0)[None, :]
    qx = np.abs(xs) - (width / 2.0 - radius)
    qy = np.abs(ys) - (height / 2.0 - radius)
    outside = np.sqrt(np.maximum(qx, 0.0) ** 2 + np.maximum(qy, 0.0) ** 2)
    return outside + np.minimum(np.maximum(qx, qy), 0.0) - radius


def tube_alpha(width: int, height: int, radius: int, bezel: int, top: int) -> np.ndarray:
    """How opaque the monitor's whole front is, from its outside edge in.

    One field for the case *and* the glass, which is the only way to get what a monitor actually
    looks like: no edge between them at all. Opaque and near-black hard against the outside,
    easing over the moulding's width into glass you can see the room through, flat from there in.
    Draw them as two shapes and there is a seam wherever they meet, and no amount of profile on
    the moulding hides it - a ring of lit metal round a dark hole reads as a glowing pill.

    Opacity is the only currency this surface has. SCREEN is near enough black that darkening it
    says nothing (see TERM_ALPHA), so every shade here is depth of glass instead: the ease from
    the rim, the shadow the case's top lip drops down it, and the raster.

    *top* is where the front sits on the panel, and it is an argument rather than a detail because
    the raster has to land on the filter's own rows. Phased any other way the two beat against
    each other and the one surface here that is literally a CRT ends up the flattest thing on it.
    """
    edge = tube_field(width, height, radius)
    inward = np.maximum(-edge, 0.0)  # how far in from the case's own edge, in pixels
    open_ = np.clip(inward / max(1.0, bezel), 0.0, 1.0)
    open_ = open_ * open_ * (3.0 - 2.0 * open_)  # smooth at both ends, so neither end is a line
    body = 255.0 - (255.0 - TERM_ALPHA) * open_
    # The case's own shadow, falling down the glass from under its top lip.
    drop = max(1.0, TUBE_SHADOW * height)
    down = np.clip(1.0 - np.arange(height, dtype=np.float32) / drop, 0.0, 1.0)
    body += TUBE_SHADOW_A * (down * down)[:, None] * open_
    # ...and the raster, on the filter's pitch and in the filter's phase, and only where there is
    # glass for it to be on - a scanline running out across the moulding is a crack in the case.
    rows = (np.arange(height) + top) % SCANLINE_EVERY == 0
    body[rows] += TERM_SCAN * open_[rows]
    return np.minimum(body, 255.0) / 255.0 * np.clip(0.5 - edge, 0.0, 1.0)


def tube_glow(width: int, height: int, radius: int, bezel: int) -> np.ndarray:
    """The phosphor the face is made of, fading up out of the dark surround towards the middle.

    The half of the bulge the opacity cannot say. Shade alone gets a flat sheet with a dark
    border; what makes a tube look like it is standing proud is that the glass is *lit* in the
    middle - not by anything on it, just the wash a driven phosphor sits in when it is switched
    on with nothing to show.

    Held very low on purpose. This runs under a live camera and under the line anybody is meant
    to be reading, and a green fog over either is worse than a flat screen.
    """
    inward = np.maximum(-tube_field(width, height, radius), 0.0)
    lit = np.clip((inward - bezel) / max(1.0, TUBE_GLOW), 0.0, 1.0)
    return lit * lit * (3.0 - 2.0 * lit) * TUBE_GLOW_A


def glare_alpha(width: int, height: int, radius: int, bezel: int) -> np.ndarray:
    """The room, wiped across the front - over the moulding as well as over the glass.

    Across *both* is the whole point, and it is what was missing while the surround was uniformly
    black. A monitor's front is one sheet: the same reflection runs over the bezel and the picture
    without a break, and that continuity is most of what tells you the thing is made of glass.
    Mask the wipe to the screen and the moulding goes dead flat beside it, which reads as a matte
    plastic frame with a shiny window cut in it.

    Stronger over the moulding than over the glass, because it is darker. Gloss black returns
    almost all of what falls on it and a driven phosphor is already putting out light of its own,
    so the same reflection is a much larger fraction of what the surround is doing.
    """
    ys = np.arange(height, dtype=np.float32)[:, None]
    xs = np.arange(width, dtype=np.float32)[None, :]
    # How much of the lamp reaches here at all. Measured in real pixels rather than in fractions
    # of each axis - the face is five times wider than it is deep, so a falloff computed on
    # normalised coordinates comes out as an ellipse lying on its side rather than as light.
    reach = np.exp(-(((xs - GLARE_X * width) ** 2 + (ys - GLARE_Y * height) ** 2)
                     / (GLARE_REACH * height) ** 2))
    # ...and the streak it draws, running down and to the right across the whole front.
    down = ys / max(1.0, height - 1)
    along = xs / max(1.0, width - 1)
    streak = np.exp(-(((down - (GLARE_AT + GLARE_TILT * along)) / GLARE_DEPTH) ** 2))
    edge = tube_field(width, height, radius)
    open_ = np.clip(np.maximum(-edge, 0.0) / max(1.0, bezel), 0.0, 1.0)
    on_case = 1.0 + GLARE_ON_CASE * (1.0 - open_ * open_ * (3.0 - 2.0 * open_))
    lit = GLARE_AMBIENT + (1.0 - GLARE_AMBIENT) * reach * streak
    return lit * GLARE_ALPHA * on_case * np.clip(0.5 - edge, 0.0, 1.0)


def sheen_alpha(width: int, height: int, radius: int) -> np.ndarray:
    """The light the case's own edge catches, all the way round it.

    Driven off which way the moulding faces rather than off where it is. The distance field's
    gradient is the surface normal, so dotting it with the direction to the lamp says how squarely
    each piece of the rim is turned into the light - brightest at the top-left where the lamp is,
    easing round to dimmest at the bottom-right, with no line anywhere for the eye to catch on.

    It was a horizontal band across the top before, and that is what put the hard wedge in the lit
    corner: the rim term follows the outline round the fillet while the band ends on a straight
    row, so the two disagreed exactly where the corner turns. A rim light has to be cut by the
    shape it is on or not cut at all.

    *SHEEN_AMBIENT* is the floor under it, and it is the difference between a moulding and a
    silhouette. Nothing in a real room is lit from one side only - there is always something
    bouncing back into the far corner - and an unlit edge reads as a hole cut in the panel rather
    than as the dark side of an object.
    """
    edge = tube_field(width, height, radius)
    gy, gx = np.gradient(edge)
    length = np.maximum(np.sqrt(gx * gx + gy * gy), 1e-6)
    ys = np.arange(height, dtype=np.float32)[:, None]
    xs = np.arange(width, dtype=np.float32)[None, :]
    lx, ly = GLARE_X * width - xs, GLARE_Y * height - ys
    reach = np.maximum(np.sqrt(lx * lx + ly * ly), 1e-6)
    facing = np.clip((gx / length) * (lx / reach) + (gy / length) * (ly / reach), 0.0, 1.0)
    near = np.clip(1.0 - np.maximum(-edge, 0.0) / max(1.0, SHEEN_D), 0.0, 1.0)
    lit = SHEEN_AMBIENT + (1.0 - SHEEN_AMBIENT) * facing * facing
    return near * near * lit * SHEEN_A * np.clip(0.5 - edge, 0.0, 1.0)


def _over(
    dst: tuple[np.ndarray, np.ndarray], colour: tuple[int, int, int], alpha: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Source-over one flat-coloured layer onto an accumulating (rgb, alpha) pair."""
    dst_rgb, dst_a = dst
    src = np.asarray(colour, dtype=np.float32)
    a = alpha[..., None]
    out_a = alpha + dst_a * (1.0 - alpha)
    carried = dst_rgb * dst_a[..., None] * (1.0 - a)
    out_rgb = (src * a + carried) / np.maximum(out_a[..., None], 1e-6)
    return out_rgb, out_a


def _to_image(rgb: np.ndarray, alpha: np.ndarray) -> Image.Image:
    rgba = np.empty((*alpha.shape, 4), dtype=np.uint8)
    rgba[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    rgba[:, :, 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


class Overlay:
    """Renders the kiosk chrome for a given window size, and remembers where it put things."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        scale = height / 480.0  # the official 7" panel is the reference layout
        self.scale = scale
        self.line = max(1, round(LINE * height))
        self.pad = max(4, round(PAD * height))
        self.radius = max(2, round(FRAME_RADIUS * height))
        self.frame = Rect(0, 0, width, height)

        def px(value: float) -> int:
            return round(value * scale)

        self.rail_w = max(4, px(RAIL))
        self.bolt_r = max(2, px(BOLT_R))
        self.font_mode = _load_font(max(11, round(27 * scale)))
        self.font_read = _load_font(max(9, round(21 * scale)))
        self.font_tab = _load_font(max(8, round(15 * scale)))
        self.font_brand = _load_font(max(7, round(14 * scale)))
        self.font_micro = _load_font(max(7, round(12 * scale)))
        self.font_caption = _load_font(max(8, round(14 * scale)))
        # Where the pod's one row sits. A centre rather than a baseline, because _text centres on
        # the y it is given, and it has to finish above the rail along the pod's bottom edge.
        self.row = round(22 * scale)
        self._clock_w = self.font_read.getlength("00:00")
        # Both tags, measured the same way and to the same width - they sit side by side and a
        # pair of slabs that differ by two pixels reads as a mistake. The face is not guaranteed
        # monospaced (see _FONT_CANDIDATES' last resort), so this is a max and not one call.
        self._tag_w = max(map(self.font_micro.getlength, (HEAT_WORD, "REC"))) + round(7 * scale) * 2
        self._gap = max(4, round(18 * scale))
        self._seg = (max(3, round(9 * scale)), max(6, round(16 * scale)), max(2, round(5 * scale)))
        # The three spines. Each is [edge, knee, knee, edge], and the ramp between the knees is
        # built from its own start rather than from a second table entry, so it is a true 45
        # whatever the rounding does to the numbers either side of it.
        depth, step = max(10, px(POD_H)), max(4, px(POD_STEP))
        ramp = depth - step
        bot, bstep = max(12, px(BOT_L)), max(4, px(BOT_L_STEP))
        blegs = bot - bstep
        rout, rstep = max(12, px(BOT_R_OUT)), max(4, px(BOT_R_STEP))
        rlegs = rout - max(4, px(BOT_R_LAND))
        # The pod is exactly as wide as what it is showing, and there is one of it per number of
        # tags. It used to be cut once at its worst case - both tags lit - which left a tag-shaped
        # hole in the middle of it whenever neither was, and that hole is bigger than the meter.
        # Three plates are three masks and three rails, built the first time each is needed and
        # kept; the alternative is dead space on the panel at all times to save a bake that
        # happens twice a session.
        self._meter_w = METER_SEGMENTS * self._seg[0] + (METER_SEGMENTS - 1) * self._seg[2]
        self._stop = max(3, round(POD_STOP * scale))
        self._tag_gap = max(2, round(POD_TAG_GAP * scale))
        self._pod_pad = max(4, px(POD_PAD))
        self.spines = {
            "bl": [(0, height - bot), (bstep, height - bot),
                   (bstep + blegs, height - bot + blegs), (bstep + blegs, height)],
            "br": [(width - rout, height), (width - rout, height - rstep),
                   (width - rout + rlegs, height - rstep - rlegs),
                   (width, height - rstep - rlegs)],
        }
        self.brackets = {
            name: Bracket((0 if name[1] == "l" else width, height), spine)
            for name, spine in self.spines.items()
        }
        # ...and the pods, one per tag count. Geometry only, so all three cost nothing to hold.
        self.pods, self.pod_boxes = {}, {}
        for tags in range(3):
            # meter, stop, [tag, gap, tag,] stop, clock - and with no tags at all the two stops
            # collapse into the one between the meter and the clock.
            content = round(
                self._meter_w + self._stop + self._clock_w
                + (tags * self._tag_w + (tags - 1) * self._tag_gap + self._stop if tags else 0)
            )
            flat = content + 2 * self._pod_pad
            left = width // 2 - flat // 2
            right = left + flat
            # Wound right to left, so that (dy, -dx) points out of the pod's own body - the same
            # sign that means "away from the corner" on the two mounts, and what lets one rail
            # routine light the correct side of all three.
            self.pods[tags] = Bracket(None, [
                (right + ramp, 0), (right + ramp, step), (right, depth),
                (left, depth), (left - ramp, step), (left - ramp, 0),
            ])
            # The readouts' own box inside the flat: everything in there is placed off this.
            self.pod_boxes[tags] = Rect(left + self._pod_pad, 0, content, depth)
        self.pod = self.pod_boxes[0]  # its resting size, and every one of them is this deep
        self.reticle_r = max(8, round(RETICLE_R * height))

        # Him, riding the big bracket's ramp. Everything about where he is comes off that ramp,
        # so moving the bracket moves him and the rail still goes round him.
        self.eye_r = max(10, round(EYE_R * height))
        eye_swell = self.eye_r + max(2, px(EYE_SHOULDER))
        self.brackets["bl"].seats = [
            Seat(self.brackets["bl"].on_ramp(0.0, eye_swell * EYE_SEAT), self.eye_r,
                 eye_swell - self.eye_r)
        ]
        self.eye_seat = self.brackets["bl"].seats[0]
        self.eye = (round(self.eye_seat.centre[0]), round(self.eye_seat.centre[1]))
        self.shoulder = eye_swell

        # ...and the two switches, sunk into the small bracket's rail rather than parting it.
        self.btn_r = max(8, px(BTN_R))
        self._dial_stroke = max(3.0, 3.6 * scale)  # the scale both dials are read against
        spacing = max(self.btn_r + 4, px(BTN_SPACING))
        self.switches = {
            name: self.brackets["br"].on_ramp(offset)
            for name, offset in zip(SWITCHES, (-spacing, spacing), strict=True)
        }
        # ...and the column a drag on the knob opens, standing in the knob's own column so that a
        # hand travelling up from it stays beside what it is setting rather than across the panel.
        # It stops short of the disc it came out of: the two are one control, and a track running
        # into the bezel would read as a thermometer bolted to a dial.
        knob_x, knob_y = self.switches[VOLUME]
        track_w = max(8, px(SLIDER_W))
        track_top = max(2, px(SLIDER_TOP))
        track_foot = round(knob_y - self.btn_r - max(2, px(SLIDER_GAP)))
        self.slider = Rect(round(knob_x - track_w / 2), track_top, track_w,
                           max(track_w, track_foot - track_top))
        self._rung = self.slider.h / SLIDER_RUNGS

        # The terminal, solved from the bottom up: the glass is as deep as two lines and their
        # padding, the moulding goes round it, and the whole thing stands on the panel's own
        # bottom edge. Written the other way round - a top edge at a chosen height with a screen
        # hung underneath - the glass ends up short of the edge at one window size and past it at
        # another, because the bottom is the only fixed thing here.
        self.caption_h = round(24 * scale)  # one line of it
        pad = max(2, px(TERM_PAD))
        bezel = max(2, px(TERM_BEZEL))
        floor_ = self.height - max(2, px(TERM_FOOT))
        roof = floor_ - CAPTION_LINES * self.caption_h - 2 * pad - 2 * bezel
        # Solved off where each mount's rail actually is at the monitor's own mid-height rather
        # than off the spine table, which is what puts the clearance where somebody looking at it
        # would measure it - and what keeps it right if either mount is ever moved.
        middle = (roof + floor_) / 2.0
        self.ear_y = middle
        clear = max(4, px(TERM_CLEAR)) + self.rail_w / 2.0
        left = round(self._rail_at("bl", middle) + clear)
        right = round(self._rail_at("br", middle) - clear)
        self.term = Rect(left, roof, right - left, floor_ - roof)
        # ...and the glass inside it, inset by the moulding on every side. Both rectangles are
        # kept: the chassis is what the plate mask and the two bolts are measured from, and the
        # tube is what the caption, the scanlines and the vignette live on. Solving the chassis
        # first and insetting is the way round that keeps the *interior* exactly two lines and
        # their padding at every window size - inset the other way and the bezel is what rounds.
        self.tube = Rect(self.term.x + bezel, self.term.y + bezel,
                         self.term.w - 2 * bezel, self.term.h - 2 * bezel)
        self.case_r = max(2, px(TERM_RADIUS))
        self.bezel = bezel
        self.bloom_r = max(1.0, BLOOM_R * scale)
        # The two clamps, one either side. Geometry only here - what they are made of is in
        # :meth:`_draw_terminal`, and it is the same rail the mounts are.
        ear_h = max(4, round(self.term.h * TERM_EAR_H))
        self.ear_w = max(4, px(EAR_RAIL))
        # A bolt sized to the member it passes through, and never smaller than half of it plus a
        # margin. That is what makes the head cover the strap's square butt at every window size
        # rather than at the one somebody looked at: the far corner of a butt cap sits half a
        # member's width from the centreline, so anything from there out hides it.
        self.ear_bolt = max(2.0, self.ear_w / 2.0 + max(1.0, EAR_BOLT * self.scale))
        half = self.ear_w // 2 + 1
        top = round(middle - ear_h / 2)
        # Each bracket's own bounds, strap included. The strap stands *on* the case's edge rather
        # than inside it - a clamp grips an edge, and one set back onto the face is a bar lying
        # across the screen - so the box reaches half a member past that edge at one end.
        self.ears = [
            Rect(round(self._rail_at("bl", middle)), top,
                 self.term.x + half - round(self._rail_at("bl", middle)), ear_h),
            Rect(self.term.right - half, top,
                 round(self._rail_at("br", middle)) - self.term.right + half, ear_h),
        ]
        # The text inside the screen, inset from the glass's own edge. It used to be measured in
        # from the two rails instead, because the glass ran under both of them and its corners
        # were a rail's half-width further out and buried; nothing is buried now, so the thing to
        # measure from is the thing anybody can see.
        inset = max(3, round(6 * scale))
        self.caption_left = round(self.tube.x + inset)
        self.caption_right = round(self.tube.right - inset)
        self.caption_top = self.tube.y + pad
        self.caption_y = self.caption_top + self.caption_h / 2  # the *first* line's middle now

        self._halo = halo_alpha(width, height)
        # One set of hairlines for the whole sheet, before the filter and the chrome are built
        # from it: the plates wear them and so do the bars bolted across them, and a scratch that
        # runs from one onto the other is most of what says they are the same piece of metal.
        self._marks = material.scratches(width, height, PLATE_SCRATCHES, (1.0, -1.0))
        self._filter = self._build_filter()
        self._backdrops: dict[int, Image.Image] = {}
        self._chromes: dict[int, Image.Image] = {}
        self._plate = self._build_plate()
        self._glass = self._build_glass()
        self._chrome_base = self._build_chrome()
        # One engine per window size: it owns the geometry, and it remembers which mood it is
        # easing out of, which is why it is built here and not per frame.
        # Where the places he looks actually are, from where he is bolted. :mod:`cyclops.eye`
        # knows their names and nothing else - it must not learn about this panel, which is what
        # lets tools/eye_sheet.py import it on its own - so this is the one seam where a name
        # becomes a direction. Unit vectors: how far he turns is the mood's `gaze`, and at nine
        # pixels of travel it is the direction that reads, not the distance.
        #
        # AHEAD is not in here and never can be. (0, 0) is the pupil dead centre, looking out of
        # the glass, and it is the only place he looks that is not on the panel.
        #
        # WORK and AWAY fall through from the defaults, because there is nothing on the screen to
        # point at: the panel resolves what it can see and the eye keeps its own fictions.
        self.places = LANDMARKS | {
            FRAME: unit(*self.eye, self.width // 2, self.height // 2),  # the reticle, exactly
            WORDS: unit(*self.eye, self.caption_left, self.caption_y),
            DIALS: unit(*self.eye, self.pod.x + self.pod.w / 2, self.pod.h / 2),
        }
        self.engine = EyeEngine(self.eye_r, self.line, SCREEN, MOODS[IDLE], places=self.places)
        # ...and the other one, for the same reason: it remembers which sentence it is printing
        # and which frame that sentence turned up on. See :class:`Typist`.
        self._typist = Typist()
        self._bases: dict[tuple[str, bool, str], Image.Image] = {}
        # The moving half of each instrument, one tile per appearance it can have. A level moves
        # when a finger moves it and a temperature moves once every five seconds, so at 25 frames
        # a second almost every frame asks for the tile the frame before it already built.
        self._knobs: dict[tuple[int | None, bool], Image.Image] = {}
        self._needles: dict[tuple[int | None, str, bool], Image.Image] = {}
        self._cursor_w = self.font_caption.getlength(CURSOR)
        self.hitboxes = self._layout()
        self.menu_card, self.menu_cells = self._menu_layout()
        self._scrim: Image.Image | None = None  # built on the first long press, then kept

    # ---- layout ----

    def _rail_at(self, name: str, y: float) -> float:
        """Where mount *name*'s rail centreline crosses the horizontal line *y*.

        Walked along :meth:`Bracket.path` rather than solved, because that path is not always a
        straight line where it is asked about: on the left the terminal's rail meets it on the arc
        that goes round his housing, a dozen pixels above where the ramp picks up again. Whichever
        crossing is nearest the middle of the panel is the one that bounds the terminal, so a
        bracket that grew a second seat would still answer this correctly.
        """
        points = self.brackets[name].path()
        hits = [
            ax + (bx - ax) * (y - ay) / (by - ay)
            for (ax, ay), (bx, by) in zip(points, points[1:], strict=False)
            if ay != by and (ay - y) * (by - y) <= 0
        ]
        return max(hits) if name[1] == "l" else min(hits)

    def _disc(self, centre: tuple[float, float], radius: float) -> Rect:
        """The bounding box of a round control, which is what a hit test gets to work with."""
        cx, cy = round(centre[0]), round(centre[1])
        r = round(radius)
        return Rect(cx - r, cy - r, r * 2, r * 2)

    def _layout(self) -> Hitboxes:
        """Three round targets in two corners. The rest of the frame is picture.

        They are discs now rather than thirds of a row, which costs area and buys the middle of
        the screen back: the tab row was 17% of the panel and the three cells here are under 6%
        between them. What keeps that honest is that they are still targets a thumb finds without
        being looked at - 72 px is about 14 mm on the 7" panel, which is over the 9 mm everybody
        agrees is the floor - and that the eye, the one you press to go looking for something, is
        by far the biggest of the three.

        Bounding boxes rather than circles because that is what the kiosk's hit test takes, and
        the two switches are spaced along the ramp so their boxes do not overlap: a tap in a
        corner shared by two controls would silently belong to whichever was tested first.
        """
        return Hitboxes(
            volume=self._disc(self.switches[VOLUME], self.btn_r),
            eye=self._disc(self.eye, self.eye_r),
            heat=self._disc(self.switches[HEAT], self.btn_r),
        )
    def _menu_layout(self) -> tuple[Rect, dict[str, Rect]]:
        """The power menu's card and its rows, sized off the panel like everything else here.

        Centred in what is left beside him rather than in the panel. It used to dodge him
        *upwards*, because he stood in the middle of the tab row and the honest centre of the
        screen was his face; with him in a corner it can dodge sideways instead, which is far
        cheaper - on the 7" panel the card only has to give up 13 px of centring to clear his
        swell, and on a small window where it would otherwise land on him it slides right until
        it does.

        Laid out once, in the constructor, because a hit test has to agree with a drawing and the
        cheapest way to make sure of that is for there to be only one of them.
        """
        width = max(160, round(MENU_W * self.height))
        row_h = max(22, round(MENU_ROW_H * self.height))
        head_h = max(14, round(MENU_HEAD_H * self.height))
        pad = max(3, round(MENU_PAD * self.height))
        height = head_h + row_h * len(MENU_ROWS) + pad * 2
        y = max(0, (self.height - height) // 2)
        x = self.frame.x + (self.width - width) // 2
        clear = self.eye[0] + self.shoulder + max(4, round(10 * self.scale))
        if x < clear:
            x = min(clear, self.frame.right - width)
        card = Rect(x, y, width, height)
        cells = {}
        row_y = card.y + pad + head_h
        for key, _ in MENU_ROWS:
            cells[key] = Rect(card.x + pad, row_y, card.w - pad * 2, row_h)
            row_y += row_h
        return card, cells

    def menu_hit(self, px: int, py: int) -> str | None:
        """Which row of the power menu a tap landed on, if the menu is up.

        Anywhere off the card is :data:`CANCEL`, which is what tapping outside a dialog has meant
        on every machine since the mouse. Anywhere on it that is not a row - the title line, the
        few pixels between the border and the first row - is nothing at all: a card you can
        dismiss by missing the thing you were aiming at is a card that answers for you.
        """
        for key, cell in self.menu_cells.items():
            if cell.contains(px, py):
                return key
        return None if self.menu_card.contains(px, py) else CANCEL

    # ---- the cached backdrop ----

    def _bracket_mask(self, tags: int) -> np.ndarray:
        """Where the chrome is, as 0..1 - the plates, less the disc his face fills.

        His body is what backs him instead, and it is the same all the way round; without this
        the wash's own edge would run across his face as a tide line. Cut at the swell rather than
        at his rim, so the hole matches what :meth:`_build_plate` fills: he is opaque out to there
        now, and a wash computed under an opaque disc is a wash nobody will ever see. The switches
        keep the wash under them, because their wells are opaque enough not to care and cutting
        two more holes in a mask is two more edges to land in the wrong place.
        """
        plate = Image.new("L", (self.width, self.height), 0)
        d = ImageDraw.Draw(plate)
        for bracket in (*self.brackets.values(), self.pods[tags]):
            bracket.plate(d)
        # ...and the terminal, which has no spine to lay down a polygon from - it is a rectangle
        # standing on the bottom edge, and where it runs under the two mounts the two footprints
        # are simply the same metal, so the union leaves no seam to line up.
        d.rounded_rectangle([self.term.x, self.term.y, self.term.right, self.term.bottom],
                            radius=self.case_r, fill=255)
        # ...and nothing for the two brackets. A plate is what a *mount* stands on - it is the
        # see-through metal a corner assembly is built out of - and giving one to a stay across
        # open picture put a broad stripy rectangle of washed, scanlined chrome behind each
        # clamp. A stay has no plate. It is a bar in the air with a shadow under it, which is
        # exactly what _draw_rail draws when there is nothing beneath it.
        holes = Image.new("L", (self.width, self.height), 0)
        hd = ImageDraw.Draw(holes)
        cx, cy, r = *self.eye, self.shoulder
        hd.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
        inside = np.asarray(plate, np.float32) / 255.0
        inside *= 1.0 - np.asarray(holes, np.float32) / 255.0
        rounded = Image.new("L", (self.width, self.height), 0)
        ImageDraw.Draw(rounded).rounded_rectangle(
            [0, 0, self.width - 1, self.height - 1], radius=self.radius, fill=255
        )
        return inside * (np.asarray(rounded, np.float32) / 255.0)

    def _build_filter(self) -> tuple[np.ndarray, np.ndarray]:
        """The tube filter - a wash, corner shading and scanlines - over the whole panel.

        Unmasked, because none of it depends on where the chrome is: it is built once and every
        plate is cut out of it. It used to be laid over a strip and a tab row that between them
        covered 29% of the panel; two mounts and a pod cover 16% of it, and the picture runs edge
        to edge behind them - this is what the chrome is *made of* rather than something sitting
        under an opaque bar. The plate is darker than it was, though, because a bracket is a
        thing rather than a tint - see PLATE_WASH, which is as far towards SCREEN as it goes.
        """
        shape = (self.height, self.width)
        base: tuple[np.ndarray, np.ndarray] = (
            np.zeros((*shape, 3), dtype=np.float32),
            np.zeros(shape, dtype=np.float32),
        )
        base = _over(base, SCREEN, np.full(shape, PLATE_WASH, dtype=np.float32))
        base = _over(base, material.STEEL_DARK, np.full(shape, PLATE_STEEL, dtype=np.float32))
        base = _over(base, GREEN, np.full(shape, TINT_ALPHA, dtype=np.float32))
        # The sheet every plate is cut from is brushed, and brushed along the panel's own
        # diagonal - the way both ramps run - so one grain serves all four brackets and none of
        # them looks like a different offcut. Laid on as a modulation either way from the wash,
        # never as a fill: the room has to keep coming through, fibres and all.
        ys, xs = np.mgrid[0 : self.height, 0 : self.width].astype(np.float32)
        across, along = (xs + ys) * math.sqrt(0.5), (xs - ys) * math.sqrt(0.5)
        grain = material.grain(across, along)
        base = _over(base, material.STEEL_LIT, np.clip(grain, 0.0, 1.0) * PLATE_GRAIN)
        base = _over(base, (0, 0, 0), np.clip(-grain, 0.0, 1.0) * PLATE_GRAIN)
        base = _over(base, material.STEEL_SPEC, self._marks * PLATE_SCRATCH_ALPHA)
        # ...and lit, by the same lamp as the bars on it: brightest where it stands, falling off
        # towards the far corners, and rubbed unevenly along the grain. Still a modulation - the
        # lift is a tenth, and the room keeps coming through the brighter part as it does the
        # duller.
        lamp = (PLATE_LAMP[0] * self.width, PLATE_LAMP[1] * self.height)
        lit = material.glare(self.width, self.height, lamp, PLATE_REACH * self.height, ambient=0.0)
        rubbed = 1.0 + material.wear(along, seed=material.SEED + 5)
        base = _over(base, material.STEEL_LIT,
                     np.clip(lit * PLATE_LIGHT + rubbed * PLATE_WEAR, 0.0, 1.0))
        base = _over(base, (0, 0, 0), vignette_alpha(self.width, self.height))
        base = _over(base, (0, 0, 0), scanline_alpha(self.width, self.height))
        return base

    def _backdrop(self, tags: int) -> Image.Image:
        """The filter cut to the plates a pod this wide makes. Built on demand, then kept."""
        cached = self._backdrops.get(tags)
        if cached is None:
            rgb, alpha = self._filter
            cached = self._backdrops[tags] = _to_image(rgb, alpha * self._bracket_mask(tags))
        return cached

    def _build_plate(self) -> Image.Image:
        """His body: the solid disc he is drawn on, on its own layer under the chrome.

        This is not the terminal's glass and does not share its argument. The screen is a window
        onto a line of text and may let the room through; this is the face of a machine, and the
        room coming through a machine is the one thing that stops it being one. He is nothing but
        thin rings with empty tile between them, so at anything short of opaque the camera ran
        through his iris, his stator and the gaps between every ring of him at once.

        Out to the *swell* rather than to his rim, because that is where the collar picks the body
        up - see :meth:`_draw_collar`. Filled to the rim instead and there is a ring of translucent
        plate left over between his rim and the collar's inner flank, which reads as a gap round
        the lens rather than as the barrel it is standing in.

        A machined well rather than a flat disc: a turned floor, a shade lighter at its middle
        than at the wall, and the shadow the bezel's near wall drops across it - deepest on the
        side towards the lamp, because that is the wall standing between the lamp and the floor.
        All of it shows only through the gaps between his rings, and that is where the depth of
        him comes from: a flat disc behind a set of rings is a badge, and a floor with a shadow
        falling across it is a socket. The disc's outline is the same ImageDraw ellipse
        :meth:`_bracket_mask` cuts the hole with, so the two agree to the pixel and no tide line
        of wash appears round him.

        A separate layer rather than part of :meth:`_build_chrome`, because the collar's rings and
        the left mount's rail are drawn over the plate and must stay over it.
        """
        layer = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        cx, cy, out = *self.eye, self.shoulder
        inn = out * COLLAR_IN
        x0, y0, x1, y1, xs, ys = self._around(out + 1)
        dist = np.hypot(xs, ys)
        # The floor falls off to the wall on a square, so the middle stays level and the drop
        # gathers where the wall is - a bowl would put the whole thing on a gradient.
        depth = (np.clip(dist / inn, 0.0, 1.0) ** 2)[..., None]
        floor = np.asarray(WELL_FLOOR, np.float32)
        rgb = floor + (np.asarray(WELL_WALL, np.float32) - floor) * depth
        rgb = rgb * (1.0 + material.GRAIN * WELL_GRAIN * material.grain(dist, self._round(xs, ys)))[
            ..., None]
        # Everything from the flank outwards stands above the floor; its shadow falls the other
        # way from the lamp, and only what lands inside the flank is floor.
        lift = max(1.0, (COLLAR_LIFT + WELL_DEPTH) * self.scale)
        shadow = material.cast((dist >= inn).astype(np.float32), lift) * WELL_SHADOW
        rgb = rgb * (1.0 - shadow * (dist < inn))[..., None]
        disc = Image.new("L", (x1 - x0, y1 - y0), 0)
        ImageDraw.Draw(disc).ellipse(
            [cx - out - x0, cy - out - y0, cx + out - x0, cy + out - y0], fill=EYE_PLATE_ALPHA
        )
        layer.alpha_composite(material.to_image(rgb, np.asarray(disc, np.float32) / 255.0),
                              (x0, y0))
        return layer

    def _around(self, reach: float) -> tuple[int, int, int, int, np.ndarray, np.ndarray]:
        """A box of the panel *reach* px round him, clipped to it, and where each pixel is from him.

        The base every field on his housing is built over. His swell comes within three pixels of
        two panel edges, so the box is clipped rather than assumed, and the two coordinate fields
        are broadcast shapes - a row and a column - so a distance or an angle off them costs one
        pass over the box and nothing over the panel.
        """
        cx, cy = self.eye
        x0, y0 = max(0, math.floor(cx - reach)), max(0, math.floor(cy - reach))
        x1 = min(self.width, math.ceil(cx + reach) + 1)
        y1 = min(self.height, math.ceil(cy + reach) + 1)
        ys = (np.arange(y0, y1, dtype=np.float32) - cy)[:, None]
        xs = (np.arange(x0, x1, dtype=np.float32) - cx)[None, :]
        return x0, y0, x1, y1, xs, ys

    def _round(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Arc length round him at his swell, in pixels, for anything turned on a lathe.

        The noise fields in :mod:`cyclops.material` run along a length and do not know they are
        on a loop, so somewhere the length has to start again and the brushing shows a seam. It
        starts under the gland, which is the one place on the collar that is always covered.
        """
        turn = (np.arctan2(ys, xs) - math.radians(GLAND_AT)) % math.tau
        return turn * self.shoulder

    def _build_glass(self) -> Image.Image:
        """The dome over him: the lamp on a sheet of glass, kept off everything that moves.

        One tile the size of his own, laid over the eye after it is painted - see :meth:`render`.
        Two things on it. The lamp's reflection, up the dome towards the lamp, which is
        :func:`material.glare` with the terminal's numbers made his; and the light a curved edge
        gathers, a faint ring just inside the rim, bright on the side facing the lamp and a shade
        darker on the side away from it. Both in the tube's own white, both faint: glass at this
        size is a suggestion or it is a smear over the face.

        The GLASS_* block says where it may and may not lie, and the rule is hard rather than
        eased: the alpha is set to nothing inside GLASS_IN and outside GLASS_OUT after the fades,
        so a spark drifting under the edge of the zone finds no edge to drift under.
        """
        r = self.eye_r
        size = 2 * r + 1
        ys = (np.arange(size, dtype=np.float32) - r)[:, None]
        xs = (np.arange(size, dtype=np.float32) - r)[None, :]
        dist = np.hypot(xs, ys) / r
        lx, ly = material.lamp_2d()
        facing = (xs * lx + ys * ly) / np.maximum(dist * r, 1e-6)
        band = np.clip((dist - GLASS_IN) / GLASS_EASE, 0.0, 1.0)
        band *= np.clip((GLASS_OUT - dist) / GLASS_EASE, 0.0, 1.0)
        spot = (r + lx * r * GLASS_AT, r + ly * r * GLASS_AT)
        glare = material.glare(size, size, spot, r * GLASS_REACH, ambient=0.0)
        rim = np.exp(-(((dist - GLASS_OUT + GLASS_EASE) / GLASS_EASE) ** 2))
        white = band * np.minimum(GLASS_GLARE * glare + GLASS_RIM * rim * np.clip(facing, 0, 1),
                                  GLASS_GLARE)
        shade = band * GLASS_SHADE * np.clip(-facing, 0.0, 1.0) * np.clip(dist - GLASS_IN, 0, 1)
        rgb, alpha = _over((np.zeros((size, size, 3), np.float32), np.zeros((size, size),
                                                                             np.float32)),
                           WHITE, white)
        rgb, alpha = _over((rgb, alpha), (0, 0, 0), shade)
        alpha[(dist < GLASS_IN) | (dist > GLASS_OUT)] = 0.0
        return _to_image(rgb, alpha)

    def _build_chrome(self) -> Image.Image:
        """The two mounts and the reticle, on transparency - everything of a fixed size.

        Drawn once and kept: nothing in here depends on the state, only on the window size. The
        pod is not in here, because it is as wide as the tags it is showing - see :meth:`_chrome`,
        which lays one on top of this. Nor is the border around the outside: it carries the state
        colour, so it belongs to the per-state base and goes on last of all.

        There is no bloom over any of it, which the doubled hairlines this replaces did have. A
        Gaussian blur over a 17 px rail is not a glow, it is a lamp - and the profile across the
        rail's own width is already doing the job the bloom was there to do, which is to say that
        the chrome has a thickness.
        """
        layer = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        # The terminal before either mount and before both instruments, because every one of them
        # is what buries an end of it. Order is the whole illusion: drawn last this is a box lying
        # on the panel, and drawn first it is a box behind it.
        self._draw_terminal(layer)
        # ...then his collar, and only then the mounts, so the left one's rail straps *over* the
        # barrel where the two share an arc. The other way round the collar's own rings are
        # written across the inner half of the rail and the joint comes apart.
        self._draw_collar(layer)
        for bracket in self.brackets.values():
            self._draw_bracket(layer, bracket)
        # The loom after the mounts and not before them, which is the opposite of the terminal's
        # rule and for the opposite reason: nothing is meant to bury these except the gland. Drawn
        # first, the corner's own webbing and the rail's cast shadow simply write over the top of
        # them - ImageDraw writes rather than composites - and the cables vanish.
        self._draw_loom(layer)
        self._draw_reticle(layer)
        # The two dial faces, which used to be baked once per state with the switches they
        # replace. Neither wears the state's accent - a volume and a board temperature are true
        # whether or not anybody is talking to him - so neither has any business being rebuilt
        # every time the state changes, and they belong here with the rail they are bolted to.
        for name in SWITCHES:
            self._draw_instrument(layer, name)
        return layer

    def _chrome(self, tags: int) -> Image.Image:
        """...and the same with a pod of the right width on it. Built on demand, then kept.

        Three of these exist at most, and each costs one rail - one Gaussian and seventeen offset
        strokes - the first time a session starts recording or a board goes hot. That is a couple
        of frames, twice a session, against a tag-shaped hole sitting in the middle of the panel
        the rest of the time.
        """
        cached = self._chromes.get(tags)
        if cached is None:
            layer = self._chrome_base.copy()
            self._draw_bracket(layer, self.pods[tags])
            cached = self._chromes[tags] = layer
        return cached

    def _rail_colour(self, across: float) -> tuple[int, int, int]:
        """The three-band profile the collar and the dial bezels still bend into rings.

        *across* runs 1 at the outer lip to 0 at the inner flank. The bars themselves no longer
        use it - they are steel under a lamp, see :meth:`_draw_rail` - and this stays until the
        rings are.

        Three bands rather than one ramp: a smooth gradient over seventeen pixels reads as a
        blur, and a chamfer reads as an edge. The lit band is well short of full phosphor -
        pushed any further it stops looking like steel catching the screen's own light and starts
        looking like a neon tube laid on the panel, which was the first thing anybody said about
        it.
        """
        steel = mix(GREEN_MID, SCREEN, STEEL)
        if across > RAIL_LIP:
            return mix(steel, GREEN, 0.34)
        if across > RAIL_BODY:
            return mix(steel, GREEN, 0.18 * (across - RAIL_BODY) / (RAIL_LIP - RAIL_BODY))
        return mix(steel, SCREEN, 0.80 * (RAIL_BODY - across) / RAIL_BODY)

    @staticmethod
    def _bar_frame(
        points: Sequence[tuple[float, float]], half: float, margin: float,
        x0: int, y0: int, width: int, height: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Where every pixel of a box sits on a mitred bar *half* wide either side of *points*.

        Signed distance across from the centreline (positive on :func:`offset_path`'s side), arc
        length along it, the unit tangent of the segment the pixel belongs to, and how far past
        a square end it is. Pixels no segment claims come back infinitely far across.

        :func:`material.bar_field` measures to the nearest point of the polyline, which rounds
        every knee; a bar cut from flat stock is mitred, and at a mitre the face and both edge
        lines have to turn through the angle without a break. The bisector of two lines is the
        set of points the same distance from both, so a pixel goes to whichever of its candidate
        segments' lines it lies nearest - and a segment is a candidate only within its own
        length plus the mitre's reach at each end, so a segment further round a curve cannot
        claim a pixel its line happens to pass close to. *margin* extends the two free ends,
        which is what lets the contact shadow wrap a square end.
        """
        pts = [points[0]]
        for p in points[1:]:
            if math.hypot(p[0] - pts[-1][0], p[1] - pts[-1][1]) > 1e-6:
                pts.append(p)
        ys = (np.arange(height, dtype=np.float32) + y0)[:, None]
        xs = (np.arange(width, dtype=np.float32) + x0)[None, :]
        dirs = [unit(*a, *b) for a, b in zip(pts, pts[1:], strict=False)]
        lens = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:], strict=False)]
        reach = [0.0] * len(pts)
        for k in range(1, len(pts) - 1):
            cos = dirs[k - 1][0] * dirs[k][0] + dirs[k - 1][1] * dirs[k][1]
            turn = math.acos(max(-1.0, min(1.0, cos)))
            reach[k] = half * min(math.tan(turn / 2.0), 2.0)
        best = np.full((height, width), np.inf, np.float32)
        across, along, tx, ty, beyond = (np.zeros((height, width), np.float32) for _ in range(5))
        run = 0.0
        last = len(dirs) - 1
        for i, ((ax, ay), (dx, dy), length) in enumerate(zip(pts, dirs, lens, strict=False)):
            t = (xs - ax) * dx + (ys - ay) * dy
            s = (xs - ax) * dy - (ys - ay) * dx
            lo = -reach[i] - (margin if i == 0 else 0.0)
            hi = length + reach[i + 1] + (margin if i == last else 0.0)
            closer = (t >= lo) & (t <= hi) & (np.abs(s) < best)
            best = np.where(closer, np.abs(s), best)
            across, along = np.where(closer, s, across), np.where(closer, run + t, along)
            tx, ty = np.where(closer, dx, tx), np.where(closer, dy, ty)
            past = np.maximum(np.maximum(-t, 0.0) if i == 0 else 0.0,
                              np.maximum(t - length, 0.0) if i == last else 0.0)
            beyond = np.where(closer, past, beyond)
            run += length
        across = np.where(np.isinf(best), np.inf, across)
        return across, along, tx, ty, beyond

    def _draw_rail(self, layer: Image.Image, points: Sequence[tuple[float, float]],
                   thick: int | None = None) -> None:
        """A bar of flat steel along the spine, under the panel's one lamp, and its shadows.

        One mitred outline - the spine offset both ways and closed, so the knees are corners and
        the ends are square - filled for coverage, and one frame off the same spine
        (:meth:`_bar_frame`) that says how far across the bar every pixel is and which edge it is
        nearest. The section is read straight off that: a chamfer's width in from either edge
        and a flat between. The lamp lights the chamfer turned towards it into one crisp line,
        the flat stays the steel's own grey brushed along its length, and the chamfer turned away
        gives back a duller glint of the lit plate under it. Nothing here decides which edge is
        which: the edge's own normal against :func:`material.lamp_2d` does, so a diagonal rail's
        bright edge is its upper-left one and a horizontal rail's is its top.

        The shadows are still most of the work. A bar with a bright edge and no shadow reads as
        a drawing of a bar; the same bar with a hard line of contact shadow hugging it and a soft
        one falling away from the lamp sits *on* something. Both go on as composites from a
        private tile, never as writes - a translucent write is a window onto the camera - and
        only over the box the bar occupies, so the eight of these a panel builds are eight small
        tiles and not eight full frames.

        Everything here is once per window size. Nothing in it may be reached from a frame.
        """
        thick = self.rail_w if thick is None else thick
        half = thick / 2.0
        lift = max(1.0, RAIL_LIFT * self.scale)
        contact = RAIL_CONTACT_W * self.scale
        drop = lift * (material.SHADOW_DROP + 3 * material.SHADOW_SOFT)
        reach = math.ceil(half + contact + drop) + 1
        rim = [*offset_path(points, half), *reversed(offset_path(points, -half))]
        x0 = max(0, math.floor(min(p[0] for p in rim)) - reach)
        y0 = max(0, math.floor(min(p[1] for p in rim)) - reach)
        x1 = min(self.width, math.ceil(max(p[0] for p in rim)) + reach + 1)
        y1 = min(self.height, math.ceil(max(p[1] for p in rim)) + reach + 1)
        if x1 <= x0 or y1 <= y0:
            return
        w, h = x1 - x0, y1 - y0
        outline = Image.new("L", (w * BAR_SS, h * BAR_SS), 0)
        ImageDraw.Draw(outline).polygon(
            [((px - x0) * BAR_SS, (py - y0) * BAR_SS) for px, py in rim], fill=255
        )
        cover = np.asarray(outline.reduce(BAR_SS), np.float32) / 255.0
        across, along, tx, ty, beyond = self._bar_frame(points, half, contact + 1.0, x0, y0, w, h)
        across = np.where(np.isfinite(across), across, float(reach))  # unclaimed: far outside
        depth = half - np.abs(across)  # in from the nearest long edge; negative outside
        # The outward normal of the edge each pixel is nearest, and whether it faces the lamp.
        side = np.sign(across)
        ex, ey = side * ty, -side * tx
        lx, ly = material.lamp_2d()
        lit = (ex * lx + ey * ly) > 0.0
        # What is under the bar: a soft shadow thrown away from the lamp, and a hard line of
        # contact shadow all the way round it - deepest on the far edge, faint on the lit one,
        # wrapping the square ends - which is what says the bar is touching the plate and not
        # floating a pixel above it.
        shadow = material.cast(cover, lift) * (RAIL_SHADOW / 255.0)
        outside = np.hypot(np.maximum(-depth, 0.0), beyond)
        hug = RAIL_CONTACT * np.clip(1.0 - outside / contact, 0.0, 1.0)
        hug = hug * np.where(lit, RAIL_CONTACT_LIT, 1.0) * (1.0 - cover)
        under = 1.0 - (1.0 - shadow) * (1.0 - hug)
        layer.alpha_composite(material.to_image(np.zeros((*cover.shape, 3), np.float32), under),
                              (x0, y0))
        # The section. Only the lit chamfer is shaded as a tilt, because that is the one the lamp
        # can see; the far one is coloured below, from the flat, and the flat itself is all but
        # level.
        edge = min(RAIL_EDGE * self.scale, half * 0.2)
        chamfer = min(RAIL_CHAMFER * self.scale, half * 0.35)
        crown = RAIL_CROWN * np.clip(np.abs(across) / max(half - chamfer, 1e-3), 0.0, 1.0)
        tilt = np.where((depth < edge) & lit, RAIL_CHAMFER_TILT, crown)
        diffuse, spec = material.shade(ex * tilt, ey * tilt, np.sqrt(1.0 - tilt * tilt))
        rub = material.wear(along)
        spec = spec * (1.0 + RAIL_WEAR * rub)
        rgb = material.steel(diffuse, spec, material.grain(across, along) * RAIL_GRAIN)
        # The far chamfer's return: the lit plate under the bar, seen in a face tilted towards
        # it. Brightest at the arris and gone by the flat, and always short of the lit edge.
        arris = np.clip(1.0 - depth / max(chamfer, 1e-3), 0.0, 1.0) * ((depth < chamfer) & ~lit)
        glint = (RAIL_RETURN * arris * arris * (1.0 + 0.5 * RAIL_WEAR * rub))[..., None]
        rgb = rgb + (np.asarray(material.STEEL_LIT, np.float32) - rgb) * glint
        # Hairlines: the sheet's, where they happen to cross, and a few of the bar's own, dragged
        # along its length. Seeded off where the bar is, so every bar is scratched differently
        # and the same way on every boot. Drawn twice over and boxed down, because a hairline
        # nearly parallel to a diagonal bar is a staircase at one pixel and a line at half.
        longest = max(zip(points, points[1:], strict=False),
                      key=lambda ab: math.hypot(ab[1][0] - ab[0][0], ab[1][1] - ab[0][1]))
        own = material.scratches(w * 2, h * 2, RAIL_SCRATCHES, unit(*longest[0], *longest[1]),
                                 seed=material.SEED + x0 * 13 + y0 * 7, spread=RAIL_SCRATCH_SPREAD,
                                 length=tuple(2 * n * self.scale for n in RAIL_SCRATCH_LEN))
        own = own.reshape(h, 2, w, 2).mean(axis=(1, 3)) * 2.0
        marks = (np.maximum(own, self._marks[y0:y1, x0:x1]) * RAIL_SCRATCH)[..., None]
        rgb = rgb * (1.0 - marks) + np.asarray(material.STEEL_SPEC, np.float32) * marks
        # A cut end is where a bar rusts first.
        total = sum(math.hypot(b[0] - a[0], b[1] - a[1])
                    for a, b in zip(points, points[1:], strict=False))
        to_end = np.minimum(along, total - along) / max(RAIL_END_W * self.scale, 1e-3)
        rgb = rgb * (1.0 - RAIL_END_WEAR * np.clip(1.0 - to_end, 0.0, 1.0))[..., None]
        layer.alpha_composite(material.to_image(rgb, cover), (x0, y0))

    def _draw_bolt(
        self, d: ImageDraw.ImageDraw, x: float, y: float, r: float | None = None
    ) -> None:
        """A socket screw countersunk into the rail, under the same lamp as everything else.

        The one detail that says a bracket is bolted on rather than drawn on. It is a sprite
        from :mod:`cyclops.material` - a flat collar of the rail's own grey with a rolled rim,
        a countersunk dish, a head at the bottom of it with a hex socket the lamp reaches down
        into, and the seat shadow and spanner's ring round it - composited rather than drawn,
        because the shadow and the ring are translucent and a translucent *write* is a hole
        through the rail onto the camera, which is what the old seat shadow was. It was a
        crowned cap screw for a while, and a crowned head on a bar reads as a bead on a tube.

        Takes the draw and not the layer so that every caller keeps its one line. The layer is
        recovered from the draw, which Pillow has exposed since 9.x; the panel runs 12.
        """
        r = self.bolt_r if r is None else r
        layer: Image.Image = d._image
        cx, cy = math.floor(x), math.floor(y)
        tile = material.screw(round(r, 2), round(x - cx, 2), round(y - cy, 2))
        half = tile.width // 2
        # A head at the panel's edge would put the tile's corner off it, which alpha_composite
        # refuses; nothing here does, but a window size that did should lose a corner of shadow
        # rather than the panel.
        left, top = cx - half, cy - half
        crop = tile.crop((max(0, -left), max(0, -top), tile.width, tile.height))
        layer.alpha_composite(crop, (max(0, left), max(0, top)))

    def _draw_bracket(self, layer: Image.Image, bracket: Bracket) -> None:
        """Stiffeners, then the rail, then the bolts at every place the rail turns.

        The ribs go down first and stay near the corner. Run them out towards the rail and they
        stop reading as webbing inside a bracket and start reading as stripes laid over the room,
        which is the one thing this layout is spending its corners to avoid. The pod has none:
        webbing braces a corner against a load, and a module hanging off the middle of an edge
        has no corner and nothing to brace.

        Nor does the one with a face on it. The loom leaves his housing into that same corner and
        there is about forty pixels of it, so webbing and cables in there together is not two
        details, it is a thicket - and of the two, the thing that says where the power goes wins
        over the thing that says the corner is stiff. `seats` is what asks the question, because
        the left mount is the only bracket that has one.
        """
        corner, a, b = bracket.corner, bracket.spine[0], bracket.spine[-1]
        if corner and not bracket.seats:
            self._draw_ribs(layer, corner, a, b)
        self._draw_rail(layer, bracket.path())
        d = ImageDraw.Draw(layer)
        # Every place the rail turns: two knees on a mount, four on the pod.
        for knee in bracket.spine[1:-1]:
            self._draw_bolt(d, *knee)
        # ...and where the rail leaves the straight to go round a face, which is the one join on
        # this panel that is carrying anything.
        for seat in bracket.seats:
            p0, p1, _, _ = bracket.shoulder(seat)
            self._draw_bolt(d, *p0, self.bolt_r - 1)
            self._draw_bolt(d, *p1, self.bolt_r - 1)

    def _draw_ribs(self, layer: Image.Image, corner: tuple[int, int], a: tuple[int, int],
                   b: tuple[int, int]) -> None:
        """RIB_N raised stiffeners across a corner, lit like the rail and shadowed like it.

        Each is a bar of the plate's own steel a few pixels wide: a shadow off its far side, a
        translucent body, a lit hairline on the edge that faces the lamp and a dark one on the
        edge that does not. All of it on a private layer and composited, because a rib written
        translucent used to be three stripes of camera showing through the plate rather than
        three pieces of metal standing on it.
        """
        w = max(1, round(RIB_W * self.scale))
        lx, ly = material.lamp_2d()
        # The rib's own normal, turned to face the lamp: that is the edge the light lands on.
        rx, ry = unit(*a, *b)
        nx, ny = (ry, -rx) if ry * lx - rx * ly > 0 else (-ry, rx)
        edge = w / 2.0 - 0.5
        drop = RAIL_LIFT * self.scale * material.SHADOW_DROP
        shade = Image.new("RGBA", layer.size, (0, 0, 0, 0))
        ribs = Image.new("RGBA", layer.size, (0, 0, 0, 0))
        sd, rd = ImageDraw.Draw(shade), ImageDraw.Draw(ribs)
        body = (*material.STEEL, round(255 * RIB_ALPHA))
        for i in range(RIB_N):
            t = 0.15 + 0.075 * i
            p = (corner[0] + (a[0] - corner[0]) * t, corner[1] + (a[1] - corner[1]) * t)
            q = (corner[0] + (b[0] - corner[0]) * t, corner[1] + (b[1] - corner[1]) * t)
            sd.line([(p[0] - lx * drop, p[1] - ly * drop), (q[0] - lx * drop, q[1] - ly * drop)],
                    fill=(0, 0, 0, RAIL_SHADOW), width=w + 1)
            rd.line([p, q], fill=body, width=w)
            rd.line([(p[0] + nx * edge, p[1] + ny * edge), (q[0] + nx * edge, q[1] + ny * edge)],
                    fill=(*material.STEEL_LIT, 150), width=1)
            rd.line([(p[0] - nx * edge, p[1] - ny * edge), (q[0] - nx * edge, q[1] - ny * edge)],
                    fill=(*material.STEEL_DARK, 180), width=1)
        soft = max(0.5, RAIL_LIFT * self.scale * material.SHADOW_SOFT)
        layer.alpha_composite(shade.filter(ImageFilter.GaussianBlur(soft)))
        layer.alpha_composite(ribs)

    def _draw_collar(self, layer: Image.Image) -> None:
        """The bezel he is seated in: a ring of worn brass, and the steel lip that holds his glass.

        A solid disc on its own is not an object, it is a hole - it has a silhouette and no
        thickness. What makes him a thing sunk into the panel is the band round the outside of
        him, and it is built the way every bar on the panel now is: fields off one distance, the
        normals of a rolled edge, and the lamp deciding which side is lit. The housing and the
        mount that straps over it are lit by the same lamp, which is what keeps the join reading
        as one assembly even though they are two metals.

        Three passes, in the order a real one is looked at. The shadow the ring drops onto the
        plate, first and underneath. Then the brass: a rolled outer edge that catches the lamp
        up-left and goes dark down-right, a face crowned just enough to be lighter on the lamp's
        side, brushed round the way a turned part is, its highlight and its colour drifting round
        the ring where it has been handled, and the sheet's own scratches running across it. Then
        the lip: a half-round of polished steel on the inner edge, which is the one bright line
        that says there is glass under it. The well inside the lip is :meth:`_build_plate`'s.

        Nothing here is written with an alpha below full - the shadow and every soft edge are
        composited from fields, or they would be windows onto the camera.
        """
        cx, cy = self.eye
        out = float(self.shoulder)
        inn = out * COLLAR_IN
        lip = max(1.5, COLLAR_LIP * self.scale)
        roll = max(1.0, COLLAR_ROLL * self.scale)
        step = max(0.5, COLLAR_STEP * self.scale)
        lift = max(1.0, COLLAR_LIFT * self.scale)
        reach = math.ceil(lift * (material.SHADOW_DROP + 3 * material.SHADOW_SOFT)) + 2
        x0, y0, x1, y1, xs, ys = self._around(out + reach)
        dist = np.hypot(xs, ys)
        safe = np.maximum(dist, 1e-6)
        ux, uy = xs / safe, ys / safe
        along = self._round(xs, ys)
        # The whole ring's shadow before any of the ring, so the ring covers its own.
        cover = np.clip(0.5 - np.maximum(inn - dist, dist - out), 0.0, 1.0)
        shadow = material.cast(cover, lift) * COLLAR_SHADOW
        layer.alpha_composite(material.to_image(np.zeros((*cover.shape, 3), np.float32), shadow),
                              (x0, y0))
        # The brass, from the lip's outer edge to the swell. Its face tilts outwards, from level
        # at the lip to DOME at the roll, so the ring reads as crowned rather than as a washer;
        # its inner edge turns the other way, a narrow roll down onto the lip.
        brass_in = inn + lip
        brass = np.clip(0.5 - np.maximum(brass_in - dist, dist - out), 0.0, 1.0)
        crown = COLLAR_CROWN * np.clip((dist - brass_in) / max(out - roll - brass_in, 1e-3),
                                       0.0, 1.0)
        nx, ny, nz = material.roll_normals(np.maximum(out - dist, 0.0), ux, uy, roll, dome=crown)
        sx, sy, sz = material.roll_normals(np.maximum(dist - brass_in, 0.0), -ux, -uy, step,
                                           dome=0.0)
        inner = dist - brass_in < step
        nx, ny, nz = np.where(inner, sx, nx), np.where(inner, sy, ny), np.where(inner, sz, nz)
        diffuse, spec = material.shade(nx, ny, nz)
        rubbed = material.wear(along)
        spec = spec * (1.0 + COLLAR_WEAR * rubbed)
        brushed = material.grain(dist, along) + COLLAR_TARNISH * rubbed
        # The metal's own colour under the lamp, and then its highlight in its own colour too:
        # `steel` lays its highlight down in STEEL_SPEC, which on brass is a cool smear.
        rgb = material.steel(diffuse, np.zeros_like(spec), brushed, colour=BRASS)
        rgb = rgb + np.asarray(BRASS_SPEC, np.float32) * (material.SPEC * spec)[..., None]
        groove = np.exp(-(((dist - (out - roll - 1.0)) / 0.6) ** 2))
        rgb = rgb * (1.0 - COLLAR_GROOVE * groove)[..., None]
        marks = (self._marks[y0:y1, x0:x1] * COLLAR_SCRATCH)[..., None]
        rgb = rgb * (1.0 - marks) + np.asarray(BRASS_SPEC, np.float32) * marks
        rgb = np.minimum(rgb, np.asarray(material.STEEL_SPEC, np.float32))
        layer.alpha_composite(material.to_image(rgb, brass), (x0, y0))
        # The lip: a bevel of steel turned to face outwards all the way across, so the whole of
        # it is lit on the lamp's side and dark on the other, one bright line round the glass.
        wire = np.clip(0.5 - np.maximum(inn - dist, dist - brass_in), 0.0, 1.0)
        diffuse, spec = material.shade(
            *material.roll_normals(np.maximum(dist - inn, 0.0), ux, uy, lip, dome=0.0))
        rgb = material.steel(diffuse, spec, material.grain(dist, along) * 0.5)
        layer.alpha_composite(material.to_image(rgb, wire), (x0, y0))
        # ...and the mount points, in the arc the rail never reaches. The bracket bolts its own
        # two where the rail leaves the straight; without these the far side of the collar is a
        # ring resting against him rather than a housing fastened down all the way round.
        d = ImageDraw.Draw(layer)
        mid = (brass_in + out) / 2.0
        for deg in COLLAR_BOLTS:
            a = math.radians(deg)
            self._draw_bolt(d, cx + mid * math.cos(a), cy + mid * math.sin(a), self.bolt_r - 1)

    def _draw_loom(self, layer: Image.Image) -> None:
        """The cables leaving his housing for the corner, and off the panel.

        A camera on a bracket has something coming out of the back of it, and until now he was the
        one piece of equipment here that was fed by nothing. The run is short by necessity - his
        swell is three pixels off both edges down there, so the pocket between him and the corner
        is all there is - and short is fine, because what the cables have to do is leave. They are
        drawn past the panel's edge on purpose: a loom that stops inside the frame is a loom with
        an end, and an end wants a connector on it.

        Each run is a round bar of braided steel under the panel's lamp - the same fields the
        rails are built from, on a section that is all roll, ribbed across rather than brushed
        along - with its shadow falling off it the other way from the light. The gland goes on
        last and is what every one of them starts under - none of them has a beginning, the same
        way the terminal's rail has no ends.
        """
        cx, cy = self.eye
        aim = math.radians(GLAND_AT)
        w = max(3, round(9 * self.scale))
        half = w / 2.0
        lift = max(1.0, LOOM_LIFT * self.scale)
        conduit = mix(material.STEEL, material.STEEL_DARK, LOOM_STEEL)
        for i in range(LOOM_N):
            a = aim + math.radians(LOOM_FAN) * (i - (LOOM_N - 1) / 2.0)
            along, across = (math.cos(a), math.sin(a)), (-math.sin(a), math.cos(a))
            # From inside the collar, so the gland has something to cover rather than to butt
            # against, and with a little sag - three cables leaving on exactly parallel lines is
            # a ribbon, and this is a bundle.
            root = (cx + (self.shoulder - w) * along[0], cy + (self.shoulder - w) * along[1])
            sag = (LOOM_N - 1 - i) * 2.5 * self.scale
            reach = LOOM_REACH * self.scale
            pts = [
                (root[0] + along[0] * reach * t + across[0] * sag * math.sin(math.pi * t),
                 root[1] + along[1] * reach * t + across[1] * sag * math.sin(math.pi * t))
                for t in (k / 12.0 for k in range(13))
            ]
            self._draw_conduit(layer, pts, half, lift, conduit)
        g = max(8, round(17 * self.scale))
        seat = (cx + (self.shoulder - g * 0.35) * math.cos(aim),
                cy + (self.shoulder - g * 0.35) * math.sin(aim))
        self._draw_gland(layer, seat, aim, g)

    def _draw_conduit(self, layer: Image.Image, pts: list[tuple[float, float]], half: float,
                      lift: float, colour: tuple[int, int, int]) -> None:
        """One run of the loom: a round section, its shadow first, over the box it occupies."""
        margin = math.ceil(half + lift * (material.SHADOW_DROP + 3 * material.SHADOW_SOFT)) + 1
        x0 = max(0, math.floor(min(p[0] for p in pts)) - margin)
        y0 = max(0, math.floor(min(p[1] for p in pts)) - margin)
        x1 = min(self.width, math.ceil(max(p[0] for p in pts)) + margin + 1)
        y1 = min(self.height, math.ceil(max(p[1] for p in pts)) + margin + 1)
        if x1 <= x0 or y1 <= y0:
            return
        dist, ox, oy, along, side = material.bar_field(pts, x0, y0, x1 - x0, y1 - y0)
        cover = np.clip(half + 0.5 - dist, 0.0, 1.0)
        shadow = material.cast(cover, lift) * (RAIL_SHADOW / 255.0)
        layer.alpha_composite(material.to_image(np.zeros((*cover.shape, 3), np.float32), shadow),
                              (x0, y0))
        # All roll and no flat: a conduit is round. The braid's ribs run round it, so the grain's
        # fibres vary along the run and hold across it - the rail's two arguments swapped.
        normals = material.roll_normals(np.maximum(half - dist, 0.0), ox, oy, half, dome=0.0)
        ribs = material.grain(along, dist * side) * LOOM_RIB
        rgb = material.steel(*material.shade(*normals), ribs, colour=colour)
        layer.alpha_composite(material.to_image(rgb, cover), (x0, y0))

    def _draw_gland(self, layer: Image.Image, seat: tuple[float, float], aim: float,
                    size: float) -> None:
        """The fitting the loom leaves through: a rolled-edged block of steel seated on the collar.

        Slightly wider where it meets the collar than where the runs leave it, so it reads as a
        fitting screwed into the housing rather than as a block sitting on top of one. Built off
        its own distance field - a rounded box in the frame of the run, tapered along it - so
        the same roll, lamp and shadow as every other part fall on it.
        """
        along, across = (math.cos(aim), math.sin(aim)), (-math.sin(aim), math.cos(aim))
        roll = max(1.0, GLAND_ROLL * self.scale)
        lift = max(1.0, LOOM_LIFT * self.scale)
        long, wide_in, wide_out = size * 0.7, size * 0.8, size * 0.62
        reach = math.hypot(long, wide_in) + lift * (material.SHADOW_DROP + 3 * material.SHADOW_SOFT)
        x0, y0 = max(0, math.floor(seat[0] - reach)), max(0, math.floor(seat[1] - reach))
        x1 = min(self.width, math.ceil(seat[0] + reach) + 1)
        y1 = min(self.height, math.ceil(seat[1] + reach) + 1)
        ys = (np.arange(y0, y1, dtype=np.float32) - seat[1])[:, None]
        xs = (np.arange(x0, x1, dtype=np.float32) - seat[0])[None, :]
        u = xs * along[0] + ys * along[1]  # along the run: negative towards the collar
        v = xs * across[0] + ys * across[1]
        # The taper: the box's half-width narrows from the collar end to the far end.
        width = wide_in + (wide_out - wide_in) * np.clip((u + long) / (2 * long), 0.0, 1.0)
        qx, qy = np.abs(u) - long + roll, np.abs(v) - width + roll
        outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
        sdf = outside + np.minimum(np.maximum(qx, qy), 0.0) - roll
        cover = np.clip(0.5 - sdf, 0.0, 1.0)
        shadow = material.cast(cover, lift) * (RAIL_SHADOW / 255.0)
        layer.alpha_composite(material.to_image(np.zeros((*cover.shape, 3), np.float32), shadow),
                              (x0, y0))
        gy, gx = np.gradient(sdf)
        glen = np.maximum(np.hypot(gx, gy), 1e-6)
        normals = material.roll_normals(np.maximum(-sdf, 0.0), gx / glen, gy / glen, roll)
        rgb = material.steel(*material.shade(*normals), material.grain(v, u))
        layer.alpha_composite(material.to_image(rgb, cover), (x0, y0))

    def _draw_reticle(self, layer: Image.Image) -> None:
        """Four right-angled corners on the lens axis, and nothing in the middle of them.

        What a camera shows you when it is looking rather than aiming. This was a gapped cross
        with graduations for exactly as long as it took somebody to look at it and say it read as
        a gun sight; the corners say frame and say nothing else, and they leave the centre of the
        picture - which is the part anybody actually points the thing at - completely clear.

        Each corner is one three-point stroke rather than two, so the bend is a joint PIL closes
        for us instead of a notch two separate legs leave at the outside of the turn. Still
        supersampled, for the ends rather than the lines: a mark this short is mostly its ends.
        """
        half = max(8, round(RETICLE_R * self.height))
        leg = max(3, round(half * RETICLE_LEG))
        stroke = max(1.0, self.line * RETICLE_W)
        cx, cy = self.width // 2, self.height // 2
        span = half + round(stroke) * 2

        def paint(t: ImageDraw.ImageDraw) -> None:
            for sx in (-1, 1):
                for sy in (-1, 1):
                    x, y = span + sx * half, span + sy * half
                    t.line(
                        [(at(x - sx * leg), at(y)), (at(x), at(y)), (at(x), at(y - sy * leg))],
                        fill=linear(GREEN_MID, 195), width=round(wide(stroke)), joint="curve",
                    )

        layer.alpha_composite(smoothed(2 * span + 1, paint), (cx - span, cy - span))


    def _base(self, state: str, recording: bool, heat: str = "") -> Image.Image:
        """Everything that holds still while the state does, built once and copied per frame.

        The state light goes on last, after the words, and falls inwards from the border. It
        was tried the other way - a margin of dark bezel with the glow spreading outwards into
        it - and it read as light leaking off the edge of the device rather than as a screen lit
        from inside. A tube blooms in front of what it is showing, so this one does too.

        Keyed on the state rather than on the halo colour, because the words change with it too.
        Baking the pod's tags and both switches in here is what keeps a frame down to
        a meter, a clock, a caption and a ring: drawing all of it every time cost 10 ms of the
        Pi's 40 ms budget, against 2.8 ms for the chrome the tab row replaced.
        """
        cached = self._bases.get((state, recording, heat))
        if cached is not None:
            return cached
        halo = HALOS.get(state, GREEN_DIM)
        # The filter, and then the chrome drawn on it. There is nothing opaque underneath either
        # of them any more: the strip, the tab row and the four scraps outside the border's
        # rounded corners are all just the wash and the scanlines over the live picture, and the
        # picture between them is not covered at all - not even by the border's own corners.
        tags = self._tag_count(state, recording, heat)
        image = Image.alpha_composite(self._backdrop(tags), self._plate)
        image = Image.alpha_composite(image, self._chrome(tags))

        d = ImageDraw.Draw(image)
        self._bake_header(d, state, recording, heat)

        # The state light, falling inwards from the border over everything drawn so far - a tube
        # blooms in front of what it is showing, not behind it. It reaches about 13 px at 480,
        # and PAD keeps every word further in than that, so nothing legible sits in it.
        rim = Image.new("RGBA", image.size, (*halo, 0))
        rim.putalpha(Image.fromarray((self._halo * 255.0).astype(np.uint8), "L"))
        image = Image.alpha_composite(image, rim)

        # ...and the border itself, crisp on top of its own glow and in the state's colour.
        ImageDraw.Draw(image).rounded_rectangle(
            [self.frame.x, self.frame.y, self.frame.right - 1, self.frame.bottom - 1],
            radius=self.radius,
            outline=(*halo, 255),
            width=self.line,
        )
        self._bases[(state, recording, heat)] = image
        return image

    # ---- where the readouts sit ----

    def _tag_count(self, state: str, recording: bool, heat: str) -> int:
        """How many tags the pod is showing, which is the whole of what sets its width."""
        return int(self._taping(state, recording)) + int(heat in HEAT_LAMP)

    @staticmethod
    def _taping(state: str, recording: bool) -> bool:
        """Is a recording actually being made? Configured to record is not the same thing.

        ``recording`` only says the setting is on. The tag has to mean "tape is running", or a
        panel sitting at ASLEEP claims to be filming the room.
        """
        return recording and session_up(state)

    def _readouts(self, tags: int) -> tuple[float, float, float]:
        """The clock's right edge, the tags' left edge, and the meter's right edge.

        One row, packed: the meter, then whichever tags are lit, then the clock, with a single
        stop between each. Nothing is reserved and nothing is padded - the pod itself is as wide
        as this comes to, so closing the gaps here is what makes the whole module narrower rather
        than only moving its contents about inside it. It was the other way round for a while and
        the reserved slot was the biggest thing on the pod, lit or not.

        What that costs is that the clock's digits sit a tag's width further right while the tape
        is running than while it is not. They do not *drift*: recording is settled for the whole
        of a session, so the position changes when a session starts and when a board goes hot,
        and both of those are moments the panel is entitled to look different.
        """
        box = self.pod_boxes[tags]
        return box.right, box.x + self._meter_w + self._stop, box.x + self._meter_w

    def _meter_x(self, right: float) -> float:
        seg_w, _, gap = self._seg
        return right - (METER_SEGMENTS * seg_w + (METER_SEGMENTS - 1) * gap)

    # ---- text ----

    def _width(self, text: str, font: ImageFont.FreeTypeFont, tracking: float) -> float:
        return font.getlength(text) + tracking * max(0, len(text) - 1)

    def _text(
        self,
        d: ImageDraw.ImageDraw,
        x: float,
        y: float,
        text: str,
        font: ImageFont.FreeTypeFont,
        fill: tuple,
        *,
        align: str = "l",
        tracking: float = 0.0,
    ) -> float:
        """Draw *text* centred vertically on *y*, letter-spaced by *tracking*. Returns its width.

        Tracking is what separates a terminal readout from a caption, and PIL has no such
        thing, so spaced text is stepped through a character at a time.
        """
        width = self._width(text, font, tracking)
        start = x if align == "l" else x - width if align == "r" else x - width / 2
        if not tracking:
            d.text((start, y), text, font=font, fill=fill, anchor="lm")
            return width
        for char in text:
            d.text((start, y), char, font=font, fill=fill, anchor="lm")
            start += font.getlength(char) + tracking
        return width

    def _wrap(
        self, text: str, font: ImageFont.FreeTypeFont, limit: float, lines: int
    ) -> list[str]:
        """*text* broken over at most *lines* of *limit*, the last one trimmed if it still runs on.

        Greedy and word-wise, which is all a caption needs: these are one short sentence, and the
        alternative - balancing the lines - would have the first one change length every time the
        second did, which on a screen of a fixed size is movement bought for nothing.

        A word longer than the whole limit still goes down and is cut mid-word by :meth:`_elide`,
        rather than being dropped or running off the glass. Rare, and always a URL or a
        token out of an API error, which is exactly the case that must not lose the beginning of
        the message it is buried in.
        """
        words = text.split()
        out: list[str] = []
        line = ""
        for i, word in enumerate(words):
            trial = f"{line} {word}" if line else word
            if line and font.getlength(trial) > limit:
                if len(out) + 1 == lines:
                    # Nowhere left to break: the rest goes down on this line and is cut there.
                    return [*out, self._elide(" ".join([line, *words[i:]]), font, limit)]
                out.append(line)
                line = word
            else:
                line = trial
        return [*out, self._elide(line, font, limit)]

    def _elide(self, text: str, font: ImageFont.FreeTypeFont, limit: float) -> str:
        """Trim to fit, with an ellipsis - an API error can be a paragraph long."""
        if font.getlength(text) <= limit:
            return text
        while text and font.getlength(text + "…") > limit:
            text = text[:-1]
        return text.rstrip() + "…"

    # ---- the frame ----

    def render(
        self,
        state: str,
        level: float,
        elapsed: float | None = None,
        recording: bool = False,
        flash: float = 0.0,
        pressed: str | None = None,
        detail: str = "",
        phase: float = 0.0,
        heat: str = "",
        hold: float = 0.0,
        menu: bool = False,
        volume: int | None = None,
        temp_c: float | None = None,
        turning: bool = False,
    ) -> np.ndarray:
        """Draw the whole chrome for this frame and return it as an RGBA numpy array.

        ``phase`` is a monotonic clock in seconds, and the only argument here that is not about
        what the panel is showing but about *when*. It is passed in rather than read here so the
        animation can be tested without a clock - the same shape as ``flash``, which the kiosk
        has always computed.

        Two things here are drawn from more than the arguments, and both remember the same pair:
        the last thing they were asked to show, and the phase it first turned up at. The eye
        eases out of the mood it was in rather than snapping to the new one
        (:meth:`cyclops.eye.EyeEngine.look`), and the caption types itself onto the terminal
        rather than arriving whole (:class:`Typist`). Neither can be told from outside, because
        a caller hands over what the panel is showing and never what it was showing a moment
        ago. Neither costs the purity worth having either, which is that a frame is a function
        of its arguments and of the order they arrived in and of nothing else: render twice and
        the second one is settled, which is what every test down here does.

        Almost nothing here moves while he is asleep, and that is deliberate and half the
        design: the eye stays shut, the border holds still, the caption stops breathing and the
        readouts have nothing to count. What is left is his breath and two instruments telling
        the truth about a box nobody is talking to. Against a panel that was quietly pulsing
        whatever it was doing, an awake one that pulses says nothing.

        ``volume`` is the level on the sink, 0 to 100, and ``temp_c`` the board's temperature -
        both ``None`` where the platform cannot say, which the dials draw as not reading rather
        than as reading nothing.

        ``turning`` is a finger on the knob, and it is what puts the column up - on the touch
        rather than on the first movement, because a control you cannot see until you have
        already started using it is one you start using blind. What the column *shows* is the
        caller's business: the level on the sink until the finger reaches the track, and the
        level under the finger after that.

        ``hold`` is how far a finger is through the long press on his face, 0 to 1, and ``menu``
        is whether that press has landed - the power menu, over everything else. The two are the
        one gesture on this panel that is not a tap, so they are the one thing here drawn from a
        clock the kiosk is holding rather than from the state.
        """
        halo = HALOS.get(state, GREEN_DIM)
        layer = self._base(state, recording, heat).copy()
        d = ImageDraw.Draw(layer)

        self._draw_readouts(d, halo, level, elapsed, self._tag_count(state, recording, heat))
        self._draw_caption(layer, state, halo, detail, phase)
        held = pressed == "eye"
        # The pointers, over the faces the chrome laid down once. Neither inverts under a thumb
        # the way the switches here used to: you do not press an instrument, you turn one and
        # read the other, so the knob answers a finger by going white under it and the gauge
        # answers the tap that opens its screen the same way.
        self._draw_hands(layer, d, volume, temp_c, pressed)
        if turning and volume is not None:
            self._draw_slider(d, volume)
        # Him, last of everything in his corner. He is the one control that never inverts under
        # a thumb: a face in photographic negative is not the same face, and half of him is over
        # the picture anyway, where there is nothing to invert. He acknowledges a tap by coming
        # up to full instead - which also holds for as long as the page behind him is loading,
        # so a slow browser looks like a box that heard you rather than one that ignored you.
        mood = self.engine.look(state, MOODS.get(state, MOODS[IDLE]), phase)
        if held:
            # Wide, bright, steady - and looking straight at you, which is what `gaze` and
            # `dart` at zero are for: an acknowledgement that carried on glancing round the
            # room would not read as one. Startled open is the right shape for it,
            # and it is the one gesture that reads the same from every mood - including asleep,
            # where you have just tapped a shut eye and it has opened to look at you.
            # Past 1.0 on purpose: `rings` is how present he is and his resting level *is* 1.0,
            # so coming "up to full" is a number above it. The machine is drawn well short of
            # full at rest (eye.RIM_LIT and friends), and this is the one gesture that saturates
            # it - which is what makes the acknowledgement read as brightening rather than as a
            # colour change he might have made on his own.
            mood = replace(mood, tint=GREEN, rings=1.7, aperture=1.0, swell=0.0,
                           voice=0.0, gaze=0.0, dart=0.0, drift=0.0)
        self.engine.paint(layer, *self.eye, mood, phase, level)
        layer.alpha_composite(self._glass, (self.eye[0] - self.eye_r, self.eye[1] - self.eye_r))
        if hold > 0.0:
            self._draw_hold(layer, hold)
        if session_up(state):
            # The teardown breathes too. He is not listening any more - the eye is already shut -
            # but the box is still working, and a panel that went stone still the moment you
            # pressed stop would look like it had stopped rather than like it was finishing.
            self._draw_rim(d, halo, phase)
        if menu:
            # Last of everything, because it is the only thing here that is asked a question
            # rather than told one: nothing behind it is live while it is up.
            self._draw_menu(layer, d, pressed)
        if flash > 0.0:
            # Green-white rather than white: a photo taken through a phosphor screen.
            d.rectangle([0, 0, self.width, self.height], fill=(214, 255, 228, int(190 * flash)))
        return np.asarray(layer)

    def _bake_header(
        self, d: ImageDraw.ImageDraw, state: str, recording: bool, heat: str = ""
    ) -> None:
        """The half of the pod that only moves when the state does: the two tags.

        One row: the meter against the pod's left edge, the clock against its right, and the two
        tags in a fixed slot between them. Every one of the three is pinned to something that
        does not move, and the slack sits in the middle where the tags are not - which is the
        same bargain the caption strikes with its cursor. Packed instead, the clock would slide
        sideways the moment the board got warm, and a panel whose numbers move while it is
        telling you the truth feels like one that is lying.

        No state word. It had a corner of its own for a long time and three other things were
        already saying it better - the border's colour, the eye's mood, and the line under the
        picture, which can say "searching the web…" where a word could only say SEARCH.

        All of it is letter-spaced, which PIL can only do a character at a time, which is
        precisely why it is baked rather than redrawn 25 times a second.
        """
        taping = self._taping(state, recording)
        _, tags, _ = self._readouts(self._tag_count(state, recording, heat))
        if taping:
            # Red, and a filled tag rather than a dot. Red is what a record light is on every
            # other machine anybody has ever used, which is worth more here than the panel's
            # preference for its own green - and filling it rather than outlining it is how this
            # tube shouts. The one thing on screen that is red without being a fault, which is
            # exactly why it is a tag with a word in it and not a lamp.
            tags += self._tag(d, tags, self.row, "REC", RED) + self._tag_gap
        colour = HEAT_LAMP.get(heat)
        if colour is not None:
            self._tag(d, tags, self.row, HEAT_WORD, colour)

    def _tag(
        self, d: ImageDraw.ImageDraw, x: float, cy: float, word: str, colour: tuple[int, int, int]
    ) -> float:
        """A filled rounded slab with a word knocked out of it, left edge at *x*. Returns width.

        The panel's way of shouting, and there are two things that do it: REC and the heat lamp.
        They were one shape typed out twice for exactly as long as it took to add the second, so
        they are one method now - which also means :attr:`_tag_w`, the width the pod is laid out
        against, is measured the same way the thing is drawn.
        """
        half = round(11 * self.scale)
        # Both tags are drawn to one width and the word centred in it, so REC and HOT side by
        # side are two slabs of the same size rather than two that nearly are.
        width = self._tag_w
        d.rounded_rectangle(
            [x, cy - half, x + width, cy + half],
            radius=max(1, round(3 * self.scale)),
            fill=(*colour, 255),
        )
        self._text(d, x + (width - self.font_micro.getlength(word)) / 2, cy, word,
                   self.font_micro, (*INK, 255))
        return width

    def _draw_readouts(
        self,
        d: ImageDraw.ImageDraw,
        halo: tuple[int, int, int],
        level: float,
        elapsed: float | None,
        tags: int,
    ) -> None:
        """The half that moves: the signal bar, and the clock counting the session up."""
        clock_right, _, meter_right = self._readouts(tags)
        whole = 0 if elapsed is None else int(elapsed)
        clock = "--:--" if elapsed is None else f"{whole // 60:02d}:{whole % 60:02d}"
        # Dim green with nothing to count, the state's accent the moment there is - the numbers
        # that only mean something during a session are the right place for the colour that only
        # appears during one.
        colour = (*GREEN_DIM, 255) if elapsed is None else (*halo, 255)
        self._text(d, clock_right, self.row, clock, self.font_read, colour, align="r")

        cy = self.row
        seg_w, seg_h, gap = self._seg
        lit = round(max(0.0, min(1.0, level)) * METER_SEGMENTS)
        x = self._meter_x(meter_right)
        for index in range(METER_SEGMENTS):
            x0 = x + index * (seg_w + gap)
            d.rectangle(
                [x0, cy - seg_h / 2, x0 + seg_w, cy + seg_h / 2],
                fill=(*halo, 255) if index < lit else (*mix(SCREEN, GREEN_DIM, METER_OFF), 255),
            )

    def _draw_caption(
        self, layer: Image.Image, state: str, halo: tuple, detail: str, phase: float
    ) -> None:
        """Plain English across the bottom of the panel, printed on the terminal's screen.

        The screen is not decoration: this text sits over the live camera, and white-on-anything
        is a coin toss. It is also where an error actually says what went wrong, which the old
        chrome could only render as a red rim. What it is printed *on* is a separate argument -
        see :meth:`_draw_terminal`.

        Printed from the top, like anything else with a prompt on it: a one-line message leaves
        the second line blank, which is what a terminal with nothing more to say looks like. The
        bubble grew upwards instead, and had to, because its bottom edge was where the tail hung
        from - the last piece of that shape still visible in this method.

        The controller's sentence wins over this module's own table, and not the other way round
        as it used to: it is the half that knows what is being searched for, which project is
        being opened and how far the teardown has got, and CAPTIONS is what is left to say when
        it knows nothing finer. Reversed, every one of those sentences would be swallowed by a
        state word during exactly the states worth narrating.
        """
        text = detail or CAPTIONS.get(state, "")
        if not text:
            # Cleared, and the typist has to hear about it: whatever turns up next is new text
            # appearing on an empty screen, even if it is the same sentence as before.
            self._typist.printed("", phase)
            return  # the strip already says the mode; saying it twice is not a caption
        busy = text.endswith(BUSY_MARK)
        if busy:
            text = text[: -len(BUSY_MARK)]  # the cursor takes the ellipsis's place, and blinks
        font = self.font_caption
        # The marker rides along on the front of the sentence so the wrap can put it where it
        # belongs rather than the drawing assuming a first line.
        x = self.caption_left
        # The cursor's room is reserved whether it is showing or not, which is what stops a line
        # that fills the screen at rest putting its cursor under the bracket half the time.
        cursor_w = self._cursor_w if busy else 0.0
        limit = self.caption_right - x - cursor_w
        lines = self._wrap(MARKER + text, font, limit, CAPTION_LINES)
        if busy and lines[-1].endswith(BUSY_MARK):
            # Cut short *and* about work in flight, which used to come out as "an M8 s…_": the
            # ellipsis the trim leaves behind, and then a cursor that was already standing in for
            # one. Two marks doing one mark's job. The cursor wins, because it is the half that
            # moves, and it means what the ellipsis meant anyway.
            lines[-1] = lines[-1][: -len(BUSY_MARK)]
        # How much of it has landed. Keyed on the wrapped lines rather than on the sentence, so
        # the schedule is built from the exact string being drawn - and so two different
        # paragraphs of API error that elide to the same visible line do not retype.
        #
        # A fault arrives whole. It is asking to be read, not watched, which is the argument
        # that already denies it the breath and the blink below, and it is also the one caption
        # that can be a paragraph long - the worst thing on this panel to watch being wiped on.
        whole = "\n".join(lines)
        printed = len(whole) if state == ERROR else self._typist.printed(whole, phase)
        typing = printed < len(whole)
        top = self.caption_top
        # The breath runs under every caption of a session that is up - it is what makes the line
        # read as a live tube rather than a printed label - and the cursor after any line that
        # ends in an ellipsis, where it means the thing everybody already reads it to mean.
        #
        # Asleep the breath stops and the cursor does not. The breath stopping is the point rather
        # than an economy: it used to run unconditionally, so a panel with nothing on it was
        # quietly pulsing 809 pixels of caption, and against that an awake panel that pulses says
        # nothing at all - and brightness stopped being his register the day the sleeping face
        # gave it up. The cursor stays because what the line says while he is asleep is a snore,
        # and a snore that holds still is a printed label. A fault gets neither: it is asking to
        # be read, not watched.
        sunk, blink = caption_pulse(phase)
        # Working breathes and blinks along with a session, though it is not one: the line under a
        # busy box is a job in flight, which is exactly what this animation is for. It is the only
        # thing on the panel that says so while the border and the readouts sit at rest.
        sunk = sunk if session_up(state) or state == WORKING else 0.0
        lit = blink and (session_up(state) or state in (IDLE, WORKING))
        colour = mix(halo if state == ERROR else GREEN, SCREEN, sunk)
        # The marker takes the accent and the sentence does not. A whole line of running text in
        # white over a live camera is harder to read than the same line in phosphor, and the
        # marker is the part that is decoration anyway - so it is the part that gets to be a
        # colour, and it breathes with the words it introduces.
        accent = (*mix(halo, SCREEN, sunk), CAPTION_ALPHA)
        # Laid out once into a list of draws rather than straight onto the panel, because the same
        # layout is wanted twice: once blurred for the halation and once crisp on top of it.
        ops: list[tuple[float, float, str, tuple]] = []
        at, y, taken = x, top + 0.5 * self.caption_h, 0
        for i, line in enumerate(lines):
            head = printed - taken  # how much of this line has landed...
            taken += len(line) + 1  # ...and the return that ends it, which costs the hand a beat
            if head <= 0:
                break  # the head is not on this line, and so cannot be on any line under it
            y = top + (i + 0.5) * self.caption_h
            at = x
            if line.startswith(MARKER):  # only ever the first, and only if the wrap left it there
                ops.append((at, y, MARKER, accent))
                at += self._width(MARKER, font, 0.0)
                line, head = line[len(MARKER):], max(0, head - len(MARKER))
            ops.append((at, y, line[:head], (*colour, CAPTION_ALPHA)))
            at += self._width(line[:head], font, 0.0)
        # Hard against the last letter printed, where a cursor belongs - it is standing in for the
        # ellipsis the phrase arrived with, not sitting beside it as a separate mark. `at` and `y`
        # are wherever the loop left them: the head while the line is still arriving, and the end
        # of the last line once it has, which is where this has always been drawn.
        #
        # Solid while it types and blinking once it has stopped. A cursor that blinked mid-word
        # would be two things moving where only one of them is the hand.
        #
        # The guard is for the resting line. Only a busy one had the cursor's width taken out of
        # its wrap (see `limit`), so a full line of text nobody reserved room for has to give the
        # mark up at the very end rather than stand it on the bracket.
        if (typing or (busy and lit)) and at + self._cursor_w <= self.caption_right:
            ops.append((at, y, CURSOR, (*colour, CAPTION_ALPHA)))
        self._print(layer, ops, font)

    def _print(self, layer: Image.Image, ops: Sequence[tuple[float, float, str, tuple]],
               font: ImageFont.FreeTypeFont) -> None:
        """Put the line on the glass: once blurred for the halation, once crisp on top of it.

        Halation is what separates a lit tube from a printed label - a glyph on a phosphor screen
        is a spot of light with a skirt round it, and the skirt is most of why a photograph of a
        CRT does not look like text on paper. It is also the one part of the terminal that is not
        baked, because it is the one part that changes with the sentence.

        Drawn *once* into a tile and composited twice. The obvious way round - draw the glow into
        a scratch layer, then draw the line again onto the panel - costs two text passes, and on
        the Pi laying this many glyphs is 2 ms while blurring them is 0.9. Doing it this way put
        the whole effect back under a millisecond and a half.

        Compositing rather than drawing onto the chrome directly, which is not just an
        optimisation: ImageDraw *writes*, so a glyph laid straight onto the layer replaces the
        glass's alpha with its own instead of adding light to it - the rule that makes every
        brightness on this panel a mix towards SCREEN (see :func:`caption_pulse`). A composite is
        the operation that means "there is more light here now", which is what a lit phosphor is.
        """
        if not ops:
            return
        tube = self.tube
        tile = Image.new("RGBA", (tube.w, tube.h), (0, 0, 0, 0))
        td = ImageDraw.Draw(tile)
        for gx, gy, glyphs, fill in ops:
            self._text(td, gx - tube.x, gy - tube.y, glyphs, font, fill)
        glow = tile.filter(ImageFilter.GaussianBlur(self.bloom_r))
        glow.putalpha(glow.getchannel("A").point(lambda a: round(a * BLOOM_ALPHA)))
        layer.alpha_composite(glow, (tube.x, tube.y))
        layer.alpha_composite(tile, (tube.x, tube.y))

    def _draw_terminal(self, layer: Image.Image) -> None:
        """The monitor the caption is printed on: a tube in a bezel, and the rail that carries it.

        This replaced a speech bubble, and the argument is the same one the words themselves
        settled a while ago. A bubble is him talking: it is the right shape for a sentence and the
        wrong shape for a readout, it has to be sized to whatever it happens to be saying, and it
        has nowhere to be when there is nothing to say. What the line actually does - narrate a
        job, name a fault, count out a teardown - is what a terminal does, so it is one: a fixed
        screen in a housing bolted into the bottom middle of the panel, the one strip of it that
        had nothing in it. The sentence changes; the machine it is printed on does not.

        What is in the housing is a monitor and is drawn as one, because for a long time it was
        drawn as a hole - a rectangle with square corners, a flat fill and a hairline chamfer,
        which is a cut-out however carefully it is shaded. The corners are the whole of it: pull
        them in and put metal all the way round, and the same rectangle is a piece of glass in a
        moulding. Everything else here follows from having a tube to be honest about - it shades
        into its corners, the bezel drops a shadow down it, the room wipes across it, and the
        letters on it bloom (:meth:`_print`, the one part of this that is not baked).

        All the rest is fixed by the window size, so all the rest is baked. What that buys is the
        half of the idea that a per-frame drawing could not have: the chassis is on the panel
        whether or not there is a caption, which is what makes it furniture rather than a slab
        that appears under some words.

        Nothing here wears the state's accent. The border does that, and the eye, and the marker
        on the line itself - a housing that changed colour with the conversation would be a fourth
        voice saying what three already say.

        The letters that go on it are :meth:`_print`, which is the only part of this that runs
        at a frame.
        """
        box = self.term
        # The whole front in one composite: the case and the glass are the same field, so there is
        # no edge between them to line up and none to see. Built out here rather than drawn
        # because ImageDraw writes rather than composites - a translucent stroke laid over this
        # would punch a hole through the glass onto the camera instead of dimming it.
        shape = (box.h, box.w)
        face: tuple[np.ndarray, np.ndarray] = (
            np.zeros((*shape, 3), dtype=np.float32),
            np.zeros(shape, dtype=np.float32),
        )
        face = _over(face, SCREEN, tube_alpha(box.w, box.h, self.case_r, self.bezel, box.y))
        face = _over(face, GREEN, tube_glow(box.w, box.h, self.case_r, self.bezel))
        face = _over(face, WHITE, glare_alpha(box.w, box.h, self.case_r, self.bezel))
        face = _over(face, WHITE, sheen_alpha(box.w, box.h, self.case_r))
        layer.alpha_composite(_to_image(*face), (box.x, box.y))
        # The two mounting brackets, one either side, and they are *rails* - the same extrusion
        # the mounts are made of, drawn by the same method, just a thinner member. That is the
        # whole of why they now look like part of this machine: a bracket built out of anything
        # else is a shape drawn next to a frame, however carefully it is shaded, and the panel
        # already has one language for "a piece of metal bolted to something".
        #
        # Each is a clamp: a strap standing across the screen's end, and an arm back to the mount.
        # Drawn after the front and lapping over it, so the screen sits *in* them.
        # The strap stops at its two bolt centres rather than running the full height of the
        # bracket. PIL gives a line butt caps, so a member drawn past its bolt leaves two square
        # corners sticking out from under the head - which is the one thing on this panel that
        # would say "drawn" rather than "made". Ending on the centre puts the whole butt inside
        # the head, and see `ear_bolt` for why the head is always big enough to hold it.
        r = self.ear_bolt
        for ear in self.ears:
            strap = self.term.x if ear.x < self.term.x else self.term.right
            self._draw_rail(layer, [(strap, ear.y + r), (strap, ear.bottom - r)], self.ear_w)
            arm = self._rail_at("bl" if ear.x < self.term.x else "br", self.ear_y)
            self._draw_rail(layer, [(arm, self.ear_y), (strap, self.ear_y)], self.ear_w)
        d = ImageDraw.Draw(layer)
        # Two bolts per strap, at its ends, and smaller than the ones on the mounts. At the ends
        # because that is where a clamp is actually fixed - a strap bolted through its middle
        # would pivot on it - and it also gets them off the arm's centreline, where a single bolt
        # sat level with the mount's own a couple of dozen pixels away and the pair of them read
        # as a face. Smaller because this is the lighter member: a bolt is sized to what it is
        # holding, and the panel already says so everywhere else.
        for ear in self.ears:
            strap = self.term.x if ear.x < self.term.x else self.term.right
            for y in (ear.y + r, ear.bottom - r):
                self._draw_bolt(d, strap, y, r)

    def _draw_rim(self, d: ImageDraw.ImageDraw, halo: tuple, phase: float) -> None:
        """Re-stroke the border, sunk by one breath. Only ever called while a session is up.

        The state light *around* it is baked - it is a full-screen composite and would cost
        milliseconds a frame - so what breathes is the crisp line on top of it, which is the
        brightest edge on the panel and the one a workshop reads from across the room. At the top
        of the breath the mix is zero and this puts back exactly the pixels ``_base`` drew, so the
        rim only ever dims from where it is now and never brightens past it.

        Redrawing over the same geometry is safe because ImageDraw does not anti-alias: the sunk
        stroke covers precisely the pixels the bright one did, with no fringe left showing.
        """
        d.rounded_rectangle(
            [self.frame.x, self.frame.y, self.frame.right - 1, self.frame.bottom - 1],
            radius=self.radius,
            outline=(*mix(halo, SCREEN, rim_breath(phase)), 255),
            width=self.line,
        )

    # ---- the two instruments ----

    def _dial_angle(self, value: float) -> float:
        """Where *value*, 0 to 1, lands on the sweep, in PIL's clockwise-from-three-o'clock."""
        return DIAL_FROM + DIAL_SWEEP * max(0.0, min(1.0, value))

    def slider_value(self, py: float) -> float:
        """What a finger at height *py* is asking the volume column for, 0 to 1.

        Absolute rather than relative: the level is *where your finger is*, not how far it has
        moved from wherever it happened to land. Relative was tried on paper and does not survive
        the geometry - the knob sits 70 px off the bottom edge of a 480 px panel, so a finger that
        grabbed it at 95 per cent would have had three quarters of the range above it and no room
        at all below.

        Clamped at both ends, which is what makes the grab itself harmless: the finger starts
        below the foot of the column and so starts at silence, and it is somewhere you have to
        drag *away* from rather than a value anybody arrived at by accident.
        """
        track = self.slider
        return max(0.0, min(1.0, (track.bottom - py) / max(1.0, track.h)))

    @property
    def dial_span(self) -> int:
        """Half-width of the tile either instrument is drawn into, in panel pixels."""
        return round(self.btn_r) + max(2, round(4 * self.scale))

    def _draw_instrument(self, layer: Image.Image, name: str) -> None:
        """One dial face: the well it is sunk into, its machined bezel, and the scale round it.

        All of it holds still for the life of the window - a bezel is the rail's own profile seen
        end-on, and a scale is a scale - so it goes into the chrome once and is never drawn again.
        What moves is a pointer, and that is a tile of its own; see :meth:`_hand`.

        Drawn through :func:`eye.smoothed`, unlike the two switches this replaces, and that is
        half of why it replaces them. Six concentric rings and a swept scale at 72 px is nothing
        but curves, PIL anti-aliases none of it, and the pair of them were the last stepped edges
        left on a panel whose eye, reticle and collar are all smooth.
        """
        cx, cy = (round(v) for v in self.switches[name])
        span, r = self.dial_span, self.btn_r

        def paint(t: ImageDraw.ImageDraw) -> None:
            middle = at(span)

            def box(rad: float, drop: float = 0.0) -> list[float]:
                reach = at(rad)
                return [middle - reach, middle - reach + at(drop) - at(0.0),
                        middle + reach, middle + reach + at(drop) - at(0.0)]

            # Something dark under it first: a disc with a chamfer and no shadow is a drawing of a
            # dial, and the same disc with something falling out from under it sits *in* the rail.
            t.ellipse(box(r + 1, 2), fill=linear(SCREEN, 140))
            t.ellipse(box(r), fill=linear(SCREEN, SWITCH_ALPHA))
            # Rings stepping down the rail's own profile rather than one stroke, so a dial reads
            # as the same piece of metal the bracket is made of. Half the stroke the switches here
            # used to carry: those had a glyph filling the well and could afford a fat rim, and a
            # dial cannot - the brightest ring inside a dial has to be its scale, or the eye reads
            # the bezel as the reading and the scale as decoration on it.
            for step in range(5):
                t.ellipse(box(r - step * 0.9), outline=linear(self._rail_colour(1.0 - step / 6.0)),
                          width=round(wide(1.2)))
            t.ellipse(box(r - 1.2), outline=linear(mix(GREEN_MID, GREEN, 0.4)),
                      width=round(wide(1.2)))
            self._paint_scale(t, span, name)

        layer.alpha_composite(smoothed(2 * span + 1, paint), (cx - span, cy - span))

    def _paint_scale(self, t: ImageDraw.ImageDraw, span: int, name: str) -> None:
        """The graduated arc a pointer is read against, inside the tile *name* is being drawn in.

        The knob's is one dim track, because a volume has no regions - it is loud where you put
        it. The gauge's is the board's own three: green until the clock starts being capped, amber
        to where it is capped in earnest, red past that. Those two breaks are
        :data:`~cyclops.stats.WARN_C` and :data:`~cyclops.stats.HOT_C` put through the very scale
        the admin page's bar uses, so the panel and the page cannot drift apart by hand.
        """
        r = self.btn_r
        middle, reach = at(span), at(r * DIAL_TRACK)
        box = [middle - reach, middle - reach, middle + reach, middle + reach]
        stroke = round(wide(self._dial_stroke))
        if name == HEAT:
            edges = (0.0, temp_percent(WARN_C) / 100.0, temp_percent(HOT_C) / 100.0, 1.0)
            for (lo, hi), colour in zip(
                zip(edges, edges[1:], strict=False), (GREEN_MID, AMBER, RED), strict=True
            ):
                t.arc(box, start=self._dial_angle(lo), end=self._dial_angle(hi),
                      fill=linear(mix(SCREEN, colour, 0.85)), width=stroke)
        else:
            t.arc(box, start=DIAL_FROM, end=DIAL_FROM + DIAL_SWEEP,
                  fill=linear(mix(SCREEN, GREEN_DIM, DIAL_OFF)), width=stroke)
        for i in range(DIAL_TICKS):
            angle = math.radians(self._dial_angle(i / (DIAL_TICKS - 1)))
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            t.line(
                [at(span + r * DIAL_TICK_IN * cos_a), at(span + r * DIAL_TICK_IN * sin_a),
                 at(span + r * DIAL_TICK_OUT * cos_a), at(span + r * DIAL_TICK_OUT * sin_a)],
                fill=linear(mix(SCREEN, GREEN_MID, 0.7)), width=round(wide(max(1.0, self.scale))),
            )

    def _hand(
        self,
        name: str,
        value: float | None,
        colour: tuple[int, int, int],
        speaker: tuple[int, int, int] | None = None,
    ) -> Image.Image:
        """The half of an instrument that moves: a pointer, its hub, and what it has covered.

        A tile rather than strokes on the frame, for the same reason the collar and the reticle
        are tiles: these are curves and a diagonal, and both read as a staircase drawn flat. It is
        cached on the value it is drawn for - see :meth:`_knob` and :meth:`_needle` - because a
        level moves when a finger moves it and a temperature moves once every five seconds, so at
        25 frames a second almost every frame wants the tile the last one had.

        ``None`` is a dial with nothing behind it - no mixer, no thermal zone - and draws its hub
        and no pointer at all, which is a gauge that is not reading rather than one reading zero.

        Both caches are bounded by what a reading can be: a whole percent for the needle and a
        step of the knob for the pointer, so a few hundred tiles of six thousand pixels is the
        worst either can come to. Against the per-state bases in :meth:`_base`, which are
        full-screen, that is not a number worth managing.
        """
        span, r = self.dial_span, self.btn_r

        def paint(t: ImageDraw.ImageDraw) -> None:
            middle = at(span)

            def hub(rad: float) -> list[float]:
                reach = at(rad)
                return [middle - reach, middle - reach, middle + reach, middle + reach]

            if value is not None:
                if name == VOLUME:
                    # What the knob has been turned past, lit over the dim track underneath it.
                    reach = at(r * DIAL_TRACK)
                    t.arc([middle - reach, middle - reach, middle + reach, middle + reach],
                          start=DIAL_FROM, end=self._dial_angle(value), fill=linear(colour),
                          width=round(wide(self._dial_stroke)))
                angle = math.radians(self._dial_angle(value))
                cos_a, sin_a = math.cos(angle), math.sin(angle)
                # A taper rather than a line: a needle with a wide root and a point is what says
                # "read the tip of this", and a stick of even width says "this is a spoke".
                root = math.radians(self._dial_angle(value) + 90.0)
                half = r * 0.075
                t.polygon(
                    [(at(span + r * DIAL_HAND * cos_a), at(span + r * DIAL_HAND * sin_a)),
                     (at(span + half * math.cos(root)), at(span + half * math.sin(root))),
                     (at(span - half * math.cos(root)), at(span - half * math.sin(root)))],
                    fill=linear(colour),
                )
            t.ellipse(hub(r * DIAL_HUB), fill=linear(colour if value is not None else GREEN_DIM))
            t.ellipse(hub(r * DIAL_HUB), outline=linear(SCREEN), width=round(wide(1)))
            if speaker is not None:
                self._paint_speaker(t, span, speaker)

        return smoothed(2 * span + 1, paint)

    def _knob(self, level: int | None, turning: bool) -> Image.Image:
        """The volume pointer at *level*, white while a finger is on it. Cached per appearance.

        The speaker rides in the same tile, because it is part of the same still picture and a
        tile that is already being cached is the cheapest place on this panel to put a shape.

        White is the whole of what the knob does under a finger, and it says the right thing at
        the right moment twice over: it and the column come up together on the touch, and once
        the finger is on the track it is the pointer following it that says the two are one
        control rather than two.
        """
        key = (level, turning)
        tile = self._knobs.get(key)
        if tile is None:
            colour = WHITE if turning else GREEN
            tile = self._knobs[key] = self._hand(
                VOLUME,
                None if level is None else level / 100.0,
                colour,
                speaker=GREEN_DIM if level is None else (WHITE if turning else GREEN_MID),
            )
        return tile

    def _needle(self, temp_c: float | None, lit: bool) -> Image.Image:
        """The heat needle for *temp_c*, in the band's own colour. Cached per whole percent."""
        percent = temp_percent(temp_c)
        band = temp_band(temp_c)
        key = (percent, band, lit)
        tile = self._needles.get(key)
        if tile is None:
            colour = WHITE if lit else HEAT_INK.get(band, GREEN_DIM)
            tile = self._needles[key] = self._hand(
                HEAT, None if percent is None else percent / 100.0, colour
            )
        return tile

    def _draw_hands(
        self,
        layer: Image.Image,
        d: ImageDraw.ImageDraw,
        volume: int | None,
        temp_c: float | None,
        pressed: str | None,
    ) -> None:
        """Both pointers, and the one reading each instrument prints in the gap under its hub.

        The gap is the quarter of the sweep neither dial uses, which is where a knob's own scale
        has always left room for a label. The knob keeps a speaker there, because its number is on
        the column a drag opens and one reading in two places is one of them being read twice. The
        gauge shows its degrees always: that is the whole of what a gauge is for, and a needle
        without a number is a mood ring.

        The words go on flat rather than into a tile: PIL renders glyphs through FreeType and they
        arrive anti-aliased already. Only the shapes need the tile.
        """
        turning = pressed == VOLUME
        for name, tile in ((VOLUME, self._knob(volume, turning)),
                           (HEAT, self._needle(temp_c, pressed == HEAT))):
            cx, cy = (round(v) for v in self.switches[name])
            layer.alpha_composite(tile, (cx - self.dial_span, cy - self.dial_span))

        cx, cy = (round(v) for v in self.switches[HEAT])
        band = temp_band(temp_c)
        colour = WHITE if pressed == HEAT else HEAT_INK.get(band, GREEN_DIM)
        word = "--" if temp_c is None else f"{round(temp_c)}°"
        self._text(d, cx, cy + round(self.btn_r * DIAL_LABEL), word, self.font_micro,
                   (*colour, 255), align="c")

    def _draw_slider(self, d: ImageDraw.ImageDraw, level: int) -> None:
        """The volume column, up for as long as a finger is on the knob.

        A ladder of rungs rather than a solid bar, and one rung per setting: the knob steps in
        fives like the page's slider does, so a column drawn continuously would show a level
        between two it can actually take. It is the pod's signal meter stood on end, in the same
        two colours, because a panel with two ways of drawing "how much of something" has one too
        many.

        Only what is lit follows the finger. The thumb is the bar across the top of the stack,
        with an ear either side of the track so it reads as something being held rather than as
        the last rung being brighter than the one under it; the number stands above the column,
        where it is out from under the hand and never moves.

        Flat, like every other filled slab here - the tag, the menu card, the pod. There is not a
        curve in it, and a stepped edge on a rectangle is not a step.
        """
        track = self.slider
        lit = round(max(0, min(100, level)) / 100.0 * SLIDER_RUNGS)
        d.rectangle([track.x - 1, track.y - 1, track.right, track.bottom],
                    fill=(*SCREEN, SWITCH_ALPHA), outline=(*GREEN_DIM, 255), width=self.line // 2)
        gap = max(1.0, self._rung * SLIDER_RUNG_GAP)
        for rung in range(SLIDER_RUNGS):
            top = track.bottom - (rung + 1) * self._rung
            d.rectangle(
                [track.x + 2, round(top), track.right - 2, round(top + self._rung - gap)],
                fill=(*GREEN, 255) if rung < lit else (*mix(SCREEN, GREEN_DIM, METER_OFF), 255),
            )
        ear = max(2, round(SLIDER_EAR * self.scale))
        half = max(2, round(SLIDER_THUMB * self.scale / 2))
        at_y = round(track.bottom - track.h * max(0, min(100, level)) / 100.0)
        d.rectangle([track.x - ear, at_y - half, track.right + ear, at_y + half],
                    fill=(*WHITE, 255))
        self._text(d, track.center[0], track.y - round(18 * self.scale), str(level),
                   self.font_mode, (*WHITE, 255), align="c")

    def _paint_speaker(self, t: ImageDraw.ImageDraw, span: int, c: tuple[int, int, int]) -> None:
        """A cone and its throat, in the knob's gap. Says which of the two dials this is.

        The gauge's half of that gap is a number in degrees; a knob's reading is the pointer, so
        what goes here instead is the one mark that says what is being turned. Inside the tile
        rather than on the frame: the cone is two diagonals, and a diagonal drawn flat at eleven
        pixels is the staircase this whole corner was rebuilt to be rid of.
        """
        r = max(3.0, self.btn_r * 0.17)
        mid = span + self.btn_r * DIAL_LABEL
        # One silhouette rather than a throat and a cone drawn separately: two shapes that share
        # an edge each own half of the pixels along it, and the shrink out of the tile averages
        # that pair into a seam down the middle of what is supposed to be one solid mark.
        t.polygon(
            [(at(span - r), at(mid - r / 3)), (at(span - r / 3), at(mid - r / 3)),
             (at(span + r * 0.9), at(mid - r)), (at(span + r * 0.9), at(mid + r)),
             (at(span - r / 3), at(mid + r / 3)), (at(span - r), at(mid + r / 3))],
            fill=linear(c),
        )


    # ---- the long press, and what it opens ----

    def _draw_hold(self, layer: Image.Image, hold: float) -> None:
        """Fill his collar in, left to right, as a finger holds his face down.

        The one gesture on this panel that is not a tap, so it is the one thing that has to say
        so while it is happening: without this, a long press is a second of a panel doing nothing
        followed by a menu, which reads as a fault that resolved itself.

        It is drawn *on the rail that is already round him* rather than beside it - the same
        radius, in full phosphor instead of the rail's own steel - so nothing new appears on the
        screen while you hold him. A line you already stopped seeing lights up from one end, and
        when it reaches the far side the menu is open. That also keeps it out of the picture:
        this panel has no room for a progress bar, and a ring drawn further out would cross the
        rail running away to either edge.
        """
        seat = self.eye_seat
        cx, cy = self.eye
        radius = seat.swell
        _, _, a0, a1 = self.brackets["bl"].shoulder(seat)
        start, sweep = a0, (a1 - a0) * max(0.0, min(1.0, hold))
        # Heavier than the hairline this used to land on, and now it has a 17 px rail to be seen
        # against: at the panel's own stroke it would be a brightness change on two pixels of
        # seventeen, which from a bench is no change at all.
        stroke = max(3, round(self.rail_w * 0.45))
        span = round(radius) + stroke

        def paint(t: ImageDraw.ImageDraw) -> None:
            middle, reach = at(span), at(radius)
            t.arc(
                [middle - reach, middle - reach, middle + reach, middle + reach],
                start=start, end=start + sweep, fill=linear(GREEN), width=round(wide(stroke)),
            )

        layer.alpha_composite(smoothed(2 * span + 1, paint), (cx - span, cy - span))

    def _draw_menu(self, layer: Image.Image, d: ImageDraw.ImageDraw, pressed: str | None) -> None:
        """The power menu: a scrim over the whole panel, and a card of rows on top of it.

        The scrim is composited rather than drawn, which is the difference between putting the
        panel out and punching a hole in it - see :func:`_mix`. Everything on the card is opaque
        for the same reason: this is the one thing on this screen you are asked to read before
        you touch it, and it does not get to be a window onto the room.
        """
        layer.alpha_composite(self._scrimmed())
        card = self.menu_card
        d.rounded_rectangle(
            [card.x, card.y, card.right - 1, card.bottom - 1],
            radius=self.radius,
            fill=(*SCREEN, 255),
            outline=(*GREEN_MID, 255),
            width=self.line,
        )
        pad = max(3, round(MENU_PAD * self.height))
        inset = card.x + pad * 2
        first = self.menu_cells[MENU_ROWS[0][0]]
        head = (card.y + pad + first.y) / 2
        self._text(
            d, inset, head, MENU_TITLE, self.font_tab, (*GREEN_DIM, 255),
            tracking=max(1.0, 2.4 * self.scale),
        )
        # The note is the half of this header that is actually load-bearing; the title only says
        # which menu you are in, and the note says which of the two sleeps this one means.
        self._text(d, card.right - pad * 2, head, MENU_NOTE, self.font_micro,
                   (*GREEN_DIM, 255), align="r")
        for key, label in MENU_ROWS:
            self._draw_menu_row(layer, d, key, label, pressed == key)

    def _draw_menu_row(
        self,
        layer: Image.Image,
        d: ImageDraw.ImageDraw,
        key: str,
        label: str,
        pressed: bool,
    ) -> None:
        """One row: a rule over it, a mark, a word, and the inversion that answers a thumb.

        The two that do something wear a glyph and full phosphor; CANCEL is centred, dimmer and
        under a rule of its own, because the way out of a menu is not one of its choices. Pressed
        inverts exactly as a tab does - it is the only feedback a screen with no travel has, and
        it should not have to be learnt twice on one panel.
        """
        cell = self.menu_cells[key]
        d.line([cell.x, cell.y, cell.right, cell.y], fill=(*GREEN_DIM, 200),
               width=max(1, self.line // 2))
        edge = max(1, round(2 * self.scale))
        if pressed:
            d.rounded_rectangle(
                [cell.x + edge, cell.y + edge, cell.right - edge, cell.bottom - edge],
                radius=max(0, self.radius - edge),
                fill=(*GREEN, 255),
            )
        ink = INK if pressed else GREEN_MID if key == CANCEL else GREEN
        tracking = max(1.0, 2.0 * self.scale)
        _, cy = cell.center
        if key == CANCEL:
            self._text(d, cell.center[0], cy, label, self.font_read, (*ink, 255),
                       align="c", tracking=tracking)
            return
        r = max(6, round(MENU_GLYPH_R * self.height))
        gap = max(6, round(14 * self.scale))
        x = cell.x + gap
        mark = self._glyph_power if key == POWER_OFF else self._glyph_restart
        mark(layer, x + r, cy, r, ink)
        self._text(d, x + 2 * r + gap, cy, label, self.font_read, (*ink, 255), tracking=tracking)

    def _scrimmed(self) -> Image.Image:
        """The wash that puts the panel out behind the card. Built once, then kept."""
        if self._scrim is None:
            self._scrim = Image.new("RGBA", (self.width, self.height), (*SCREEN, MENU_SCRIM))
        return self._scrim

    def _glyph_power(
        self, layer: Image.Image, cx: int, cy: int, r: int, colour: tuple[int, int, int]
    ) -> None:
        """IEC 5009, the mark on every power button ever made: a broken ring, a bar in the gap.

        Smoothed like the eye rather than stroked flat like the tab glyphs. Those are 32 px of
        straight lines and one circle; this is a ring with a gap in the top of it, and a stepped
        gap reads as a broken ring rather than as a deliberate one.
        """
        span = round(r * 1.15) + max(2, round(3 * self.scale))
        stroke = max(2, round(3 * self.scale))

        def paint(t: ImageDraw.ImageDraw) -> None:
            middle, reach = at(span), at(r)
            t.arc(
                [middle - reach, middle - reach, middle + reach, middle + reach],
                start=-62, end=242, fill=linear(colour), width=round(wide(stroke)),
            )
            t.line(
                [middle, at(span - r * 1.12), middle, at(span - r * 0.12)],
                fill=linear(colour), width=round(wide(stroke)),
            )

        layer.alpha_composite(smoothed(2 * span + 1, paint), (cx - span, cy - span))

    def _glyph_restart(
        self, layer: Image.Image, cx: int, cy: int, r: int, colour: tuple[int, int, int]
    ) -> None:
        """A ring with a head on it, going round again - the refresh mark, which is what a
        restart is: the same box, from the top."""
        span = round(r * 1.15) + max(2, round(3 * self.scale))
        stroke = max(2, round(3 * self.scale))
        end = math.radians(232)  # where the arc stops, up and to the left, and where the head is

        def paint(t: ImageDraw.ImageDraw) -> None:
            middle, reach = at(span), at(r)
            t.arc(
                [middle - reach, middle - reach, middle + reach, middle + reach],
                start=-48, end=232, fill=linear(colour), width=round(wide(stroke)),
            )
            # The head sits on the end of the sweep and points along it. Its tangent, on a
            # screen whose y runs downwards, is (-sin, cos) - which is the direction the arc was
            # travelling when it stopped, and the reason this reads as motion rather than as a
            # ring with a lump on it.
            tip = (-math.sin(end), math.cos(end))
            out = (math.cos(end), math.sin(end))
            point = (r * out[0], r * out[1])
            head = r * 0.85
            corners = [
                (point[0] + tip[0] * head, point[1] + tip[1] * head),
                (point[0] + out[0] * head * 0.6, point[1] + out[1] * head * 0.6),
                (point[0] - out[0] * head * 0.6, point[1] - out[1] * head * 0.6),
            ]
            t.polygon(
                [(at(span + x), at(span + y)) for x, y in corners], fill=linear(colour)
            )

        layer.alpha_composite(smoothed(2 * span + 1, paint), (cx - span, cy - span))


# How many rows of the panel one pass of :func:`composite` blends. The whole frame at once is
# the obvious way to write it and it is the slow way: at 800 wide, a full-height 16-bit
# intermediate is 2.3 MB and every one of the six passes over it streams in from memory, while a
# 24-row strip is 115 KB and stays in the Pi 5's L2 between them. Measured on the panel, whole
# frame against strips: 8.4 ms and 4.9 ms. The floor is broad - anything from 16 to 48 rows is
# within noise of the best - so this is a plateau to sit on rather than a number to tune.
COMPOSITE_STRIP = 24


def composite(frame_bgr: np.ndarray, rgba: np.ndarray) -> np.ndarray:
    """Alpha-blend an RGBA overlay onto a BGR frame, in place-ish, without touching PIL again.

    The arithmetic is 16-bit fixed point, not float. ``frame + ((over - frame) * a) >> 7`` with
    the alpha halved to 0-128 is the widest form that cannot overflow an int16 (255 * 128 fits,
    255 * 255 does not), which matters because the float version's temporaries are twice the
    width and there are more of them: 19.5 ms a frame against 4.9 ms, measured on the Pi with a
    real overlay, and this runs 25 times a second. Seven bits of alpha rather than eight costs
    at most one level on about 6% of pixels, on chrome drawn in three flat greens.

    Still allocates - see :meth:`cyclops.record.PanelSource.publish`, which hands the returned
    frame to an encoder on another thread and needs it to be nobody else's buffer.
    """
    import cv2  # local import keeps this module importable without a camera stack

    out = np.empty_like(frame_bgr)
    for y in range(0, frame_bgr.shape[0], COMPOSITE_STRIP):
        band = rgba[y : y + COMPOSITE_STRIP]
        under = frame_bgr[y : y + COMPOSITE_STRIP]
        over = cv2.cvtColor(band, cv2.COLOR_RGBA2BGR)  # RGB -> BGR to match the frame
        weight = ((band[:, :, 3].astype(np.int16) + 1) >> 1)[:, :, None]
        blended = over.astype(np.int16)
        blended -= under
        blended *= weight
        blended >>= 7
        blended += under  # back within 0-255: the result never leaves the two ends it is between
        out[y : y + COMPOSITE_STRIP] = blended
    return out


# Unsharp masking, for a camera that cannot be asked to do better. The endoscope streams
# 640x480 JPEG at quality ~44 and ignores the protocol's resolution command, so this is the only
# remaining lever on how much of the room you can actually make out - see docs/camera.md.
#
# The floor and the ceiling are what separate this from a sharpen slider. Detail below the floor
# is the sensor's noise and the encoder's blocking, and amplifying that is how a sharpened cheap
# camera comes to look like a cheap camera someone has sharpened. The ceiling caps how far any
# one pixel may travel, which is what kills the white halo an unsharp mask otherwise draws down
# every high-contrast edge - a face against a bright window grew one immediately without it.
# Measured on a Pi frame: 2.9x the Laplacian variance for +7% in the flat-area noise floor.
SHARPEN_RADIUS = 1.1  # gaussian sigma of the blur that defines "detail", in source pixels
SHARPEN_KERNEL = (5, 5)  # ...over a kernel stated rather than derived - see sharpen()
SHARPEN_AMOUNT = 0.8  # how much of the detail layer goes back on top
SHARPEN_FLOOR = 4  # ...but only where the detail is at least this strong
SHARPEN_CEILING = 12  # ...and no pixel may move further than this
# Off for now. It is the second-largest thing the render loop does - about 6 ms of every 40 ms
# frame, a seventh of a core at 25 fps - and turning it off is the cheapest way to look at the
# panel without it and decide whether the detail was worth the cost. Everything below is intact;
# flip this back to True to have it again. Note this is the switch for the *function*, so it
# also takes the sharpening off the photos handed to the model (see cyclops.webcam.for_model),
# not only off the preview.
SHARPEN = False


def sharpen(frame_bgr: np.ndarray) -> np.ndarray:
    """Unsharp-mask a frame, gated so it does not amplify what the JPEG encoder invented.

    A no-op while :data:`SHARPEN` is off, in which case the frame is handed straight back - both
    callers pass the result to a resize that allocates, so nobody is left holding an alias.

    Built from OpenCV primitives rather than the obvious numpy, because this runs on every
    preview frame inside the kiosk's 40 ms budget: the numpy version of the same arithmetic is
    several times slower for a result that differs by at most 3 levels in 1% of pixels.

    Two things here are performance, not taste, and both were measured on the Pi. The kernel is
    stated because leaving it to OpenCV derives a 9x9 from the sigma and spends 3.6 ms on the
    blur where 5x5 spends 1.7 ms - and at sigma 1.1 everything outside 5x5 is past 2.3 sigma and
    weighs nothing. And *frame_bgr must be contiguous*: this makes eight passes over it, and
    OpenCV copies a non-contiguous input on every one of them, which took the same function from
    8 ms to 30 ms. :func:`fit_to_window` is what guarantees that, and the camera hands it a
    contiguous frame to begin with.
    """
    import cv2  # local import keeps this module importable without a camera stack

    if not SHARPEN:
        return frame_bgr

    blur = cv2.GaussianBlur(frame_bgr, SHARPEN_KERNEL, SHARPEN_RADIUS)
    edges = cv2.threshold(
        cv2.absdiff(frame_bgr, blur), SHARPEN_FLOOR - 1, 255, cv2.THRESH_BINARY
    )[1]
    boosted = cv2.addWeighted(frame_bgr, 1.0 + SHARPEN_AMOUNT, blur, -SHARPEN_AMOUNT, 0)
    ceiling = (SHARPEN_CEILING,) * 4  # a scalar here would only reach the blue channel
    boosted = cv2.min(
        cv2.max(boosted, cv2.subtract(frame_bgr, ceiling)), cv2.add(frame_bgr, ceiling)
    )
    out = frame_bgr.copy()
    cv2.copyTo(boosted, edges, out)  # leave the flat areas exactly as they arrived
    return out


def fit_to_window(frame_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    """Centre-crop to the window's aspect, then scale - so faces keep their proportions.

    The webcam is 16:9 and the official Pi panel is 5:3; stretching one to the other makes
    everyone look wrong, so the sides get trimmed instead.

    A frame that has to be *enlarged* to fill the panel is sharpened first and then enlarged
    cubically. Both halves of that matter and both are cheap: the endoscope's 640x480 is smaller
    than the panel in both axes, so every preview pixel is invented by the interpolator, and
    INTER_LINEAR invents blurry ones. Sharpening happens before the resize because it is a third
    of the pixels there and therefore a third of the cost, and because sharpening after an
    upscale sharpens the interpolation's own softness rather than the picture's detail. A frame
    being *reduced* - a real webcam at 1280x720 - skips both: INTER_AREA is already sharp, and
    there is nothing there to rescue.
    """
    import cv2  # local import keeps this module importable without a camera stack

    h, w = frame_bgr.shape[:2]
    want = width / height
    have = w / h
    if have > want:  # too wide - trim left and right
        new_w = int(h * want)
        x0 = (w - new_w) // 2
        frame_bgr = frame_bgr[:, x0 : x0 + new_w]
    elif have < want:  # too tall - trim top and bottom
        new_h = int(w / want)
        y0 = (h - new_h) // 2
        frame_bgr = frame_bgr[y0 : y0 + new_h, :]
    if frame_bgr.shape[0] > height:
        return cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
    # Trimming rows leaves the buffer contiguous and this costs nothing; trimming columns does
    # not, and then paying for one copy here is far cheaper than letting OpenCV make its own
    # inside every call that follows. See sharpen() for what that was worth measuring.
    return cv2.resize(
        sharpen(np.ascontiguousarray(frame_bgr)), (width, height), interpolation=cv2.INTER_CUBIC
    )


def platform_font_note() -> str:
    """Which font face got picked - useful when the Pi and the Mac disagree."""
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            return path
    return f"PIL default bitmap font (no TrueType found on {sys.platform})"


def message(width: int, height: int, text: str) -> np.ndarray:
    """A black frame with one line centred on it, as BGR - the panel with nothing to show.

    Drawn here rather than in :mod:`cyclops.kiosk` because this is where the fonts already live.
    Dim phosphor rather than white: it is a caption explaining a black rectangle, not a thing to
    read across the room, and it has to sit under the same green chrome as everything else.
    """
    image = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    font = _load_font(max(16, height // 26))
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    draw.text(
        ((width - (right - left)) // 2 - left, (height - (bottom - top)) // 2 - top),
        text,
        font=font,
        fill=GREEN_MID,
    )
    return np.array(image)[:, :, ::-1].copy()  # RGB -> BGR, as everything downstream expects
