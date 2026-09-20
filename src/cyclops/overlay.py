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

His bracket's rail runs along its ramp into his housing, and is bolted to its brass on either
side of him. That is the join: it is what makes him part of the bracket rather than a badge
sitting on it, and it is why his corner is the big one - a face wants room, and the two
instruments opposite are bolted straight through their rail and take a third of the space. Those
two were a shutter and a microphone until the box grew a button of its own, which does both
without anybody having to find a 12 mm target on glass; they are a volume knob and a heat gauge
now, which is the pair of things this panel could not otherwise be told or asked. While he is
asleep the only thing on this panel that moves is his own breath: no ring turns, nothing blinks,
no readout changes, and nothing beckons - there is nothing left in that corner to press. That is
what makes any of the rest of it read as awake: the *mechanism* stopping, rather than the
creature holding its breath.

Everything that holds still while the state does - the halo, the brackets and their bolts, the
reticle, the pod's heat lamp, the switches at rest - is built once and cached, keyed on the state.
A Pi rendering this at 25 fps has 40 ms for the whole loop and the camera wants most of them; what
is left for a frame here is a signal meter, a clock, the record light, a caption, one ring, the
border line and the eye. The record light is the one tag that is not baked, because it is the one
that blinks - it is a kept tile the frame composites, like the meter and the clock's own glyphs,
so what a blink costs is a composite and not the blur behind it. The terminal is baked with the rest of the chrome except for the line printed on it, whose
halation is 1.3 ms of that - measured on the Pi against the same frame drawn crisp-only, and
worth having at the price because it is what makes the glass read as lit rather than printed.
Every part of the eye moves, so none of it is cached at all; measured on the Pi, he is 10.5
ms of a 15.5 ms frame, which is the single largest thing this loop does and is meant to be - he is
the only part of the panel anybody looks at. He grew from r60 to r88 when he moved into the
corner and took about 3.5 ms with him, which is the whole of the difference between this and the
tab row; the board sat at 59 C and 0x0 throttled afterwards, so it is a price that is being paid
out of headroom rather than out of frames. He came back down to r66 when r88 turned out to take
up too much of the screen, and the Pi figures above are r88's until somebody re-measures them;
on the Mac the smaller eye is about a fifth cheaper. The number is worth keeping honest, because
it was wrong here for a long time: this line once claimed a tenth of a millisecond, which was
the figure before he was ever supersampled. Re-measure with `deploy/push.sh && ssh
cyclops@cyclops.local` and a timing loop around `Overlay.render`, not by reasoning about it.
"""

from __future__ import annotations

import math
import sys
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import lru_cache, wraps
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from . import material
from .devices import CAMERA, MUSIC, STORAGE, Device
from .eye import (
    AHEAD,
    AWAY,
    BLINK_DRIFT,
    DIALS,
    FRAME,
    LANDMARKS,
    POINT,
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
from .point import PanelMark
from .stats import HOT_C, WARN_C, temp_band, temp_percent
from .tutorial import Tutorial

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


def session_live(state: str) -> bool:
    """Is there somebody on the other end - as opposed to a session being attempted?

    Stricter than :func:`session_up` by exactly the wait for the socket, the way :func:`awake`
    below is stricter by exactly the teardown. It exists for the lid and for nothing else. The
    caption, the halo and the ring in the button all go optimistic the instant you ask, because
    what they are reporting is that the box heard you - but a lid is a claim about the far end,
    and one that winds open over a socket that has not connected yet is a box telling you it is
    listening before it can hear.
    """
    return session_up(state) and state not in (STARTING, CONNECTING)


def working_over(state: str, busy: bool) -> str:
    """*state*, or WORKING when something is running in the background that it does not cover.

    Two states get replaced and the second one is the whole reason this is a function rather than
    a comparison. IDLE is the obvious half: nobody is talking to him and the child is filing a
    session, so a panel that snores through it is lying. DRAWING is the half that was missed on
    the first attempt, and it is the commonest case by far - you ask for a diagram *in* a
    conversation, so the state for the whole of that half minute is DRAWING and never IDLE, and
    gating on IDLE alone meant the one path anybody would actually take never showed the face.

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
# ...and the two steps down from it, which exist for one reason: the volume knob goes over to
# this hue entire while a companion holds his voice, and an instrument is not one colour. It is a
# bright hand and arc, mid graduations and a nearly-out track, and a knob that lost those three
# levels on the way across would read as a sticker rather than as the same instrument relit.
# Stepped by what GREEN_MID and GREEN_DIM are of GREEN, so the two hues wear the same clothes.
BLUE_MID = (66, 116, 176)
BLUE_DIM = (44, 78, 118)
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


# The pilot lamp on the small mount's plate: a brass fitting with a glass dome in it, lit from
# inside, in the state's own colour. See :meth:`Overlay._draw_pilot` for the fitting and
# :meth:`Overlay._pilot_tile` for the light.
#
# It says nothing the border does not already say, and that is the point of where it is rather
# than an argument against it: the border, the eye and the terminal are all on the left or round
# the edge, and the bottom-right corner is the one place on this panel a glance lands with nothing
# in it. What it adds over the border is a SHAPE - a border can only be a colour, and a lamp can
# hurry, breathe or hold still - which is the difference between "he is up" and "he is up and
# hunting".
#
# Two shapes and no more. A square wave is a thing being SWITCHED, which is what the record
# light already argues at length (see REC_PERIOD_S), and STEADY is a lamp that is simply on,
# which is the commonest thing a lamp is. There was a breath here for the live states and it is
# gone: a session that is up is not a thing that fades in and out, and Marco's call was that
# while he is live the lamp just burns.
STEADY, BLINK = "steady", "blink"


@dataclass(frozen=True)
class Pilot:
    """What the lamp is doing in one state: a colour, a shape, and how fast.

    ``colour`` of None is a dark lamp, and that is a state rather than an omission: asleep, the
    fitting is there and unlit, so the lamp coming on is itself the news that something is
    happening. It is also what keeps a sleeping panel still - see
    ``tests/test_eye.py::test_only_two_things_move_while_he_is_asleep``, which allows the eye and
    the caption to move at IDLE and nothing else at all.
    """

    colour: tuple[int, int, int] | None
    shape: str = STEADY
    period_s: float = 0.0
    duty: float = 0.5  # BLINK: the lit share of one period. Under a half is a flash with a rest

    def level(self, phase: float) -> float:
        """How brightly the lamp burns at *phase*, 0 dark .. 1 full."""
        if self.colour is None:
            return 0.0
        if self.shape == BLINK and self.period_s > 0.0:
            return 1.0 if phase % self.period_s < self.period_s * self.duty else 0.0
        return 1.0


# One row per state, and three things the lamp can be: out, hurrying in amber, or burning.
#
# A live session is one look and not five. The states a conversation actually passes through -
# listening, speaking, looking at a photograph, searching, drawing - all wear the same bright
# phosphor green and hold still, because what the lamp is answering there is "he is up", and
# five shades of the same answer is five things to learn for one fact. The eye is the part of
# this panel that says what KIND of up, in far more detail than a lamp could, and the terminal
# spells it out underneath. This one burns.
#
# That green is GREEN and not the mood's own white on purpose - it is the phosphor the rest of
# the chrome is drawn in, so a lit lamp reads as this panel's own hardware rather than as a
# fifth accent colour. The other three rows keep MOODS' colours, because amber, blue and red
# each mean something the green cannot.
#
# The periods are coprime-ish with the panel's other clocks - 1.0 (REC), 1.2 (the caption's
# cursor), 2.4 (the caption's breath), 3.7 (the border's). Two things blinking on one panel that
# fall into step read as one mechanism rather than two, and the fix is arithmetic rather than
# taste; test_no_lamp_period_locks_to_the_panels_own is what keeps it that way.
LAMPS = {
    # Dark, and the only row here that draws nothing. Marco's call, and the strongest version of
    # what this lamp is for: with the fitting unlit, the lamp coming on IS the signal.
    IDLE: Pilot(None),
    # Amber and hurrying, both ends of a session. Half a second is about as fast as a lamp can
    # blink and still read as deliberate rather than as a fault.
    STARTING: Pilot(AMBER, BLINK, 0.44),
    CONNECTING: Pilot(AMBER, BLINK, 0.44),
    STOPPING: Pilot(AMBER, BLINK, 0.7),  # the same transition, run backwards and slower
    # ...and then a session is up, and the lamp simply burns. Five states, one look: the
    # conversation is live and that is the whole of what this is saying.
    LISTENING: Pilot(GREEN),
    SPEAKING: Pilot(GREEN),
    LOOKING: Pilot(GREEN),
    SEARCHING: Pilot(GREEN),
    DRAWING: Pilot(GREEN),
    # Blue, and a short flash with a long rest rather than an even blink: nobody is in a
    # conversation, and a lamp ticking over in the corner is exactly what that is.
    WORKING: Pilot(BLUE, BLINK, 1.57, duty=0.22),
    # Red and dead still. The fault is the one thing on this panel that does not move - see
    # test_only_the_broken_face_holds_still - and a nagging red light would be this panel
    # shouting where every other part of it has agreed to say it once, in colour.
    ERROR: Pilot(RED, STEADY),
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

BUSY_MARK = "…"  # the last character of a caption that ends in a blinking cursor. Stripped
# before the line is drawn - it marks the cursor rather than being punctuation - so a caption
# carrying it reads on the glass as exactly the words in front of it. Declared here rather than
# beside the drawing because CAPTIONS below is its first user.

# ... and the resting line underneath: what is true about a state when nothing finer is known.
# The controller sends a better sentence whenever it has one - what is being searched for, which
# project is being opened, which step of the teardown is running - and that wins; this is what the
# panel falls back on. Every state has one, so the line is never blank while a session is up.
CAPTIONS = {
    # Never actually read: this state exists only while cyclops.tasks has a sentence, and that
    # sentence is what the caption shows. Here because every state has a resting line and the one
    # that could get away without it is the one that would be blank on the day something changed.
    WORKING: "working…",
    IDLE: "Press button to start" + BUSY_MARK,  # the one caption on this panel that is an
    # instruction, and the one state that needs one: a shut steel lid is not a control, and
    # nothing else in the corner beckons while he is asleep. He snored here for a while, which
    # was him rather than a readout and read beautifully to anybody who already knew what to do.
    #
    # The mark is not punctuation and is never drawn - it is what puts the blinking cursor after
    # the line, and the line needs one for a reason the snore never did: with the lid down,
    # nothing else on this panel moves at all. His breath is behind steel now. A cursor is the
    # whole of what is left saying the box is running rather than stopped, and after an
    # instruction it is the oldest convention there is for a machine waiting to be told.
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
    #
    # And the one row with the steel across it. The cover is a property of being asleep rather
    # than of any face - see eye.COVER_PIVOT - so it is set here, once, and every other row in
    # this table is drawn exactly as it was before the cover existed. A state arriving without a
    # row falls back to this one and gets the lid with it, which is right: an unknown state is
    # not a face this panel knows how to put up.
    IDLE: Mood(tint=GREEN_MID, aperture=0.24, swell=0.14, breath_s=6.5, spin=2.5, sway=1.6,
               drift=0.30, cover=1.0),
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
    # out of its own glass. Every ten seconds or so he turns to the picture he is sitting on,
    # HOLDS it long enough for you to see that he is looking at it, and comes back. That is the
    # whole of it, and it is two names and three numbers.
    #
    # 0.22 over a 2.75 s window, measured rather than guessed: he holds you for eight or fourteen
    # seconds, the look itself runs 2.4 to 2.6 s, and 79% of frames over ten minutes have him on
    # you - roughly what a person listening does. Swept over every start phase, a twelve-second
    # strip sampled once a second catches that look on two or three consecutive frames 87% of the
    # time and misses it entirely twice in a hundred. The old row was 0.26 over 2.6 s with the
    # hold written as a *share* of the window, which came out at 1.1 s: he arrived and left inside
    # a single sample, and three critics in a row read it as a dropped frame rather than a look.
    #
    # The iris is the other half of "he is here". It swelled 0.07 - a one-per-cent change in the
    # pupil, measured, which is not a breath, it is a rounding error - and the only thing on the
    # face that visibly moved with it was the specular core. 0.16 opens him from half to two
    # thirds and back every four seconds, which is a lens breathing rather than a lamp flickering.
    LISTENING: Mood(
        tint=WHITE, aperture=0.50, swell=0.16, breath_s=4.0, spin=7.0, blink_s=4.4,
        look=(AHEAD, FRAME), gaze=0.70, dart=0.22, dart_s=2.75, drift=0.05,
    ),
    # Talking: a faster breath and a wider iris, because he is doing the thing rather than waiting
    # to. The one row left that opens to level at all, and the reason the knob still exists: the
    # level here *is* his own voice coming back, so a face moving with it is a face moving with
    # what it is saying. That is the case the gesture was always right for. A tenth, because his
    # own words should show on him without him mouthing them.
    #
    # He holds your eye while he talks, and what he looks at when he does look away is his own
    # caption - the one place on the panel that is what he is saying, and the reason WORDS is
    # named second rather than anywhere else in the list (see eye.FOCUS). It is the same gesture
    # as reading back what you have just written. Fewer looks than listening and a longer window
    # between them, which is the opposite of a person (speakers avert more than listeners do) and
    # right for this one: he is a face on a panel, and a panel that looked away while answering
    # you would read as not answering. The look itself is as long as listening's - a glance at his
    # own line that lasted half a second would be a twitch towards the caption, not a check of it.
    SPEAKING: Mood(
        tint=WHITE, aperture=0.66, swell=0.19, breath_s=1.1, voice=0.10, spin=13.0, blink_s=5.5,
        look=(AHEAD, WORDS), gaze=0.50, dart=0.22, dart_s=3.7, drift=0.04,
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
MARKER = "› "  # what every caption opens with, and the smallest thing that wears the accent
CAPTION_LINES = 2  # how far a sentence may wrap before it is cut short instead, and now also
# how deep the terminal's screen is - the glass is cut to its text rather than the other way
# round. One line meant every phrase worth reading - a fault, a search, what a tool is doing -
# was trimmed to a stub ending in an ellipsis with most of the panel's width still free beside
# it. Two is where it stops: a third is a screen deep enough to start eating the picture, and a
# caption that big is a dialogue box rather than something said in passing.
# While a walkthrough is up, the top row of the glass is its bar and the sentence has what is left
# - elided rather than wrapped, so a long search line can never climb into the bar. The case stays
# the height it is: the bar goes *in* a row, not over the terminal.
BAR_ROWS = 1
BAR_WELL = 0.55  # how much black is over the glass in an empty step's cell - a well, not a lamp
CAPTION_ALPHA = 253
BLOOM_R = 2.6  # reference px of skirt round a lit glyph. 1.3 ms a frame on the Pi...
BLOOM_ALPHA = 0.80  # ...and how much of the letter's own alpha goes into it. Both are held well
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
# The record light, which is the one thing on this panel that blinks. A square wave and not a
# breath, by the argument caption_pulse makes about its cursor: a thing being *switched* rather
# than a thing being dimmed, and a record light has been switched on every machine that ever had
# one. One second, because that is the rate the clock beside it counts at and the two are the
# same instrument saying the same thing - and because a whole second is what a camcorder does.
REC_PERIOD_S = 1.0
REC_DUTY = 0.65  # ...lit for rather more than half of it. Even would read as a warning; this
# reads as a lamp that is on, interrupted, rather than as one that is missing half the time.
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

HALO_CORE = 0.0208  # fraction of the height the state light is at full brightness by, measured
# in from the panel's outer edge. It starts HALO_LIP short of the surround's inner arris, so the
# frame's own inner chamfer is lit by it and the pool of light sits *under the lip* rather than
# on the metal's face - a cove, which is where a light that has to be a light and not a wash on a
# machined surface belongs. The outer end of the band is dark because the surround is standing in
# front of it, and a bloom in front of the picture cannot also be in front of the case.
HALO_FALLOFF = 0.0125  # and how far it reaches inwards past that before it is gone. The two
# together are the light's whole reach in from the panel edge, which is what
# test_he_rides_the_ramp_and_his_rim_is_off_the_bezel measures his clearance against
HALO_LIP = 2.0  # reference px of the surround's inner chamfer the light reaches back onto
HALO_PEAK = 0.45  # alpha at the border, falling away to nothing before it reaches any text
TINT_ALPHA = 0.022  # green wash under the chrome - phosphor cast, not a colour filter. It was
# 0.05, and a twentieth of GREEN is +7 of green bias on its own: with SCREEN under it the plate
# came out at +32 while the bars on it sat at +8, so the two read as different materials and the
# duller one read as paint. The cast is worth keeping - the tube does throw light on its own
# chassis - but it has to be the faintest thing on the plate rather than its colour.
SCANLINE_EVERY = 3  # every third row of the chrome is darkened...
SCANLINE_ALPHA = 0.17  # ...by this much, which is a CRT at arm's length and not a zebra
VIGNETTE_FROM = 0.46  # where the corner shading starts, as a fraction of the half-diagonal
VIGNETTE_ALPHA = 0.42
# The strip and the tab row this layout replaced used to be opaque near-black, which bought
# contrast at the price of a third of the panel - 29% of a feed you are holding down a pipe to
# see what is at the bottom of. The brackets carry the filter and nothing else, so the picture
# runs edge to edge behind them and the chrome earns its contrast from its own opaque glyphs
# rather than from a bar.
TERM_ALPHA = 205  # the pod's window, and the one see-through well left on the panel. It used
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

# ---- the surround ----
#
# The panel's outermost element, and the one thing on it nothing else is bolted to. It was a two
# pixel stroke for four rounds: a coherence critic sampled 2,240 perimeter pixels and found ONE
# colour in all of them, the same 2 px on all four edges with no variation by direction, against
# 1,518 colours on the same perimeter of the panel we are being judged against. A boundary with
# no material in it reads as part of the projected layer rather than as the case the projection
# is in, which is why that critic put it first out of five.
#
# So it is a machined surround now, in the panel's own bar language turned inside out: a crown
# chamfer on the outer arris, a face crowned so gently it is a fall and not a curve, and a steep
# chamfer on the inner one, over a radiused opening cut in a square plate - so the corners are
# solid metal with a mitre where the two crowns meet, which is the one place a frame is ever
# actually looked at. Nothing in here decides which of the two chamfers is the bright one: the
# chamfer's own normal against material.lamp_2d does. The top and left rails therefore carry
# their specular on the OUTSIDE, the bottom and right ones carry it on the INSIDE, and the face
# falls monotonically from whichever end the lamp is at to the other, which ends up the darkest
# metal in the section. One lamp cannot light four edges the same way and this does not pretend
# it can.
#
# The state stroke stays exactly where it was, on the outer arris: read it as the light let into
# the frame's edge, which is what carries the session's colour to the far side of a workshop and
# what breathes (:meth:`_draw_rim`). The crown is built INWARD from it.
FRAME_W = 11.0  # reference px the surround runs in from the panel's outer edge, of which the
# outermost `line` are under the state stroke. Wider and it eats the bottom rail's spine and the
# collar round his face; narrower and there is no width for a chamfer, which is the whole point
FRAME_CROWN = 2.4  # reference px of the outer chamfer, the edge the lamp is on...
FRAME_UNDER = 2.6  # ...and of the inner one, which is the terminator on the lit edges
FRAME_CROWN_TILT = 0.88  # sin of the crown's steepest facet - near end-on, so where it faces the
FRAME_UNDER_TILT = 0.92  # lamp it mirrors it, and where it does not it takes nothing at all
FRAME_FACE_TILT = 0.42  # sin of the tilt the face has reached by its two chamfers, outward at
# the outer end and inward at the inner one. This is what makes the top rail's face two thirds
# brighter than the bottom rail's off the same steel: a face that only domes reads as one bar
# lit four ways, and the panel we are matching runs 176 across its top rail against 80 across
# its bottom one
FRAME_FACETS = 2  # flats a chamfer is milled into. A chamfer that airbrushes from face to
# silhouette is a fillet; a real one steps, and the steps are what say it was cut
FRAME_FACET_AA = 0.7  # px of each step that is rolled, which is the step's anti-aliasing and
# nothing else: a hard step following the opening's radius rasterises as a comb of teeth
FRAME_ARRIS_WAVE = 0.55  # how much the crown's width breathes along the run, either way...
FRAME_ARRIS_PITCH = 0.30  # ...and how slowly. Between them they walk the brightest row of the
# surround in and out by a pixel over a few tens of columns, which is what a straight edge on
# real stock does and what a rendered section cannot: a specular pinned to one row for eight
# hundred columns is the tell that a section was extruded rather than lit
FRAME_WEAR = 0.30  # how much the crown's polish comes and goes along a run. Down from the
# mirror, never up - a highlight already reflecting the lamp cannot reflect more of it
FRAME_BREAK_AT = 0.78  # sin past which the arris is a broken edge rather than a rolled one...
FRAME_BREAK = 0.55  # ...and how much of its highlight survives being knocked off
FRAME_STEEL = 0.70  # how far the face is put from STEEL towards STEEL_LIT. Above the plate it
# stands on and a shade under the brackets bolted to it: the case is the dullest steel that is
# still obviously steel, because everything else on the panel has to read in front of it
FRAME_FACE_GLOSS = 0.20  # of the material's highlight the face keeps. The arris keeps all of it
FRAME_GRAIN = 0.98  # of the material's brushing, drawn along the run rather than across it
FRAME_GRAIN_DARK = 0.45  # ...and how much of it goes the other way - brushed steel scatters
FRAME_DRIFT = 0.15  # how much the face's light comes and goes along a run, either way. Half
# what a bracket's rail carries: a rail is a foot long and a hand has been all over it, and this
# runs the whole perimeter, where a forty-level swing between two columns reads as a stain
FRAME_SCRATCHES = 26  # hairlines dragged along each run of the surround, of which a handful
FRAME_SCRATCH = 0.30  # cross any one stretch of it, and how pale one shows where it does
FRAME_SHEET_MARK = 0.55  # ...and how much of the sheet's own long hairlines carry onto the case.
# Knocked back rather than dropped: they are what says the plate and the case are one piece of
# metal, and at full weight a 60 px diagonal across a 9 px rail is a crack and not a scratch
FRAME_SCRATCH_SPREAD = 16.0  # degrees either side of the run they wander
FRAME_PITS = 90  # pits over the sheet, of which the surround keeps the ones that land on it...
FRAME_PIT_DEPTH = 0.44  # ...and how much of the face's light goes at the bottom of one
FRAME_LIFT = 3.0  # how proud the surround stands of the picture, which sets the shadow it drops
FRAME_SHADOW = 0.58  # alpha of that shadow where it is deepest. Soft, and it recovers over a
# dozen rows: a shadow that is an opaque floor for five rows and then gone is a painted band
FRAME_CONTACT = 0.80  # alpha of the hard line where the surround meets the picture...
FRAME_CONTACT_W = 2.6  # ...how far in from its inner arris that line reaches...
FRAME_CONTACT_LIT = 0.40  # ...and how much of it survives on an edge whose inner chamfer is the
# one the lamp is on, because there the light gets under the lip
FRAME_SCREW = 4.2  # reference px: the head in each corner of the plate. Smaller than a rail's,
# because it is holding a bezel down and not a member on - and the panel already says everywhere
# else that a fixing is sized to what it is holding
FRAME_SCREW_AT = 0.62  # of the band, down the corner's diagonal. Out on the gusset where the
# plate is deepest, which is the one place on a nine-pixel run a head can sit clear of both
# chamfers - and the one place a bezel is actually fixed

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
# line and the one turned away is a dark one, with a hard line of contact shadow under it and a
# soft one falling the other way from the light. That is what makes a bracket read as something
# bolted to the panel rather than as a line drawn on it, and it is why the rail is this thick: at
# a hairline there is no width for a chamfer. It was a round-edged section for a while and read
# as a tube - a bright edge fading to dark is a pipe whatever it is made of, and flat stock is
# what a bracket is cut from.
#
# One lit edge, and only one. For a round the far chamfer gave back a glint of the lit plate
# under it (RAIL_RETURN), and three critics measured a hairline at 212 on the edge turned *away*
# from the lamp against 211 on the edge turned towards it and read the whole rail system as an
# emboss - a bar lit from two sides. The far edge is dark now, the face grades away from the
# light, and nothing on the shadow side of anything is a light line.
RAIL = 17.0  # reference pixels, and the one number the whole bracket language rests on
RAIL_STEEL = 0.58  # how far the bar's face is put from STEEL towards STEEL_LIT. Bar stock is
# the brightest metal on the panel short of a highlight: it stands proud and square to the lamp
# where the plate under it lies back in its own shadow. At STEEL itself a face came out at 82
# against a plate at 88 - the same slab twice, which is what three critics measured and called
# fused. At this it runs 145 down to 105 under the lamp and 120 to 85 in the far corner, which
# is two to three times the plate it is bolted to wherever it is standing
RAIL_EDGE = 2.0  # reference px of the rolled edge the lamp lights. Not the width of a bright
# line - the width of the turn: the roll runs from the face's crown at this depth to fully
# end-on at the outline, and the angle that mirrors the lamp is met about two thirds of the way
# out, which is where the ridge lands and why it lands a pixel or two IN from the silhouette
RAIL_ARRIS_WAVE = 0.40  # how much that width breathes along the run, either way...
RAIL_SWEEP_PITCH = 0.25  # how slowly the slow drift runs, against the pitch of the fast one
RAIL_ARRIS_PITCH = 0.35  # ...and how slowly, against the pitch the polish comes and goes at.
# Between them they walk the brightest row of a bar in and out by a pixel over a few tens of
# pixels, which is what a straight edge on real stock does and what a rendered section cannot
RAIL_CHAMFER = 2.0  # reference px of the chamfer turned away from it, which is the dark edge:
# wide enough to be read at arm's length, narrow enough to stay an edge rather than a band;
# capped for a thin member so a clamp's nine-pixel strap keeps a face between its two chamfers
RAIL_BREAK_AT = 0.72  # sin of the tilt past which an arris is a broken edge rather than a
RAIL_BREAK = 0.5  # rolled one, and how much of its highlight survives being broken
RAIL_FAR_TILT = 0.78  # sin of the far chamfer's slope. Steep: this is the terminator, the last
# band of the section before the contact shadow, and it is turned far enough out of the lamp to
# take no light at all. It was 0.60, at which it came out a mid grey and the bar had a bright
# edge, a face, and then nothing to stop it - a stroke on a fill rather than a section
RAIL_FACE_GLOSS = 0.22  # how much of the material's highlight the face keeps. The arris is
# polished by the tool that cut it and the face is not, and the lamp's lobe is broad enough that
# a face crowned far enough to grade at all sits on the shoulder of it: at full gloss the first
# three pixels in from the lit chamfer came out at 200 too, which is a four-pixel highlight
# however narrow the chamfer is. One bright line means the face under it is matt
RAIL_CROWN = 0.12  # sin of the tilt the face has reached by its chamfers, from none at its
# middle - towards the lamp on the lit half, away from it on the other, so the face falls from
# its bright edge to its dark one. It was 0.02 to keep the face from reading as a tube, and a
# face with no gradient at all was read as dead flat instead: 87 to 95 across the whole width,
# where a frame member in the reference falls about forty from its lit edge to its far one. Back
# from 0.16 now that RAIL_SHUT carries part of that fall: the two together put a horizontal run
# at forty levels across its face and a strut at twenty-six, and the crown alone put the first
# at thirty-five and the second at twelve
RAIL_LIFT = 3.0  # how proud the bar stands of what it is bolted to, which sets its cast shadow
RAIL_CONTACT = 0.95  # alpha of the hard contact line where the bar meets the plate. Near
# total: what is under a bar touching a plate is not a dark tint, it is occluded, and the fifth
# percentile of a lit box round our chrome sat at 20 where the panel it is judged against
# reaches 10. The band is under two pixels wide, so this is a line and not a shadow...
RAIL_CONTACT_W = 1.8  # ...how far out from the edge it reaches before it is gone...
RAIL_CONTACT_LIT = 0.55  # ...and how much of it survives on the edge the lamp lights
RAIL_WEAR = 0.30  # how much the highlight comes and goes along a length - handled steel is
# polished where hands have been and dull between. It only ever takes the ridge *down* from a
# true mirror of the lamp (see `polish`), so at the old 0.45 most of a run sat a third under the
# ceiling and the bar never blew anywhere along its length...
RAIL_DRIFT = 0.030  # ...and how much the face under it does, either way, at the same pitch...
RAIL_SWEEP = 0.018  # ...over a slow one four times its length, which is the light in the room
# rather than the hands on the bar. These two used to be 0.30 and 0.20, which is half the face's
# light moving either way: on a strut whose shading spans 54 levels the texture swung 45, so the
# noise was larger than the thing it was meant to be noise ON and the section stopped reading.
# Texture that outruns its own gradient is a grain overlay, not a surface. Together with
# RAIL_GRAIN the three of them now hold every face inside about fifteen per cent of its own
# local value, which is where wear stays a finish rather than becoming the subject. Halved again
# from 0.055 and 0.030 because form has to outrank texture along a member as well as across it:
# the panel this is judged against varies about three levels down the length of a bar and forty
# across its section, and the lengthwise drift that is *form* is the lamp's own falloff, not
# these. Measured, the foot rail's face now moves sixteen levels over 145 px of run against a
# hundred and thirty-five across its width
RAIL_GRAIN = 0.55  # of the material's brushing. Under one and the bar is smoother than the
# sheet it is on; under half and at arm's length it is plastic
RAIL_RIDGE_WEAR = 0.18  # how much of the lit arris a chip takes, at the brushing's own pitches
# read along the bar instead of across it. This is the fix for the one measurement that says
# "rendered" louder than any other - a specular that stayed within 0.7 of a level for 158 pixels
# of its run, where nothing sawn, deburred and then handled has an edge that straight. About half
# the pixels of a ridge are chipped at all and the worst of them lose a sixth, which is a
# granular worn line rather than a wire. Down only: a broken edge cannot reflect more than a
# clean one, and the ridge's ceiling stays exactly where the material says it is
RAIL_TURN_FALL = 1.5  # how fast a turned bar's far half rolls away from its crown, as a power
# of how far out of the section a pixel is. `turned` on :meth:`_draw_rail` says whether a bar
# has that section at all, 0 for sawn flat stock and 1 for round bar: a flat member's far half
# stays near the face's own angle, and a round one keeps rolling until the light dies well
# inside its own silhouette, which is what puts a core shadow in the section instead of a floor.
# The two frame rails are round bar and everything bolted to them is flat plate, which is one
# more thing telling a member from the thing it carries
RAIL_SHUT = 0.52  # how much of the room the far side of a *flat* bar loses to the plate under
# it, which is the fall that does not care which way the bar runs. The crown does care: it is
# the lamp's bearing projected onto the section, and on a 45-degree strut that projection is
# 0.40 where a horizontal run gets 0.93. So the same crown that fell 35 levels across the bar
# along the bottom edge fell 12 across the strut bolted to it, and three of our seven runs came
# out flat to within the brushing - "a flat fill bracketed by edge strokes", measured four
# rounds running. Occlusion is geometry rather than bearing and lands the same on all of them.
# Up from 0.42, which left a flat member's far chamfer at 24 over a crest of 224 - a section
# whose darkest metal was a tenth of its brightest where the panel this is matched against runs
# a member from 203 down to 12. In-member range is the measurement that says a bar was lit
# rather than filled, and this is the number that buys it on the runs the crown cannot
RAIL_OCCLUDE = 0.70  # ...and how much a turned one loses, which is more, because a round bar
RAIL_OCC_AT = 0.30  # curls right under itself; and how far down the section either of them
# starts. A dark half sitting at the ambient floor is a *fill* at 39 levels; a bar touching a
# plate cannot see the room down there at all, and the last of its far half goes to about 20 -
# which is the core shadow every critic measured the absence of. Ambient occlusion, and it
# belongs to the diffuse floor rather than to the colour: see material.steel's `ambient`
RAIL_BOUNCE = 0.34  # ...and how much light comes back UP off the plate onto the last of it,
RAIL_BOUNCE_AT = 0.52  # over the outer half of the far side. This is the fourth event in the
# section and the one that says the bar is round: core shadow, then a line of reflected light
# under it. Capped hard - the whole point of the round bar is one specular, and a bounce that
# gets near it is the two-lights fault this panel has been marked down for three rounds running.
# It is also scaled by how squarely the member's section lies to the lamp, because the crest it
# has to stay under is: see `bear` in :meth:`_draw_rail`. Between them the foot rail's bounce
# measures a tenth of its own crest and the head rail's corner legs six hundredths, where a flat
# 0.36 put those legs at four fifths. It reaches only the rows the section actually turns under
RAIL_GRAIN_DARK = 0.45  # ...and how much of it goes the other way. Brushed steel scatters more
# than it swallows, so its tooth is bright specks on a face rather than noise about a mean: a
# symmetric field measures as a grain overlay laid over the render, which is what it is
RAIL_PITS = 6  # pits drawn over each bar's box, of which a couple land on the bar...
RAIL_PIT_DEPTH = 0.5  # ...and how much of the face's light goes at the bottom of one
RAIL_SUN = 0.60  # what is left of the lamp at the far corner of the panel, as a fraction of
# what a bar under it gets. One lamp means one falloff: the bar this is judged against runs its
# top frame at a median of 116 and its bottom-right corner at 37, and a rail system of one
# brightness end to end is the flattest thing on a panel however well each bar is shaded. The
# field is :func:`material.glare` about PLATE_LAMP, the same one the plate itself is lit by
RAIL_SUN_SPEC = 0.30  # ...as a power, for the highlight. A face in the far corner is dimmer
# because less light reaches it; a *highlight* there is a picture of the lamp, and the lamp is
# the same lamp from anywhere on an eight-inch panel. Taking the full falloff off the ridge as
# well put our bottom-right corner at a chamfer of 101 over a face of 90 - a member with no
# section - where the reference holds 157 to 179 over a face of 85 in the same corner. Not zero:
# the ridge is a reflection of a lamp seen at a longer grazing angle out there, and a corner
# whose highlight is as hot as the one under the lamp says the lamp is everywhere
RAIL_SCRATCHES = 26  # hairlines drawn over each bar's box, of which seven or eight cross it.
# Wear on metal is an event and events are bright: a scuff takes the oxide off and what is under
# it catches the lamp. Eight of them over a whole mount was a clean bar with a mark on it
RAIL_SCRATCH_SPREAD = 12.0  # degrees either side of the bar's own run they wander - dragged
# along it, not across it, but never all at one angle: two dozen parallel hairlines are brushing,
# and brushing is already in the grain. A scuff is an event and events do not line up
RAIL_SCRATCH_LEN = (8.0, 52.0)  # reference px, shortest to longest: long enough to read as a
# scratch and never the whole bar
RAIL_SCRATCH = 0.52  # how pale a hairline shows where it crosses a bar - its own and the
# sheet's, because a scratch that stops at the rail is a scratch on a drawing
RAIL_END_WEAR = 0.13  # how much darker the last few pixels of a bar are, where a cut end rusts
RAIL_END_W = 3.0  # ...and how many pixels that is
RAIL_LIP = 0.78  # fraction of a ring's width that is the lit chamfer, for the collar and the
RAIL_BODY = 0.28  # dial bezels, which still bend the old three-band profile - see _rail_colour
RAIL_SHADOW = 230  # alpha of the soft shadow the rail casts onto whatever is behind it - its
# plate, the glass, the room. It was 150, which is a tint: under a bar in the reference the
# ground goes to a fifth of what it was, and that drop is most of what lifts a bar off a surface
SPINE_DROP = 21.5  # reference px up from the panel's bottom edge to the foot rail's centreline.
# It was 10, and the state light's cove runs rows 464 to 469 at 480 - the whole of that bar's lit
# half. Same fault as the head rail's and the same answer: the bar stands a pixel inboard of the
# cove, the cove lights the picture between the case and the bar, and the metal keeps its own
# colour. See HEAD_DROP, which carries the measurements
SPINE_W = 10.0  # ...and its section, which is a little over half a mount's. Ten and not nine
# so that the bar's top edge lands on a whole row: at nine the one blown pixel of its ridge fell
# across the join between two rows and read as a two-pixel band forty levels down. The whole
# member and its shadow live in the outer twenty-seven pixels of a 480-pixel frame - under six
# per cent of its height, and none of it over the middle of the picture. That budget is the
# entire argument for it: three critics in a row said the hardware has no spine and every
# bracket dead-ends in air, and one slim bar along the bottom is the cheapest sentence that
# answers them. It is drawn before every other member, so the monitor stands in front of it and
# the mounts land on it
HEAD_DROP = 21.5  # reference px down from the top edge to the head rail's centreline. It was
# 11.5, which put the bar's lit edge at 6.5 - hard against the border, and inside the cove the
# state light pools in. Two faults came out of that one number, and both were measured. The
# surround's crown sits at row 2 and the bar's at row 8, a five-pixel pair of full-strength
# speculars with a trough between: one lamp cannot light one edge twice, and the eye reads the
# pair as a single fat moulding rather than as a case with a rail inside it. And rows 10 to 15
# are the cove - HALO_CORE and HALO_FALLOFF - so the bar's whole face sat in a green wash at a
# third alpha, which took the metal to +25 of green bias and flattened its own fall to two
# levels over six rows. At 21.5 the bar's outer edge is at 16.5, a pixel clear of the cove, and
# its ridge is fifteen pixels from the surround's with a lit strip of picture between them.
#   The half is not a rounding: a bar whose edge lands on a whole row samples its roll at depths
# 0, 1, 2 and the angle that mirrors the lamp - a sine of 0.40 - falls between the first two, so
# the ridge came out seven levels under what the same section reaches on every 17 px bar on this
# panel, all of which have their edges on halves.
#   What the member buys: the module across the top used to be bolted to nothing at all, and
# three quarters of the top edge was bare picture with the module's corner fixings hanging in
# it. Now the module's legs come down POD_STEP and land on the bar's top edge instead of passing
# through it, which is a load path anybody can trace
HEAD_TURN = 34.0  # ...and how far down each side it turns before it stops. A bar that runs off
# the edge of the frame dies in the border's glow; one that turns the corner and ends is a
# gusset, and the two together are what say the top of this panel is a frame rather than a lid
HEAD_BEND = 9.0  # reference px of radius its formed corner is bent to. It used to be the frame's
# own radius offset inwards, so the gap to the panel's edge stayed constant round the turn; a bar
# standing further in than that radius cannot be concentric with it at all, so the corner is bent
# to a former of its own now - which is what a fabricated corner in bar actually is
HEAD_STEP = 15.0  # degrees of the corner's arc per straight segment of the centreline. The bend
# is a handful of pixels across, so this is already finer than a pixel; any smaller and the
# mitre solver is doing arithmetic nobody can see
SPINE_SADDLE = 1.6  # how much thicker the collar that grips it is than the bar itself...
SPINE_GRIP = 9.0  # ...and how far along it that collar reaches either side of the member. A
# clamp said in silhouette: the section swells where it grips and nowhere else, which needs no
# bolt, no plate and no second part to be read as one
BAR_SS = 4  # a bar's outline is filled this many times over and boxed down - its anti-aliasing
BOLT_R = 7.0  # a socket head, sunk through the rail wherever it turns
BOLT_FALL = 0.5  # how much of the lamp's falloff across the panel a head takes, as a power. Half
# of it: a bar is a matt face and dims with the light on it, but a head is turned and proud and
# keeps catching the lamp long after the sheet round it has gone down. At the full falloff the
# fixing in the bottom-right corner came out at 123 against the 203 it is being matched against.
RIB_N = 3  # stiffeners across the deep corner of a bracket
RIB_W = 4.0  # ...their section, in reference px
RIB_ALPHA = 0.36  # how much steel a rib lays over the plate. Translucent like the plate it
# stiffens, so the room keeps running behind it.
RIB_FACET = 0.45  # how far the two halves of a web's face are set either side of STEEL, towards
# the lit and the dark. A machined chamfer breaks into facets with a hard step at each break;
# it does not airbrush, and a four-pixel web with one flat band between two hairlines was a
# stroke with a highlight on it. Two facets is all the width there is room for, and they put a
# monotonic step down from the lit arris to the dark one - which is the section, in miniature
RIB_CREST = 105  # alpha of the one lit hairline along a rib's near edge. It was 150, against a
# plate half as dark as this one; on the plate as it is now that was a bright line in the corner
# of the panel furthest from the lamp
RIB_SHADOW = 120  # ...and of the shadow it drops. A web four pixels wide stands a fraction as
# proud as a bar does, so it may not drop a bar's shadow: at RAIL_SHADOW the three of them read
# as three dark scratches gouged across the corner rather than as stiffeners standing on it
PLATE_INK = (9, 12, 13)  # what a bracket's plate is washed with. It used to be SCREEN, which is
# the green-black the phosphor's own furniture is drawn in, and that is the whole of why the
# plate measured +32 of green bias against a rail at +8: a plate is *steel* seen through the
# tube's glass, so it takes the metal's neutrality and not the tube's colour. A shade to the blue
# for the same reason the steel is - cold metal beside warm phosphor is what tells them apart at
# a glance. Its luminance is within a level of SCREEN's, so nothing built on the plate restages.
PLATE_WASH = 0.66  # how far a bracket's plate is put towards PLATE_INK. Not opaque: a bracket you
# cannot see the room through is a bar, and this layout exists to stop having those. But dark -
# it was 0.34, and at that the plate came out at the same luminance as the bars bolted across it
# (87 against 88), so rail, gusset and plate fused into one grey mass and the corner under the
# instruments read as a light wedge. A plate is the thing a bar stands *off*: near black, with
# the room still moving in it, and the bar two or three to one against it.
PLATE_STEEL = 0.10  # ...and the grey stirred into that wash. A plate is gunmetal seen through
# the tube's glass rather than the glass alone, and a wash with no grey in it is a tint.
PLATE_GRAIN = 0.11  # how much the brushing shows through the wash, either way. More than it was,
# because on a plate this dark the grain is the one thing saying it is a sheet and not a shadow
PLATE_SCRATCHES = 70  # hairlines across the whole sheet, of which the plates keep about a sixth
PLATE_SCRATCH_ALPHA = 0.34  # ...and how deep the deepest of them cuts. Dark on the plate and
# pale on the bars (RAIL_SCRATCH), which is not two minds about it: a groove in a sheet lying in
# its own shadow is a dark line, and the same groove in stock standing up in the light shows the
# bare metal in its wall. Ten levels or so either way - a scratch, not a stripe
PLATE_PITS = 110  # pits across the whole sheet, of which the plates keep a couple of dozen...
PLATE_PIT_DEPTH = 0.42  # ...and how much of the little light a plate has goes at each
PLATE_LIGHT = 0.055  # how far towards STEEL_LIT the sheet is lifted where the lamp is nearest it.
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
# ---- the USB module, flush into the top-left corner ----
#
# The one corner of this panel with nothing in it, and the thing that belongs there is what is
# plugged into the box: it comes and goes, it belongs to the machine rather than to the session,
# and it must never be somewhere a glance has to hunt for. Built like the status pod and out of
# the same parts - a plate that runs off the panel's edges, a machined rail along the inner
# edges only, a bolt at every knee, and the name cut into the rail rather than printed on the
# green. A first pass as a tall boxed panel was rejected for spending the whole left-hand edge
# on rails; this is the pod's own depth and nothing more.
#
# It runs off the TOP and the LEFT edges with no steel between it and them, which is the whole
# difference between a chassis part and a box drawn on top of one. Its steel is on the two edges
# that meet the rest of the machine: the bottom, and the chamfered right-hand end.
USB_PAD = 10.0  # reference px inside its ends, before the first device and after the last
USB_GAP = 12.0  # ...and between two of them
USB_GLYPH = 19.0  # the square a category mark is cut in
USB_GLYPH_GAP = 3.0  # ...and the air under it, before the name
USB_TOP = 3.0  # from the panel's top edge down to the top of the glyph
USB_NAME_W = 92.0  # the most a name may take before it is cut short. A column is as wide as its
# name, so this is what stops one badly-behaved product string eating the whole corner
USB_MIN_W = 126.0  # the narrowest the flat ever gets, which the legend in its rail sets and not
# the contents: a pocket for USB DEVICES has to fit inside the module it names
USB_CLEAR = 10.0  # px of picture kept between the module's top corner and the pod's plate. The
# pod is measured at its WIDEST - both tags lit - so nothing up here moves when REC comes on
USB_EMPTY = "NO USB"  # what it says when there is nothing on the bus. It is always there: a
# module that vanished would be a part falling off the machine every time a cable came out
USB_LEGEND = "USB DEVICES"  # cut into the rail along its bottom, the way CYCLOPS is cut into the
# pod's. Not a label on the green - a label on the green is a sticker, and this is a nameplate

POD_H = 50.0  # its depth, which is one row of readout and the rail under it
POD_STEP = 16.0  # the square drop off the top edge before the splay starts
POD_PAD = 16.0  # inside the flat, either side of the readouts
POD_STOP = 12.0  # between the meter, the tags and the clock...
POD_TAG_GAP = 8.0  # ...and the tighter one between two tags, which read as one group
# What the pod's rail frames is a window, not a plate: the readouts are lamps behind glass, and
# the glass is the terminal's own (TERM_ALPHA, TERM_SCAN) let into a narrow flange of the rail's
# steel. The flange is what makes it an instrument fitted into the frame rather than a hole in
# it, and the reveal is what makes the glass sit *below* the steel.
POD_LAND = 4.0  # reference px of flange inside the rail, from the rail's foot to the glass.
# It was three visible pixels and a hairline chamfer at their edge, which is a line and not a
# face: nothing to grade along, nothing to brush, nowhere to put a mark. Then it was seven, and
# seven is a *plate* - which cost the pane the room, and an instrument is read glass-first. The
# window used to be 32 px of a 58 px module, 57 %; a real one gives its glass better than 70 and
# leaves the metal a rail. Four is a face wide enough to roll, rake and cut a scale into, and
# every pixel taken off it is a pixel the window got.
POD_REVEAL = 1.0  # ...and how far inside the flange's edge the glass starts, so the steepest of
# the roll is steel and the glass meets it in a dark seam rather than on a bright line
POD_ROLL_TILT = 0.46  # sin of the tilt the cove has reached where the glass starts. The flange
# leans *away* from the window now, not into it: it descends into the recess, so the lamp reaches
# less of it the nearer the glass it gets and the darkest steel on the module is the pixel the
# pane sits against. It leaned the other way for two rounds and carried a blown line along its
# inner arris, which put a second specular three pixels above the rail's own crest - one steel
# flange firing two highlights of near-identical intensity across the whole width of the panel's
# most prominent chrome run. Three pixels of separation is not two members with any thickness in
# them, it is one edge drawn twice with an offset, and every whole-panel critic measured it.
# So there is one member here and one crest on it: the rail's. This face is its root.
POD_STOCK = 0.58  # how far the cove's steel is put from STEEL towards STEEL_LIT, which is where
# RAIL_STEEL puts the bar's own face. Same stock and the same finish, because it is the same bar
POD_BOUNCE = 0.15  # how much the pane gives back into the cove, hard against the glass, as a
# fraction of the light the cove has of its own...
POD_BOUNCE_W = 2.2  # ...and the px it reaches back over. A face turned from the lamp is allowed
# exactly this much and no highlight: it is a sixth of the lift the rail's crest carries, it is
# raked along the run with everything else on the module, and it never makes a second peak
POD_BOUNCE_LIT = (5, 13, 8)  # ...and what colour that light is. The bounce above says how much
# of the pane the cove gets back; this says that a green window lights the steel beside it green,
# which every critic of this module has measured us not doing: ours came back +1.1 green-excess
# hard against the glass and +0.9 ten rows down - dead neutral, which is a window painted on a
# plate rather than one lit behind it. Held to a measured peak of +6 and gone in five pixels,
# because metal that reads green is a worse fault than metal with no bounce at all
POD_BOUNCE_REACH = 5.0  # px of cove the colour carries over: the whole run from the glass to the
# rail's crest, which is where this face ends and the bar's own begins
# The rake. A drift across the module was a linear ramp of a quarter, which measured as a face
# falling eleven levels over a hundred and eighty pixels - a light with a *direction* and no
# position, which is the same flat every panel that has never been photographed has. The lamp
# stands off the module's left shoulder instead and its light falls off with the square of the
# distance, so the left end of every face on this module is half as bright again as the right,
# and the fall is a curve rather than a slope.
POD_RAKE_OUT = 54.0  # reference px outboard of the module's left knee the lamp stands...
POD_RAKE_UP = 44.0  # ...and above the panel's own top edge
POD_RAKE_REACH = 244.0  # ...how far its light carries, in reference px. Short enough that the
# right-hand knee gets about half of what the left one does
POD_RAKE_FLOOR = 0.26  # ...and what the far end still gets from the room, so nothing on the
# module goes to the black a hole would
POD_RAKE_CEIL = 1.06  # the most the near end may be lifted over the face's own shading, which
# is what keeps the left knee from blowing out to match the rest going dark
POD_WEAR_HOT = (250, 253, 250)  # what a hairline dragged across the cove goes towards: the
# lamp's own colour and not the steel's, because bare metal under a scratch mirrors the lamp
POD_GRAIN = 1.6  # of the material's brushing, on the flange. A rolled edge catches the brush
# harder than a flat does, and a face four pixels deep has to say metal in four rows
POD_SCRATCHES = 34  # hairlines dragged across the pod's own box, of which the flange keeps the
# few that cross it - on top of the sheet's own, which run onto it from the plates
POD_SCRATCH = 0.52  # ...and how pale the palest of them shows. Wear on steel is *bright*: a
# hairline is where the finish came off and the bare metal under it catches the lamp. Dark
# specks at the same rate read as a grain overlay laid over the render, which is what ours did
POD_PITS = 14  # single pixels over the pod's box where the finish has chipped...
POD_PIT = 0.55  # ...and how far towards black one goes. Few and shallow, against the hairlines
POD_TICK_PITCH = 14.0  # reference px between the graduations cut into the bottom flange, which
# is the meter's own segment pitch: the scale the bar beside it is read against. A blank pocket
# sat there before and two critics called it what it was - a groove that holds nothing. There is
# no word this panel needs that it is not already saying, so the mark it carries is a machined
# scale rather than a name, and a scale is legible at three pixels where a name is not
POD_TICK_W = 1.0  # px of the cut...
POD_TICK_A = 0.72  # ...how far into the dark its floor goes...
POD_TICK_LIP = 0.28  # ...and how much brighter than the steel it is cut into the wall beyond it
# comes back, which is the one lit line any groove has and it is on the side away from the lamp.
# A fraction of the face's own light rather than a step towards STEEL_LIT: a wall lit to one
# number is a highlight that does not know where on the module it stands, and on a cove that
# grades from 60 to 120 it was firing brighter than the face it was cut into
POD_TICK_LONG = 4  # every this many, a graduation that runs the whole face rather than half
POD_SEAM = 2.5  # px over which the glass eases from opaque at the reveal to the terminal's
# depth - the dark seam round any pane set in a frame, and the terminal's own bezel ease. The
# phosphor's cast and the room's reflection stop at it: it is the one part of the glass that
# is in the frame's shadow all the way round
POD_RECESS = 3.0  # px the glass sits below the flange, which sets the shadow the lip nearest
# the lamp drops onto it - with the seam, the whole of what says recess rather than decal
POD_SHADOW_A = 0.85  # how dark that shadow is where it is deepest
POD_AO = 13.0  # reference px in from its edge over which the recess's walls shade the glass,
# all the way round - the fall to near-black along the bottom of any pane set down in a frame,
# and what puts the readouts *in* the window rather than printed on it
POD_AO_A = 0.78  # ...and how dark it is right at the wall
POD_REBATE = 4.6  # reference px of that fall that are a *rebate* rather than a falloff: hard
# against the reveal, near black, and the same width all the way round. A pane whose frame meets
# it at one dark pixel is a decal however deep the shading behind it goes; the shape has to say
# that the bezel holds the glass, and the shape is a step down into a shadow you can measure
POD_REBATE_A = 0.90  # ...and how dark that step is, which is most of the way to a socket floor
POD_LIP_SHADE = 6.0  # reference px down from the panel's edge the border's shadow falls on the
# glass. The window is open at the top and runs up under the border, which makes the border its
# lip, and a lip drops a shadow on whatever is under it
POD_LIP_SHADE_A = 0.85  # ...and how dark, hard against the edge
POD_GLARE_A = 0.42  # how much of the room the glass gives back along its upper edge...
POD_GLARE_FLOOR = 0.07  # ...the little it gives back everywhere, because no glass goes dead
# black - and what lifts the pane off its own seam. Low: the segments are read against this,
# and a floor lifted by a wash is a floor the meter has to shout over
POD_GLARE_REACH = 0.62  # ...how far the lamp's light carries, in window widths, and...
POD_GLARE_STREAK = (0.20, 0.30, 0.10)  # ...the wipe it makes: where down the left edge it
# passes, how broad it is, how far it drops on its way across - all as fractions of the glass's
# own depth, not of the pod's. One soft blob at the top left, under the border's own shadow and
# falling away across the pane, rather than a band the whole width of the window: a reflection
# that covers a face evenly is a tint, and the lamp is in one place
POD_SWEEP_A = 0.44  # the cover glass's own reflection over that: one lobe with a *hard leading
# edge*, which is the half a gaussian blob cannot do. Glass is curved, so the room's edge lands
# on it as a line - the reference rises fifty levels in a single row and then trails off over
# ten - and a reflection with no edge anywhere in it reads as a wash however bright it is
POD_SWEEP_AT = 0.19  # where down the pane the lobe passes at its left edge...
POD_SWEEP_TILT = 0.16  # ...how far further down it has got by the right...
POD_SWEEP_LEAD = 0.035  # ...how sharply it comes on, as a fraction of the pane's depth...
POD_SWEEP_TRAIL = 0.34  # ...and how slowly it goes off. The asymmetry is the whole of it
POD_SWEEP_X = 0.17  # where across the module the lobe is brightest...
POD_SWEEP_WIDE = 0.30  # ...and how much of its width it covers. A reflection that runs the
# whole length of a pane is the lamp smeared into a band; this one is a *thing* the glass is
# giving back, and it has two ends
POD_SWEEP_LAMP = (255, 240, 205)  # ...and what colour it gives back. This lobe sits at the
# lamp's own end of the pane and is raked by it, so it is the *lamp* the glass is reflecting, and
# a workshop lamp is warm. Everything else on this pane returns the room, which is not: a critic
# measured our glass at B-R +1 to +15 and the reference at -9 to +31, and read the difference as
# a window reflecting one flat white instead of a room with two lights in it
POD_RETURN_A = 0.22  # the rim return: the pane's far edge gives the room back a second time,
# which is what says the glass has a thickness and the reveal is behind it
POD_RETURN_W = 2.2  # px of the far edge that does it
POD_FALL_A = 0.46  # how far into the dark the rake takes the pane at the end away from the
# lamp. The steel is raked by multiplication and the glass cannot be - a pane is mostly what is
# behind it - so the same lamp reaches it as a black laid over the far end, which is what turns
# a lit rectangle into a window with a top-left and a bottom-right
POD_TINT_A = 0.26  # the phosphor's own wash on the glass, at the lamp's end of it. It was a
# flat 0.07 everywhere, which is a tint; this is a field, brightest where the lamp is and gone
# by the far knee, so the pane has a top-left and a bottom-right like every other lit thing here
POD_SEPTUM_A = 0.88  # how far towards black the mask between two of the meter's windows takes
# the glass. A segmented meter is a dark plate with eight windows cut in it and the metal
# between two of them is a septum - ours had lit glass there, so with the skirt cut back to
# nothing the five pixels between two cells still came back at the wash's own 108, against a
# cell at 189: 43 %, and the count still could not be read from a pace. The reference's gaps
# measure *darker* than its own glass and this is the whole of why. Deep enough that the pair
# still separates once a lens or a pair of eyes has softened the panel by a pixel and a half,
# which is the distance the count actually has to survive
POCKET_W = 60.0  # reference px of the legend pocket milled into the rail under the window...
POCKET_H = 10.0  # ...how much of the rail's face it takes, which leaves two rows of face above
# it and one below rather than being a band across the bar...
POCKET_R = 2.5  # ...and the radius the slot drill leaves in its corners
POCKET_IN = 15.5  # px in from the pod's lower-right knee its right edge sits: off centre, where
# a service label is actually stuck, and clear of the bolt through the knee
POCKET_DROP = 0.5  # px below the rail's centreline its own centre sits
POCKET_DEEP = 2.0  # px the floor is cut below the face, which is what sets its shadow
POCKET_FLOOR = 0.26  # how far towards black that floor goes - about forty-five counts under
# this rail's face, which is what a milled pocket measures on the reference. Multiplicative, so
# the rail's own section, brushing and hairlines all carry through the floor: a pocket with a
# flat interior is a decal of a pocket lying on a bar
POCKET_WALL = 1.3  # px of wall round it...
POCKET_SHADE = 0.60  # ...how dark the two walls the lamp cannot reach are...
POCKET_LIP = 0.72  # ...and how much of the steel's own highlight the two it can reach give
# back. Shadow above and light below, which is the opposite way round from the bezel's chamfer
# outside it - the two recesses must not read as the same stroke drawn twice
POCKET_CAST = 0.35  # the shadow the near wall throws across the floor
POCKET_TOOTH = 0.11  # the mill's marks on the floor, running across the pocket where the rail's
# brushing runs along it: a cut face does not keep the finish of the face it was cut into
MARK_INSET = 5.5  # reference px in from each end of the pocket the engraved label starts. The
# pocket was milled and left empty, and an empty slot the exact shape of a nameplate reads as a
# missing part - a critic called it the loudest missing small thing on the module. What went in
# it first was a datum cross and a drilled index row, on the argument that this panel gave its
# words up on purpose; three critics measured that row as one glyph stamped nine times - adjacent
# cells correlating at r=0.47 - and read it as a font that had failed to render. A nameplate is
# not a word the panel is saying, it is the name of the thing the panel is part of, which is what
# every instrument in a workshop carries and the one legend nobody has to read twice
MARK_WORD = "CYCLOPS"  # what is cut into it. One word, and the machine's own
MARK_CAP = 6.2  # px of cap height, which is the pocket's floor less a row of wall at each end
MARK_WIDE = 0.68  # a capital's width as a fraction of that, before the row is stretched to the
# plate: caps this narrow are what a pantograph cuts, and they leave the counters of C, O and P
# open at a size where a filled typeface closes them up
MARK_STROKE = 1.35  # px the engraving cutter is wide. Single-line letterforms - the cutter walks
# the centre of each stroke and the width of the stroke *is* the width of the tool
MARK_ENAMEL = (236, 233, 224)  # what the cut is filled with. A cut left bare reads as printed:
# the winner's strokes stand ninety counts *over* their plate, not fifty under it, and the way a
# nameplate gets that is enamel rubbed into the engraving. Warm rather than white, and neutral
# enough that the plate is still steel: this is the only pale thing on the module that is paint
MARK_FILL = 0.54  # how much of it a fully cut pixel carries...
MARK_HARD = 0.60  # ...and the gamma on the coverage, which is the cutter's own wall: a milled
# channel has vertical sides, so a pixel half inside one is most of the way to full rather than
# half of it. Without this a stroke that straddles two columns loses half its contrast and
# the word goes grey
MARK_SHOULDER = 0.62  # how dark the lip of the cut on the lamp's own side goes...
MARK_SHOULDER_W = 0.95  # ...and the px of it that does. One line above every stroke, and none
# below: the shoulder the lamp cannot get down to is the only shadow a filled cut has left

# The alphabet the cutter walks, one polyline per stroke, in a box that runs 0..1 left to right
# and 0..1 from the cap line down to the baseline. Single-line letterforms, because that is what
# a pantograph makes and because at a cap height of six pixels a typeface's bowls fill in: what
# distinguishes C from O here is a counter three pixels tall, and a filled face has none.
MARK_ALPHABET = {
    "C": (((.95, .26), (.68, .0), (.30, .0), (.0, .30), (.0, .70), (.30, 1.), (.68, 1.),
           (.95, .74)),),
    "L": (((.0, .0), (.0, 1.), (.95, 1.)),),
    "O": (((.30, .0), (.65, .0), (.95, .30), (.95, .70), (.65, 1.), (.30, 1.), (.0, .70),
           (.0, .30), (.30, .0)),),
    "P": (((.0, 1.), (.0, .0), (.62, .0), (.95, .25), (.62, .50), (.0, .50)),),
    "S": (((.95, .24), (.66, .0), (.28, .0), (.0, .22), (.14, .45), (.81, .57), (.95, .78),
           (.66, 1.), (.28, 1.), (.0, .78)),),
    "Y": (((.0, .0), (.48, .46)), ((.95, .0), (.48, .46)), ((.48, .46), (.48, 1.))),
    # ...and the rest of what the USB module's rail needs cut into it. Same hand: one stroke per
    # run, counters left open, nothing a six-pixel cap would close up.
    "B": (((.0, .0), (.0, 1.)), ((.0, .0), (.62, .0), (.88, .24), (.62, .48), (.0, .48)),
          ((.0, .48), (.68, .48), (.95, .74), (.68, 1.), (.0, 1.))),
    "D": (((.0, .0), (.0, 1.), (.58, 1.), (.95, .68), (.95, .32), (.58, .0), (.0, .0)),),
    "E": (((.95, .0), (.0, .0), (.0, 1.), (.95, 1.)), ((.0, .50), (.72, .50))),
    "I": (((.48, .0), (.48, 1.)),),
    "N": (((.0, 1.), (.0, .0), (.95, 1.), (.95, .0)),),
    "U": (((.0, .0), (.0, .70), (.30, 1.), (.65, 1.), (.95, .70), (.95, .0)),),
    "V": (((.0, .0), (.48, 1.)), ((.95, .0), (.48, 1.))),
    " ": (),
}

BOT_L = 170.0  # the bottom-left bracket's reach along both edges - BOT_R_OUT's, so the two mirror
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
TERM_FOOT = 4.0  # how far the case stands off the panel's own bottom edge. It sat on it while
# it was a slab bolted between the two mounts, where the edge was one of the things holding it;
# a monitor on its own brackets is a thing with air all the way round, and the border's glow
# running under it is what says so. Two px of that air paid for the wider moulding: the glass is
# solved from the bottom up (see the layout in `Overlay.__init__`), so a moulding two px deeper
# would otherwise have carried the whole screen and the line printed on it up the panel with it
# - and the eye's own table of where the panel's landmarks are (eye.LANDMARKS) says where the
# caption is. The case grew two px each way instead, and the line's own rows did not move.
TERM_W = 352.0  # the case's width, reference px, and it stands in the middle of the panel. It used
# to be solved from the two rails, which was off-centre the moment the two mounts stopped being
# the same size and moved every time either one did. It shrinks, rather than cross TERM_CLEAR,
# on a window too narrow for it.
TERM_CLEAR = 7.0  # how far the monitor's case stands clear of each mount's rail, so that
# all four of its corners are its own. It used to run from the middle of one mount's bottom rail
# to the middle of the other's, buried at both ends for the lower half of its depth - which read
# as bolted in, and which cost the two ends of the shape. A monitor is a thing you can see the
# whole of; buried ends make it a slot again however round its corners are. Two px of that gap
# went the same way the two below the case did (see TERM_FOOT): the case is solved outward from
# its own rails, so a moulding two px wider would otherwise have taken them out of the glass,
# and the glass is where the caption is measured from - and where the eye's own table of the
# panel's landmarks says it is. The screen is the pixels it always was; the frame round it grew.
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
TERM_BEZEL = 9.0  # the frame round the glass: a bar of the panel's steel bent round the tube
# and rolled at both edges - down to the panel on the outside, down into the recess on the
# inside - lit by the one lamp like every other bar here, so the top rail's outer edge is bright
# and its inner edge dark, and the bottom rail the other way about. For a long time nothing drew
# an edge between case and glass at all, because a bright *green* lip round the aperture had
# read as a glowing pill; that was the phosphor's fault and not the ring's. Grey steel is a
# frame, and a frame is what says the glass is set down into something rather than painted on.
# Seven px could not carry a section. The reference spends eleven on one - a px of dark reveal,
# two of crest, a groove, two of secondary ridge and four of dark inner wall - and at seven ours
# had to choose between a face and a terminator: two px of wall, which is a line and not a
# thickness. Nine buys the fourth plane back. The case grows UPWARD for it, because the bottom
# of this thing is pinned to the panel's own edge (see TERM_FOOT) and the glass inside it is
# solved from two lines of type, so nothing that has to be read moves.
TERM_PAD = 3.0  # inside the glass, above the first line and below the last. It was 6.0 while the
# glass ran to the chassis edges and the text had nothing but its own padding holding it off
# them; the moulding is that separation now.
TERM_RADIUS = 22.0  # the corner of the front. One radius and not two: the case and the glass are
# the same shape at different depths into the same field, so there is one fillet to turn and no
# pair of concentric ones to keep from drifting apart.
TERM_ROLL = 2.6  # reference px of the frame's outer edge that turns down to the panel, and the
# one number this round's whole argument about the frame rests on. Both edges are rolled now
# (see TERM_REVEAL for the inner one), and the lamp decides which of them is bright rather than
# the code: a ring standing over a recess under one overhead light is lit on its two UP-facing
# surfaces, which are the top rail's outer roll and the bottom rail's inner one. Lighting only
# the outer edge got the top rail right and left the bottom rail a soft dark band no brighter
# than the glass beside it - the readout lost its lower edge and collapsed into a lozenge at a
# pace away. On the reference the bottom rail's inner chamfer is the brightest pixel of the
# whole element, brighter than the top rail's crest by a third.
TERM_TURN = 1.5  # how far past end-on each rolled edge carries, as a fraction of a quarter
# turn. Over one on purpose: a round-over runs on down onto the panel, so its outermost pixel
# has turned AWAY from the lamp and the crest sits a fraction inside the silhouette - which is
# what lets it wander with TERM_WOBBLE instead of being pinned to the outermost row of the part.
TERM_QUIRK = 0.36  # how much of the light the shallow groove behind each rolled edge
# loses to the bead standing over it...
TERM_QUIRK_AT = 0.75  # ...and how far past that roll its middle sits, in reference px. This is
# what makes the bar two machined surfaces instead of one gradient: a bead, a shadow groove,
# then the face. Read across, the reference's top rail goes 24 / 127 / 152 / 76 / 93 / 97 / 75
# / 24 - a crest, a dip, and a second broader lobe - where ours ran one monotonic ramp from the
# crest to the seam, which is a painted gradient with a stroke at each end.
TERM_CROWN = 0.32  # of a quarter turn: how far the face itself has come round by its own two
# edges, from flat in its middle. Nothing rolled is dead flat, and this is the ONE term that
# shades the moulding across its section rather than along its length. At a tenth of a turn the
# face was flat enough that the quirk groove was the only thing on it, so the bottom rail read
# 160, 84, 59, 70, 79, 35 outward - a second bright stroke sitting on the far edge of a bar lit
# from above, which is the loudest tell of a shape that was drawn instead of lit. A face this
# crowned falls monotonically from the edge the lamp is on to the one it is not, the way the
# reference's does (a 2 px crest, then 38 counts down the face, then the far edge darkest).
TERM_WOBBLE = 0.38  # reference px the whole cross-section wanders in and out along the length
# of the bar. Rolled stock is not one thickness, and this is the term that takes the specular OFF
# a single row: without it the crest's brightest pixel sat on the same row in 24 of 26 columns,
# which is the signature of a gradient extruded along a shape rather than of a piece of metal.
TERM_LIFT = 2.0  # how proud the frame stands of the panel behind it, which sets the shadow it
# drops - down and to the right, away from the lamp, like every rail's. Less than a rail's: it
# is a frame let into a bay, not a spine standing on a plate.
TERM_SHADOW = 0.45  # ...and how dark that shadow is where it is deepest
TERM_CONTACT = 0.55  # ...and how dark the band hard against its outer edge is. Where a part
# meets a plate the room cannot get in, and that band is what draws the one dark contour round
# the whole frame. Without it the top rail's specular is the outermost thing on the part, and a
# bright line with nothing behind it reads as a stroke round a rectangle rather than as a lit edge.
TERM_CONTACT_W = 2.0  # ...and how wide it is, in reference px
TERM_GLOSS = 0.26  # how much of the steel's specular the frame keeps. It was an eighth, on the
# argument that a full-strength highlight put STEEL_SPEC along BOTH edges of the bar and made an
# emboss of it. The argument was right and the cure was wrong: an emboss is a bar lit at its top
# outer edge and its bottom outer edge. Lit at the top OUTER and the bottom INNER, which is what
# the rolled inner edge now gives, it is a ring standing over a hole under one lamp - exactly
# what the reference measures - and turning the highlight down to an eighth only meant the whole
# element held 27 near-white pixels against the reference's 640.
TERM_FACE = 1.62  # the gain on the light reaching the frame at all, at the end of the bar the
# lamp is nearest...
TERM_REACH = 3.0  # ...and how far that light carries, in frame heights, from the point the
# glass reflects it from (GLARE_X, GLARE_Y). One inverse-square fall from one lamp, and it
# replaces the pair of ramps that used to shade this part top-to-bottom and left-to-right. A
# bench lamp off the top-left corner is nearly as far from the bottom rail as from the top one
# and three times as far from the right end as from the left, so the fall along the LENGTH is
# the big one; the old ramps had it backwards - 1.86 top to bottom and 0.94 left to right, which
# is a dome overhead, and steel under a dome reads as painted plastic.
TERM_FALL_FLOOR = 0.36  # what is left of the lamp at the far end of the bar - the room, which
# is the only reason the right-hand end is not black
TERM_ARRIS = 0.92  # how much of the lamp the polished arris of a rolled edge gives back where
# it is turned squarely end-on to it. This is the one thing on the frame allowed above
# material.STEEL_SPEC, which is the ceiling on lit STEEL and not on a mirror: the reference's
# ridge runs to 231 where our whole palette stopped at 212, and every critic this round asked
# for a blown ridge at 240-255 against ours ceilinged at 209-214.
TERM_ARRIS_TIGHT = 1.6  # ...and how tight it is. High: an arris is one pixel of a bar, and a
# blown line two or three wide is a stroke round a rectangle again.
TERM_BOUNCE = 0.30  # what a face turned AWAY from the lamp keeps of the light on the face turned
# towards it. ONE MEMBER, ONE SPECULAR: this moulding is a single ring under a single lamp
# standing above it, so the only surface on it that can mirror the source is the top rail's outer
# roll. The bottom rail's inner roll is turned up into the well, and what reaches it is a bounce
# off the lit glass and the panel - a fraction of the lamp, arriving second-hand. Left at full
# strength the ring carried two speculars of equal brightness which did not even share a falloff:
# 224 along the top rim against 223 along the bottom across x=290-590, and past x=550 the bottom
# was 1.3x the brighter of the two. That is what one part lit twice, from opposite sides, by two
# people who never met, looks like - and a critic reading the whole panel picked it out of every
# element on it. A bounce also borrows its LENGTHWISE run from the face it bounces off (see
# `crest` in :func:`tube_frame`), because a reflection cannot outlive its own source.
TERM_ARRIS_FALL = 0.34  # what the arris keeps at the far end of the bar. Much more than the
# face does, and the reference says so plainly: its top crest falls 231 to 144 down the length
# while its bottom ridge only goes 210 to 199. A mirror returns the source wherever it can see
# it; it is the scattered light that runs out with the distance.
TERM_WELL = 0.11  # how much light the inner half of the frame loses to the well it looks into...
TERM_WELL_FROM = 0.20  # ...and how far across the bar that starts, as a fraction of its width.
# A frame stands over a hole: the nearer its inner edge, the less of the room any part of it
# can see, so it darkens all the way in. This is ambient and nothing else, which is why it is
# now half what it was - it is symmetric about the section, and on the bottom rail it was
# pulling down the one chamfer the lamp is actually on. The grade across the moulding belongs
# to TERM_CROWN, which knows which way the lamp is; this only ever fills the corner between the
# frame and the pane's own wall shade (TUBE_SIDE_A), so the two meet without a step.
TERM_REVEAL = 3.4  # px of the frame's inner edge that turn down into the well. Rolled, like the
# outer edge, and the shading is left to the lamp instead of being taken off by hand: on the top
# rail this edge faces down into the recess and goes black, which is the seam that says the
# glass sits behind the frame; on the bottom rail the same edge faces up out of it and takes the
# brightest specular on the part. One geometry, two opposite results, because there is one lamp.
# It is nearly twice as wide as it was, and that width is the whole of what gives a 7 px moulding
# the thickness of a 13 px one: the reference's dark inner wall is four px at 0.36 of its glass
# field, ours was two at 0.50, and a frame whose aperture ends on a two-pixel line is a frame
# read as a hairline from a pace away however well its crest is lit.
TERM_DRIFT = 0.9  # how far the face's own brightness wanders along the length of the bar, in
# units of material.GRAIN - so about a tenth either way, slowly. Brushed grain alone is fine
# noise and averages out over any patch big enough to look at: measured along this face it is
# a standard deviation of 2.5 levels against the reference's 5.8, which is why a frame with
# grain on it still reads as one flat grey. Rolled stock is not one thickness and handled
# stock is not one polish, and this is the term that says so.
TERM_WEAR = 0.45  # how much the frame's highlight comes and goes along its length - handled
# steel is polished where hands have been and dull between
TERM_SCRATCHES = 26  # hairlines over the frame's own box, of which the frame keeps the few that
# cross it - a scratch that stops at an edge is a scratch on a drawing
TERM_SCRATCH = 0.38  # ...and how pale the palest of them shows, as a fraction of the light
# already on that part of the face. A scratch is metal turned, not a decal laid over one: it has
# to go dark at the dark end of the bar instead of crossing the whole part at one brightness.
TERM_PITS = 6  # dark specks in the frame's steel, from whatever has been dropped on it...
TERM_PIT_A = 0.18  # ...and how much of the light one takes away. Four times as many and
# half again as deep before this, and between them they made the wear on this part symmetric -
# 1.8% of the face brighter than its neighbours and 1.7% darker, which the eye reads as a grain
# overlay dropped on top rather than as damage. Wear on handled steel is bright: the reference
# runs 2.98% bright outliers against 0.17% dark, because what a workshop does to a bar is polish
# it and scratch it.
TERM_INK_CEIL = 296  # the sum the glass may approach, and never reach, over the rows the line
# prints on. The reader that finds the caption on this panel finds it by summing the channels
# past 300 (tests/test_caption.py), so anything the pane gives back above that IS a letter as
# far as this machine is concerned, and a reflection standing in for a letter is a word lost.
TERM_INK_KNEE = 274  # ...and where the pane starts being held back towards it. A hard clip at
# the ceiling is worse than the glare it prevents: it lays a plateau of one exact value across
# whatever it catches, which is the posterisation this pane was pulled up for in the first
# place. Compressed into the ceiling instead, every level below it stays a level of its own.
# It sat 24 lower, and low enough that the pane's own brightest quarter lived inside the
# compression rather than under it: half the top-left of the glass came out at 97-102 whatever
# was actually computed there. The knee is a safety net now and not a shaper - the field is
# built to pass under it (see GLARE_HALO), and what this catches is the skirt of the one
# specular and nothing else.
TUBE_TOP = (33, 70, 45)  # the phosphor at the top of the glass...
TUBE_BOTTOM = (19, 40, 26)  # ...and at the bottom of it. Opaque, both: the module docstring has
# always listed this glass among the parts you cannot see through, and it was only 80% of the way
# there - close enough that a contrast stretch of the pane found the workshop, a finger and a
# clamp knob sitting behind the caption. A screen you can see the room through is a hole. The
# fall from the top colour to the bottom one is the curve: a tube read from a little above shows
# the lit room in its top half and the floor in its bottom, and that alone is most of what says
# the face is not flat.
TUBE_RAKE = 0.52  # how far the vertical fall leans as it crosses the pane, in fractions of the
# whole fall per pane width. It is what stops every row of the phosphor being one value: the
# tube is lit from up and to the LEFT, so the same brightness is reached higher up the glass at
# the right-hand end than at the left, and the iso lines run diagonally as they do on any lit
# cylinder. Leaning the same way the lamp falls rather than against it matters - the pane's
# tile ladder has to keep decreasing in both axes with no local maximum outside the corner the
# lamp is over, which is the one thing this glass has been winning on.
TUBE_GLOW_A = 0.07  # the pod's window keeps the flat wash this glass used to have, and
# TERM_ALPHA and TERM_SCAN with it: that pane is a slot with lit readouts behind it rather than
# a tube, it is a fifth the size, and none of the modelling below would read at that scale.
TUBE_LIT_A = 0.34  # how much brighter the glass is where the lamp stands in it. It was two
# thirds again as much and half as wide, which had the *phosphor* shading 60 at the left of the
# pane to 37 at the right - the room's lamp lighting the inside of a tube, backwards. An emissive
# face is flat (the reference's measures 49 / 48 / 46 across) and all the direction on it belongs
# to what its front surface REFLECTS: the corner blowout and the lit rim, which are SHEEN_* below.
# What is left here is the lift the lamp's own haze puts in the near corner of a picture tube.
# It is a smaller lift than it was and a much tighter one (see TUBE_LIT_Y and TUBE_LIT_H), and
# the two changes are one change: it used to be a cloud half again taller than the pane, so it
# lifted the rows the caption prints on as much as the rows above them and the field came out
# FLAT for fifteen rows under the first line. Pulled up under the lip it grades again, and the
# glass under the type falls from 93 to 71 - which is where the line's contrast against it came
# from, since the phosphor the letters are made of cannot get any brighter than GREEN.
TUBE_LIT_X = 0.16  # ...and where that is, in case widths across...
TUBE_LIT_Y = 0.15  # ...and case heights down: the upper-left quarter, the same lamp the frame
# is lit by. A pane with a bright quarter is lit; a pane at one brightness with marks on it is a
# texture, and that is what this was - flat 58 the whole width with two hairlines drawn on it.
TUBE_LIT_W = 0.40  # how far that lift carries across the glass...
TUBE_LIT_H = 0.52  # ...and how far down it, both as fractions of the case
TUBE_LIT_FALL = 1.3  # how the lift falls away from its middle. A gaussian is 2 here, and a
# gaussian is the wrong shape for this: flat across its own peak and then gone, which spreads
# one soft cloud evenly over half the pane. Under one puts a definite bright patch where the
# lamp is and still leaves a long tail across the rest, which is what a photograph of a lit
# screen has - a corner you can point at, and no edge anywhere.
TUBE_GRAIN = 0.055  # the fine tooth under everything else on the pane, either way, as a
# fraction of the light there - a couple of levels. A tube's face is grains of phosphor behind a
# shadow mask and is never a fill: masked of its text, ours held 218 distinct luminances in a
# 190x40 patch and ran 36 identical pixels along the average row, and at y=435 it was literally
# constant for 38 columns. The reference holds 1085 values and never runs flat past five. This
# is the cheapest level of the whole element and one of the two that decide whether the pane
# reads as glass or as a painted field.
TUBE_SPECK = 0.45  # of that tooth again, per channel rather than shared: the phosphor
# triads themselves, which is why a photograph of a tube holds a thousand distinct values in
# a patch where a fill holds two hundred. Small enough that it never tints anything.
TUBE_SCAN = 0.13  # how much darker every third row of the phosphor is: the tube's own raster,
# on the panel filter's pitch and in its phase (see *top*), so the two reinforce and never beat
TUBE_INSET = 3.0  # px in from the frame's inner lip where the room's reflections stop. Every
# glare mark is cut to this rather than to the glass itself, because a hairline that runs off the
# pane and onto the moulding is a scratch on the drawing and not on the screen - two of them
# crossed the top rail at the same brightness as the ones on the glass before this was here.
# The rebate. The glass is not a fill inside a frame, it is a pane sitting at the bottom of a
# well, and the whole of what says so is the band round its inside edge. Each wall of that well
# gets the SAME curve - a contact shadow that is nearly black on the boundary pixel, deepest one
# or two px in where the wall shades itself, and then a monotonic recovery to the pane's own
# light - and they differ only in how deep and how far. That is why the numbers below come in
# threes: how far in the trough sits, how wide it is, and how much of the glass's light it takes.
# It is applied as a GAIN on the finished pane rather than as black laid over it, because a
# shadow is multiplicative: the same curve has to fall across the phosphor, the room's wipe and
# the corner blowout alike, and an opaque floor painted at a fixed level is a stripe that nothing
# is casting. Measured against a 45-level field, the reference runs 19, 13, 18, 23, 28, 33, 38,
# 39 inward from its right-hand wall and 27, 14, 11, 19, 28, 34, 40 up from its bottom one; ours
# ran 64, 94, 90, 56, 45, 44 - a BRIGHT stroke where a recess needs its darkest line.
TUBE_LIP = 3.5  # px below the top lip where the shadow it casts down the pane is deepest. Not
# on the lip: the first two or three px of glass are the one place the pane gives back the
# frame's own lit inner face (SHEEN_STRIP), and the shadow of an overhang starts BELOW what it
# is reflected in. The reference reads 50, 64, 61, 57, 43, 39, 48 down from its top lip - a lift
# to a third over field, then the trough, then the field - and ours read 55, 66, 62, 55, 64, 63
# because the shadow was sitting on top of the reflection and the two cancelled.
TUBE_LIP_A = 0.46  # ...and how much of the pane's light it takes there. Still the shallowest of
# the four, because this one is a cast shadow and not an occlusion - the lamp is wide and it
# fills - but it has to be a trough somebody can see: lift, trough, field going down the top of
# the pane is the whole recess signature, and at 0.28 the three read 78 / 88 / 80 down a column,
# which is a ripple. It is also what keeps the specular's own skirt off the first line of type.
TUBE_FEATHER = 1.7  # px the top trough takes to come out of its deepest, either side
TUBE_FOOT = 1.1  # px above the bottom lip where the floor of the well is darkest...
TUBE_FOOT_W = 2.6  # ...how far that reaches up the pane...
TUBE_FOOT_A = 0.62  # ...and how much it takes. The deepest of the four and the narrowest: the
# bottom of a well under a lamp from above is simply where no light gets, and the reference ramps
# its last four rows to a quarter of field (33 -> 12) where ours BRIGHTENED into its own edge.
TUBE_SIDE_AT = 1.5  # px in from each end wall where its contact shadow is deepest...
TUBE_SIDE = 3.6  # ...and how far it takes to recover, so an end reads as a wall and not a line
TUBE_SIDE_A = 0.72  # ...and how much of the pane's light it takes at its deepest
TUBE_WALL = 0.10  # the one band that goes the other way: the polished end wall bouncing the
# lamp back onto the pane, which is what the reference has and a flat field does not - its
# column means lift 44 -> 48 over the last 20 px before the frame where ours were dead flat.
TUBE_WALL_AT = 15.0  # ...how far in from each end it is brightest...
TUBE_WALL_W = 9.0  # ...and how broad. Wide and low: a wall this size is a dull mirror, and what
# it returns is a lift you can measure rather than a stroke you can see.
TUBE_BLEND = 4.0  # px over which one wall's profile gives way to the next one's. Switched hard
# on whichever wall is nearest and the diagonal through each corner is a visible seam between
# two different shadows; blended, a corner carries both and simply goes darker, which is what
# the corner of a well does.
# The room, and where it is coming from. Everything below is the one lamp :data:`material.LAMP`
# points at: a white source up and to the left of the panel, which is where a bench light is and
# where anybody reads a highlight from without having to be told. Two terms make its reflection
# in the glass - how much of its light reaches a point at all, which falls away with distance
# from the source, and the streak it draws down the face, which is what a long glossy surface
# does with a small bright thing. The frame takes none of this: it is steel, and steel is lit
# by its own normals under the same lamp, so a reflection painted over it would be two lamps.
GLARE_X = 0.02  # the source, in face widths across...
GLARE_Y = -0.16  # ...and in face heights down, so it sits just off the top-left corner
GLARE_REACH = 2.4  # how far its light carries, in face heights
GLARE_ALPHA = 0.10  # how much of the tube's white the glass gives back where it lands hardest.
# Held down by the line printed on it as much as by taste: a reflection over a word is a word
# lost, and anything on the glass past a brightness is read as ink (tests/test_caption.py).
GLARE_AT = 0.42  # where down the left-hand edge the streak passes...
GLARE_DEPTH = 0.12  # ...how broad it is either side of that...
GLARE_TILT = 1.5  # ...and how far down the face its middle travels on its way across. It
# crosses the whole depth in two thirds of the width, which is steeper than a letterbox would
# seem to allow, because a streak that lay along the glass read as one more scanline. It starts
# a quarter of the way down rather than in the corner: the corner is where the phosphor is
# already brightest, and a white wipe laid over that is the one place on this pane where glare
# could put a word out.
GLARE_AMBIENT = 0.05  # the floor under the wipe, for the corner furthest from the lamp. It was
# going dead black without it, which no glass in a room does - low now, because the pane has its
# own gradient to be lit by and a flat white wash over that is what flattened it before.
GLARE_HALO = 0.11  # how much of the lamp the pane scatters round the corner it is nearest -
# the soft wide bloom under the hard specular, which is what a cover with any depth to it does
# with a bright source. A pane with a specular and no bloom under it is a sticker on a flat
# field. It was near three times this, doing the job the phosphor's own lift does now, and that
# was the whole of why the upper-left third of the glass came out FLAT: white is expensive in
# the one currency this pane has to spend, which is the sum of the three channels the panel's
# own reader takes for a letter past 300 (see TERM_INK_CEIL). Twenty-six rows of it sat pinned
# to the ceiling at 97-102 with nothing to tell them apart. Green buys a fifth more luminance
# per unit of that sum than white does - the same reason the lift in :func:`tube_glow` is the
# phosphor turned up rather than a wash laid over it - so the lifting is green now and the white
# is spent where it is worth most, on the one specular above the first line.
GLARE_HALO_W = 0.22  # ...how far it carries across the face, in face widths...
GLARE_HALO_H = 0.72  # ...and down it, in face heights. An ellipse half again as wide as it is
# tall, not a disc: this face is a letterbox cut out of a tube, so it is much straighter
# across its width than down its depth, and the reflection of a small source is drawn out
# along the axis a surface is straight on and pulled in along the one it is bent on.
GLARE_MARKS = 11  # fine hairlines polished into the glass, clustered mid-pane, which only show
# where the lamp reaches them - at arm's length the glare on a real screen is mostly these
GLARE_MARK_TILT = -45.0  # degrees from horizontal the polishing ran at: up and to the right
GLARE_MARK_SPREAD = 8.0  # ...and how far off that any one line strays, in degrees
GLARE_MARK_A = 0.22  # how much white the palest of them adds where the lamp is full on it
GLARE_MARK_BOX = (0.22, 0.28, 0.30, 0.42)  # the patch of the face they are clustered on - left,
# top, width, height, as fractions of the case. Mid-pane and towards the lamp, because a mark the
# light does not reach is not there at all, and one over the far end of the line is over a word.
GLARE_MARK_LEN = (7.0, 22.0)  # ...and how long they run, in reference px. Short and few: a
# polishing mark is a flick of a cloth, and thirty-six of them at full length was a screen door.
GLARE_MARK_BLUR = 0.7  # px they are softened by. A hairline drawn a pixel wide and left sharp
# is aliasing at 800x480; a scratch in glass seen through the glass is never that crisp.
GLARE_LAMP = (234, 238, 235)  # the lamp itself, as the glass gives it back. Off-white and 2%
# saturated, a step above material.STEEL_SPEC, because a mirror returns the source where metal
# only scatters it - and undyed, which is the whole point: our brightest non-text glass pixel
# was L 99 at (70,103,82), still 32% green, so the pane never returned a light source at all and
# read as matte paint over a lit field. The reference's peaks at L 234 on (230,236,231). It is
# what the frame's own arris mirrors as well as the glass, so its green bias is a bias on METAL:
# held to +3.5 on G - (R+B)/2, where the panel's rule for a metal surface is +7. A lamp has no
# colour to lend anyway - what green there is on this element belongs to the phosphor.
SHEEN_IN = 1.0  # px in from the pane's own edge that its reflected rim runs - on the front
# surface, and hard against the lip, because it is the frame's own lit inner face the glass is
# giving back and the shadow of that same lip starts BELOW it (see TUBE_LIP).
SHEEN_W = 1.8  # ...and how wide it is, in px to where it has fallen to a third. A band of two
# or three px and not a hairline: the reference's is four px (49, 65, 61, 57 against a 45 field),
# and a one-px stroke on the boundary of an aperture reads as a drawn outline rather than as a
# reflection. It is also what keeps the hot end of it clear of the two rows the line is printed
# on - a specular over a word is a word lost, and past a brightness the panel's own reader takes
# one for a letter (tests/test_caption.py).
SHEEN_LIT = (0.50, 1.0)  # weight of the rim band and of the specular. A cover square to the
# viewer in its middle mirrors the dark room and gives nothing back; only where it has turned
# does it return anything, and only the rim that has turned TOWARDS the lamp is bright. The band
# is the frame's lit inner face reflected in the pane and runs the whole length of it; the
# specular is the LAMP, and a lamp is a shape and not a stripe - so the specular is the brighter
# of the two and it is the only place on this glass with an outline.
SHEEN_SPEC_X = 0.056  # where the lamp's own image sits, in pane widths across...
SHEEN_SPEC_AT = 1.1  # ...and in px below the frame's inner lip. High on the pane, and it has to
# be. The first row the panel's own caption reader looks at is five px under the lip, and past a
# brightness it takes a reflection for a letter (tests/test_caption.py) - so the strip above the
# first line is the only place on this glass where a light source may actually blow out. Below
# it the pane is held under TERM_INK_CEIL, which is a ceiling of L 98 on anything white.
SHEEN_SPEC_W = 0.031  # how far it carries across the pane, to a third...
SHEEN_SPEC_H = 2.4  # ...and down it, in px. Wide and shallow: a bench lamp seen in a cylinder
# is stretched along the curve, and the pane's curve runs the long way. It replaces a second rim
# running down the pane's left edge - two rims meeting in a corner is a corner that is bright,
# which is not the same thing as a light source being in the glass. The critic measuring this
# pane found it "flat to +/-2 lum over 20x22 px, one linear gradient, tinted vector fill", and a
# reflection with no shape of its own is exactly what that reads as.
SHEEN_A = 0.98  # how much of GLARE_LAMP the pane gives back where the specular sits on the rim,
# which is the corner nearest the lamp. This is the one blown specular on glass on the whole
# panel and it wants to be blown: near white, high on the pane, and gone within four pixels.
SHEEN_STRIP = 0.11  # what the rim keeps where the lamp does not reach it - the lit inner face
# of the frame, which the pane goes on giving back the whole length of its top edge. It is
# what puts the reference's discrete four-pixel strip (49, 65, 61, 57 against a 45 field) all
# the way along; ours measured 1.01 / 1.12 / 0.36 of its own field over the same rows, which is
# no strip at all - the frame's shadow was laid on the same three px and cancelled it. It is
# doubled now that TUBE_LIP has moved the shadow down off it, and it is the top half of the
# recess signature: a band a third over field, and then a trough under it.
SHEEN_REACH = 0.62  # how far the lamp carries along the rim, in pane heights. Short - the blowout
# is a corner and not a stripe - but long enough that the strip under the top lip is still a
# measurable 40% over the field at the far end, which is what the reference does.
# The eye. He rides the left bracket's ramp, sunk into it - `EYE_SEAT` is that depth as a
# fraction of the swell's radius, and acos(0.3) is a 72-degree shoulder, which is where the rail
# meets his collar on either side. Part of him is in the bracket and part is over the
# picture, which is the same join the tab row used to make and the reason he reads as part of the
# machine rather than as a badge stuck on it.
EYE_R = 0.1375  # 66 px at 800x480. It was 88, and took up too much of the screen
EYE_SHOULDER = 16.0  # reference px between his rim and his swell, the collar's outer edge
EYE_SEAT = 0.3  # less deep than half, so his rim keeps clear of the border's glow on both edges
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
LUMA = (0.299, 0.587, 0.114)  # what the panel is measured with, and what a ceiling is read in
STILL_CEILING = 206.0  # the brightest luminance anything that never moves may reach anywhere on
# his housing. The pupil peaks at 253 and is the one live thing on this panel; a 33 px blob of
# blown brass on the bezel sat in the same band for every frame of a six second strip and read
# as a second, dead eye at eleven o'clock. Every other part of the panel keeps material's own
# STEEL_SPEC ceiling of 212 - this is three counts under it, which is invisible as a colour and
# decisive as a rule: the housing is held below the band the living thing in the middle owns.
# Applied to the *fields* each part is composited from, so it holds hue and only takes level.
NEAR_CEILING = 144.0  # ...and the same rule again, harder, for the metal that stands right
# against the glass. A ceiling one count under the pupil is enough out on the swell, where a
# specular is a bezel's width away from the eye and reads as a different object; it is not
# enough on the ring that holds the pane down. Two blind critics measured the same fault
# independently: a still highlight on the well's own lip carried 96% of the moving blades' Weber
# contrast twenty pixels from them, and a dead-still edge as loud as the living ones beside it is
# how motion stops reading - the eye is given no reason to prefer the thing that is changing.
# So the housing's ceiling falls with radius: nothing still inside the seam between the two
# metals may pass this, and it climbs back to STILL_CEILING across the brass, where the lamp is
# allowed to land hard again. The bevels stay - a chamfer reads from its dark side and its
# gradient as much as from its shine - what they lose is the right to compete.
NEAR_EASE = 5.0  # reference px of brass the ceiling takes to climb back. Short, because it is a
# ceiling and not a shading: it only touches a pixel that was over it, and across the brass's
# inner face that turns a flat-topped clipped band back into a section that climbs outwards.
COLLAR_IN = 0.86  # the collar's inner flank, as a fraction of the swell
COLLAR_LIP = 6.0  # reference px of the steel bevel on the bezel's inner edge. Wide enough to be
# a machined face rather than a wire: at two and a half it read as a drawn white stroke round the
# glass, which is what a bezel does *not* look like from a pace away.
COLLAR_BEVEL = 0.35  # sin of the tilt that face keeps right across itself, so the whole of it is
# lit on the lamp's side and dark on the other: the brightness has to go round the ring, or the
# bevel is a white stroke again with a shadow on the far side of it.
COLLAR_BEVEL_SHINE = 0.26  # ...and how much of the lamp it gives back. A machined face, not a
# mirror: at full it saturates flat across the whole lit half and the gradient disappears.
# It was 0.55, which put the lit half of this ring at 200 and over - a second bright ring
# concentric with the pupil, twenty pixels outside the blades, holding still. See NEAR_CEILING:
# the ring the glass is bedded in is the last place on the housing allowed a hard specular, and
# what it keeps instead is its section - a graded face, an index cut across it, a reveal under
# the brass and a black seam - all of which read at this level and none of which needs 200.
COLLAR_REVEAL = 2.4  # reference px of the lip's outer edge that lie in the brass's shadow. The
# brass stands proud of the steel - the lip is sunk under it, holding the glass down - so the
# last pixels before the seam are occluded all the way round. Without this the lip is brightest
# where it is *flattest*, which is at its outer edge everywhere except on the lamp's own
# bearing: this lamp has more height (0.63) than reach (0.78), so a face square to the viewer
# takes more of it than an edge-on one does. A section that climbs outwards on every bearing but
# one is a ring drawn at a constant offset, which is the tell every critic of this panel has
# measured, and a reveal is what a real bezel has there anyway.
COLLAR_REVEAL_DARK = 0.62  # ...and how much of the light is gone at the bottom of it
COLLAR_CATCH = 0.22  # what the rolled edge at the glass picks up away from the lamp, as a
# fraction of the way to STEEL_LIT. A face turned right away from a lamp goes matte and reads as
# paint; a rolled edge over there still gathers the room along its length, and it is that
# terminating line - not a second highlight out on the face - that says the dark half is metal.
# Broad rather than confined to the far bearing, because a roll gathers the room everywhere the
# lamp's own highlight is not, and capped: any light on this panel arriving from anywhere but
# the lamp is held to a third of the member's own crest, which is what stops a fill becoming a
# second lamp.
COLLAR_INDEX = 24  # the index cut round the lip, one mark every fifteen degrees...
COLLAR_INDEX_LONG = 4  # ...one in four of them deeper, so it reads in quarters
COLLAR_INDEX_TICK = 0.30  # how far a short one reaches down the face, as a fraction of its width
COLLAR_INDEX_DEEP = 0.62
# A fifth of the marks the brass carries, and cut with the same tool (TICK_W): this is the ring
# you read a setting off, not the one you count degrees on. It costs the section nothing, which
# is the point of putting it here - a graduation is angular, so it puts detail on a band without
# adding a band to the radial profile, and this panel has already been told it wins on having
# fewer, larger rings than the photograph does.
COLLAR_SEAM = (14, 15, 12)  # the reveal where the brass meets the bevel - the dark line every
# two-part bezel has, and without it the two metals read as one band that changed colour. Near
# black rather than merely dim: what is at the bottom of a reveal is nothing, and a groove whose
# floor is a grey reads as a painted line.
COLLAR_ROLL = 3.0  # reference px of the brass's outer edge that roll down to the plate
COLLAR_STEP = 2.0  # ...and of its inner edge that go dark into the reveal beside the bevel
COLLAR_GROOVE = 0.45  # how dark the turned line just inside the roll is - the one mark a lathe
# leaves on every bezel, and what separates the face from the roll at a glance
COLLAR_LIFT = 3.0  # how proud the bezel stands of the plate, which is what sets its shadow
COLLAR_SHADOW = 0.80  # ...and how dark that shadow is where it is deepest. A part standing three
# pixels off a plate under one lamp occludes nearly all of it right against the edge; at 0.62 the
# plate under the bezel was still readable there, which is a part resting on its own picture.
COLLAR_WEAR = 0.6  # how much the brass's highlight comes and goes round the ring: handled where
# a thumb lands on it, dull between
COLLAR_TARNISH = 0.6  # how much the same slow drift shows in the brass itself, as multiples of
# the brushing - old brass is not one colour, it is polished in patches
COLLAR_SCRATCH = 0.34  # how pale the sheet's hairlines show where they cross the brass, and the
# only wear on this bezel that is not symmetric: a scratch takes the tarnish off and catches the
# lamp, so wear on metal is a bright outlier and never a dark one. Noise either way reads as a
# grain overlay laid over the drawing; the reference's wear is bright at seventeen to one.
COLLAR_TICKS = 120  # the graduation cut round the bezel's face, one every three degrees...
COLLAR_LONG = 5  # ...one in five of them run deeper, so it reads in fifteens
COLLAR_TICK = 0.34  # how far a short one reaches down the face, as a fraction of its width
COLLAR_TICK_LONG = 0.72
TICK_W = 0.6  # half the width of any graduation on his housing, in px. Not per-ring: a scale at
# this size is one pixel wide or it is a stripe, and every scale on this panel was cut with the
# same tool - his bezel's, and the two instruments' beside him.
TICK_DARK = 0.55  # how far into the metal the cut goes...
TICK_GLINT = 0.30  # ...and how much the wall on its far side comes back up. A dark line on a
# flat band is a printed scale; a dark line with a lit wall beside it is a cut one.
COLLAR_CROWN = 0.40  # sin of the tilt the brass face has reached where its roll begins. Far
# more than a bar's DOME, because a bezel is not a bar: it is turned with a rounded section, and
# the fall from its lit side to its dark side across a dozen pixels is most of what says so.
COLLAR_FACETS = 3  # ...and how many facets that crown is cut in. A turned bezel of this size is
# not polished round: it comes off the tool in a few flats, and the photograph everything here is
# measured against steps its chamfers by about thirty counts a facet rather than airbrushing
# them. Quantising the *tilt* rather than the colour is what makes the step land where the metal
# actually changes direction, so the same three flats run all the way round the ring.
BRASS = (96, 84, 58)  # a flat face of it square to the viewer. Well short of the metal in a
# catalogue: this is brass seen by a phosphor tube, and it borrows what little colour it has.
# Never orange, and never brighter than STEEL_SPEC in luminance - in luminance and not in any
# one channel, because a per-channel ceiling clips a warm metal's red first and hands the
# picture a brass bezel whose highlights are green.
BRASS_SPEC = (198, 169, 128)  # where the lamp lands hardest on it. Warm where steel's is cool,
# because a highlight carries the metal's own colour, and no brighter than the steel's ceiling.
# Both of these lost two or seven counts of green this round. Greenbias - G above the mean of R
# and B - is the number that separates hardware from phosphor on this panel, and it is held at
# +7 everywhere: the brass was at +9 on its face and +14 on its highlight, which is what a
# critic measured as "the brass band washes olive-green" under the west fixing. It is still
# brass and it is still warm - R over B by 38 on the face and 70 on the highlight - it simply
# no longer borrows the tube's colour to be warm with.
BRASS_BOUNCE = 0.12  # how much of the brass the steel lip picks up on the side turned away from
# the lamp. A ring of steel set inside a ring of brass has no other light on that side, and it is
# what stops the two metals reading as one cool band with a warm one beside it.
# It was 0.34, which is where a bounce stops being a fill and becomes a second lamp: laid flat
# across a face that was already climbing outwards, it made the lip's brightest pixel its outer
# edge on every bearing but the lamp's. COLLAR_CATCH puts the same amount of light back on the
# rolled edge, where a bounce actually lands.
BRASS_BLOWN = (250, 246, 232)  # ...and the lamp's own image in the polished roll, which is not
# the metal at all: a specular is the source reflected, so it is the source's colour and it is
# allowed past every ceiling the metal has. One line, a pixel or two wide, on one arc of one
# edge - the panel's speculars all top out at 209-214 and a rolled brass edge under a bench lamp
# does not, and this is the difference between a shaded drawing and a photographed part.
COLLAR_BLOWN = 0.80  # how far towards it the hottest pixel of that line goes. It was 0.95, and
# at 0.95 the line cleared 240 over a third of the ring - the pupil's own band, on a part that
# never moves. See STILL_CEILING: the line stays, and what it is not allowed to do is compete.
COLLAR_BLOWN_AT = 0.45  # of the roll, how far in from the outer edge the line sits: where the
# roll's tangent has come round to face the lamp squarely
COLLAR_BLOWN_W = 0.85  # ...how wide it is, as a gaussian's sigma in px. Narrower than the level
# it lost, on purpose: a specular on turned stock in the reference peaks and is gone in five
# pixels, so taking the height out of one has to buy a faster fall or it becomes a satin smear.
COLLAR_BLOWN_ARC = 4.5  # the power on how squarely a pixel faces the lamp, which is what keeps
# this a short arc on the lamp's side rather than a ring right round
COLLAR_BLOWN_WANDER = 1.4  # px its radius drifts round the ring. Nothing is turned perfectly and
# nothing is polished evenly, and a highlight that holds one radius all the way round is a line
# drawn at a constant offset - which is exactly what the last round measured on this panel.
COLLAR_BLOWN_DRIFT = 0.55  # ...and how much of the line's brightness the same handling takes
# away between the places a thumb has been, so it varies along its run as well as across it
COLLAR_TERMINATOR = 0.72  # how far below even the room's own light the far edge of the roll is
# taken. AMBIENT keeps an unlit face off black, which is right for a face; an edge rolling away
# from the lamp is occluded by the part it belongs to, and that goes to nothing.
STEEL_BLOWN = (246, 250, 247)  # the same lamp in the polished lip that holds the glass down,
# cool where the brass's is warm. Two metals, one source, and each gives it back in its own hue.
COLLAR_LIP_BLOWN = 0.10  # how far towards it the hottest pixel of the lip goes. It was 0.72,
# and 0.88 before that: three rounds of taking level off one line, and it was the wrong lever
# every time. What a lamp's image on a lip a pixel from the glass costs is not brightness, it is
# the well - a still line at that radius is measured in the same annulus as the blades and it is
# read against them. So this round moves it instead of only dimming it: out onto the face, in
# under NEAR_CEILING, and narrow. What is left is a glint on turned steel, which is all the lip
# was ever asked for; the lamp's own image on this housing lives on the brass's roll, sixteen
# pixels further out, where it has a bezel between it and the eye.
COLLAR_LIP_AT = 3.4  # ...and how far out from the glass's edge that line sits, in px. Past the
# middle of the face rather than on the corner: the corner is where the section has to go DARK
# for the glass to read as set into something, and a line there is a stroke round the pupil at
# the one radius nothing on this panel may put one.
COLLAR_LIP_W = 0.55  # ...and its sigma, narrower than the brass's own. A glint that has lost
# most of its height has to buy a faster fall or it spreads into a satin band, which is the one
# thing a machined lip may not look like.
WELL_FLOOR = (9, 27, 15)  # the disc he is drawn on, at its middle...
WELL_WALL = (2, 6, 4)  # ...and where it meets the wall. Dark enough that no test counting
# phosphor ever sees it, light enough that a shadow falling on it has something to fall on -
# SCREEN is so near black that a shadow on SCREEN is nothing at all.
# Both were two and a half times this for a round, and the eye lost every blind vote it was in.
# What is behind him is not a lit surface, it is the inside of an instrument: a dark cavity with
# a machined floor, and everything you see in there is the phosphor drawn on it. The ratio
# between these two, the fall, the lamp and the grain are all unchanged - only the level is,
# because the level was the whole fault.
WELL_FALL = 1.38  # how sharply the one goes to the other with radius. Above one, so the middle
# stays level and the drop gathers at the wall: a plain square is a bowl, and a bowl reads as a
# painted vignette rather than as a floor with a wall round it.
WELL_LAMP = (6, 14, 8)  # what the half of the disc facing the lamp gains over the half away
# from it. The one thing that stops a dark disc reading as a hole cut in the panel.
WELL_SHADE = 0.38  # ...and what the half turned away from it loses, as a fraction of itself.
# The pane is being shone at, not lit from within: what it gives back on the side facing the
# room falls away to nothing on the side turned from it, and "nothing" is darker than the flat.
# Multiplicative, so the fall takes level and not hue - the cavity stays green where it is
# visible at all. This is the only way the lamp gets any more swing across the well: every
# version of this that bought amplitude by *lifting* the lit side put the well's black floor up
# with it and cost the turning ring the stroke contrast the last round was won on. Taking the
# far side down buys the same swing and pays for it in the right direction - a deeper moat, a
# darker median, and more Weber on every mark out there, all measured.
# All three are grey-green rather than phosphor-green, and that is the difference between glass
# with light on it and a lit screen: the pane is not a source, it is a thing being shone at.
# They keep that tint where every piece of metal on this panel has had its taken away, and the
# reason is that this is not metal: it is the ground the iris is drawn on, seen through glass,
# and it is 20 counts of luminance. Neutralising it at constant luminance was measured and cost
# the face 0.157 saturation to 0.153 - the blades are blended against it - which is the one
# trade this round is not allowed to make. What IS metal down here is the groove turned into
# it, and that is where the tint comes out: see SEAT_METAL.
WELL_HATCH = 0.10  # the pane's own brushing, either way, running on the lamp's diagonal
_HALF_ROOT = math.sqrt(0.5)  # a panel pixel measured along that diagonal, and across it
WELL_SPOT = 0.030  # the pane's reflection of the lamp, as a fraction of the way to WHITE...
WELL_SPOT_AT = 0.42  # ...how far up it towards the lamp that sits, of the flank radius...
WELL_SPOT_LONG = 0.62  # ...and how far it spreads along the lamp's line and across it. Drawn
WELL_SPOT_WIDE = 0.40  # out rather than round, the way a reflection on anything curved is.
WELL_SCRATCH = 0.055  # how far towards WHITE the sheet's hairlines lift the pane where they cross
WELL_DEPTH = 1.5  # reference px the floor sits below the bezel's lip, which is how far the near
# wall's shadow reaches across it
WELL_SHADOW = 1.6  # how dark that shadow is under the wall - over one, and clipped, so it is
# black where the wall meets the floor rather than merely dim. A well you can see the bottom of
# all the way to its edge is a disc with a ring painted round it.
WELL_GRAIN = 0.5  # how much the floor's turning marks show, as a fraction of material.GRAIN
WELL_CEILING = 195  # no pixel of the well may reach the 200 the face's colour is measured over
# (`test_he_changes_colour_with_what_he_is_doing`), or the disc behind him starts voting on his
# mood. Enforced rather than argued: the last thing built here is a hard scale onto this sum.
# The seat the stator drum runs in: a groove turned into the well's floor at the radius the
# knurl rides, so the grip band is a serrated edge running in a track rather than a row of marks
# floating on a disc. Built into the well rather than drawn in his tile, because the tile is
# transparent wherever nothing is written on it and this groove is the same pixels in every
# frame of every mood - the knurl turns in it, which is what it is there to be read against.
#
# A band of brass with a scale on it stood here for a round, and it was the single worst thing on
# the panel: pale, warm, and parked at 0.614-0.727 where the blade tips sweep, so his most
# saturated feature was the one part of the face that never moved. Brass belongs to the bezel,
# where the panel's palette puts it and where nothing is trying to move underneath it.
SEAT_IN = 0.685  # of his rim: the groove's inner wall - clear of the diaphragm's own reach...
SEAT_OUT = 0.735  # ...and its outer, where the stator's vanes take over
SEAT_DEEP = 0.60  # how far the floor of the groove is taken down from the well's own, which is
# what makes the knurl look sunk into it rather than laid across it
SEAT_WALL = 2.2  # px of each wall, which is where the depth is actually read: a groove is two
# lines - one dark where the wall turns from the lamp, one lit where it turns into it
SEAT_LIP = 2.95  # how much of the lamp the wall facing it gives back, as a fraction of the way
# from the sunk floor back up to the well's own colour. Over one, so the lit wall comes out
# *brighter* than the floor around the groove: a wall tilted into the lamp catches more of it
# than the flat does, and one that only climbs back to level is a fade rather than an edge.
# Well over, now that only one wall of the two lights at all: the lit wall reads 24 counts over
# its own baseline where the lamp lands and 7 under it opposite, which is a groove with a
# section, where at 1.45 across both walls it was a two count wobble on a black band. It fits
# the bezel's lamp to within a degree and a half at an amplitude of 12.5, which is the number
# the material critics ask every annulus of this housing for. It is still a turned edge in a
# dark cavity: its brightest pixel is 52, against a pupil at 253.
SEAT_GRAIN = 1.9  # its turning marks, as a fraction of material.GRAIN. Coarser than the floor's
# because the groove was cut with the tool still in the work, not skimmed flat afterwards.
SEAT_SHADOW = 0.7  # how dark the shadow the near wall drops across the floor of the groove is
SEAT_METAL = (33, 42, 37)  # the groove is machined steel and not more of the pane, so it is
# neutral where the floor round it is green: greenbias +7 against the floor's +15, which is the
# ceiling this panel now holds every metal surface to. Laid on at the *level* the floor has
# already reached at that radius rather than at its own, so the hue changes across the edge of
# the cut and nothing else does - the well keeps its depth, its vignette, its lamp and its
# shadow, and the one band of it a lathe touched stops wearing the tube's colour. Green belongs
# to the phosphor drawn over this and to the glass above it.
SEAT_ARC = 1.5  # the power on how squarely a pixel of the groove faces the lamp. The same
# argument as MOUTH_ARC and the reason it is here: `rise` is positive on the inner wall and
# negative on the outer one, and the outer one geometrically faces the lamp wherever the inner
# one does not - so a groove four pixels wide came out lit on BOTH sides of the ring, from
# opposite directions, which is an emboss and not a cut. A groove at the bottom of a cavity does
# not see the room from over there: what reaches it comes in over the bezel's lip on the lamp's
# own bearing, so one wall lights and the other goes into its own shade.
SEAT_TERMINATOR = 0.55  # ...and how far under the sunk floor that far wall goes
SEAT_FACETS = 2  # how many flats each wall of the groove is cut in. The same argument as
# COLLAR_FACETS at a smaller size: two is all a two-pixel wall has room for, and two hard steps
# still read as a cut edge where a smooth ramp reads as a fade.

# The mouth of the well: the countersink between the outermost thing he draws and the flank the
# bezel's lip stands on. Until now that band was bare floor, and it is why every critic could fit
# the panel's lamp to the collar's brass and to nothing inside it - the light story stopped dead
# at the steel. This is the one piece of *static* metal with room to carry it inwards: nothing he
# draws reaches past the castellated ring, so a turned land here is lit by the same lamp as the
# bezel without a directional wash ever touching a blade or the floor they turn over. That
# distinction is the whole design. A wash over the well makes a mark's brightness a function of
# where it has rotated to, which destroys the one cue that proves the ring is turning; a lit ring
# of metal at a fixed radius is a part, and a part may be lit.
#
# One section, read outwards: a relief groove at the bottom of the cut, a face that ramps up in
# flats to a single lit crest, and then the reveal under the lip - the darkest metal in the whole
# cross-section, because the bezel is standing over it.
#
# It was twice this wide for a pass and that was the wrong answer: a countersink reaching down to
# the castellated ring put lit grey metal under the one ring you can watch turn, and the moat
# between the still bezel and the moving eye - which is the whole reason this face reads as two
# planes - closed up. Everything the lamp does out here now happens outside 0.91, where nothing
# is drawn, and the annulus the material critics measure is only two pixels of that. That trade
# is deliberate and it is stated in the summary: a lamp fitted across the *whole* mid annulus
# needs a wash over the floor, and a wash over the floor is what lost the last round.
MOUTH_IN = 0.938  # of his radius: the bottom of the cut. There are five pixels of bare floor
# between the castellated ring and the rim's own stroke, and this is them: further out and the
# rim writes over the whole section, further in and the cut is under a ring that turns.
MOUTH_LAND = 0.26  # of the mouth's width: the relief groove at the foot before the face climbs.
# Every cut has one, and it is what stops the chamfer reading as the floor tipping up.
MOUTH_FOOT = 0.50  # sin of the tilt the face already has where it leaves the relief...
MOUTH_TILT = 0.85  # ...and what it has reached at the crest. Both steep, and that is the whole
# trick: a face square to the viewer still takes two thirds of this lamp, so a cone that starts
# flat is lit right the way round and lifts the well's black floor with it. A cone this steep is
# past the terminator on its far side - it goes to AMBIENT there - so the light it carries into
# the annulus is bought entirely out of the dark half and none of it out of the floor.
MOUTH_FACETS = 2  # flats across that ramp - see COLLAR_FACETS. Two and not three: there are
# four pixels of section here once the rim's own stroke has taken the outer two, and a flat
# narrower than a pixel is a gradient with extra steps in it.
MOUTH_ARC = 1.9  # the power on how squarely a pixel of the cut faces the lamp. A countersink
# at the bottom of a bore does not see the room, it sees the lamp through the hole the bezel
# leaves, so the arc that is lit at all is short. It was 3.0 and that was too short to be the
# same lamp as the bezel outside it: a lobe that narrow carries a huge peak-to-trough swing and
# almost no first harmonic, so the ring measured as a bright spot rather than as a lit annulus.
# At 1.9 the fit is 12.8 counts of amplitude against the brass's 55 to 84, on the same bearing.
MOUTH_METAL = 0.62  # how far the mouth's steel is put from STEEL towards STEEL_DARK. Dark: it
# is inside a cavity, under an overhanging bezel, and its unlit side has to sit down with the
# well's own floor or the cavity has a grey ring painted round the inside of it
MOUTH_SHINE = 0.44  # how much of the lamp its crest gives back
MOUTH_GRAIN = 1.5  # its turning marks, as a fraction of material.GRAIN. Coarser than the
# floor's, like the seat's: a countersink is cut, not skimmed
MOUTH_REVEAL = 3.0  # reference px of it that go into the lip's shadow at the outer edge
MOUTH_DARK = 0.86  # ...and how far under the room's own light that reveal and the relief go
MOUTH_LIFT = 4.3  # how proud the crest stands of the well's floor, which is what sets the
# shadow it drops back down into the well - multiplicative on the floor, and recovering over a
# dozen pixels rather than stopping at an opaque line. It reaches further than it did, and that
# is bought rather than free: this shadow is the one lever on the ring band's black median that
# does not touch the floor's own level, and the median is half of what the moat between the
# still bezel and the turning eye is made of.
MOUTH_SHADOW = 0.92
# The glass over him. A dome held down by the lip, and the one part of him that is not drawn
# every frame: the lamp's reflection on it and the light it gathers along its edge are a tile
# built once and laid over the eye after it is painted. It has to stay off everything that moves.
# The iris and everything inside it travel with his gaze, as far as IRIS + GAZE_SHIFT plus the
# width of a stroke, and a highlight over a moving spark makes the spark's brightness a function
# of where it is - which is exactly the flicker the sleeping face is not allowed. And it has to
# stay off the rim, which the scan sweep is measured on. What is left is the band between, the
# stator and the castellated ring, and that is where a dome's glare falls anyway.
GLASS_IN = 0.10  # of his radius: where the dome's own reflection may start. Almost nowhere -
# it washes the whole face and stops only at the spark, which is right anyway: the spark is the
# light, not a surface for the light to land on. It used to stop at 0.62, and the reason given
# for that was wrong. The first attempt at a stronger dome failed two tests at once, and the
# radius was pulled back until they passed on the assumption that reach was what they objected
# to. It was not. Re-measured one variable at a time: at this alpha the reach may go to 0.10 with
# everything green, and the only test that ever objected to reach is the sleeping face's peak
# (test_he_turns_and_breathes_while_he_is_asleep), and only at 0.0, where the wash covers the
# spark itself and the breath moves the brightest pixel under it. The other objection was always
# to ALPHA and never to radius - see GLASS_GLARE.
GLASS_OUT = 0.955  # ...and where it must have ended, short of the rim
GLASS_EASE = 0.05  # how far past each of those it fades in and out
GLASS_AT = 0.62  # how far up the dome towards the lamp its reflection sits, as a fraction of him
GLASS_REACH = 0.72  # ...and how far that reflection spreads, in the same units. Both were half
# this: the reflection sat high on the rim and stayed there, which reads as a lit edge and not as
# a curved face with a lamp somewhere above it
GLASS_GLARE = 0.20  # its alpha where it is brightest, in the tube's own white. Three times what
# it was, and the one number the glass is actually bounded by: at 0.26 with this reach the white
# on his sleeping face outweighs its own dim green and test_he_changes_colour_with_what_he_is_
# doing reads him as awake. The asleep face is the binding one because it is the dimmest, so a
# neutral wash moves it furthest. Traded down from 0.26 to buy the reach above, which is the
# right way round: a reflection that stops at a ring inside the eye is not a reflection.
GLASS_RIM = 0.15  # the light the dome gathers along its edge on the side facing the lamp
GLASS_SHADE = 0.14  # ...and how much the far side of it darkens what is under it
# The loom: three runs of flexible steel conduit leaving the back of his housing through a gland,
# the same lamp on them as on everything else. Conduit and not cable, because everything on this
# panel is metal or glass; darker than the bars, because it is braid and not a machined face.
LOOM_STEEL = 0.45  # how far the conduit is put from STEEL towards STEEL_DARK
LOOM_LIFT = 2.0  # how proud a run stands of the plate, which sets its shadow
LOOM_RIB = 1.6  # how much the braid's ribbing shows, as multiples of a bar's brushing
LOOM_GLOSS = 0.42  # ...and how much of the material's highlight a braided sleeve keeps. Not a
# machined arris: a woven jacket scatters, so its light is a broad graded band round the section
# and not a mirror of the lamp. At full gloss a nine-pixel cable put a one-pixel line at 161
# against 25 on either side of it, in the darkest corner of the panel, and three critics in a
# row read the pair of them as lens flare rather than as metal turning to the light. The one
# specular on this panel belongs to the bars; the loom in front of them has a sheen
GLAND_ROLL = 2.0  # reference px of the gland's edges that roll
# Where the loom leaves him, in PIL's degrees - straight at the panel's own corner, because that
# is the only direction with any run in it. His swell comes within five pixels of both the left
# edge and the bottom one, so the pocket between him and the corner is the whole cable budget:
# about 41 px at 800x480, which is enough for a gland and three cables and nothing else.
GLAND_AT = 135.0
LOOM_N = 3
LOOM_FAN = 11.0  # degrees between one cable and the next
LOOM_REACH = 70.0  # reference px, which is past the corner: they are meant to leave the panel
# The mount points round the collar besides the two where the bracket's rail meets it, over his
# shoulders at about -118 and +28. These two say the rest of the collar is bolted down as well.
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

# The pilot lamp, on the plate under the ramp rather than bolted through it like the two dials.
# It is not a control and never becomes one - see LAMPS - so it wants the plate, which is the one
# part of this mount nothing has to be able to hit.
#
# Measured rather than chosen: with both dials, the ramp, the foot rail, the corner webbing and
# the machined surround taken out, the empty plate in this corner is Rect(720, 380, 61, 61) and
# the largest circle that fits it is r=35 at (744, 414). PILOT_DEPTH puts the lamp within two
# pixels of that centre while being a fact about the mount rather than two magic numbers: it is
# on the perpendicular bisector of the ramp, which is also what makes it sit square between the
# knob and the gauge instead of nearer one of them.
#
# The r=35 is a ceiling on the FITTING PLUS ITS LIGHT, and it is load-bearing twice over. The
# circle is bounded by the two dials' own discs, and tests/test_eye.py asserts that both dial
# hitboxes are byte-identical in every state - so a glow that reaches either one is a glow that
# breaks a test, and rightly: light from a state lamp landing on a volume knob would make the
# knob look like it meant something.
PILOT_DEPTH = 60.0  # in from the ramp's centreline, reference px
PILOT_R = 28.0  # the brass fitting's outer edge...
PILOT_BEZEL_IN = 0.66  # ...and the share of it inside the bezel's mouth, where the glass starts
PILOT_SEAT = 0.12  # the shadowed reveal the dome is set down into, as a share of the radius
# The ring's section, across its width from the glass out - his collar's, at a quarter of the
# size. See Overlay._pilot_bezel: Marco asked for the brass round the eye rather than a second
# idea about brass, so these track COLLAR_* wherever the scale lets them.
PILOT_ROLL = 0.40  # how much of the ring's width rolls over at the outer edge...
PILOT_STEP = 2.0  # ...and reference px of its inner flank going dark into the seat
PILOT_CROWN = 0.40  # sin of the tilt the face has reached where its roll begins - COLLAR_CROWN
PILOT_FACETS = 3  # ...and the flats it is cut in. A turned bezel of this size gets three
PILOT_GROOVE = 0.45  # the turned line just inside the roll, the one lathe mark that is not a scale
# The graduation. The TOOL is shared with every other scale on this panel (see _graduation, and
# TICK_W, which is not an argument); the PITCH is not, and must not be. The collar cuts 120 marks
# at a radius near a hundred - five and a half pixels a mark - and the same count here would be
# one and a half, which is a texture rather than a scale. This is the count that lands on the
# collar's own arc-length pitch, so the two rings read as indexed by one machine.
PILOT_TICKS = 28
PILOT_LONG = 4  # one in four deeper, so it reads in quarters
PILOT_TICK = 0.34  # how far a short one reaches down the face - COLLAR_TICK
PILOT_TICK_LONG = 0.72
PILOT_TARNISH = 0.6  # the slow drift in the brass itself - COLLAR_TARNISH
PILOT_SCRATCH = 0.34  # how pale the sheet's hairlines show where they cross - COLLAR_SCRATCH
PILOT_TERMINATOR = 0.72  # how far under the room's own light the far edge of the roll goes
PILOT_BLOWN = 0.80  # the one line where the roll carries the lamp's own image, not the metal's
PILOT_BLOWN_AT = 0.45  # of the roll, how far in from the outer edge that line sits
PILOT_BLOWN_W = 0.85  # ...its sigma in px...
PILOT_BLOWN_ARC = 4.5  # ...and how squarely a pixel must face the lamp to carry any of it, which
# is what keeps it an arc on the lit side instead of a ring all the way round
PILOT_BLOWN_WANDER = 1.4  # px its radius drifts. Nothing is turned perfectly and a line that is
# the same distance from the edge the whole way round is a drawn one
PILOT_LIFT = 2.0  # how far the fitting stands off the plate, for the shadow it drops
PILOT_SHADOW = 0.62
PILOT_WEAR = 0.35  # brass on a bench machine is handled brass
# The dome, and the light in it.
PILOT_DOME = 0.80  # the cap's tilt at its rim: 1 is a hemisphere, this is the low dome Marco picked
PILOT_FALL = 0.7  # how fast the light inside gives out towards that rim, and it is under one on
# purpose. A lit lens is BRIGHT nearly all the way to the glass and then stops at the seat; the
# falloff is what rounds it, not what shapes it. At 1.5 the lens dimmed by half across its own
# face and the lamp read as a smudge rather than as something switched on - which is the half of
# "blurry" that is not the blur. What gives the edge is the seat, not the gradient.
PILOT_CORE = 0.55  # the hot middle, as a share of the glass's radius
PILOT_CORE_A = 0.95  # ...and how solid it is at full brightness
PILOT_BODY_A = 0.88  # the rest of the dome, which is lit glass rather than the filament
PILOT_GLARE_AT = 0.42  # where the bench lamp's own reflection sits, out towards it
PILOT_GLARE_REACH = 0.30  # ...and how wide. A coin of glass gives back a spot, not a wash
PILOT_GLARE_A = 0.70  # white, whatever colour the lamp is burning: a reflection is the room's
PILOT_RIM_A = 0.34  # the cut edge of the glass, lit from inside all the way round
PILOT_GLASS_A = 0.93  # how solid the unlit bead is. Not 1: a dead lamp is dark glass, not a hole
PILOT_BLOWN = 0.55  # where the knurl's crown carries the lamp's own image rather than the brass's
PILOT_SEAT_DARK = 0.62  # the undercut the bead is set down into, under the ring standing over it
# ...and the light it throws, which is the half Marco asked for by name. One falloff off the
# edge of the glass, and it is ANALYTIC rather than a blurred mask. This went through two blurs
# first - PIL's, which is uint8 and quantised the outer halo to six distinct values, and then a
# separable Gaussian written out in float, which fixed the banding and did not fix the softness.
# The softness was never the blur's resolution. It was that a blurred mask can only be laid OVER
# the metal, and a coloured layer at half alpha takes a knurl's bright flank and its dark flank
# to the same green. A falloff in closed form can be evaluated at the supersampled grid the rest
# of the part is built on and ADDED to the metal's own shading, which keeps every edge the ring
# had. See Overlay._pilot_tile.
PILOT_THROW = 11.0  # how far past the glass the light carries, reference px...
PILOT_THROW_REACH = 2.2  # ...and how many of those are worth building a tile for. Past this the
# falloff is under a level and the corner's own cap usually bites first anyway
PILOT_ON_METAL = 0.26  # ...how much of it lands on the ring's CHAMFER, which is the only face
# turned inwards far enough to see the lens. The roll and the land face the room and get none:
# at 0.62 across the whole ring the brass measured 74 green dead and 217 lit, which is not a lit
# fitting but a green one, and three channels saturating together is a section going flat...
PILOT_ON_PLATE = 0.30  # ...and how much pools on the plate outside the fitting, where there is
# no shading of ours underneath to add to and a soft layer is the honest thing
PILOT_STEPS = 16  # brightnesses a tile is cached at. Every row of LAMPS is STEADY or BLINK, so
# in practice this is one tile per colour; it is here against the day a row wants to fade

# Both dials sweep 270 degrees with the gap at the bottom, which is where the gap is on every knob
# anybody has ever turned and every gauge anybody has ever read. PIL measures clockwise from three
# o'clock, so that is 135 (down-left, empty) through the top and round to 405 (down-right, full),
# and the bottom quarter is left free for the reading to sit in.
DIAL_FROM = 135.0
DIAL_SWEEP = 270.0
DIAL_TICKS = 7  # graduations across the sweep, both ends included
# The scale runs well inside the bezel, with the graduations between the two: a green arc drawn up
# against a green chamfer is a green chamfer, and that is what the first cut of this looked like.
DIAL_TRACK = 0.615
DIAL_TICK_IN = 0.720  # the graduations stop short of the bezel's foot, so they sit on the face
DIAL_TICK_OUT = 0.800  # and not in the shadow it drops down the wall of the recess. They stand
# clear of the sweep as well as of the ring: arc and graduations fused into one eight-pixel band
# is a band, and a scale is read by the gap between the mark and the thing pointing at it
DIAL_TICK_W = 1.35  # reference px. Heavier than a hairline: a graduation is the one mark on a
# gauge with no shape of its own to be recognised by, so if it does not carry it is not there -
# but taller than it is wide, which is the whole of what makes a mark POINT at a centre. At 1.7
# by 2.2 a graduation was as wide as it was long and a critic read the set of them as nubs stuck
# on the outside of the arc rather than as a scale; the aspect matters more than the weight
DIAL_HAND = 0.45  # the pointer's tip. Short of the sweep by more than the sweep's own bloom,
# which is what keeps a pointer a pointer: the skirt of light round the arc reaches inwards too,
# and a tip that ends inside it is a tip nobody - and no test - can tell from the scale
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
DIAL_LABEL = 0.585  # where the reading sits in the gap under the hub, clear of the pointer's
# reach: a needle at either end of the sweep points down into that quarter, and a number it
# grazes on the way past is a number you read twice to be sure of. It was 0.63, which put the
# bottom-left corner of the digits two pixels INTO the bezel - a mark printed on the face drawn
# over the metal holding the face in. See _dial_aperture, which is what makes that impossible
# rather than merely unlikely; this is what keeps the number off the edge of it

# What an instrument is made of. A face sunk into the rail, glass over it, and a bezel holding
# the glass in - all of it under the panel's one lamp (:mod:`cyclops.material`), so the bezel's
# crown is bright where the bars' rolls are, the face is shaded by the ring standing over it,
# and the glass gives the lamp back from one place.
#
# The bezel is the panel's own steel and not brass. It was brass for one round, on the argument
# that an instrument is the part of a machine that came from somewhere else, and at 72 px across
# it read as a sticker of a different gauge stuck onto a steel bracket - a flat olive ring with
# no light on it at all. The only brass left on this panel is the ring round his eye, which is
# the one thing that is not furniture.
#
# The ring's section is four bands, and they are what a lathe leaves rather than what an airbrush
# does. Outside in: a quarter-round ROLL turning down to the bracket, which is the one face on the
# whole ring square enough to the lamp to catch it, so the ring has ONE highlight; a flat LAND,
# which is the body tone the highlight rides on and is nearly square to the viewer; a CUT chamfer
# falling away in to the glass, which faces the other way and therefore sees no lamp at all, only
# what bounces back off the far side of the crown; and under that the seat the crystal sits in.
#
# It was a symmetrical crown for one round - a crest in the middle falling away to both edges -
# and that put the lamp on it TWICE, half a turn apart, at very nearly the same brightness. Three
# critics read that as a flat printed ring under two lamps rather than as one machined part under
# one, and they were right: a chamfer cut down towards the glass is in the crown's shadow, and
# what it gives back is a bounce and not a second lamp.
DIAL_BEZEL_IN = 0.86  # of the radius: where the crown stops and the groove under it begins.
# Three per cent further out than it was, which is a quarter of the ring's width and nine per
# cent more face to read: a critic counted 38 per cent of our dial's area spent on inert metal
# against the reference's 25, and the part of an instrument you look at is the plate
DIAL_BEZEL_ROLL = 0.44  # of the band: the outer quarter-round, which takes the one highlight
DIAL_BEZEL_ROLL_TILT = 0.97  # sin of the tilt it has reached where it meets the bracket
DIAL_BEZEL_ROLL_ROUND = 0.80  # under one keeps the turn gentle where it leaves the land
DIAL_BEZEL_LAND = 0.11  # sin of the tilt the flat between the two keeps - a machined face is
# never quite square to the viewer, and a face that is has no gradient across it at all
DIAL_BEZEL_CUT = 0.30  # of the band: the chamfer cut down to the glass, facing in and down
DIAL_BEZEL_CUT_TILT = 0.78  # sin of the tilt it reaches at the seam
DIAL_BEZEL_CUT_GLOSS = 0.13  # how much of the crown's highlight a cut face keeps. Nearly none:
# it is the one band on the ring a polishing cloth cannot reach into
DIAL_BEZEL_CUT_SHUT = 0.36  # ...and how much of the lamp the crown standing over it keeps off,
# which is what turns the old mirrored second highlight into the dim bounce it should have been
DIAL_BEZEL_ROVE = 0.30  # of the roll: how much narrower the roll is on the side facing the lamp,
# so the highlight's radius travels round the ring instead of tracing one perfect circle...
DIAL_BEZEL_ROVE_N = 0.46  # ...and how much of that again wanders at random, so its brightest
# point is not on one row for the whole sweep
DIAL_BEZEL_BLOWN = 1.00  # how far the crown's brightest line is pushed past the steel's own
DIAL_BEZEL_RIDGE = 3.4  # ...and how narrow it is: a blown highlight is a line, not a band
DIAL_SPARK = 1.22  # of STEEL_SPEC: what a highlight that has blown out reaches. The steel's own
# ceiling is what a *material* gives back; a specular is the lamp, and the lamp is brighter
DIAL_BEZEL_SCRATCH = 14  # hairlines dragged across each instrument's metal. Bright only: a
# scratch takes the finish off and what is under it catches more light, never less
DIAL_BEZEL_SCRATCH_A = 0.42
DIAL_SEAT_ROVE = 0.32  # of the seat's width: how much it wanders round one ring
DIAL_TARNISH = 0.28  # of the polish: how far one ring's finish may sit from the other's
DIAL_SEED_STEP = 31  # so the two instruments are two parts and not one sprite stamped twice
DIAL_BEZEL_TURNED = 0.38  # of the way from the bars' STEEL to STEEL_LIT. A bezel is turned and
# then polished, and a polished face gives back more of the room than a sawn and brushed one -
# which is what makes the ring read as the brightest metal on the panel rather than the dullest
DIAL_BEZEL_FILL = 0.16  # how far the lamp's own term wraps past the terminator, so a slope that
# has just turned out of the light does not go straight to the ambient floor...
DIAL_BEZEL_SHUT = 0.50  # ...and how much of the room the slope facing the other way still sees.
# The panel's ambient is what a face square to the viewer sees of the room; a slope on a ring
# bolted to a bracket sees the bracket instead, so it goes below that floor. Without this the
# ring has no black in it anywhere and reads as a moulding rather than as turned metal.
# It was 0.70, back when it scaled the whole colour and so carried the section's contrast on the
# shadow side as well as its own. Now that it moves the ambient alone this is the only thing on
# the dark half that still knows which way each band of the section is leaning, so it needs the
# range: the difference between a rim with a crown, a land and a chamfer in it and a dark arc
#
# ...and none of that varies round the ring, which is what three critics measured and called a
# stroke. The lamp is a long way above and to the left of a ring five pixels wide and three
# proud: the far half of the crown stands between it and everything behind, so that half sees
# no lamp at all and the well it is bedded in instead of the room. A flat annulus square to the
# viewer takes the same light at every azimuth - true, and the reason the last cut measured at
# CV 0.13 all the way round - so what has to travel round the ring is not the surface's answer
# to the lamp but whether the lamp reaches it. These two are that, and they are what turn a
# donut into a rim with a lit side and a dark one.
DIAL_BEZEL_CAST = 0.18  # what of the lamp still finds the far side of the ring...
DIAL_BEZEL_WRAP = 1.25  # ...and how quickly the crown's own shadow takes it round there
DIAL_BEZEL_WELL = 0.74  # ...and what the room gives a ring bedded in a plate, on that side
DIAL_BEZEL_BOUNCE = 0.36  # ...and what comes back UP off that plate onto the skirt of the roll,
# added to the ambient there. The plate is lit all the way round the ring, so the metal standing
# over it is not black anywhere either: a critic sampled our crown every ten degrees and read 19
# to 30 for twelve consecutive samples - over half the circumference unlit - against a reference
# whose ring never leaves metal at any azimuth. Diffuse, and the same at every bearing, which is
# what makes it a floor under the shadow side rather than a second highlight on it
DIAL_BEZEL_BOUNCE_AT = 0.42  # of the roll, from its outer edge in: how much of it has turned
# far enough down to see the plate at all. A pixel, near enough. Spread over the whole roll it
# is a wash that lifts the ring's mean and takes the azimuthal gradient with it; kept to the
# skirt it is the one lighter line along the outer edge every photographed bezel has
DIAL_BEZEL_POLISH = 2.8  # how much brighter the roll's highlight is than the land's: a ring is
# polished where a cloth reaches, and the flat in the middle keeps the bars' own dull shine
DIAL_BEZEL_SHINE = 1.8  # how tight the ring's highlight is. Tighter than a bar's, because a
# turned face is smoother than a sawn one
DIAL_BEZEL_GRAIN = 1.05  # of the bars' brushing. Round the ring, because it came off a lathe.
# Held to an eighth of the local face value: at two and a half times the bars' amount the tooth
# swung further either way than the shading it was laid over, and a texture louder than the form
# under it is a noise overlay rather than a finish
DIAL_WEAR = 0.42  # how much the bezel's highlight comes and goes round the ring
DIAL_BEZEL_EDGE = 1.7  # reference px of the outer silhouette darkened against the bracket...
DIAL_BEZEL_EDGE_DARK = 0.62  # ...and by how much where the ring stands in its own shadow, so
DIAL_BEZEL_EDGE_LIT = 0.34  # ...against where the lamp is still on it. The metal has to end on
# a line and not dissolve into what it is bolted to, but a line of one value all the way round
# is the drawn outline this ring spent two rounds being mistaken for
DIAL_BEZEL_LIP = 0.9  # reference px of the crown's inner edge that turn down into the groove...
DIAL_BEZEL_LIP_DARK = 0.85  # ...and how far they go dark on the way where the crown stands
DIAL_BEZEL_LIP_RAKE = 0.60  # over them. On the other side of the ring that same inward turn is
# the one face on the whole part the lamp reaches over the crystal to touch, and it draws the
# pale line round the inside of the far half that every photograph of a bezel has. A BOUNCE and
# not a light: a quarter of what the crown gives back, which is what keeps it from reading as a
# second lamp - the mistake this ring has already been rebuilt once to undo
DIAL_BEZEL_SEAM = 2.4  # reference px of near-black groove between the crown and the glass. The
# one line that says the glass is *inside* the ring and not printed level with it
DIAL_SEAM = 0.27  # of STEEL_DARK: how black the bottom of that groove goes where the crown is
DIAL_SEAM_RAKE = 5.6  # ...and how far a lamp that reaches over the glass and down into it on
# the other side lifts that floor. An undercut is a shadow, and a shadow of one depth drawn all
# the way round a ring is a stroke: the reveal is widest and blackest on the side the crown
# stands between it and the lamp, and narrows to a lifted hairline where the light rakes in
DIAL_SEAM_SHUT = 1.00  # of the groove's width where the crown shuts the lamp out of it...
DIAL_SEAM_OPEN = 0.58  # ...against where it rakes straight down the wall
DIAL_BEZEL_RAKE_WRAP = 2.2  # how tightly that raking light stays on the arc of the ring
# actually turned to receive it. A bounce that spreads out over half the circumference is a
# second lamp, and this part has already been rebuilt once for having one
DIAL_LEAK = 0.55  # of the seat's width, from its inner edge out: how far the light out of the
# crystal's cut edge carries down the undercut before the seat's own black takes over. Of the
# seat and not a count of pixels, so wherever the seam has wandered the green stops at the metal
DIAL_LEAK_A = 0.46  # ...and how much of the phosphor comes back out of it there. A pane over a
# lit dial is a light pipe: it carries the face's light sideways and lets it go where the glass
# is cut, which is why every photograph of an instrument has a green line round the inside of
# the bezel and none of it on the bezel. Kept inside the seat, so the ring's own metal never
# sees it - see DIAL_GREENBIAS, which this is laid down after and deliberately outside of
DIAL_GREENBIAS = 6.0  # levels of green over the mean of red and blue any metal on an instrument
# may hold. Green on this panel belongs to the phosphor and to the glass over it; a bezel that
# shares it reads from a pace away as a green painted ring rather than as steel, and the two
# stop being different materials. Clamped rather than set, so retuning the palette in
# :mod:`cyclops.material` moves the ring's colour and this cap never has to be touched again
DIAL_LIFT = 3.0  # how proud an instrument stands of the rail it is bolted through
DIAL_SHADOW = 0.60  # ...and how dark the shadow it drops is, where deepest
DIAL_FACE_A = 0.95  # how much of the camera the face keeps out. Higher than the well a switch
# left, because the bracket running under a translucent face crossed it with a straight diagonal
# step and a face behind glass with a crack across it is a broken instrument
DIAL_RECESS = 0.26  # of the radius: how far across the face the bezel's own shadow falls
DIAL_RECESS_A = 0.99  # how opaque the face goes at the foot of that shadow
DIAL_RECESS_ALL = 0.45  # how much of that shadow the face gets all the way round, from contact
# with the ring rather than from the lamp: the band that says the face is under something
DIAL_WALL = 1.5  # reference px of the recess's far wall the lamp reaches down
DIAL_GLARE_AT = 0.50  # of the glass radius: where the lamp's reflection sits, towards the lamp
DIAL_GLARE_REACH = 0.42  # of the glass radius: how far it spreads. One small hard reflection,
# not a wash: a dome the size of a coin gives the lamp back from a spot the size of a pinhead
DIAL_GLARE_A = 0.21  # its alpha where it is brightest
DIAL_GLASS_A = 0.03  # the veil the glass lays over the whole face, so it is a glass and not air
DIAL_GLASS_TILT = 0.075  # ...and how much more of it the half leaning towards the lamp gives
# back than the half leaning away, which is the gradient across the pane
DIAL_GLASS_EDGE = 3.5  # reference px of the dome's edge steep enough to give the lamp back
DIAL_GLASS_EDGE_A = 0.40  # ...how brightly, where the rim faces the lamp
DIAL_GLASS_EDGE_ROOM = 0.16  # ...and what the rim on the far side still gives back, of the room
DIAL_BLOOM_R = 2.0  # reference px of skirt round the lit scale, so the arc and the graduations
DIAL_BLOOM_A = 0.80  # ...read as printed in phosphor under the glass and not inked on it
# What reaches the phosphor PRINTED ON THE FACE, which is under the glass and not on it. Every
# mark on a dial - the arc, the graduations, the pointer, the reading - is ink on the plate, so
# the bezel's shadow falls across it where the ring stands over it and the lamp's reflection lies
# over it where the crystal gives the lamp back. Painting the glass on top of a flat set of marks
# instead is what made three critics call the scale a sticker: it carried no tonal drift at all,
# three distinct luminances in eighty-three samples round the sweep.
#
# It is a MULTIPLY and not a wash, and that is the whole of why it is safe: a wash adds the same
# light to the dim track and the lit fill and closes the gap between them until the value cannot
# be read, where a multiply leaves their ratio exactly where it was at every angle. What moves is
# how much light the whole print is under, which is what a real gauge does.
DIAL_PRINT_TILT = 0.30  # of the sheen: what the pane leaning towards the lamp gives the print...
DIAL_PRINT_GLARE = 1.00  # ...and what the reflection on it does where it crosses a mark
DIAL_PRINT_SHADE = 0.20  # how much of the bezel's own shadow the print under it takes. It bites
# on the graduations, which stand furthest out, and barely at all on the sweep inside them, which
# is what makes the drift along the sweep non-monotonic instead of one ramp round the ring
DIAL_PRINT_DARK = 0.72  # the dimmest a mark may get, and this is not a taste: a lit fill dimmed
DIAL_PRINT_BRIGHT = 1.16  # past three quarters stops clearing the empty track it is drawn over
DIAL_TRACK_ROVE = 0.016  # of the radius: how far the printed sweep wanders off a true circle. A
# scale is printed on a plate, not struck by a compass, and a stroke of dead constant radius is
# the one thing on an instrument nothing physical does
DIAL_ARC_STEP = 3.0  # degrees per segment of it
DIAL_HAND_LIFT = 1.2  # how far a pointer stands off the face, which sets its shadow
DIAL_HAND_SHADOW = 0.55
DIAL_CAP = 0.25  # how far the hub's cap is pushed towards the tube's white where it catches the
# lamp. Short of where the pointer would stop reading as phosphor and start reading as held.
DIAL_CAP_R = 0.5  # of the hub: the highlight's radius...
DIAL_CAP_OFF = 0.35  # ...and how far off the hub's centre it sits, towards the lamp
DIAL_CAP_SEAT = 62.0  # degrees either side of the lamp's bearing where the cap's turned-down
DIAL_CAP_SEAT_LIT = 0.42  # edge still leans into the light, and how far towards the cap's own
# colour it comes there. It was one black ring at one value all the way round, and that is an
# OUTLINE: the one mark on an instrument otherwise built out of fields that a critic could name
# as drawn rather than lit. A cap is seated in a face, and a seat is a shadow with a side to it
DIAL_READ_LIGHT = 0.98  # the least of the lamp the reading under the hub is allowed to read by.
# The pane's own tilt bottoms out exactly where the number sits - the far corner of the face
# from the lamp - and a gauge's one number came out at two thirds brightness because of it,
# which a critic measured against the reference and called our biggest liability. The number is
# printed in phosphor and phosphor emits; what the tilt models is how much of the ROOM the
# crystal hands back to a mark, and that is not what a lit mark is made of
DIAL_READ_STRIKES = 2  # how many times the reading is struck. A glyph twelve pixels tall is
# more anti-aliased edge than it is stroke, and a hairline that arrives half covered reads as
# half a hairline; struck again, every partly covered pixel of a stroke goes opaque. It is the
# only way to put weight into this number without putting height into it, and height is what
# the gap under the hub has none of - see DIAL_READ_PT
DIAL_READ_PT = 12  # reference px of face for it. Two points down from the panel's own brand,
# and that point is what buys the whole thing: at 14 the bottom-left of "60" reached r=30.5 on a
# ring whose metal starts at 27.8, so the one number on the instrument was printed ACROSS the
# bezel holding the glass in. There is one gap on a dial and everything in it - the number, the
# speaker, the pointer swinging over both - has to fit inside the glass
DIAL_MARK = 0.139  # of the radius: the speaker in the knob's half of that gap, for the same
# reason. It was 0.17, whose bottom corner stood a pixel and a half out on the ring
DIAL_AWAY_CUTS = ("phone", "chevron", "slash", "leaving")  # the mark the knob shows while a
# companion holds his voice, in the same gap and the same box as the cone. Four cuts because this
# one is looked at rather than reasoned about - see Overlay._paint_away and factory/html/003.html
DIAL_AWAY = "phone"
DIAL_SS = 3  # the face's fields are drawn this many times over and boxed down
_DIAL_LIGHT: dict[tuple[int, float], np.ndarray] = {}  # see Overlay._dial_light: one per window
_DIAL_APERTURE: dict[tuple[int, float], np.ndarray] = {}  # ...and Overlay._dial_aperture
_DIAL_FONT: dict[int, ImageFont.FreeTypeFont] = {}  # the reading's face, one per size
# The pilot lamp, one tile per window size and appearance - a colour of None being the dead one.
# Module-level rather than per-instance, which is the same bargain _DIAL_LIGHT strikes and for a
# measured reason: a lit appearance is a whole fitting rebuilt with light in it (see
# Overlay._pilot_tile) and costs about fourteen milliseconds, and the panel builds every colour
# up front so that a state change is never the thing that drops a frame. Held per instance that
# was four builds per Overlay, which the test suite constructs a great many of - six seconds of
# it. Nothing here depends on anything but the geometry and the colour, so one is enough.
_PILOT_TILES: dict[tuple[int, int, tuple[int, int, int] | None, int], Image.Image] = {}

# The reticle: four corners on the lens axis and nothing else. It was a cross with graduations
# for about an hour, which is exactly as long as it took somebody to say it looked like a gun
# sight - and they were right. Corners say "the frame is here" and say nothing else.
RETICLE_R = 0.112  # 54 px at 480, from the middle out to any edge of the box
RETICLE_LEG = 0.5  # of that reach, per leg; the rest of the edge is the gap on the axis
RETICLE_W = 1.4  # of a hairline: marks this short need the weight back to read as drawn
RETICLE_ALPHA = 195

# What the model points with (cyclops.point). A ring is the reticle's own four corners at this
# fraction of its size - the same instrument, smaller - rather than a circle, because a circle
# round a component on a live picture is the gun sight the reticle stopped being.
MARK_R = 0.6
MARK_N_R = 0.024  # the disc a numbered marker is set in, of the panel's height
MARK_ALPHA = 215  # brighter than the resting reticle: this is being shown to somebody, now
MARK_GAP = 0.014  # of the width, between a mark and the label beside it
LABELS_KEPT = 24  # rendered labels held before the lot is dropped; see _label_tile
MARK_HEAD_R = 0.022  # an arrow's head, of the panel's height
MARK_HEAD_SPREAD = 0.45  # radians each side of the shaft; ~26 degrees, an arrow and not a dart

RIM_PERIOD_S = 3.7  # one breath of the border, slower than the caption's and not a multiple of it
RIM_DEPTH = 0.14  # how far it sinks towards SCREEN - a mix, not an alpha, and a quarter of what
# the caption may do, because this is the one thing readable across a workshop

METER_SEGMENTS = 8  # steps in the signal bar
# How much colour is stirred into the chrome for the parts that are meant to look faded. These
# are mixes rather than alphas on purpose - see _mix: drawing them translucently would not dim
# them, it would open a window onto whatever the camera is pointed at.
METER_OFF = 0.45  # an unlit signal segment, where one is still a filled slab (the slider)
METER_CELL = 0.90  # ...and the outline of an empty cell in the pod's glass, which has nothing
# inside it and has to be found against dark glass rather than against the wash
LAMP_BLOOM_R = 3.8  # reference px of the tight skirt round a lit segment or a lit tag, seen
# through the pod's glass, and...
LAMP_BLOOM_A = 1.00  # ...how much of the lamp's own alpha goes into it. The caption's
# halation, a shade tighter and brighter: a bar is a harder edge than a letter.
LAMP_HALO_R = 7.0  # ...and the soft one round that, which is the lamp lighting the glass near
# it rather than its own edge blooming, and...
LAMP_HALO_A = 0.58  # ...how much of it there is. Both baked into a tile per reading rather than
# blurred per frame - see _meter.
# Both were half this, and at half the light stopped at the segment's own edge: the gaps between
# eight lit cells measured within twenty levels of the dead glass at the far end of the window,
# which is a bar drawn on a window rather than eight lamps behind one. A lamp behind glass lights
# the glass, and the count is read off the block of light as much as off the cells in it.
# Both skirts used to be blurred out of the lamp's own RGBA, which blurs colour into the
# transparency around it as well as alpha: every pixel of skirt came back a mix of the lamp's
# colour and black, so the further out it went the *darker* the light got. A light does not do
# that. The skirt is blurred off the lamp's coverage alone now and worn in the lamp's own colour.
LAMP_SPILL = 0.62  # how much of the skirt survives past the glass, onto the flange and the rail
# below the window. Not all of it - the frame stands in front of the pane - but not none either:
# a lit instrument puts colour on the metal around it, and a window whose light stops dead at
# its own reveal is a picture of a window.
LAMP_BOUNCE_R = 13.0  # reference px of the third and widest skirt: the light the instrument
# throws on the metal below it, which is a different thing from the bloom on its own glass and
# decays over a different distance. The reference's rail runs +36 green-excess at the row
# touching the pane and is still +12 fourteen rows down - a falloff, where ours measured a flat
# +8 the whole way, which is a global tint and reads as one
LAMP_BOUNCE_A = 0.55  # ...and how much of it lands. Cut to the module: a skirt this wide runs
# off the bottom of the rail, and light haze over the camera is not a bounce
SEG_EDGE = 2.0  # px over which a lit cell falls from its core to its own edge...
SEG_FLOOR = 0.90  # ...and what is left of it there. A segment is a lens with a lamp behind it,
# so it is hottest in the middle and never one number: ours held exactly two values across a
# whole cell, and the reference holds sixty-two. It was 0.78, and 0.78 against a far wall that
# gives SEG_RIM *back* is a lens that is dim along its top edge and hot along its bottom one -
# which a critic measured as a dim band cut across the top of every lit segment and read,
# correctly, as a highlight being composited so that it subtracts. A lamp does not get darker
# where something crosses it; the falloff stays, because the cell is a lens, but it is a tenth
# now rather than a fifth and it is under the halation the cell lays outside its own rim
SEG_RIM = 0.42  # how much brighter the wall the light leaves by is than the core...
SEG_RIM_W = 1.3  # ...and how far into the cell that return reaches. On the side away from the
# lamp, like every other return on this panel
SEG_WANDER = 0.10  # how far off its own centre a cell's core sits, as a fraction of the cell...
SEG_VARY = 0.045  # ...and how much hotter or colder than its neighbours it burns, both seeded off
# the cell's own index. Two lit cells that are byte-identical are one cell blitted twice - ours
# were, to the pixel, and the reference's seven all differ
SEG_WELL = 0.52  # how far below the glass round it an empty cell's floor sits. It used to sit
# level with it - interior 40 against a field of 41 - which is a cell drawn on the mask rather
# than milled into it, and there is nothing for the lit ones to be read against
SEG_WELL_NEAR = 1.28  # of the outline's colour, the walls on the lamp's own side. A well's near
# walls are the ones the light cannot get down to - but an aperture in a mask is a *lit edge on
# all four sides*, and ours carried a stroke on the right and the bottom only, with the top and
# the left left at the floor's own value. That reads as a bevel bug rather than as an empty
# cell, and half a scale nobody can see is a scale nobody can count...
SEG_WELL_RIM = 2.35  # ...and the far ones still carry more than the near ones, because the lamp
# is a place. The reference's empty cell is a hairline at L148 round an interior at L58; ours
# measured L80 on L33, half as bright as the thing it has to be read against
SEG_WELL_RAKE = 0.72  # how much of the module's own fall an empty cell's rim takes. All of it
# and the far end of the scale dissolves; none of it and eight apertures ignore the window they
# are cut in, which is what two critics measured last round
SEG_GRAIN = 0.030  # the phosphor's own tooth inside a lit cell, running with the raster. Small,
# and the single thing that turns a cell from a handful of levels into continuous tone: ours held
# two, the reference sixty-two, and at a glance the difference is between a lit thing and a swatch
SEG_SCAN = 0.26  # of the glass's raster the emitters take. At full it quantised a lit cell to
# two values twenty-six levels apart - a 13 % stripe running straight through the lit element,
# which is the loudest thing in the window and it is an artefact
SEG_BLOOM_R = 0.62  # px the halation a lit cell lays outside its own window falls off over...
SEG_BLOOM_A = 0.44  # ...and the alpha it starts at, hard against the rim, which puts about a
# fifth of the cell's colour on the pixel next to it and nothing two pixels out. Tight,
# and drawn into the cell's own tile rather than blurred off it, because the cells are windows
# in a mask and the metal between two of them is a septum: a cell may fatten its own edge by a
# pixel and it may not cross to its neighbour. Two skirts blurred over four and seven pixels
# used to do this and they welded two lit cells into one slab - the gap between them came back
# +43 over the glass round it, a +115 halo sat on the metal above the bar, and a full-height
# column of light ran up through the top of the window. A meter that cannot be counted has no
# reason to be segmented
CLOCK_PAD = 2  # reference px round a baked clock glyph, so its own halation is not clipped by
# the edge of its tile
CLOCK_HALATION = 0.55  # how far towards the screen's own black the pixel of bleed round a glyph
# is taken, which is what a lamp behind glass has round it and paint does not...
CLOCK_CORE = 0.35  # ...and how far its middle is pushed towards the tube's white. Light that is
# bright enough desaturates; a readout that stays at exactly its own chroma however bright it
# gets is ink. Both come out of one stroked draw, baked per character - see _digit
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
    a 36 px disc twice in 170 px would be more swell than rail. The eye is sunk EYE_SEAT of its
    swell instead, and the rail runs into his collar on either side.
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
        """The mount's outline along its rail: in from one edge, round the swell of whatever is
        seated on it, out to the other.

        What the clamps and the screen's clearance are measured against. The rail itself is drawn
        straight along the spine and runs in behind a seat's housing (see Overlay._draw_bracket),
        so round a seat this is the collar's outer edge rather than any steel.

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


def _by_size(build):
    """Memoise a chassis layer on the panel's size, so every Overlay of a size shares one.

    The machined chrome - plate, surround, glass, chrome sheet, lamp filter - is a pure function
    of width and height: the same steel, the same lamp, the same seeded hairlines every time.
    Building it per Overlay cost a third of a second a panel, which is most of a test run and a
    visible slice of the kiosk's start on the Pi.

    What this buys has to be written down, because it is the thing the next change breaks: these
    layers are now SHARED between every panel of a given size, so nothing may draw into one in
    place. Composite onto a copy - :meth:`_chrome` shows the shape with
    ``self._chrome_base.copy()``, and ``Image.alpha_composite`` hands back a new image rather
    than touching either side.
    """
    memo: dict[tuple[int, int], object] = {}

    @wraps(build)
    def cached(self):
        key = (self.width, self.height)
        if key not in memo:
            memo[key] = build(self)
        return memo[key]

    return cached


@lru_cache(maxsize=64)
def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """First real TrueType face that exists on this machine; PIL's bitmap default if none do.

    Cached: six faces are asked for per panel, and reading the same TTF off the disk six times
    over is the rest of what an Overlay used to cost. A face is read from and never written to.
    """
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
    framing: Rect  # the reticle, which is how far in the module is looking


def frame_band(height: int) -> int:
    """How far in from the panel's outer edge the machined surround runs, in window pixels."""
    return max(4, round(FRAME_W * height / 480.0))


def opening_field(width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """How far outside the picture's rounded opening each pixel is, and the way out of it.

    The opening is a rounded rectangle of FRAME_RADIUS inset by :func:`frame_band` - the window
    cut in the plate. The plate's own outer edge is square, which is why the state stroke on it
    is: a light let into an arris follows the arris, and a rounded line across a square corner is
    the one thing on a frame nobody can read as a made edge. The distance is signed - positive in
    the metal, negative on the picture - and the direction is the unit vector from the nearest
    point on the opening outwards, which is what the inner chamfer's normal and the light under
    its lip are both built from. Both are :func:`halo_alpha`'s and
    :meth:`Overlay._build_surround`'s, and they have to agree to the pixel or the light lands
    beside the lip instead of under it.
    """
    band = frame_band(height)
    radius = max(2.0, FRAME_RADIUS * height)
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    half_w = max(0.0, cx - band - radius)
    half_h = max(0.0, cy - band - radius)
    xs = np.arange(width, dtype=np.float32)[None, :] - cx
    ys = np.arange(height, dtype=np.float32)[:, None] - cy
    out_x = xs - np.clip(xs, -half_w, half_w)  # from the opening's core rect to the pixel
    out_y = ys - np.clip(ys, -half_h, half_h)
    reach = np.maximum(np.hypot(out_x, out_y), 1e-6)
    return reach - radius, out_x / reach, out_y / reach


def halo_alpha(width: int, height: int) -> np.ndarray:
    """A 0..1 mask of the state light, pooled under the surround's lip and gone a little inside.

    Built once per window size. It used to be a squared ramp off the distance to the nearest
    edge, peaking on the panel's outermost pixel; there is a machined surround standing there
    now, and a bloom cannot be in front of the case it is inside. So the band starts HALO_LIP
    short of the surround's inner arris - lighting the frame's own inner chamfer, which is what
    makes the light read as coming from under a lip rather than as a wash laid over metal - and
    falls away inwards from there over the picture.

    HALO_CORE and HALO_FALLOFF are both measured in from the panel's outer edge, so their sum is
    the light's whole reach and the clearance tests can go on reading it straight off them.
    """
    edge = max(2, round(HALO_CORE * height))
    fall = max(3, round(HALO_FALLOFF * height))
    lip = max(1.0, HALO_LIP * height / 480.0)
    outside, _, _ = opening_field(width, height)
    dist = frame_band(height) - outside  # in from the panel's outer edge, square with the opening
    ramp = np.clip((edge + fall - dist) / fall, 0.0, 1.0)
    # ...and up across the chamfer rather than onto it in one step. Nothing on the face, nothing
    # on the arris the section's darkest metal is, and full where the metal ends: the undercut
    # stays the dark line it has to be and the light sits under it, which is what a cove is. A
    # one-pixel rise also walked on and off in whole steps round the corner and left a staircase.
    under = np.clip((dist - (frame_band(height) - lip)) / lip, 0.0, 1.0)
    return (under * ramp * ramp * HALO_PEAK).astype(np.float32)


def _facets(depth: np.ndarray, width: np.ndarray | float) -> np.ndarray:
    """A chamfer *width* px wide cut into FRAME_FACETS flats: 1 at the arris, 0 at the face.

    A chamfer that airbrushes from the face to the silhouette is a fillet. A milled one steps,
    and the steps - thirty counts or so apiece at this width - are what say it was cut.

    Each step is rolled over FRAME_FACET_AA of a pixel, which is not a fillet: ImageDraw does not
    anti-alias and neither does a hard threshold on a distance field, so a step that follows the
    opening's corner radius came out as a comb of one-pixel teeth under a critic's 8x crop. Sub
    pixel is the only place a step on a curve may be soft.
    """
    wide = np.maximum(width, 1e-3)
    # ...and the roll is never wider than the flat it is rolling off, which is what stops it
    # leaking out of the chamfer entirely. On a 1.3 px chamfer 0.7 px of roll came to more than
    # one flat, so a pixel nine deep - the far chamfer, the darkest metal there is - still read a
    # sliver of crown and was shaded as one. That flattened the whole section by half.
    soft = np.minimum(FRAME_FACET_AA * FRAME_FACETS / wide, 1.0)
    level = np.clip(1.0 - depth / wide, 0.0, 1.0) * FRAME_FACETS
    flat = np.floor(level)
    roll = np.clip((level - flat - 1.0) / soft + 1.0, 0.0, 1.0)
    return np.minimum((flat + roll) / FRAME_FACETS, 1.0)


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


def _glass(edge: np.ndarray, bezel: float) -> np.ndarray:
    """Coverage of the glass: everything inside the frame's inner edge, anti-aliased at it.

    Every pass laid on the glass is cut by this and nothing laid on the frame is, so the frame
    is the one place the two meet - and it is steel all the way through, so there is no seam.
    A *bezel* wider than the frame's own is how the reflections are held off the moulding: see
    :data:`TUBE_INSET`, which is the only reason this takes a float.
    """
    return np.clip(-edge - bezel + 0.5, 0.0, 1.0)


def _outward(edge: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The unit direction out of the shape at every pixel: the normal of anything built on it."""
    gy, gx = np.gradient(edge)
    length = np.maximum(np.hypot(gx, gy), 1e-6)
    return gx / length, gy / length


def _pits(width: int, height: int, count: int, seed: int, scale: float) -> np.ndarray:
    """Dark specks in a plate, 0..1: whatever has been dropped on it over the years.

    Scratches are the wear anybody notices and pits are the wear nobody does, which is exactly
    why a plate needs both - a face carrying only lines reads as a face that has been drawn on.
    """
    rng = np.random.default_rng(seed)
    sheet = Image.new("L", (width, height), 0)
    d = ImageDraw.Draw(sheet)
    for _ in range(count):
        x, y = rng.uniform(0.0, width), rng.uniform(0.0, height)
        r = rng.uniform(0.4, 1.2) * scale
        d.ellipse([x - r, y - r, x + r, y + r], fill=int(rng.uniform(110, 255)))
    return np.asarray(sheet, np.float32) / 255.0


def tube_alpha(width: int, height: int, radius: int) -> np.ndarray:
    """Coverage of the front: opaque glass, out to a pixel short of the frame's own edge.

    The glass is opaque, which is what the module docstring has always said it was - it lists
    this pane with the dial wells among the parts you cannot see through - and what it actually
    was until now is 80% of the way there. That last fifth is a whole defect on its own: stretch
    the contrast of the pane and the workshop is in it, a finger and a clamp knob behind the
    caption, and the green the camera shows through competes with the green the letters are made
    of. A screen you can see the room through is a hole in the panel with a lamp behind it.

    What is under the frame does not matter, because the frame is steel all the way through;
    except at its outer edge, where its anti-aliased pixel has to meet the room and not a dark
    fringe, so the backing stops a pixel short of it.
    """
    return np.clip(-0.5 - tube_field(width, height, radius), 0.0, 1.0)


def tube_glow(width: int, height: int, radius: int, bezel: int, top: int) -> np.ndarray:
    """The phosphor the glass is made of, as a colour at every pixel of the front.

    Three things, and the first two are the ones that were missing. A vertical fall from
    :data:`TUBE_TOP` to :data:`TUBE_BOTTOM`, which is the curve: a tube read from a little above
    reflects the lit room in its top half and the floor in its bottom, and a pane at one
    brightness is a texture however much is drawn on it. A lift where the lamp stands in the
    glass, up and to the left, falling away across and down - so the pane is *lit* rather than
    filled, and the marks on it come up in the light and vanish out of it. And the tube's own
    raster, on the panel filter's pitch and in its phase, which is why *top* is an argument
    rather than a detail: phased any other way the two beat against each other and the one
    surface here that is literally a CRT ends up the flattest thing on the panel.

    The lift is the phosphor's own colour turned up rather than white laid over it. Both look
    the same at a glance and only one of them survives the caption: white doubles the red and
    blue for the luminance it buys, and the line printed on this glass is found by summing the
    channels (tests/test_caption.py). Green is the cheap way to be bright here, which is also
    the true one - it is the tube that is brighter over there, not the room.
    """
    ys = np.arange(height, dtype=np.float32)[:, None]
    xs = np.arange(width, dtype=np.float32)[None, :]
    # ...raked, so the fall runs across the pane as well as down it. A tube read from a little
    # above AND a little to the right of the lamp does not shade in horizontal bands: its iso
    # lines lean, and a field whose every row is one value is the thing that measures as "one
    # linear gradient". The rake is in the same direction as the lamp's own fall, so the ladder
    # down and the ladder across still both run the one way and the pane never brightens away
    # from the light.
    across = xs / max(1.0, float(width)) - 0.5
    down = np.clip((ys - bezel) / max(1.0, height - 2 * bezel), 0.0, 1.0)
    # Leaned as a gamma on the ramp and not as an offset to it, so that the fall still runs the
    # pane's whole depth at every point along its length. Offset instead, the left third of the
    # glass spends its first ten rows clipped at the top colour and comes out FLATTER than the
    # thing this is here to fix.
    down = down ** (1.0 - 2.0 * TUBE_RAKE * across)
    down = down * down * (3.0 - 2.0 * down)
    face = np.asarray(TUBE_TOP, np.float32) + (
        np.asarray(TUBE_BOTTOM, np.float32) - np.asarray(TUBE_TOP, np.float32)
    ) * down[..., None]
    cx, cy = TUBE_LIT_X * width, TUBE_LIT_Y * height
    rx, ry = max(1.0, TUBE_LIT_W * width), max(1.0, TUBE_LIT_H * height)
    reach = np.sqrt(((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2)
    lit = np.exp(-(reach ** TUBE_LIT_FALL))
    face = face * (1.0 + TUBE_LIT_A * lit)[..., None]
    rows = (np.arange(height) + top) % SCANLINE_EVERY == 0
    raster = np.zeros((height, 1), np.float32)
    raster[rows] = TUBE_SCAN
    face = face * (1.0 - raster)[..., None]
    # ...and the tooth under all of it: grains of phosphor behind a shadow mask, a level or two
    # either way, the same on every channel so it is brightness and not colour noise. Without it
    # the pane posterises into plateaus - whole rows of one value - which is the single loudest
    # thing that says a surface was filled rather than photographed.
    rng = np.random.default_rng(material.SEED + 11)
    tooth = rng.uniform(-1.0, 1.0, (height, width, 1)).astype(np.float32)
    speck = rng.uniform(-1.0, 1.0, (height, width, 3)).astype(np.float32)
    return face * (1.0 + TUBE_GRAIN * (tooth + TUBE_SPECK * speck))


def rebate_gain(width: int, height: int, radius: int, bezel: int, scale: float) -> np.ndarray:
    """How much of its own light the glass keeps at every pixel, for being sunk in a rebate.

    A GAIN, and that is the whole idea. What was here before laid black over the pane at a fixed
    alpha, which paints a floor: a stripe of one level for five or six rows and then a jump back
    to whatever was underneath, with nothing in the picture actually casting it. A shadow is
    multiplicative. The same curve has to fall across the phosphor, the room's wipe and the
    corner blowout alike and come back out of it over the same distance wherever it lands - so
    this is one number per pixel, applied to the finished pane, and every pass above it stays
    honest about its own light.

    Four walls, one curve each, and they differ only in three numbers: how far in from the wall
    the trough sits, how wide it is, and how much it takes. Evaluated on the pane's own distance
    field so it follows the corner radius, and blended between the walls by which of them is
    nearest - which is what puts both of a corner's shadows into the corner without either of
    them running along the wrong side.

        top     a CAST shadow, and the only one that does not start on its own boundary: the
                first two px of glass are the frame's lit inner face reflected in it
                (:data:`SHEEN_STRIP`), and an overhang casts below what it is mirrored in. So
                the pane reads lift, trough, field going down - the recess signature.
        bottom  the deepest and the narrowest. Not cast at all: the floor of a well under a lamp
                from above is where no light gets, and the last four rows go to a quarter of
                field.
        ends    the widest, because the end wall of a letterbox is the tallest thing round it.

    ...and one band that goes the other way, on the ends only: :data:`TUBE_WALL`, the polished
    wall bouncing the lamp back onto the pane a dozen px in. Without it the recovery from the
    contact shadow overshoots into a flat field and stops, which measures as a dead 45 for
    forty-five px; with it the column means lift and settle, which is what the reference does.
    """
    edge = tube_field(width, height, radius)
    ys = np.arange(height, dtype=np.float32)[:, None] + 0.5
    xs = np.arange(width, dtype=np.float32)[None, :] + 0.5
    # Distance to each wall's own line, which is only used to decide WHICH wall a pixel belongs
    # to; how deep it is there is measured on the true field below, so the corners stay round.
    walls = (ys - bezel, (height - bezel) - ys, xs - bezel, (width - bezel) - xs)
    walls = tuple(np.broadcast_to(w, (height, width)) for w in walls)
    nearest = np.minimum(np.minimum(walls[0], walls[1]), np.minimum(walls[2], walls[3]))
    blend = max(1.0, TUBE_BLEND * scale)
    share = [np.exp(-np.maximum(w - nearest, 0.0) / blend) for w in walls]
    total = sum(share)
    share = [s / total for s in share]
    ends = share[2] + share[3]

    def mix(top: float, bottom: float, side: float) -> np.ndarray:
        return share[0] * top + share[1] * bottom + ends * side

    at = mix(TUBE_LIP, TUBE_FOOT, TUBE_SIDE_AT) * scale
    soft = np.maximum(mix(TUBE_FEATHER, TUBE_FOOT_W, TUBE_SIDE) * scale, 0.6)
    deep = mix(TUBE_LIP_A, TUBE_FOOT_A, TUBE_SIDE_A)
    d = np.maximum(-edge - bezel, 0.0)  # how far in from the aperture, on the glass
    sink = deep * np.exp(-(((d - at) / soft) ** 2))
    # ...and the wall's own bounce, which is a reflection and so cannot outlive its source. The
    # near wall stands in the lamp and throws a little of it back onto the pane; the far one is
    # in the frame's own shadow and has nothing to throw. Left at one strength along the length
    # it lifted the last column of the pane back above the one before it - a local maximum in
    # the corner furthest from the lamp, which is the single fault this glass has been winning
    # on other people having.
    reach = max(1.0, GLARE_REACH * height)
    near = 1.0 / (1.0 + ((xs - GLARE_X * width) ** 2 + (ys - GLARE_Y * height) ** 2) / reach ** 2)
    lift = TUBE_WALL * ends * (near / near.max()) * np.exp(
        -(((d - TUBE_WALL_AT * scale) / max(1.0, TUBE_WALL_W * scale)) ** 2)
    )
    return (1.0 - sink) * (1.0 + lift)



def glare_alpha(width: int, height: int, radius: int, bezel: int, scale: float) -> np.ndarray:
    """The room, wiped across the glass: the streak a convex face smears the lamp into, and the
    polishing marks that only show where it lands.

    It is :func:`material.glare` - the same reflection every sheet of glass on the panel gives
    back - cut to the pane and held :data:`TUBE_INSET` px clear of the moulding. Not over the
    frame: that is steel, lit by its own normals under the same lamp, and a reflection painted
    over a bar that is already lit is two lamps. The inset is not fussiness either; without it
    two of these hairlines ran across the top rail at the same brightness as the ones on the
    glass, which reads as a decal over the whole assembly rather than as wear on one part of it.

    The wipe starts a quarter of the way down the left edge rather than in the corner. The
    corner is where the phosphor is brightest already, and stacking a white streak on top of the
    one bright quarter of the pane is both the flattest place to put it and the only place on
    this glass where a reflection could put a word out.

    The hairlines are what a real screen's glare is mostly made of at arm's length. Few, short,
    clustered, all running the one way a cloth was dragged, softened by a fraction of a pixel
    because a scratch in glass seen through the glass is never crisp, and weighted by the lamp's
    own light without its ambient floor, so they come up in the wipe and vanish in the dark end.
    """
    edge = tube_field(width, height, radius)
    lamp = (GLARE_X * width, GLARE_Y * height)
    lit = material.glare(width, height, lamp, GLARE_REACH * height, GLARE_AMBIENT,
                         (GLARE_AT, GLARE_DEPTH, GLARE_TILT))
    heading = math.radians(GLARE_MARK_TILT)
    # Drawn on their own patch of the pane and pasted, rather than sown over the whole box. A
    # cloth is wiped once in one place: eleven marks spread over a letterbox this wide are eleven
    # separate accidents, and a cluster is what says a hand was here.
    mx, my, mw, mh = GLARE_MARK_BOX
    bx, by = int(mx * width), int(my * height)
    bw, bh = max(8, int(mw * width)), max(8, int(mh * height))
    marks = np.zeros((height, width), np.float32)
    marks[by:by + bh, bx:bx + bw] = material.scratches(
        bw, bh, GLARE_MARKS, (math.cos(heading), math.sin(heading)),
        seed=material.SEED + 3, spread=GLARE_MARK_SPREAD,
        length=(GLARE_MARK_LEN[0] * scale, GLARE_MARK_LEN[1] * scale),
    )
    soft = Image.fromarray((marks * 255.0).astype(np.uint8), "L")
    soft = soft.filter(ImageFilter.GaussianBlur(max(0.3, GLARE_MARK_BLUR * scale)))
    marks = np.asarray(soft, np.float32) / 255.0
    # The marks are lit by how much of the lamp reaches them and not by the wipe, which is a
    # narrow band and would show three of them and hide the rest. What a cloth leaves is all
    # over the pane; what decides whether you can see one is where the light is.
    caught = material.glare(width, height, lamp, GLARE_REACH * height, 0.0)
    ys = np.arange(height, dtype=np.float32)[:, None]
    xs = np.arange(width, dtype=np.float32)[None, :]
    halo = GLARE_HALO * np.exp(
        -(((xs - lamp[0]) / max(1.0, GLARE_HALO_W * width)) ** 2
          + ((ys - lamp[1]) / max(1.0, GLARE_HALO_H * height)) ** 2)
    )
    pane = _glass(edge, bezel + TUBE_INSET * scale)
    return (lit * GLARE_ALPHA + halo + marks * caught * GLARE_MARK_A) * pane


def sheen_alpha(width: int, height: int, radius: int, bezel: int, scale: float) -> np.ndarray:
    """What the pane's own curve gives back: the frame's lit lip along the top, and the lamp.

    Two things, and they are not the same kind of thing. A picture tube's face is convex: in its
    middle it is square to the viewer, so it mirrors the dark room and returns nothing, and only
    at the top rim has it turned far enough towards the lamp to show anything. What it shows
    there is the frame's own lit inner face - a band, running the whole length of the pane and
    dying along it as the lamp gets further away. That is the RIM, and a rim has no shape of its
    own; it is an edge condition.

    What the pane also has to show is the SOURCE, and a source does have a shape: one ellipse,
    wide and shallow, high on the glass near the corner the lamp stands over, falling to field
    over a few px. Painted as the place two rims happen to cross - which is what was here - the
    brightest thing on the glass has no outline, and a critic measuring a twenty-pixel block of
    it found "97 to 102 in every row, flat to +/-2 lum, one linear gradient": a pane that reads
    as tinted vector fill rather than as lit glass. A reflection you can put a ruler round is
    most of the difference.

    The other three rims get nothing. They used to get a flat bounce off the bench, and the bounce
    is what put a 1.53x stroke of white on the right-hand end of the pane and a 1.15x lift along
    its bottom - a BRIGHT line standing exactly where the rebate's darkest one has to be. An
    aperture outlined in light is a decal; the same aperture terminating in shadow is a pane in
    a rebate, and there is nothing else on this glass that says which of the two it is. What
    lifts the far end back up now is the wall's own bounce a dozen px in (:data:`TUBE_WALL`),
    which is broad, set back, and under the frame's shadow rather than on top of it.

    ONE specular, and it is held in the strip of glass above the first line of type. The rim is a
    band in one coordinate rather than a distance from the pane's outline, and the specular is an
    ellipse pinned a px under the lip, for the same reason: ridden round the corner radius
    instead, either would cross the line, and a reflection over a word is a word lost - past a
    brightness the panel's own reader takes one for a letter (tests/test_caption.py).
    """
    edge = tube_field(width, height, radius)
    xs = np.arange(width, dtype=np.float32)[None, :]
    ys = np.arange(height, dtype=np.float32)[:, None]
    sigma = max(0.8, SHEEN_W * scale)
    inset = SHEEN_IN * scale

    def rim(coord: np.ndarray, at: float) -> np.ndarray:
        return np.exp(-(((coord - at) / sigma) ** 2))

    lit_top, lit_spec = SHEEN_LIT
    top = rim(ys, bezel + inset) * lit_top
    # ...and ONE specular: the lamp's own image in the face, which is an ellipse with an edge to
    # it rather than the place two rims happen to cross. A rim band is what the pane gives back
    # of the frame it is set in and it runs the whole length; a specular is what it gives back of
    # the SOURCE, and a source has a size, a shape and a falloff to field over a few px. Without
    # one the pane measures flat to a couple of levels over a twenty-pixel block and reads as
    # tinted fill however carefully the field under it is graded.
    spec = lit_spec * np.exp(
        -(((xs - SHEEN_SPEC_X * width) / max(1.0, SHEEN_SPEC_W * width)) ** 2
          + ((ys - bezel - SHEEN_SPEC_AT * scale) / max(0.8, SHEEN_SPEC_H * scale)) ** 2)
    )
    # How much of the lamp reaches each point of the rim at all. The lamp is the same point the
    # rest of the glass reflects (GLARE_X, GLARE_Y), off the top-left corner, so the blowout is
    # in the corner and the strip under the top lip fades along the length - which is the only
    # thing in this pane that says which way the light is coming from.
    reach = max(1.0, SHEEN_REACH * height)
    lx, ly = GLARE_X * width, GLARE_Y * height
    near = 1.0 / (1.0 + ((xs - lx) ** 2 + (ys - ly) ** 2) / reach ** 2)
    # ...measured against the corner of the pane itself, which is the nearest the glass gets to
    # the lamp and therefore the one place the reflection is allowed to blow right out.
    corner = bezel + inset
    peak = 1.0 / (1.0 + ((corner - lx) ** 2 + (corner - ly) ** 2) / reach ** 2)
    # Screened, so where the specular sits on the rim the pane returns more than either alone
    # without either of them clipping, and taken down the length by the lamp's own distance - to
    # a floor, because the rim goes on reflecting the lit frame at the far end of the pane even
    # where the lamp itself does not reach.
    lamp = (top + spec - top * spec) * (
        SHEEN_STRIP + (1.0 - SHEEN_STRIP) * np.clip(near / peak, 0.0, 1.0)
    )
    return np.minimum(lamp, 1.0) * SHEEN_A * _glass(edge, bezel)


def tube_frame(
    width: int, height: int, radius: int, bezel: int, roll: float, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """The steel frame round the glass, as an (rgb, coverage) pair the layer can composite.

    A bar of the panel's steel bent round the tube, off the same distance field as the glass so
    the two can never drift apart. What is drawn is not four shaded rails but one CROSS-SECTION
    carried round a rounded rectangle, and the lamp decides what each side of it looks like.

    Read from the outside in, that section is: an edge rolled down to the panel, a crowned
    face, and the inner edge rolled down into the well. It is symmetric, and every asymmetric
    thing about the finished part comes from :data:`material.LAMP` standing up and to the left
    of it. Two of those are worth naming, because the round before this got both wrong.

    The first is that this ring gets ONE specular, and it goes on the top rail's outer roll -
    the face turned towards the lamp. The bottom rail's inner roll is turned up into the well and
    was lit as brightly, on the argument that an inset frame lights both of its up-facing edges.
    It does, but not equally and not independently: what reaches the underside of a ring is a
    bounce off the lit glass and the panel rather than the lamp itself, so it is a fraction of the
    top rail's light and it dies along the length exactly as the top rail's does. Lit twice
    instead, the part measured 224 along the top rim against 223 along the bottom over the whole
    of x=290-590 - a ratio of 1.00 at every sample, and 1.3 past x=550 - with two different
    falloffs, which is the signature of one member lit by two people. See :data:`TERM_BOUNCE`,
    which is the fraction, and `crest`, which is the borrowed run. The reference's own readout
    resolves the same corner far harder than this does: its bottom moulding sits at 12-27 against
    a 213 top rim, an eighth, where ours holds between a fifth and three tenths.

    (What must not come back with it is the earlier mistake at the other end: the outset
    convention - light along the top edge, dark along the bottom - left the bottom rail peaking
    at L 56 against the top rail's 147 and sitting within a few levels of the glass beside it,
    so at a pace the readout lost its lower edge and went soft. A capped bounce is not that: it
    is still the up-facing inner roll that carries what light there is, and it still steps off
    into the well.)

    The second is the quirk groove (:data:`TERM_QUIRK`), which is not a shape but a shadow:
    a bead standing proud of a face under a low lamp cannot light the strip of face directly
    behind its own crest. That is why the reference's top rail has a dip inside its outer roll
    and its bottom rail one outside its inner roll and neither has both, and it is what makes
    the bar read as two machined surfaces instead of one gradient - a crest, a dip, then a
    broader second lobe. With the rolls it gives the face a profile worth measuring across: a
    blown ridge, a mid-tone, a terminator, and the contact shadow outside it that
    :meth:`_draw_terminal` lays down.

    Nothing here is flat along its length either. One lamp at one point (:data:`TERM_REACH`)
    lights the near end of the bar two thirds again as brightly as the far end and takes the
    specular with it; the whole section wanders a third of a pixel in and out (:data:`TERM_WOBBLE`)
    so the crest is not the same row from one end to the other; and the steel is brushed, worn
    brighter and duller by turns, scratched, and pitted - with the scratches taking their brightness
    from the light already on that piece of face, so they go dark at the dark end of the bar instead
    of crossing the whole part at one grey.
    """
    edge = tube_field(width, height, radius)
    gx, gy = _outward(edge)
    ys = np.arange(height, dtype=np.float32)[:, None]
    xs = np.arange(width, dtype=np.float32)[None, :]
    along = xs * -gy + ys * gx
    # How far in from the outer silhouette each pixel is, and out from the glass - the pair the
    # section is read against, wandering a fraction of a pixel along the length, because rolled
    # stock is not one thickness and a specular pinned to one row for the whole run of a part is
    # the thing that reads as a gradient extruded along a shape.
    inward = np.maximum(-edge, 0.0)
    section = inward + TERM_WOBBLE * scale * material.wear(along, seed=material.SEED + 4)
    outward_ = float(bezel) - section
    # The section, as one signed lean: + where the surface turns out of the frame, - where it
    # turns into it. Two rolls turning opposite ways, and a crowned face between them.
    out_roll, in_roll = max(1.0, roll), max(1.0, TERM_REVEAL * scale)
    quirk = max(0.6, TERM_QUIRK_AT * scale)
    lean = TERM_TURN * (
        np.clip(1.0 - section / out_roll, 0.0, 1.0) - np.clip(1.0 - outward_ / in_roll, 0.0, 1.0)
    )
    lean = lean + TERM_CROWN * (1.0 - 2.0 * np.clip(section / max(1.0, float(bezel)), 0.0, 1.0))
    # The section as an angle rather than as a tilt, so a roll can turn PAST end-on and go dark.
    # A round-over does not stop at vertical, it carries on down onto the panel, and its last
    # pixel is turned away from the lamp: which means the crest is not the outermost pixel but
    # whichever one has come round to face the lamp, and THAT moves as the section wanders. It is
    # the whole of why the crest is no longer a constant-offset line pinned to one row.
    turn = lean * (math.pi / 2.0)
    nx, ny, nz = gx * np.sin(turn), gy * np.sin(turn), np.cos(turn)
    tilt = np.abs(np.sin(turn))
    diffuse, spec = material.shade(nx, ny, nz)
    spec = spec * TERM_GLOSS * (1.0 + TERM_WEAR * material.wear(along))
    texture = material.grain(inward, along) + TERM_DRIFT * material.wear(
        along, seed=material.SEED + 7
    )
    rgb = material.steel(diffuse, spec, texture)
    # How much of the lamp reaches each point of the part at all: one inverse-square fall from
    # one point, the same point the glass reflects. It replaces a pair of straight ramps that
    # shaded this top-to-bottom and left-to-right, and it reverses which of those matters. A
    # bench lamp off the top-left corner is nearly as far from the bottom rail as from the top
    # one and three times as far from the right end as from the left, so the fall runs along the
    # LENGTH: the left end of the frame comes out about twice the right, and the far end of the
    # top rail's specular keeps about half of what its near end has. The old ramps had it 1.86
    # top-to-bottom and 0.94 end-to-end, which is a dome overhead and reads as painted plastic.
    reach = max(1.0, TERM_REACH * height)
    lamp_x, lamp_y = GLARE_X * width, GLARE_Y * height
    near = 1.0 / (1.0 + ((xs - lamp_x) ** 2 + (ys - lamp_y) ** 2) / reach ** 2)
    # ...and then the half of the ring that never sees the lamp at all. What lights the bottom
    # rail is what the lamp lights ABOVE it, so both the level and the run along the length are
    # the top rail's: a bounce is a fraction of a lit surface and it has to die where the surface
    # it bounces off does. Measuring it at the crest's own row rather than at its own is the
    # whole of what stops the two rims having falloffs of their own - see :data:`TERM_BOUNCE`.
    down = np.clip(gy, 0.0, 1.0)  # 1 on the faces turned away from the lamp, 0 along the top rail
    crest = 1.0 / (1.0 + ((xs - lamp_x) ** 2 + (0.5 * bezel - lamp_y) ** 2) / reach ** 2)
    near = near * (1.0 - down) + crest * down
    bounce = 1.0 - (1.0 - TERM_BOUNCE) * down
    fall = TERM_FACE * bounce * (TERM_FALL_FLOOR + (1.0 - TERM_FALL_FLOOR) * near / near.max())
    # ...and the quirk groove each rolled edge throws back across the face. It is an occlusion
    # and not a shape: the bead stands proud, the lamp is low across it, and the strip of face
    # immediately behind a LIT crest is the one place on the bar the light cannot reach. Behind
    # an unlit crest there is nothing to occlude, which is why the reference's top rail has its
    # dip inside the outer roll and its bottom rail has one outside the inner roll and neither
    # has both. Modelled as tilt instead, each groove put a second bright hairline on whichever
    # rail its far wall happened to face the lamp on.
    lx, ly = material.lamp_2d()
    faces = gx * lx + gy * ly  # +1 where a rail's outer edge is the one turned to the lamp
    groove = TERM_QUIRK * (
        np.clip(faces, 0.0, 1.0) * np.exp(-(((section - out_roll - quirk) / quirk) ** 2))
        + np.clip(-faces, 0.0, 1.0) * np.exp(-(((outward_ - in_roll - quirk) / quirk) ** 2))
    )
    rgb = rgb * (fall * (1.0 - groove))[..., None]
    marks = material.scratches(width, height, TERM_SCRATCHES, (1.0, 0.0), seed=material.SEED + 5,
                               length=(12.0 * scale, 60.0 * scale))
    pits = _pits(width, height, TERM_PITS, material.SEED + 9, scale)
    rgb = rgb * (1.0 + TERM_SCRATCH * marks - TERM_PIT_A * pits)[..., None]
    rgb = np.minimum(rgb, np.asarray(material.STEEL_SPEC, np.float32))
    # ...and the well's own shade across the inner half of the bar: a frame stands over a hole,
    # and the nearer its inner edge, the less of the room any part of it can see. It was three
    # times this deep, which took the face from 135 to 47 in six pixels and left the ring
    # looking like two different parts; the inner roll is the seam now, so this only has to
    # grade the face towards it.
    depth = np.clip(section / max(1.0, float(bezel)), 0.0, 1.0)
    well = np.clip((depth - TERM_WELL_FROM) / max(1e-3, 1.0 - TERM_WELL_FROM), 0.0, 1.0) ** 2
    rgb = rgb * (1.0 - TERM_WELL * well)[..., None]
    # Last, and over the steel's own ceiling: the arris. Where a rolled edge is turned end-on to
    # the lamp it stops scattering and starts MIRRORING, and what a mirror returns is the source
    # - :data:`GLARE_LAMP`, the same off-white the glass gives back two pixels away. That is the
    # difference between a highlight and a blown one: material.STEEL_SPEC is as bright as lit
    # steel gets and the reference's ridge runs half a stop above it, because it is not steel you
    # are looking at there, it is the lamp. One pixel wide, on the two up-facing edges only, and
    # taken down the length by the lamp's own distance so it decays instead of striping the part.
    lit_edge = np.clip((nx * lx + ny * ly) / np.maximum(tilt, 1e-6), 0.0, 1.0)
    carry = TERM_ARRIS_FALL + (1.0 - TERM_ARRIS_FALL) * near / near.max()
    # ...and only on the half of the ring the lamp can actually reach. A mirror returns what is
    # in front of it, and there is nothing in front of a down-facing arris but the bench.
    blown = (TERM_ARRIS * (tilt * tilt * lit_edge * lit_edge) ** TERM_ARRIS_TIGHT * carry
             * (1.0 - down))
    rgb = rgb + (np.asarray(GLARE_LAMP, np.float32) - rgb) * blown[..., None]
    cover = np.clip(0.5 - edge, 0.0, 1.0) * np.clip(bezel - inward + 0.5, 0.0, 1.0)
    return rgb, cover


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


def _stroke_sdf(
    xs: np.ndarray, ys: np.ndarray, runs: Sequence[Sequence[tuple[float, float]]]
) -> np.ndarray:
    """Distance from every pixel of an (xs, ys) grid to the nearest of *runs*.

    The centrelines of an engraving. A single-line letterform has no outline to fill: the tool
    walks the path and the stroke is as wide as the tool, so the whole glyph is this distance
    thresholded at half the cutter's width - and the same field gives the walls of the cut their
    direction, which is what lights them.
    """
    best = np.full(np.broadcast(xs, ys).shape, np.inf, np.float32)
    for run in runs:
        for (ax, ay), (bx, by) in zip(run, run[1:], strict=False):
            dx, dy = bx - ax, by - ay
            along = np.clip(((xs - ax) * dx + (ys - ay) * dy) / max(dx * dx + dy * dy, 1e-6),
                            0.0, 1.0)
            best = np.minimum(best, np.hypot(xs - (ax + along * dx), ys - (ay + along * dy)))
    return best


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
        self.frame_w = frame_band(height)
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
        # Centred in the pane rather than hung off its top: at 22 the bar had twelve pixels of
        # glass over it and three under, which reads as a part fitted in the wrong hole however
        # well the part is made.
        self.row = round(19 * scale)
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
        # Kept, because the USB module in the other corner is cut to the same section.
        self.pod_depth, self.pod_step, self.pod_ramp = depth, step, ramp
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
        # Built once and composited wherever it is pointing; see _corners. Per instance rather
        # than through _by_size because it is about a hundred pixels square and an Overlay is
        # only rebuilt when the window changes size.
        self._reticle = self._corners(self.reticle_r, RETICLE_ALPHA)
        # A mark's two fixed shapes, kept at full and faded by scaling alpha at composite time.
        # Measured on the Pi: rebuilding these per frame as the gesture faded cost ~2 ms a mark
        # on a loop that has 33 ms for everything, because each one is a supersampled draw. A
        # kept tile and a LUT over its alpha channel is the same picture for a tenth of that -
        # the same bargain _meter() and _digit() strike, and the same one that let REC blink.
        self._mark_r = max(6, round(RETICLE_R * MARK_R * height))
        self._mark_ring = self._corners(self._mark_r, MARK_ALPHA)
        self._mark_disc = self._disc_tile(max(6, round(MARK_N_R * height)), MARK_ALPHA)
        self._labels: dict[str, tuple[Image.Image, int]] = {}  # see _label_tile

        # Him, riding the big bracket's ramp. Everything about where he is comes off that ramp,
        # so moving the bracket moves him and the rail still meets his collar either side.
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
        # ...and the pilot lamp on the plate under them, square between the two. Off the ramp
        # like they are, so it moves with the mount rather than sitting at a pixel someone typed.
        self.pilot_r = max(4, px(PILOT_R))
        self.pilot = self.brackets["br"].on_ramp(0.0, px(PILOT_DEPTH))
        # How far its light carries past the fitting. The tile is built at this reach, so this is
        # the one number that decides whether the glow can reach a dial - see PILOT_DEPTH.
        #
        # Whichever carries further, the light or the shadow the fitting drops, and then capped
        # at the room the corner actually has. The cap is measured off the two dials rather than
        # typed: a hitbox is a square round a disc, so what the lamp has to clear is the near
        # EDGE of each square, and the panel asserts in test_eye.py that both of those squares
        # are byte-identical in every state. A tile overlapping one would fail that test even
        # where its light is numerically zero, because the assertion is on bytes not brightness.
        lift = max(1.0, PILOT_LIFT * scale)
        spread = math.ceil(max(
            self.pilot_r + lift * (material.SHADOW_DROP + 3 * material.SHADOW_SOFT),
            self.pilot_r * PILOT_BEZEL_IN + PILOT_THROW_REACH * max(1.0, PILOT_THROW * scale),
        )) + 1
        room = min(self.pilot[0] - (self.switches[VOLUME][0] + self.btn_r),
                   self.pilot[1] - (self.switches[HEAT][1] + self.btn_r))
        self.pilot_reach = max(self.pilot_r + 1, min(spread, math.floor(room) - 1))
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
        # TERM_W wide and centred, but never closer than TERM_CLEAR to either mount's rail -
        # measured where each rail actually is at the monitor's own mid-height rather than off the
        # spine table, which is what puts the clearance where somebody looking at it would see it.
        middle = (roof + floor_) / 2.0
        self.ear_y = middle
        clear = max(4, px(TERM_CLEAR)) + self.rail_w / 2.0
        half_w = min(px(TERM_W) / 2.0, width / 2.0 - (self._rail_at("bl", middle) + clear),
                     self._rail_at("br", middle) - clear - width / 2.0)
        left = round(width / 2.0 - half_w)
        right = width - left
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

        # One set of hairlines for the whole sheet, before the filter and the chrome are built
        # from it: the plates wear them and so do the bars bolted across them, and a scratch that
        # runs from one onto the other is most of what says they are the same piece of metal.
        self._marks = material.scratches(width, height, PLATE_SCRATCHES, (1.0, -1.0))
        self._filter = self._build_filter()
        self._backdrops: dict[int, Image.Image] = {}
        self._chromes: dict[int, Image.Image] = {}
        # One signal-bar tile per (segments lit, accent), built the first time that reading is
        # shown and kept: the glow round a lit segment is a blur, and a blur is not a frame cost.
        self._meters: dict[tuple[int, tuple[int, int, int]], tuple[Image.Image, int]] = {}
        # ...and the walkthrough's bar, the same idea: one tile per (steps, steps lit), halation
        # baked in, so what a frame pays for it is one composite. Ten by ten at most.
        self._bars: dict[tuple[int, int], Image.Image] = {}
        # ...and one tile per character of the clock, for the same reason and a harder one: the
        # digits change every second, so they cannot be baked into the strip, and PIL's stroked
        # text - which is how a glyph gets its halation - costs three times a plain one. Twelve
        # characters by a handful of accents is a bounded cache; a tile per *string* would be
        # three thousand six hundred of them.
        self._digits: dict[tuple[str, tuple[int, int, int]], tuple[Image.Image, int]] = {}
        # ...and the record light, for the same reason one step further on: it blinks, so it
        # cannot be baked into the strip either. Keyed on the tag count alone, because that is
        # the whole of what moves it - the word and the colour are constants and the x comes off
        # _readouts - which makes this a cache of at most two small tiles.
        self._rec: dict[int, tuple[Image.Image, tuple[int, int]]] = {}
        # ...the USB module, one per width it comes to, and one tile per thing in it. A width
        # is a handful of numbers and a bench is a handful of cables, so both are bounded by what
        # is actually plugged into the box rather than by anything here.
        self._usb: dict[float, Image.Image] = {}
        self._usb_entries: dict[tuple[str, str], Image.Image] = {}
        self.glow_r = max(1.0, LAMP_BLOOM_R * scale)
        self.halo_r = max(1.0, LAMP_HALO_R * scale)
        self._skirt = math.ceil(3 * self.halo_r)  # how far a lamp's light reaches past its edge
        self._plate = self._build_plate()
        self._surround = self._build_surround()
        self._glass = self._build_glass()
        self._chrome_base = self._build_chrome()
        # Every colour the lamp can burn, built now rather than on the state change that first
        # asks for one. A lit appearance is a whole fitting (see _pilot_tile) and costs about
        # fourteen milliseconds here, which is half a frame on this machine and more than a whole
        # one on the Pi - so built lazily it is a dropped frame at the exact moment somebody has
        # just pressed the button and is watching. There are four of them, they are 85 px square,
        # and this is the constructor that already builds the whole chassis.
        for lamp in {p.colour for p in LAMPS.values() if p.colour is not None}:
            self._pilot_tile(lamp, PILOT_STEPS)
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
        self._knobs: dict[tuple[int | None, bool, bool], Image.Image] = {}
        self._needles: dict[tuple[int | None, str, bool], Image.Image] = {}
        # ...and the still half of the knob, a second time, in the hue it wears while a companion
        # holds his voice. Built on the first claim rather than with the window: most sessions
        # never hand the voice anywhere, and this is a tile nobody has asked to see yet.
        self._away: tuple[Image.Image, int] | None = None
        self._cursor_w = self.font_caption.getlength(CURSOR)
        self.hitboxes = self._layout()
        # What a pointed-at label has to stay off; see _label_anchor. Everything opaque on this
        # panel and nothing else - the brackets and their plates are the tube filter over the
        # live picture, and rimmed text reads over those.
        self._label_clear = (
            self.term,
            self._disc(self.switches[VOLUME], self.btn_r),
            self._disc(self.switches[HEAT], self.btn_r),
            self._disc(self.eye, self.eye_r),
        )
        self.menu_card, self.menu_cells = self._menu_layout()
        self._scrim: Image.Image | None = None  # built on the first long press, then kept

    # ---- layout ----

    def _rail_at(self, name: str, y: float) -> float:
        """Where mount *name*'s rail centreline crosses the horizontal line *y*.

        Walked along :meth:`Bracket.path` rather than solved, because that path is not always a
        straight line where it is asked about: on the left, high enough up, a clamp meets the arc
        of his housing's edge rather than the ramp. Whichever
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
        """Three round targets in two corners, and the reticle. The rest of the frame is picture.

        They are discs now rather than thirds of a row, which costs area and buys the middle of
        the screen back: the tab row was 17% of the panel and the three cells here are under 6%
        between them. What keeps that honest is that they are still targets a thumb finds without
        being looked at - 72 px is about 14 mm on the 7" panel, which is over the 9 mm everybody
        agrees is the floor - and that the eye, the one you press to go looking for something, is
        by far the biggest of the three.

        Bounding boxes rather than circles because that is what the kiosk's hit test takes, and
        the two switches are spaced along the ramp so their boxes do not overlap: a tap in a
        corner shared by two controls would silently belong to whichever was tested first.

        The fourth is the reticle, and it is exactly the mark that is already drawn there: the
        corners say "the frame is here", so they are the thing to press to ask for a different
        one. It gets no target of its own beyond them - 108 px at 480, about 21 mm on the 7"
        panel - because a control drawn nowhere is one nobody finds, and this one was already on
        the glass. It is tested last of the four and, sitting in the middle of a panel whose
        other controls are in the corners, overlaps none of them on any window this is drawn at.
        """
        return Hitboxes(
            volume=self._disc(self.switches[VOLUME], self.btn_r),
            eye=self._disc(self.eye, self.eye_r),
            heat=self._disc(self.switches[HEAT], self.btn_r),
            framing=self._disc((self.width // 2, self.height // 2), self.reticle_r),
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

    @_by_size
    def _build_filter(self) -> tuple[np.ndarray, np.ndarray]:
        """The tube filter - a wash, corner shading and scanlines - over the whole panel.

        Unmasked, because none of it depends on where the chrome is: it is built once and every
        plate is cut out of it. It used to be laid over a strip and a tab row that between them
        covered 29% of the panel; two mounts and a pod cover 16% of it, and the picture runs edge
        to edge behind them - this is what the chrome is *made of* rather than something sitting
        under an opaque bar. The plate is much darker than it was, though, because a bracket is a
        thing rather than a tint - see PLATE_WASH, which is as far towards PLATE_INK as it goes.

        Dark, and a *surface*: brushed along the panel's own diagonal, grooved by the sheet's
        hairlines, pitted where the finish has gone through, lifted where the lamp stands over it
        and falling away towards the corners. A plate the same grey as the bars bolted across it
        is not a plate, it is more of the same slab; a plate with nothing in its face is a shadow.
        """
        shape = (self.height, self.width)
        base: tuple[np.ndarray, np.ndarray] = (
            np.zeros((*shape, 3), dtype=np.float32),
            np.zeros(shape, dtype=np.float32),
        )
        base = _over(base, PLATE_INK, np.full(shape, PLATE_WASH, dtype=np.float32))
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
        # A hairline in a sheet lying back in its own shadow is a groove full of shadow, so the
        # sheet's marks go on dark here - the bars, standing up in the light, take the same marks
        # pale (see :meth:`_draw_rail`). Then the pitting, which is dark wherever it lands.
        base = _over(base, (0, 0, 0), self._marks * PLATE_SCRATCH_ALPHA)
        base = _over(base, (0, 0, 0),
                     material.pits(self.width, self.height, PLATE_PITS, seed=material.SEED + 9)
                     * PLATE_PIT_DEPTH)
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

    @_by_size
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

        A disc of dark glass rather than a flat fill: a radial vignette from a dim green middle
        to near black at the wall, the half of it facing the lamp a shade lifted, a fine brushing
        on the lamp's own diagonal, one drawn-out reflection of the lamp up towards it, the groove
        the stator drum runs in (:meth:`_stator_seat`), the sheet's hairlines where they cross, and
        the shadow the bezel's near wall drops - deepest on the side towards the lamp, because that
        is the wall standing between the lamp and the floor.
        All of it shows only through the gaps between his rings, and that is where the
        depth of him comes from: a flat disc behind a set of rings is a badge, and a lit floor
        with a shadow falling across it is a socket. The disc's outline is the same ImageDraw
        ellipse :meth:`_bracket_mask` cuts the hole with, so the two agree to the pixel and no
        tide line of wash appears round him.

        A separate layer rather than part of :meth:`_build_chrome`, because the collar's rings and
        the left mount's rail are drawn over the plate and must stay over it.
        """
        layer = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        cx, cy, out = *self.eye, self.shoulder
        inn = out * COLLAR_IN
        x0, y0, x1, y1, xs, ys = self._around(out + 1)
        dist = np.hypot(xs, ys)
        # The floor falls off to the wall faster than it started, so the middle stays level and
        # the drop gathers where the wall is - a plain square would put the whole thing on a
        # gradient and read as a painted vignette.
        depth = (np.clip(dist / inn, 0.0, 1.0) ** WELL_FALL)[..., None]
        floor = np.asarray(WELL_FLOOR, np.float32)
        rgb = floor + (np.asarray(WELL_WALL, np.float32) - floor) * depth
        lx, ly = material.lamp_2d()
        facing = (xs * lx + ys * ly) / np.maximum(dist, 1e-6)
        # Both halves, and the far one is the half that does the work - see WELL_SHADE.
        reach = 1.0 - 0.5 * depth[..., 0]
        rgb = rgb + np.asarray(WELL_LAMP, np.float32) * (np.clip(facing, 0.0, 1.0) * reach)[
            ..., None]
        rgb = rgb * (1.0 - WELL_SHADE * np.clip(-facing, 0.0, 1.0) * reach)[..., None]
        # Two textures, and they belong to two different surfaces: the floor was turned on a
        # lathe, so its marks run round; the pane over it was brushed off the sheet, so its marks
        # run on the one diagonal every brushed thing on this panel runs on.
        turned = material.grain(dist, self._round(xs, ys))
        brushed = material.grain((xs + ys) * _HALF_ROOT, (xs - ys) * _HALF_ROOT,
                                 seed=material.SEED + 9)
        rgb = rgb * (1.0 + material.GRAIN * WELL_GRAIN * turned + WELL_HATCH * brushed)[..., None]
        # One reflection of the lamp, up the pane towards it and drawn out along its line. The
        # only pale mark inside his rim that is not phosphor, and it is a twenty-fourth of the way
        # to white: glass at this size is a suggestion or it is a smear over the face.
        sx, sy = xs - lx * inn * WELL_SPOT_AT, ys - ly * inn * WELL_SPOT_AT
        u = (sx * lx + sy * ly) / (inn * WELL_SPOT_LONG)
        v = (sx * -ly + sy * lx) / (inn * WELL_SPOT_WIDE)
        pale = WELL_SPOT * np.exp(-(u * u + v * v))
        pale = pale + self._marks[y0:y1, x0:x1] * WELL_SCRATCH
        rgb = rgb + (np.asarray(WHITE, np.float32) - rgb) * pale[..., None]
        # Everything from the flank outwards stands above the floor; its shadow falls the other
        # way from the lamp, and only what lands inside the flank is floor.
        lift = max(1.0, (COLLAR_LIFT + WELL_DEPTH) * self.scale)
        shadow = np.clip(material.cast((dist >= inn).astype(np.float32), lift) * WELL_SHADOW,
                         0.0, 1.0)
        rgb = rgb * (1.0 - shadow * (dist < inn))[..., None]
        # The mouth after that shadow and not under it: the bezel's lip stands *on* the
        # countersink, so what the near wall occludes is the floor inboard of it. See
        # :meth:`_well_mouth`, which drops that shadow itself.
        rgb = self._well_mouth(rgb, dist, xs, ys)
        rgb = self._stator_seat(rgb, dist, xs, ys)
        rgb = rgb * np.minimum(1.0, WELL_CEILING / np.maximum(rgb.sum(-1, keepdims=True), 1e-3))
        disc = Image.new("L", (x1 - x0, y1 - y0), 0)
        ImageDraw.Draw(disc).ellipse(
            [cx - out - x0, cy - out - y0, cx + out - x0, cy + out - y0], fill=EYE_PLATE_ALPHA
        )
        layer.alpha_composite(material.to_image(rgb, np.asarray(disc, np.float32) / 255.0),
                              (x0, y0))
        return layer

    def _stator_seat(self, rgb: np.ndarray, dist: np.ndarray, xs: np.ndarray,
                     ys: np.ndarray) -> np.ndarray:
        """The groove the knurl runs in, turned into the well's own floor.

        A cross-section rather than a band: the floor drops, the wall facing the lamp catches it
        and the wall turned away goes dark, and the near wall drops a shadow across the floor
        between them. Four pixels wide and no two of them the same value, which is the whole of
        what says the well has a step in it - a flat annulus in another colour is a printed ring,
        however carefully it is graded round.

        Nothing pale, and the one metal down here is steel: what is inside his rim is a dark
        cavity and the light drawn on it, and the panel's brass stays on the bezel outside where
        nothing turns underneath it. The groove takes its *level* from the pane it is cut in and
        its *hue* from SEAT_METAL, which is how it stops carrying the tube's green without
        moving a count of the well's own shading.
        """
        inn, out = self.eye_r * SEAT_IN, self.eye_r * SEAT_OUT
        wall = max(1.0, SEAT_WALL * self.scale)
        band = np.clip(0.5 + (out - dist), 0.0, 1.0) * np.clip(0.5 + (dist - inn), 0.0, 1.0)
        lx, ly = material.lamp_2d()
        # Which way each wall faces: the inner one looks outwards, the outer one looks in, so one
        # of them is into the lamp wherever the other is away from it, all the way round.
        into = (xs * lx + ys * ly) / np.maximum(dist, 1e-6)
        # Each wall in flats rather than as a ramp - see SEAT_FACETS. Quantised before the two
        # are subtracted, so the inner wall's steps and the outer one's are the same cut.
        rise = (self._facets(1.0 - (dist - inn) / wall, SEAT_FACETS)
                - self._facets(1.0 - (out - dist) / wall, SEAT_FACETS))
        # One wall and one side - see SEAT_ARC. Only the wall the lamp's own bearing turns into
        # lights; the one opposite it goes under the sunk floor rather than catching a second
        # arc of the same lamp from the other end of the ring.
        lit = np.clip(rise, 0.0, 1.0) * np.clip(into, 0.0, 1.0) ** SEAT_ARC * SEAT_LIP
        # The groove's own metal, at whatever level the floor has reached where it is cut - see
        # SEAT_METAL. Taking the level from the pane and the hue from the steel is what keeps
        # the well's vignette, its lamp and its depth unchanged across the edge of the cut.
        metal = np.asarray(SEAT_METAL, np.float32)
        metal = metal * ((rgb @ np.asarray(LUMA, np.float32))
                         / float(np.dot(SEAT_METAL, LUMA)))[..., None]
        floor = metal * (1.0 - SEAT_DEEP)
        face = floor + (metal - floor) * lit[..., None]
        face = face * (1.0 - SEAT_TERMINATOR * np.clip(-rise, 0.0, 1.0))[..., None]
        face = face * (1.0 + material.GRAIN * SEAT_GRAIN
                       * material.grain(dist, self._round(xs, ys), seed=material.SEED + 3))[
            ..., None]
        # ...and the shadow the near wall throws over the floor, which is what gives the groove a
        # depth rather than two lit edges with a gap between them.
        shade = material.cast(np.clip(1.0 - rise, 0.0, 1.0) * band, wall) * SEAT_SHADOW
        face = face * (1.0 - np.clip(shade * band, 0.0, 1.0))[..., None]
        return rgb + (face - rgb) * band[..., None]

    def _graduation(self, dist: np.ndarray, turn: np.ndarray, top: float, face: float,
                    ticks: int, longer: int, short: float, deep: float) -> np.ndarray:
        """A scale cut into a ring: what to multiply its colour by, and 1 where the metal is uncut.

        *turn* is where round the ring a pixel is in radians and *dist* how far out. The ticks
        hang inwards from *top* and reach *short* of the *face*'s width down it, except one in
        *longer* of them, which reaches *deep*. All an engraving needs to know is how far across
        the nearest tick a pixel is, and that is one round and one multiply off the angle.

        The width is not an argument and neither is the depth - see :data:`TICK_W`. Every scale
        on this panel was cut with the same tool, and one that varies pitch, width and depth from
        ring to ring reads as decoration rather than as a machine that was indexed.
        """
        pitch = math.tau / ticks
        index = np.round(turn / pitch)
        off = (turn - index * pitch) * np.maximum(dist, 1e-6)
        reach = np.where(index % longer == 0, deep, short) * face
        down = np.clip(top - dist + 1.0, 0.0, 1.0) * np.clip(dist - (top - reach), 0.0, 1.0)
        wall = np.clip(TICK_W + 1.5 - off, 0.0, 1.0) * np.clip(off - TICK_W, 0.0, 1.0)
        return (1.0 - TICK_DARK * np.clip(TICK_W + 0.5 - np.abs(off), 0.0, 1.0) * down
                + TICK_GLINT * wall * down)

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

    def _stilled(self, rgb: np.ndarray, dist: np.ndarray | None = None) -> np.ndarray:
        """A field of his housing held under its ceiling, its hue untouched.

        Scaled rather than clipped per channel: a clip turns a warm highlight cool as soon as one
        channel lands on the ceiling, which is the one thing brass may not do. Every part of him
        that never moves goes through here, so the only pixels on this panel in the pupil's band
        are the pupil's.

        *dist* is how far each pixel is from his centre, and where it is given the ceiling is the
        one :meth:`_near_ceiling` sets rather than the flat :data:`STILL_CEILING` - a housing that
        may take the lamp hard out on the swell and hardly at all against the glass. A part that
        does not know where it is on him keeps the flat ceiling and stays under the harder one by
        being nowhere near the well.
        """
        lum = rgb @ np.asarray(LUMA, np.float32)
        ceiling = STILL_CEILING if dist is None else self._near_ceiling(dist)
        return rgb * np.minimum(1.0, ceiling / np.maximum(lum, 1e-3))[..., None]

    def _near_ceiling(self, dist: np.ndarray) -> np.ndarray:
        """What a still pixel may reach at each radius: :data:`NEAR_CEILING` in, and full out.

        Flat at the hard ceiling everywhere inside the seam between the two metals - the whole
        lip, the mouth of the well and anything screwed down over them - then a smoothstep back
        to :data:`STILL_CEILING` across the brass's inner face. The step is smooth rather than
        linear because this is the one field on the housing that is not a section: an edge in a
        ceiling would draw a ring of its own at exactly the radius the ceiling exists to keep
        rings off.
        """
        seam = self.shoulder * COLLAR_IN + max(1.5, COLLAR_LIP * self.scale)
        t = np.clip((dist - seam) / max(NEAR_EASE * self.scale, 1e-3), 0.0, 1.0)
        return NEAR_CEILING + (STILL_CEILING - NEAR_CEILING) * (t * t * (3.0 - 2.0 * t))

    def _radius(self, x0: int, y0: int, w: int, h: int) -> np.ndarray:
        """How far each pixel of a box is from his centre, for the parts not built off one.

        The loom's runs and their gland are built in the frame of the cable rather than in his,
        so they carry no radius of their own - and they are bolted to the collar, which puts
        their inner corners inside the seam :meth:`_near_ceiling` measures from.
        """
        cx, cy = self.eye
        ys = (np.arange(y0, y0 + h, dtype=np.float32) - cy)[:, None]
        xs = (np.arange(x0, x0 + w, dtype=np.float32) - cx)[None, :]
        return np.hypot(xs, ys)

    @staticmethod
    def _facets(tilt: np.ndarray, flats: int) -> np.ndarray:
        """A tilt that climbs smoothly, cut into *flats* steps: a chamfer off a tool, not a fade.

        Rounded up rather than down so the outermost flat keeps the full tilt it was asked for -
        the crest is where the specular is, and losing a step of it there costs the section its
        one bright line.
        """
        return np.ceil(np.clip(tilt, 0.0, 1.0) * flats) / flats

    def _well_mouth(self, rgb: np.ndarray, dist: np.ndarray, xs: np.ndarray,
                    ys: np.ndarray) -> np.ndarray:
        """The countersink round the well's mouth, and the shadow it drops back into the well.

        The MOUTH_* block says why this exists at all. Mechanically it is one cross-section over
        eight pixels: a face whose outward tilt climbs in flats from the foot to the crest, so it
        is brightest where the ring faces the lamp and near black where it faces away - the same
        lamp, at the same bearing, as the bezel outside it - and then a reveal at the outer edge
        taken under even the room's own light, because the bezel's lip is standing over it.

        The order matters. The face is laid first and the reveal cut into it, so the darkest
        pixel of the section is the last one before his rim rather than a line drawn beside it;
        then the crest's own shadow goes down onto the floor inboard, which is multiplicative on
        whatever the floor is doing and recovers over a dozen pixels the way the reference's do.
        """
        inn, out = self.eye_r * MOUTH_IN, float(self.eye_r)
        safe = np.maximum(dist, 1e-6)
        ux, uy = xs / safe, ys / safe
        band = np.clip(0.5 + (out - dist), 0.0, 1.0) * np.clip(0.5 + (dist - inn), 0.0, 1.0)
        # The section: a flat relief at the foot, then a tilt climbing outwards in flats, so the
        # crest under the lip is the one part square enough to the lamp to carry a specular.
        across = np.clip((dist - inn) / max(out - inn, 1e-3), 0.0, 1.0)
        climb = np.clip((across - MOUTH_LAND) / max(1.0 - MOUTH_LAND, 1e-3), 0.0, 1.0)
        tilt = MOUTH_FOOT + (MOUTH_TILT - MOUTH_FOOT) * self._facets(climb, MOUTH_FACETS)
        nx, ny, nz = ux * tilt, uy * tilt, np.sqrt(np.maximum(1.0 - tilt * tilt, 0.0))
        diffuse, spec = material.shade(nx, ny, nz)
        turned = material.grain(dist, self._round(xs, ys), seed=material.SEED + 11)
        face = material.steel(diffuse, spec * MOUTH_SHINE, turned * MOUTH_GRAIN,
                              colour=mix(material.STEEL, material.STEEL_DARK, MOUTH_METAL))
        # ...and the two ends of it. The relief is the bottom of the cut and the reveal is what
        # the lip stands over; neither is a fade, because a groove floor and an overhang both
        # occlude, so both go under even AMBIENT and the section ends dark at both edges.
        reveal = np.clip(1.0 - (out - dist) / max(MOUTH_REVEAL * self.scale, 0.5), 0.0, 1.0)
        relief = np.clip(1.0 - across / max(MOUTH_LAND, 1e-3), 0.0, 1.0)
        face = face * (1.0 - MOUTH_DARK * np.maximum(reveal, relief))[..., None]
        # ...and how much of the room this cut can see at all - see MOUTH_ARC. Down at the bottom
        # of a bore, only the arc pointing at the lamp does; the rest is closed in by the bezel
        # standing over it, and closed in is what a cavity is.
        lx, ly = material.lamp_2d()
        seen = np.clip((ux * lx + uy * ly), 0.0, 1.0) ** MOUTH_ARC
        face = face * (material.AMBIENT + (1.0 - material.AMBIENT) * seen)[..., None]
        rgb = rgb + (face - rgb) * band[..., None]
        # The crest is proud of the floor, so it drops a shadow back down into the well - away
        # from the lamp, and only where there is floor left to fall on.
        lift = max(1.0, MOUTH_LIFT * self.scale)
        drop = material.cast((dist >= inn).astype(np.float32), lift) * MOUTH_SHADOW
        return rgb * (1.0 - np.clip(drop, 0.0, 1.0) * (dist < inn))[..., None]

    @_by_size
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

    @_by_size
    def _build_surround(self) -> Image.Image:
        """The machined surround the whole chassis sits in, and the shadow it drops inwards.

        A square plate with a radiused window cut in it, FRAME_W deep, shaded off two fields:
        how far each pixel is from the panel's outer edge (which drives the crown, and mitres at
        45 degrees where two edges meet) and how far it is out of the opening (which drives the
        inner chamfer, and follows the corner round). Between them the face, crowned outward at
        the crown's end and inward at the chamfer's, so it falls monotonically from whichever end
        the lamp is at - and which end that is comes from the chamfer's own normal against
        :func:`material.lamp_2d`, never from a per-edge constant. The top and left rails end up
        with their one specular on the outside, the bottom and right ones on the inside, and the
        far edge of every one of them is the darkest metal in its section.

        Then the same finish every other steel face on this panel wears: brushed along the run,
        drifting along it, the sheet's own hairlines plus a few dragged along each stretch, the
        odd pit, and the lamp's falloff across the panel (:meth:`_sunlight`), so the bottom right
        of the surround is a good deal duller than the top left even where the section agrees.

        Underneath it, two shadows onto the picture: a soft one thrown away from the lamp, which
        is why the light rails have a band of shade inside them and the dark ones do not, and a
        hard line of contact hugging the opening all the way round. Both composited black, so
        they multiply the camera and recover over a dozen rows instead of painting a floor on it.

        Once per window size. Nothing here may be reached from a frame.
        """
        band = float(self.frame_w)
        outside, out_x, out_y = opening_field(self.width, self.height)
        cover = np.clip(0.5 + outside, 0.0, 1.0)
        xs = np.arange(self.width, dtype=np.float32)[None, :]
        ys = np.arange(self.height, dtype=np.float32)[:, None]
        near_x = np.minimum(xs, self.width - 1 - xs)
        near_y = np.minimum(ys, self.height - 1 - ys)
        upright = np.broadcast_to(near_x < near_y, cover.shape)  # nearer a left or right edge
        # The outward normal of the panel edge each pixel is nearest. Picking one edge rather
        # than blending two is what puts a hard mitre down the diagonal of every corner.
        edge_x = np.where(upright, np.sign(xs - (self.width - 1) / 2.0), 0.0)
        edge_y = np.where(upright, 0.0, np.sign(ys - (self.height - 1) / 2.0))
        # The section: a crown at the outer arris, a chamfer at the inner one, a face between.
        deep = np.minimum(near_x, near_y)  # in from the panel's outer edge, whichever is nearest
        run = np.where(upright, np.broadcast_to(ys, cover.shape),
                       np.broadcast_to(xs, cover.shape))
        # ...and the crown's width breathes along the run, so the row the ridge lands on walks in
        # and out by a pixel every few tens of columns. A specular pinned to one row for eight
        # hundred of them is the tell that a section was extruded rather than lit.
        wave = material.wear(run * FRAME_ARRIS_PITCH, seed=material.SEED + 16)
        # Measured from the state stroke's inner edge, not the panel's: the outer `line` pixels
        # are the light let into the arris and the crown is the first metal anybody can see. Off
        # the panel edge instead, both of the chamfer's flats fell under the stroke and the one
        # visible row of it was the tail of the second - a 210 specular became a 160 face.
        crown = _facets(np.maximum(deep - self.line, 0.0),
                        FRAME_CROWN * self.scale * (1.0 + FRAME_ARRIS_WAVE * wave))
        under = _facets(outside, FRAME_UNDER * self.scale)
        # The face crowns outward towards the plate's own edge and rolls inward towards the
        # opening, off two ramps rather than one. One ramp read off the opening alone put the
        # whole corner gusset at the outer end of the section - a bright 8 px triangle in every
        # corner - because a corner is a long way outside the opening whichever way you measure.
        # Off both, the deep corner is flat plate, which is what a corner of a plate is.
        rise = np.clip(1.0 - deep / band, 0.0, 1.0)
        fall = np.clip(1.0 - outside / band, 0.0, 1.0)
        # The three bands are mixed by their own weights rather than switched on a threshold: a
        # chamfer that is one twentieth present still overrode the face where it was tested for
        # zero, and the whole section went with it.
        arris = np.clip(crown + under, 0.0, 1.0)
        tilt = (FRAME_FACE_TILT * (rise - fall) * (1.0 - arris)
                + FRAME_CROWN_TILT * crown - FRAME_UNDER_TILT * under)
        # The crown faces the way its own panel edge does; the inner chamfer faces the way out of
        # the opening, which is the same thing on a run and turns with the radius at a corner.
        dir_x = edge_x + (out_x - edge_x) * under
        dir_y = edge_y + (out_y - edge_y) * under
        length = np.maximum(np.hypot(dir_x, dir_y), 1e-6)
        dir_x, dir_y = dir_x / length, dir_y / length
        steep = np.abs(tilt)
        diffuse, spec = material.shade(dir_x * tilt, dir_y * tilt,
                                       np.sqrt(np.maximum(1.0 - tilt * tilt, 0.0)))
        # A cut edge is deburred, not moulded: past FRAME_BREAK_AT the arris scatters where a
        # rolled one would mirror, which is what stops the silhouette matching the ridge.
        spec = spec * np.where(steep > FRAME_BREAK_AT, FRAME_BREAK, 1.0)
        spec = spec * (FRAME_FACE_GLOSS + (1.0 - FRAME_FACE_GLOSS) * arris)
        section = np.where(upright, np.broadcast_to(xs, cover.shape),
                           np.broadcast_to(ys, cover.shape))
        tooth = material.grain(section, run, seed=material.SEED + 11)
        tooth = np.where(tooth > 0.0, tooth, tooth * FRAME_GRAIN_DARK)
        rub = material.wear(run, seed=material.SEED + 12)
        # Half a run of handled steel is duller than the other half, and that is the other thing
        # that makes the ridge wander. It goes on the highlight only: the drift below is a
        # multiply on the *diffuse* light, and a lamp is not brighter where the metal is polished.
        spec = spec * (1.0 - FRAME_WEAR * (1.0 - rub) / 2.0)
        drift = FRAME_DRIFT * rub / material.GRAIN
        rgb = material.steel(diffuse, spec, tooth * FRAME_GRAIN + drift,
                             mix(material.STEEL, material.STEEL_LIT, FRAME_STEEL))
        rgb = self._surround_wear(rgb, upright)
        rgb = rgb * self._sunlight(0, 0, self.width, self.height)[..., None]
        # What goes under it. `cast` pushes the surround's own coverage away from the lamp, so
        # the shadow of the top and left rails falls into the picture and that of the bottom and
        # right rails falls off the panel - which is the whole reason a shadow is worth drawing.
        lift = max(1.0, FRAME_LIFT * self.scale)
        shadow = material.cast(cover, lift) * FRAME_SHADOW
        lx, ly = material.lamp_2d()
        under_lit = (-out_x * lx - out_y * ly) > 0.0  # the inner chamfer faces the lamp here
        hug = FRAME_CONTACT * np.clip(1.0 + outside / max(FRAME_CONTACT_W * self.scale, 1e-3),
                                      0.0, 1.0)
        hug = hug * np.where(under_lit, FRAME_CONTACT_LIT, 1.0) * (1.0 - cover)
        dark = 1.0 - (1.0 - shadow) * (1.0 - hug)
        alpha = cover + dark * (1.0 - cover)
        layer = material.to_image(rgb * (cover / np.maximum(alpha, 1e-6))[..., None], alpha)
        self._surround_screws(layer)
        return layer

    def _surround_screws(self, layer: Image.Image) -> None:
        """One socket screw through each corner of the plate - where a bezel is actually fixed.

        The same head as every other fixing on the panel and built the same way: its own bearing
        to the lamp (:func:`material.bearing`), its own clocking, its own grime and its own share
        of the falloff, so the four of them are four screws and not one screw pasted four times.
        It goes on the corner gusset rather than on a run, because a nine pixel run has a crown
        down one edge and a chamfer down the other and nothing in between for a head to sit on.
        """
        r = max(2.0, FRAME_SCREW * self.scale)
        at = FRAME_SCREW_AT * self.frame_w
        lamp = (PLATE_LAMP[0] * self.width, PLATE_LAMP[1] * self.height)
        for x in (at, self.width - 1 - at):
            for y in (at, self.height - 1 - at):
                cx, cy = math.floor(x), math.floor(y)
                ax, ay = unit(x, y, *lamp)
                mark = (cx * 73856093 ^ cy * 19349663) % 65521
                tile = material.screw(
                    round(r, 2), round(x - cx, 2), round(y - cy, 2), round(ax, 3), round(ay, 3),
                    round(mark % 360 * math.pi / 1080.0, 3),  # a hex repeats every sixty degrees
                    mark, round(float(self._sunlight(cx, cy, 1, 1)[0, 0]) ** BOLT_FALL, 3),
                )
                half = tile.width // 2
                left, top = cx - half, cy - half
                crop = tile.crop((
                    max(0, -left), max(0, -top),
                    tile.width - max(0, left + tile.width - self.width),
                    tile.height - max(0, top + tile.height - self.height),
                ))
                layer.alpha_composite(crop, (max(0, left), max(0, top)))

    def _surround_wear(self, rgb: np.ndarray, upright: np.ndarray) -> np.ndarray:
        """Hairlines and pitting over the surround's face - two sheets, one per run direction.

        A single sheet of scratches would drag every hairline the same way round all four rails,
        and a scratch that runs across a rail instead of along it is a scratch on a texture rather
        than on a bar. The sheet's own marks go over the top of both, because a hairline that
        carries from the plate onto the case is most of what says they are one piece of metal.
        """
        length = tuple(2 * n * self.scale for n in RAIL_SCRATCH_LEN)
        drawn = [
            # Twice over and boxed down, as the rails' are: a hairline at one pixel on a nine
            # pixel rail is a staircase, and at half a pixel it is a line.
            material.scratches(self.width * 2, self.height * 2, FRAME_SCRATCHES, along,
                               seed=material.SEED + seed, spread=FRAME_SCRATCH_SPREAD,
                               length=length)
            .reshape(self.height, 2, self.width, 2).mean(axis=(1, 3)) * 2.0
            for along, seed in (((1.0, 0.0), 13), ((0.0, 1.0), 14))
        ]
        marks = np.maximum(np.where(upright, drawn[1], drawn[0]),
                           FRAME_SHEET_MARK * self._marks)
        marks = (FRAME_SCRATCH * marks)[..., None]
        rgb = rgb * (1.0 - marks) + np.asarray(material.STEEL_SPEC, np.float32) * marks
        pitted = material.pits(self.width, self.height, FRAME_PITS, seed=material.SEED + 15)
        return rgb * (1.0 - FRAME_PIT_DEPTH * pitted)[..., None]

    @_by_size
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
        # The webbing before any of the hardware. It is what is stiffening the *plate*, so it
        # belongs to the plate, and drawn with its own bracket it went on over the foot rail:
        # measured, the rail's specular ran flat at 150 out to x=755 and then broke into a
        # 13 px sawtooth dipping to 51 - a two-thirds modulation, which is a see-through bar.
        # Panel graphics, hatching and washes go behind hardware, always; a member you can see
        # the plate through is not a member.
        self._draw_webbing(layer)
        # ...then the two frame rails, because everything else on the panel hangs off them: the
        # mounts stand on the foot rail and the module hangs from the head rail.
        self._draw_spine(layer)
        self._draw_head(layer)
        # The terminal before either mount and before both instruments, because every one of them
        # is what buries an end of it. Order is the whole illusion: drawn last this is a box lying
        # on the panel, and drawn first it is a box behind it.
        self._draw_terminal(layer)
        # ...then the mounts, and his collar with the left one - see _draw_bracket, which lays it
        # between that mount's rail and its bolts.
        for bracket in self.brackets.values():
            self._draw_bracket(layer, bracket)
        # The loom after the mounts and not before them, which is the opposite of the terminal's
        # rule and for the opposite reason: nothing is meant to bury these except the gland. Drawn
        # first, the corner's own webbing and the rail's cast shadow simply write over the top of
        # them - ImageDraw writes rather than composites - and the cables vanish.
        self._draw_loom(layer)
        # The two dial faces, which used to be baked once per state with the switches they
        # replace. Neither wears the state's accent - a volume and a board temperature are true
        # whether or not anybody is talking to him - so neither has any business being rebuilt
        # every time the state changes, and they belong here with the rail they are bolted to.
        for name in SWITCHES:
            self._stamp_instrument(layer, name, *self._instrument_tile(name))
        # ...and the pilot lamp on the plate under them, last because it stands on everything
        # else in that corner: the ribs, the foot rail's shadow and the mount's own plate are all
        # what it is screwed down to. Unlit here - the light in it is the only part that knows
        # what the box is doing, and that is a tile the frame composites. See _draw_pilot.
        self._draw_pilot(layer)
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
            # The window first and the rail over it: the rail is the frame, its shadow falls on
            # the flange, and the bolts through its knees sit over both.
            self._draw_pod_face(layer, tags)
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

    def _sunlight(self, x0: int, y0: int, width: int, height: int) -> np.ndarray:
        """How much of the one lamp reaches a box of the panel, RAIL_SUN..1.

        The same field :meth:`_build_filter` lights the sheet with, sampled over one part's box
        instead of the whole panel, so a bar and the plate under it agree about where the lamp
        is. Without it every bar on the panel is the same brightness whatever corner it is in,
        which is the flattest a panel can look however carefully each bar is shaded: a lamp that
        does not fall off is not a lamp, it is a fill.
        """
        lamp = (PLATE_LAMP[0] * self.width - x0, PLATE_LAMP[1] * self.height - y0)
        return material.glare(width, height, lamp, PLATE_REACH * self.height, ambient=RAIL_SUN)

    def _draw_rail(self, layer: Image.Image, points: Sequence[tuple[float, float]],
                   thick: int | None = None, turned: float = 0.0) -> None:
        """A bar of steel along the spine, under the panel's one lamp, and its shadows.

        One mitred outline - the spine offset both ways and closed, so the knees are corners and
        the ends are square - filled for coverage, and one frame off the same spine
        (:meth:`_bar_frame`) that says how far across the bar every pixel is and which edge it is
        nearest. The section is read straight off that, and it is four bands: the edge the lamp
        is on ROLLS over - through every angle between the face and the silhouette - so it puts
        a knocked-back arris on the outline, a blown ridge a pixel or two inside it where the
        surface passes through the angle that mirrors the lamp, and then the face, crowned so
        gently that it is a fall and not a curve; and the edge turned away is a steep chamfer
        taking no light at all, which is the terminator the contact shadow starts under.
        Nothing here decides which edge is which: the edge's own normal against
        :func:`material.lamp_2d` does, so a diagonal rail's bright edge is its upper-left one and
        a horizontal rail's is its top.

        Which edge is lit is one thing and whether that edge is lit *at all* is another, and the
        second is what the brushing settles. The bar is ground along its own run, so its
        highlight answers to the lamp anisotropically (:data:`material.BRUSH`, the bar's tangent
        passed in as the fibre). Isotropic, a member's ridge dies whenever the direction that
        mirrors the lamp happens to lie along the member's own axis - which is exactly where a
        45-degree strut sits, and ours measured a lit edge of 119 over a face of 111 beside a
        horizontal at 198 over 137. A bar whose section falls but whose edge does not catch is a
        filled shape.

        The fall across the face is two things, because one of them cannot do the job alone. The
        crown is the lamp's bearing projected onto the section, so it is worth forty levels on a
        horizontal run and twelve on a strut; the far half's ambient occlusion (RAIL_SHUT) is
        geometry and lands the same on every bearing. Together the face falls monotonically from
        its lit chamfer to a far edge that is the darkest metal on the member, whichever way the
        member runs, which is the whole of what a single lamp looks like.

        The face is brushed along its length, is crossed by the sheet's hairlines and a couple of
        dozen of its own, and pitted here and there. None of that is decoration: a face with
        nothing in it measures flat to within a level and is read as a fill at a glance. Nor may
        it be the subject - texture along a run stays well under the section's own gradient, or
        the member reads as a swatch with noise on it. Over all of it, the lamp's own falloff
        across the panel (:meth:`_sunlight`), which the face takes whole and the ridge takes a
        fraction of: see RAIL_SUN_SPEC.

        The shadows are still most of the work. A bar with a bright edge and no shadow reads as
        a drawing of a bar; the same bar with a hard line of contact shadow hugging it and a soft
        one falling away from the lamp sits *on* something. Both go on as composites from a
        private tile, never as writes - a translucent write is a window onto the camera - and
        only over the box the bar occupies, so the eight of these a panel builds are eight small
        tiles and not eight full frames.

        *turned* is what makes the two frame rails a different member from everything bolted to
        them. At 0 the section is sawn flat stock: the far half holds the face's own crown until
        the chamfer takes it away. At 1 it is round bar, and the far half keeps rolling until the
        lamp has left it entirely - which, with the plate shutting the underside in
        (RAIL_OCCLUDE) and throwing a little back up onto its last row (RAIL_BOUNCE), is the
        difference between a section and a slab. Measured across a flat bar the reading was
        specular, falloff, and then six rows that never left 126: four bands and only two events.
        A turned one gives the four the eye actually looks for - a ridge in the upper third, a
        monotonic fall, a core shadow at about a fifth of the face, and a line of reflected light
        under it at a third of the ridge, which is where a bounce has to stay so that the member
        still has exactly one specular on it.

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
        # The section, all of it a tilt under the one lamp, and the face between the two edges
        # crowned towards whichever it is nearer - towards the lamp on the lit half and away
        # from it on the other, so it falls from its bright edge to its dark one with no second
        # light anywhere on it.
        #
        # The lit edge is a ROLL and not a bevel, and that is the whole of the cross-section.
        # A flat chamfer holds its whole face at one angle, which is one brightness, which is a
        # stroke: the flat chamfer sat at a sine of 0.75 where the angle that mirrors this
        # lamp is 0.43, so no pixel on the bar ever reflected it and the ridge topped out at 213
        # whatever the ceiling allowed. A roll passes through every angle between the face and
        # the silhouette, so it *must* cross the mirror somewhere - a pixel or two in from the
        # outline, with the steeper arris outside it darker again. Four bands fall out of it
        # without being drawn: rolled edge, blown ridge, mid face, and the far chamfer turned
        # right away from the lamp, which is the terminator.
        #
        # Where the ridge lands is not the same all the way along, because the width of the roll
        # is not: RAIL_ARRIS_WAVE breathes it slowly, which walks the brightest row in and out
        # by a pixel. A specular pinned to one row for twenty-six columns running is the tell
        # that a section was extruded rather than lit, and it was ours in twenty-four of them.
        wave = material.wear(along * RAIL_ARRIS_PITCH, seed=material.SEED + 6)
        edge = np.minimum(RAIL_EDGE * self.scale * (1.0 + RAIL_ARRIS_WAVE * wave), half * 0.45)
        chamfer = min(RAIL_CHAMFER * self.scale, half * 0.3)
        crown = RAIL_CROWN * np.clip(np.abs(across) / max(half - chamfer, 1e-3), 0.0, 1.0)
        arris = depth < np.where(lit, edge, chamfer)
        rolled = np.clip(1.0 - depth / np.maximum(edge, 1e-3), 0.0, 1.0)
        # How far out of the section a pixel is, 0 on the centreline and 1 on the outermost row
        # the outline actually fills. Measured against half a pixel in from the silhouette
        # because that is where full coverage stops: normalising on `half` itself puts the end
        # of the section in the anti-aliasing, where nothing can be read.
        out = np.clip(np.abs(across) / max(half - 0.5, 1e-3), 0.0, 1.0)
        # A turned bar keeps rolling away past its own crown on the far half, so the light dies
        # inside the silhouette instead of at it; sawn stock holds the crown until the chamfer.
        away = crown + (1.0 - crown) * out**RAIL_TURN_FALL
        far = np.where(arris, RAIL_FAR_TILT, np.maximum(crown, turned * away))
        tilt = np.where(lit, crown + (1.0 - crown) * rolled, far)
        # Brushed along its own length, so the highlight answers to the lamp anisotropically -
        # see material.BRUSH. Without the fibre the mirror direction for a 45-degree strut lies
        # almost along the strut's own axis, no normal on it ever crosses that direction, and
        # the member comes out with no lit edge at all beside the horizontal it is bolted to.
        diffuse, spec = material.shade(ex * tilt, ey * tilt, np.sqrt(1.0 - tilt * tilt),
                                       fibre=(tx, ty))
        # What the far half can see of the room, which is what turns its dark side from a floor
        # into a section. Two things and they land in different places: the plate shuts the
        # underside in (deepest at the contact), and the plate also throws a little light back up
        # onto the last of it. Occlusion first, bounce over it, and the bounce is held well under
        # the crest so the member keeps one specular - see RAIL_BOUNCE.
        #
        # Every bar is shut in on its far side, not only a turned one. A flat bar's face is a
        # shallow cylinder and the tilt across it is worth about twenty levels on a horizontal
        # run - but only eight on a 45-degree one and nine on a vertical, because the lamp's
        # bearing barely projects onto those sections at all. Measured, that left three of our
        # seven runs with a face flat to within the brushing, which is what a critic reads as a
        # fill bracketed by edge strokes. Occlusion does not care which way a bar points: the
        # far half of any bar lying on a plate sees less of the room than the near half, and it
        # is what carries the fall on the runs the crown cannot.
        under = np.clip((out - RAIL_OCC_AT) / (1.0 - RAIL_OCC_AT), 0.0, 1.0) * ~lit
        # ...and the bounce sits INSIDE the far chamfer, never on it. Read out to the silhouette
        # it landed on the terminator itself, so the section came out core shadow, then a lift
        # on its very last row: a bar that gets brighter at the outline is a bar with a light
        # behind it. The reference's order is falloff, one row of reflected light, then the
        # chamfer that ends the member - 203 124 75, then 86 85, then 35 19 - because the row
        # that can see the plate is the one still turned down towards it, and the chamfer beyond
        # it is turned out of the picture entirely. So: rise to the last row before the arris,
        # and stop there.
        edge_at = max((half - chamfer) / max(half - 0.5, 1e-3), RAIL_BOUNCE_AT + 1e-3)
        bounce = np.clip((out - RAIL_BOUNCE_AT) / (edge_at - RAIL_BOUNCE_AT), 0.0, 1.0) ** 2
        bounce = bounce * ~lit * ~arris
        shut = RAIL_SHUT + (RAIL_OCCLUDE - RAIL_SHUT) * turned
        # How squarely this member's section lies to the lamp, which is how big a crest it has to
        # sit under. The bounce off the plate is much the same whichever way a bar runs; the
        # crest beside it is not, and the ceiling a bounce has to stay under is a fraction of
        # *that*. Left flat, the same reflected line that reads as a bounce on the foot rail -
        # a seventh of its crest - came out four fifths of the crest on the head rail's corner
        # legs, where the lamp barely grazes the section at all: two comparable highlights on
        # one member, which is a panel with two lamps on it however each was arrived at.
        bear = np.abs(ex * lx + ey * ly)
        room = (material.AMBIENT * (1.0 - shut * under * under)
                + turned * RAIL_BOUNCE * bear * bounce)
        rub = material.wear(along)
        # ...and how polished the bar is where the ridge crosses it. Down from the mirror, never
        # up: a highlight that is already reflecting the lamp cannot reflect more of it, and
        # multiplying past 1 only pushed the ceiling somewhere no lamp is. Half the length of a
        # handled bar is duller than the other half, and that is what makes the ridge wander.
        polish = 1.0 - RAIL_WEAR * (1.0 - rub) / 2.0
        # ...and the last sliver of the arris itself is knocked off. A bar is sawn and deburred,
        # not moulded: past RAIL_BREAK_AT the surface is a broken edge, which scatters where a
        # rolled one mirrors. Without it the outline came out within twenty levels of the ridge
        # and the ridge had nothing to sit inset from - the two together are the whole read.
        broken = np.where(tilt > RAIL_BREAK_AT, RAIL_BREAK, 1.0)
        spec = spec * polish * broken * np.where(arris, 1.0, RAIL_FACE_GLOSS)
        # Where the arris has been knocked about it reflects nothing at all, and that is the fix
        # for the loudest "this was rendered" measurement on the panel: a highlight that stayed
        # within 0.7 of a level for a hundred and fifty-eight pixels of its run. `polish` and the
        # arris wave are both slow - a bend every thirty pixels - and neither could break it,
        # because a ridge sitting on its own ceiling barely moves when the specular under it
        # does. So this is a multiply on the finished colour, and only on the two lit rows: the
        # brushing again with its arguments the other way round, fibres that vary ALONG the bar
        # and run across it, which is the transverse tooling a deburred edge carries and the one
        # field here that changes from one pixel to the next down a run. Down only - a chipped
        # edge cannot reflect more of the lamp than a clean one.
        nick = np.clip(material.grain(along, across, seed=material.SEED + 11), -1.0, 1.0)
        chip = RAIL_RIDGE_WEAR * np.clip(-nick, 0.0, 1.0) * (lit & arris)
        face = mix(material.STEEL, material.STEEL_LIT, RAIL_STEEL)
        # Brushed steel scatters more than it swallows: the tooth's lit side throws light back
        # at you and its shaded side only loses the little it had. Symmetric noise over a face
        # is a grain overlay and measures as one - ours came back +1.82% bright against -1.74%
        # dark where the panel we are matching is +2.98 against -0.17.
        tooth = material.grain(across, along)
        tooth = np.where(tooth > 0.0, tooth, tooth * RAIL_GRAIN_DARK)
        # ...and how much light the face itself is taking here, which is two slow fields: the one
        # the highlight breathes with, and a sweep four times as long that is the room rather
        # than the hands. Both go in with the brushing, because all three are the same multiply
        # on the *diffuse* light and none of them has any business touching the highlight - a
        # specular is a picture of the lamp, and the lamp is not brighter where the bar is
        # polished. Laid over the finished colour instead, they pushed the ridge past the tube.
        sweep = material.wear(along * RAIL_SWEEP_PITCH, seed=material.SEED + 8)
        drift = (RAIL_DRIFT * rub + RAIL_SWEEP * sweep) / material.GRAIN
        # The lamp is one lamp standing over the top left of the panel, and a bar in the far
        # corner takes less of it. The face takes all of that falloff; the highlight takes a
        # fraction of it, because a specular is not a lit surface, it is a picture of the lamp
        # reflected in one - and a picture of the lamp is nearly as bright wherever on the panel
        # you stand. Taken off the finished colour instead, as it was, the ridge in the bottom
        # right corner fell with its own face and the member lost its section: ours measured a
        # chamfer of 101 over a face of 90 there, against 157-179 over 85 on the panel this one
        # is judged against, which is the whole of that corner's read in one number.
        sun = self._sunlight(x0, y0, w, h)
        lamp_spec = sun**RAIL_SUN_SPEC
        rgb = material.steel(diffuse * sun, spec * lamp_spec, tooth * RAIL_GRAIN + drift,
                             face, ambient=room * sun)
        rgb = rgb * (1.0 - chip)[..., None]
        # Hairlines: the sheet's, where they happen to cross, and a few of the bar's own, dragged
        # along its length. Seeded off where the bar is, so every bar is scratched differently
        # and the same way on every boot. Drawn twice over and boxed down, because a hairline
        # nearly parallel to a diagonal bar is a staircase at one pixel and a line at half.
        longest = max(zip(points, points[1:], strict=False),
                      key=lambda ab: math.hypot(ab[1][0] - ab[0][0], ab[1][1] - ab[0][1]))
        seed = material.SEED + x0 * 13 + y0 * 7
        own = material.scratches(w * 2, h * 2, RAIL_SCRATCHES, unit(*longest[0], *longest[1]),
                                 seed=seed, spread=RAIL_SCRATCH_SPREAD,
                                 length=tuple(2 * n * self.scale for n in RAIL_SCRATCH_LEN))
        own = own.reshape(h, 2, w, 2).mean(axis=(1, 3)) * 2.0
        marks = (np.maximum(own, self._marks[y0:y1, x0:x1]) * RAIL_SCRATCH)[..., None]
        # Under the same lamp as the bar it is cut into: a scratch is bare metal catching the
        # room, not a light of its own, so it dims with the corner it is in.
        bare = np.asarray(material.STEEL_SPEC, np.float32) * lamp_spec[..., None]
        rgb = rgb * (1.0 - marks) + bare * marks
        # ...and the odd pit where the finish has gone through. Kept off the lit chamfer: a hole
        # in the one bright line reads as a break in the bar rather than as a mark on it.
        pitted = material.pits(w, h, RAIL_PITS, seed=seed) * ((depth >= edge) | ~lit)
        rgb = rgb * (1.0 - RAIL_PIT_DEPTH * pitted)[..., None]
        # A cut end is where a bar rusts first.
        total = sum(math.hypot(b[0] - a[0], b[1] - a[1])
                    for a, b in zip(points, points[1:], strict=False))
        to_end = np.minimum(along, total - along) / max(RAIL_END_W * self.scale, 1e-3)
        rgb = rgb * (1.0 - RAIL_END_WEAR * np.clip(1.0 - to_end, 0.0, 1.0))[..., None]
        layer.alpha_composite(material.to_image(rgb, cover), (x0, y0))

    def _spine_y(self) -> float:
        """Where the foot rail's centreline runs, in panel rows.

        SPINE_DROP up from the bottom edge, or far enough up that the bar's lower edge tucks
        under the monitor's foot - whichever is higher. The case stands in *front* of the rail,
        and a window size that left a pixel of steel showing below it would turn the one member
        everything hangs off into a stripe behind a box.

        SPINE_DROP clears the state light's cove as well - see :meth:`_draw_head`, which had the
        same problem at the other end of the panel and the same answer.

        On a half row, always, which is the head rail's rule and for a harder reason here. The
        section is read off how far out of the bar a pixel is, and a centreline on a whole row
        puts the outermost sample at eight ninths of the way out: the row that carries the core
        shadow and the row of reflected light under it are simply never sampled, and the bar ends
        in a fade instead of in a section. Landed on a half, the ten rows sit symmetrically at
        plus and minus a half through four and a half, and both ends of the section are pixels.
        """
        half = max(4, round(SPINE_W * self.scale)) / 2.0
        return min(round(self.height - SPINE_DROP * self.scale - 0.5) + 0.5,
                   round(self.term.bottom - half) - 0.5)

    def _draw_spine(self, layer: Image.Image) -> None:
        """One bar across the whole bottom edge, and the thing that holds the panel together.

        Every part on here was bolted to something and nothing was bolted to anything: the two
        mounts ran their legs off the bottom of the frame, the monitor's clamps reached down into
        open picture, and three critics in a row wrote down the same sentence - the hardware has
        no spine, so nothing explains what holds what. Measured, the band along the bottom edge
        carried a highlight in three and a half per cent of its columns where the panel this one
        is judged against carries one in eighty-nine, and that whole difference is one continuous
        member along the bottom with everything else hanging off it.

        So: SPINE_W of round bar on SPINE_DROP, the same steel as every other bar here and the
        turned section the head rail has (:meth:`_draw_rail`), cut to the frame's own rounded
        corners so it does not poke out past them. It is half a mount's section, which keeps two
        sizes of member on the panel rather than three, and it lives in the outer six per cent of
        the frame's height, so it buys the load path for almost none of the picture.

        First member on the chrome layer, which is what makes it structural rather than
        decorative: the monitor stands in front of it, the eye's housing passes over it, and each
        mount's leg comes down and is gripped by a collar on it (see :meth:`_draw_bracket`).
        Only the plate's own webbing is behind it - see :meth:`_draw_webbing`.

        Once per window size, on a private layer so the corner cut is a mask and not a write.
        """
        y = float(self._spine_y())
        self._edge_rail(layer, [(0.0, y), (float(self.width), y)])

    def _draw_head(self, layer: Image.Image) -> None:
        """...and one across the top edge, turning down into a gusset at each corner.

        The other half of the load path, and the answer to the one thing every design critic has
        written down about this panel: the module across the top is bolted to nothing. Its two
        legs came down to row zero and stopped, its corner fixings sat over open picture, and
        the whole top edge - three quarters of it bare photo - explained nothing about what holds
        the thing up. A bar there and the legs land on it, which is what the fixings already
        through their knees have been claiming all along.

        The same section and the same width as the foot rail, because two sizes of member on a
        panel is a language and three is a pile of parts.

        It used to hug the edge - HEAD_DROP put its lit arris at 6.5, immediately inboard of the
        surround - and that was two mistakes at once. The surround's own crown sits at row 2 and
        this bar's at row 8, five pixels apart with a trough between them: one member under one
        lamp, lit twice, which is the fault every critic on the coherence lens has written down.
        And the state light pools in a cove *just* inside the surround, rows 10 to 15, which is
        exactly where this bar's face was: measured, the green bias of the metal there ran to
        +25 against +8 for the same steel a pixel further in, and the cove's own gradient
        flattened the section's fall to within two levels over six rows. Hardware standing in a
        wash is not hardware.

        So HEAD_DROP now stands the bar clear of both: the cove lights the picture between the
        case and the rail, as a cove is meant to, and the bar is its own member with its own
        single specular fifteen pixels away from the surround's. The module's legs come down to
        POD_STEP and land on its top edge, which they used to pass straight through.

        Where the foot rail runs off both ends into the frame's rounded corners, this one turns
        into them and stops: HEAD_BEND of formed corner, then HEAD_TURN of leg down the side.
        Its own former's radius rather than the frame's, because a bar this far inboard cannot be
        concentric with a corner it is inside of - and a member that ends *somewhere* is the
        difference between a frame and a stripe.

        Drawn with the foot rail, before everything: the module, the mounts and the companion's
        own housing all pass in front of it.
        """
        inset = HEAD_DROP * self.scale
        turn = inset + HEAD_TURN * self.scale
        bend = max(1.0, HEAD_BEND * self.scale)  # the radius the corner is formed to
        near, far = inset + bend, self.width - inset - bend
        self._edge_rail(layer, [
            (inset, turn),
            *arc_points(near, near, bend, 180.0, 270.0, HEAD_STEP),
            *arc_points(far, near, bend, 270.0, 360.0, HEAD_STEP),
            (self.width - inset, turn),
        ])

    def _edge_rail(self, layer: Image.Image, points: Sequence[tuple[float, float]]) -> None:
        """One of the two frame rails, cut to the panel's own rounded corners.

        Both hug an edge, so both would otherwise poke out past the radius into the four scraps
        of picture outside the border. The cut is a mask over a private layer rather than a write,
        because a write of anything translucent is a hole onto the camera.
        """
        bar = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        self._draw_rail(bar, points, max(4, round(SPINE_W * self.scale)), turned=1.0)
        corner = Image.new("L", (self.width, self.height), 0)
        ImageDraw.Draw(corner).rounded_rectangle(
            [0, 0, self.width - 1, self.height - 1], radius=self.radius, fill=255
        )
        cut = np.asarray(bar, np.float32)
        cut[:, :, 3] *= np.asarray(corner, np.float32) / 255.0
        layer.alpha_composite(Image.fromarray(cut.astype(np.uint8), "RGBA"))

    def _draw_webbing(self, layer: Image.Image) -> None:
        """Every bracket's stiffeners, first of everything, so no hardware is drawn through.

        Webbing is a feature of the plate and not of the rail on it, so it goes down with the
        plate's other graphics and the members come over the top. It used to be drawn inside
        :meth:`_draw_bracket`, which is after :meth:`_draw_spine`, so the bottom-right corner's
        three ribs hatched straight across the foot rail's highlight: measured, the rail's
        specular ran flat at 150 out to x=755 and then broke into a thirteen-pixel sawtooth
        dipping to 51.

        And cut to the picture's own opening, because the ribs cross a corner and the corner they
        cross is the panel's - so their far ends used to run out over the case's face, dipping it
        from 88 to 35 on the same thirteen-pixel pitch. The plate is behind the surround, and so
        is everything drawn on the plate.

        The one with a face on it has none: see :meth:`_draw_bracket` for why.
        """
        web = Image.new("RGBA", layer.size, (0, 0, 0, 0))
        for bracket in self.brackets.values():
            if bracket.corner and not bracket.seats:
                self._draw_ribs(web, bracket.corner, bracket.spine[0], bracket.spine[-1])
        band = frame_band(self.height)
        inside = Image.new("L", layer.size, 0)
        ImageDraw.Draw(inside).rounded_rectangle(
            [band, band, self.width - 1 - band, self.height - 1 - band],
            radius=max(1, self.radius - band), fill=255,
        )
        cut = np.asarray(web, np.float32)
        cut[:, :, 3] *= np.asarray(inside, np.float32) / 255.0
        layer.alpha_composite(Image.fromarray(cut.astype(np.uint8), "RGBA"))

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

        No two of them are the same tile. They used to be: one sprite cached on (radius, offset),
        which put the *same* crescent, the same grime and the same clocked hex on every fixing
        from the pod's knees to the bottom-left mount - two of them 434 px apart came back 82%
        bit-identical, and five of the seven measured the same to the decimal. So each one is
        built from where it is. The lamp is a bench light standing over the top-left of the
        panel, not a sun: :func:`material.bearing` turns this head's own vector to it into its
        own light, which swings the rim's bright crescent round by tens of degrees between the
        pod and the bottom-right mount and takes the seat shadow with it. On top of that, its
        position clocks the hex - a driver leaves no two sockets at the same angle - and seeds
        the grime round its rim and the lean of the spanner's ring; and :meth:`_sunlight` takes
        the whole head down by however far off the lamp it sits.

        Takes the draw and not the layer so that every caller keeps its one line. The layer is
        recovered from the draw, which Pillow has exposed since 9.x; the panel runs 12.
        """
        r = self.bolt_r if r is None else r
        layer: Image.Image = d._image
        cx, cy = math.floor(x), math.floor(y)
        lamp = (PLATE_LAMP[0] * self.width, PLATE_LAMP[1] * self.height)
        ax, ay = unit(x, y, *lamp)
        # Two odd primes off the head's own pixel: near neighbours land far apart, so the two
        # bolts through one knee are as unlike each other as the two ends of the panel.
        mark = (cx * 73856093 ^ cy * 19349663) % 65521
        tile = material.screw(
            round(r, 2), round(x - cx, 2), round(y - cy, 2), round(ax, 3), round(ay, 3),
            round(mark % 360 * math.pi / 1080.0, 3),  # a hex repeats every sixty degrees
            mark, round(float(self._sunlight(cx, cy, 1, 1)[0, 0]) ** BOLT_FALL, 3),
        )
        half = tile.width // 2
        # A head at the panel's edge would put the tile's corner off it, which alpha_composite
        # refuses; nothing here does, but a window size that did should lose a corner of shadow
        # rather than the panel.
        left, top = cx - half, cy - half
        crop = tile.crop((max(0, -left), max(0, -top), tile.width, tile.height))
        layer.alpha_composite(crop, (max(0, left), max(0, top)))

    def _draw_bracket(self, layer: Image.Image, bracket: Bracket) -> None:
        """The rail, then the bolts at every place it turns.

        The stiffeners are not in here any more - see :meth:`_draw_webbing`, which puts every
        bracket's down before any hardware at all, because a rib drawn after the foot rail
        hatches through it. They still stay near the corner: run them out towards the rail and
        they stop reading as webbing inside a bracket and start reading as stripes laid over the
        room, which is the one thing this layout is spending its corners to avoid. The pod has
        none: webbing braces a corner against a load, and a module hanging off the middle of an
        edge has no corner and nothing to brace.

        Nor does the one with a face on it. The loom leaves his housing into that same corner and
        there is about forty pixels of it, so webbing and cables in there together is not two
        details, it is a thicket - and of the two, the thing that says where the power goes wins
        over the thing that says the corner is stiff. `seats` is what asks the question, because
        the left mount is the only bracket that has one.
        """
        # Straight along the ramp even where he is seated on it. The rail used to bend round him
        # at his swell, and a 17 px half-round round a 66 px face was more rail than face; it runs
        # behind his housing now, and the collar goes on over it below.
        self._draw_rail(layer, bracket.spine)
        # Where a leg comes down onto the foot rail, the rail swells round it. That swell is the
        # clamp: no bolt, no plate, no second part - the silhouette alone says the two are gripped
        # together, and the section stays the section it was everywhere else.
        grip = max(4, round(SPINE_GRIP * self.scale))
        for x, y in (bracket.spine[0], bracket.spine[-1]):
            if y >= self.height - 1:
                self._draw_rail(layer, [(x - grip, self._spine_y()), (x + grip, self._spine_y())],
                                max(5, round(SPINE_W * SPINE_SADDLE * self.scale)))
        # ...then his housing, over the rail and under the bolts: the rail runs into the collar,
        # its brass edge goes all the way round him, and the bolts are what fix the two together.
        if bracket.seats:
            self._draw_collar(layer)
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

        Webbing lives in the deepest corner of the panel, which is the furthest thing on it from
        the lamp, so its steel is taken down by the same falloff every bar is (:meth:`_sunlight`).
        Left at full it was three bright diagonals across a plate that had just gone near black -
        a rib reads as a rib because it is a shade up from what it stiffens, not because it is
        the brightest thing in the corner.
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
        sun = float(self._sunlight(round(corner[0]), round(corner[1]), 1, 1)[0, 0])
        # The web's own hairline takes the falloff the way a bar's ridge does - as a fraction,
        # because it is a reflection of the lamp and not a lit face (see RAIL_SUN_SPEC). Taking
        # all of it, the three stiffeners in the deepest corner of the panel came out a dozen
        # levels off the plate they stand on, which is a gusset with no relief on it at all.
        crest = (*(round(c * sun**RAIL_SUN_SPEC) for c in material.STEEL_LIT), RIB_CREST)
        # The face between the two arrises, in two facets rather than one flat band. A web this
        # narrow has no room for a ramp, and a ramp is not what a milled chamfer does anyway: it
        # breaks into facets and each one holds its own angle, so the section steps down from the
        # lit arris instead of sliding. Both go on at the same alpha, so a facet is a change of
        # angle and not a change of how much room shows through the plate.
        alpha = round(255 * RIB_ALPHA)
        facets = (
            (mix(material.STEEL, material.STEEL_LIT, RIB_FACET), 1.0),
            (mix(material.STEEL, material.STEEL_DARK, RIB_FACET), -1.0),
        )
        step = max(1, w // 2)
        for i in range(RIB_N):
            t = 0.15 + 0.075 * i
            p = (corner[0] + (a[0] - corner[0]) * t, corner[1] + (a[1] - corner[1]) * t)
            q = (corner[0] + (b[0] - corner[0]) * t, corner[1] + (b[1] - corner[1]) * t)
            sd.line([(p[0] - lx * drop, p[1] - ly * drop), (q[0] - lx * drop, q[1] - ly * drop)],
                    fill=(0, 0, 0, RIB_SHADOW), width=w + 1)
            for tone, side in facets:
                off = side * step / 2.0
                rd.line([(p[0] + nx * off, p[1] + ny * off), (q[0] + nx * off, q[1] + ny * off)],
                        fill=(*(round(c * sun) for c in tone), alpha), width=step)
            rd.line([(p[0] + nx * edge, p[1] + ny * edge), (q[0] + nx * edge, q[1] + ny * edge)],
                    fill=crest, width=1)
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
        mount that runs into it are lit by the same lamp, which is what keeps the join reading
        as one assembly even though they are two metals.

        Four passes, in the order a real one is looked at. The shadow the ring drops onto the
        plate, first and underneath. Then the brass: a rolled outer edge that catches the lamp
        up-left and goes dark down-right, a face crowned just enough to be lighter on the lamp's
        side, brushed round the way a turned part is, its highlight and its colour drifting round
        the ring where it has been handled, and the sheet's own scratches running across it. Then
        the lip: a machined face of steel tilted out towards the glass all the way across, lit at
        ten o'clock and dark at four, with one bright line where it rolls over into the well, an
        index cut round its face, a catch along that same roll on the half the lamp never
        reaches, and a reveal at its outer edge where the brass stands over it. Then the seam
        between the two, a pixel of black. The well inside is the plate's.

        The lip's section is the one that had to be argued out. Left as a plain graded face it
        was brightest at its OUTER edge on every bearing but the lamp's - this lamp has more
        height than reach, so a face square to the viewer takes more of it than an edge-on one
        does - and a bright band holding one radius all the way round a ring is the tell that it
        was drawn rather than turned. The reveal is what a real bezel has there, it is not a
        function of where the lamp is, and it puts the fall back the right way: crest at the
        glass, graded face, dark reveal, black seam, brass.

        Each of the two metals carries a full section across its own width rather than a fill and
        an edge: a blown line of the lamp itself a pixel or two in from the lit edge, a graded
        face, a terminator on the far edge taken under the room's own light, and the ring's cast
        shadow occluding the plate beside it. That is four bands over a dozen pixels, and it is
        the difference between a ring that is coloured and a ring that is turned. The blown line's
        radius and brightness both wander round the ring - nothing is polished evenly, and a
        highlight holding one radius all the way round is the tell that it was drawn.

        Nothing here is written with an alpha below full - the shadow and every soft edge are
        composited from fields, or they would be windows onto the camera.
        """
        cx, cy = self.eye
        out = float(self.shoulder)
        inn = out * COLLAR_IN
        lip = max(1.5, COLLAR_LIP * self.scale)
        reveal = min(lip * 0.5, max(1.0, COLLAR_REVEAL * self.scale))
        roll = max(1.0, COLLAR_ROLL * self.scale)
        step = max(0.5, COLLAR_STEP * self.scale)
        lift = max(1.0, COLLAR_LIFT * self.scale)
        reach = math.ceil(lift * (material.SHADOW_DROP + 3 * material.SHADOW_SOFT)) + 2
        x0, y0, x1, y1, xs, ys = self._around(out + reach)
        dist = np.hypot(xs, ys)
        safe = np.maximum(dist, 1e-6)
        ux, uy = xs / safe, ys / safe
        along = self._round(xs, ys)
        lx, ly = material.lamp_2d()
        # His body again, over anything the panel's own hardware wrote across it. The foot rail
        # is laid down before this - the bezel has to strap over the rail and not the other way
        # round - and :meth:`_build_plate` is a layer *under* the chrome, so the rail's top
        # chamfer ran a still, near-white chord straight through the well: eighty pixels under
        # the pupil, at 189 against the pupil's own 253, and dead still in every frame of a six
        # second strip. Two blind critics named it independently, and both preferred the older
        # face that did not have it. A housing whose bezel stands in front of a rail and whose
        # glass does not is welded flat onto the backplate; what a cavity does where a member
        # passes behind it is stay dark. The plate is opaque out to the swell, so this only ever
        # puts back what was already underneath, and it goes down before the ring's own shadow
        # so that shadow still falls across the floor inboard of the flank.
        well = np.clip(inn + 0.5 - dist, 0.0, 1.0)
        body = self._plate.crop((x0, y0, x1, y1))
        body.putalpha(Image.fromarray(
            (np.asarray(body.getchannel("A"), np.float32) * well).astype(np.uint8), "L"))
        layer.alpha_composite(body, (x0, y0))
        # The whole ring's shadow before any of the ring, so the ring covers its own.
        cover = np.clip(0.5 - np.maximum(inn - dist, dist - out), 0.0, 1.0)
        shadow = material.cast(cover, lift) * COLLAR_SHADOW
        layer.alpha_composite(material.to_image(np.zeros((*cover.shape, 3), np.float32), shadow),
                              (x0, y0))
        # The brass, from the bevel's outer edge to the swell. Its face tilts outwards, from
        # level at the reveal to DOME at the roll, so the ring reads as crowned rather than as a
        # washer. Its inner edge does NOT roll: it stops at the reveal and goes dark there.
        # A roll turned inwards faces up and to the left on the FAR side of a ring, catches the
        # lamp there and puts a second bright arc opposite the first - which is an emboss, not a
        # lit part, and it is the one thing every critic of this panel measured. One lamp, one
        # highlight, and this edge is not it: what a reveal a pixel wide does is go dark.
        brass_in = inn + lip
        brass = np.clip(0.5 - np.maximum(brass_in - dist, dist - out), 0.0, 1.0)
        crown = COLLAR_CROWN * self._facets(
            (dist - brass_in) / max(out - roll - brass_in, 1e-3), COLLAR_FACETS)
        nx, ny, nz = material.roll_normals(np.maximum(out - dist, 0.0), ux, uy, roll, dome=crown)
        diffuse, spec = material.shade(nx, ny, nz)
        shut = np.clip(1.0 - (dist - brass_in) / step, 0.0, 1.0)
        diffuse, spec = diffuse * (1.0 - shut), spec * (1.0 - shut)
        rubbed = material.wear(along)
        spec = spec * (1.0 + COLLAR_WEAR * rubbed)
        brushed = material.grain(dist, along) + COLLAR_TARNISH * rubbed
        # The metal's own colour under the lamp, and then its highlight in its own colour too:
        # `steel` lays its highlight down in STEEL_SPEC, which on brass is a cool smear.
        rgb = material.steel(diffuse, np.zeros_like(spec), brushed, colour=BRASS)
        rgb = rgb + np.asarray(BRASS_SPEC, np.float32) * (material.SPEC * spec)[..., None]
        groove = np.exp(-(((dist - (out - roll - 1.0)) / 0.6) ** 2))
        rgb = rgb * (1.0 - COLLAR_GROOVE * groove)[..., None]
        rgb = rgb * self._graduation(dist, along / out, out - roll, out - roll - brass_in,
                                     COLLAR_TICKS, COLLAR_LONG, COLLAR_TICK,
                                     COLLAR_TICK_LONG)[..., None]
        sheet = self._marks[y0:y1, x0:x1]
        marks = (sheet * COLLAR_SCRATCH)[..., None]
        rgb = rgb * (1.0 - marks) + np.asarray(BRASS_SPEC, np.float32) * marks
        # Held to the steel's ceiling in luminance and not per channel. A per-channel clip lands
        # on red first on a warm metal, which leaves green the tallest channel it has: 177 px of
        # this face read G above R and came out at greenbias +31 with a p99 of +32, which is a
        # brass bezel with green highlights on it and is exactly the fault this round is about.
        # At his own radius, so the ceiling this face takes is the one _near_ceiling sets: near
        # the seam it is NEAR_CEILING and out on the roll it is the housing's full 206.
        rgb = self._stilled(rgb, dist)
        # ...and the two things that happen at the edges of the roll, both of them past what a
        # shaded face is allowed to reach. The far edge is occluded by the part it belongs to, so
        # it goes under the room's own light; the near one carries the lamp's own image, which is
        # the lamp's colour and not the metal's, and is the one mark on this panel that may go
        # brighter than STEEL_SPEC.
        facing = ux * lx + uy * ly
        rolled = np.clip(1.0 - (out - dist) / roll, 0.0, 1.0)
        away = np.clip(-facing, 0.0, 1.0) ** 1.5
        rgb = rgb * (1.0 - COLLAR_TERMINATOR * away * rolled)[..., None]
        ridge = out - roll * COLLAR_BLOWN_AT + COLLAR_BLOWN_WANDER * material.wear(
            along, seed=material.SEED + 5)
        hot = (np.exp(-(((dist - ridge) / COLLAR_BLOWN_W) ** 2))
               * np.clip(facing, 0.0, 1.0) ** COLLAR_BLOWN_ARC
               * (1.0 - COLLAR_BLOWN_DRIFT * np.clip(-rubbed, 0.0, 1.0))
               + sheet * np.clip(facing, 0.0, 1.0) * rolled)
        blown = np.clip(COLLAR_BLOWN * hot, 0.0, 1.0)[..., None]
        rgb = rgb + (np.asarray(BRASS_BLOWN, np.float32) - rgb) * blown
        layer.alpha_composite(material.to_image(self._stilled(rgb, dist), brass), (x0, y0))
        # The bevel: a face of steel turned outwards all the way across, rolling to edge-on at
        # the glass. `COLLAR_BEVEL` is what makes it a ring of metal rather than a stroke - the
        # whole width of it is lit at ten o'clock and dark at four, so the brightness goes round
        # the ring instead of sitting on one pixel of it, and the roll at the inner edge puts the
        # single bright line where the glass starts.
        wire = np.clip(0.5 - np.maximum(inn - dist, dist - brass_in), 0.0, 1.0)
        # Faceted across its width the same way the brass's crown is, and by the same trick: the
        # rise the roll is asked for is quantised, so the flats land where the metal turns.
        rise = self._facets(1.0 - (dist - inn) / lip, COLLAR_FACETS)
        diffuse, spec = material.shade(
            *material.roll_normals((1.0 - rise) * lip, ux, uy, lip, dome=COLLAR_BEVEL))
        # The highlight falls off with the roll, not with the face. Left flat, the flattest facet
        # out by the seam comes nearest to mirroring the lamp and puts a SECOND bright stroke on
        # the far edge of a section that already has one at the glass - which is the tell the
        # reference never shows: one specular on the chamfer that faces the lamp, then a fall
        # across the face, then the far edge is the darkest metal in the section.
        rgb = material.steel(diffuse, spec * COLLAR_BEVEL_SHINE * rise,
                             material.grain(dist, along) * 0.5 + COLLAR_WEAR * rubbed)
        # ...and the brass around it, bounced onto the half of it the lamp never reaches. A ring
        # of steel set inside a ring of brass has nothing else lighting that side, and without it
        # the bezel reads as one cool band with a warm one beside it rather than as two parts of
        # the same instrument.
        rgb = rgb + (np.asarray(BRASS, np.float32) * BRASS_BOUNCE
                     * (1.0 - diffuse)[..., None])
        # ...the index cut round it, from the same tool as the brass's scale and at a fifth of
        # its pitch. It hangs off the face rather than off the reveal, so the marks are read
        # against metal and not against the shadow the brass drops.
        rgb = rgb * self._graduation(dist, along / out, brass_in - reveal, lip - reveal,
                                     COLLAR_INDEX, COLLAR_INDEX_LONG, COLLAR_INDEX_TICK,
                                     COLLAR_INDEX_DEEP)[..., None]
        # ...the catch on the rolled edge at the glass, which is the whole of what the far side
        # of this ring gets. See COLLAR_CATCH: it goes on the edge and not on the face, and it
        # is capped at a third of the crest the lamp's own side carries.
        rgb = rgb + (np.asarray(material.STEEL_LIT, np.float32) - rgb) * (
            COLLAR_CATCH * rise * rise * (1.0 - np.clip(facing, 0.0, 1.0)))[..., None]
        # ...and the reveal at its outer edge, under the brass standing over it. All the way
        # round, because an overhang occludes on every bearing - it is the one band of this
        # section that is not a function of where the lamp is.
        rgb = rgb * (1.0 - COLLAR_REVEAL_DARK * np.clip(
            1.0 - (brass_in - dist) / reveal, 0.0, 1.0))[..., None]
        rgb = self._stilled(rgb, dist)
        # ...and the lamp itself in the lip, on the arc that faces it: the same line the brass's
        # roll carries, at the other end of the section and in the steel's own hue, so the two
        # metals are lit by one source and say so. Out on the face and not on the corner at the
        # glass - see COLLAR_LIP_AT - and narrow, because what is left of it has to read as a
        # glint on a turned edge rather than as satin.
        lit = np.exp(-(((dist - (inn + COLLAR_LIP_AT + COLLAR_BLOWN_WANDER * material.wear(
            along, seed=material.SEED + 6))) / COLLAR_LIP_W) ** 2))
        lit = lit * np.clip(facing, 0.0, 1.0) ** COLLAR_BLOWN_ARC
        lit = np.clip(COLLAR_LIP_BLOWN * lit, 0.0, 1.0)[..., None]
        rgb = rgb + (np.asarray(STEEL_BLOWN, np.float32) - rgb) * lit
        layer.alpha_composite(material.to_image(self._stilled(rgb, dist), wire), (x0, y0))
        # ...and the reveal between the two metals. A pixel of black is what says they are two
        # parts bolted together rather than one ring that changed colour half way across.
        seam = np.clip(0.5 - np.abs(dist - brass_in), 0.0, 1.0)
        layer.alpha_composite(material.to_image(
            np.broadcast_to(np.asarray(COLLAR_SEAM, np.float32), (*seam.shape, 3)), seam), (x0, y0))
        # ...and the mount points, in the arc the rail never reaches. The bracket bolts its own
        # two where the rail leaves the straight; without these the far side of the collar is a
        # ring resting against him rather than a housing fastened down all the way round.
        d = ImageDraw.Draw(layer)
        mid = (brass_in + out) / 2.0
        for deg in COLLAR_BOLTS:
            a = math.radians(deg)
            self._draw_bolt(d, cx + mid * math.cos(a), cy + mid * math.sin(a), self.bolt_r - 1)
        # ...and one last pass holding everything inside the seam under NEAR_CEILING, whatever
        # drew it. A bolt head is twelve pixels across on a face eight wide, so a fixing through
        # this bezel necessarily overhangs the lip it clamps, and `_draw_bolt` is the panel's -
        # shared with the rail's knees and both instruments - so it cannot be asked to know how
        # near the glass it has landed. Holding the *region* rather than the member is also what
        # makes the rule survive: inside the seam, nothing that never moves may pass the ceiling,
        # however it got there.
        held = np.asarray(layer.crop((x0, y0, x1, y1)), np.float32)
        lum = held[..., :3] @ np.asarray(LUMA, np.float32)
        under = np.where(dist <= brass_in,
                         np.minimum(1.0, NEAR_CEILING / np.maximum(lum, 1e-3)), 1.0)
        held[..., :3] *= under[..., None]
        layer.paste(Image.fromarray(held.round().astype(np.uint8), "RGBA"), (x0, y0))

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
        # ...and under the same falloff as every bar on the panel (:meth:`_sunlight`), which the
        # loom was the one part here not taking. It runs into the deepest corner there is, and a
        # nine-pixel cable mirroring the lamp as hard down there as the head rail does under it
        # came out as two hard white hairlines against a face of 25 - read, correctly, as flare
        # rather than as metal turning to the light. The face takes all of it and the highlight a
        # fraction, exactly as RAIL_SUN_SPEC says.
        sun = self._sunlight(x0, y0, x1 - x0, y1 - y0)
        diffuse, spec = material.shade(*normals)
        rgb = material.steel(diffuse * sun, spec * LOOM_GLOSS * sun**RAIL_SUN_SPEC, ribs,
                             colour=colour)
        layer.alpha_composite(material.to_image(
            self._stilled(rgb, self._radius(x0, y0, x1 - x0, y1 - y0)), cover), (x0, y0))

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
        # Held under the collar's own ceiling, and at the collar's own radius: its rolled corner
        # mirrors the lamp almost exactly, and unheld it was the brightest thing on the whole
        # panel outside the pupil - a fitting nobody is meant to look at, at 248. Its inner end
        # is screwed down inside the seam, so that end takes NEAR_CEILING with the lip beneath it.
        layer.alpha_composite(material.to_image(
            self._stilled(rgb, self._radius(x0, y0, x1 - x0, y1 - y0)), cover), (x0, y0))

    def _draw_framing(self, d: ImageDraw.ImageDraw, name: str, fade: float) -> None:
        """The framing's name inside the reticle, fading out.

        It goes in the one part of this panel deliberately kept empty, and it is the only thing
        that ever draws there - which is what earns it the middle rather than a corner. The mark
        under it says where the frame is; this says which frame, at the moment that changes, and
        then gets out of the way. The picture behind it is the real answer and arrives about a
        second later, which is the whole reason a word is needed at all: without one the panel
        holds a still frame for a second and looks like it missed the tap.

        Outlined rather than plated. Everything else here with words in it has a bracket or a tab
        under them; this lands on bare picture, which on a bench is as likely to be a sunlit
        window as a dark part, and a rim of the panel's own near-black holds the letters against
        either for eight short draws - where a plate would have to be built, faded and thrown
        away every frame of the second it is up.
        """
        alpha = round(255 * min(1.0, max(0.0, fade)))
        if alpha <= 0:
            return
        cx, cy = self.width // 2, self.height // 2
        tracking = max(1.0, 2.0 * self.scale)
        rim = max(1, round(1.5 * self.scale))
        for dx in (-rim, 0, rim):
            for dy in (-rim, 0, rim):
                if dx or dy:
                    self._text(d, cx + dx, cy + dy, name, self.font_mode, (*SCREEN, alpha),
                               align="c", tracking=tracking)
        self._text(d, cx, cy, name, self.font_mode, (*GREEN, alpha), align="c", tracking=tracking)

    def _corners(self, half: int, alpha: int) -> Image.Image:
        """Four right-angled corners around a point, as a tile. Nothing in the middle of them.

        What a camera shows you when it is looking rather than aiming. This was a gapped cross
        with graduations for exactly as long as it took somebody to look at it and say it read as
        a gun sight; the corners say frame and say nothing else, and they leave the centre of the
        picture - which is the part anybody actually points the thing at - completely clear.

        Each corner is one three-point stroke rather than two, so the bend is a joint PIL closes
        for us instead of a notch two separate legs leave at the outside of the turn. Still
        supersampled, for the ends rather than the lines: a mark this short is mostly its ends.

        A tile rather than a stroke onto the layer, and built about its own centre rather than
        the panel's, because the reticle travels now: it leaves the lens axis for whatever the
        model is pointing at (:mod:`cyclops.point`). Composited per frame at wherever that is,
        which costs one paste of about a hundred pixels square - against the supersampled draw
        this is, which is why it is built once and kept. The same shape at ``MARK_R`` of the
        size is what a ``ring`` mark is drawn with, so the thing pointing and the thing pointed
        at are visibly the same instrument.
        """
        leg = max(3, round(half * RETICLE_LEG))
        stroke = max(1.0, self.line * RETICLE_W)
        span = half + round(stroke) * 2

        def paint(t: ImageDraw.ImageDraw) -> None:
            for sx in (-1, 1):
                for sy in (-1, 1):
                    x, y = span + sx * half, span + sy * half
                    t.line(
                        [(at(x - sx * leg), at(y)), (at(x), at(y)), (at(x), at(y - sy * leg))],
                        fill=linear(GREEN_MID, alpha), width=round(wide(stroke)), joint="curve",
                    )

        return smoothed(2 * span + 1, paint)

    def _disc_tile(self, r: int, alpha: int) -> Image.Image:
        """The well a numbered marker's digit sits in: dark, rimmed, and its own size."""

        def paint(t: ImageDraw.ImageDraw) -> None:
            box = [at(0.5), at(0.5), at(2 * r - 0.5), at(2 * r - 0.5)]
            t.ellipse(box, fill=linear(SCREEN, alpha), outline=linear(GREEN, alpha),
                      width=round(wide(max(1.0, self.line))))

        return smoothed(2 * r + 1, paint)

    @staticmethod
    def _faded(tile: Image.Image, fade: float) -> Image.Image:
        """A kept tile at a fraction of its alpha, for a shape that is going out.

        One 256-entry lookup over one channel, against redrawing the thing supersampled: the
        difference between a mark costing a tenth of a millisecond and costing two. Nothing else
        about the tile changes as a gesture fades, which is what makes this sound.
        """
        if fade >= 1.0:
            return tile
        out = tile.copy()
        out.putalpha(tile.getchannel("A").point(lambda a: round(a * fade)))
        return out

    def _draw_reticle(self, layer: Image.Image, cx: int, cy: int) -> None:
        """The reticle, wherever it is this frame. One composite of a tile built at startup."""
        span = self._reticle.width // 2
        layer.alpha_composite(self._reticle, (cx - span, cy - span))

    def _label_tile(self, text: str) -> tuple[Image.Image, int]:
        """A word with its near-black rim, drawn once and kept. Also how far in the word starts.

        :meth:`_draw_framing`'s technique and for its reason, which applies to every label a
        mark carries: this lands on a live camera frame, as likely to be a sunlit window as a
        dark part, and eight offset draws hold it against either - where a plate behind it would
        have to be built, faded and thrown away every frame.

        Kept rather than drawn per frame because of what those nine draws cost. Measured on the
        Pi: one five-letter label was 2.3 ms of a 33 ms frame - more than the ring, the marker,
        the arrow and the reticle put together, and the shapes were the things that looked
        expensive. A label's text does not change while a gesture is up; only its alpha does.
        """
        kept = self._labels.get(text)
        if kept is not None:
            return kept
        if len(self._labels) > LABELS_KEPT:
            self._labels.clear()  # a bench full of pointing, not a leak; cheap to refill
        rim = max(1, round(1.5 * self.scale))
        pad = rim + 1
        width = round(self._width(text, self.font_read, 0.0)) + 2 * pad
        height = round(self.font_read.size * 1.8) + 2 * pad
        tile = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        t = ImageDraw.Draw(tile)
        middle = height / 2
        for dx in (-rim, 0, rim):
            for dy in (-rim, 0, rim):
                if dx or dy:
                    self._text(t, pad + dx, middle + dy, text, self.font_read,
                               (*SCREEN, MARK_ALPHA))
        self._text(t, pad, middle, text, self.font_read, (*GREEN, MARK_ALPHA))
        self._labels[text] = (tile, pad)
        return tile, pad

    def _rimmed_text(self, layer: Image.Image, x: float, y: float, text: str,
                     fade: float) -> None:
        """A kept label composited so its word starts at *x* and sits centred on *y*."""
        tile, pad = self._label_tile(text)
        layer.alpha_composite(self._faded(tile, fade),
                              (round(x) - pad, round(y - tile.height / 2)))

    def _label_anchor(self, ax: float, ay: float, offset: float,
                      width: float) -> tuple[float, float]:
        """Where a label actually goes, given the mark at *ax*,*ay* and how far clear of it.

        Right of the mark by default, flipped to its other side when that would run off the
        panel, and clamped inside either way. The thing being pointed at is as likely to be at
        the edge of the picture as anywhere else, and half a word hanging off the glass points
        at nothing.
        """
        pad = max(6, round(10 * self.scale))
        x = ax + offset
        if x + width > self.width - pad:
            x = ax - offset - width
        x = max(pad, min(x, self.width - width - pad))
        y = min(max(pad, ay), self.height - pad)
        # ...and clear of the four things on this panel a word cannot be read through: the
        # terminal's glass, the two instruments sunk into the bottom-right rail, and his face.
        # Everything else the chrome is made of is the tube filter over the live picture, which
        # rimmed text sits on perfectly well. Lifted rather than moved aside, because the mark it
        # belongs to is what fixes its column - slide it sideways and it labels the wrong thing.
        for box in self._label_clear:
            if x < box.right and x + width > box.x and y + pad > box.y and y - pad < box.bottom:
                y = box.y - pad
        return x, max(pad, y)

    def _draw_number(self, layer: Image.Image, d: ImageDraw.ImageDraw, mark: PanelMark,
                     fade: float, alpha: int) -> int:
        """A numbered marker: the digit sunk in a disc, for "these two, in that order"."""
        r = self._mark_disc.width // 2
        layer.alpha_composite(self._faded(self._mark_disc, fade), (mark.x - r, mark.y - r))
        self._text(d, mark.x, mark.y, str(mark.n), self.font_tab, (*GREEN, alpha), align="c")
        return r

    def _draw_arrow(self, layer: Image.Image, d: ImageDraw.ImageDraw, mark: PanelMark,
                    alpha: int) -> int:
        """A shaft from one point to another, with a head on the end. Returns the head's reach.

        The shaft goes straight onto the layer rather than through a tile: an arrow can cross
        the whole panel, and supersampling 800x480 to smooth one line is four times the panel's
        area spent on a stroke nobody reads the edges of. The head is a tile, because a head is
        all ends - the same reason the reticle's corners are.
        """
        stroke = max(1.0, self.line * RETICLE_W)
        d.line([(mark.x, mark.y), (mark.x2, mark.y2)], fill=(*GREEN_MID, alpha),
               width=round(stroke))
        size = max(5, round(MARK_HEAD_R * self.height))
        span = size + round(stroke) * 2
        if mark.x == mark.x2 and mark.y == mark.y2:
            return span  # nowhere to point: the shaft is a dot and a head has no direction
        along = math.atan2(mark.y2 - mark.y, mark.x2 - mark.x)

        def paint(t: ImageDraw.ImageDraw) -> None:
            for turn in (MARK_HEAD_SPREAD, -MARK_HEAD_SPREAD):
                angle = along + math.pi + turn
                t.line(
                    [(at(span), at(span)),
                     (at(span + math.cos(angle) * size), at(span + math.sin(angle) * size))],
                    fill=linear(GREEN_MID, alpha), width=round(wide(stroke)), joint="curve",
                )

        layer.alpha_composite(smoothed(2 * span + 1, paint), (mark.x2 - span, mark.y2 - span))
        return span

    def _draw_marks(self, layer: Image.Image, d: ImageDraw.ImageDraw,
                    marks: Sequence[PanelMark], fade: float) -> None:
        """What the model is pointing at, over the picture, fading out with the gesture.

        Everything here is drawn on bare camera frame, which is the one surface on this panel
        with no chassis behind it - so all of it is the phosphor and the near-black rim, and
        none of it is plated. A ring is the reticle's corners at :data:`MARK_R`, which is what
        makes the mark and the instrument that travelled to it read as the same thing.
        """
        fade = min(1.0, max(0.0, fade))
        alpha = round(MARK_ALPHA * fade)
        if alpha <= 0:
            return
        ring = self._faded(self._mark_ring, fade)
        span = ring.width // 2
        dot = max(2, round(3 * self.scale))
        for mark in marks:
            anchor = (mark.x, mark.y)
            if mark.kind == "ring":
                layer.alpha_composite(ring, (mark.x - span, mark.y - span))
                reach = span
            elif mark.kind == "n":
                reach = self._draw_number(layer, d, mark, fade, alpha)
            elif mark.kind == "arrow":
                reach = self._draw_arrow(layer, d, mark, alpha)
                anchor = (mark.x2, mark.y2)  # a label belongs at the end it points to
            else:  # tag: the thing itself is the mark, so this is only a place
                d.ellipse([mark.x - dot, mark.y - dot, mark.x + dot, mark.y + dot],
                          fill=(*GREEN, alpha))
                reach = dot
            if mark.label:
                gap = max(4, round(MARK_GAP * self.width))
                width = self._width(mark.label, self.font_read, 0.0)
                x, y = self._label_anchor(anchor[0], anchor[1], reach + gap, width)
                self._rimmed_text(layer, x, y, mark.label, fade)


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
        # The filter, and then the chrome drawn on it. There is nothing opaque underneath either
        # of them any more: the strip, the tab row and the four scraps outside the border's
        # rounded corners are all just the wash and the scanlines over the live picture, and the
        # picture between them is not covered at all - not even by the border's own corners.
        tags = self._tag_count(state, recording, heat)
        image = Image.alpha_composite(self._backdrop(tags), self._plate)
        # ...then the case, and then everything bolted to it. The surround goes on under the
        # chrome and over his body, which is the order it is in: the brackets' rails run under
        # the frame's inner arris instead of off the edge of the screen, his collar stands on it,
        # and the pod sits over the top rail - all of which is what a chassis boundary is for.
        image = Image.alpha_composite(image, self._surround)
        image = Image.alpha_composite(image, self._chrome(tags))

        d = ImageDraw.Draw(image)
        self._bake_header(d, state, recording, heat)

        # No state light and no border. Both were taken off on 2026-09-07 at the owner's word:
        # a green ring round the whole picture, changing colour with the session, was the last
        # thing on this panel still drawn as a HUD rather than as a machine. What it was for has
        # not gone away - the panel and the room still have to agree about whether a session is
        # running - it is the FACE that says so now, in the tint every mood carries, which is
        # where somebody looks anyway. See test_the_face_is_where_the_state_is_read.
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
        framing: str = "",
        framing_fade: float = 0.0,
        awake: bool | None = None,
        reticle: tuple[int, int] | None = None,
        marks: Sequence[PanelMark] = (),
        marks_fade: float = 0.0,
        look_at: tuple[int, int] | None = None,
        handed_over: bool = False,
        tutorial: Tutorial | None = None,
        plugged_in: Sequence[Device] = (),
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

        ``framing`` is which of the module's three the reticle has just been tapped to, and
        ``framing_fade`` how much of its second is left - the same shape as ``flash``, and for
        the same reason: what fades is the kiosk's to time, and what is drawn is ours.

        ``handed_over`` is a companion on the LAN holding his voice, and it turns the volume knob
        blue entire - hand, arc, hub, graduations, track and the mark in its gap - and touches
        nothing else on the panel. The knob is an icon for as long as it is up, not a control:
        the kiosk takes no presses on it either. Whenever the claim is held, session or not, the
        case worth catching is the glance at the panel *before* you start talking, when the voice
        is already routed away and this is the only thing that could tell you.

        ``tutorial`` is a walkthrough in progress. It takes the terminal's top row for a bar - one
        cell per step, lit up to and including the one being done - and its step's label becomes
        the sentence under it whenever ``detail`` has nothing to say.

        ``plugged_in`` is what is on the USB bus, from :mod:`cyclops.devices`, and it becomes up
        to three labels on the head rail. A label turning up *is* the feedback that something
        was plugged in - there is no toast and nothing to dismiss - so this is composited rather
        than baked, and none of it animates: a rail that breathed would be the panel asking to
        be looked at over a cable that was already plugged in an hour ago.
        """
        halo = HALOS.get(state, GREEN_DIM)
        layer = self._base(state, recording, heat).copy()
        d = ImageDraw.Draw(layer)

        self._draw_readouts(d, halo, level, elapsed, self._tag_count(state, recording, heat),
                            self._taping(state, recording), phase)
        # ...and the module in the opposite corner, which says what is plugged into the box.
        # Composited rather than baked, because a device arriving IS the feedback that it
        # arrived, and a layer built once per state cannot say so without being thrown away.
        self._draw_usb(layer, plugged_in)
        self._draw_caption(layer, state, halo, detail, phase, tutorial)
        held = pressed == "eye"
        # The pointers, over the faces the chrome laid down once. Neither inverts under a thumb
        # the way the switches here used to: you do not press an instrument, you turn one and
        # read the other, so the knob answers a finger by going white under it and the gauge
        # answers the tap that opens its screen the same way.
        self._draw_hands(layer, d, volume, temp_c, pressed, handed_over)
        # ...and the light in the pilot lamp beside them, which is the fourth thing on this panel
        # that says what the box is doing and the only one in that corner. Before the slider, so
        # a column coming up under a thumb covers it the way it covers everything else there.
        self._draw_pilot_light(layer, state, phase)
        if turning and volume is not None:
            self._draw_slider(d, volume)
        # The reticle, which is on the lens axis unless something is being pointed at. It used to
        # be baked into the chrome with everything else that holds still; it travels now, so it is
        # a composite of one kept tile - see _corners. The marks go over it and the framing's word
        # over both, which is the order they arrive in: the frame, the answer, the tap.
        self._draw_reticle(layer, *(reticle or (self.width // 2, self.height // 2)))
        if marks and marks_fade > 0.0:
            self._draw_marks(layer, d, marks, marks_fade)
        if framing and framing_fade > 0.0:
            self._draw_framing(d, framing, framing_fade)
        # Him, last of everything in his corner. He is the one control that never inverts under
        # a thumb: a face in photographic negative is not the same face, and half of him is over
        # the picture anyway, where there is nothing to invert. He acknowledges a tap by coming
        # up to full instead - which also holds for as long as the page behind him is loading,
        # so a slow browser looks like a box that heard you rather than one that ignored you.
        mood = self.engine.look(state, MOODS.get(state, MOODS[IDLE]), phase)
        # The lid answers to one fact and it is not the state: it is open while a session is up
        # and shut otherwise. Handed in rather than read off `state`, because by the time the
        # state gets here a background job may have turned it into WORKING - and the two cases
        # that hides are opposites. A box tidying up after a session must stay shut; a box making
        # a picture in the middle of one must stay open. Left None, the state's own mood decides,
        # which is what the preview harness and every test that renders a mood directly want.
        if awake is not None:
            mood = replace(mood, cover=0.0 if awake else 1.0)
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
        if look_at is not None and marks_fade > 0.0:
            # He looks at what he is pointing at, for as long as the mark is up. A place rather
            # than an angle, because that is the only way gaze is expressed here (eye.gaze_at) -
            # and it is written into `places` per frame because, unlike every other name in that
            # map, this one is wherever the mark went. `gaze` rides the fade so the pupil is
            # drawn home with the gesture rather than dropping you the instant it ends; `dart`
            # and `drift` off, because glancing round the room while pointing at something is
            # exactly what somebody who meant it would not do.
            self.engine.places[POINT] = unit(*self.eye, *look_at)
            mood = replace(mood, look=(POINT,), gaze=MOODS[LOOKING].gaze * marks_fade,
                           dart=0.0, drift=0.0)
        self.engine.paint(layer, *self.eye, mood, phase, level)
        layer.alpha_composite(self._glass, (self.eye[0] - self.eye_r, self.eye[1] - self.eye_r))
        if hold > 0.0:
            self._draw_hold(layer, hold)
        if menu:
            # Last of everything, because it is the only thing here that is asked a question
            # rather than told one: nothing behind it is live while it is up.
            self._draw_menu(layer, d, pressed)
        if flash > 0.0:
            # Green-white rather than white: a photo taken through a phosphor screen.
            d.rectangle([0, 0, self.width, self.height], fill=(214, 255, 228, int(190 * flash)))
        return np.asarray(layer)

    # ---- the pod ----

    def _pod_field(
        self, tags: int, x0: int, y0: int, w: int, h: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The pod's lip as a distance field over one box of the panel: negative inside, the
        unit direction from the lip to each pixel, which inside the window is inwards, and how
        far along the lip its nearest point is.

        The lip is where the flange's flat ends and the reveal starts - inside the rail by the
        land's width. The polygon is closed well above the panel so that bar_field's nearest-
        edge sign is a clean inside/outside and its closing edge never lands on a pixel: the
        window is open at the top, because the module hangs off the frame.
        """
        pod = self.pods[tags]
        lip = offset_path(pod.spine, -(self.rail_w / 2.0 + max(1.0, POD_LAND * self.scale)))
        above = -4.0 * self.rail_w
        ring = [*lip, (lip[-1][0], above), (lip[0][0], above), lip[0]]
        dist, ox, oy, along, side = material.bar_field(ring, x0, y0, w, h)
        return -dist * side, ox, oy, along

    def _pod_glass(self, sdf: np.ndarray) -> np.ndarray:
        """Coverage of the glass proper - inside the lip by the reveal's width, anti-aliased."""
        return np.clip(0.5 - (sdf + max(1.0, POD_REVEAL * self.scale)), 0.0, 1.0)

    def _pod_rake(self, tags: int, x0: int, y0: int, w: int, h: int) -> np.ndarray:
        """How much of the module's own key light reaches each pixel of one of its boxes.

        The lamp is a *place*, off the left shoulder of the module and above the panel, and its
        light falls off with distance - so the left knee is lit, the right knee is half lit, and
        every face between them grades. A drift across the module did this as a straight ramp
        before, and a straight ramp is a light with a direction and no position: the same eleven
        levels over a hundred and eighty pixels whichever end you read it from, which measures as
        a lamp at infinity and looks like no lamp at all.

        Normalised at the left knee and capped, so this only ever takes brightness away. The
        flange, the reveal, the scale cut into it and the glass are all multiplied by it, which
        is the whole point: one lamp, and everything on the module agrees about where it is.
        """
        pod = self.pods[tags]
        lamp = (pod.spine[-1][0] - POD_RAKE_OUT * self.scale, -POD_RAKE_UP * self.scale)
        reach = POD_RAKE_REACH * self.scale
        lit = material.glare(w, h, (lamp[0] - x0, lamp[1] - y0), reach, ambient=POD_RAKE_FLOOR)
        knee = pod.spine[-2]  # the left one, nearest the lamp: the face's own full brightness
        near = material.glare(1, 1, (lamp[0] - knee[0], lamp[1] - knee[1]), reach,
                              ambient=POD_RAKE_FLOOR)[0, 0]
        return np.minimum(lit / max(near, 1e-6), POD_RAKE_CEIL)

    def _pod_scale(
        self, rgb: np.ndarray, tags: int, xs: np.ndarray, sdf: np.ndarray,
        land: float, reveal: float, rake: np.ndarray,
    ) -> np.ndarray:
        """The graduation cut into the flange under the window, at the meter's own pitch.

        What was here was a blank pocket, and three critics called it what it was: a groove that
        holds nothing and says nothing. A face this deep cannot hold a plate - the pane took the
        room, which is the right trade - but it can hold a *scale*, and a scale is the one mark
        an instrument may carry that is not a word. It lines up with the segments above it, so
        the bar is read against a graduation the way every meter in the world is read, and every
        fourth one runs the whole face.

        A cut is three levels, the same three the window itself is: the floor turned out of the
        lamp's way and near black, the near wall darker still, and the far wall - the one on the
        side away from the lamp - giving back a fraction of what the face it is cut into has.
        Raked with everything else, so the marks at the far end are as dim as that steel is.
        """
        pitch = max(4.0, POD_TICK_PITCH * self.scale)
        box = self.pod_boxes[tags]
        # Off the meter's own left edge, so the marks stand under the cells rather than beside
        # them, and stopping short of the right-hand knee where the rail turns.
        first = box.x + self._seg[0] / 2.0
        index = np.round((xs - first) / pitch)
        cut = np.abs(xs - (first + index * pitch))
        wall = max(1.0, POD_TICK_W * self.scale)
        long_ = (np.abs(index % POD_TICK_LONG) < 0.5) & (index >= 0)
        # Under the bar and no further: this is the meter's scale, and a graduation running on
        # under the clock would be a ruler the panel has nothing to measure with.
        within = (xs >= box.x - pitch) & (xs <= box.x + self._meter_w + pitch / 2.0)
        deep = np.where(long_, land - 0.2, reveal + 0.5 + (land - reveal) * 0.55)
        face = within & (sdf > reveal + 0.4) & (sdf < deep)
        groove = np.clip(1.0 - cut / wall, 0.0, 1.0) * face
        # The far wall of the cut: one pixel beyond it, on the side the lamp is not.
        beyond = np.clip(1.0 - np.abs(cut - wall - 0.5) / 1.0, 0.0, 1.0) * face * (xs > first)
        dark = np.asarray(material.STEEL_DARK, np.float32)
        sunk = (POD_TICK_A * groove)[..., None]
        rgb = rgb * (1.0 - sunk) + dark * sunk
        # ...and the wall lifts its own steel rather than reaching for a colour of its own, so a
        # graduation at the dark end of the cove stays as dark as the cove.
        return rgb * (1.0 + POD_TICK_LIP * beyond * rake)[..., None]

    def _pod_mask(self, tags: int, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Coverage of the mask between the meter's windows: one septum per gap, at its pitch.

        A segmented meter is a dark plate with eight windows cut in it, and what lies between
        two windows is that plate. Ours had lit glass there, and that - not the skirt, once the
        skirt was cut - is what a critic measured when he said our lit cells fuse into a slab:
        with the bloom taken out of the gap entirely the five pixels still came back at the
        wash's own brightness, because they *were* the wash. The reference's gaps measure darker
        than its own glass.

        The windows' own height and no more. A plate would go over and under them too, and the
        light a lit cell throws up onto the glass and down onto the rail is the thing this
        module won its last round on: the mask stops the light of one cell reaching its
        neighbour, which is all it is for.

        On the pane and not in the meter's tile, because it is part of the instrument rather
        than part of the reading: it is there when every cell is dark, and it costs a frame
        nothing because the pane is baked.
        """
        box = self.pod_boxes[tags]
        seg_w, seg_h, gap = self._seg
        pitch = seg_w + gap
        # Where in one cell-and-gap each column falls, and the half-pixel the cells' own
        # anti-aliasing already covers taken off each end of the septum.
        at = np.mod(xs - box.x, pitch)
        along = np.clip(0.5 + gap / 2.0 - np.abs(at - (seg_w + gap / 2.0)), 0.0, 1.0)
        run = (xs >= box.x) & (xs <= box.x + self._meter_w)  # the eight cells and no further
        down = np.clip(0.5 + (seg_h + 1) / 2.0 - np.abs(ys - self.row), 0.0, 1.0)
        return along * run * down

    def _draw_pod_face(self, layer: Image.Image, tags: int) -> None:
        """The instrument in the pod: a dark glass window let into a steel flange, under the rail.

        The rail is the frame. What it framed used to be the same brushed wash the mounts are
        cut from, with the meter and the clock lying flat on it, and a readout lying on a plate
        is a label. These are lamps, and lamps sit behind glass: so the flat is a window now -
        the terminal's glass at the terminal's depth, on the filter's raster - set down inside a
        flange of the rail's own steel. Open at the top, because the module hangs
        off the frame: the glass runs up under the border the way the terminal's runs under its
        moulding, and a lip along the top would make it a box sitting on the panel.

        Fields off one distance function, the way the rail and the terminal are. The flange is
        not a member: it is the *root* of the rail's chamfer, a cove that leaves the bar's foot
        flat and descends POD_ROLL_TILT into the recess by the time the glass starts, leaning
        away from the window the whole way. So the lamp reaches less of it the nearer the glass
        it gets, the darkest steel on the module is the pixel the pane sits against, and the one
        specular anywhere on this run is the rail's own crest a few pixels outboard - which the
        cove now climbs into instead of competing with. It carried its own blown arris for two
        rounds, three pixels off the crest, and that read exactly as what it was: one edge drawn
        twice. What is left on the cove is a grade, and a grade is enough - it is brushed along
        its length, it is raked by a lamp standing off its left shoulder so the right end of
        every face is half the brightness of the left, it wears the sheet's hairlines and its
        own where the lamp is on it to catch them, and a graduation at the meter's pitch is cut
        into the bottom run from the glass down: four pixels of steel that is different in every
        one of them, because a face that is one grey for two hundred pixels is a fill and
        everybody can see that it is.

        The glass goes opaque hard against the reveal and eases to the terminal's depth over a
        few pixels, which is the dark seam round any pane in a frame; a rebate a few pixels wide
        steps down into near black all the way round, which is the shape that says the bezel
        *holds* the pane rather than being drawn beside it; the walls shade it further in; the
        lip nearest the lamp drops a shadow onto it by exactly the depth the glass sits down;
        and the border along the top, which is this window's own lip, drops its own. The glass
        itself is opacity, the phosphor's cast raked with everything else, a glare, and over
        that the cover's own reflection - one lobe with a hard leading edge and a soft trail,
        plus the second return off the far rim that says the glass has a thickness.

        Once per pod width, before the rail, so the rail's shadow falls on the land and the bolts
        sit over both. Nothing in here may be reached from a frame.
        """
        pod = self.pods[tags]
        half = self.rail_w / 2.0
        land = max(1.0, POD_LAND * self.scale)
        reveal = max(1.0, POD_REVEAL * self.scale)
        x0 = max(0, math.floor(min(x for x, _ in pod.spine)) - 1)
        x1 = min(self.width, math.ceil(max(x for x, _ in pod.spine)) + 2)
        y0, y1 = 0, min(self.height, math.ceil(pod.spine[2][1]) + 1)
        if x1 <= x0 or y1 <= y0:
            return
        w, h = x1 - x0, y1 - y0
        sdf, ox, oy, _ = self._pod_field(tags, x0, y0, w, h)
        glass = self._pod_glass(sdf)

        # Nearer the lamp is brighter, across the whole module - and "nearer" is a distance from
        # a place, not a fraction of the way across a box. See _pod_rake. Built before the steel
        # now, because the bounce off the pane is raked along the run like everything else.
        rake = self._pod_rake(tags, x0, y0, w, h)
        # The steel: the cove at the root of the rail's chamfer. The tilt runs from nothing at
        # the bar's foot to POD_ROLL_TILT where the glass starts, and it leans *away* from the
        # window the whole way - (ox, oy) is the way from the lip to the pixel, outwards on the
        # flange and inwards past it, so away is the negative of it inside the reveal and the
        # thing itself outside. Squared, so the steel nearest the bar stays flat and the descent
        # is all in the last two pixels, which is what a cove does. That leaves this face with
        # no highlight of its own at all: the normal is turned from the lamp everywhere it is
        # turned at all, so the run reads dark at the pane and climbs into the one crest.
        ys = (np.arange(h, dtype=np.float32) + y0)[:, None]
        xs = (np.arange(w, dtype=np.float32) + x0)[None, :]
        roll = np.clip((land - sdf) / max(1.0, land - reveal), 0.0, 1.0)
        tilt = POD_ROLL_TILT * roll * roll
        away = np.where(sdf < 0.0, -1.0, 1.0).astype(np.float32)
        nx, ny = ox * tilt * away, oy * tilt * away
        nz = np.sqrt(np.maximum(1.0 - tilt * tilt, 0.0))
        diffuse, spec = material.shade(nx, ny, nz)
        # The pane's return into the cove, hard against the glass and gone two pixels out: the
        # one light a face turned from the lamp may have, and it is worth a sixth of the lift
        # the crest carries. Raked, so it fades along the run with the lamp that caused it.
        back = np.clip(1.0 - (sdf + reveal) / max(1.0, POD_BOUNCE_W * self.scale), 0.0, 1.0)
        diffuse = diffuse * (1.0 + POD_BOUNCE * back * rake)
        # How much of the lamp this pixel can see, against a face square to it: what the wear and
        # the chips are worth here. A hairline shows because bare metal under it catches the
        # lamp, so on the turned part of the cove there is less of it to catch, and a mark drawn
        # at one brightness down a face that grades is a decal laid over the render.
        light = np.clip(diffuse / material.LAMP[2], 0.0, 1.0)
        rgb = material.steel(diffuse, spec, POD_GRAIN * material.grain(ys, xs),  # along the flange
                             colour=mix(material.STEEL, material.STEEL_LIT, POD_STOCK))
        rgb = np.minimum(rgb * rake[..., None], np.asarray(material.STEEL_SPEC, np.float32))
        # ...and the colour of it. The lift above is how much light the pane returns; this is
        # what colour a green window's light is on the steel it stands in, falling off over the
        # whole cove and raked with the lamp that lit the pane in the first place. Additive and
        # small: at its hottest row it is +5 of green-excess on metal that otherwise measures
        # +1, which is a bounce - twice that and the flange has stopped being steel.
        spill = np.clip(1.0 - (sdf + reveal) / max(1.0, POD_BOUNCE_REACH * self.scale), 0.0, 1.0)
        rgb = rgb + np.asarray(POD_BOUNCE_LIT, np.float32) * (spill * rake)[..., None]
        rgb = self._pod_scale(rgb, tags, xs, sdf, land, reveal, rake)
        # The wear: the sheet's own hairlines where they run onto the flange and the pod's own
        # dragged across it, both deterministic off material.SEED. Bright, and only a few chips
        # against them: what a used face carries is where the finish came *off* and the bare
        # metal under it catches the lamp. Darkening as many pixels as it brightens is a grain
        # overlay, which is a filter laid over a render rather than a history the part has.
        own = material.scratches(w, h, POD_SCRATCHES, (1.0, 0.0), seed=material.SEED + 9,
                                 spread=30.0, length=(20.0 * self.scale, 60.0 * self.scale))
        marks = (np.maximum(own, self._marks[y0:y1, x0:x1]) * POD_SCRATCH * rake * light)[..., None]
        rgb = rgb * (1.0 - marks) + np.asarray(POD_WEAR_HOT, np.float32) * marks
        pits = material.scratches(w, h, POD_PITS, (1.0, 0.0), seed=material.SEED + 11,
                                  spread=180.0, length=(0.5, 1.2))
        rgb = rgb * (1.0 - POD_PIT * light * (pits > 0.5))[..., None]
        flange = (1.0 - glass) * np.clip(0.5 + (half + land) - sdf, 0.0, 1.0)
        layer.alpha_composite(_to_image(rgb, flange), (x0, y0))

        # The glass: the terminal's well, opaque at the seam; the phosphor's cast; the shadow the
        # lip drops onto it; the raster in the filter's phase; and the room along its upper
        # edge. Built as an (rgb, alpha) pair and composited, never written - a translucent
        # write is a hole onto the camera.
        pane: tuple[np.ndarray, np.ndarray] = (
            np.zeros((h, w, 3), dtype=np.float32), np.zeros((h, w), dtype=np.float32)
        )
        # How far in from the seam each pixel is, 0 at the reveal and 1 on the open pane: the
        # opacity eases on it, and the cast and the glare are cut to it.
        open_ = np.clip(np.maximum(-(sdf + reveal), 0.0) / max(1.0, POD_SEAM * self.scale), 0, 1)
        open_ = open_ * open_ * (3.0 - 2.0 * open_)
        pane = _over(pane, SCREEN, 1.0 - (1.0 - TERM_ALPHA / 255.0) * open_)
        # The phosphor's wash, raked with the steel: brightest at the lamp's end of the pane and
        # gone by the far knee. A cast that covers a pane evenly is a tint on a decal.
        pane = _over(pane, GREEN, POD_TINT_A * open_ * rake * rake)
        # The mask between the meter's windows, behind everything the glass itself does - the
        # glare, the cover's reflection and the raster all lie over it, because it is a plate
        # behind the pane and they are on its face. See :meth:`_pod_mask`.
        pane = _over(pane, (0, 0, 0), POD_SEPTUM_A * self._pod_mask(tags, xs, ys))
        # The rebate: a step down into near black, hard against the reveal and the same width all
        # the way round, and then the walls shading further in - deepest at the wall and gone
        # POD_AO in, squared so it is a falloff and not a band. The step is the shape that says
        # the bezel holds the glass; the falloff is what puts the readouts *in* the window
        # instead of printed on it. A pane that meets its frame at one dark pixel has neither.
        inset = np.maximum(-(sdf + reveal), 0.0)
        rebate = np.clip(1.0 - inset / max(1.0, POD_REBATE * self.scale), 0.0, 1.0)
        pane = _over(pane, (0, 0, 0), POD_REBATE_A * rebate)
        wall = np.clip(inset / max(1.0, POD_AO * self.scale), 0.0, 1.0)
        pane = _over(pane, (0, 0, 0), POD_AO_A * (1.0 - wall) ** 2)
        # ...and the same lamp over the length of it. Steel is raked by multiplying it; a pane is
        # mostly whatever is behind it, so the far end of this one is raked by laying black over
        # it. Without this the window is evenly lit end to end, which no window ever is.
        pane = _over(pane, (0, 0, 0), POD_FALL_A * (1.0 - np.clip(rake, 0.0, 1.0)))
        recess = max(1.0, POD_RECESS * self.scale)
        pane = _over(pane, (0, 0, 0), material.cast(1.0 - glass, recess) * POD_SHADOW_A)
        pane = _over(pane, (0, 0, 0), np.broadcast_to(self._raster(y0, h)[:, None], (h, w)))
        # The room, over the glass's own depth - the pane ends where the reveal starts, well
        # above the rail's centreline this box runs down to, and a streak placed as a fraction
        # of the box would land on the steel.
        deep = math.ceil(pod.spine[2][1] - half - land - reveal)
        shine = np.zeros((h, w), dtype=np.float32)
        shine[:deep] = material.glare(w, deep, (GLARE_X * w, GLARE_Y * deep),
                                      POD_GLARE_REACH * w, POD_GLARE_FLOOR, POD_GLARE_STREAK)
        pane = _over(pane, WHITE, shine * rake * POD_GLARE_A * open_)
        # The cover's own reflection over that: a lobe that comes on in a row and goes off over
        # ten, running downhill across the pane, and the second return the far rim gives back.
        # Both cut to the open pane and raked, so the reflection is on the lamp's side of the
        # glass and the far end of the window is dark.
        down = ys / max(1.0, float(deep))
        across = (xs - x0) / max(1.0, w - 1.0)
        edge = (down - (POD_SWEEP_AT + POD_SWEEP_TILT * across)) / POD_SWEEP_LEAD
        trail = (down - (POD_SWEEP_AT + POD_SWEEP_TILT * across)) / POD_SWEEP_TRAIL
        lobe = np.exp(-(((across - POD_SWEEP_X) / POD_SWEEP_WIDE) ** 2))
        sweep = np.where(edge < 0.0, np.exp(-edge * edge * 9.0), np.exp(-trail * trail)) * lobe
        pane = _over(pane, POD_SWEEP_LAMP, POD_SWEEP_A * sweep * rake * open_)
        lx, ly = material.lamp_2d()
        away = np.clip(ox * lx + oy * ly, 0.0, 1.0)  # (ox, oy) points in: this is the far rim
        rim = np.clip(1.0 - inset / max(1.0, POD_RETURN_W * self.scale), 0.0, 1.0)
        pane = _over(pane, WHITE, POD_RETURN_A * rim * rim * away * open_)
        # ...and the border's shadow over the top of it, because the border is this window's lip
        # and the reflection is on the glass underneath it, not over it. Squared for the same
        # reason the walls are: a shadow with an edge on it is a stripe.
        lip = np.clip(1.0 - ys / max(1.0, POD_LIP_SHADE * self.scale), 0.0, 1.0)
        pane = _over(pane, (0, 0, 0), np.broadcast_to(POD_LIP_SHADE_A * lip * lip, (h, w)))
        layer.alpha_composite(_to_image(pane[0], pane[1] * glass), (x0, y0))

    def _raster(self, y0: int, h: int) -> np.ndarray:
        """How much darker each of *h* rows from panel row *y0* is behind the pod's glass.

        The filter's pitch and the filter's phase, at the terminal's strength: the one tube.
        """
        rows = (np.arange(h) + y0) % SCANLINE_EVERY == 0
        return rows.astype(np.float32) * (TERM_SCAN / 255.0)

    def _lamp(
        self, lamps: Image.Image, emitter: np.ndarray | None, colour: tuple[int, int, int],
        left: int, top: int, tags: int, scan: float = 1.0,
    ) -> tuple[Image.Image, tuple[int, int]]:
        """*lamps* - solid shapes in their own light - as seen through the pod's glass.

        The skirt is blurred off *emitter*, the coverage of the lit part alone, and worn in
        *colour*: twice over, tight and bright for the edge's own bloom and wide and faint for
        the glass lit round it. Pass ``None`` for a lamp that is not allowed one - the meter's
        cells are windows in a mask, and a mask's whole job is to stop the light of one window
        reaching its neighbour's gap, so the bar draws its own pixel of halation inside each
        cell and asks for no skirt at all.

        The skirt used to be blurred out of the lamps' own RGBA, and PIL blurs the colour
        channels through the transparency around a shape as well as the alpha - so every pixel
        of skirt came back part lamp and part black, and the further from the lamp it went the
        darker its light got. That is not what light does, and it is why a critic
        could measure our bloom at four levels one bar-height out where the reference has
        thirty-five: the alpha was there and the colour had been blurred out of it.

        Then what the window does to it. The lamp is behind the glass and the frame is in front,
        so the body of a lamp is cut to the pane - but a lit instrument puts colour on the metal
        round it, and LAMP_SPILL of the skirt carries past the reveal onto the flange and the
        rail. The raster runs through the lamps at *scan* of its depth on the glass, dimming
        their own colour on its rows and adding nothing where they are not, which is what keeps
        the tile's box from showing as a darker rectangle.

        *left* and *top* are where the tile would land on the panel. It comes back cut to the
        panel with where it now lands, because a skirt round something on the top row reaches
        above the panel and alpha_composite refuses a negative corner. Never done per frame:
        every caller bakes the result.
        """
        w, h = lamps.width, lamps.height
        body = np.asarray(lamps, np.float32)
        sdf, _, _, _ = self._pod_field(tags, left, top, w, h)
        glass = self._pod_glass(sdf)
        over = body[:, :, 3] / 255.0 * glass
        if emitter is None:
            alpha, rgb = over, body[:, :, :3]
        else:
            source = np.clip(emitter, 0.0, 1.0)
            mask = Image.fromarray((source * 255.0).astype(np.uint8), "L")
            def blurred(radius: float) -> np.ndarray:
                round_ = mask.filter(ImageFilter.GaussianBlur(radius))
                return np.asarray(round_, np.float32) / 255.0
            skirt = np.clip(LAMP_BLOOM_A * blurred(self.glow_r)
                            + LAMP_HALO_A * blurred(self.halo_r), 0.0, 1.0)
            # The body is cut to the pane. Past it the skirt keeps LAMP_SPILL of itself and the
            # widest of the three - the light thrown down onto the metal - is laid over that,
            # both cut to the module: the rail's outer edge is where this instrument stops, and
            # a skirt this wide would otherwise haze the camera under the whole pod.
            land = max(1.0, POD_LAND * self.scale)
            module = np.clip(0.5 + (self.rail_w + land) - sdf, 0.0, 1.0)
            metal = np.clip(module - glass, 0.0, 1.0)
            bounce = LAMP_BOUNCE_A * blurred(max(1.0, LAMP_BOUNCE_R * self.scale))
            under = np.clip(skirt * glass + (LAMP_SPILL * skirt + bounce) * metal, 0.0, 1.0)
            alpha = over + under * (1.0 - over)
            lit = np.asarray(colour, np.float32) * (under * (1.0 - over))[..., None]
            rgb = (body[:, :, :3] * over[..., None] + lit) / np.maximum(alpha, 1e-6)[..., None]
        rgb = rgb * (1.0 - scan * self._raster(top, h))[:, None, None]
        tile = _to_image(rgb, alpha)
        cut = max(0, -top), max(0, -left)
        if cut != (0, 0):
            tile = tile.crop((cut[1], cut[0], tile.width, tile.height))
        return tile, (left + cut[1], top + cut[0])

    def _meter(self, lit: int, halo: tuple[int, int, int]) -> tuple[Image.Image, int]:
        """The signal bar at *lit* of METER_SEGMENTS, in *halo*: one tile per reading, kept.

        Eight cells in the glass, and every one of them lands on exactly the pixels the eight
        rectangles it replaces used to fill - _meter_x still says where the bar starts - and the
        tile comes back with the row it starts on, cut to the panel by :meth:`_lamp`.

        A lit cell is a lens with a lamp behind it, not a rectangle of paint. It is hottest a
        pixel or two inside its own edge and falls off to SEG_FLOOR by the rim; the wall the
        light leaves by - the one away from the lamp, like every other return on this panel -
        gives some of it back brighter than the core; and where the core sits and how hard it
        burns wander cell to cell off the cell's own index. Ours held *two* luminance values
        across a whole cell and its two lit cells were byte-identical; the reference holds
        sixty-two and no two of its seven agree, and that difference is the whole of whether a
        meter is a light or a swatch.

        An empty cell is a well: a floor a little darker than the glass round it, walls on the
        far side lit by what the room gives them and near walls in their own shade - but lit on
        all four, because an aperture in a mask has an edge all the way round and ours drew two
        of them at the floor's own value, so half the scale could not be counted at all. All of
        it raked by the module's own lamp: an empty cell is lit by the room and a lit one is a
        light, so the first fades along the window and the second does not.

        Its light stops at its own window. What a lit cell lays outside the cell rect is one
        pixel of halation at SEG_BLOOM_R, drawn into this tile: no skirt is asked of
        :meth:`_lamp` at all. The two blurred skirts that used to do it reached four and seven
        pixels, which is further than the septum between two windows is wide, and the meter came
        back as one lit slab with a halo on the metal above it.

        The window round the bar is the same shape whichever pod is up - every pod pads the
        meter off its left-hand ramp by the same POD_PAD - so one tile per reading serves all
        three and is cut to the glass once. Nine readings by a handful of accents is a few
        dozen tiles of a few kilobytes each, and one composite of one of them is what a frame
        pays, against eight rectangle fills before and no glow at all.
        """
        key = (lit, halo)
        cached = self._meters.get(key)
        if cached is not None:
            return cached
        seg_w, seg_h, gap = self._seg
        m = self._skirt  # the margin _draw_readouts places this tile by, left and right
        # Only the halation stands outside a cell now, so the tile is the bar's own height and
        # two pixels: it used to run down to the far edge of the rail to carry a bounce that no
        # longer exists, and a frame paid to composite three times the transparent pixels.
        pad = max(1, math.ceil(2.5 * SEG_BLOOM_R * self.scale))
        w, h = self._meter_w + 1 + 2 * m, seg_h + 1 + 2 * pad
        _, _, meter_right = self._readouts(0)
        left = math.floor(self._meter_x(meter_right)) - m
        top = math.floor(self.row - seg_h / 2) - pad
        # The module's own lamp over the bar's box. A lit cell is a lamp and does not take it -
        # a light does not dim because it is standing further from another light - but an empty
        # one is a hole in a mask, lit by the room the way the glass round it is, and ours held
        # its rim at 133, 131, 134, 129, 133, 129 left to right while the window it sits in fell
        # by four to one. Two critics measured that as emitters ignoring the surface they are
        # in, and the dead cells competing with the live ones at 0.70 of a lit fill.
        #
        # SEG_WELL_RAKE of the module's fall and not all of it, because an aperture in a mask is
        # lit by the whole window and not only by the lamp at one end of it: taking the fall
        # whole put the last cell's edge a third under the first's, and a critic read the far end
        # of the scale as dissolving. The drift stays - it is what says these cells are cut in a
        # surface that is lit from one side - and no cell now steps more than a few counts off
        # its neighbour, which is the other half of what a scale has to do.
        cell_rake = 1.0 - SEG_WELL_RAKE * (1.0 - self._pod_rake(0, left, top, w, h))
        xs = np.arange(w, dtype=np.float32)[None, :]
        ys = np.arange(h, dtype=np.float32)[:, None]
        lx, ly = material.lamp_2d()
        jitter = np.random.default_rng(material.SEED + 17).uniform(-1.0, 1.0, (METER_SEGMENTS, 3))
        off = np.asarray(mix(SCREEN, GREEN_DIM, METER_CELL), np.float32)
        accent = np.asarray(halo, np.float32)
        body: tuple[np.ndarray, np.ndarray] = (
            np.zeros((h, w, 3), np.float32), np.zeros((h, w), np.float32)
        )
        # The phosphor's tooth, laid on every cell alike: fibres along the raster, a speck under
        # them. One field for the whole bar, so two cells at the same brightness still differ.
        tooth = 1.0 + SEG_GRAIN * material.grain(ys, xs)
        for index in range(METER_SEGMENTS):
            x = m + index * (seg_w + gap)
            cx, cy = x + seg_w / 2.0, pad + seg_h / 2.0
            hw, hh = (seg_w + 1) / 2.0, (seg_h + 1) / 2.0
            ax, ay = np.abs(xs - cx) - hw, np.abs(ys - cy) - hh
            edge = np.maximum(ax, ay)
            cover = np.clip(0.5 - edge, 0.0, 1.0)
            # Which wall each pixel is nearest, and whether that wall is turned from the lamp.
            gx = np.where(ax >= ay, np.sign(xs - cx), 0.0)
            gy = np.where(ax >= ay, 0.0, np.sign(ys - cy))
            away = np.clip(-(gx * lx + gy * ly), 0.0, 1.0)
            if index >= lit:
                # A well: the floor under the glass, and a rim brighter on the far walls than on
                # the near ones but standing off the floor on all four. The stroke lands on the
                # pixels PIL's outline used to. Its rim takes the same per-cell offset and the
                # same tooth a lamp does, because six empty cells drawn to one number are one
                # cell blitted six times.
                ring = np.clip(1.0 - np.abs(edge + 0.5), 0.0, 1.0)
                floor_ = np.clip(0.5 - (edge + 1.0), 0.0, 1.0)
                body = _over(body, (0, 0, 0), SEG_WELL * floor_)
                worn = (1.0 + SEG_VARY * jitter[index][2]) * tooth * cell_rake
                wall = off * (worn * (SEG_WELL_NEAR
                                      + (SEG_WELL_RIM - SEG_WELL_NEAR) * away))[..., None]
                over = ring[..., None]
                body = (body[0] * (1.0 - over) + wall * over, np.maximum(body[1], ring))
                continue
            # A lamp. Its core sits a fraction of a cell off centre and burns a few per cent
            # hotter or colder than its neighbour's, so no two lit cells - and no two of the
            # skirts they lay on the glass - are one blit. Off the cell's own index and not off
            # the order they are drawn in: a cell is a physical thing and looks the same at every
            # reading, so the eighth one may not change its mind when the seventh lights.
            jx, jy, amp = jitter[index]
            reach = max(1.0, SEG_EDGE * self.scale)
            inward = -np.maximum(np.abs(xs - cx - jx * SEG_WANDER * seg_w) - hw,
                                 np.abs(ys - cy - jy * SEG_WANDER * seg_h) - hh)
            t = np.clip(inward / reach, 0.0, 1.0)
            core = SEG_FLOOR + (1.0 - SEG_FLOOR) * t * t * (3.0 - 2.0 * t)
            back = np.clip(1.0 + (edge + 0.5) / max(0.5, SEG_RIM_W * self.scale), 0.0, 1.0)
            value = (core * (1.0 + SEG_VARY * amp) + SEG_RIM * back * back * away) * tooth
            # Its halation, and only its own: a Gaussian pixel of the cell's colour outside the
            # window, under the cell itself. At SEG_BLOOM_R nothing of it reaches the middle of a
            # septum, so the gap between two lit cells falls to the glass floor and the bar can
            # be counted - which is the only reason it is drawn in cells.
            out = np.clip(edge + 0.5, 0.0, None) / max(0.4, SEG_BLOOM_R * self.scale)
            bleed = SEG_BLOOM_A * (1.0 + SEG_VARY * amp) * np.exp(-out * out) * (1.0 - cover)
            body = _over(body, halo, bleed)
            over = cover[..., None]
            glow = np.clip(accent * value[..., None], 0.0, 255.0)
            body = (body[0] * (1.0 - over) + glow * over, np.maximum(body[1], cover))
        lamps = _to_image(body[0], body[1])
        tile, (_, row) = self._lamp(lamps, None, halo, left, top, 0, SEG_SCAN)
        cached = self._meters[key] = (tile, row)
        return cached

    def _digit(self, char: str, colour: tuple[int, int, int]) -> tuple[Image.Image, int]:
        """One character of the clock as a lamp behind glass: a tile per (character, accent).

        A glyph on this panel is a lit thing, and a lit thing has a halation - a pixel of its own
        colour, dimmed towards the screen's black, bleeding into the glass round it - and a core
        hot enough to have lost some of its colour, because light bright enough to be read across
        a workshop desaturates. Ours was one flat fill at exactly its own chroma, which a critic
        measured peaking at 128 against a reference's 208 and called what it was: paint.

        Both of those are one stroked ``d.text``, and a stroked text costs three plain ones - on
        every frame, in every state. So the glyph is baked instead. Twelve characters by a
        handful of accents is a bounded cache, and five composites of a twenty-pixel tile is
        cheaper than the single plain draw this replaced.

        Opaque, both of them: ImageDraw writes rather than composites, so a translucent halation
        would be a ring of camera round every digit. Returns the tile and the margin its own
        light is allowed, which is what the caller places it by.
        """
        key = (char, colour)
        cached = self._digits.get(key)
        if cached is not None:
            return cached
        pad = max(1, round(CLOCK_PAD * self.scale))
        size = (math.ceil(self.font_read.getlength(char)) + 2 * pad,
                round(self.font_read.size) + 4 * pad)
        tile = Image.new("RGBA", size, (0, 0, 0, 0))
        ImageDraw.Draw(tile).text(
            (pad, size[1] / 2), char, font=self.font_read, anchor="lm",
            fill=(*mix(colour, WHITE, CLOCK_CORE), 255),
            stroke_width=max(1, round(self.scale)),
            stroke_fill=(*mix(colour, SCREEN, CLOCK_HALATION), 255),
        )
        cached = self._digits[key] = (tile, pad)
        return cached

    def _rec_tile(self, tags: int) -> tuple[Image.Image, tuple[int, int]]:
        """The record light as a kept tile, for a pod showing *tags* of them.

        The one tag that is not baked into the base. It blinks, and a thing that blinks cannot
        live in a layer that is built once per state and copied - so it is built once here
        instead and composited by the frame that wants it, which is what :meth:`_meter` and
        :meth:`_digit` next to it already do with their own lamps.

        Its x is :meth:`_readouts`' and not the caller's: REC is always the first tag, so the
        count is the whole of what decides where it sits, and taking it from the same place the
        layout does is what stops the lit tag and the gap it leaves being two different boxes.
        """
        cached = self._rec.get(tags)
        if cached is None:
            _, at, _ = self._readouts(tags)
            cached = self._rec[tags] = self._tag_tile("REC", RED, tags, at, self.row)
        return cached

    def _pocket(self, layer: Image.Image, cx: float, cy: float, w: float) -> None:
        """The legend pocket milled into the rail under the window: empty, and lit as a socket.

        The one thing this module had none of was evidence of manufacture - a critic could
        measure that there was not an engraved mark anywhere on its face, while the panel it is
        judged against carries a stamped plate on the same bar. This is that pocket, and there
        is nothing in it: the panel gave its words up on purpose - the border says the state,
        the eye says the mood, the terminal says the rest - so what is left of a legend is the
        recess it would have been stamped into. A word in here would be a word back.

        Lit by the one lamp, like the socket in every cap screw on the panel: the two walls
        turned away from it are a dark line, the two it reaches down give back the steel's own
        highlight, and the near wall throws a shadow across the floor that recovers over three
        or four pixels. Shadow above, light below - which is the other way round from the
        bezel's chamfer a few pixels above it, so the two recesses cannot read as one stroke
        drawn twice.

        All of it is modulation: black and STEEL_LIT at an alpha over whatever is already there,
        so the rail's section, its brushing and its hairlines all run on through the floor. An
        opaque floor would be a picture of a pocket lying on a bar.

        Milled with the header and not with the face, because the rail is laid over the flange
        and this is cut into the rail - :meth:`_bake_header` is the first of the pod's own
        passes that happens after it. It is the same numpy a tag already pays for at a bake,
        over a box a tenth the size, and a bake is not a frame.
        """
        h = POCKET_H * self.scale
        deep = max(1.0, POCKET_DEEP * self.scale)
        m = math.ceil(2.0 * deep)  # room for the shadow the near wall throws, and for the AA
        left, top = math.floor(cx - w / 2.0) - m, math.floor(cy - h / 2.0) - m
        tw = math.ceil(cx + w / 2.0) + m - left
        th = math.ceil(cy + h / 2.0) + m - top
        xs = (np.arange(tw, dtype=np.float32) + left)[None, :]
        ys = (np.arange(th, dtype=np.float32) + top)[:, None]
        radius = max(1.0, POCKET_R * self.scale)
        dx = np.abs(xs - cx) - (w / 2.0 - radius)
        dy = np.abs(ys - cy) - (h / 2.0 - radius)
        sdf = (np.hypot(np.maximum(dx, 0.0), np.maximum(dy, 0.0))
               + np.minimum(np.maximum(dx, dy), 0.0) - radius)
        mouth = np.clip(0.5 - sdf, 0.0, 1.0)
        # Which way each wall faces, off the pocket's own distance field - the same read the
        # hex socket in material.bolt takes: the wall the light gets down is the far one.
        lx, ly = material.lamp_2d()
        rise_y, rise_x = np.gradient(sdf)
        facing = np.clip(-(rise_x * lx + rise_y * ly)
                         / np.maximum(np.hypot(rise_x, rise_y), 1e-6), 0.0, 1.0)
        # A wall a pixel deep, full for the whole of that pixel: the mouth lands on pixel edges
        # (see the half in POCKET_IN and POCKET_DROP), so the row against it is the wall and the
        # next row is already floor. Held up over the first half pixel, or the only row there is
        # of it comes out at two thirds and the pocket has no rim at all.
        deepen = np.maximum(-sdf, 0.0) / max(1.0, POCKET_WALL * self.scale)
        wall = np.clip(1.5 - deepen, 0.0, 1.0) * mouth
        pocket: tuple[np.ndarray, np.ndarray] = (
            np.zeros((th, tw, 3), np.float32), np.zeros((th, tw), np.float32)
        )
        pocket = _over(pocket, (0, 0, 0), POCKET_FLOOR * mouth)
        tooth = np.clip(0.5 + 0.5 * material.grain(xs, ys, material.SEED + 21), 0.0, 1.0)
        pocket = _over(pocket, (0, 0, 0), POCKET_TOOTH * tooth * mouth)
        # What the metal round it drops on the floor, cut to the floor: the same multiplicative
        # shadow the bolts and the rail cast, which recovers rather than stopping at a floor.
        pocket = _over(pocket, (0, 0, 0), POCKET_CAST * material.cast(1.0 - mouth, deep) * mouth)
        pocket = _over(pocket, (0, 0, 0), POCKET_SHADE * wall * (1.0 - facing))
        pocket = _over(pocket, material.STEEL_LIT, POCKET_LIP * wall * facing)
        layer.alpha_composite(_to_image(*pocket), (left, top))

    def _mark(self, layer: Image.Image, cx: float, cy: float, w: float, word: str) -> None:
        """The name of the machine, engraved into the legend pocket and filled with enamel.

        The pocket was milled and left blank, and a slot the exact shape of a nameplate with
        nothing in it reads as a missing part. What went into it first was a datum cross and a
        row of drilled dimples, on the argument that this panel had given its words up on
        purpose. Three critics measured that row and all three read it the same way: one glyph
        stamped nine times, adjacent cells correlating at r=0.47, a font that had failed to
        render. A nameplate is not the panel saying something - it is the name of the thing the
        panel is part of, which every instrument in a workshop carries and nobody reads twice.

        Single-line letterforms out of MARK_ALPHABET, walked by a cutter MARK_STROKE wide: at a
        cap height of six pixels a typeface closes its own counters up and C, O and P come back
        as one blob each, where a pantograph's stroke leaves them open. The row is stretched to
        land its last letter on the plate's far inset, so the word fills the pocket rather than
        however much of it the advance happened to come to.

        Then the polarity a cut mark actually has when it is meant to be read across a bench:
        the channel is filled, the fill catches the one lamp at its own albedo and stands eighty
        counts *over* the plate, and the only shadow left is the lip of the cut on the lamp's own
        side - one line above every stroke and none below. Cut and left bare it measured fifty
        counts *under* the plate, which is what printing looks like, not what milling does.

        All modulation over what the pocket left there, so the mill's tooth and the shadow its
        near wall throws run on through the letters. Laid out off the pocket's own box rather
        than a box of its own, because it is cut into the pocket: move POCKET_IN and the mark
        follows. Baked with the header, right after the floor it is cut into and long before the
        bolts, so nothing here composites over hardware.
        """
        h = POCKET_H * self.scale
        m = math.ceil(2.0 * max(1.0, POCKET_DEEP * self.scale))
        left, top = math.floor(cx - w / 2.0) - m, math.floor(cy - h / 2.0) - m
        tw = math.ceil(cx + w / 2.0) + m - left
        th = math.ceil(cy + h / 2.0) + m - top
        xs = (np.arange(tw, dtype=np.float32) + left)[None, :]
        ys = (np.arange(th, dtype=np.float32) + top)[:, None]
        cap = max(3.0, MARK_CAP * self.scale)
        wide = cap * MARK_WIDE
        tool = max(1.0, MARK_STROKE * self.scale)
        inset = MARK_INSET * self.scale
        # Ink to ink between the two insets, so the tool's own width is inside the plate at both
        # ends rather than the centreline being flush with it and half the stroke over the wall.
        first = cx - w / 2.0 + inset + tool / 2.0
        last = cx + w / 2.0 - inset - tool / 2.0 - wide
        step = (last - first) / max(1, len(word) - 1)
        cap_top = cy - cap / 2.0
        runs: list[list[tuple[float, float]]] = []
        for index, char in enumerate(word):
            at = first + index * step
            for path in MARK_ALPHABET[char]:
                runs.append([(at + px * wide, cap_top + py * cap) for px, py in path])
        sdf = _stroke_sdf(xs, ys, runs) - tool / 2.0
        # A milled channel has vertical walls, so a pixel half inside one is most of the way to
        # a full pixel of enamel rather than half of it - see MARK_HARD. Linear coverage on a
        # stroke this narrow halves its contrast wherever one straddles two columns, and the word
        # goes grey exactly where the letters are.
        cut = np.clip(0.5 - sdf, 0.0, 1.0)
        lx, ly = material.lamp_2d()
        rise_y, rise_x = np.gradient(sdf)
        facing = np.clip(-(rise_x * lx + rise_y * ly)
                         / np.maximum(np.hypot(rise_x, rise_y), 1e-6), 0.0, 1.0)
        shoulder = max(0.6, MARK_SHOULDER_W * self.scale)
        # The pixel just outside the stroke, on the side the lamp cannot get down: the lip the
        # enamel does not reach. `facing` is 1 on the far wall, as it is in the pocket itself.
        lip = np.clip(1.0 - np.abs(sdf - shoulder / 2.0) / shoulder, 0.0, 1.0) * (1.0 - cut)
        mark: tuple[np.ndarray, np.ndarray] = (
            np.zeros((th, tw, 3), np.float32), np.zeros((th, tw), np.float32)
        )
        mark = _over(mark, (0, 0, 0), MARK_SHOULDER * lip * (1.0 - facing))
        mark = _over(mark, MARK_ENAMEL, MARK_FILL * cut ** MARK_HARD)
        layer.alpha_composite(_to_image(*mark), (left, top))

    def _bake_header(
        self, d: ImageDraw.ImageDraw, state: str, recording: bool, heat: str = ""
    ) -> None:
        """The half of the pod that only moves when the state does: the heat lamp.

        One row, packed - the meter, whichever tags are lit, then the clock, with one stop
        between each (see :meth:`_readouts` for what that costs the clock and why it is worth
        it). The lamp goes here and not in :meth:`_draw_readouts` because it changes with the
        board and with nothing else, and the base is keyed on that.

        REC used to be baked here beside it and is not any more: it blinks, so it is a tile the
        frame composites - :meth:`_rec_tile`. What is still its business here is the *slot*. The
        tag's width is counted into the layout by :meth:`_tag_count` whether the light happens
        to be on this frame or off it, so the lamp lands in second place for the whole of a
        recording and does not slide left every time the red goes out.

        The legend pocket and the mark cut into it come along with them - see :meth:`_pocket`
        for why they are milled here rather than with the rest of the face - because this is the
        pass that runs after the rail, and the rail is what they are cut into.

        No state word. It had a corner of its own for a long time and three other things were
        already saying it better - the border's colour, the eye's mood, and the line under the
        picture, which can say "searching the web…" where a word could only say SEARCH.
        """
        count = self._tag_count(state, recording, heat)
        self._pocket(d._image, *self._legend_box(self.pods[count].spine[2], MARK_WORD))
        self._mark(d._image, *self._legend_box(self.pods[count].spine[2], MARK_WORD), MARK_WORD)
        _, at, _ = self._readouts(count)
        if self._taping(state, recording):
            at += self._tag_w + self._tag_gap  # REC's slot, kept whether it is lit or not
        colour = HEAT_LAMP.get(heat)
        if colour is not None:
            self._tag(d, at, self.row, HEAT_WORD, colour, count)

    def _legend_box(self, knee: tuple[float, float], word: str) -> tuple[float, float, float]:
        """Where a nameplate cut into a rail sits, given the knee its rail turns at: (cx, cy, w).

        One box for both passes - :meth:`_pocket` mills it and :meth:`_mark` cuts into what was
        milled - because two boxes that have to agree are two boxes that can disagree, and what
        they would disagree about is whether the word is inside its own pocket.

        The width follows the word: POCKET_W is what CYCLOPS needs, and the engraving stretches
        to fill whatever it is given, so a longer legend in a pocket sized for a shorter one is
        a row of letters at half the pitch of every other one on the panel.
        """
        w = POCKET_W * self.scale * len(word) / len(MARK_WORD)
        return (knee[0] - POCKET_IN * self.scale - w / 2.0,
                knee[1] + POCKET_DROP * self.scale, w)

    def _tag_tile(
        self, word: str, colour: tuple[int, int, int], tags: int, x: float, cy: float,
    ) -> tuple[Image.Image, tuple[int, int]]:
        """A tag as one finished tile: the slab, its glow, and the word knocked out of it.

        The whole tag in a single image, and where on the panel it lands - the shape :meth:`_lamp`
        hands back and :meth:`_meter` and :meth:`_digit` both keep. That is what lets REC blink:
        the blur is paid once here, and a frame that wants the tag pays one composite for it.

        The panel's way of shouting, and there are two things that do it: REC and the heat lamp.
        They were one shape typed out twice for exactly as long as it took to add the second, so
        they are one method now - which also means :attr:`_tag_w`, the width the pod is laid out
        against, is measured the same way the thing is drawn.
        """
        half = round(11 * self.scale)
        # Both tags are drawn to one width and the word centred in it, so REC and HOT side by
        # side are two slabs of the same size rather than two that nearly are.
        width = self._tag_w
        # A lamp behind the pod's glass, like the segments beside it: the slab drawn once into a
        # tile and its glow laid under it. Never per frame - the caller keeps the tile.
        m, left, top = self._skirt, math.floor(x), cy - half
        slab = Image.new("RGBA", (math.ceil(width) + 1 + 2 * m, 2 * half + 1 + 2 * m), (0, 0, 0, 0))
        ImageDraw.Draw(slab).rounded_rectangle(
            [m + x - left, m, m + x - left + width, m + 2 * half],
            radius=max(1, round(3 * self.scale)),
            fill=(*colour, 255),
        )
        shape = np.asarray(slab, np.float32)[:, :, 3] / 255.0
        tile, (px, py) = self._lamp(slab, shape, colour, left - m, top - m, tags)
        # The word goes in after the glow rather than before it: ImageDraw writes where
        # alpha_composite blends, so ink laid down first would simply be painted over by the
        # lamp. Its place comes off where the tile *landed* and not off where it was asked for,
        # because a skirt this wide runs off the top of the panel and _lamp crops what does -
        # which for a tag on the top rail is always. Measured from the uncropped corner the word
        # sits a skirt's height too high, inside the slab.
        self._text(ImageDraw.Draw(tile), x + (width - self.font_micro.getlength(word)) / 2 - px,
                   cy - py, word, self.font_micro, (*INK, 255))
        return tile, (px, py)

    def _tag(
        self, d: ImageDraw.ImageDraw, x: float, cy: float, word: str,
        colour: tuple[int, int, int], tags: int,
    ) -> float:
        """A filled rounded slab with a word knocked out of it, left edge at *x*. Returns width.

        What the heat lamp uses, which is baked into the base like everything else that only
        moves when the state does. REC goes through :meth:`_tag_tile` directly - see
        :meth:`_rec_tile` - because it has to be able to go out and come back between frames.
        """
        layer: Image.Image = d._image
        layer.alpha_composite(*self._tag_tile(word, colour, tags, x, cy))
        return self._tag_w

    # ---- what is plugged in ----

    def usb_room(self) -> float:
        """The rightmost x the module's top corner may reach: the pod at its widest, less air.

        Measured against ``pods[2]`` and never against the pod actually being drawn, so the
        module is the same width whether the tape is running or the board is hot. What that
        costs is that the corner is only as big as it is - see :meth:`usb_fit`, which drops what
        will not go in it.
        """
        return min(x for x, _ in self.pods[2].spine) - USB_CLEAR * self.scale

    def _usb_col(self, device: Device) -> tuple[str, float]:
        """One device's name, cut to what a column may take, and how wide that column comes out."""
        name = self._elide(device.name, self.font_micro, USB_NAME_W * self.scale)
        return name, max(USB_GLYPH * self.scale, self.font_micro.getlength(name))

    def usb_fit(self, found: Sequence[Device]) -> tuple[list[tuple[Device, str, float]], float]:
        """The devices the corner can actually show, and the x its flat ends at.

        Greedy and in bus order, so a column keeps its place for as long as the thing is plugged
        in. The module is as wide as what it is showing and not a pixel wider - the pod's own
        rule - with two floors under it: the legend cut into its rail has to fit inside it, and
        with nothing on the bus at all it is still there saying so.
        """
        pad, gap = USB_PAD * self.scale, USB_GAP * self.scale
        room = self.usb_room() - self.pod_ramp - 2 * pad
        shown: list[tuple[Device, str, float]] = []
        used = 0.0
        for device in found:
            name, w = self._usb_col(device)
            step = w if not shown else w + gap
            if used + step > room:
                continue
            shown.append((device, name, w))
            used += step
        if not shown:
            used = self.font_read.getlength(USB_EMPTY)
        return shown, max(USB_MIN_W * self.scale, round(used + 2 * pad))

    def usb_spine(self, right: float) -> list[tuple[float, float]]:
        """Its outline, wound like the pod's so ``(dy, -dx)`` still points out of its own body.

        The pod's right half, mirrored onto the left edge: square down from the top corner, then
        a true 45 into the flat, then along the bottom and off the left edge of the panel. Four
        points and no corner point - :class:`Bracket` closes the polygon against the corner it is
        given, which for this one is the panel's own top-left.
        """
        return [(right + self.pod_ramp, 0), (right + self.pod_ramp, self.pod_step),
                (right, self.pod_depth), (0, self.pod_depth)]

    def usb_box(self, right: float) -> Rect:
        """Everything the module can reach, rail's shadow and all - the tile the frame pastes.

        Wider and deeper than the plate it draws, because a rail on this panel throws a shadow
        past its own edge and a bolt blooms past its own head. One box for the crop and for the
        test that nothing outside it moved, so the two cannot come to different answers.
        """
        return Rect(0, 0, math.ceil(right) + self.pod_ramp + self._skirt,
                    self.pod_depth + self.rail_w)

    def _usb_chassis(self, right: float) -> Image.Image:
        """The module with nothing in it: plate, rail, bolts, and the legend cut into the rail.

        Built once per width and kept, the way :attr:`pods` is built once per tag count. Nothing
        in here depends on *which* devices are plugged in - only on how much room they came to -
        so plugging a different stick into the same port costs one dictionary lookup.

        A full-panel layer rather than a tile, because every part of it - the rail's mitres, the
        bolts' falloff, the pocket's lamp - is laid out in panel coordinates and would otherwise
        have to be told twice where it is. It is cropped to the module's own box on the way out,
        so what the frame composites is a couple of hundred pixels wide.
        """
        cached = self._usb.get(right)
        if cached is not None:
            return cached
        module = Bracket((0, 0), self.usb_spine(right))
        # The plate: the same wash, corner shading and scanlines every other plate on this panel
        # is cut from, through the module's own footprint. Never an opaque fill - a plate you
        # cannot see the room through is a lid, and every other plate here is a window.
        mask = Image.new("L", (self.width, self.height), 0)
        module.plate(ImageDraw.Draw(mask))
        # Square into (0, 0), and NOT cut back to the case's rounded corner. It was cut, on the
        # reasoning that a part running off a corner the case rounds is a part sticking out -
        # and what that actually left was a quarter-disc of the plate missing and the surround's
        # brightest corner, the one directly under the lamp, shining through the hole. A grey nub
        # in the corner of the green is the one thing this module cannot have: it is the reading
        # that says overlay rather than chassis. The plate covers the surround along the whole
        # top edge already, so covering it round the corner as well is the same part, finished.
        cut = np.asarray(mask, np.float32) / 255.0
        rgb, alpha = self._filter
        layer = _to_image(rgb, alpha * cut)
        # ...then the steel, on the two edges that meet the rest of the machine, and a bolt where
        # it turns - which is the same pass the pod and both mounts go through.
        self._draw_bracket(layer, module)
        # ...and the nameplate milled into that rail, after it, because it is cut INTO it.
        box = self._legend_box(module.spine[2], USB_LEGEND)
        self._pocket(layer, *box)
        self._mark(layer, *box, USB_LEGEND)
        box = self.usb_box(right)
        cached = self._usb[right] = layer.crop((box.x, box.y, box.right, box.bottom))
        return cached

    def _usb_entry(self, device: Device, name: str, w: float) -> Image.Image:
        """One device as a column: its category glyph, and its name under it.

        Kept per device, like the record light and for the same reason - it is composited by the
        frame rather than baked into anything, because a thing being plugged in is the whole of
        the feedback that it was plugged in, and a baked layer cannot say so without being
        thrown away.
        """
        key = (device.category, name)
        cached = self._usb_entries.get(key)
        if cached is not None:
            return cached
        span = USB_GLYPH * self.scale
        gap = USB_GLYPH_GAP * self.scale
        height = round(span + gap + self.font_micro.size + 2)
        tile = smoothed((round(w), height),
                        lambda t: self._usb_glyph(t, device.category, (w - span) / 2.0, 0.0, span))
        self._text(ImageDraw.Draw(tile), w / 2.0, span + gap + self.font_micro.size / 2.0,
                   name, self.font_micro, (*GREEN, 255), align="c")
        self._usb_entries[key] = tile
        return tile

    def _usb_glyph(self, t: ImageDraw.ImageDraw, category: str, x: float, y: float,
                   span: float) -> None:
        """The category, as the mark anybody already knows, in a *span* square at (*x*, *y*).

        His word on all four, and the argument for taking it: a camera body, an eighth note, a
        floppy disk and the USB trident are marks people have been reading for thirty years, and
        a glyph nobody has to learn is the only kind worth nineteen pixels. Drawn as silhouettes
        rather than as pictures, because at nineteen pixels a picture is a smudge and the outline
        is all that survives.
        """
        c = GREEN
        cx, cy, r = x + span / 2.0, y + span / 2.0, span / 2.0
        if category == CAMERA:
            # A body with a round lens and the viewfinder bump on its shoulder.
            t.rectangle([at(cx - r * 0.34), at(y + r * 0.10), at(cx + r * 0.04),
                         at(y + r * 0.42)], fill=linear(c))
            t.rounded_rectangle([at(x), at(y + r * 0.36), at(x + span), at(y + span)],
                                radius=round(wide(r * 0.22)), fill=linear(c))
            t.ellipse([at(cx - r * 0.42), at(cy + r * 0.24 - r * 0.42),
                       at(cx + r * 0.42), at(cy + r * 0.24 + r * 0.42)],
                      fill=linear(SCREEN))
            t.ellipse([at(cx - r * 0.20), at(cy + r * 0.24 - r * 0.20),
                       at(cx + r * 0.20), at(cy + r * 0.24 + r * 0.20)], fill=linear(c))
        elif category == MUSIC:
            # An eighth note: head, stem, flag. The one mark here that is taller than it is wide.
            stem = max(1.0, r * 0.20)
            t.rectangle([at(cx + r * 0.16), at(y), at(cx + r * 0.16 + stem), at(y + span * 0.80)],
                        fill=linear(c))
            t.polygon([(at(cx + r * 0.16 + stem), at(y)),
                       (at(x + span), at(y + r * 0.44)),
                       (at(x + span), at(y + r * 0.92)),
                       (at(cx + r * 0.16 + stem), at(y + r * 0.46))], fill=linear(c))
            t.ellipse([at(cx - r * 0.86), at(y + span * 0.56),
                       at(cx + r * 0.22), at(y + span)], fill=linear(c))
        elif category == STORAGE:
            # A floppy disk: the shutter across its top and the label across its bottom.
            t.rounded_rectangle([at(x), at(y), at(x + span), at(y + span)],
                                radius=round(wide(r * 0.16)), fill=linear(c))
            t.rectangle([at(cx - r * 0.40), at(y), at(cx + r * 0.40), at(y + r * 0.62)],
                        fill=linear(SCREEN))
            t.rectangle([at(cx - r * 0.56), at(y + span - r * 0.66), at(cx + r * 0.56),
                         at(y + span)], fill=linear(SCREEN))
        else:
            # The USB trident, which is on the end of the cable they are holding.
            shaft = max(1.0, r * 0.20)
            t.rectangle([at(cx - shaft / 2), at(y + r * 0.22), at(cx + shaft / 2), at(y + span)],
                        fill=linear(c))
            t.polygon([(at(cx), at(y)), (at(cx - r * 0.42), at(y + r * 0.50)),
                       (at(cx + r * 0.42), at(y + r * 0.50))], fill=linear(c))
            for side, tip in ((-1, "square"), (1, "round")):
                arm_y = y + span * (0.52 if side < 0 else 0.68)
                end = cx + side * r * 0.72
                t.line([at(cx), at(arm_y + r * 0.26)], fill=linear(c), width=round(wide(shaft)))
                t.line([at(cx), at(arm_y + r * 0.26), at(end), at(arm_y + r * 0.26)],
                       fill=linear(c), width=round(wide(shaft)))
                t.line([at(end), at(arm_y + r * 0.26), at(end), at(arm_y)],
                       fill=linear(c), width=round(wide(shaft)))
                if tip == "square":
                    t.rectangle([at(end - r * 0.26), at(arm_y - r * 0.30),
                                 at(end + r * 0.26), at(arm_y + r * 0.04)], fill=linear(c))
                else:
                    t.ellipse([at(end - r * 0.26), at(arm_y - r * 0.28),
                               at(end + r * 0.26), at(arm_y + r * 0.24)], fill=linear(c))

    def _draw_usb(self, layer: Image.Image, found: Sequence[Device]) -> None:
        """The module and whatever is in it, composited onto the frame."""
        shown, right = self.usb_fit(found)
        layer.alpha_composite(self._usb_chassis(right), (0, 0))
        if not shown:
            self._text(ImageDraw.Draw(layer), (right + USB_PAD * self.scale) / 2.0,
                       (self.pod_depth - self.rail_w / 2.0) / 2.0,
                       USB_EMPTY, self.font_read, (*GREEN_DIM, 255), align="c")
            return
        at_x = USB_PAD * self.scale
        for device, name, w in shown:
            layer.alpha_composite(self._usb_entry(device, name, w),
                                  (round(at_x), round(USB_TOP * self.scale)))
            at_x += w + USB_GAP * self.scale

    def _draw_readouts(
        self,
        d: ImageDraw.ImageDraw,
        halo: tuple[int, int, int],
        level: float,
        elapsed: float | None,
        tags: int,
        taping: bool = False,
        phase: float = 0.0,
    ) -> None:
        """The half that moves: the record light, the signal bar, and the clock counting up."""
        clock_right, _, meter_right = self._readouts(tags)
        # Red, and a filled tag rather than a dot. Red is what a record light is on every other
        # machine anybody has ever used, which is worth more here than the panel's preference for
        # its own green - and filling it rather than outlining it is how this tube shouts. The one
        # thing on screen that is red without being a fault, which is exactly why it is a tag with
        # a word in it and not a lamp.
        #
        # And it blinks, which is the other half of what everybody already knows a record light
        # does. A square wave rather than a breath, by the argument caption_pulse makes about its
        # cursor: this is a thing being switched, not a thing being dimmed. Off the phase and not
        # off a frame count, because the loop runs at 25 fps with the camera up and 5 behind the
        # admin page, and a blink counted in frames would gallop and stall with it.
        #
        # The slot it sits in is held by _tag_count whether this frame draws it or not, so the
        # lamp beside it and the clock after it do not shuffle left every second.
        if taping and phase % REC_PERIOD_S < REC_PERIOD_S * REC_DUTY:
            d._image.alpha_composite(*self._rec_tile(tags))
        whole = 0 if elapsed is None else int(elapsed)
        clock = "--:--" if elapsed is None else f"{whole // 60:02d}:{whole % 60:02d}"
        # Phosphor at rest with nothing to count, the state's accent the moment there is - the
        # numbers that only mean something during a session are the right place for the colour
        # that only appears during one. The phosphor at full and not the middle green, for the
        # same reason the resting meter is: these dashes sit beside a lit bar behind one pane.
        #
        # Character by character out of :meth:`_digit`, because the string changes every second
        # and the glyphs do not: five composites of a cached tile, which is cheaper than the one
        # plain d.text this used to be and a third of the price of the stroked one it wants.
        colour = GREEN if elapsed is None else halo
        layer: Image.Image = d._image
        at = clock_right - self._width(clock, self.font_read, 0.0)
        for char in clock:
            tile, pad = self._digit(char, colour)
            layer.alpha_composite(tile, (round(at) - pad, self.row - tile.height // 2))
            at += self.font_read.getlength(char)

        # The bar is a cached tile per reading - see _meter - and one composite of it is the
        # whole of what a frame pays for it. The layer is recovered from the draw, as _draw_bolt
        # does, so render keeps its one line.
        #
        # A segment is a lamp, and a lamp is lit or it is not. Asleep the accent is the dim green
        # the empty cells are already outlined in, so a lit cell drawn in it came out ten levels
        # brighter than an empty one's border and the count could not be read from a pace. A
        # resting panel's lit cells are the phosphor at full instead - still green, which is the
        # whole of what asleep promises - and every awake state's are its own accent, as before.
        lit = round(max(0.0, min(1.0, level)) * METER_SEGMENTS)
        tile, row = self._meter(lit, GREEN if halo == GREEN_DIM else halo)
        layer: Image.Image = d._image
        layer.alpha_composite(tile, (math.floor(self._meter_x(meter_right)) - self._skirt, row))

    def _draw_caption(
        self, layer: Image.Image, state: str, halo: tuple, detail: str, phase: float,
        tutorial: Tutorial | None = None,
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

        A walkthrough's step sits between the two: anything the controller has to say still wins
        the row, and the step comes back once it has said it. The bar above never moves for it.
        """
        rows, first = CAPTION_LINES, 0
        step = ""
        if tutorial is not None:
            layer.alpha_composite(self._bar(tutorial.total, tutorial.number),
                                  (self.caption_left, self.caption_top))
            rows, first, step = CAPTION_LINES - BAR_ROWS, BAR_ROWS, tutorial.current
        text = detail or step or CAPTIONS.get(state, "")
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
        lines = self._wrap(MARKER + text, font, limit, rows)
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
        top = self.caption_top + first * self.caption_h
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

    def _bar(self, count: int, lit: int) -> Image.Image:
        """The walkthrough's bar: *count* cells across the top row, the first *lit* of them lit.

        The signal meter's idiom at the terminal's scale - a lit cell is the phosphor at full and
        an empty one a dark well with a dim rim - and its economics: one tile per reading, with
        the halation blurred in once rather than per frame, the way :meth:`_print` lays a glyph's.
        Drawn into its own tile and composited, never onto the layer: ImageDraw writes, and a
        cell drawn straight onto the chrome would punch a hole through the glass under it.
        """
        key = (count, lit)
        cached = self._bars.get(key)
        if cached is not None:
            return cached
        width = self.caption_right - self.caption_left
        _, cell_h, gap = self._seg
        pitch = (width - gap * (count - 1)) / count
        top = (self.caption_h - cell_h) // 2
        wells = Image.new("RGBA", (width, self.caption_h), (0, 0, 0, 0))
        lamps = wells.copy()
        for index in range(count):
            x0 = round(index * (pitch + gap))
            box = [x0, top, round(index * (pitch + gap) + pitch) - 1, top + cell_h - 1]
            if index < lit:
                ImageDraw.Draw(lamps).rectangle(box, fill=(*GREEN, 255))
            else:
                ImageDraw.Draw(wells).rectangle(box, fill=(0, 0, 0, round(255 * BAR_WELL)),
                                                outline=(*GREEN_DIM, 255))
        glow = lamps.filter(ImageFilter.GaussianBlur(self.bloom_r))
        glow.putalpha(glow.getchannel("A").point(lambda a: round(a * BLOOM_ALPHA)))
        wells.alpha_composite(glow)
        wells.alpha_composite(lamps)
        self._bars[key] = wells
        return wells

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

    def _terminal_front(self) -> np.ndarray:
        """Coverage of the monitor's case over the whole panel, 0..1: what stands in the light.

        HARDWARE IS OPAQUE AND TOPMOST, and this is the one place on the panel where the rule was
        being broken by something drawn *after* the hardware. The state light is a cove - it pools
        under the surround's inner lip and falls inwards over the picture - and :meth:`_base` lays
        it over everything, which is right for the picture and wrong for the case bolted to the
        panel in front of it. Unmasked it put 0.31 of the state's own colour across the bottom
        moulding at one constant level from one end of it to the other, and that single composite
        did three separate things wrong: it washed a piece of steel in green (GREEN_DIM asleep,
        which measured +26.5 of green bias on metal held to +7), it lit a down-facing member as
        brightly as the up-facing one, and being flat in x it flattened the only falloff that
        member had - the bottom rail held 186-220 along its whole length while the top rail
        decayed 222 -> 141 beside it.

        So the pool runs BEHIND the case, which is where a pool of light under a lip runs when
        there is a monitor bolted in front of it. Built once, in __init__, off the same distance
        field the front itself is drawn from, so the two can never disagree about where it is.
        """
        front = np.zeros((self.height, self.width), np.float32)
        rows = slice(max(0, self.term.y), min(self.height, self.term.bottom))
        cols = slice(max(0, self.term.x), min(self.width, self.term.right))
        edge = tube_field(self.term.w, self.term.h, self.case_r)
        cover = np.clip(0.5 - edge, 0.0, 1.0)
        front[rows, cols] = cover[: rows.stop - rows.start, : cols.stop - cols.start]
        return front

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

        And the glass is opaque, which the module docstring has always claimed and this method
        was never quite doing. Four fifths is not opaque: stretch the contrast of the pane and
        the workshop is behind the caption, and the green coming through the back competes with
        the green the letters are made of. The pane has its own light now instead - a fall from
        top to bottom for the curve, a lift where the lamp stands in it, the shadow the lip casts
        across the top of it, one catch-light in the upper-left corner and one wipe of the room
        - all of it clipped to the pane, none of it crossing onto the metal.

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
        scale = self.scale
        # The shadow the frame drops on the panel, first, so the front covers its own the way a
        # real one does. On a box a margin wider than the front, because a shadow falls outside
        # the thing that drops it - and only there, since inside it the recess is the shade.
        #
        # Two darks, not one. The cast shadow falls down and to the right, away from the lamp,
        # and says the frame stands proud; the contact band hugs the whole outline, and says it
        # is *seated*. Without the second one the top rail's specular is the outermost pixel of
        # the part, and a bright line with nothing behind it reads as a stroke round a rectangle.
        # The padded field is the same rounded rect grown by the margin, so it is the front's own
        # distance carried out into the pad rather than a blur standing in for one.
        lift = max(1.0, TERM_LIFT * scale)
        pad = math.ceil(lift * (material.SHADOW_DROP + 3 * material.SHADOW_SOFT)) + 1
        out = tube_field(box.w + 2 * pad, box.h + 2 * pad, self.case_r + pad) + pad
        front = np.clip(0.5 - out, 0.0, 1.0)
        cast = material.cast(front, lift) * TERM_SHADOW
        contact = np.clip(1.0 - out / max(1.0, TERM_CONTACT_W * scale), 0.0, 1.0) ** 2
        shadow = (1.0 - (1.0 - cast) * (1.0 - contact * TERM_CONTACT)) * (1.0 - front)
        tile = material.to_image(np.zeros((*shadow.shape, 3), np.float32), shadow)
        left, top = box.x - pad, box.y - pad
        tile = tile.crop((max(0, -left), max(0, -top), tile.width, tile.height))
        layer.alpha_composite(tile, (max(0, left), max(0, top)))
        # The glass, in passes on one (rgb, alpha) pair: the phosphor it is made of and the lamp
        # standing in it, the room wiped across it and the one place its curve catches the lamp.
        # Built out here rather than drawn because ImageDraw writes rather than composites - a
        # translucent stroke laid over this would punch a hole through the glass onto the camera
        # instead of dimming it.
        face: tuple[np.ndarray, np.ndarray] = (
            tube_glow(box.w, box.h, self.case_r, self.bezel, box.y),
            tube_alpha(box.w, box.h, self.case_r),
        )
        face = _over(face, GLARE_LAMP, glare_alpha(box.w, box.h, self.case_r, self.bezel, scale))
        face = _over(face, GLARE_LAMP, sheen_alpha(box.w, box.h, self.case_r, self.bezel, scale))
        # ...and the rebate over all three of them, LAST and as a multiplier. The pane is at the
        # bottom of a well and the walls of that well shade whatever the pane is showing there -
        # the phosphor, the wipe and the corner blowout alike - so this is not another pass with
        # a colour of its own, it is a gain on the ones above (:func:`rebate_gain`). Laid as
        # black underneath them instead, each of those passes then paints its own light back over
        # the shadow and the aperture ends on a bright line, which is the opposite of a recess.
        rgb, alpha = face
        rgb *= rebate_gain(box.w, box.h, self.case_r, self.bezel, scale)[..., None]
        face = (rgb, alpha)
        # ...and one ceiling on all of it, over the rows the line prints on. No pixel of the
        # glass there may reach the brightness at which the panel's own reader takes it for a
        # letter (tests/test_caption.py finds ink by summing the channels), because a reflection
        # over a word is a word lost. Enforced rather than argued, like the well's own
        # WELL_CEILING, and enforced on the finished pane rather than on each pass - so every
        # pass stays honest and only their sum is held, and the rest of the glass, which is the
        # strip above the first line where the rim reflection runs and the corner it blows out
        # in, takes all the light it should.
        block = slice(self.caption_top - box.y,
                      self.caption_top - box.y + CAPTION_LINES * self.caption_h)
        held = rgb[block]
        total = np.maximum(held.sum(-1, keepdims=True), 1e-3)
        room = float(TERM_INK_CEIL - TERM_INK_KNEE)
        eased = TERM_INK_KNEE + room * (
            1.0 - np.exp(-np.maximum(total - TERM_INK_KNEE, 0.0) / room)
        )
        rgb[block] = held * np.minimum(1.0, eased / total)
        layer.alpha_composite(_to_image(rgb, alpha), (box.x, box.y))
        # ...and the frame over it, off the same field. The roll is the rail's, capped so a frame
        # at a small window keeps a flat between its two edges rather than becoming a wire.
        roll = min(max(1.5, TERM_ROLL * scale), self.bezel / 2.0)
        frame = tube_frame(box.w, box.h, self.case_r, self.bezel, roll, scale)
        layer.alpha_composite(material.to_image(*frame), (box.x, box.y))
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
        """Half-width of the tile either instrument's hand and scale are drawn into, in panel px.

        The face itself is drawn into a wider one, because its shadow falls past its own edge.
        """
        return round(self.btn_r) + max(2, round(4 * self.scale))

    def _instrument_tile(self, name: str, away: bool = False) -> tuple[Image.Image, int]:
        """One instrument, less its pointer, and how far out from its centre the tile reaches.

        Built as fields the way the bars are, not as rings of stroke: one distance from the centre,
        and the shadow, the recess, the glass and the bezel's rolled edge all come off it, lit by
        the panel's one lamp through :mod:`cyclops.material`. Nothing here decides which side of
        anything is bright. The scale alone is still drawn, through :func:`eye.smoothed`, because
        ticks and arcs are strokes and a stroke is what PIL is for.

        Bottom to top, each a composite onto one private tile: the shadow, so the instrument
        covers its own; the face, which is the well the old switch left and keeps its opacity, so
        the rail runs under it as it always did; the scale, printed on the face and dimmed by
        what the face's own shading leaves it (:meth:`_dial_light`), because ink on a plate is
        under everything the plate is under; the glass, which lies over the scale because that is
        where glass is; and the bezel last, because its inner edge is what cuts the glass off
        clean.

        The two instruments are given a different seed and a different amount of polish. They
        are the same part off the same lathe under the same lamp, and they have not had the same
        life; drawing both from one set of numbers is what let a critic measure our two bezels as
        one sprite, mean absolute difference 0.4 of a level where the reference's two differ by 18.

        *away* is the knob relit for a companion holding his voice, and it is the same instrument
        by the same numbers with a scale in the other hue - which is the point, and the reason it
        is built here rather than painted over the top somewhere. The graduations, the unlit
        track and the skirt they bleed onto the face are all under the glass and inside the
        bezel; a second pass at any of them from outside would have to reach through both, and
        what that actually draws is a green halo round a blue dial (measured, and looked at).
        It drops its shadow, because the instrument it covers has already dropped that one and
        two of the same shadow is a ring of dirt round the bezel.
        """
        r = self.btn_r
        # Which instrument this is decides how much finish is left on its ring; its seed decides
        # where the brushing, the wear and the scratches on it fall.
        index = SWITCHES.index(name)
        seed = material.SEED + DIAL_SEED_STEP * index
        spread = 1.0 - 2.0 * index / max(len(SWITCHES) - 1, 1)
        polish = DIAL_BEZEL_TURNED * (1.0 + DIAL_TARNISH * spread)
        lift = max(1.0, DIAL_LIFT * self.scale)
        reach = math.ceil(r + lift * (material.SHADOW_DROP + 3 * material.SHADOW_SOFT)) + 1
        dist, ux, uy = self._dial_grid(reach)
        lx, ly = material.lamp_2d()
        facing = ux * lx + uy * ly  # 1 where the way out from the centre is the way to the lamp
        disc = np.clip(0.5 - (dist - r) * DIAL_SS, 0.0, 1.0)

        tile = Image.new("RGBA", (2 * reach + 1, 2 * reach + 1), (0, 0, 0, 0))
        if not away:
            _, cover = self._boxed(np.zeros((*disc.shape, 3), np.float32), disc)
            shadow = material.cast(cover, lift) * DIAL_SHADOW
            tile.alpha_composite(
                material.to_image(np.zeros((*cover.shape, 3), np.float32), shadow))
        tile.alpha_composite(self._dial_layer(*self._face(dist, facing, disc, r)))
        span = self.dial_span
        scale = self._printed(smoothed(
            2 * span + 1, lambda t: self._paint_scale(t, span, name, away)))
        # The scale is printed in phosphor, so it bleeds onto the face the way the caption bleeds
        # onto its glass: the skirt first, then the crisp marks over it.
        tile.alpha_composite(self._phosphor_bloom(scale, span, 0.0),
                             (reach - span, reach - span))
        tile.alpha_composite(scale, (reach - span, reach - span))
        tile.alpha_composite(self._dial_layer(*self._dial_glass(dist, facing, reach, r)))
        # The ring's phosphor with it: what leaks into the seat is the light off this face, and
        # a blue dial in a ring with a green reveal round it is two instruments in one hole.
        tile.alpha_composite(self._dial_layer(
            *self._bezel(dist, ux, uy, r, seed, polish, BLUE if away else GREEN)))
        return tile, reach

    def _stamp_instrument(self, layer: Image.Image, name: str, tile: Image.Image,
                          reach: int) -> None:
        """Put an instrument's tile on *layer*, centred on where that instrument is bolted."""
        cx, cy = (round(v) for v in self.switches[name])
        # A window small enough to put the tile's corner off the panel should lose a corner of
        # shadow rather than the panel; alpha_composite refuses a negative corner.
        left, top = cx - reach, cy - reach
        crop = tile.crop((max(0, -left), max(0, -top), tile.width, tile.height))
        layer.alpha_composite(crop, (max(0, left), max(0, top)))

    def _dial_grid(self, reach: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Distance from the instrument's centre and the unit way out from it, DIAL_SS times over.

        Sample centres, not corners: the tile is ``2 * reach + 1`` panel pixels with the centre on
        the middle one, and a sample sits a half-step into its share of that pixel.
        """
        size = (2 * reach + 1) * DIAL_SS
        grid = (np.arange(size, dtype=np.float32) + 0.5) / DIAL_SS - 0.5 - reach
        xs, ys = grid[None, :], grid[:, None]
        dist = np.hypot(xs, ys)
        safe = np.maximum(dist, 1e-6)
        return dist, xs / safe, ys / safe

    @staticmethod
    def _boxed(rgb: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """A supersampled (rgb, alpha) pair averaged down by DIAL_SS, colour weighted by cover."""
        h, w = alpha.shape
        cover = alpha.reshape(h // DIAL_SS, DIAL_SS, w // DIAL_SS, DIAL_SS).mean(axis=(1, 3))
        lit = (rgb * alpha[..., None]).reshape(h // DIAL_SS, DIAL_SS, w // DIAL_SS, DIAL_SS, 3)
        return lit.mean(axis=(1, 3)) / np.maximum(cover[..., None], 1e-6), cover

    def _dial_layer(self, rgb: np.ndarray, alpha: np.ndarray) -> Image.Image:
        """One supersampled field of the instrument, boxed down to the tile a composite wants."""
        return material.to_image(*self._boxed(np.broadcast_to(rgb, (*alpha.shape, 3)), alpha))

    def _dial_light(self) -> np.ndarray:
        """How much of the lamp reaches a mark printed on the face, over an instrument's tile.

        The bezel's shadow and the crystal's reflection, and nothing else. A mark on a dial is
        ink on the plate: the ring standing over the plate shades it where the shadow falls, and
        the reflection lying on the crystal above it lifts it back where the two cross. So the
        arc is neither flat nor a single ramp - it dips under the ring's shadow at the top of the
        sweep and comes back up through the reflection, which is what a photograph of a gauge
        does and what a set of marks painted at one value never does.

        Multiplied on, never added. Adding light closes the gap between the dim track and the
        lit fill until the value cannot be read off the ring at all - measured on the reference
        as a fill that bottoms out *below* the empty track it is drawn over - where a multiply
        leaves that ratio untouched at every angle and moves only the light they are both under.
        Floored and capped for the same reason: a reading has to read.

        One array per window size, held in :data:`_DIAL_LIGHT`, because nothing in it moves.
        """
        span, r = self.dial_span, self.btn_r
        key = (span, r)
        got = _DIAL_LIGHT.get(key)
        if got is None:
            r_in = r * DIAL_BEZEL_IN
            lx, ly = material.lamp_2d()
            ys, xs = (a.astype(np.float32) for a in np.mgrid[-span:span + 1, -span:span + 1])
            dist = np.hypot(xs, ys)
            facing = (xs * lx + ys * ly) / np.maximum(dist, 1e-6)
            depth = np.clip(1.0 - (r_in - dist) / max(DIAL_RECESS * r, 1e-3), 0.0, 1.0)
            shade = depth * depth * (DIAL_RECESS_ALL
                                     + (1.0 - DIAL_RECESS_ALL) * np.clip(facing, 0.0, 1.0))
            reach = max(1.0, DIAL_GLARE_REACH * r_in)
            spot = np.exp(-((xs - lx * DIAL_GLARE_AT * r_in) ** 2
                            + (ys - ly * DIAL_GLARE_AT * r_in) ** 2) / (reach * reach))
            sheen = np.clip(DIAL_PRINT_TILT * (0.5 + 0.5 * facing)
                            + DIAL_PRINT_GLARE * spot, 0.0, 1.0)
            lit = ((DIAL_PRINT_DARK + (DIAL_PRINT_BRIGHT - DIAL_PRINT_DARK) * sheen)
                   * (1.0 - DIAL_PRINT_SHADE * shade))
            got = _DIAL_LIGHT[key] = lit.astype(np.float32)[..., None]
        return got

    def _printed(self, tile: Image.Image) -> Image.Image:
        """*tile*'s colour put under :meth:`_dial_light`, its coverage left alone.

        Colour only: dimming a mark's alpha would eat into the face behind it, and what is being
        modelled is a mark lit less brightly rather than a mark less there.
        """
        lit = np.asarray(tile).astype(np.float32)
        lit[:, :, :3] *= self._dial_light()
        return Image.fromarray(np.clip(lit, 0, 255).astype(np.uint8), "RGBA")

    def _face(self, dist: np.ndarray, facing: np.ndarray, disc: np.ndarray, r: float
              ) -> tuple[np.ndarray, np.ndarray]:
        """The recess: the well the switch left, shaded by the bezel standing over it.

        The floor is nearly, and deliberately not quite, opaque. It used to keep the well's old
        opacity, on the argument that the rail runs under it and always had; what that actually
        drew was the bracket's straight diagonal edge stepping ten levels across the middle of
        the face, and a lens with a crack across it is a broken instrument. It is still a
        modulation and not a fill - the camera is under everything on this panel - just a much
        heavier one, and what it buys is a face dark and even enough for the arc to glow on.

        Two shadows, not one. The bezel's own falls on the side towards the lamp, and under it
        runs a band all the way round that has nothing to do with the lamp: it is where the face
        meets the ring standing on it, and a part in contact with another part is dark at the
        join whichever way the light comes from. The far wall of the recess is the one place the
        lamp reaches down into, so it alone is lifted towards steel.
        """
        r_in = r * DIAL_BEZEL_IN
        floor = DIAL_FACE_A
        depth = np.clip(1.0 - (r_in - dist) / max(DIAL_RECESS * r, 1e-3), 0.0, 1.0)
        shade = depth * depth * (DIAL_RECESS_ALL
                                 + (1.0 - DIAL_RECESS_ALL) * np.clip(facing, 0.0, 1.0))
        alpha = floor + (DIAL_RECESS_A - floor) * shade
        wall = np.clip(1.0 - (r_in - dist) / max(DIAL_WALL * self.scale, 0.5), 0.0, 1.0)
        wall = wall * np.clip(-facing, 0.0, 1.0) * (dist <= r_in)
        screen = np.asarray(SCREEN, np.float32)
        rgb = screen + (np.asarray(material.STEEL, np.float32) - screen) * wall[..., None]
        return rgb, alpha * disc

    def _dial_glass(self, dist: np.ndarray, facing: np.ndarray, reach: int, r: float
               ) -> tuple[np.ndarray, np.ndarray]:
        """The dome over the face: a veil, the lamp given back from one place, and a rim.

        The terminal's glare with its numbers made this small: the lamp's reflection sits up
        towards the lamp on a dome, and no streak, because a dome is not a long face. It is one
        reflection and it is small - a coin of glass under a bench lamp gives back a spot the
        size of a pinhead, and a soft wash over half the face is a smear on the lens rather than
        a light in the room. The edge of the dome turns steep enough to give the room back all
        the way round, brightly where it faces the lamp and barely anywhere else, so what the rim
        adds is a thin reflection and not a second highlight arguing with the first.

        Everything here is the tube's own white, thin: a glass that whitens the scale under it is
        a fogged glass, and the tests that read the scale would agree.
        """
        r_in = r * DIAL_BEZEL_IN
        lx, ly = material.lamp_2d()
        size = (2 * reach + 1) * DIAL_SS
        # material.glare addresses samples by index; the instrument's centre is at index
        # (reach + 0.5) * DIAL_SS - 0.5, and the lamp's reflection a fraction of the glass out.
        middle = (reach + 0.5) * DIAL_SS - 0.5
        lamp = (middle + lx * DIAL_GLARE_AT * r_in * DIAL_SS,
                middle + ly * DIAL_GLARE_AT * r_in * DIAL_SS)
        lit = material.glare(size, size, lamp, DIAL_GLARE_REACH * r_in * DIAL_SS, ambient=0.0)
        # The rim reflection is brightest where the glass disappears under the bezel's groove,
        # not at the glass's own edge: the last couple of pixels of it are in shadow under the
        # ring, and a reflection drawn there is one nobody can see.
        rim = r_in - max(1.0, DIAL_BEZEL_SEAM * self.scale)
        edge = np.clip(1.0 - (rim - dist) / max(DIAL_GLASS_EDGE * self.scale, 0.5), 0.0, 1.0)
        edge = edge * (DIAL_GLASS_EDGE_ROOM
                       + (1.0 - DIAL_GLASS_EDGE_ROOM) * np.clip(facing, 0.0, 1.0) ** 2)
        # ...and the pane itself is not one brightness: it leans towards the lamp, so the half of
        # it turned that way gives back more of the room than the half turned from it. Ten levels
        # across the face, which is what stops a dark disc reading as a hole.
        slope = np.clip(0.5 + 0.5 * facing, 0.0, 1.0)
        cover = np.clip(0.5 - (dist - r_in) * DIAL_SS, 0.0, 1.0)
        alpha = (DIAL_GLASS_A + DIAL_GLASS_TILT * slope
                 + DIAL_GLARE_A * lit + DIAL_GLASS_EDGE_A * edge) * cover
        return np.asarray(WHITE, np.float32), alpha

    def _bezel(self, dist: np.ndarray, ux: np.ndarray, uy: np.ndarray, r: float, seed: int,
               polish: float, phosphor: tuple[int, int, int] = GREEN
               ) -> tuple[np.ndarray, np.ndarray]:
        """The steel ring: a rolled outer edge that takes the one highlight, a flat land, a
        chamfer cut down to the glass, and the two dark lines that say how deep it all is.

        Four bands across five pixels, and the point of them is that a surface has a SECTION.
        Along one radius on the lit side the lamp writes a groove at 10, the chamfer's shoulder
        at 95, a land that holds flat, a blown crown line past 240 and a flank falling back to
        the contact shadow - five or six turns, which is what a machined part does and what a
        radial gradient cannot do at all.

        The section alone is not enough, and that is what three critics measured next: a flat
        annulus square to the viewer takes the same light at every azimuth, so a ring built out
        of a section and nothing else comes out at one value all the way round - ours read 120
        to 125 in eight octants, a spread of a fifth, where the reference swings four times. So
        what travels round this ring is not the surface's answer to the lamp but WHETHER THE
        LAMP REACHES IT. A five-pixel ring standing three proud of a well in a plate shades its
        own far side; that side sees the plate instead of the room. DIAL_BEZEL_CAST is what is
        left of the lamp there, DIAL_BEZEL_WELL what is left of the room, and between them the
        ring has a lit side and a dark one whichever way the metal happens to be leaning.

        Only the roll is ever lit. The chamfer faces in and down, so on the far side of the ring
        it is turned towards the lamp and would take a second highlight of its own - which is
        precisely what an earlier cut of this drew, two speculars of near-equal strength half a
        turn apart, and what every critic called a printed ring under two lamps. A chamfer is
        cut, not polished, and it stands under the crown: it keeps a thirtieth of the shine and
        loses a third of the lamp to the metal over it. What is left is a bounce - present, so
        the far side is metal and not a hole, capped at a quarter of the crown's own highlight,
        and never mistakable for a light.

        Three strokes, not one band. Outside in: the crown with its one blown line; the lip
        where the crown turns down, shadowed where the crown stands over it and RAKED where the
        lamp comes in over the crystal instead; and the undercut the glass is set into, which is
        three pixels of near black on the lit side and under two of a lifted hairline opposite.
        A shadow of one depth drawn all the way round a ring is a stroke, and a stroke is what
        this was.

        Occlusion moves the AMBIENT and nothing else. What a face can see of the room sets the
        floor it sits on; it does not dim the lamp that reaches it or the highlight it mirrors,
        and material.steel takes a field for exactly that. Scaling the finished colour by it -
        which is what this did - flattens the shadow side's section along with its brightness,
        and a ring whose far half has no crown, land or chamfer left in it is a painted donut at
        any value. What keeps that half out of black is the plate it stands on: the plate is lit
        all the way round, so the skirt of the roll turned down towards it takes a little back.

        And the one thing on the ring that is green: the cut edge of the crystal, showing at the
        top of the undercut and lit by the face inside it. That is what makes the junction a
        REVEAL rather than a drawn black stroke - a critic measured forty to fifty levels of
        green excess over three pixels there on the reference and exactly none on ours. It is
        laid down last, past the clamp that keeps green off steel, and it stops inside the seat.

        A lathe-turned part, so the brushing runs round the ring rather than along a bar. The
        wear on it is BRIGHT ONLY: a scratch takes the finish off and what is underneath catches
        more light than the finish did, never less, and symmetric noise either side of the mean
        is a grain overlay rather than a used part. What it catches is all the light there is
        and not only the lamp's, at the root of it, because bare metal at the bottom of a
        hairline is near enough a mirror: gated on the lamp alone, no mark on this ring survived
        the half of it the lamp does not reach.

        *seed* and *polish* are what stop the two instruments being one sprite stamped twice -
        same lamp, same section, different history and a different amount of finish left on the
        ring. A critic measured our pair as matching to four tenths of a level across the whole
        annulus where the reference's two differ by eighteen; two parts on one panel are two
        parts.
        """
        r_in = r * DIAL_BEZEL_IN
        lx, ly = material.lamp_2d()
        theta = (np.arctan2(uy, ux) - math.atan2(-ly, -lx)) % (2.0 * math.pi)
        along = theta.astype(np.float32) * r
        facing = ux * lx + uy * ly
        # The seat is not a machined fit either: it wanders a fraction of a pixel round the ring,
        # by a different amount on each of the two instruments, which is the cheapest thing on
        # the part that stops the second one being the first one stamped again.
        seam = (max(1.0, DIAL_BEZEL_SEAM * self.scale)
                * (1.0 + DIAL_SEAT_ROVE * material.wear(along, seed + 9)))
        cover = (np.clip(0.5 - (dist - r) * DIAL_SS, 0.0, 1.0)
                 * np.clip(0.5 + (dist - (r_in - seam)) * DIAL_SS, 0.0, 1.0))
        band = max(r - r_in, 1e-3)
        # The roll is narrower where the ring turns into the lamp, so the crown's brightest line
        # sits further out there than it does on the shadow side and travels between the two -
        # three pixels of it on a thirty-six pixel radius. Plus a slow wander with no reason
        # behind it but the part not being perfect, which is what takes the highlight off one row.
        rove = 1.0 - DIAL_BEZEL_ROVE * facing + DIAL_BEZEL_ROVE_N * material.wear(along, seed + 5)
        roll = np.maximum(DIAL_BEZEL_ROLL * band * rove, 0.6)
        cut = max(DIAL_BEZEL_CUT * band, 0.6)
        # How far the surface has turned and which way: out and down on the roll, in and down on
        # the chamfer, and the land in between holding the small tilt a machined flat keeps.
        out = np.clip(1.0 - (r - dist) / roll, 0.0, 1.0) ** DIAL_BEZEL_ROLL_ROUND
        into = np.clip(1.0 - (dist - r_in) / cut, 0.0, 1.0)
        lean = (DIAL_BEZEL_LAND + (DIAL_BEZEL_ROLL_TILT - DIAL_BEZEL_LAND) * out
                - (DIAL_BEZEL_CUT_TILT + DIAL_BEZEL_LAND) * into)
        tilt = np.clip(np.abs(lean), 0.0, 0.995)
        sign = np.sign(lean)
        nx, ny = sign * ux * tilt, sign * uy * tilt
        nz = np.sqrt(np.maximum(1.0 - tilt * tilt, 0.0))
        diffuse, spec = material.shade(nx, ny, nz)
        # The room, so the face turned from the lamp is dark metal and not a hole; the polish,
        # which lives on the roll and not on the land; and the crown standing over the chamfer,
        # which is what keeps the chamfer's answer to the lamp down to a bounce.
        shut = 1.0 - DIAL_BEZEL_CUT_SHUT * into
        # Which side of the ring this is, 1 turned to the lamp and 0 turned from it. Everything
        # that travels round the ring hangs off it - it is the difference between a rim and a
        # stroke - and it is a fact about the ASSEMBLY, not about the surface: a five-pixel ring
        # standing three proud of a plate shades its own far side, whichever way the metal there
        # happens to be leaning.
        key = np.clip(0.5 + 0.5 * facing, 0.0, 1.0)
        cast = key ** DIAL_BEZEL_WRAP
        lamp_on = DIAL_BEZEL_CAST + (1.0 - DIAL_BEZEL_CAST) * cast
        diffuse = (DIAL_BEZEL_FILL + (1.0 - DIAL_BEZEL_FILL) * diffuse) * shut * lamp_on
        spec = spec ** DIAL_BEZEL_SHINE * (1.0 + (DIAL_BEZEL_POLISH - 1.0) * out)
        spec = spec * (1.0 - (1.0 - DIAL_BEZEL_CUT_GLOSS) * into)
        spec = spec * (1.0 + DIAL_WEAR * np.clip(material.wear(along, seed), 0.0, 1.0))
        # A surface in the crown's shadow cannot mirror a lamp it cannot see. Without this the
        # chamfer on the far side answers the lamp at seven tenths of the crown's own highlight
        # and the ring reads as a part lit from both sides at once.
        spec = spec * lamp_on
        brushing = DIAL_BEZEL_GRAIN * material.grain(dist, along, seed)
        # What a slope on a ring sees instead of the room is the bracket it is bolted to, so the
        # far one goes below the panel's own ambient - and so does the whole far side of the
        # ring, which is bedded in a plate and sees it rather than the room. Without these the
        # ring has no black in it anywhere.
        #
        # It is the AMBIENT and only the ambient: what a surface can SEE moves the floor it sits
        # on, and never the lamp's own share or the highlight (see material.steel). Scaling the
        # whole colour instead - which is what this did - takes the shadow side's section away
        # with its brightness, and a ring whose far half holds no crown, no land and no chamfer
        # is a painted donut whatever value it is drawn at.
        seen = np.clip(nx * lx + ny * ly + 1.0, 0.0, 1.0)
        edge = np.clip(1.0 - (r - dist) / max(DIAL_BEZEL_EDGE * self.scale, 0.5), 0.0, 1.0)
        room = (material.AMBIENT
                * (DIAL_BEZEL_SHUT + (1.0 - DIAL_BEZEL_SHUT) * seen)
                * (DIAL_BEZEL_WELL + (1.0 - DIAL_BEZEL_WELL) * cast)
                # ...plus what comes back up off the plate onto the skirt of the roll, which is
                # the only part of the ring turned down far enough to see it. A bounce and not a
                # lamp, and deliberately the SAME all the way round: the plate is an annulus of
                # dull steel round the whole ring, so what it hands back has no direction in it
                # at all. On the lit side the crown's own highlight swamps it; on the dark side
                # it is the floor that keeps the metal metal. Anything with a bearing here is a
                # second light, and this part has been rebuilt once already for having one.
                #
                # Not on the last pixel of it: that one is the contact line, shut against the
                # plate it is standing on, and it sees less of the room than anything else on
                # the part rather than more. Which is what makes the bounce read - it comes up
                # to a defined lighter line and then stops, instead of washing out over the
                # silhouette into the shadow the instrument drops.
                + DIAL_BEZEL_BOUNCE * (1.0 - edge)
                * np.clip((out - DIAL_BEZEL_BOUNCE_AT) / (1.0 - DIAL_BEZEL_BOUNCE_AT), 0.0, 1.0))
        rgb = material.steel(diffuse, spec, brushing, ambient=room,
                             colour=mix(material.STEEL, material.STEEL_LIT, polish))
        # ...and the one line where the crown has actually blown out. material.steel stops at
        # STEEL_SPEC, which is as bright as the metal itself can be; the lamp in it is brighter
        # than that, and a highlight that never reaches the top of the range is a matte part.
        blown = np.clip(spec, 0.0, 1.0) ** DIAL_BEZEL_RIDGE
        spark = np.minimum(np.asarray(material.STEEL_SPEC, np.float32) * DIAL_SPARK, 255.0)
        rgb = rgb + (spark - rgb) * np.clip(DIAL_BEZEL_BLOWN * blown, 0.0, 1.0)[..., None]
        lip = np.clip(1.0 - (dist - r_in) / max(DIAL_BEZEL_LIP * self.scale, 0.5), 0.0, 1.0)
        rolled = DIAL_BEZEL_EDGE_DARK - (DIAL_BEZEL_EDGE_DARK - DIAL_BEZEL_EDGE_LIT) * key
        # The arc of the ring actually turned to take the light that comes over the crystal -
        # not simply "the other half", which spreads a bounce over a hundred and eighty degrees
        # and reads as a lamp of its own.
        raked = np.clip(-facing, 0.0, 1.0) ** DIAL_BEZEL_RAKE_WRAP
        rgb = rgb * ((1.0 - rolled * edge)
                     * (1.0 - DIAL_BEZEL_LIP_DARK * key * lip)
                     * (1.0 + DIAL_BEZEL_LIP_RAKE * raked * lip))[..., None]
        # ...and the undercut the glass is set down into, which is the innermost of the ring's
        # three strokes and the one a critic measured as two pixels of one value all the way
        # round. It is a shadow: three pixels of it and near black where the crown stands
        # between it and the lamp, and less than two of a floor lifted four times as high where
        # the light reaches over the crystal and rakes down the wall.
        reveal = np.maximum(seam * (DIAL_SEAM_OPEN + (DIAL_SEAM_SHUT - DIAL_SEAM_OPEN) * key),
                            0.5)
        groove = np.clip((r_in - dist) / reveal, 0.0, 1.0)
        floor = DIAL_SEAM * (1.0 + (DIAL_SEAM_RAKE - 1.0) * raked)
        ink = np.asarray(material.STEEL_DARK, np.float32) * floor[..., None]
        rgb = rgb * (1.0 - groove)[..., None] + ink * groove[..., None]
        # ...and the hairlines, under the same lamp as the rest of it. A scratch takes the finish
        # off and what is underneath catches more light than the finish did - more of the light
        # that is THERE. Added flat, they came out as bright specks lying in the ring's own
        # shadow, which is a decal of a scratch rather than a scratch.
        #
        # ALL the light that is there, which is what this had wrong: gated on the lamp alone, no
        # mark on the ring survived the half of it the lamp does not reach, and a critic measured
        # our high-frequency residual at half the reference's and called the shadow side smeared.
        # A scratch in a part standing in a room is lit by the room, and it is fresh metal at the
        # bottom of it - near enough a mirror beside the brushing round it, so it hands back a
        # far larger share of what little light there is than the finish does. The root is that:
        # it barely moves a hairline lying in the crown's own highlight and doubles the one lying
        # in the shadow, which is where a used part shows its history and ours showed none.
        drag = (self._dial_scratches(dist.shape[0], seed) * (1.0 - groove) * (1.0 - edge)
                * np.sqrt(room + (1.0 - material.AMBIENT) * diffuse))
        rgb = rgb + drag[..., None]
        # Green is the phosphor's and the glass's. Whatever the panel's steel is tuned to, no
        # pixel of an instrument's ring leaves here carrying more of it than a metal may.
        rgb[..., 1] = np.minimum(rgb[..., 1],
                                 0.5 * (rgb[..., 0] + rgb[..., 2]) + DIAL_GREENBIAS)
        # ...and then the one green thing on the ring, which is not the ring's colour at all: the
        # rim of the crystal, edge-lit by the face under it. A crystal is a light pipe - the dial
        # emits, the pane carries it sideways, and it comes out where the glass is cut, which is
        # the sliver of it still showing at the bottom of the undercut. So it is the LAST thing
        # laid down, past the clamp that keeps green off the steel, and it is the same all the
        # way round because a phosphor has no bearing: the shadow side gets it too.
        #
        # This is the reveal. A dark stroke round a lit face is a drawn line; a groove with the
        # face's own light in the bottom of it is a groove, and it is the one cue a critic found
        # missing outright - forty to fifty levels of green excess over three pixels on the
        # reference, exactly none on ours past r=28.5.
        #
        # A LINE and not a fill, and that is the whole of the tuning: light comes out of the
        # crystal where the crystal is CUT, which is one edge at the top of the undercut, and
        # filling the seat with it puts the brightest thing on the shadow side of the ring in
        # the groove - so the section runs face, groove, crown all downhill and the reveal
        # disappears again from the other direction. Measured from the seat's inner edge and
        # sized off the seat, so it cannot reach the metal wherever the seam has wandered to.
        leak = np.clip(1.0 - (dist - (r_in - seam)) / np.maximum(DIAL_LEAK * seam, 0.5), 0.0, 1.0)
        rgb = rgb + np.asarray(phosphor, np.float32) * (DIAL_LEAK_A * leak * leak)[..., None]
        return rgb, cover

    def _dial_scratches(self, size: int, seed: int) -> np.ndarray:
        """Hairlines dragged across one instrument's metal, in levels to be added on.

        Bright only, and that is the finding rather than a preference: on the reference, wear
        shows up as bright outliers against the local mean and hardly any dark ones, where a
        symmetric noise field - which is all the ring carried before - lands the same number
        either side and reads as a grain overlay laid over the part rather than as damage done
        to it.
        """
        drag = material.scratches(size, size, DIAL_BEZEL_SCRATCH, (-1.0, 0.45), seed,
                                  spread=52.0, length=(6.0 * DIAL_SS, 22.0 * DIAL_SS))
        return drag * (DIAL_BEZEL_SCRATCH_A * 255.0)

    def _track_at(self, deg: float) -> tuple[float, float]:
        """Where the printed sweep runs at *deg*, and how heavy it is there.

        Not a true circle and not one constant weight. A scale is printed on a plate, and a
        stroke laid down by a machine that has been in service wanders a pixel either way and
        thickens where it was laid on. Two harmonics rather than one, so it is a wander and not
        an ellipse, and the same function for the empty track and the filled arc over it - one
        printed mark, drawn twice, so the fill sits exactly on the track it is filling.
        """
        turn = math.radians(deg)
        wob = (math.sin(turn * 2.3 + 0.7) + 0.6 * math.sin(turn * 5.1 + 2.4)) / 1.6
        weight = 1.0 + 0.06 * math.sin(turn * 3.7 - 1.1)
        return self.btn_r * (DIAL_TRACK + DIAL_TRACK_ROVE * wob), weight

    def _sweep(self, t: ImageDraw.ImageDraw, span: int, lo: float, hi: float,
               colour: tuple[int, int, int]) -> None:
        """A stretch of the printed scale from *lo* to *hi* degrees, stamped rather than struck.

        PIL's ``arc`` is a perfect annulus of one radius and one width, which is the one thing
        nothing printed ever is; a critic measured the peak radius of our sweep as holding to
        half a pixel round the whole ring against the reference's pixel and a half. Laying it
        down as overlapping dots costs a few dozen ellipses in a tile that is built once and lets
        the radius and the weight both breathe along the run.
        """
        steps = max(1, round(abs(hi - lo) / DIAL_ARC_STEP))
        ink = linear(colour)
        for i in range(steps + 1):
            deg = lo + (hi - lo) * i / steps
            rad, weight = self._track_at(deg)
            turn = math.radians(deg)
            cx, cy = at(span + rad * math.cos(turn)), at(span + rad * math.sin(turn))
            half = wide(self._dial_stroke * weight) / 2.0
            t.ellipse([cx - half, cy - half, cx + half, cy + half], fill=ink)

    def _paint_scale(self, t: ImageDraw.ImageDraw, span: int, name: str,
                     away: bool = False) -> None:
        """The graduated arc a pointer is read against, inside the tile *name* is being drawn in.

        The knob's is one dim track, because a volume has no regions - it is loud where you put
        it. The gauge's is the board's own three: green until the clock starts being capped, amber
        to where it is capped in earnest, red past that. Those two breaks are
        :data:`~cyclops.stats.WARN_C` and :data:`~cyclops.stats.HOT_C` put through the very scale
        the admin page's bar uses, so the panel and the page cannot drift apart by hand.

        *away* is the knob's scale in the handover hue, struck a second time over the green one
        baked into the chrome - see :meth:`_hand`. Only the knob has one: a board temperature is
        true whoever is holding his voice, and a gauge that changed colour with it would be
        saying something it does not know.
        """
        mid, dim = (BLUE_MID, BLUE_DIM) if away else (GREEN_MID, GREEN_DIM)
        if name == HEAT:
            edges = (0.0, temp_percent(WARN_C) / 100.0, temp_percent(HOT_C) / 100.0, 1.0)
            for (lo, hi), colour in zip(
                zip(edges, edges[1:], strict=False), (GREEN_MID, AMBER, RED), strict=True
            ):
                self._sweep(t, span, self._dial_angle(lo), self._dial_angle(hi),
                            mix(SCREEN, colour, 0.85))
        else:
            self._sweep(t, span, DIAL_FROM, DIAL_FROM + DIAL_SWEEP,
                        mix(SCREEN, dim, DIAL_OFF))
        r = self.btn_r
        for i in range(DIAL_TICKS):
            angle = math.radians(self._dial_angle(i / (DIAL_TICKS - 1)))
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            t.line(
                [at(span + r * DIAL_TICK_IN * cos_a), at(span + r * DIAL_TICK_IN * sin_a),
                 at(span + r * DIAL_TICK_OUT * cos_a), at(span + r * DIAL_TICK_OUT * sin_a)],
                # Full GREEN_MID rather than a mix down towards the screen: a graduation here is
                # one pixel wide and four long, and stirred any further into the black it went
                # missing from a pace away - which is the distance this panel is read from.
                fill=linear(mid), width=round(wide(max(1.0, DIAL_TICK_W * self.scale))),
            )

    def _hand(
        self,
        name: str,
        value: float | None,
        colour: tuple[int, int, int],
        speaker: tuple[int, int, int] | None = None,
        reading: str | None = None,
        away: bool = False,
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

        The pointer stands off the face, so it drops a shadow onto it - the same soft shadow
        every raised part on this panel drops, from :func:`material.cast`, baked into the tile
        under the crisp pointer. The speaker mark is printed on the face and drops none, which is
        why it is drawn into the tile after the shadow and not before it.

        *reading* is the gauge's number, which rides here for the same reason the speaker does:
        it belongs to the gap under the hub, it changes only when the reading does, and a tile
        that is already being cached is the cheapest place on this panel to put anything.
        """
        span, r = self.dial_span, self.btn_r
        lx, ly = material.lamp_2d()

        def paint(t: ImageDraw.ImageDraw) -> None:
            def hub(rad: float, dx: float = 0.0, dy: float = 0.0) -> list[float]:
                reach = at(rad)
                cx, cy = at(span + dx), at(span + dy)
                return [cx - reach, cy - reach, cx + reach, cy + reach]

            if value is not None:
                if name == VOLUME:
                    # What the knob has been turned past, lit over the dim track underneath it -
                    # down the same wandering line, so the fill lies on the track and not beside
                    # it. Nothing is drawn at all at nothing: a stamp at the foot of an empty
                    # sweep is a knob claiming a level it has not been turned to.
                    if value > 0.0:
                        self._sweep(t, span, DIAL_FROM, self._dial_angle(value), colour)
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
            # The hub is a cap, and a cap is domed: one highlight pushed towards the lamp, inside
            # a dark rim. Its colour is the pointer's own pushed a little towards the tube's
            # white, not the white itself - the hub is phosphor like the rest of the hand.
            cap = colour if value is not None else (BLUE_DIM if away else GREEN_DIM)
            t.ellipse(hub(r * DIAL_HUB), fill=linear(cap))
            off = r * DIAL_HUB * DIAL_CAP_OFF
            t.ellipse(hub(r * DIAL_HUB * DIAL_CAP_R, lx * off, ly * off),
                      fill=linear(mix(cap, WHITE, DIAL_CAP)))
            # ...and the seat it stands in, which is a shadow and not an outline: dark where the
            # cap's turned-down edge leans out of the lamp, all but gone where it leans in.
            seat, wall = hub(r * DIAL_HUB), round(wide(1))
            bearing = math.degrees(math.atan2(ly, lx))
            t.arc(seat, bearing + DIAL_CAP_SEAT, bearing - DIAL_CAP_SEAT + 360.0,
                  fill=linear(SCREEN), width=wall)
            t.arc(seat, bearing - DIAL_CAP_SEAT, bearing + DIAL_CAP_SEAT,
                  fill=linear(mix(SCREEN, cap, DIAL_CAP_SEAT_LIT)), width=wall)

        hand = self._printed(smoothed(2 * span + 1, paint))
        cover = np.asarray(hand)[:, :, 3].astype(np.float32) / 255.0
        lift = max(0.5, DIAL_HAND_LIFT * self.scale)
        shadow = material.cast(cover, lift) * DIAL_HAND_SHADOW
        tile = material.to_image(np.zeros((*cover.shape, 3), np.float32), shadow)
        if speaker is not None:
            tile.alpha_composite(self._printed(smoothed(
                2 * span + 1, lambda t: self._paint_speaker(t, span, speaker, away))))
        tile.alpha_composite(self._phosphor_bloom(hand, span, r * DIAL_HUB + self.scale))
        tile.alpha_composite(hand)
        if reading is not None:
            self._print_reading(tile, reading, colour)
        # Hardware is opaque and hardware is topmost. This tile goes on OVER the bezel, because
        # it moves and the ring is baked into the chrome; so anything in it that reached past
        # the glass would be a mark printed on the face drawn across the metal holding the face
        # in. The reading is what found it - two pixels of the bottom-left of "60" were on the
        # ring - and the mask is what stops the next mark finding it again.
        clipped = np.asarray(tile).astype(np.float32)
        clipped[:, :, 3] *= self._dial_aperture()
        return Image.fromarray(np.rint(clipped).astype(np.uint8), "RGBA")

    def _dial_aperture(self) -> np.ndarray:
        """The glass, as coverage: how much of a mark printed on the face is not under the ring.

        The bezel's seat wanders a third of its width round each instrument (DIAL_SEAT_ROVE), so
        what a moving mark may reach is the innermost the ring's inner edge ever comes on either
        of them - not its nominal radius. One array per window size, because nothing in it moves.
        """
        span, r = self.dial_span, self.btn_r
        key = (span, r)
        got = _DIAL_APERTURE.get(key)
        if got is None:
            reach = (r * DIAL_BEZEL_IN
                     - max(1.0, DIAL_BEZEL_SEAM * self.scale) * (1.0 + DIAL_SEAT_ROVE))
            ys, xs = (a.astype(np.float32) for a in np.mgrid[-span:span + 1, -span:span + 1])
            got = _DIAL_APERTURE[key] = np.clip(reach - np.hypot(xs, ys) + 0.5, 0.0, 1.0)
        return got

    def _print_reading(self, tile: Image.Image, word: str, colour: tuple[int, int, int]) -> None:
        """The gauge's number, in the gap under the hub of the tile it belongs to.

        Flat rather than through :func:`smoothed`, because FreeType hands glyphs over
        anti-aliased already and supersampling them is a fourfold tile for nothing. In the
        cached tile rather than on the frame, though, which is where it used to go: it is drawn
        with its own skirt now, and PIL renders a stroked glyph by dilating the mask - a sixth of
        a millisecond every frame, for a number that changes once every five seconds.

        The skirt is what every other lit mark on this face gets from :meth:`_phosphor_bloom`:
        phosphor scatters in the glass over it, and a number that arrives with a hard edge is the
        one mark on the instrument that reads as ink rather than as light.

        And struck twice. What carries a number across a workshop is the WEIGHT of its strokes,
        and weight is what the gap under the hub has no room to buy with height: a face big
        enough for the reference's three-pixel stroke stands two pixels out on the ring, and
        PIL's own stroke_width grows the glyph as much at the top - where the pointer swings -
        as at the sides, where there is room. A second strike grows it in neither direction. It
        takes the half-covered pixels along each stroke to opaque instead, which is where a
        twelve-pixel glyph keeps most of its ink: mean 110 to 125 and a stroke run of 2.5 px,
        against the reference's 126 and 3.0.
        """
        span, drop = self.dial_span, round(self.btn_r * DIAL_LABEL)
        under = max(DIAL_READ_LIGHT, float(self._dial_light()[span + drop, span, 0]))
        ink = tuple(round(c * under) for c in colour)
        size = max(7, round(DIAL_READ_PT * self.scale))
        font = _DIAL_FONT.get(size) or _DIAL_FONT.setdefault(size, _load_font(size))
        word_tile = Image.new("RGBA", tile.size, (0, 0, 0, 0))
        pen = ImageDraw.Draw(word_tile)
        for _ in range(DIAL_READ_STRIKES):
            pen.text((span, span + drop), word, font=font, fill=(*ink, 255), anchor="mm")
        tile.alpha_composite(self._phosphor_bloom(word_tile, span, 0.0))
        tile.alpha_composite(word_tile)

    def _phosphor_bloom(self, tile: Image.Image, span: int, keep_out: float) -> Image.Image:
        """The skirt a lit mark bleeds onto the face under the glass, out of *keep_out* of it.

        Everything on a dial that glows glows the same way the caption does, and for the same
        reason: phosphor scatters in the glass over it, so a mark a pixel wide is read from a
        pace away as a mark with a halo. Without it the arc and the pointer are ink printed on
        the face rather than light coming off it, which is what a critic means by "a sticker".

        The hub is held out of it by *keep_out*. It is the one thing on the hand that is drawn
        with nothing behind it - a dial with no reading still has its hub - and a bloom round a
        parked hub would put light in the middle of an instrument that is admitting it cannot
        read. Blooming the tile and cutting the middle out costs one array and one blur, which
        is why it is done here rather than by painting the marks a second time.
        """
        ys, xs = np.mgrid[-span:span + 1, -span:span + 1]
        outside = np.clip(np.hypot(ys, xs) - keep_out, 0.0, 1.0)
        lit = np.asarray(tile).copy()
        lit[:, :, 3] = (lit[:, :, 3] * outside).astype(np.uint8)
        glow = Image.fromarray(lit, "RGBA").filter(
            ImageFilter.GaussianBlur(max(1.0, DIAL_BLOOM_R * self.scale)))
        glow.putalpha(glow.getchannel("A").point(lambda a: round(a * DIAL_BLOOM_A)))
        return glow

    def _knob(self, level: int | None, turning: bool, away: bool = False) -> Image.Image:
        """The volume pointer at *level*, white while a finger is on it. Cached per appearance.

        The speaker rides in the same tile, because it is part of the same still picture and a
        tile that is already being cached is the cheapest place on this panel to put a shape.

        White is the whole of what the knob does under a finger, and it says the right thing at
        the right moment twice over: it and the column come up together on the touch, and once
        the finger is on the track it is the pointer following it that says the two are one
        control rather than two.

        *away* is a companion holding his voice, and it beats a finger, because while it is up
        there is no finger to beat: the kiosk stops taking the knob's presses at all (see
        :meth:`cyclops.kiosk.Kiosk._on_mouse`). The knob is not a control that has gone quiet,
        it is not a control - it is an icon of where his voice went, and every lit thing on it
        goes over to the one hue at once. Hand, arc, hub, graduations, track and the mark in the
        gap: a single mark changing was the version that was too easy to miss.

        *away* is in the key rather than swapped into a tile after the fact, which is the whole
        of what it costs: the appearance is two knobs instead of one per level and turning, and
        the pair is built once each and then handed back for as long as nothing moves.
        """
        key = (level, turning, away)
        tile = self._knobs.get(key)
        if tile is None:
            colour = BLUE if away else (WHITE if turning else GREEN)
            tile = self._knobs[key] = self._hand(
                VOLUME,
                None if level is None else level / 100.0,
                colour,
                speaker=BLUE if away else (
                    GREEN_DIM if level is None else (WHITE if turning else GREEN_MID)),
                away=away,
            )
        return tile

    def _needle(self, temp_c: float | None, lit: bool) -> Image.Image:
        """The heat needle for *temp_c*, and the degrees it is reading. Cached on both.

        The word is in the key as well as the percent, and not because it is tidy: the scale runs
        a degree and a bit to the percent, so two temperatures that round to different degrees
        can land on the same point of the sweep, and a tile keyed on the sweep alone would hand
        one of them the other's number.
        """
        percent = temp_percent(temp_c)
        band = temp_band(temp_c)
        word = "--" if temp_c is None else f"{round(temp_c)}°"
        key = (percent, band, lit, word)
        tile = self._needles.get(key)
        if tile is None:
            colour = WHITE if lit else HEAT_INK.get(band, GREEN_DIM)
            tile = self._needles[key] = self._hand(
                HEAT, None if percent is None else percent / 100.0, colour, reading=word
            )
        return tile

    def _draw_hands(
        self,
        layer: Image.Image,
        d: ImageDraw.ImageDraw,
        volume: int | None,
        temp_c: float | None,
        pressed: str | None,
        handed_over: bool = False,
    ) -> None:
        """Both pointers, each with whatever it prints in the gap under its hub.

        The gap is the quarter of the sweep neither dial uses, which is where a knob's own scale
        has always left room for a label. The knob keeps a speaker there, because its number is on
        the column a drag opens and one reading in two places is one of them being read twice. The
        gauge shows its degrees always: that is the whole of what a gauge is for, and a needle
        without a number is a mood ring.

        Two composites and nothing else. The gauge's number used to go on here, flat, on the
        argument that FreeType hands glyphs over anti-aliased already - which is true, and stayed
        true when the number was given the skirt every other lit mark on the face has. What was
        no longer true was the cost: PIL draws a stroked glyph by dilating its mask, and that is
        a sixth of a millisecond a frame for a word that changes once every five seconds. It is
        in the needle's own cached tile now; *d* is what :meth:`render` hands every painter, and
        this one has nothing left to draw flat.
        """
        if handed_over:
            # The knob's whole instrument again, in the other hue, over the green one in the
            # chrome. Cached, so a claim coming and going costs one composite of a 91-pixel
            # square either way and never a rebuild - which matters, because a backgrounded
            # companion tab drops and retakes the claim about once a minute.
            self._stamp_instrument(layer, VOLUME, *self._away_dial())
        turning = pressed == VOLUME
        for name, tile in ((VOLUME, self._knob(volume, turning, handed_over)),
                           (HEAT, self._needle(temp_c, pressed == HEAT))):
            cx, cy = (round(v) for v in self.switches[name])
            layer.alpha_composite(tile, (cx - self.dial_span, cy - self.dial_span))

    def _away_dial(self) -> tuple[Image.Image, int]:
        """The knob's instrument in the handover hue, built the first time it is wanted.

        Not in the chrome, and deliberately: the chrome is baked per state and per heat band,
        and putting a fourth key on it would double a cache of full-screen layers to say a thing
        about one instrument in the corner. One tile of nine thousand pixels says it instead.
        """
        if self._away is None:
            self._away = self._instrument_tile(VOLUME, away=True)
        return self._away

    # ------------------------------------------------------------------ the pilot lamp

    def _draw_pilot(self, layer: Image.Image) -> None:
        """The lamp with nothing burning in it, baked into the chrome with the two dials.

        The dead lamp is the one appearance that is true whatever the box is doing, so it is the
        one that belongs here. Every lit appearance is the same build with light in it - see
        :meth:`_pilot_tile`, which this is a call to.
        """
        self._paste_pilot(layer, self._pilot_tile(None, 0))

    def _paste_pilot(self, layer: Image.Image, tile: Image.Image) -> None:
        """One pilot tile onto *layer*, clipped if the window is small enough to push it off."""
        cx, cy = (round(v) for v in self.pilot)
        left, top = cx - self.pilot_reach, cy - self.pilot_reach
        crop = tile.crop((max(0, -left), max(0, -top), tile.width, tile.height))
        layer.alpha_composite(crop, (max(0, left), max(0, top)))

    def _pilot_tile(self, colour: tuple[int, int, int] | None, step: int) -> Image.Image:
        """The whole fitting at one appearance: dead if *colour* is None, else burning.

        **The metal is rebuilt with the light in it rather than having the light laid over it,
        and that is the whole of this method's design.** The first cut of this baked one dead
        fitting and composited a coloured glow on top, which is what a lamp looks like in a
        drawing program and not what one looks like on a bench: a flat layer at half alpha
        replaces half of every pixel under it, so the knurl's bright flank and its dark flank
        converge on the same green and the ring goes soft. Marco called it blurry twice, and
        both times it was this - not the resolution, which was always the dials' own.

        So the lamp's light is an argument to the metal's shading, added in the light's own
        colour the way his collar adds its specular, and the section survives intact: a knurl
        under a green lamp is a green knurl with all of its contrast, not a green disc.

        The cost of that is a build per appearance instead of one build ever, and it is affordable
        for exactly one reason - the table only asks for full brightness or none (see LAMPS, where
        every row is STEADY or BLINK), so this is at most one tile per colour. *step* is kept in
        the key against the day a row wants to fade, which would make it PILOT_STEPS per colour
        and still small.
        """
        key = (self.width, self.height, colour, step)
        tile = _PILOT_TILES.get(key)
        if tile is not None:
            return tile
        reach, r = self.pilot_reach, self.pilot_r
        r_in = r * PILOT_BEZEL_IN
        dist, ux, uy = self._dial_grid(reach)
        level = 0.0 if colour is None else max(0.0, min(1.0, step / PILOT_STEPS))
        lamp = np.asarray(colour or (0, 0, 0), np.float32)
        # How much of the lamp's own light reaches each sample. One radial falloff off the edge
        # of the glass, and analytic rather than a blurred mask: a Gaussian over a disc has to be
        # computed at the tile's own resolution and then quantised into it, and both of those
        # cost sharpness that this cannot. It is also simply the honest shape - a disc of lit
        # glass throwing light onto the metal ringing it falls off with distance from the glass.
        throw = max(1.0, PILOT_THROW * self.scale)
        glow = np.exp(-np.clip((dist - r_in) / throw, 0.0, None) ** 2) * level
        sun = float(self._sunlight(*(round(v) for v in self.pilot), 1, 1)[0, 0])

        tile = Image.new("RGBA", (2 * reach + 1, 2 * reach + 1), (0, 0, 0, 0))
        disc = np.clip(0.5 - (dist - r) * DIAL_SS, 0.0, 1.0)
        _, cover = self._boxed(np.zeros((*disc.shape, 3), np.float32), disc)
        lift = max(1.0, PILOT_LIFT * self.scale)
        # The shadow it drops, and the pool it throws beyond its own edge. The pool goes over the
        # shadow because a lamp lights the plate it is standing on more than its own body shades
        # it; both are outside the fitting, which covers them.
        shade = material.cast(cover, lift) * PILOT_SHADOW
        tile.alpha_composite(material.to_image(np.zeros((*cover.shape, 3), np.float32), shade))
        if level > 0.0:
            pool = self._boxed_field(glow * (dist > r)) * PILOT_ON_PLATE
            tile.alpha_composite(material.to_image(
                np.broadcast_to(lamp, (*pool.shape, 3)), np.clip(pool, 0.0, 1.0)))
        tile.alpha_composite(self._dial_layer(
            *self._pilot_bead(dist, reach, r_in, lamp, level)))
        tile.alpha_composite(self._dial_layer(
            *self._pilot_bezel(dist, ux, uy, r, r_in, sun, lamp, glow)))
        _PILOT_TILES[key] = tile
        return tile

    def _pilot_bead(self, dist: np.ndarray, reach: int, r_in: float, lamp: np.ndarray,
                    level: float) -> tuple[np.ndarray, np.ndarray]:
        """The glass dome: dark with the room in it, plus whatever is burning behind it.

        A low dome and not a hemisphere, which is what Marco picked: the glass stands a little
        proud of the brass and turns away at the rim rather than rolling over. ``PILOT_DOME`` is
        that tilt at the edge, and the cosine of it does two jobs - it darkens the rim, because a
        surface turned away shows less of what is under it, and it is the same field the light
        inside uses to decide how much escapes.

        Dead, it is deliberately not black: a bead of glass over a cold filament is near enough
        the screen's own green-black, where a hole is what it looks like at full alpha with
        nothing in it, and a hole is what this corner already had.

        Lit, the light is added to that rather than drawn over it, so the dome's own shading and
        the room's reflection are both still in the result. The reflection is laid down last for
        that reason: it is the room's light and not the lamp's, so it is the same spot at the
        same brightness whether the thing is burning amber, green or not at all, and a lens whose
        highlight vanished when it lit would stop reading as glass.
        """
        lit = self._pilot_glare(dist, reach, r_in)
        nz = self._pilot_dome(dist, r_in)
        cover = np.clip(0.5 - (dist - r_in) * DIAL_SS, 0.0, 1.0)
        screen = np.asarray(SCREEN, np.float32)
        dead = np.asarray(material.STEEL_DARK, np.float32)
        rgb = screen + (dead - screen) * (1.0 - nz)[..., None]
        if level > 0.0:
            core = np.clip(1.0 - dist / max(PILOT_CORE * r_in, 1e-3), 0.0, 1.0) ** 2
            body = nz ** PILOT_FALL
            emit = np.clip(PILOT_CORE_A * core + PILOT_BODY_A * body, 0.0, 1.0) * level
            rgb = rgb + lamp * emit[..., None]
            # ...and the cut edge of the glass where it goes under the ring, lit from inside all
            # the way round because a phosphor has no bearing. It is what makes the join a reveal
            # rather than a drawn line, the argument the dials' crystals make about their own.
            seat = np.clip(1.0 - (r_in - dist) / max(0.5, PILOT_SEAT * self.pilot_r), 0.0, 1.0)
            rgb = rgb + lamp * (PILOT_RIM_A * level * seat * seat)[..., None]
        rgb = rgb + (np.asarray(WHITE, np.float32) - rgb) * np.clip(
            PILOT_GLARE_A * lit, 0.0, 1.0)[..., None]
        return np.minimum(rgb, 255.0), PILOT_GLASS_A * cover

    @staticmethod
    def _pilot_glare(dist: np.ndarray, reach: int, r_in: float) -> np.ndarray:
        """The bench lamp's own reflection on the bead, 0..1 over the tile.

        ``material.glare`` addresses samples by index, exactly as the dials' glass does: the
        bead's centre is at ``(reach + 0.5) * DIAL_SS - 0.5`` and the reflection a fraction of it
        out towards the lamp. A coin of glass under a bench lamp gives back a spot, not a wash.
        """
        lx, ly = material.lamp_2d()
        size = dist.shape[0]
        middle = (reach + 0.5) * DIAL_SS - 0.5
        at = (middle + lx * PILOT_GLARE_AT * r_in * DIAL_SS,
              middle + ly * PILOT_GLARE_AT * r_in * DIAL_SS)
        return material.glare(size, size, at, max(1.0, PILOT_GLARE_REACH * r_in * DIAL_SS),
                              ambient=0.0)

    @staticmethod
    def _pilot_dome(dist: np.ndarray, r_in: float) -> np.ndarray:
        """How square to the viewer the dome's glass is, 1 in the middle .. 0 edge-on.

        One field, two users - the bead's own shading and the share of the light inside that
        escapes - which is what keeps a lit lamp and a dead one the same piece of glass.
        """
        tilt = np.clip(dist / max(r_in, 1e-3), 0.0, 1.0) * PILOT_DOME
        return np.sqrt(np.maximum(1.0 - tilt * tilt, 0.0))

    def _pilot_bezel(self, dist: np.ndarray, ux: np.ndarray, uy: np.ndarray, r: float,
                     r_in: float, sun: float, lamp: np.ndarray, glow: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
        """The brass ring, turned and graduated - his collar's section at a quarter of the size.

        Marco asked for the brass round the eye rather than something new, and this is that part's
        own vocabulary rather than an imitation of it: a face cut in flats to a crown, a lathe's
        turned groove just inside the roll, a graduation indexed round it, the sheet's own
        hairlines showing pale where they cross, and one blown line where the roll mirrors the
        bench lamp. Same constants where the size allows, so the two read as parts off one lathe -
        see :meth:`_draw_collar`, which this is the small cousin of.

        **The knurl it replaces was the wrong part.** A knurl is what you put on something a hand
        turns; nobody turns a pilot lamp. The graduation is what a machined bezel on an instrument
        panel actually carries, and it is what the only other brass on this panel carries.

        **The pitch is the one thing not copied.** ``_graduation``'s docstring is firm that the
        tool is shared and never the spacing: the collar cuts 120 marks at a radius of about a
        hundred, which is five and a half pixels a mark. Cut at this radius the same count would
        be one and a half, which is not a scale but a texture. PILOT_TICKS is chosen to land on
        the collar's own arc-length pitch instead, so the two rings look indexed by one machine.

        **What the lamp inside does to the brass stays on the inner flank.** This is the part
        Marco sent back twice. Light thrown at the whole ring is not what a pilot lamp does - the
        crown and the face are turned towards the room and cannot see the lens at all, and
        colouring them turned a brass fitting into a coloured disc: measured, the ring's green
        went 74 to 217 and took the section with it, three channels saturating being three
        channels agreeing. Only the flank falling into the seat looks at the glass.
        """
        lx, ly = material.lamp_2d()
        theta = np.arctan2(uy, ux).astype(np.float32)
        along = theta * r  # arc length round the ring, for the brushing and the wear
        roll = max(PILOT_ROLL * (r - r_in), 0.8)
        step = max(PILOT_STEP * self.scale, 0.6)
        face = max(r - roll - r_in, 1e-3)
        # The crown: the face's tilt climbing to the roll, cut in flats rather than faded, which
        # is the difference between a chamfer off a tool and an airbrush.
        crown = PILOT_CROWN * self._facets((dist - r_in) / face, PILOT_FACETS)
        nx, ny, nz = material.roll_normals(np.maximum(r - dist, 0.0), ux, uy, roll, dome=crown)
        diffuse, spec = material.shade(nx, ny, nz)
        # ...and the inner flank going dark into the seat the bead sits in.
        shut = np.clip(1.0 - (dist - r_in) / step, 0.0, 1.0)
        diffuse, spec = diffuse * (1.0 - shut), spec * (1.0 - shut)
        rubbed = material.wear(along, material.SEED + 21)
        spec = spec * (1.0 + PILOT_WEAR * rubbed) * sun ** RAIL_SUN_SPEC
        brushed = material.grain(dist, along, material.SEED + 21) + PILOT_TARNISH * rubbed
        # The metal's own colour under the lamp, and then its highlight in its own colour too:
        # `steel` lays its highlight down in STEEL_SPEC, which on brass is a cool smear.
        rgb = material.steel(diffuse * sun, np.zeros_like(spec), brushed, colour=BRASS)
        rgb = rgb + np.asarray(BRASS_SPEC, np.float32) * (material.SPEC * spec)[..., None]
        # The one mark a lathe leaves that is not a scale: the turned line just inside the roll.
        turned = np.exp(-(((dist - (r - roll - 1.0)) / 0.6) ** 2))
        rgb = rgb * (1.0 - PILOT_GROOVE * turned)[..., None]
        rgb = rgb * self._graduation(dist, theta, r - roll, face,
                                     PILOT_TICKS, PILOT_LONG, PILOT_TICK,
                                     PILOT_TICK_LONG)[..., None]
        # ...and the hairlines, pale where they cross brass, exactly as they are on his collar.
        marks = (self._pilot_marks(dist.shape[0]) * PILOT_SCRATCH)[..., None]
        rgb = rgb * (1.0 - marks) + np.asarray(BRASS_SPEC, np.float32) * marks
        # Held to the ceiling in luminance and never per channel: a clip lands on red first on a
        # warm metal, which leaves green the tallest channel it has - a brass ring with green
        # highlights, which is the one thing brass may not do.
        rgb = self._stilled(rgb)
        # The two things that happen at the edges of the roll, both past what a shaded face may
        # reach. The far edge is occluded by the part it belongs to, so it goes under the room's
        # own light; the near one carries the lamp's own image, which is the lamp's colour and
        # not the metal's.
        facing = ux * lx + uy * ly
        rolled = np.clip(1.0 - (r - dist) / roll, 0.0, 1.0)
        away = np.clip(-facing, 0.0, 1.0) ** 1.5
        rgb = rgb * (1.0 - PILOT_TERMINATOR * away * rolled)[..., None]
        ridge = r - roll * PILOT_BLOWN_AT + PILOT_BLOWN_WANDER * material.wear(
            along, material.SEED + 26)
        hot = (np.exp(-(((dist - ridge) / PILOT_BLOWN_W) ** 2))
               * np.clip(facing, 0.0, 1.0) ** PILOT_BLOWN_ARC)
        rgb = rgb + (np.asarray(BRASS_BLOWN, np.float32) - rgb) * np.clip(
            PILOT_BLOWN * hot, 0.0, 1.0)[..., None]
        # Green is the phosphor's and the glass's: no pixel of a METAL leaves here carrying more
        # of it than a metal may. The pilot's own light goes on after, because a lamp shining on
        # brass is light lying on a metal and not a metal that has turned green.
        rgb[..., 1] = np.minimum(rgb[..., 1],
                                 0.5 * (rgb[..., 0] + rgb[..., 2]) + DIAL_GREENBIAS)
        rgb = np.minimum(rgb + lamp * (PILOT_ON_METAL * shut * glow)[..., None], 255.0)
        cover = (np.clip(0.5 - (dist - r) * DIAL_SS, 0.0, 1.0)
                 * np.clip(0.5 + (dist - r_in) * DIAL_SS, 0.0, 1.0))
        return rgb, cover

    def _pilot_marks(self, size: int) -> np.ndarray:
        """The sheet's hairlines where the lamp is bolted, 0..1, at the tile's own sampling.

        Taken out of :data:`_marks` rather than generated here, which is the whole point of there
        being one sheet: a scratch that runs off the plate and onto the fitting is most of what
        says the two are the same piece of metal, and one rolled locally would stop dead at the
        part's edge. Nearest-neighbour up to the supersampled grid, because a hairline is already
        a pixel wide on the sheet and interpolating it only makes it two.
        """
        cx, cy = (round(v) for v in self.pilot)
        reach = self.pilot_reach
        span = 2 * reach + 1
        patch = np.zeros((span, span), np.float32)
        y0, x0 = max(0, cy - reach), max(0, cx - reach)
        y1, x1 = min(self.height, cy + reach + 1), min(self.width, cx + reach + 1)
        patch[y0 - (cy - reach):y1 - (cy - reach), x0 - (cx - reach):x1 - (cx - reach)] = (
            self._marks[y0:y1, x0:x1])
        return np.repeat(np.repeat(patch, DIAL_SS, 0), DIAL_SS, 1)[:size, :size]

    @staticmethod
    def _boxed_field(field: np.ndarray) -> np.ndarray:
        """One supersampled field averaged down by DIAL_SS. :meth:`_boxed` without the colour."""
        h, w = field.shape
        return field.reshape(h // DIAL_SS, DIAL_SS, w // DIAL_SS, DIAL_SS).mean(axis=(1, 3))

    def _draw_pilot_light(self, layer: Image.Image, state: str, phase: float) -> None:
        """The lamp lit for *state* at *phase*, or nothing at all if the table says it is dark.

        The early return is the whole of why a sleeping panel is exactly as still as it was: at
        IDLE there is no tile, no composite and no pixel touched, so two resting frames stay
        byte-identical without this having to be gated on a predicate somebody would later have
        to remember. A dark row in the table *is* the gate.
        """
        pilot = LAMPS.get(state, LAMPS[IDLE])
        if pilot.colour is None:
            return
        step = round(pilot.level(phase) * PILOT_STEPS)
        if step <= 0:
            return
        self._paste_pilot(layer, self._pilot_tile(pilot.colour, step))

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

    def _paint_speaker(self, t: ImageDraw.ImageDraw, span: int, c: tuple[int, int, int],
                       away: bool = False) -> None:
        """A cone and its throat, in the knob's gap. Says which of the two dials this is.

        The gauge's half of that gap is a number in degrees; a knob's reading is the pointer, so
        what goes here instead is the one mark that says what is being turned. Inside the tile
        rather than on the frame: the cone is two diagonals, and a diagonal drawn flat at eleven
        pixels is the staircase this whole corner was rebuilt to be rid of.

        *away* is a companion holding his voice, and it puts a different mark in the same gap -
        see :meth:`_paint_away`. The mark is the smallest part of what changes: the whole
        instrument goes over to :data:`BLUE` with it, and the kiosk stops taking its presses.
        One mark on a green knob was the version that shipped first, and it was too easy to miss.
        """
        if away:
            self._paint_away(t, span, c)
            return
        r, mid = self._mark_box(span)
        # One silhouette rather than a throat and a cone drawn separately: two shapes that share
        # an edge each own half of the pixels along it, and the shrink out of the tile averages
        # that pair into a seam down the middle of what is supposed to be one solid mark.
        t.polygon(
            [(at(span - r), at(mid - r / 3)), (at(span - r / 3), at(mid - r / 3)),
             (at(span + r * 0.9), at(mid - r)), (at(span + r * 0.9), at(mid + r)),
             (at(span - r / 3), at(mid + r / 3)), (at(span - r), at(mid + r / 3))],
            fill=linear(c),
        )

    def _mark_box(self, span: int) -> tuple[float, float]:
        """How big a mark in the knob's gap may be, and where its middle sits.

        Ten pixels across, at the panel's own scale, and that is not a style choice: the gap
        under the hub is bounded above by the pointer swinging over it and below by the bezel -
        the mark's bottom lands 26.1 px from the middle of a face whose glass stops at 27.8 (see
        DIAL_MARK, and :meth:`_dial_aperture`, which is what would quietly shave a mark that
        reached further). Every cut of the handover mark is drawn inside this same square, so
        two of them differ in shape and in nothing else.
        """
        return max(3.0, self.btn_r * DIAL_MARK), span + self.btn_r * DIAL_LABEL

    def _paint_away(self, t: ImageDraw.ImageDraw, span: int, c: tuple[int, int, int]) -> None:
        """The mark for "his voice is somewhere else", in the gap the cone usually has.

        Which of the cuts in :data:`DIAL_AWAY_CUTS` is drawn is Marco's call, from real renders
        on the real panel - see ``factory/html/003.html``. They are all four here so the choice
        is one word rather than a rewrite, and they all four obey :meth:`_mark_box`.

        What none of them may say is *broken*. The knob has stopped being a control while the
        claim is held, and that is deliberate; what it must not read as is a fault, a dead sink
        or a panel that has lost its speaker. The hue does the work of saying "elsewhere" and
        this mark does the work of saying *where* - which is why the one cut that reads as muted
        is the one cut that cannot be picked.
        """
        r, mid = self._mark_box(span)
        cut = DIAL_AWAY
        if cut == "phone":
            # A different silhouette, not a decorated cone: upright and narrow against the
            # cone's sideways wedge, which is the one difference that survives a glance from a
            # pace away. The screen is knocked back out of it, because a solid slab this size is
            # a domino - it is the frame around a dark rectangle that says "a device".
            # Half as wide as it is tall, which is the proportion doing all the work: against the
            # cone's sideways wedge an upright is the one difference that survives a glance from
            # a pace away, and it is the proportion rather than any detail that says "a phone".
            # A square of the same area reads as a button and a wider one as a battery.
            w = r * 0.50
            t.rounded_rectangle([at(span - w), at(mid - r), at(span + w), at(mid + r)],
                                radius=wide(r * 0.24), fill=linear(c))
            # One knockout, not a frame: a 5 px body has no room for a margin round a screen,
            # and the earpiece slot is the detail that survives being averaged down to size.
            t.line([at(span - w * 0.45), at(mid - r * 0.58),
                    at(span + w * 0.45), at(mid - r * 0.58)],
                   fill=(0, 0, 0, 0), width=round(wide(1.1)))
        elif cut == "chevron":
            # The cone, and the sound carrying on past the gap without it. A chevron rather
            # than an arrow: at ten pixels an arrowhead is three pixels of head on two of shaft,
            # which averages down to a blob - and a *diagonal* arrow beside the cone's wedge
            # reads as a tick, which is the one thing worse than reading as nothing.
            t.polygon(
                [(at(span - r), at(mid - r * 0.22)), (at(span - r * 0.68), at(mid - r * 0.22)),
                 (at(span - r * 0.28), at(mid - r * 0.74)),
                 (at(span - r * 0.28), at(mid + r * 0.74)),
                 (at(span - r * 0.68), at(mid + r * 0.22)), (at(span - r), at(mid + r * 0.22))],
                fill=linear(c),
            )
            t.line([at(span + r * 0.26), at(mid - r * 0.74), at(span + r * 0.96), at(mid),
                    at(span + r * 0.26), at(mid + r * 0.74)],
                   fill=linear(c), width=round(wide(1.4)), joint="curve")
        elif cut == "slash":
            self._paint_speaker(t, span, c)
            # Knocked out under the bar so the bar reads over the cone rather than merging into
            # it: a hairline of its own colour laid straight on a solid shape of that colour is
            # not a bar at all.
            for width, colour in ((wide(2.6), (0, 0, 0, 0)), (wide(1.2), linear(c))):
                t.line([at(span - r * 0.85), at(mid + r * 0.85),
                        at(span + r * 0.85), at(mid - r * 0.85)], fill=colour, width=round(width))
        else:
            # The cone pulled back to the left, and its sound gone off the right-hand side
            # without it: two arcs with a gap where they used to leave the throat. Detached is
            # the whole of the idea - arcs still touching the cone is the loudspeaker glyph
            # every volume control on earth draws, and it would read as "louder".
            t.polygon(
                [(at(span - r), at(mid - r / 4)), (at(span - r * 0.62), at(mid - r / 4)),
                 (at(span - r * 0.24), at(mid - r * 0.80)),
                 (at(span - r * 0.24), at(mid + r * 0.80)),
                 (at(span - r * 0.62), at(mid + r / 4)), (at(span - r), at(mid + r / 4))],
                fill=linear(c),
            )
            for reach, arc in ((r * 0.62, 62.0), (r * 1.18, 50.0)):
                t.arc([at(span + r * 0.38 - reach), at(mid - reach),
                       at(span + r * 0.38 + reach), at(mid + reach)],
                      start=-arc, end=arc, fill=linear(c), width=round(wide(1.2)))


    # ---- the long press, and what it opens ----

    def _draw_hold(self, layer: Image.Image, hold: float) -> None:
        """Fill his collar in, left to right, as a finger holds his face down.

        The one gesture on this panel that is not a tap, so it is the one thing that has to say
        so while it is happening: without this, a long press is a second of a panel doing nothing
        followed by a menu, which reads as a fault that resolved itself.

        It is drawn *on the brass that is already round him* rather than beside it - the same
        band, in full phosphor instead of the metal - so nothing new appears on the screen while
        you hold him. A line you already stopped seeing lights up from one end, and when it
        reaches the far side the menu is open. That also keeps it out of the picture: this panel
        has no room for a progress bar. It runs between the two bolts where the mount's rail
        meets the collar, over the top of him, which is the arc the rail used to take.
        """
        seat = self.eye_seat
        cx, cy = self.eye
        out = seat.swell
        brass_in = out * COLLAR_IN + max(1.5, COLLAR_LIP * self.scale)
        radius = (brass_in + out) / 2.0
        _, _, a0, a1 = self.brackets["bl"].shoulder(seat)
        start, sweep = a0, (a1 - a0) * max(0.0, min(1.0, hold))
        # The brass's whole width: anything narrower is a brightness change on a band that is
        # already thin, which from a bench is no change at all.
        stroke = max(2, round(out - brass_in))
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
