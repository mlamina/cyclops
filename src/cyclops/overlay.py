"""The kiosk's chrome: a Pip-Boy-style terminal bezel drawn over the live camera frame.

OpenCV can only draw Hershey stroke fonts, which look like a 1980s oscilloscope. Everything
here is rendered into one RGBA layer with PIL instead - real TrueType text, anti-aliased
shapes, a numpy-built halo and scanline field - and handed to :mod:`cyclops.kiosk` as a numpy
array to alpha-blend onto the frame. Geometry doubles as the hit-test map: every interactive
element returns its rectangle, so a tap can be resolved without a second layout.

The layout is four corner brackets and a picture. Each bracket's rail comes in square to one
panel edge, ramps across the corner at 45 degrees and lands square on the other; the two top
ones carry two rows of readout each, and the two bottom ones carry the controls. This replaced a
readout strip and a three-cell tab row that between them covered 29% of the panel - the brackets
cover about 14%, and the middle of the screen, which is what somebody holding a camera down a
pipe is actually looking at, is now nothing but picture and a four-arc reticle on the lens axis.

What made that affordable was giving up the words. SNAP, WAKE UP and GO TO SLEEP were what
forced the controls into a row across the bottom: a cell has to be as wide as its label. Two
glyphs and a face need a disc each, and a disc can go in a corner. What the words used to say is
said by the microphone filling in while he is up, by the state word in the opposite corner, and
by the line under the picture.

A bracket is drawn to look like one. The rail is an extrusion rather than a stroke - a shadow
cast inwards onto the plate, then a profile across its width: a lit chamfer on the outer lip, a
body falling away, a flank in shadow - with socket-head bolts wherever it turns and stiffeners
across the deep corner. The plate itself is still see-through: the tube filter (a phosphor wash,
corner shading and a scanline field) laid over the live picture rather than an opaque bar, so
the camera shows through the chrome as well as between it. The border carries the session state
in its colour and glows inwards from it, which is the one thing that has to be readable across a
workshop without reading any words - and it says so in *hue* rather than in brightness, because
dim green and bright green are the same colour to anyone more than a pace away.

His eye is the one thing here that is a face rather than a readout, and it is what lets the rest
stay this terse: a glance at the bottom-left corner answers "is he there, and what is he up to",
so the readouts are left to spell it out only for whoever is close enough to read them. He is the
boot mark brought to life - :mod:`cyclops.eye` draws the splash's iris-inside-HUD-rings with the
rings turning and the iris breathing, and :data:`MOODS` says how, one row per state. Tapping him
opens what the box has kept, because what you ask a face is what it remembers.

His bracket's rail leaves the straight, goes round him and comes back. That shoulder is the
join: it is what makes him part of the bracket rather than a badge sitting on it, and it is why
his corner is the big one - a face wants room, and the two switches opposite are bolted straight
through their rail and take a third of the space. While he is asleep the only things on this
panel that move are his own breath and the microphone: no ring turns, nothing blinks, no readout
changes. That is what makes any of the rest of it read as awake - the *mechanism* stopping,
rather than the creature holding its breath. The switch breathes because it is the only control
left to press, and a control nobody finds is worse than one that beckons.

Everything that holds still while the state does - the halo, the brackets and their bolts, the
reticle, the mode word, the switches at rest - is built once and cached, keyed on the state. A
Pi rendering this at 25 fps has 40 ms for the whole loop and the camera wants most of them; what
is left for a frame here is a signal meter, a clock, a caption, one ring, the border line and the
eye. Every part of the eye moves, so none of it is cached at all; measured on the Pi, he is 10.5
ms of a 13 ms frame, which is the single largest thing this loop does and is meant to be - he is
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
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .eye import EyeEngine, Mood, at, breath, linear, mix, smoothed, wide

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


def session_up(state: str) -> bool:
    """Is there a session at all - from the tap that starts one to the last file it writes?

    The panel's coarsest question: the caption breathes, the border breathes, the microphone
    fills in. The kiosk asks it too, so that what the switch shows and what the switch does can
    never drift apart - see ``_toggle_session``.
    """
    return state not in (IDLE, ERROR)


def awake(state: str) -> bool:
    """Is Cyclops himself up - eye open, listening, able to answer?

    Stricter than :func:`session_up` by exactly the teardown, which can run for a minute while
    the video is muxed. Both are right about their own half: the eye shuts on the tap because
    that is what you asked for, and the cell stays lit with the caption saying "saving the
    video…" because that is what is still true.
    """
    return state not in (IDLE, ERROR, STOPPING)


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
SCREEN = (5, 15, 10)  # the green-black the brackets and their plates are made of
INK = (3, 11, 7)  # text on a filled tab

# The halo answers one question only from across a room - is the agent up? - and the state's
# accent answers it in colour rather than in brightness, which is the only half of it that
# survives the distance. Everything wearing this moves together: the border and its inward glow,
# the mode word, the signal meter, the session clock, the caption's marker, and the mic and word
# on the button that is holding the session open. The rest of the chrome stays green, because a
# panel where everything is an accent has none.
HALOS = {
    IDLE: GREEN_DIM,
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
# What the strip calls each state. Kept here rather than taken from the controller's ``detail``
# because two of these states are the kiosk's own invention and the controller has never heard
# of them; the controller's sentence goes in the caption underneath instead.
#
# Asleep, waking, sleeping - because the control in the other corner is a microphone you press to
# wake him, and a box asked to wake up does not answer STANDBY. The five in the middle read as a
# creature
# doing something and are left alone, and so is FAULT: the metaphor does not get to swallow the
# one word that has to be believed.
LABELS = {
    IDLE: "ASLEEP",
    STARTING: "WAKING",
    STOPPING: "SLEEPING",
    CONNECTING: "WAKING",
    LISTENING: "LISTENING",
    SPEAKING: "SPEAKING",
    LOOKING: "OPTICS",
    SEARCHING: "SEARCH",
    DRAWING: "DRAWING",
    ERROR: "FAULT",
}
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
HEAT_WORD = "HOT"

# ... and the resting line underneath: what is true about a state when nothing finer is known.
# The controller sends a better sentence whenever it has one - what is being searched for, which
# project is being opened, which step of the teardown is running - and that wins; this is what the
# panel falls back on. Every state has one, so the line is never blank while a session is up.
CAPTIONS = {
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
    # The gaze is the newest of these and the one worth reading as a set rather than a row at a
    # time: `gaze` is how far he wanders, `dart` how much of that is flicking rather than
    # drifting, `dart_s` how often, and `look_x`/`look_y` a lean he holds under all of it. What
    # separates a creature from a turret is that the mix differs per state - a hunting eye flicks
    # and a staring one does not - and it costs a number rather than a branch.
    #
    # He drifts in his sleep and never darts: a sleeping face that flicks about is a dreaming
    # one, and this panel is not claiming that. It is also the third thing that moves while he is
    # asleep, after the rings and the breath, and that scarcity is what makes awake read as
    # awake - so it is kept small.
    IDLE: Mood(tint=GREEN_MID, aperture=0.24, swell=0.14, breath_s=6.5, spin=2.5, sway=1.6,
               gaze=0.30),
    # Coming round: the iris only half up, the rings running fast, and a highlight sweeping the
    # rim - a thing spinning itself up rather than a thing paying attention. It looks about while
    # it does it, which is the difference between waking and booting.
    STARTING: Mood(tint=AMBER, aperture=0.34, swell=0.10, breath_s=1.5, spin=54.0, scan=88.0,
                   gaze=0.55, dart=0.45, dart_s=1.1),
    CONNECTING: Mood(tint=AMBER, aperture=0.34, swell=0.10, breath_s=1.5, spin=54.0, scan=88.0,
                     gaze=0.55, dart=0.45, dart_s=1.1),
    # Winding down. The same transition run backwards, which is what the rings do - and the gaze
    # settles as it goes, because a thing finishing is not still looking for anything.
    STOPPING: Mood(tint=AMBER, aperture=0.10, swell=0.04, breath_s=3.0, spin=-22.0, gaze=0.14),
    # Awake and attending. A resting breath, a barely-moving ring set, and the one mood whose
    # iris opens to your voice - which is the panel saying it can hear you. It looks around a
    # little and flicks now and then: attending, not staring you down.
    LISTENING: Mood(
        tint=WHITE, aperture=0.52, swell=0.07, breath_s=4.0, voice=0.30, spin=7.0, blink_s=4.4,
        gaze=0.38, dart=0.30, dart_s=2.9,
    ),
    # Talking: a faster breath and a wider iris, because he is doing the thing rather than
    # waiting to. Barely opens to level here - the level *is* his own voice coming back. He
    # holds your eye while he talks, which is why this wanders less than listening does.
    SPEAKING: Mood(
        tint=WHITE, aperture=0.70, swell=0.17, breath_s=1.1, voice=0.10, spin=13.0, blink_s=5.5,
        gaze=0.22, dart=0.15, dart_s=3.7,
    ),
    # Looking at a photo. Wide, still, and it does not blink: this is a stare. The gaze is barely
    # off zero for the same reason - a stare that wanders is not one.
    LOOKING: Mood(tint=WHITE, aperture=0.88, swell=0.02, breath_s=6.0, spin=3.0, gaze=0.10),
    # Hunting. Narrowed to a point, breathing fast, rings tearing round with a scanning arc, and
    # the eye flicking all over: nearly the full travel, several times a second. This is the one
    # mood where the gaze is the loudest thing about him, and it should be - he is looking *for*
    # something rather than *at* something.
    SEARCHING: Mood(tint=WHITE, aperture=0.36, swell=0.06, breath_s=0.9, spin=155.0, scan=118.0,
                    gaze=0.95, dart=0.85, dart_s=0.6),
    # Drawing. Deliberate, and turning the other way, because it is making rather than looking -
    # and looking down at the work while it does, which is the one mood with a lean it holds.
    DRAWING: Mood(tint=WHITE, aperture=0.46, swell=0.05, breath_s=2.2, spin=-34.0,
                  gaze=0.26, look_x=-0.30, look_y=0.42),
    # A fault. Still and red, and pointedly not pulsing - not even the sleeping breath: a thing
    # that throbs is asking to be watched, and this one is asking to be read. The caption
    # underneath says what broke, and it is the only face on the panel that never moves at all
    # - which now includes its gaze: `gaze` at 0 with no lean is a face staring dead ahead
    # forever, and that is exactly what a fault should do.
    ERROR: Mood(tint=RED, aperture=0.20, swell=0.0, breath_s=0.0, spin=0.0),
}

# A caption that ends in an ellipsis is a caption that moves, and that is the whole test the line
# uses: "searching the web…" walks its dots, "listening — talk to me" holds still. Every phrase
# the controller publishes obeys the same rule, which is why none of them has to say twice whether
# it is a job or a state. Nearly always that means work in flight; the exception is the snore,
# which is not work but is just as much a thing going on.
BUSY_MARK = "…"
MARKER = "› "  # what every caption opens with, and the smallest thing that wears the accent
CAPTION_LINES = 2  # how far a sentence may wrap before it is cut short instead. One line meant
# every phrase worth reading - a fault, a search, what a tool is doing - was trimmed to a stub
# ending in an ellipsis, in a bubble with most of the picture's width still free beside it. Two
# is where it stops: a third would have the bubble standing taller than his head, and a caption
# that big is a dialogue box rather than something said in passing.
CAPTION_ALPHA = 245
CAPTION_DOTS = 3
DOT_PERIOD_S = 1.2  # one sweep of the three dots...
BREATH_PERIOD_S = 2.4  # ...and one breath of the phosphor, at half that rate so the two never lock
BREATH_DEPTH = 0.30  # how far the text sinks towards the slab at the bottom of a breath

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
PLATE_ALPHA = 170  # the one dark backing left: the caption bubble, which sits on the picture.
# Backing rather than a bar: it is there to stop white-on-anything, and every point of alpha past
# what that needs is a point of the room taken away from somebody holding a camera down a pipe to
# look at it. The edge below is what carries the shape, so the fill can afford to be thin.
BUBBLE_EDGE_ALPHA = 215  # ...and the outline, which stays nearly solid. Fill and edge used to be
# one number, so thinning the fill dissolved the bubble's own shape along with it - which is the
# half that has to survive whatever the camera is pointed at
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
# The rail on that spine is an extrusion rather than a stroke: a shadow cast inwards onto the
# plate, then a profile drawn across its width - a lit chamfer on the outer lip, a body falling
# away, a flank in shadow. That is what makes a bracket read as something bolted to the panel
# rather than as a line drawn on it, and it is why the rail is this thick: at a hairline there is
# no width for a profile to happen in.
RAIL = 17.0  # reference pixels, and the one number the whole bracket language rests on
RAIL_LIP = 0.78  # fraction of the rail's width that is the lit chamfer
RAIL_BODY = 0.28  # ...and where the body gives way to the flank in shadow
RAIL_SHADOW = 165  # alpha of the shadow the rail casts onto its own plate
BOLT_R = 7.0  # a socket head, sunk through the rail wherever it turns
RIB_N = 3  # stiffeners across the deep corner of a bracket
PLATE_WASH = 0.34  # how far a bracket's plate is put towards SCREEN. Not opaque: a bracket you
# cannot see the room through is a bar, and this layout exists to stop having those.

# The spines, at the reference. Anything on the right or the bottom is written as a distance in
# from that edge, so one table lays out all four and mirrors without a second.
#
# The two bottom brackets are deliberately unequal. The left one is a housing - it has a face in
# it, and a face wants room. The right one is a bracket with two switches bolted through it, and
# every pixel it does not take is a pixel of the room somebody is holding a camera down a pipe
# to look at.
TOP_H = 78.0  # the top brackets' depth, and two lines of readout fit in it
TOP_STEP = 26.0  # the square landing before the top edge
TOP_L_FLAT = 218.0  # where the left bracket's ramp starts
TOP_R_FLAT = 190.0  # ...and the right one's, in from the right edge
BOT_L = 250.0  # the bottom-left bracket's reach along both edges
BOT_L_STEP = 38.0  # its square landings
BOT_R_OUT = 170.0  # the bottom-right bracket's reach in from the right edge...
BOT_R_STEP = 38.0  # ...its landing on the bottom edge...
BOT_R_LAND = 26.0  # ...and the shorter one on the right, which is what makes it the small one

# The eye. He rides the left bracket's ramp, sunk halfway into it - `EYE_SEAT` is that depth as a
# fraction of the swell's radius, and acos(0.5) is a 60-degree shoulder, which is where the rail
# leaves the straight and goes round him. Half of him is in the bracket and half is over the
# picture, which is the same join the tab row used to make and the reason he reads as part of the
# machine rather than as a badge stuck on it.
EYE_R = 0.1833  # 88 px at 800x480, against 60 in the row this replaced
EYE_SHOULDER = 16.0  # reference px between his rim and the rail's centreline round him
EYE_SEAT = 0.5
EYE_PLATE_ALPHA = 205  # the disc behind him. Lighter than the caption's slab on purpose: this
# one sits over the middle of the picture, and a porthole you cannot see through is a hole

# The two switches, bolted straight through the small bracket's rail rather than sitting in a
# plate of their own. Neither has a word: a glyph that needs a label is the wrong glyph, and the
# words were what forced the controls into a row across the bottom in the first place.
#
# Shutter nearer the middle of the panel, microphone nearer the corner, so the left-to-right
# order the tab row taught still holds on a diagonal.
BTN_R = 36.0
BTN_SPACING = 52.0  # along the ramp, either side of its middle
SWITCHES = ("shutter", "wake")

# The reticle: four arcs on the lens axis and nothing else. It was a cross with graduations for
# about an hour, which is exactly as long as it took somebody to say it looked like a gun sight -
# and they were right. A broken ring says "lens" and says nothing else.
RETICLE_R = 0.1625  # 78 px at 480
RETICLE_ARC = 68.0  # degrees of ring per quadrant; the rest is gap

# The other thing that moves on a sleeping panel, his breath being the first. Everything else
# holds still - that stillness is what makes awake read as awake - but the button that ends it may
# say so, because a control that does nothing until you find it is worth pointing at. A slow
# swell, not a flash: this is an invitation, and a panel blinking at you across a workshop is an
# alarm.
WAKE_PERIOD_S = 2.9  # seconds a breath takes, and not a multiple of any other on this panel
WAKE_GLOW = 1.0  # how far the glyph and its bezel travel towards the colour he will be

RIM_PERIOD_S = 3.7  # one breath of the border, slower than the caption's and not a multiple of it
RIM_DEPTH = 0.14  # how far it sinks towards SCREEN - a mix, not an alpha, and a quarter of what
# the caption may do, because this is the one thing readable across a workshop

METER_SEGMENTS = 8  # steps in the signal bar
# How much colour is stirred into the chrome for the parts that are meant to look faded. These
# are mixes rather than alphas on purpose - see _mix: drawing them translucently would not dim
# them, it would open a window onto whatever the camera is pointed at.
METER_OFF = 0.45  # an unlit signal segment
RING_MIX = 0.55  # the ring around the live microphone
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

    def __init__(self, corner: tuple[int, int], spine: Sequence[tuple[int, int]],
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
        other."""
        points: list[tuple[float, float]] = [self.spine[0], self.spine[1]]
        for seat in self.seats:
            p0, p1, a0, a1 = self.shoulder(seat)
            points.append(p0)
            points += arc_points(*seat.centre, seat.swell, a0, a1)[1:-1]
            points.append(p1)
        points += [self.spine[2], self.spine[3]]
        return points

    def plate(self, d: ImageDraw.ImageDraw) -> None:
        """Fill the bracket's footprint into a mask: the gusset, plus every seat's swell."""
        d.polygon([self.corner, *self.spine], fill=255)
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


def caption_pulse(phase: float) -> tuple[float, int]:
    """How far the caption has sunk, and how many dots trail it, at monotonic time *phase*.

    A raised cosine rather than a square wave: this is a phosphor tube, and a line snapping on
    and off reads as a fault light rather than as work being done. What comes back is a *mix*
    and not an alpha, which is not a detail - see :func:`_mix`. PIL writes into the chrome layer
    rather than compositing onto it, so a translucent letter is not a dimmer letter, it is a
    window onto whatever the camera is pointed at, punched through the slab that was put there
    to stop exactly that.

    Time rather than frames, because the loop does not run at one rate - 25 fps with the camera
    up, 5 while the admin page covers the panel, 4 asleep - and a dot per frame would gallop and
    stall along with it.
    """
    step = DOT_PERIOD_S / (CAPTION_DOTS + 1)
    return BREATH_DEPTH * breath(phase, BREATH_PERIOD_S), int((phase % DOT_PERIOD_S) / step)


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

    shutter: Rect
    eye: Rect  # the middle cell, which is his face and opens what the box has kept
    wake: Rect


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
        # The four spines. Each is [edge, knee, knee, edge], and the ramp between the knees is
        # built from its own start rather than from a second table entry, so it is a true 45
        # whatever the rounding does to the numbers either side of it.
        top, step = max(10, px(TOP_H)), max(4, px(TOP_STEP))
        ramp = top - step
        bot, bstep = max(12, px(BOT_L)), max(4, px(BOT_L_STEP))
        blegs = bot - bstep
        rout, rstep = max(12, px(BOT_R_OUT)), max(4, px(BOT_R_STEP))
        rlegs = rout - max(4, px(BOT_R_LAND))
        self.spines = {
            "tl": [(0, top), (px(TOP_L_FLAT), top), (px(TOP_L_FLAT) + ramp, step),
                   (px(TOP_L_FLAT) + ramp, 0)],
            "tr": [(width, top), (width - px(TOP_R_FLAT), top),
                   (width - px(TOP_R_FLAT) - ramp, step), (width - px(TOP_R_FLAT) - ramp, 0)],
            "bl": [(0, height - bot), (bstep, height - bot),
                   (bstep + blegs, height - bot + blegs), (bstep + blegs, height)],
            "br": [(width - rout, height), (width - rout, height - rstep),
                   (width - rout + rlegs, height - rstep - rlegs),
                   (width, height - rstep - rlegs)],
        }
        self.brackets = {
            name: Bracket((0 if name[1] == "l" else width, 0 if name[0] == "t" else height), spine)
            for name, spine in self.spines.items()
        }

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
        spacing = max(self.btn_r + 4, px(BTN_SPACING))
        self.switches = {
            name: self.brackets["br"].on_ramp(offset)
            for name, offset in zip(SWITCHES, (-spacing, spacing), strict=True)
        }

        # What makes the status line a bubble rather than a slab is the tail, and only the tail.
        # The corners were rounded for a while and it was the wrong borrowing: every other edge on
        # this panel is square or is a full arc, and a softened box in the middle of it read as a
        # chat app dropped onto a terminal.
        self.caption_tail = max(3, round(11 * scale))
        self.caption_root = max(4, round(15 * scale))
        self.caption_lean = max(2, round(6 * scale))
        self.caption_h = round(24 * scale)  # one line of it
        self.caption_nose = max(3, round(14 * scale))  # tail tip to his shoulder
        # The bubble's left edge is the fixed one, pinned beside him, so the tail always lands in
        # the same place - on him - and the sentence grows away to the right. It clears his
        # *swell* rather than his rim, because the rail goes round him out there and a tail
        # coming out of the rail is a tail coming out of the bracket.
        #
        # It hangs off his shoulder now rather than straight out from his side, so the gap is
        # solved for rather than added on: the tail's tip is put exactly `caption_nose` clear of
        # the swell along whatever diagonal it happens to lie on. Written as `x + nose` instead,
        # the constant would mean fourteen pixels at one window size and thirty at another, which
        # is how a tail stops looking like it comes out of anything.
        self.caption_bottom = self.eye[1] - self.eye_r + round(15 * scale)
        reach = self.shoulder + self.caption_nose
        drop = self.caption_bottom + self.caption_tail - self.eye[1]
        across = math.sqrt(reach * reach - drop * drop) if abs(drop) < reach else reach
        self.caption_left = math.ceil(self.eye[0] + across) + self.caption_lean
        # ...and it stops short of the switches rather than at the frame's own padding. A
        # two-line sentence at full width has its bottom edge and its tail at very nearly the
        # height of the microphone, and a bubble clipping the top of a control is a bubble that
        # has made the control look broken.
        self.caption_right = min(
            self.width - self.pad - round(34 * scale),
            self.switches["wake"][0] - self.btn_r - round(12 * scale),
        )
        self.caption_y = self.caption_bottom - self.caption_h / 2  # the bottom line's middle

        self.font_mode = _load_font(max(11, round(27 * scale)))
        self.font_read = _load_font(max(9, round(21 * scale)))
        self.font_tab = _load_font(max(8, round(15 * scale)))
        self.font_brand = _load_font(max(7, round(14 * scale)))
        self.font_micro = _load_font(max(7, round(12 * scale)))
        self.font_caption = _load_font(max(8, round(14 * scale)))

        # Where the two rows of each top bracket sit. Centres rather than baselines, because
        # _text centres on the y it is given - and the mode word's row has to clear the rail,
        # which is `rail_w` wide and centred on the spine.
        self.row_top = round(25 * scale)
        self.row_bottom = round(52 * scale)
        self.read_pad = self.pad + round(10 * scale)

        self._halo = halo_alpha(width, height)
        self._backdrop = self._build_backdrop()
        self._plate = self._build_plate()
        self._chrome = self._build_chrome()
        # One engine per window size: it owns the geometry, and it remembers which mood it is
        # easing out of, which is why it is built here and not per frame.
        self.engine = EyeEngine(self.eye_r, self.line, SCREEN, MOODS[IDLE])
        self._bases: dict[tuple[str, bool, str], Image.Image] = {}
        # The readout strip's right-hand group is laid out from the frame edge inwards, and in a
        # monospaced face every width in it is a constant, so it is worked out here rather than
        # per frame - and, more to the point, the baked half and the drawn half then agree.
        self._clock_w = self.font_read.getlength("00:00")
        self._rec_w = self.font_micro.getlength("REC") + round(7 * scale) * 2
        self._dots_w = self.font_caption.getlength("." * CAPTION_DOTS)
        self._gap = max(4, round(18 * scale))
        self._seg = (max(3, round(9 * scale)), max(6, round(16 * scale)), max(2, round(5 * scale)))
        self.hitboxes = self._layout()
        self.menu_card, self.menu_cells = self._menu_layout()
        self._scrim: Image.Image | None = None  # built on the first long press, then kept

    # ---- layout ----

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
            shutter=self._disc(self.switches["shutter"], self.btn_r),
            eye=self._disc(self.eye, self.eye_r),
            wake=self._disc(self.switches["wake"], self.btn_r),
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

    def _bracket_mask(self) -> np.ndarray:
        """Where the four brackets are, as 0..1 - their plates, less the disc his face fills.

        His plate is what backs him instead, and it is the same all the way round; without this
        the wash's own edge would run across his face as a tide line. The switches keep the wash
        under them, because their wells are opaque enough not to care and cutting two more holes
        in a mask is two more edges to land in the wrong place.
        """
        plate = Image.new("L", (self.width, self.height), 0)
        d = ImageDraw.Draw(plate)
        for bracket in self.brackets.values():
            bracket.plate(d)
        holes = Image.new("L", (self.width, self.height), 0)
        hd = ImageDraw.Draw(holes)
        cx, cy, r = *self.eye, self.eye_r
        hd.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
        inside = np.asarray(plate, np.float32) / 255.0
        inside *= 1.0 - np.asarray(holes, np.float32) / 255.0
        rounded = Image.new("L", (self.width, self.height), 0)
        ImageDraw.Draw(rounded).rounded_rectangle(
            [0, 0, self.width - 1, self.height - 1], radius=self.radius, fill=255
        )
        return inside * (np.asarray(rounded, np.float32) / 255.0)

    def _build_backdrop(self) -> tuple[np.ndarray, np.ndarray]:
        """The tube filter - a wash, corner shading and scanlines - inside the brackets only.

        It used to be laid over a strip and a tab row that between them covered 29% of the panel.
        Four corner brackets cover about 14%, and the picture runs edge to edge behind all of it:
        this is what the chrome is *made of* rather than something sitting under an opaque bar.
        The plate is darker than it was, though, because a bracket is a thing rather than a tint -
        see PLATE_WASH, which is as far towards SCREEN as it goes and no further.
        """
        shape = (self.height, self.width)
        base: tuple[np.ndarray, np.ndarray] = (
            np.zeros((*shape, 3), dtype=np.float32),
            np.zeros(shape, dtype=np.float32),
        )
        base = _over(base, SCREEN, np.full(shape, PLATE_WASH, dtype=np.float32))
        base = _over(base, GREEN, np.full(shape, TINT_ALPHA, dtype=np.float32))
        base = _over(base, (0, 0, 0), vignette_alpha(self.width, self.height))
        base = _over(base, (0, 0, 0), scanline_alpha(self.width, self.height))
        rgb, alpha = base
        return rgb, alpha * self._bracket_mask()

    def _build_plate(self) -> Image.Image:
        """The disc behind the eye, on its own layer under the chrome.

        The same argument the caption's slab makes: he is drawn in thin rings over a live camera,
        and a green hairline over whatever the lens is pointed at is a coin toss. Lighter than
        the caption's slab, though - that one backs a line of text you have to read, and this one
        backs a face you only have to recognise, so the room can still ghost through behind him.

        A separate layer rather than part of :meth:`_build_chrome`, because that one blurs its own
        alpha to make the bloom, and a filled disc this size through a Gaussian blur is not a
        glow, it is a lamp.
        """
        layer = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        cx, cy, r = *self.eye, self.eye_r
        ImageDraw.Draw(layer).ellipse(
            [cx - r, cy - r, cx + r, cy + r], fill=(*SCREEN, EYE_PLATE_ALPHA)
        )
        return layer

    def _build_chrome(self) -> Image.Image:
        """The four brackets and the reticle, on transparency.

        Drawn once and kept: nothing in here depends on the state, only on the window size. The
        border around the outside is not in here - it carries the state colour, so it belongs to
        the per-state base and goes on last of all.

        There is no bloom over any of it, which the doubled hairlines this replaces did have. A
        Gaussian blur over a 17 px rail is not a glow, it is a lamp - and the profile across the
        rail's own width is already doing the job the bloom was there to do, which is to say that
        the chrome has a thickness.
        """
        layer = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        for bracket in self.brackets.values():
            self._draw_bracket(layer, bracket)
        self._draw_reticle(layer)
        return layer

    def _rail_colour(self, across: float) -> tuple[int, int, int]:
        """The rail seen end-on. *across* runs 1 at the outer lip to 0 at the inner flank.

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

    def _draw_rail(self, layer: Image.Image, points: Sequence[tuple[float, float]]) -> None:
        """An extrusion, not a stroke: a shadow cast onto the plate, then a profile across it.

        The shadow is what does most of the work. A rail with a chamfer and no shadow reads as a
        drawing of a rail; the same rail with something dark falling off its inner edge sits *on*
        something. It is blurred, so it is composited from a layer of its own rather than drawn -
        blurring the chrome in place would drag every hairline on the panel with it.

        Both halves are built here rather than at a frame: this whole layer is cached per window
        size, so the cost is four Gaussians once and nothing at all at 25 fps.
        """
        thick = self.rail_w
        shade = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        ImageDraw.Draw(shade).line(
            [tuple(p) for p in offset_path(points, -thick * 0.3)],
            fill=(0, 0, 0, RAIL_SHADOW), width=thick + max(2, round(6 * self.scale)),
            joint="curve",
        )
        layer.alpha_composite(shade.filter(ImageFilter.GaussianBlur(max(1.5, 5.0 * self.scale))))
        d = ImageDraw.Draw(layer)
        half = thick / 2.0
        for i in range(thick):
            at_ = half - 0.5 - i
            d.line([tuple(p) for p in offset_path(points, at_)],
                   fill=(*self._rail_colour((at_ + half) / thick), 255), width=2, joint="curve")
        # The lit lip and the flank, crisp on top of the bands they end - two pixels each, which
        # is what stops a seventeen-band profile reading as a smudge with a bright side.
        d.line([tuple(p) for p in offset_path(points, half - 0.6)],
               fill=(*mix(GREEN_MID, GREEN, 0.5), 255), width=2, joint="curve")
        d.line([tuple(p) for p in offset_path(points, -(half - 0.6))],
               fill=(*SCREEN, 240), width=2, joint="curve")

    def _draw_bolt(
        self, d: ImageDraw.ImageDraw, x: float, y: float, r: float | None = None
    ) -> None:
        """A socket head sunk through the rail: a shadowed seat, a lit rim above, a dark hex.

        The one detail that says a bracket is bolted on rather than drawn on, and it costs a
        circle and a hexagon. The rim is two arcs and not one ring, which is the whole of the
        depth: light from above means the top half of a countersink is bright and the bottom
        half is not.
        """
        r = self.bolt_r if r is None else r
        d.ellipse([x - r - 1, y - r, x + r + 1, y + r + 2], fill=(0, 0, 0, 130))
        d.ellipse([x - r, y - r, x + r, y + r], fill=(*mix(GREEN_MID, SCREEN, 0.72), 255))
        d.arc([x - r, y - r, x + r, y + r], start=170, end=350, fill=(*GREEN, 245), width=2)
        d.arc([x - r, y - r, x + r, y + r], start=350, end=530, fill=(*SCREEN, 220), width=2)
        d.polygon(
            [(x + r * 0.52 * math.cos(math.radians(a)), y + r * 0.52 * math.sin(math.radians(a)))
             for a in range(0, 360, 60)],
            fill=(*SCREEN, 255), outline=(*mix(GREEN_MID, SCREEN, 0.45), 190),
        )

    def _draw_bracket(self, layer: Image.Image, bracket: Bracket) -> None:
        """Stiffeners, then the rail, then the bolts at every place the rail turns.

        The ribs go down first and stay near the corner. Run them out towards the rail and they
        stop reading as webbing inside a bracket and start reading as stripes laid over the room,
        which is the one thing this layout is spending its corners to avoid.
        """
        d = ImageDraw.Draw(layer)
        corner, a, b = bracket.corner, bracket.spine[0], bracket.spine[-1]
        for i in range(RIB_N):
            t = 0.15 + 0.075 * i
            d.line(
                [(corner[0] + (a[0] - corner[0]) * t, corner[1] + (a[1] - corner[1]) * t),
                 (corner[0] + (b[0] - corner[0]) * t, corner[1] + (b[1] - corner[1]) * t)],
                fill=(*mix(GREEN_MID, SCREEN, 0.5), 170), width=max(1, round(4 * self.scale)),
            )
        self._draw_rail(layer, bracket.path())
        d = ImageDraw.Draw(layer)
        for knee in bracket.spine[1:3]:
            self._draw_bolt(d, *knee)
        # ...and where the rail leaves the straight to go round a face, which is the one join on
        # this panel that is carrying anything.
        for seat in bracket.seats:
            p0, p1, _, _ = bracket.shoulder(seat)
            self._draw_bolt(d, *p0, self.bolt_r - 1)
            self._draw_bolt(d, *p1, self.bolt_r - 1)

    def _draw_reticle(self, layer: Image.Image) -> None:
        """Four arcs on the lens axis, and nothing in the middle of them.

        What a camera shows you when it is looking rather than aiming. This was a gapped cross
        with graduations for exactly as long as it took somebody to look at it and say it read as
        a gun sight; the arcs say lens and say nothing else, and they leave the centre of the
        picture - which is the part anybody actually points the thing at - completely clear.

        Supersampled, unlike the corner ticks it replaces: a stepped circle in the middle of an
        otherwise smooth panel is the most conspicuous kind of aliasing there is, and this layer
        is built once so the tile costs nothing at a frame.
        """
        r = max(8, round(RETICLE_R * self.height))
        cx, cy = self.width // 2, self.height // 2
        span = r + self.line * 2

        def paint(t: ImageDraw.ImageDraw) -> None:
            middle, reach = at(span), at(r)
            for quadrant in range(4):
                start = quadrant * 90 - RETICLE_ARC / 2
                t.arc(
                    [middle - reach, middle - reach, middle + reach, middle + reach],
                    start=start, end=start + RETICLE_ARC, fill=linear(GREEN_MID, 230),
                    width=round(wide(self.line)),
                )

        layer.alpha_composite(smoothed(2 * span + 1, paint), (cx - span, cy - span))


    def _base(self, state: str, recording: bool, heat: str = "") -> Image.Image:
        """Everything that holds still while the state does, built once and copied per frame.

        The state light goes on last, after the words, and falls inwards from the border. It
        was tried the other way - a margin of dark bezel with the glow spreading outwards into
        it - and it read as light leaking off the edge of the device rather than as a screen lit
        from inside. A tube blooms in front of what it is showing, so this one does too.

        Keyed on the state rather than on the halo colour, because the words change with it too.
        Baking the brand, the mode, the REC tag and both switches in here is what keeps a frame
        down to a meter, a clock, a caption and a ring: drawing all of it every time cost 10 ms of
        the Pi's 40 ms budget, against 2.8 ms for the chrome the tab row replaced.
        """
        cached = self._bases.get((state, recording, heat))
        if cached is not None:
            return cached
        halo = HALOS.get(state, GREEN_DIM)
        # The filter, and then the chrome drawn on it. There is nothing opaque underneath either
        # of them any more: the strip, the tab row and the four scraps outside the border's
        # rounded corners are all just the wash and the scanlines over the live picture, and the
        # picture between them is not covered at all - not even by the border's own corners.
        image = Image.alpha_composite(_to_image(*self._backdrop), self._plate)
        image = Image.alpha_composite(image, self._chrome)

        d = ImageDraw.Draw(image)
        self._bake_header(d, state, halo, recording, heat)
        for name in SWITCHES:
            self._draw_switch(d, name, state, halo, pressed=False)

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

    @staticmethod
    def _taping(state: str, recording: bool) -> bool:
        """Is a recording actually being made? Configured to record is not the same thing.

        ``recording`` only says the setting is on. The tag has to mean "tape is running", or a
        panel sitting at ASLEEP claims to be filming the room.
        """
        return recording and session_up(state)

    def _readouts(self, taping: bool) -> tuple[float, float, float]:
        """Right edges of the clock, the REC tag and the signal meter, laid out edge inwards.

        Two rows now rather than one line: the meter and its label on top, the clock underneath.
        That is what lets the top-right bracket be a corner rather than a bar - a single row of
        SIG, eight segments, REC and a clock is 290 px wide, and stacked it is 190. The clock has
        the bottom row to itself, so its right edge is the only one of the three that is not on
        the meter's row and the REC tag sits beside the meter instead of between it and the
        clock.
        """
        right = self.frame.right - self.read_pad
        meter = right
        rec = self._meter_x(meter) - self._gap
        return right, rec, meter

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
        second did, in a bubble that is already redrawn as the sentence changes.

        A word longer than the whole limit still goes down and is cut mid-word by :meth:`_elide`,
        rather than being dropped or spilling out of the bubble. Rare, and always a URL or a
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
    ) -> np.ndarray:
        """Draw the whole chrome for this frame and return it as an RGBA numpy array.

        ``phase`` is a monotonic clock in seconds, and the only argument here that is not about
        what the panel is showing but about *when*. It is passed in rather than read here so a
        frame is a pure function of its arguments and the caption's animation can be tested
        without a clock - the same shape as ``flash``, which the kiosk has always computed.

        Almost nothing here moves while he is asleep, and that is deliberate and half the
        design: no ring turns, the eye stays shut, the border holds still, the caption stops
        breathing and the readouts have nothing to count. What is left is his breath and the
        microphone - the creature, and the way out of him. Against a panel that was quietly
        pulsing whatever it was doing, an awake one that pulses says nothing.

        ``hold`` is how far a finger is through the long press on his face, 0 to 1, and ``menu``
        is whether that press has landed - the power menu, over everything else. The two are the
        one gesture on this panel that is not a tap, so they are the one thing here drawn from a
        clock the kiosk is holding rather than from the state.
        """
        halo = HALOS.get(state, GREEN_DIM)
        layer = self._base(state, recording, heat).copy()
        d = ImageDraw.Draw(layer)

        self._draw_readouts(d, halo, level, elapsed, self._taping(state, recording))
        self._draw_caption(d, state, halo, detail, phase)
        held = pressed == "eye"
        if pressed in SWITCHES and not held:
            # Redrawn over the switch the base has at rest: an inverted control is the only
            # feedback a screen with no travel can give, and it lasts a handful of frames.
            self._draw_switch(d, pressed, state, halo, pressed=True)
        # After the pressed switch, not before it: the ring used to be drawn, painted over by an
        # inverted cell and then drawn again in ink. One order, one draw, one colour.
        if awake(state):
            inverted = pressed == "wake"
            ring = mix(halo, INK, RING_MIX) if inverted else mix(SCREEN, halo, 1.0 - RING_MIX)
            self._draw_ring(d, ring, level)
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
                           voice=0.0, gaze=0.0, dart=0.0)
        self.engine.paint(layer, *self.eye, mood, phase, level)
        if hold > 0.0:
            self._draw_hold(layer, hold)
        if session_up(state):
            # The teardown breathes too. He is not listening any more - the eye is already shut -
            # but the box is still working, and a panel that went stone still the moment you
            # pressed stop would look like it had stopped rather than like it was finishing.
            self._draw_rim(d, halo, phase)
        elif state == IDLE and pressed != "wake":
            self._draw_invite(d, phase)
        if menu:
            # Last of everything, because it is the only thing here that is asked a question
            # rather than told one: nothing behind it is live while it is up.
            self._draw_menu(layer, d, pressed)
        if flash > 0.0:
            # Green-white rather than white: a photo taken through a phosphor screen.
            d.rectangle([0, 0, self.width, self.height], fill=(214, 255, 228, int(190 * flash)))
        return np.asarray(layer)

    def _bake_header(
        self,
        d: ImageDraw.ImageDraw,
        state: str,
        halo: tuple[int, int, int],
        recording: bool,
        heat: str = "",
    ) -> None:
        """The half of the two top brackets that only moves when the state does.

        Who he is over what he is doing on the left; the SIG caption and the REC tag on the
        right. Two rows rather than one because that is what turns a bar into a corner: a bracket
        wide enough for CYCLOPS, a divider and LISTENING on one line would reach a third of the
        way across the panel, and stacked it stops at an eighth.

        All of it is letter-spaced, which PIL can only do a character at a time, which is
        precisely why it is baked rather than redrawn 25 times a second.
        """
        x = self.frame.x + self.read_pad
        track = max(1.0, 2.0 * self.scale)

        # A step up from GREEN_DIM, which is what it wore on the old strip. The bracket's plate
        # is darker than that strip was and the name sits on the busiest corner of it, so the
        # faintest phosphor on the panel stopped being legible there.
        brand = self._text(
            d, x, self.row_top, "CYCLOPS", self.font_brand, (*GREEN_MID, 255), tracking=track
        )
        # The heat lamp goes on the brand's row rather than beside the mode word, which is the
        # widest thing in the bracket and has nowhere to put it. A warning beside the name of the
        # box is still the first place somebody wondering why it is misbehaving will look.
        colour = HEAT_LAMP.get(heat)
        if colour is not None:
            self._tag(d, x + brand + self._gap, self.row_top, HEAT_WORD, colour)
        self._text(
            d, x, self.row_bottom, LABELS.get(state, "—"), self.font_mode, (*halo, 255),
            tracking=track * 0.7,
        )

        taping = self._taping(state, recording)
        _, rec_right, meter_right = self._readouts(taping)
        cy = self.row_top
        if taping:
            # Red, and a filled tag rather than a dot. Red is what a record light is on every
            # other machine anybody has ever used, which is worth more here than the panel's
            # preference for its own green - and filling it rather than outlining it is how this
            # tube shouts. The one thing on screen that is red without being a fault, which is
            # exactly why it is a tag with a word in it and not a lamp.
            self._tag(d, rec_right - self._rec_w, cy, "REC", RED)
        else:
            # SIG only when there is room for it. With the tape running the label gives its place
            # to the tag, which is the one of the two that is telling you something you did not
            # already know from the eight segments beside it.
            self._text(
                d,
                self._meter_x(meter_right) - round(9 * self.scale),
                cy,
                "SIG",
                self.font_micro,
                (*GREEN_DIM, 255),
                align="r",
            )

    def _tag(
        self, d: ImageDraw.ImageDraw, x: float, cy: float, word: str, colour: tuple[int, int, int]
    ) -> float:
        """A filled rounded slab with a word knocked out of it, left edge at *x*. Returns width.

        The panel's way of shouting, and there are two things that do it: REC and the heat lamp.
        They were one shape typed out twice for exactly as long as it took to add the second, so
        they are one method now - which also means :attr:`_rec_w`, the width the right-hand group
        is laid out against, is measured the same way the thing is drawn.
        """
        half = round(11 * self.scale)
        pad = round(7 * self.scale)
        width = self.font_micro.getlength(word) + pad * 2
        d.rounded_rectangle(
            [x, cy - half, x + width, cy + half],
            radius=max(1, round(3 * self.scale)),
            fill=(*colour, 255),
        )
        self._text(d, x + pad, cy, word, self.font_micro, (*INK, 255))
        return width

    def _draw_readouts(
        self,
        d: ImageDraw.ImageDraw,
        halo: tuple[int, int, int],
        level: float,
        elapsed: float | None,
        taping: bool,
    ) -> None:
        """The half that moves: the signal bar, and the clock counting the session up."""
        clock_right, _, meter_right = self._readouts(taping)
        whole = 0 if elapsed is None else int(elapsed)
        clock = "--:--" if elapsed is None else f"{whole // 60:02d}:{whole % 60:02d}"
        # Dim green with nothing to count, the state's accent the moment there is - the numbers
        # that only mean something during a session are the right place for the colour that only
        # appears during one.
        colour = (*GREEN_DIM, 255) if elapsed is None else (*halo, 255)
        self._text(d, clock_right, self.row_bottom, clock, self.font_read, colour, align="r")

        cy = self.row_top
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
        self, d: ImageDraw.ImageDraw, state: str, halo: tuple, detail: str, phase: float
    ) -> None:
        """Plain English over the bottom of the picture, in a bubble coming out of his face.

        The bubble is not decoration: this text sits on the live camera, and white-on-anything is
        a coin toss. It is also where an error actually says what went wrong, which the old
        chrome could only render as a red rim. What it is *shaped* like is a separate argument -
        see :meth:`_bubble`.

        The controller's sentence wins over this module's own table, and not the other way round
        as it used to: it is the half that knows what is being searched for, which project is
        being opened and how far the teardown has got, and CAPTIONS is what is left to say when
        it knows nothing finer. Reversed, every one of those sentences would be swallowed by a
        state word during exactly the states worth narrating.
        """
        text = detail or CAPTIONS.get(state, "")
        if not text:
            return  # the strip already says the mode; saying it twice is not a caption
        busy = text.endswith(BUSY_MARK)
        if busy:
            text = text[: -len(BUSY_MARK)]  # the dots take the ellipsis's place, and move
        font = self.font_caption
        # Laid out from his rim outwards, and the marker rides along on the front of the sentence
        # so the wrap can put it where it belongs rather than the drawing assuming a first line.
        x, inset = self.caption_left, round(8 * self.scale)
        # The dots are reserved on every line whether any of them are showing or not. On the last
        # one that is what stops the bubble's far edge shuffling four times a second while they
        # count; on the others it costs at most a word's placement, which is cheaper than working
        # out which line will turn out to be the widest before wrapping it.
        dots_w = self._dots_w if busy else 0.0
        limit = self.caption_right - x - inset * 2 - dots_w
        lines = self._wrap(MARKER + text, font, limit, CAPTION_LINES)
        if busy and lines[-1].endswith(BUSY_MARK):
            # Cut short *and* about work in flight, which used to come out as "an M8 s…..": the
            # ellipsis the trim leaves behind, followed by the dots that were already standing in
            # for one. Four marks doing one mark's job. The dots win, because they are the half
            # that moves, and they mean what the ellipsis meant anyway.
            lines[-1] = lines[-1][: -len(BUSY_MARK)]
        widths = [font.getlength(line) for line in lines]
        widths[-1] += dots_w  # the dots trail the last line, so only that one has to make room
        right = x + max(widths) + inset * 2
        # It grows *up*. The bottom edge is where the tail hangs from and the tail has his
        # shoulder to clear, so that edge is layout; a second line has nowhere to go but upwards,
        # over a part of the picture where there is nothing but his own head anyway.
        bottom = self.caption_bottom
        top = bottom - len(lines) * self.caption_h
        self._bubble(d, x, right, top, bottom)
        # The breath runs under every caption of a session that is up - it is what makes the line
        # read as a live tube rather than a printed label - and the dots under any line that ends
        # in an ellipsis, where they mean the thing everybody already reads them to mean.
        #
        # Asleep the breath stops and the dots do not. The breath stopping is the point rather
        # than an economy: it used to run unconditionally, so a panel with nothing on it was
        # quietly pulsing 809 pixels of caption, and against that an awake panel that pulses says
        # nothing at all - and brightness stopped being his register the day the sleeping face
        # gave it up. The dots stay because what the line says while he is asleep is a snore, and
        # a snore that holds still is a printed label. A fault gets neither: it is asking to be
        # read, not watched.
        sunk, walking = caption_pulse(phase)
        sunk = sunk if session_up(state) else 0.0
        lit = walking if session_up(state) or state == IDLE else 0
        colour = mix(halo if state == ERROR else GREEN, SCREEN, sunk)
        # The marker takes the accent and the sentence does not. A whole line of running text in
        # white over a live camera is harder to read than the same line in phosphor, and the
        # marker is the part that is decoration anyway - so it is the part that gets to be a
        # colour, and it breathes with the words it introduces.
        accent = (*mix(halo, SCREEN, sunk), CAPTION_ALPHA)
        for i, line in enumerate(lines):
            y = top + (i + 0.5) * self.caption_h
            at = x + inset
            if line.startswith(MARKER):  # only ever the first, and only if the wrap left it there
                at += self._text(d, at, y, MARKER, font, accent)
                line = line[len(MARKER):]
            at += self._text(d, at, y, line, font, (*colour, CAPTION_ALPHA))
            if busy and lit and i == len(lines) - 1:
                # Hard against the last letter, where an ellipsis belongs - these are standing in
                # for the one the phrase arrived with, not sitting beside it as a separate mark.
                self._text(d, at, y, "." * lit, font, (*colour, CAPTION_ALPHA))

    def _bubble(
        self, d: ImageDraw.ImageDraw, x: float, right: float, top: float, bottom: float
    ) -> None:
        """The slab the caption sits on, shaped like what it is: him saying something.

        The line was always his - the marker, the plain English, the breath under it - and a
        rectangle was the one part of it that read as a readout. Three rounded corners and a
        tail hanging off the fourth, leaning down and to the left towards the face it comes
        out of, which is the shape everybody has been reading since before there were screens.

        The tail hangs off the *left* because that is where he stands, and off the bottom rather
        than out of the side because the bubble's left edge moves with the sentence: a tail on
        the side would point straight at him under a long caption and into the picture under a
        short one, while one under the corner leans the same way whatever the line says.

        It is drawn with an edge, which the slab it replaces did not need. A dark fill only
        has a shape where there is something behind it to differ from, and behind this there is
        as often as not a black picture - an unlit room, or no camera at all - where a bubble and
        a rectangle are the same invisible dark patch. The edge is what makes the shape survive
        its own background, and it is the faintest phosphor on the panel because the words inside
        it are what anybody is actually reading.

        Drawn flat rather than through :func:`eye.smoothed`, like every other filled shape on
        this panel - the tag, the menu card, the tab cells. Only the strokes are supersampled,
        because a stepped hairline reads as a fault and a stepped edge on a slab does not.
        """
        tip = (x - self.caption_lean, bottom + self.caption_tail)
        root = (x + self.caption_root, bottom)
        fill = (*SCREEN, PLATE_ALPHA)
        edge = (*GREEN_DIM, BUBBLE_EDGE_ALPHA)
        line = max(1, self.line // 2)
        d.rectangle([x, top, right, bottom], fill=fill, outline=edge, width=line)
        # The tail, then the bottom edge between its corners wiped back to fill, then its own two
        # sides - which is what makes the two shapes one silhouette rather than a box with a
        # pennant taped under it. ImageDraw writes rather than composites, so the wipe is a wipe.
        d.polygon([(x, bottom), root, tip], fill=fill)
        d.line([(x, bottom), root], fill=fill, width=line)
        d.line([root, tip], fill=edge, width=line)
        d.line([tip, (x, bottom)], fill=edge, width=line)

    # ---- the two switches ----

    def _draw_switch(
        self, d: ImageDraw.ImageDraw, name: str, state: str, halo: tuple, pressed: bool
    ) -> None:
        """One switch, sunk through the small bracket's rail: a well, a bezel, and a glyph.

        Three appearances, and they have to stay distinguishable without a word between them.
        At rest it is a dark well with a chrome bezel and a phosphor glyph. While a session is
        up the microphone fills in and takes the state's colour, which is now the whole of what
        that control says about itself - the row it came from said GO TO SLEEP, and there is
        nothing to read here. Under a thumb it inverts, which is the only feedback a touchscreen
        with no travel can give.

        The bezel is drawn as five rings stepping down the rail's own profile rather than as one
        stroke, so a switch reads as the same piece of metal the bracket is made of. It is the
        cheapest way to make two discs sitting on a diagonal look bolted through it instead of
        parked on it.
        """
        cx, cy = self.switches[name]
        r = self.btn_r
        live = name == "wake" and session_up(state)
        ink = halo if name == "wake" and state != IDLE else GREEN_MID
        glyph = INK if pressed else ink
        d.ellipse([cx - r - 2, cy - r, cx + r + 2, cy + r + 3], fill=(0, 0, 0, 120))
        d.ellipse([cx - r, cy - r, cx + r, cy + r],
                  fill=(*halo, 255) if pressed else (*SCREEN, SWITCH_ALPHA))
        if not pressed:
            for step in range(5):
                edge = r - step
                d.ellipse([cx - edge, cy - edge, cx + edge, cy + edge],
                          outline=(*self._rail_colour(1.0 - step / 6.0), 255), width=2)
            d.ellipse([cx - r + 2, cy - r + 2, cx + r - 2, cy + r - 2],
                      outline=(*mix(GREEN_MID, GREEN, 0.5), 255), width=2)
        radius = round(self.btn_r * 0.5)
        if name == "shutter":
            self._glyph_aperture(d, round(cx), round(cy), radius, glyph)
        else:
            self._glyph_mic(d, round(cx), round(cy), radius + 2, glyph, state if live else IDLE)


    def _glyph_aperture(self, d: ImageDraw.ImageDraw, cx: int, cy: int, r: int, c: tuple) -> None:
        """Take a photo now. A six-bladed aperture, swept the one way round."""
        stroke = max(2, round(3 * self.scale))
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(*c, 255), width=stroke)
        for blade in range(6):
            outer = math.radians(blade * 60)
            inner = outer + math.radians(60)  # the sweep is what makes it read as an iris
            d.line(
                [
                    cx + r * math.cos(outer),
                    cy + r * math.sin(outer),
                    cx + r * 0.34 * math.cos(inner),
                    cy + r * 0.34 * math.sin(inner),
                ],
                fill=(*c, 255),
                width=stroke,
            )

    def _draw_invite(self, d: ImageDraw.ImageDraw, phase: float) -> None:
        """Breathe the microphone switch, because it is the only thing left to do.

        Drawn per frame over the switch the base baked at rest, which is why the well goes down
        first and the bezel and the glyph on top of it: the base's own copies are underneath, and
        this covers them.

        It swells towards the accent rather than up the green, so the glyph and its bezel change
        colour and not just brightness - which is the point of the accent everywhere else on this
        panel, and here it is also a promise: the switch wears the colour the whole screen turns
        when you press it. The one white thing on a sleeping panel is the way off it.

        It matters more than it did. The control this replaces had the words WAKE UP written
        across a third of the panel and the breath was a flourish on top of them; this one is a
        36 px microphone in a corner with nothing written anywhere, so the breath is now most of
        how anybody finds it.

        Only while he is asleep proper. Not on a fault - a red panel with a green button
        beckoning at you is a machine asking to be prodded rather than read, and the line under
        the picture is where a fault has something to say.
        """
        cx, cy = self.switches["wake"]
        r = self.btn_r
        swell = breath(phase, WAKE_PERIOD_S)
        colour = mix(GREEN_MID, WHITE, WAKE_GLOW * swell)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(*SCREEN, SWITCH_ALPHA))
        for step in range(5):
            edge = r - step
            d.ellipse([cx - edge, cy - edge, cx + edge, cy + edge],
                      outline=(*self._rail_colour(1.0 - step / 6.0), 255), width=2)
        d.ellipse([cx - r + 2, cy - r + 2, cx + r - 2, cy + r - 2],
                  outline=(*colour, 255), width=2)
        self._glyph_mic(d, round(cx), round(cy), round(r * 0.5) + 2, colour, IDLE)


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

    def _draw_ring(self, d: ImageDraw.ImageDraw, colour: tuple, level: float) -> None:
        """The ring around the microphone, which swells with your voice as the eye's pupil does.

        Drawn per frame rather than baked with the switch, for the obvious reason that it is the
        only part of that corner with anything to say between one frame and the next. It sits
        *outside* the switch's own bezel now rather than inside its glyph: the switch is a disc
        with a machined edge, and a ring drawn within that edge would be read as part of it and
        would stop being a meter.
        """
        cx, cy = self.switches["wake"]
        ring = round(self.btn_r + 5 * self.scale + 7 * self.scale * max(0.0, min(1.0, level)))
        d.ellipse(
            [cx - ring, cy - ring, cx + ring, cy + ring],
            outline=(*colour, 255),
            width=max(1, self.line // 2),
        )

    def _glyph_mic(
        self, d: ImageDraw.ImageDraw, cx: int, cy: int, r: int, c: tuple, state: str
    ) -> None:
        """Start or stop the agent. Hollow while it is down, filled in while it is listening.

        A microphone rather than the eye that was here: what this tab starts is a conversation,
        and the camera it also opens is the shutter's business - the aperture beside this shows
        Cyclops a picture, and the panel behind both is already a viewfinder. An eye over the word
        SESSION said the wrong one of the two things this box does, and the eye is a face in the
        opposite corner now rather than a control at all.

        Filled is the whole state indicator, and there is no word beside it any more, so it has
        to carry the whole message at 36 px on a panel seen from across a bench: an outline that
        gained a detail when live would read as neither.
        """
        stroke = max(2, round(3 * self.scale))
        live = awake(state)

        # Four parts, and they must not overlap. The first cut of this made the head 1.84 r tall,
        # which left the cradle and the stem drawn straight through it - at 32 px that reads as a
        # bullet with fins rather than as a microphone. The head gets the top half and stops; the
        # cradle's arms rise past its waist without reaching its crown; the stand has the rest.
        half = round(r * 0.44)
        head = [cx - half, cy - round(r * 0.94), cx + half, cy + round(r * 0.19)]
        if live:
            d.rounded_rectangle(head, radius=half, fill=(*c, 255))
        else:
            d.rounded_rectangle(head, radius=half, outline=(*c, 255), width=stroke)

        # An arc over an ellipse wider than the head, so its arms end beside the head and not
        # under it. Drawn 0 to 180, which is the lower half.
        arms = round(r * 0.69)
        d.arc(
            [cx - arms, cy - round(r * 0.69), cx + arms, cy + round(r * 0.56)],
            start=0, end=180, fill=(*c, 255), width=stroke,
        )
        d.line([cx, cy + round(r * 0.56), cx, cy + round(r * 0.88)], fill=(*c, 255), width=stroke)
        # The foot is what stops it reading as a pill hung on a hook.
        foot, base = round(r * 0.38), cy + round(r * 0.88)
        d.line([cx - foot, base, cx + foot, base], fill=(*c, 255), width=stroke)

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
# remaining lever on how much of the room you can actually make out - see the README.
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
