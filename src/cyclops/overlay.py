"""The kiosk's chrome: a Pip-Boy-style terminal bezel drawn over the live camera frame.

OpenCV can only draw Hershey stroke fonts, which look like a 1980s oscilloscope. Everything
here is rendered into one RGBA layer with PIL instead - real TrueType text, anti-aliased
shapes, a numpy-built halo and scanline field - and handed to :mod:`cyclops.kiosk` as a numpy
array to alpha-blend onto the frame. Geometry doubles as the hit-test map: every interactive
element returns its rectangle, so a tap can be resolved without a second layout.

The layout is two corner mounts, a status pod and a picture. A mount's rail comes in square to
one panel edge, ramps across the corner at 45 degrees and lands square on the other, and the two
of them carry the three controls; the pod is that same shape turned inwards, hanging off the
middle of the top edge with the readouts in it. This replaced a strip and a three-cell tab row
that between them covered 29% of the panel - the chrome here covers 16%, and the middle of
the screen, which is what somebody holding a camera down a pipe is actually looking at, is
nothing but picture and a four-arc reticle on the lens axis.

What made that affordable was giving up the words. SNAP, WAKE UP and GO TO SLEEP were what
forced the controls into a row across the bottom: a cell has to be as wide as its label. Two
dials and a face need a disc each, and a disc can go in a corner. What the words used to say is
said by the ring in the button, by the border's colour, and by the line under
the picture - which can say "searching the web…" where a word could only say SEARCH. The state
word went the same way and for the same reason: three things were already saying it better.

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
# the signal meter, the session clock, the caption's marker, and the microphone on the switch
# that is holding the session open. The rest of the chrome stays green, because a
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

# The reticle: four arcs on the lens axis and nothing else. It was a cross with graduations for
# about an hour, which is exactly as long as it took somebody to say it looked like a gun sight -
# and they were right. A broken ring says "lens" and says nothing else.
RETICLE_R = 0.1625  # 78 px at 480
RETICLE_ARC = 58.0  # degrees of ring per quadrant; the rest is gap

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
        # It hangs *below* the reticle rather than across it. The four arcs are the one thing on
        # this panel drawn over the middle of the picture, and a bubble crossing them turns two
        # deliberate marks into one accident - so the bubble's own top edge, at its tallest, is
        # what clears them, and the tail then lands beside his equator rather than above it.
        self.caption_bottom = (
            self.height // 2 + self.reticle_r + max(4, round(8 * scale))
            + CAPTION_LINES * self.caption_h
        )
        reach = self.shoulder + self.caption_nose
        drop = self.caption_bottom + self.caption_tail - self.eye[1]
        across = math.sqrt(reach * reach - drop * drop) if abs(drop) < reach else reach
        self.caption_left = math.ceil(self.eye[0] + across) + self.caption_lean
        # ...and it stops short of *both* switches rather than at the frame's own padding. Now
        # that the line sits low enough to clear the reticle it is level with the shutter as well
        # as the microphone, and a bubble clipping the top of a control is a bubble that has made
        # the control look broken.
        self.caption_right = min(
            self.width - self.pad - round(34 * scale),
            min(cx for cx, _ in self.switches.values()) - self.btn_r - round(12 * scale),
        )
        self.caption_y = self.caption_bottom - self.caption_h / 2  # the bottom line's middle

        self._halo = halo_alpha(width, height)
        self._filter = self._build_filter()
        self._backdrops: dict[int, Image.Image] = {}
        self._chromes: dict[int, Image.Image] = {}
        self._plate = self._build_plate()
        self._chrome_base = self._build_chrome()
        # One engine per window size: it owns the geometry, and it remembers which mood it is
        # easing out of, which is why it is built here and not per frame.
        self.engine = EyeEngine(self.eye_r, self.line, SCREEN, MOODS[IDLE])
        self._bases: dict[tuple[str, bool, str], Image.Image] = {}
        # The moving half of each instrument, one tile per appearance it can have. A level moves
        # when a finger moves it and a temperature moves once every five seconds, so at 25 frames
        # a second almost every frame asks for the tile the frame before it already built.
        self._knobs: dict[tuple[int | None, bool], Image.Image] = {}
        self._needles: dict[tuple[int | None, str, bool], Image.Image] = {}
        self._dots_w = self.font_caption.getlength("." * CAPTION_DOTS)
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

        His plate is what backs him instead, and it is the same all the way round; without this
        the wash's own edge would run across his face as a tide line. The switches keep the wash
        under them, because their wells are opaque enough not to care and cutting two more holes
        in a mask is two more edges to land in the wrong place.
        """
        plate = Image.new("L", (self.width, self.height), 0)
        d = ImageDraw.Draw(plate)
        for bracket in (*self.brackets.values(), self.pods[tags]):
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
        base = _over(base, GREEN, np.full(shape, TINT_ALPHA, dtype=np.float32))
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
        for bracket in self.brackets.values():
            self._draw_bracket(layer, bracket)
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
        which is the one thing this layout is spending its corners to avoid. The pod has none:
        webbing braces a corner against a load, and a module hanging off the middle of an edge
        has no corner and nothing to brace.
        """
        d = ImageDraw.Draw(layer)
        corner, a, b = bracket.corner, bracket.spine[0], bracket.spine[-1]
        for i in range(RIB_N if corner else 0):
            t = 0.15 + 0.075 * i
            d.line(
                [(corner[0] + (a[0] - corner[0]) * t, corner[1] + (a[1] - corner[1]) * t),
                 (corner[0] + (b[0] - corner[0]) * t, corner[1] + (b[1] - corner[1]) * t)],
                fill=(*mix(GREEN_MID, SCREEN, 0.5), 170), width=max(1, round(4 * self.scale)),
            )
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
                    start=start, end=start + RETICLE_ARC, fill=linear(GREEN_MID, 195),
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
        volume: int | None = None,
        temp_c: float | None = None,
        turning: bool = False,
    ) -> np.ndarray:
        """Draw the whole chrome for this frame and return it as an RGBA numpy array.

        ``phase`` is a monotonic clock in seconds, and the only argument here that is not about
        what the panel is showing but about *when*. It is passed in rather than read here so a
        frame is a pure function of its arguments and the caption's animation can be tested
        without a clock - the same shape as ``flash``, which the kiosk has always computed.

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
        self._draw_caption(d, state, halo, detail, phase)
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
                           voice=0.0, gaze=0.0, dart=0.0)
        self.engine.paint(layer, *self.eye, mood, phase, level)
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
        same bargain the caption strikes with its dots. Packed instead, the clock would slide
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
