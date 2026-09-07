"""Cyclops' eye: the one thing on the panel that is a face rather than a readout.

The boot mark brought to life. ``assets/splash.png`` is a diaphragm inside concentric HUD rings,
and this is that drawing with the rings turning, the iris breathing, the core lit and the whole
thing tinted by whatever the box is doing - so a glance at the bottom-left corner answers "is
he there, what is he up to, and is he looking at me" without reading a word.

Three things make him read as alive rather than as a spinner, and they are worth naming because
each one was arrived at by trying the alternative first:

*The core.* The middle of him is a disc of radial filaments converging on a hot centre, and the
falloff is free - the threads are radial, so their density goes as 1/r and the centre glows
because it is *crowded* rather than because anything was shaded. A dark hole was tried and is
the reason the old face never looked back at you: the one part of an eye that has to have
something in it was the one part with nothing in it.

*The gaze.* He looks around. The whole optic - blades, bezel and core - slides inside a shell
that never moves, so it reads as an eye in a socket. Two things sell it and both are free: the
shell staying put, and the brow staying put with it. The brow is a highlight on the outer glass,
and a highlight does not travel with the thing underneath it, so the core slides out from under
its own catchlight.

*The asymmetry.* A row of indicators at one clock position, a lug that answers nothing, a seam
at an angle that agrees with none of the rings. Symmetry reads as a printed badge; one thing out
of step reads as a part.

Everything the eye does is a :class:`Mood`: fourteen numbers and a colour. The engine is only the
mechanism that draws them, which is what makes the eye tunable - a state is not code here, it is
a row in a table (``overlay.MOODS``), and a new one costs a line. The parameters are meant to be
pushed around: ``tools/eye_sheet.py`` renders every mood over a strip of time, so a change can be
looked at rather than argued about.

Nothing in here knows about the panel, the palette or the camera. It takes a radius, a colour and
a clock, and draws. That is what lets the preview harness import it on its own, and it is why the
state table lives in :mod:`cyclops.overlay` where the phosphor colours are.

Adding a part is meant to be cheap. If one of :class:`Pen`'s shapes already draws it, it is a
radius and a brightness among the constants below plus one line in :meth:`EyeEngine._shell` or
:meth:`EyeEngine._optic`; if it needs a new shape, it is also one method on ``Pen``. Anything
that must move with his gaze goes in ``_optic`` and is drawn through the shifted pen it is
handed; anything that is the socket goes in ``_shell``; the brow is neither, and is drawn last
on the unshifted pen. What is *not* cheap is a *second* instance of an existing part at another
radius - the part methods read their own brightnesses, so that means turning those into
arguments first. Nothing here is arranged for it, because nothing has wanted it yet.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from copy import copy
from dataclasses import dataclass, replace

import numpy as np
from PIL import Image, ImageDraw

SUPERSAMPLE = 4  # how many times over-size he is drawn before being averaged onto the panel.
# PIL will not anti-alias anything, and he is nothing but curves: every arc, tick and blade in
# here is a hard-edged run of pixels, which on a circle 120 px across reads as a staircase rather
# than as a ring. Drawing him four times over and shrinking is the whole fix. Four is where the
# stairs stop showing at this size; eight costs four times as much and looks the same from a
# bench, and one puts back exactly the pixels this drew before any of it existed.

GAMMA = 2.2  # the panel's, and the reason the shrink happens in light rather than in numbers.
# Averaging two 8-bit values does not average two brightnesses. A pixel half covered by a bright
# line, averaged as numbers, comes out at half its value - which is under a quarter of its light
# - so a ring made mostly of half-covered pixels goes grey and the smoothing reads as a fade. The
# tile is drawn in linear light and converted back on the way out instead, which is what keeps a
# smoothed hairline as bright as the jagged one it replaces.
LINEAR = tuple(round(255 * (v / 255) ** GAMMA) for v in range(256))
SRGB = np.array([round(255 * (v / 255) ** (1 / GAMMA)) for v in range(256)], dtype=np.uint8)

# ---------------------------------------------------------------- the shape, outside in
#
# All of these are fractions of the eye's radius, so the whole drawing scales with him and the
# preview harness can render him at any size without a second set of numbers. What survives being
# 120 px across on a panel seen from a bench is the limit on every one of them: the splash mark
# has seven concentric rings and at this size the inner three turn to porridge.

# Outside in, roughly - the bands overlap, because a machine's parts do. Every number the
# drawing uses lives here: if a literal appears in _shell or _optic it is a bug, because the
# whole point of this block is that the eye can be retuned without reading the code.
RIM = 1.00  # the outer ring. The shoulder arc that joins him to the tab row sits outside it
RIM_W = 1.6  # ...and how wide, as a multiple of the panel's line

CASTLE = 0.885  # a ring that steps radially inward for a short arc, five times round
CASTLE_N = 5
CASTLE_JOG = -0.055  # inward, because a notch that steps out reads as a cog tooth
CASTLE_ARC = 30  # degrees of each step
CASTLE_AT = 18  # where notch zero sits before the ring starts turning
CASTLE_W = 1.2  # as a multiple of the thin line
# The splash mark's signature and the one thing that stops a ring reading as a ring: a true
# circle is a drawn shape, and a circle with square jogs cut into it is a *part*.

VANE_IN, VANE_OUT = 0.735, 0.795  # chunky stator blades: a shroud, not a knurl
VANE_N = 18
VANE_ARC = 4.2  # half-width in degrees, so a vane is a plate rather than a line

DATUM = 0.79  # a hairline ring, there to carry the asymmetry rather than to be seen
GREEBLE_AT = 128  # where the row of indicators sits on it
GREEBLE_N = 4
GREEBLE_TALL = 0.028  # half the height of one indicator
GREEBLE_SPREAD = 5.4  # degrees between them
GREEBLE_BAR = 2.0  # the bar above the row, in multiples of GREEBLE_TALL...
GREEBLE_BAR_ARC = 13  # ...and how far round it reaches
LUG_AT = 292  # ...and a lug somewhere else entirely, answering nothing
LUG_ARC = 5.0
LUG_THICK = 0.03
SEAM_AT = 43  # a seam at an angle that agrees with none of the rings
SEAM_IN = 0.70  # running from here out to the castellated ring

KNURL_IN, KNURL_OUT = 0.685, 0.735  # a grip band, and only part of the way round. It runs in
# the seat the panel turns into the well behind him (`SEAT_IN` in :mod:`cyclops.overlay`), which
# is clear of everything that follows his gaze: the diaphragm reaches 0.60 and the gaze carries
# it another 0.105, so a mark inside 0.705 is a mark a blade tip sweeps across. It sat at 0.617
# for a round, under the tips and over a band of pale brass, and the two together took the one
# ring on him you can watch turn and froze it - measured, at r60-65, from a temporal spread of
# 8.4 levels to 0.3. Bright on a dark seat, and out where nothing else is moving.
KNURL_N = 22
KNURL_FROM, KNURL_SPAN = 196, 214

DOTS = 0.760  # ...and the dotted ring out where the stator can pass in front of it. It used to
# sit at 0.72, in the open; a row of marks appearing and going again behind eighteen vanes says
# "turning" harder than the same row with nothing in front of it ever did.
DOT_N = 16

IRIS = 0.60  # the diaphragm's outer edge, and everything inside it moves with his gaze
LEAF_N = 8
LEAF_TWIST = 12  # where leaf zero sits. Fixed, and deliberately so: a real diaphragm's blades
# move when the aperture moves and not otherwise, and this set already travels with his gaze -
# spinning them as well would fight that, which is the one motion here doing narrative work.
HATCH = 0.085  # the hatch pitch inside a leaf, as a fraction of the iris
HOLE_MIN = 0.16  # the hole the leaves leave, as a fraction of the iris, shut...
HOLE_MAX = 0.78  # ...and wide open. The core can only be as big as this, which is what makes a
# stare a wide hot core and a fault a narrow one without a single number per state.
BEZEL = 1.14  # the ring the core is set into, as a multiple of the hole
BEZEL_W = 1.1
INDEX_IN, INDEX_OUT = 0.605, 0.700  # the brass ring the diaphragm is set in, as fractions of
# him. It sits immediately outside the blades' own edge and travels with them, because it is the
# housing they are mounted in and not a mark on the socket behind. That is also what keeps it
# clear of the band the "rings turn" tests read, and it is what a static one got wrong the first
# time this was tried: pinned to the socket it lay across the blade tips wherever he looked, and
# took the motion out of the one band that had it.
INDEX_TICKS = 60  # one cut every six degrees...
INDEX_MAJOR = 5  # ...and every fifth of them run the full width, which is what makes a ring of
# marks read as a scale rather than as a knurl
INDEX_TICK = 0.45  # how far across the face a minor cut reaches, as a fraction of the width
INDEX_LIT = 0.34  # the flat of the brass where it is turned away from the lamp...
INDEX_TOP = 0.50  # ...and where it faces it. Two arcs rather than a grade, for the reason
# LEAF_FACE gives: a gradient inside this tile is a numpy pass over half a million pixels a frame
INDEX_FROM = 200  # where the lit arc starts and how far it runs, set on the panel's own lamp -
INDEX_SPAN = 160  # up and to the left - so the ring agrees with every bevel outside him
INDEX_CUT = 0.16  # the cut itself. A groove in brass is darker than the face it is cut into
INDEX_LIP = 0.60  # the polished outer edge, one hair wide: a machined ring reads from its edge
# before it reads from its face
BRASS_EYE = (178, 128, 38)  # what the ring is made of, and the one part of him that is not his
# tint. Darker than the collar's brass outside him (overlay.BRASS) because it is seen through the
# same glass as the phosphor, and because the face's colour is measured over what it lights: at
# INDEX_TOP this lands near 160 of the 200 that measurement starts counting at, so the ring is a
# brass part on a green eye rather than a warm eye.
CORE_N = 52  # filaments in the core at full size - see Pen.core for why it is a ceiling
CORE_HUB = 0.10  # the filaments start here rather than at a point, or they pile up into
# a blob at the centre and the falloff stops being a falloff
CORE_EDGE = 0.97  # ...and stop just short of the rim, so the rim reads as a rim
CORE_RINGS = (0.44, 0.72)  # two faint circles across them, which says "lens" and not "fan"
HOT = 0.30  # the hot middle, as a fraction of the core...
SPARK = 0.13  # ...and the brightest part of that again
SPARK_FLOOR = 1.5  # ...but never a smaller radius than this in panel pixels. sqrt(2) is the
# bound that makes the spark cover a whole pixel at any sub-pixel position; see Pen.core.
BROW_AT = 0.40  # the catchlight, on the outer glass
BROW_FROM, BROW_TO = 198, 256
BROW_W = 1.2

# ---------------------------------------------------------------- how present each part is
#
# Against the mood's own tint, before ``Mood.rings`` scales all of them. The hierarchy is the
# reference's rather than a HUD's: the machine is cold and the core is hot. That is a real trade
# and not a preference - the rim is the outermost thing on him and used to be the brightest, and
# turning it down is what makes the core read as the only lit part of a dark instrument.
RIM_LIT = 0.44  # the seam between his glass and the bezel's steel. Not the brightest ring on him
# - it was 0.60 once and read as a drawn green stroke bounding the whole instrument, which is the
# first thing the eye landed on and the last thing a bezel does. What holds the glass is the
# collar's machined lip, outside his rim and in metal; this is the line of phosphor under it, and
# it has to be bright enough to be the boundary between a dark cavity and a lit piece of steel.
CASTLE_LIT = 0.50
VANE_LIT = 0.34
DATUM_LIT = 0.34
GREEBLE_LIT = 0.75
GREEBLE_DARK = 0.26  # the one indicator in the row that is out. A row all lit is a decoration
GREEBLE_BAR_LIT = 0.40
LUG_LIT = 0.55
SEAM_LIT = 0.35
DOT_LIT = 0.44
KNURL_LIT = 0.34  # a lit serration, read against the dark seat it runs in. It was 0.05 for a
# round, on the theory that it was a cut in brass; a dark mark on a pale band is the one way of
# drawing this that has no contrast to move with, and it cost the ring its motion.
LEAF_LIT = 0.012  # the flat of a blade at its hinge, out at the iris...
LEAF_FACE = 0.085  # ...and at the aperture edge, where it has turned into the lamp. Stepped
# rather than graded - `LEAF_BANDS` chords across the same segment, each one flat, because a real
# gradient inside this tile is a numpy pass over half a million pixels every frame. Three steps
# and the hatch across them is four values on a plate that had one.
#
# Both are a quarter of what they were, and dropping them is the whole of this round's repair.
# The ramp was right and the level was not: at 0.13 -> 0.36 the plate came out a mid grey-green,
# the same drawing read 20% weaker everywhere, and two motion critics preferred the flat dark
# blade it replaced. A diaphragm is a stack of blackened leaves in a lens - it is nearly black,
# and everything you can see of it is the light along its edges. Keep the ramp, drop the floor.
LEAF_BANDS = 3
HATCH_LIT = 0.235  # ...and the hatch across it, which is the only thing telling two blades apart.
# Brighter than any of the three bands, now they are all dark: what shows on a blackened leaf is
# where the light catches its rolling marks, and a *darker* line on a plate this dark is nothing
# at all. It was 0.08 against a plate of 0.36, which is a cut - the right mark on the wrong
# plate, and worth naming because the pair have to be retuned together or neither reads.
EDGE_LIT = 0.92  # the bright chord where one blade lies over the next
EDGE_SHADE = 0.02  # ...and the dark line beside it on the blade's own face: the blade's
# thickness, seen edge-on. A lit line on its own is a drawn edge; a lit line with a dark one
# against it is a bevel, and a plate with a bevel has a thickness.
EDGE_BEVEL = 1.0  # how far onto the blade that facet lies, in multiples of the thin line
IRIS_LIT = 0.95
BEZEL_LIT = 0.55
BEZEL_HI = 0.85  # the bezel where the panel's lamp lands on it, up and to the left. On the
# moved centre with the rest of the optic, and deliberately so: the brow is a highlight on the
# glass and stays put, this is a highlight on the metal under it and travels - two reflections
# parting company is what says there is a pane between them.
BEZEL_HI_FROM, BEZEL_HI_TO = 204, 294  # PIL degrees, centred where the lamp is
CORE_FLOOR = 0.10
THREAD_LIT = 0.60
CORE_RING_LIT = 0.26
CORE_EDGE_LIT = 0.90
BROW_LIT = 0.55
HEAT = (255, 255, 255)  # the top of the tube. The core is the one place the eye may go paler
# than its own tint, and it is why it reads as hot rather than merely bright.
HOT_MIX = 0.55  # how far the hot middle is pushed towards HEAT...
SPARK_MIX = 0.85  # ...and the spark inside it. Both are deliberately short of 1.0: a core that
# goes fully white stops wearing the mood's colour, and the panel says severity in colour.

# ---------------------------------------------------------------- what turns, and how
#
# Counter-rotation, as multiples of the mood's spin. Rings that all turn together read as one
# disc; rings that disagree read as a mechanism. Four channels, and the wander offsets each of
# them differently - see :func:`wander`.
KNURL_SPIN = 1.0  # the knurl. Not the blades: see LEAF_TWIST for why they hold their angle
CASTLE_SPIN = -0.62  # the castellated ring, and the seam
DOT_SPIN = 0.31  # the dotted ring, and the greeblies riding it - which is most of what tells
# you he is running at all, because it is the one mark on him you can follow
SCAN_SPIN = 2.3  # the sweeping highlight, when a mood asks for one...
SCAN_DIM = 0.42  # ...and how far the rim is turned *down* underneath it while it sweeps
RINGS = 4  # the four channels above, and what `ring` indexes in wander(). The vanes and the
# diaphragm are deliberately not among them - not everything on him turns.

RING_GEAR = 34.0  # degrees a ring indexes round per unit of sideways gaze, times its own share.
# The ring set is geared to the eye: when he turns to look at something the mechanism carrying
# him indexes with it, forward on the knurl and back on the castellation, and it stays there for
# as long as he holds the place. This is the answer to a measurement rather than a flourish -
# three critics counted the castellated ring at -4.5 degrees a frame, EVERY frame, variance zero,
# and one of them put it best: a still frame around a moving thing cannot itself be the thing
# that moves most steadily. The alternative on the table was more `sway`, which is a second
# unrelated motion laid over the first; this makes the loudest motion in the picture a
# *consequence* of the thing you are supposed to be watching. Sideways rather than the full
# vector because a ring turns about one axis, and it takes the sign with it: a mood that looks
# down at its own hands indexes the set the other way from one that looks up at the pod.
# The scan arc is left out of it - a rotor sweeps at its own rate whatever the gear train does.

# ...and the wander on top of those, for a mood that asks for one - see `Mood.sway` and
# :func:`wander`. Fixed multiples of one rate are a gear train: it turns, but every arrangement
# it reaches it has reached before, and a few seconds in, the eye stops reading it as alive.
TAU = 2.0 * math.pi
SWAY_S = 16.3  # the shorter of the two periods the wander is built out of. The longer is the
# golden ratio times it - see BLINK_DRIFT, of which this is the same argument - so their ratio
# is irrational, the two never come back into step, and no arrangement of the rings is ever
# held twice. Both are long against a glance at the panel, which is the point: what you see in
# any few seconds is rings speeding up, falling back and turning over, not a cycle.

# ---------------------------------------------------------------- looking around
GAZE_SHIFT = 0.130  # how far the optic travels at full gaze, as a fraction of the eye. It was
# 0.105, which is 9 px of spark travel at his panel size and is not a look - across six seconds
# on a seven-inch panel it reads as a still picture with a mechanism turning behind it. What caps
# it is what the diaphragm has to slide past: the blade tips reach IRIS + this, and the ring set
# they cross starts at KNURL_IN. Crossing it *now and then* is parallax and is what makes the
# move read; sitting on it is what froze the knurl the round it lived at 0.617, so the ceiling is
# the vane ring at VANE_IN, which the tips must not reach even at the hunt's 0.90 of full gaze.
GAZE_LEAD = 0.65  # how much further the hot middle goes than the core around it, as a fraction
# of the core. It is what puts a pupil inside the lens rather than dragging one flat disc about,
# and it is where most of the travel now comes from: the plate is fenced in by the ring set and
# the pupil is not, so the eye rolls inside its own lens the way a real one does in its socket.
LEAD_EDGE = 0.86  # ...but no further than this: the hot middle's own outer edge, as a fraction
# of the core's radius, so the pupil rides inside the lens and never over the rim drawn at
# CORE_EDGE. It only bites on the moods that look hardest - the hunt and the stare - where the
# lead alone would carry the pupil out over the blades, and it bites inside one saccade.
DRIFT_S = 7.3  # the shorter of the gaze's two drift periods; the longer is the golden ratio
# times it, for the reason every other pair of periods on this panel is irrational.
SACCADE_S = 0.13  # how long a dart takes. Shorter than this is indistinguishable from a dropped
# frame; longer and it reads as a pan rather than as a flick.

# ---------------------------------------------------------------- the places he looks
#
# Names in here and nothing else. *Which direction* each of them lies in depends on where he is
# bolted and what is on the panel beside him, and this module knows neither - see the module
# docstring, and see tools/eye_sheet.py, which is the reason it must not learn.
# :mod:`cyclops.overlay` resolves the three that are really on the screen off its own geometry
# and hands the map to :class:`EyeEngine`.
#
# AHEAD is (0, 0), and that it is the anchor for nearly every mood is the whole of the model: the
# pupil dead centre is him looking out of the glass at whoever is standing there. Every other name
# is something he may take his eyes off you for. This is what replaced a walk over arbitrary
# angles, and the difference is not subtle - the old one had a reach *floor*, so the one direction
# it could never manage was straight at you.
AHEAD = "ahead"
FRAME = "frame"  # the middle of the picture he is sitting on: the camera's own reticle
WORDS = "words"  # the line of text under it, where his own caption is printed
DIALS = "dials"  # the readouts along the top
WORK = "work"    # down at his own hands. Nothing on the panel is there, and that is the point
AWAY = "away"    # off past the edge of the thing entirely

# The reference layout's answers, so the preview harness renders the eye the panel actually has
# rather than a plausible one. The overlay recomputes the three geometric ones for its own window
# size and overrides these; a test keeps the two in step, because nothing else would notice the
# day the bracket moves.
LANDMARKS: dict[str, tuple[float, float]] = {
    AHEAD: (0.0, 0.0),
    FRAME: (0.911, -0.413),   # up and to the right of him, at 800x480
    WORDS: (0.952, 0.305),    # down and to the right, at the head of his own line
    DIALS: (0.644, -0.765),   # steeply up, at the pod
    WORK: (-0.55, 0.84),      # down and to his left, off the panel: the bench
    AWAY: (-0.80, -0.60),     # up and to his left, at nothing at all
}

DART_PICK = 0.4142135624  # sqrt(2) - 1, and a *second* irrational on purpose.
# Two walks decide a glance: whether this window is spent away from the anchor, and which place it
# is spent on. Walk them both on the golden ratio and the second is a function of the first -
# frac(3*phi*w) is frac(3*frac(phi*w)) - so where he looks becomes a function of whether he looks
# at all, and the far end of the list starves. Measured over 20000 windows at dart 0.26 with four
# places: on 3*phi the fourth gets 199 glances against about 1667 each for the other three; on
# this, 1306/1294/1306/1295. 1, phi and sqrt(2) are rationally independent, which is the whole of
# why. Not a PRNG and not hash(), for the reason :func:`blink` gives.

DWELL_MIN, DWELL_MAX = 2.4, 3.2  # how long he holds a place once he has arrived at it, in
# SECONDS, and the one number this round turns on. It was a share of the window - 0.42 to 0.72 of
# `dart_s` - which is the thing that made the old gaze read as a twitch: a share scales with the
# span, so a mood asking for a longer, more deliberate rhythm got a *shorter* look out of it, and
# at listening's window the hold came out 1.1 to 1.9 s. Measured on the render at one frame a
# second: out 21.5 px on one frame, back 18.1 px on the next, and nothing at the far end at all.
# That is a flinch, or a dropped frame. A creature looks, HOLDS, and then comes back, and the
# hold is the whole of what says it was looking rather than glitching. Under two seconds there is
# nothing to see; past four he has stopped attending to you and is staring at something else.
DART_FLOOR = 3.0 * SACCADE_S  # 0.39 s, and the shortest window a mood may ask for: a saccade
# out, a beat at the far end, and a saccade home. A hold is clipped to `dart_s - SACCADE_S` so
# that the way back always fits inside the window - what comes out is continuous whatever the
# table says, and the next window always starts with him arrived. Asserted on the table rather
# than clamped here, because a mood that asks for a window this short wants telling.
FOCUS = 2.0  # how much harder the first place a mood names pulls than the ones after it. `look`
# is a priority and not a menu - the thing that is happening gets most of his attention and the
# rest get what is left - which is the difference between a face watching something and one
# sweeping a room. Squaring the picking walk gives the first of the hunt's four named places 50%
# of its glances and the last 13%; at 1.0 it is a quarter each, which is a turret on a schedule.

MICRO_S = 1.7   # the tremor's shorter period, the same shape as the drift an order faster...
MICRO = 0.10    # ...and how far a *held* gaze still moves, as a fraction of the full travel. An
# eye that stops moving is a dead one, so this is under every fixation. It has to stay well below
# a peek or it stops being a tremor and becomes the wander this model was written to remove: about
# 1.6 px of spark travel at his panel size, which is a live edge rather than a motion.
MICRO_LIT = 0.20  # the gaze below which the tremor fades out with it rather than snapping off at
# zero. A mood that holds still - the fault, the working face, the tap acknowledgement - has to
# hold *completely* still, and has to get there without a step at the end of the crossfade.

BLINK_S = 0.34  # one blink, down and back up: eight or nine frames at 25 fps. It was 0.22, which
# is five - long enough to be seen running and not long enough to be *caught*. A strip sampled
# once a second lands inside a 0.22 s lid one time in twenty, so three critics in a row looked at
# a face blinking fourteen times a minute and reported that nothing blinks. This still cannot be
# mistaken for a dropped frame, and it now shows up in a still.
BLINK_DRIFT_S = 2.6  # how far a blink may wander inside its window: a blink on a fixed period
# is a status LED, not a creature
BLINK_DRIFT = 0.6180339887  # golden ratio, so no two consecutive gaps come out the same length
LID_LEAD = 0.10  # how far *before* a saccade the gaze-evoked lid starts down - see
# :func:`gaze_blink`. A creature blinks as it changes what it is looking at, and it starts
# closing before it sets off, so the lid is at the bottom while the optic is crossing and on its
# way up as it arrives. Blink, then look. Ahead of the flick rather than on it because a lid that
# shut on arrival would hide the one frame the whole move exists to put on the screen.

MOOD_EASE_S = 0.45  # how long the eye takes to become a different mood. Colour that snaps reads
# as a warning lamp; colour that travels reads as the same creature changing its mind


def mix(base: tuple[int, int, int], other: tuple[int, int, int], amount: float) -> tuple:
    """*amount* of *other* stirred into *base*, so a tinted fill can be one opaque colour.

    The rule everything small on this panel obeys. PIL's ImageDraw *writes* into an RGBA image
    rather than compositing onto it, so a fill at 40% alpha does not come out 40% dimmer - it
    stamps a 40%-opaque patch and the picture behind the panel shows through it. That is fine for
    a whole cell of see-through chrome and no good at all for a ring that has to stay legible
    over whatever the camera happens to be pointed at. Those are mixed towards the screen's own
    black and drawn opaque instead.
    """
    a = max(0.0, min(1.0, amount))
    return tuple(round(b + (o - b) * a) for b, o in zip(base, other, strict=True))


def breath(phase: float, period: float) -> float:
    """0 at the top of a cycle, 1 at the bottom, on a raised cosine.

    The one shape everything alive on this panel moves with. A square wave here reads as a fault
    light rather than as something breathing.
    """
    if period <= 0.0:
        return 0.0
    return 0.5 - 0.5 * math.cos(2.0 * math.pi * (phase % period) / period)


def blink(phase: float, every: float) -> float:
    """1.0 between blinks, 0.0 at the bottom of one. ``every`` of 0 means a mood that never does.

    One blink per window of *every* seconds, but not at the top of it: each window's is offset by
    a different fraction of :data:`BLINK_DRIFT_S`, so consecutive gaps differ and the thing reads
    as a face rather than as an indicator lamp. The offsets come off the golden ratio because it
    is the least rational number there is - the sequence never settles into a period anybody can
    anticipate - and because a multiply and a mod is the whole cost. Not a PRNG, and emphatically
    not ``hash()`` of anything, which is salted per process and would give the box a different
    blink on every boot.
    """
    if every <= 0.0:
        return 1.0
    window = math.floor(phase / every)
    started = window * every + min(BLINK_DRIFT_S, every - BLINK_S) * ((window * BLINK_DRIFT) % 1.0)
    since = phase - started
    if not 0.0 <= since < BLINK_S:
        return 1.0
    return 1.0 - breath(since, BLINK_S)


def wander(
    phase: float, spin: float, share: float, sway: float, ring: int, sway_s: float = SWAY_S
) -> float:
    """Where one ring has got to by *phase*, in degrees: its own steady turn, pushed as it goes.

    *share* is the ring's multiple of the mood's spin - the counter-rotation it would have on
    its own - and *sway* is how hard it is pushed off that. At 0 this is the plain product the
    rings used to turn on, so a mood that does not ask for a wander is untouched.

    The push is two sines whose periods are in the golden ratio (see :data:`SWAY_S`), offset per
    ring so no two are ever doing the same thing at the same time. Written on the *angle* rather
    than on the rate, which is what makes the amplitude here mean something: a term of period P
    contributes ``sway * spin`` degrees a second at its steepest whatever P is, so the two
    periods change how long a surge lasts without changing how hard it pushes.

    The push is scaled by the mood's own spin rather than by the ring's share of it, which is
    the difference between a set that varies and a set that rearranges: proportional wander
    leaves the slowest ring slowest forever, while this lets the dotted ring - a third of the
    knurl's rate - outrun it, fall behind it and turn back under it.

    Offsets walked by the golden ratio for the reason :func:`blink` walks its own by it: it is
    the least rational number there is, so no two rings fall into step. Not a PRNG and pointedly
    not ``hash()``, which is salted per process and would give the box a different eye on every
    boot - and this has to be the same creature each time it wakes up.
    """
    turn = spin * share * phase
    if sway <= 0.0 or spin == 0.0:
        return turn
    short = sway_s if sway_s > 0.0 else SWAY_S
    long = short * (1.0 + BLINK_DRIFT)  # the golden ratio times the short one
    for i, period in enumerate((short, long)):
        offset = ((ring + 1) * (i + 1) * BLINK_DRIFT) % 1.0
        turn += sway * spin * period / TAU * math.sin(TAU * (phase / period + offset))
    return turn


def steady(gx: float, gy: float) -> tuple[float, float]:
    """A gaze vector no longer than 1, so a diagonal is a direction and not 1.41 times as hard.

    Small and separate because getting it wrong is not a small bug: normalising by a magnitude
    that has already been clamped leaves the basis vectors longer than unit, and everything built
    on them - the shift, the lead - comes out scaled by up to two.
    """
    m = math.hypot(gx, gy)
    return (gx / m, gy / m) if m > 1.0 else (gx, gy)


@dataclass(frozen=True)
class Mood:
    """How the eye looks and moves. One per state, and the only thing that differs between them.

    Fourteen numbers and a colour. Every one of them is meant to be pushed around - see the
    module docstring - so none of them is allowed to be load-bearing on its own: an eye with
    every parameter at zero is a dim ring looking straight ahead, not a crash.
    """

    tint: tuple[int, int, int]  # the eye's own colour, whatever the border happens to be doing
    aperture: float = 0.5  # how far the iris stands open at rest, 0 shut .. 1 wide
    swell: float = 0.06  # ...and how much of that the breath gives and takes back
    breath_s: float = 4.0  # seconds per breath. A resting creature is twelve to fifteen a minute
    voice: float = 0.0  # how much further the level on the meter opens it, on top of the
    # breath. Only a mood hearing its *own* sound should touch this: an iris moving with the
    # room is a mouth, not an eye - see overlay.MOODS, where listening pointedly leaves it at 0.
    spin: float = 6.0  # degrees a second the ring set turns; the sign is a direction
    sway: float = 0.0  # ...and how far each ring wanders off that rate, as a multiple of it.
    # 0 is a gear train, every ring locked to every other. Past 1 a ring's wander outruns its
    # own rate and it turns over now and then - which is the whole point of it, since a set that
    # only ever varies its speed still reads as one mechanism running unevenly.
    sway_s: float = SWAY_S  # ...and how long one surge of that wander lasts. Amplitude and rate
    # are separate on purpose - see :func:`wander`, where a term pushes just as hard whatever its
    # period is - so this is the knob for *how often* a ring turns over, where `sway` is only how
    # hard it is pushed. A sleeping face surges over sixteen seconds because that is a creature
    # breathing; a working one wants its parts changing speed several times in a glance.
    # It snaps between moods rather than easing; see :meth:`lerp` for the measurement.
    core: float = 1.0  # how much of an eye there is behind the blades, 0 .. 1. At 0 the
    # diaphragm meets in the middle and nothing is drawn inside it - no hole, no bezel, no spark -
    # which is the difference between a narrowed eye and a shut one. Every other number in here
    # describes a face looking at something, at some width and in some direction; this is the one
    # that says there is nobody looking out. It scales the hole rather than gating it so that
    # closing is an animation: he winds the iris down as he goes to work.
    rings: float = 1.0  # how present he is at all. 1 is his own resting level and
    # not a ceiling - above it he comes up towards full, which is what a tap uses
    scan: float = 0.0  # length in degrees of a bright arc sweeping the rim, 0 for none
    blink_s: float = 0.0  # mean seconds between blinks; 0 for a mood that does not blink, and
    # that means every lid he has - see :func:`gaze_blink`, which is also switched off by it
    # ...and where he is looking. See :func:`gaze_at`. A mood with `gaze` at 0 and no lean stares
    # dead ahead forever, which is exactly the drawing that existed before any of this.
    look: tuple[str, ...] = (AHEAD,)  # the places he looks, in order of how much they matter.
    # The first is his anchor - where he sits by default and where a look comes back to - and the
    # first after that is what he is attending to, which gets most of the glances he does take.
    # See :data:`FOCUS`: the list is a ranking, so putting the caption second under SPEAKING is
    # the whole of "he checks his own line while he talks". A mood naming one place never takes
    # its eyes off it. Names rather than vectors, because this module does not know where
    # anything is - see LANDMARKS.
    gaze: float = 0.0  # how far he actually turns to look at a place, 0 .. 1 of the full travel.
    # It scales the anchor as well as the glances, so 0 is dead centre whatever `look` says -
    # which is what the fault, the working face and the tap acknowledgement all rely on.
    dart: float = 0.0  # how much of his attention goes anywhere but the anchor, 0 .. 1: the
    # share of windows spent on a glance. 0 is a mood that never looks away from its anchor.
    dart_s: float = 1.7  # one window of his attention. Short is restless, long is deliberate,
    # and it may not go under DART_FLOOR. Does not ease between moods - see lerp: it indexes a
    # sequence rather than scaling one
    drift: float = 0.0  # a slow float on top of whatever he is fixed on, as a fraction of the
    # full travel. Not the same thing as MICRO: that is the tremor every open eye has whether it
    # is asked for or not, and this is a mood asking to wander. It is what a sleeping face is
    # made of, and what nothing awake and attending should have much of.

    def lerp(self, other: Mood, t: float) -> Mood:
        """Part-way from this mood to *other*, colour included."""
        k = max(0.0, min(1.0, t))
        if k <= 0.0:
            return self
        if k >= 1.0:
            return other
        return replace(
            other,
            tint=mix(self.tint, other.tint, k),
            **{
                name: getattr(self, name) + (getattr(other, name) - getattr(self, name)) * k
                # Every number that eases is named here, and the list is the whole of it: lerp
                # builds its result out of `other`, so a field left off this line is taken
                # wholesale from the destination and snaps. Add a field to Mood and you must
                # decide about it here.
                #
                # The rule for deciding is *amplitudes ease, periods do not*. A half-way
                # amplitude is a real value - half as far, half as bright. A half-way period is
                # not: `blink_s` and `dart_s` are both used as `floor(phase / period)` to index
                # a sequence, so a period sliding through a crossfade makes that index jump
                # about, and the eye spends the 0.45 s darting at random. `breath_s` is a period
                # too and *is* eased, which is the exception that shows the rule - it is used as
                # a phase, not as an index, so it slides instead of jumping.
                #
                # `sway_s` is a period and is NOT eased, and "it goes into a sine rather than a
                # floor" is not enough to save it - that was tried. wander() uses it as
                # `sin(phase / period)`, and `phase` is monotonic seconds since the kiosk started,
                # so sliding the period from 16.3 to 1.6 sweeps that argument through thousands of
                # turns inside 0.45 s. Measured: the knurl lurches between -60 and +165 degrees a
                # second during the crossfade, against a mood whose steady rate is 52, and it does
                # it at every uptime. Snapped, the same transition stays inside its own rate. The
                # amplitude eases and the period does not, which is the rule above after all.
                #
                # `look` is not a number at all and so cannot be on this line: it is taken from
                # the destination and snaps, so the target changes once, at the instant the mood
                # does. That is a saccade, and it is fine - what the rule above forbids is a
                # *sliding* index, which changes dozens of times inside the 0.45 s. Changing
                # where you are looking once is what a creature does when it is told something.
                #
                # `dart` is eased even though it is compared against a walk rather than scaling
                # one, and that is safe for a reason worth writing down before somebody "fixes"
                # it: the ease is monotone in t and t is monotone in time, so `walk(w) < dart`
                # flips at most *once* per window however far the value travels. One flip is one
                # saccade. `dart_s` is the counterexample sitting in the same expression - slide
                # that and floor(phase / dart_s) sweeps through hundreds of windows, which is the
                # measured disaster the paragraph above records.
                for name in ("aperture", "swell", "breath_s", "voice", "spin", "sway", "core",
                             "rings", "scan", "gaze", "dart", "drift")
            },
        )


def gaze_at(phase: float, mood: Mood,
            places: dict[str, tuple[float, float]] | None = None) -> tuple[float, float]:
    """Where the eye is looking at *phase*: a place he is fixed on, and the tremor under it.

    He holds an anchor - ``mood.look[0]``, and for nearly every mood that is :data:`AHEAD`, which
    is (0, 0), which is straight out of the glass at whoever is standing there. Attention comes in
    windows of ``dart_s``, and a walk decides whether each one is spent on that anchor or on a
    look at one of the mood's other places - weighted towards the first one it names, because
    that is the thing that is happening. Consecutive anchor windows are the *same* target, so
    nothing happens between them and they merge: what comes out is a long hold on you, broken by
    looks that are themselves held.

    That the look is *held* is the point, and :data:`DWELL_MIN` is where it is said. He goes, he
    arrives, he stays two and a half to three seconds - two or three frames of a strip sampled
    once a second - and then he comes back. Out and straight back is not a look; it is a flinch,
    and it is what this read as for seven rounds.

    Between two targets he is ballistic - :data:`SACCADE_S` on a smoothstep, quick and not
    instantaneous - and while he is fixed there is always :data:`MICRO` under him, because an eye
    that stops moving is a dead one.

    The sequence is a golden-ratio walk for the reason :func:`blink` gives: deterministic,
    identical on every boot, and never settling into a round anybody can anticipate. Not a PRNG
    and pointedly not ``hash()``. *Which* place is picked walks a second irrational - see
    :data:`DART_PICK`, where sharing one is measured going wrong.

    Pure and O(1): a handful of modulos and sines, whatever the uptime.

    Returns a vector no longer than 1. ``gaze`` at 0 returns (0, 0) whatever ``look`` says, so a
    fault holds dead centre however long the panel has been up.
    """
    spots = LANDMARKS if places is None else places
    home = spots.get(mood.look[0], (0.0, 0.0)) if mood.look else (0.0, 0.0)
    bx, by = home
    n = len(mood.look)
    if mood.gaze > 0.0 and mood.dart > 0.0 and mood.dart_s > 0.0 and n > 1:
        window = math.floor(phase / mood.dart_s)
        since = phase - window * mood.dart_s
        held = _hold(window, mood, n)  # how much of this window is spent away, 0 if none of it
        where, _ = _stray(window, mood.dart, n)
        out = spots.get(mood.look[where], home) if where else home
        came = _settled(window - 1, mood, n, spots, home)  # where the last window left him
        if since < held:
            bx, by = _flick(came, out, since)
        else:
            # The way home, and its start is *where he actually was* when he turned round rather
            # than the place he was heading for. Those are the same only when the peek outlasted a
            # saccade, which at a short window it does not - take the place instead and he
            # teleports out and then eases back, a step of 0.23 of the travel in one frame at
            # dart_s 0.35 against a ballistic bound of 0.058.
            bx, by = _flick(_flick(came, out, held) if held > 0.0 else came, home, since - held)
    gx, gy = bx * mood.gaze, by * mood.gaze
    # The tremor, and a mood's own float on top of it. Two sines per axis on periods in the golden
    # ratio, which never come back into step - a wander that repeats is a windscreen wiper. Added
    # after the gaze scale rather than before it: this is the eyeball being alive, not part of the
    # excursion, so it does not shrink with a mood that looks less far.
    for amount, short, ox, oy in ((mood.drift, DRIFT_S, 0.37, 0.11),
                                  (MICRO * min(1.0, mood.gaze / MICRO_LIT), MICRO_S, 0.83, 0.29)):
        if amount <= 0.0:
            continue
        long = short * (1.0 + BLINK_DRIFT)
        gx += amount * 0.5 * (math.sin(TAU * phase / short)
                              + math.sin(TAU * (phase / long + ox)))
        gy += amount * 0.5 * (math.sin(TAU * (phase / short + 0.61))
                              + math.sin(TAU * (phase / long + oy)))
    return steady(gx, gy)


def _flick(a: tuple[float, float], b: tuple[float, float], since: float) -> tuple[float, float]:
    """*since* seconds into a saccade from *a* to *b*. Ballistic, and finished by SACCADE_S."""
    k = 1.0 if SACCADE_S <= 0.0 else min(1.0, max(0.0, since) / SACCADE_S)
    k = k * k * (3.0 - 2.0 * k)  # smoothstep: a dart is quick, not instantaneous
    return a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k


def _stray(window: int, dart: float, n: int) -> tuple[int, float]:
    """Which of *n* places attention *window* is spent on, and the walk it was picked with.

    0 is the anchor, and is what most windows come back. The pick is weighted by :data:`FOCUS`
    towards the *first* place the mood names, because `look` is a priority - the thing that is
    happening - rather than a list of directions to take turns over.

    The second number is the picking walk itself, untouched by the weighting: free, uniform, and
    used for how long the glance lasts, so a look is never quite the same length twice without a
    third walk. Uniform is the whole reason it is handed back raw rather than as the leftover of
    the weighted index, which is bunched at one end and would make every look the shortest one.
    """
    if n < 2 or dart <= 0.0 or (window * BLINK_DRIFT) % 1.0 >= dart:
        return 0, 0.0
    part = (window * DART_PICK) % 1.0
    return 1 + int(part**FOCUS * (n - 1)), part


def _hold(window: int, mood: Mood, n: int) -> float:
    """How many seconds of *window* are spent away from the anchor. 0 for one spent on it.

    A look goes out, is HELD for :data:`DWELL_MIN` to :data:`DWELL_MAX` seconds, and comes back
    inside its own window - that is what makes it a look rather than a change of mind - *unless*
    the next window is a glance too, in which case he stays out and goes straight on to the next
    thing. Without that, a hunting eye at dart 0.9 would bounce place, home, place, home twice a
    second, which is a metronome and not a search.

    The hold is clipped so the way home fits: a mood whose window is shorter than a dwell gets
    everything but one saccade of it, which is what keeps a fast, chaining mood continuous.
    """
    where, part = _stray(window, mood.dart, n)
    if where == 0:
        return 0.0
    if _stray(window + 1, mood.dart, n)[0] != 0:
        return mood.dart_s
    return min(mood.dart_s - SACCADE_S, DWELL_MIN + (DWELL_MAX - DWELL_MIN) * part)


def _settled(window: int, mood: Mood, n: int, spots: dict[str, tuple[float, float]],
             home: tuple[float, float]) -> tuple[float, float]:
    """Where *window* left him: the place he was on if he never came home, the anchor if he did."""
    where, _ = _stray(window, mood.dart, n)
    if where == 0 or _hold(window, mood, n) < mood.dart_s:
        return home
    return spots.get(mood.look[where], home)


def gaze_blink(phase: float, mood: Mood) -> float:
    """The lid he drops as he sets off to look at something. 1.0 whenever it is open.

    The pairing that sells a creature and that the spontaneous blink alone never gets: a face
    blinks *when it changes what it is looking at*, so the lid and the saccade are one event
    rather than two things happening on unrelated clocks. Three critics measured the old face,
    found its one blink nowhere near its one look, and read the blink as a flicker.

    Only on the way out, so there is one lid per look and not two - and only for a mood that
    blinks at all. ``blink_s`` at 0 means a face that does not blink for any reason, which is
    what the stare, the working face and the fault all rely on: the knob has to shut off the
    whole apparatus, not just the part of it that runs on its own clock.

    Pure, O(1) and off the same window arithmetic as :func:`gaze_at` - it costs one floor and one
    modulo more than the gaze already spends.
    """
    n = len(mood.look)
    if mood.blink_s <= 0.0 or mood.gaze <= 0.0 or mood.dart <= 0.0 or mood.dart_s <= 0.0 or n < 2:
        return 1.0
    window = math.floor((phase + LID_LEAD) / mood.dart_s)
    if _stray(window, mood.dart, n)[0] == 0:
        return 1.0  # a window spent on the anchor: nothing sets off, so no lid
    since = phase + LID_LEAD - window * mood.dart_s
    if since >= BLINK_S:
        return 1.0
    return 1.0 - breath(since, BLINK_S)


def at(p: float) -> float:
    """A panel coordinate - or a radius, which maps the same way - in the oversampled tile.

    A pixel at *p* covers the run from ``p * SUPERSAMPLE`` to ``p * SUPERSAMPLE + SUPERSAMPLE-1``
    over there, and PIL's boxes are inclusive of both ends, so the middle of that run is half a
    sample short of the far edge. Getting this wrong by that half-sample is not a rounding error
    you can ignore: it moves every ring off the centre the tile was cut around.

    Centres and radii, then, and not corners - a bounding box is built out of the two, because a
    corner put through here comes out a third of a pixel wide of where it belongs.
    """
    return p * SUPERSAMPLE + (SUPERSAMPLE - 1) / 2.0


def wide(stroke: float) -> float:
    """A stroke width over there, which is the plain multiple - a width has no ends to be off."""
    return max(1.0, stroke * SUPERSAMPLE)


def linear(colour: tuple[int, int, int], alpha: int = 255) -> tuple:
    """A colour in the form the tile wants it: linear light, multiplied by its own coverage.

    Both halves are undone by :func:`_straighten` on the way out, and both are there for the same
    reason - an average is only meaningful over numbers that mean something added together.
    Brightness in the panel's gamma does not (see :data:`GAMMA`), and neither does a colour
    holding out on how much of the pixel it actually covers.
    """
    return (*(LINEAR[c] * alpha // 255 for c in colour), alpha)


def smoothed(
    size: int | tuple[int, int], paint: Callable[[ImageDraw.ImageDraw], None]
) -> Image.Image:
    """*paint* drawn into a tile ``SUPERSAMPLE`` times *size*, handed back at *size*, unstepped.

    The whole of the anti-aliasing on this panel, in four lines. What *paint* draws must be laid
    out through :func:`at` and :func:`wide` and coloured through :func:`linear`; everything else
    about it is ordinary PIL.

    The ground is black at nothing, and that is load-bearing rather than tidy: the shrink averages
    the empty pixels in with the drawn ones, so what comes out is already multiplied by its own
    coverage - the form a composite wants, and the only one that does not leave a dark fringe down
    the outside of every curve.

    *size* is one number for a square tile, which is what every dial and the eye want, or a
    ``(width, height)`` pair for something that is not - the terminal's screen is six times as
    wide as it is deep, and squaring it off would supersample a quarter of a megapixel of empty
    tile to draw one letterbox.
    """
    w, h = (size, size) if isinstance(size, int) else size
    tile = Image.new("RGBA", (w * SUPERSAMPLE, h * SUPERSAMPLE), (0, 0, 0, 0))
    paint(ImageDraw.Draw(tile))
    # reduce() rather than resize(): the same box average, told up front that the ratio is a whole
    # number, and half the time for it on the Pi.
    return _straighten(tile.reduce(SUPERSAMPLE))


def _straighten(tile: Image.Image) -> Image.Image:
    """Divide the shrunken tile back out by its own coverage, and return it to the panel's gamma.

    The second half of the trick in :meth:`EyeEngine.paint`, and the reason it runs *after* the
    shrink rather than before it: fifty-odd thousand numbers on the tile as it lands instead of a
    million on the tile as it was drawn, which is the difference between a rounding error in the
    frame budget and a dropped frame.

    Where nothing was drawn there is nothing to divide by, and those pixels stay at zero - fully
    transparent, so what they carry never reaches the panel anyway.
    """
    px = np.asarray(tile)
    alpha = px[..., 3]
    scale = np.divide(255.0, alpha, out=np.zeros(alpha.shape, np.float32), where=alpha > 0)
    rgb = (px[..., :3] * scale[..., None]).clip(0.0, 255.0).astype(np.uint8)
    # The alpha band rides through untouched: it is a coverage, not a brightness, and has no
    # gamma to be in.
    return Image.fromarray(np.dstack((SRGB[rgb], alpha)), "RGBA")


class Pen:
    """The vocabulary the eye is drawn in: one shape per method, all of it centred on him.

    Everything here works in the oversampled tile's coordinates and takes radii as absolute
    lengths in it, so a part is written the way it is described - "a ring at 0.79 of him, a
    quarter lit" - and the arithmetic to get there happens once, in :class:`EyeEngine`.

    Two halves, and the split is worth knowing before adding to it. The top half is generic -
    points, rings, discs, bands, spokes, loops - and takes everything it needs as arguments. The
    bottom half is *this eye's own parts*, and each of those reads its own brightnesses from the
    block at the top of the module rather than taking a dozen arguments. That is what keeps the
    call sites in :class:`EyeEngine` down to a radius and a rotation, and what makes the constant
    block the one place to retune the drawing.

    A pen is cheap, which is what lets :meth:`shifted` carry the gaze: hand the optic a pen whose
    centre has moved and every part drawn through it moves with it.
    """

    def __init__(self, d: ImageDraw.ImageDraw, cx: float, cy: float, screen: tuple[int, int, int],
                 tint: tuple[int, int, int], lit: float, stroke: float, thin: float) -> None:
        self.d, self.cx, self.cy = d, cx, cy
        self.screen, self.tint, self.lit = screen, tint, lit
        self.stroke, self.thin = stroke, thin
        self.hair = max(1.0, thin * 0.6)

    def shifted(self, dx: float, dy: float) -> Pen:
        """The same pen, drawing about a centre *dx, dy* away.

        This is most of the gaze - every part drawn through the returned pen moves, and no part
        has to know that it can. The remainder is the core's own lead: its hot middle travels a
        little further than the disc around it, which is what makes a pupil inside a lens rather
        than one flat disc being dragged about. See :meth:`core`.

        A copy rather than a rebuild, so a pen that gains a field does not silently stop being
        carried across here.
        """
        moved = copy(self)
        moved.cx, moved.cy = self.cx + dx, self.cy + dy
        return moved

    # ---- colour ----

    def shade(self, strength: float) -> tuple:
        """The tint at *strength*, mixed towards the screen's own black and drawn opaque.

        Mixed in the panel's own gamma, because that is where the palette was chosen and where
        every other cell on the screen mixes, then handed over in the tile's terms.
        """
        return linear(mix(self.screen, self.tint, strength * self.lit))

    def heat(self, towards: float) -> tuple:
        """The tint pushed *towards* the top of the tube.

        The one place the eye is allowed to go paler than its own colour, which is what makes
        the core read as hot rather than merely bright.
        """
        return linear(mix(self.screen, mix(self.tint, HEAT, towards), self.lit))

    # ---- points ----

    def point(self, rad: float, deg: float) -> tuple[float, float]:
        a = math.radians(deg)
        return self.cx + rad * math.cos(a), self.cy + rad * math.sin(a)

    def arc_pts(self, rad: float, a0: float, a1: float, n: int = 24) -> list:
        return [self.point(rad, a0 + (a1 - a0) * i / n) for i in range(n + 1)]

    def box(self, rad: float) -> list[float]:
        return [self.cx - rad, self.cy - rad, self.cx + rad, self.cy + rad]

    # ---- shapes ----

    def ring(self, rad: float, strength: float, width: float) -> None:
        self.d.ellipse(self.box(rad), outline=self.shade(strength), width=max(1, round(width)))

    def disc(self, rad: float, strength: float, dx: float = 0.0, dy: float = 0.0) -> None:
        self.d.ellipse([self.cx + dx - rad, self.cy + dy - rad,
                        self.cx + dx + rad, self.cy + dy + rad], fill=self.shade(strength))

    def band(self, rad: float, a0: float, a1: float, strength: float, width: float) -> None:
        self.d.arc(self.box(rad), start=a0, end=a1, fill=self.shade(strength),
                   width=max(1, round(width)))

    def spoke(self, r0: float, r1: float, deg: float, strength: float, width: float) -> None:
        self.d.line([self.point(r0, deg), self.point(r1, deg)], fill=self.shade(strength),
                    width=max(1, round(width)))

    def loop(self, rad: float, strength: float, width: float, n: int = 64) -> None:
        """A circle drawn as a polyline, for when it has to sit on a moved centre exactly."""
        self.d.line(self.arc_pts(rad, 0.0, 360.0, n), fill=self.shade(strength),
                    width=max(1, round(width)), joint="curve")

    def sweep(self, rad: float, a0: float, a1: float, strength: float, width: float) -> None:
        """An arc drawn as a polyline: :meth:`band` for a moved centre, as :meth:`loop` is for
        :meth:`ring`.

        No rounded joints, unlike :meth:`loop`: a joint is an ellipse per vertex and is most of
        the cost of the line, and at a step of six degrees the notch it would fill is a
        hundredth of a pixel wide before the tile is even shrunk.
        """
        self.d.line(self.arc_pts(rad, a0, a1, max(6, int(abs(a1 - a0) / 6))),
                    fill=self.shade(strength), width=max(1, round(width)))

    # ---- the parts ----

    def castle(self, rad: float, n: int, jog: float, arc: float, start: float,
               strength: float, width: float) -> None:
        """A ring that steps radially by *jog* for *arc* degrees, *n* times round.

        Walked as one polyline with a vertex at each corner rather than sampled at a fixed step,
        which is what keeps the steps square: a sampled version puts the radial jump between two
        samples and draws it as a diagonal.
        """
        notches = [(start + i * 360.0 / n - arc / 2, start + i * 360.0 / n + arc / 2)
                   for i in range(n)]
        pts, here = [], notches[-1][1] - 360.0
        for a0, a1 in notches:
            pts += self.arc_pts(rad, here, a0, max(3, int(abs(a0 - here) / 6)))
            pts += [self.point(rad + jog, a0)]
            pts += self.arc_pts(rad + jog, a0, a1, max(2, int(arc / 6)))
            pts += [self.point(rad, a1)]
            here = a1
        pts += self.arc_pts(rad, here, notches[0][0] + 360.0, 8)
        self.d.line(pts, fill=self.shade(strength), width=max(1, round(width)), joint="curve")

    def dots(self, rad: float, n: int, size: float, strength: float, start: float = 0.0) -> None:
        for i in range(n):
            x, y = self.point(rad, start + i * 360.0 / n)
            self.d.ellipse([x - size, y - size, x + size, y + size], fill=self.shade(strength))

    def knurl(self, r0: float, r1: float, n: int, strength: float, start: float,
              span: float) -> None:
        for i in range(n):
            self.spoke(r0, r1, start + span * i / n, strength, self.hair * 1.4)

    def vanes(self, r0: float, r1: float, n: int, arc: float, strength: float) -> None:
        for i in range(n):
            a = i * 360.0 / n
            self.d.polygon(self.arc_pts(r1, a - arc, a + arc, 3)
                           + self.arc_pts(r0, a + arc, a - arc, 3), fill=self.shade(strength))

    def lug(self, rad: float, deg: float, arc: float, thick: float, strength: float) -> None:
        self.d.polygon(self.arc_pts(rad + thick, deg - arc, deg + arc, 4)
                       + self.arc_pts(rad - thick, deg + arc, deg - arc, 4),
                       fill=self.shade(strength))

    def greebles(self, rad: float, deg: float, tall: float) -> None:
        """A row of indicators at one clock position, one of them dark, under a short bar.

        The one asymmetric mark on him, and the reason it rides a turning ring: it is the only
        thing on the eye you can actually follow, so it is most of what says he is running at
        all. The dark one is not a mistake - a row with every lamp lit reads as a decoration,
        and a row with one out reads as a readout.
        """
        for i in range(GREEBLE_N):
            a = deg + (i - (GREEBLE_N - 1) / 2) * GREEBLE_SPREAD
            self.spoke(rad - tall, rad + tall, a,
                       GREEBLE_LIT if i < GREEBLE_N - 1 else GREEBLE_DARK, self.thin * 1.3)
        self.band(rad + tall * GREEBLE_BAR, deg - GREEBLE_BAR_ARC, deg + GREEBLE_BAR_ARC,
                  GREEBLE_BAR_LIT, self.hair)

    def index(self, r0: float, r1: float) -> None:
        """The brass ring the diaphragm is set in, and the scale cut round its face.

        Two arcs and not a gradient: the side turned towards the lamp is a step brighter than the
        side turned away, which is the whole of the modelling and costs two calls. The cuts are
        darker than the face because that is what a groove in metal is, and every fifth runs the
        full width so the ring reads as a scale rather than as a knurl. The polished lip on the
        outer edge goes on last - a machined ring is read from its edge before its face.
        """
        mid, wide = (r0 + r1) / 2.0, r1 - r0
        face = max(1, round(wide))
        self.d.ellipse(self.box(mid), outline=self.shade(INDEX_LIT), width=face)
        self.d.arc(self.box(mid), start=INDEX_FROM, end=INDEX_FROM + INDEX_SPAN,
                   fill=self.shade(INDEX_TOP), width=face)
        cut = self.shade(INDEX_CUT)
        for i in range(INDEX_TICKS):
            deg = i * 360.0 / INDEX_TICKS
            reach = r1 if i % INDEX_MAJOR == 0 else r0 + wide * INDEX_TICK
            self.d.line([self.point(r0, deg), self.point(reach, deg)],
                        fill=cut, width=max(1, round(self.hair)))
        self.d.ellipse(self.box(r1), outline=self.shade(INDEX_LIP),
                       width=max(1, round(self.hair)))

    def leaves(self, iris: float, hole: float, n: int, twist: float, pitch: float) -> None:
        """The diaphragm: *n* straight-edged blades cutting a regular polygon hole out of a disc.

        Each blade is the circular segment its own chord cuts off the iris, every chord tangent
        to the hole - so what the n of them leave behind is the polygon the core sits in. The
        hatch runs parallel to each blade's own edge and is worked out rather than masked: a line
        at distance *t* from the centre is ``2*sqrt(iris^2 - t^2)`` long, so it stops exactly at
        the blade's edge for the cost of a square root.

        The hatch is not decoration. Filled flat, the eight blades merge into one dark ring and
        the diaphragm stops being readable at all; it and the banding below are what say these
        are separate plates lying over each other.

        Nor is a blade one value. It is a plate tilted out of the plane on its way to the hole,
        so it takes more of the lamp the nearer the aperture it gets. The segments successive
        chords cut off the iris nest inside one another, so the whole grade costs `LEAF_BANDS`
        polygons and no gradient at all: lay the widest down at the aperture's value and step
        outwards, each one smaller and a shade darker, until the last is the sliver at the hinge.
        """
        facet, off = self.shade(EDGE_SHADE), self.thin * EDGE_BEVEL
        # Every colour and width this loop uses, worked out once: eight blades and ten hatch
        # lines each is a hundred trips through `shade` a frame for three answers.
        hatch, lit, hair = self.shade(HATCH_LIT), self.shade(EDGE_LIT), max(1, round(self.hair))
        steps = [(math.degrees(math.acos(max(-1.0, min(1.0, (hole + (iris - hole) * k
                                                            / LEAF_BANDS) / iris)))),
                  self.shade(LEAF_FACE + (LEAF_LIT - LEAF_FACE) * k / max(1, LEAF_BANDS - 1)))
                 for k in range(LEAF_BANDS)]
        half = steps[0][0]
        for i in range(n):
            a = twist + i * 360.0 / n
            seg = self.arc_pts(iris, a - half, a + half, 20)
            self.d.polygon(seg, fill=steps[0][1])
            for wide, tone in steps[1:]:
                self.d.polygon(self.arc_pts(iris, a - wide, a + wide, 6), fill=tone)
            ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
            t = hole + pitch * 0.5
            while t < iris - pitch * 0.2:
                ln = math.sqrt(max(0.0, iris * iris - t * t))
                mx, my = self.cx + t * ca, self.cy + t * sa
                self.d.line([mx - ln * sa, my + ln * ca, mx + ln * sa, my - ln * ca],
                            fill=hatch, width=hair)
                t += pitch
            self.d.line([seg[0], seg[-1]], fill=lit, width=max(1, round(self.thin)))
            # The facet, a hairline onto the blade from its lit edge. Under the iris loop at
            # both ends, so it has no ends of its own to read.
            self.d.line([(seg[0][0] + ca * off, seg[0][1] + sa * off),
                         (seg[-1][0] + ca * off, seg[-1][1] + sa * off)],
                        fill=facet, width=max(1, round(self.hair)))
        self.loop(iris, IRIS_LIT, self.stroke)

    def core(self, rad: float, lead: tuple[float, float]) -> None:
        """Filaments converging on a hot centre, with the centre led out towards his gaze.

        The glow is free. The threads are radial, so their density goes as 1/r and the middle is
        bright because it is *crowded* - there is no gradient anywhere in here, and there is none
        in the reference this came from either.

        The thread count is a ceiling rather than a fixed number: past about one thread every two
        panel pixels at the rim they stop resolving and are just cost, and on a narrow aperture
        the core is small enough for that to bite.
        """
        n = max(12, min(CORE_N, round(TAU * rad / (1.2 * SUPERSAMPLE))))
        self.disc(rad, CORE_FLOOR)
        thread, hair = self.shade(THREAD_LIT), max(1, round(self.hair))
        # CORE_HUB leaves a gap the filaments do not fill, and it is meant to spend its life
        # under the hot middle. At the lead this eye now uses it does not: left where it is, that
        # gap is a small dark disc at the tile's centre while the pupil is off to one side, which
        # reads as a second and truer pupil - and dragging it the whole way instead skews the fan
        # until it stops being radial and turns into a moire. Both were rendered; this is the
        # third thing. The inner ends hold still until the hot middle is about to slide off the
        # gap and then follow only as far as it takes to stay under it, so the fan is exactly
        # radial while he is looking at you and no more skewed than it has to be when he is not.
        # Free either way: the same lines, moved.
        dx, dy = lead
        mag = math.hypot(dx, dy)
        over = max(0.0, mag - rad * (HOT - CORE_HUB)) / mag if mag > 0.0 else 0.0
        hx, hy = dx * over, dy * over
        for i in range(n):
            a = i * 360.0 / n
            x, y = self.point(rad * CORE_HUB, a)
            self.d.line([(x + hx, y + hy), self.point(rad * CORE_EDGE, a)],
                        fill=thread, width=hair)
        for k in CORE_RINGS:
            self.loop(rad * k, CORE_RING_LIT, self.hair, 40)
        self.loop(rad, CORE_EDGE_LIT, self.thin)
        # The spark has a floor under it in panel pixels, and it is load-bearing: it is the
        # brightest thing on the face, and a spark that does not fully cover at least one panel
        # pixel changes value as it slides between them - which would turn his gaze into a
        # flicker and make a sleeping face read as dimming rather than as moving, which is the
        # one thing the sleeping face is not allowed to do.
        #
        # The floor is sqrt(2) rounded up, and that bound is derived rather than picked. A disc
        # covers a whole panel pixel only if all four of that pixel's corners are inside it, and
        # the worst alignment puts the disc's centre on a corner - where the nearest pixel's far
        # corner is sqrt(2) away. So sqrt(2) is what *guarantees* it at every sub-pixel position.
        #
        # Below it, behaviour is unreliable rather than uniformly worse, which is worth writing
        # down because it is the trap: swept over 576 sub-pixel positions at eye_r 40, a floor of
        # 1.2 dilutes the spark at 4 of them (peak 691 against 699) while 1.0 passes all 576.
        # 1.0 is not safer - it is a rasterising coincidence at that one radius, and the kind of
        # thing that comes back on a different panel. Take the bound, not the measurement.
        self.d.ellipse([self.cx + dx - rad * HOT, self.cy + dy - rad * HOT,
                        self.cx + dx + rad * HOT, self.cy + dy + rad * HOT],
                       fill=self.heat(HOT_MIX))
        spark = max(SPARK_FLOOR * SUPERSAMPLE, rad * SPARK)
        self.d.ellipse([self.cx + dx - spark, self.cy + dy - spark,
                        self.cx + dx + spark, self.cy + dy + spark], fill=self.heat(SPARK_MIX))


class EyeEngine:
    """Draws the eye at one size, and remembers which mood it is on its way to.

    Geometry is worked out once per size and the mood per frame, for the reason the whole overlay
    is built that way: a Pi has 40 ms for the loop and the camera wants most of them. Every part
    of him moves, so none of it is cached at all.

    The drawing is in three parts and the split is the design rather than tidiness.
    :meth:`_shell` is the socket and never moves with his gaze; :meth:`_optic` is the eye in it
    and always does; and the brow is drawn last on the unshifted pen, because it is a highlight
    on the outer glass in front of both. A new part joins whichever of the three it belongs to
    and gets the gaze - or the lack of it - for free.
    """

    def __init__(self, radius: int, line: int, screen: tuple[int, int, int], resting: Mood,
                 places: dict[str, tuple[float, float]] | None = None) -> None:
        self.r = radius
        # Where the names in `Mood.look` actually are, from where he is bolted. That is geometry,
        # so it lives here with the rest of it; optional, so the preview harness can build an
        # engine without a panel and still get the panel's own answers - see LANDMARKS.
        self.places = dict(LANDMARKS if places is None else places)
        self.screen = screen
        self.stroke = max(1, line)
        self.thin = max(1, line // 2)
        self._mood = resting  # what it is now...
        self._from = resting  # ...what it was before that, and when it changed
        self._at = 0.0
        self._key = ""
        # The index ring, one picture per brightness he is ever drawn at. See :meth:`_index`.
        self._rings_seen: dict[float, Image.Image] = {}
        # His tile is his bounding box and not a pixel more - 121 px square as it lands on the
        # 800x480 panel - which is what makes drawing him four times over affordable inside a
        # 40 ms frame. His centre in it is also the rim's radius, the rim being the tile drawn
        # edge to edge, and the strokes are worked out here because none of them ever changes.
        self._size = 2 * radius + 1
        self._c = at(radius)
        self._stroke = wide(self.stroke)
        self._thin = wide(self.thin)

    # ---- the mood ----

    def look(self, key: str, mood: Mood, phase: float) -> Mood:
        """The mood to draw right now, eased out of whatever the last one was.

        The eye is the only thing on the panel that crossfades. The border snaps between its
        three colours because it is a signal and wants to be read the instant it changes; the
        eye is a face, and a face that jumped from green to amber between two frames would read
        as a different creature rather than as this one waking up.
        """
        if key != self._key:
            self._from = self._mood
            self._at = phase
            self._key = key
        elapsed = phase - self._at
        t = 1.0 if MOOD_EASE_S <= 0 else max(0.0, min(1.0, elapsed / MOOD_EASE_S))
        self._mood = self._from.lerp(mood, t)
        return self._mood

    # ---- the drawing ----

    def aperture(self, mood: Mood, phase: float, level: float) -> float:
        """How far the iris actually stands open this frame: rest, breath, voice, then the lids.

        Two lids and not one, and they are two because they are two different things: one runs on
        its own clock because an eye that never blinks is glass (:func:`blink`), and one falls
        because he is about to look somewhere (:func:`gaze_blink`). Both multiply rather than
        subtract, so either closes the eye all the way from wherever it happened to be - a mood
        that sits half open still blinks shut, not to a quarter.
        """
        open_ = mood.aperture + mood.swell * breath(phase, mood.breath_s)
        open_ += mood.voice * max(0.0, min(1.0, level))
        lids = blink(phase, mood.blink_s) * gaze_blink(phase, mood)
        return max(0.0, min(1.0, open_)) * lids

    def paint(
        self, img: Image.Image, cx: int, cy: int, mood: Mood, phase: float, level: float
    ) -> None:
        """The whole eye, onto *img* with its centre at *cx, cy*.

        Drawn into a tile of its own and shrunk onto the panel rather than stroked straight onto
        it, because PIL will not anti-alias and he is nothing but curves. Everything about the
        drawing is the same either way - :data:`SUPERSAMPLE` of 1 puts back the pixels this used
        to put down - so the smoothing is a property of how he lands, not of how he is drawn.

        The tile is opaque only where he is, so what shows between the rings is still his plate
        and the room through it.
        """
        tile = smoothed(self._size, lambda d: self._draw(d, mood, phase, level))
        # The index ring is the one part of him whose picture never changes - only where it sits.
        # Sixty cuts and three circles is half a millisecond of PIL if they are stroked every
        # frame, which is the whole per-frame budget for a thing that is the same picture each
        # time; so it is drawn once per brightness and pasted at the offset his gaze has reached.
        # Its own radii are clear of the core and the bezel, so landing it last changes no pixel
        # that another part of him would have won.
        gx, gy = gaze_at(phase, mood, self.places)
        ring = self._index(round(mood.rings, 2))
        half = ring.width // 2
        tile.alpha_composite(ring, (round(self.r + gx * self.r * GAZE_SHIFT) - half,
                                    round(self.r + gy * self.r * GAZE_SHIFT) - half))
        img.alpha_composite(tile, (cx - self.r, cy - self.r))

    def _index(self, lit: float) -> Image.Image:
        """The brass ring, drawn once at each brightness his mood asks for, and kept.

        Keyed on `rings` rounded to a hundredth: a crossfade walks that number, and a table with
        one entry per frame of a walk is not a cache. A hundredth is finer than the panel can
        show, and the moods between them leave it a few dozen entries for the life of a kiosk -
        the same bargain `Overlay._bases` makes, and bounded for the same reason.
        """
        ring = self._rings_seen.get(lit)
        if ring is not None:
            return ring
        span = 2 * round(self.r * INDEX_OUT) + 3
        mid = at((span - 1) / 2.0)
        ring = smoothed(span, lambda d: Pen(
            d, mid, mid, self.screen, BRASS_EYE, max(0.0, lit), self._stroke, self._thin,
        ).index(at(self.r * INDEX_IN), at(self.r * INDEX_OUT)))
        self._rings_seen[lit] = ring
        return ring

    def _draw(self, d: ImageDraw.ImageDraw, mood: Mood, phase: float, level: float) -> None:
        """The whole eye at the centre of its tile: the socket, then the eye, then the glass."""
        # `rings` is not clamped at 1: it is how present he is, and the parts below are drawn
        # well short of full on purpose (see the *_LIT block), so "brighter than usual" needs
        # somewhere above 1 to go. That is what the tap acknowledgement uses. `Pen.shade` mixes
        # through `mix`, which clamps, so an absurd value saturates rather than raising.
        pen = Pen(d, self._c, self._c, self.screen, mood.tint,
                  max(0.0, mood.rings), self._stroke, self._thin)
        # The gaze first, because the socket is geared to it now - see RING_GEAR. It is still one
        # call: the shell is handed the answer rather than asking for its own.
        gx, gy = gaze_at(phase, mood, self.places)
        self._shell(pen, mood, phase, gx)
        self._optic(pen.shifted(gx * self._c * GAZE_SHIFT, gy * self._c * GAZE_SHIFT),
                    mood, phase, level, gx, gy)
        # The brow last, and on the *unshifted* pen. It is a highlight on the outer glass, and a
        # highlight does not travel with what is under it - which is the whole depth cue, and the
        # reason the eye reads as turning rather than as sliding about.
        pen.band(self._c * BROW_AT, BROW_FROM, BROW_TO, BROW_LIT, self._thin * BROW_W)

    def _shell(self, pen: Pen, mood: Mood, phase: float, gx: float) -> None:
        """The socket. Nothing in here *travels* with his gaze, which is the whole of what it is
        for - but the rings do turn with it, because they are what carries him. See RING_GEAR.

        Not everything in here turns, either, and the exceptions are deliberate: the rim and the
        vanes hold their angle so there is something fixed to read the turning parts against. A
        set where every ring moves is very nearly as hard to read as one where none does.
        """
        r = self._c
        # Each ring is asked where it has got to rather than all of them being read off one
        # clock. With no sway that is the same product it always was; with one they drift apart,
        # overtake and turn back under each other - see :func:`wander`. On top of that the set is
        # indexed by however far he has turned, which is what stops the steady rate being the
        # most legible thing on him: the rings hold a rate while he holds a place, and lurch
        # round when he looks - the same event, on two parts.
        gear = RING_GEAR * gx
        fast = wander(phase, mood.spin, KNURL_SPIN, mood.sway, 0, mood.sway_s) + gear * KNURL_SPIN
        mid = wander(phase, mood.spin, CASTLE_SPIN, mood.sway, 1, mood.sway_s) + gear * CASTLE_SPIN
        slow = wander(phase, mood.spin, DOT_SPIN, mood.sway, 2, mood.sway_s) + gear * DOT_SPIN

        # The rim, and the sweep when a mood asks for one. A radar sweep is built the way one is:
        # the trace is bright and the ring under it is faint. The highlight used to be mixed
        # towards white over a rim at full, which worked while the accent was a colour and
        # stopped working the day the accent became the tube's own white - there is nowhere paler
        # than pale to go. Turning the rim down instead needs no headroom above the tint and
        # cannot be defeated by any of it.
        scanning = mood.scan > 0.0
        pen.ring(r * RIM, RIM_LIT * (SCAN_DIM if scanning else 1.0), self._stroke * RIM_W)
        if scanning:
            start = wander(phase, mood.spin, SCAN_SPIN, mood.sway, 3, mood.sway_s) % 360.0
            pen.band(r * RIM, start, start + mood.scan, RIM_LIT,
                     self._stroke * RIM_W + self._thin)

        # The order below is load-bearing and is NOT the radial order. ImageDraw writes rather
        # than composites, so the last part drawn over a shared band is the one you see: the
        # vanes go on last and lie over the datum ring, the greeblies and the dotted ring, which
        # is what makes them read as a shroud in front rather than as another ring among them.
        # Sorting these lines outside-in looks tidier and quietly redraws the eye - it was tried,
        # and it moved 0.4% of his pixels.
        #
        # Nothing is drawn on the seat's band but the knurl, and the seat itself is not here at
        # all: it is turned into the well behind him, and what the tile leaves alone in that band
        # is what shows of it.
        pen.castle(r * CASTLE, CASTLE_N, r * CASTLE_JOG, CASTLE_ARC, CASTLE_AT + mid, CASTLE_LIT,
                   self._thin * CASTLE_W)
        pen.ring(r * DATUM, DATUM_LIT, pen.hair)
        pen.greebles(r * DATUM, GREEBLE_AT + slow, r * GREEBLE_TALL)
        pen.lug(r * DATUM, LUG_AT + slow, LUG_ARC, r * LUG_THICK, LUG_LIT)
        pen.spoke(r * SEAM_IN, r * CASTLE, SEAM_AT + mid, SEAM_LIT, self._thin)
        pen.dots(r * DOTS, DOT_N, max(1.0, pen.hair), DOT_LIT, slow)
        pen.knurl(r * KNURL_IN, r * KNURL_OUT, KNURL_N, KNURL_LIT, KNURL_FROM + fast, KNURL_SPAN)
        pen.vanes(r * VANE_IN, r * VANE_OUT, VANE_N, VANE_ARC, VANE_LIT)

    def _optic(self, pen: Pen, mood: Mood, phase: float, level: float,
               gx: float, gy: float) -> None:
        """The eye in the socket: blades, the bezel, and the core. All of it follows his gaze.

        The core can only be as big as the hole the blades leave, and the hole grows as the iris
        opens - so a wide-awake face has a big hot core and a dozing one has a small ember, for
        free and by construction, off the one number every mood already sets.
        """
        r = self._c
        iris = r * IRIS
        open_ = self.aperture(mood, phase, level)
        hole = iris * (HOLE_MIN + (HOLE_MAX - HOLE_MIN) * open_) * max(0.0, min(1.0, mood.core))
        pen.leaves(iris, hole, LEAF_N, LEAF_TWIST, max(2.0 * SUPERSAMPLE, iris * HATCH))
        # A hole narrower than the spark's own floor is not a small eye, it is a closed one, and
        # the two have to be told apart here because :meth:`Pen.core` cannot: it guarantees a
        # spark of at least one whole panel pixel however small the core is (and must - see the
        # note there), so drawing it at a hole the blades have already met across would leave the
        # brightest mark on the whole face sitting on a shut diaphragm.
        if hole <= SPARK_FLOOR * SUPERSAMPLE:
            return
        pen.loop(hole * BEZEL, BEZEL_LIT, self._stroke * BEZEL_W)
        pen.sweep(hole * BEZEL, BEZEL_HI_FROM, BEZEL_HI_TO, BEZEL_HI, self._stroke * BEZEL_W)
        # The pupil rolls inside the lens, and stops at its rim rather than climbing over it -
        # see LEAD_EDGE. One hypot and a min: the moods that look hardest are the only ones that
        # ever reach the stop, and none of them reaches it except at the far end of a saccade.
        room, mag = LEAD_EDGE - HOT, math.hypot(gx, gy)
        lead = GAZE_LEAD if mag * GAZE_LEAD <= room else room / mag  # mag 0 takes the first
        pen.core(hole, (gx * hole * lead, gy * hole * lead))
