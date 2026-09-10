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

# ---------------------------------------------------------------- the cover, in front of it all
#
# A steel diaphragm at his full radius: a lens cap, not another ring on him. It is the one part
# of this drawing that is not phosphor - see :meth:`Pen.steel` - and the one part that is not
# there at all most of the time. While he is awake every blade is wound clear of the rim and
# nothing below is drawn, so an awake eye is the same pixels it was before any of this existed.
# That is the whole licence for it: metal inside his radius measurably killed the face when it
# was there while he was looking at you, and this is only there while nobody is.
COVER_N = 6  # blades. Six is the reference's, and it is also the fewest that still meet in a
# spiral rather than in a star: each blade has to sweep 360/n and a wider sweep is a fatter blade.
COVER_PIVOT = 1.00  # where a blade turns, as a fraction of him: on the rim, like a lens's.
# It sets the whole mechanism. A blade's cutting edge is an arc of *his own radius* whose centre
# rides at this distance from the pivot, so at rest that centre is dead centre, the arc IS the rim
# and every blade lies outside the drawing. Wind them all on by the same angle and the centres
# swing out to 2 * COVER_PIVOT * sin(turn/2); the n discs still overlap in a curved polygon whose
# inradius is what is left of him, and that polygon is the aperture. Nothing scales and nothing
# is masked: the hole is what n rigid shapes happen to leave.
COVER_SHUT = math.degrees(2.0 * math.asin(min(1.0, 1.0 / (2.0 * COVER_PIVOT))))  # ...and how far
# each blade has to wind for that inradius to reach zero. 60 degrees at a pivot on the rim, and
# derived rather than typed so that moving the pivot moves the throw with it.
COVER_AT = 96  # where blade zero's pivot sits. Off the vertical, so the shut spiral does not
# line up with the seam, the greeblies or the lug - the eye's one rule about symmetry.
COVER_SLICES = 40  # how many flat wedges one blade's shading is cut into, along its own arc.
# The blade is drawn as a fan of quads between its cutting circle and its neighbour's, each one
# a constant value, because the shading has to run ALONG the arc rather than across the disc and
# nothing cheaper does that. Twenty is where the steps stop being steps at his size and start
# being the grain - each wedge is four or five panel pixels of arc, and COVER_GRAIN breaks the
# ramp up inside that. At twenty they were nine, and a jitter that wide is not grain, it is a
# fan: the wedges read as the facets they are. A real gradient here would be a numpy pass over a
# quarter of a million pixels every frame, which is the same trade `leaves` made and lost.
COVER_GROOVE = 1.42  # the machined arc along a blade's back, as a multiple of its cutting arc's
# radius about the same centre. Rigid, like everything else here: it is a fixed radius about a
# point the blade carries, so it turns with the blade and cannot slide across its face.
COVER_SEAT = 0.968  # where the collar's inner edge sits, as a fraction of him: under three panel
# pixels of ring. It was 0.90, which is a fifth of his radius and made the brightest thing in the
# frame a bezel - the blades have to own the face, and a collar is a rim, not a mount.
COVER_STEP = 6.0  # degrees between samples along any of the arcs the lines are drawn on. The
# chord it leaves is a tenth of a panel pixel; `sweep` uses the same step for the same reason.
COVER_BLEED_N = 6  # ...and how many pieces the seam's own light is cut into, so it can fade
COVER_LAMP = 249  # PIL degrees, the same lamp BEZEL_HI_FROM/TO is centred on. A blade's face
# takes its value from how squarely it looks at it, so the six of them are six different greys
# and the disc reads as turned metal rather than as a printed spiral.

# How present each part of it is - against STEEL rather than against the mood, and NOT scaled by
# `rings`. A tap brightens the phosphor because that is him answering; the cover is a lump of
# metal in front of him and answers nothing.
STEEL = (233, 242, 245)  # brushed steel at its brightest, a shade cool - the top of the
# specular rather than the colour of a face, since everything below is a fraction of it. The one
# colour on this panel that is not the phosphor's, and it is why the cover reads as a part rather
# than as a drawing of one.
COVER_FLOOR = 0.10  # the plate the blades ride on, seen through the seams and under the collar
COVER_DARK = 0.12  # a blade's face turned away from the lamp: near black, and it has to be. A
# blade is a piece of metal in a room with one light in it, and metal says so by having a range -
# nothing on the disc near white and nothing near black is a paper cutout, whatever it is shaped
# like. This was 0.42 to 0.92 for a round, which is a fifth of the range there is, and it read as
# grey card laid over a lit face.
COVER_LIGHT = 0.64  # ...and turned towards it. The diffuse half, and it runs along the blade's
# own arc rather than across the disc, so the six blades catch the light at six different points
# along themselves and the highlight walks round the spiral instead of landing on all of them.
COVER_SPEC = 0.34  # the specular band on top of that, which is what takes the brightest part of
# a blade to the top of the tube - and just to it, rather than past it. A band that clips spends
# its middle on a flat plateau, which is the one part of a blade with no grain in it and the part
# the eye goes to first...
COVER_GLINT = 0.30  # ...and how wide it is, in radians of the blade's own arc. Narrow: a broad
# specular is a diffuse, and the band has to be a band.
COVER_GRAIN = 0.06  # how far consecutive wedges are pushed off the ramp, which is the brushing.
# Walked on the golden ratio for the reason :func:`blink` walks its own by it - not a PRNG, so
# the same blade has the same grain on every boot, and no two wedges in a row agree.
COVER_MACHINED = 0.78  # the machined arc along a blade's back, as a share of that blade's own
# value: a mark ON the metal, so it has to move with the metal rather than being one grey.
COVER_EDGE = 1.00  # the chamfer down a blade's cutting edge: the top of the tube, and the one
# line that says "machined" at this size...
COVER_EDGE_HELD = 0.90  # ...but only once the lid is home. See COVER_MUTED: this and the
# specular are the two things on the cover that reach the top of the tube, so they are the two
# that have to wait until there is no face left for them to out-shine.
COVER_GAP = 0.03  # ...and the dark line under it where one plate lies over the next
COVER_SEAM = 0.95  # the light that gets past that, in the mood's own colour, on the plate below.
# The one thing on the cover that is his rather than the steel's, and the whole of what a shut
# face says: he is behind it, and the light is still on.
COVER_SEAM_FADE = 0.15  # ...and what is left of it out at the rim. Brightest at the hub, because
# that is where the light is and because a seam lit evenly along its length is a drawn line.
COVER_COLLAR = 0.86  # the collar where the lamp lands on it...
COVER_COLLAR_LO = 0.14  # ...and away from it
COVER_COLLAR_N = 32  # ...and how many arcs that grade is cut into. A ring drawn dark with one
# bright arc laid over it is what this was first, and the two ends of that arc are steps: a
# machined ring has no seam in it, so the grade has to go all the way round.
COVER_LIP = 0.72  # the machined lip along its inner edge, which is what gives the assembly a
# depth: the blades run into a shadow and the ring above them catches the light
COVER_MUTED = 2.0  # how hard the specular is held back while any of him is still showing, as the
# power `shut` is raised to. The eye is a display and the cover is not: a near-white band on the
# steel while the core is still in shot would make the metal the brightest thing on a face that
# is meant to be the only lit part of the picture. It comes up as the lid arrives, which is also
# what a plate swinging into a light actually does, and by the time it is at full there is no eye
# left to lose.

COVER_SHUT_S = 0.80  # how long it takes to wind across, and the slower of the two on purpose.
# Shutting is the gesture - he is going to sleep - and a lid that falls in a fifth of a second is
# a shutter rather than a cover.
COVER_OPEN_S = 0.42  # ...and how long to clear him again. Waking is not a deliberation.
COVER_HEFT = 0.80  # the shut's travel, pushed towards its own start before the ease. Under 1 the
# set breaks hard and then settles into the seal: three quarters of the throw in the first half
# of the window, and the last of it creeping home. That is a driven mechanism arriving at a stop
# rather than a wipe, and it is the shape a strip of this actually shows - measured on eight
# frames, the aperture goes 1.00, 0.82, 0.60, 0.40, 0.24, 0.12, 0.03, 0, which is a picture of
# something *closing* rather than eight evenly spaced holes. Linear is a wipe, and pushing the
# other way (a slow break and a slam) spends five of those eight frames on an eye with nothing
# in front of it - which was tried, and reads as a lid that sticks.
COVER_SPRING = 0.55  # ...and the open's, pushed harder the same way: a fifth of the steel is off
# him in the first twentieth of the window. What you see of an open is its first three frames.

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
    cover: float = 0.0  # whether the steel is across him: 0 wound clear, 1 shut. Not a fraction
    # anybody sets in between - it is a destination, and :meth:`EyeEngine.shut` is what takes its
    # time getting there. A mood at 0 is drawn exactly as it was before the cover existed.

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
                # `cover` is not eased, and it is the clearest case on the line: it is a
                # destination rather than an amplitude, and something else owns the journey.
                # Halfway through a crossfade the cover is not half across - it is wherever
                # :meth:`EyeEngine.shut` has wound it to on a clock twice as long as this one,
                # which is the only place that answer can live if the close is to look mechanical
                # rather than to inherit the colour fade's timing.
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

    def steel(self, strength: float) -> tuple:
        """Metal at *strength*, mixed towards the screen's own black and drawn opaque.

        Deliberately not :meth:`shade`, on two counts. It does not wear the mood's colour, because
        a cover that went amber with him would be a drawing of a cover; and it is not scaled by
        ``lit``, because ``rings`` is how present *he* is - a tap brings the phosphor up to full
        and must not also polish the lid over it.
        """
        return linear(mix(self.screen, STEEL, strength))

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

    def cover(self, rad: float, shut: float) -> None:
        """The steel across him: :data:`COVER_N` rigid blades wound about their own fixed pivots.

        Real diaphragm kinematics, and the reason none of this is a mask or a hole that grows -
        see :data:`COVER_PIVOT` for the construction. *shut* becomes one angle, every blade turns
        by it about a pivot that never moves, and the aperture is whatever the rigid shapes leave
        in the middle. A blade cannot stretch between two frames because there is nowhere in the
        arithmetic for it to stretch: its cutting edge, its chamfer and its machined arc are each
        a fixed radius about a point carried at a fixed distance from the pivot, and the only
        thing any of them is a function of is the angle.

        What each blade *is* is the crescent between its own cutting circle and the next blade's -
        which is what a real one shows, since the rest of it is under its neighbour. That is not a
        saving, it is the only arrangement that works: the blades are a cyclic pile and a
        painter's algorithm cannot order one, so drawing them whole leaves the last two down
        covering the other four and a six-bladed iris comes out with two faces.

        And what each blade is *drawn* as is a fan of flat wedges along that crescent, because
        the shading has to run along the blade's own arc rather than across the disc - see
        :data:`COVER_SLICES`. Every wedge is a quad from the cutting circle out to whichever of
        the neighbour's circle and the rim it meets first, both solved rather than clipped, so the
        fan tiles the crescent exactly and no part of it needs a mask.

        The floor under all of it is not scenery: it runs from the aperture's widest corner out
        to the rim, so nothing of him can show between two blades however they land.
        """
        turn = math.radians(COVER_SHUT * shut)
        piv = rad * COVER_PIVOT
        out = 2.0 * piv * math.sin(turn / 2.0)  # how far a cutting arc's centre has swung out
        half = math.pi / COVER_N
        # What the n of them leave: `rad - out` across the aperture's flats, and this to its
        # corners, which is where the floor has to start if it is never to intrude on the hole.
        corner = math.sqrt(max(0.0, rad * rad - (out * math.sin(half)) ** 2)) - out * math.cos(half)
        floor = self.steel(COVER_FLOOR)
        if corner <= 0.0:
            self.d.ellipse(self.box(rad), fill=floor)
        else:
            self.d.ellipse(self.box(rad), outline=floor, width=max(1, round(rad - corner)))

        # Where each blade's cutting arc is centred. All of them before any of them is drawn: a
        # blade is defined against its neighbour, so they have to exist first.
        hubs = []
        for i in range(COVER_N):
            phi = math.radians(COVER_AT + i * 360.0 / COVER_N)
            hinge = phi + math.pi + turn
            hubs.append((self.cx + piv * (math.cos(phi) + math.cos(hinge)),
                         self.cy + piv * (math.sin(phi) + math.sin(hinge))))

        for i, hub in enumerate(hubs):
            self._blade(rad, hub, hubs[(i + 1) % COVER_N], shut)
        self._collar(rad, corner)

    def _blade(self, rim: float, hub: tuple, near: tuple, shut: float) -> None:
        """One blade: the fan of wedges, then the seam that its cutting edge makes with the next.

        The seam is three hairlines and they are three different things - the chamfer catching
        the lamp on this plate, the dark under its thickness, and his own light coming past it
        onto the plate below. One of those alone is a drawn line; the three together are an edge.
        """
        gap = math.hypot(near[0] - hub[0], near[1] - hub[1])
        if gap <= 0.0:
            return
        towards = math.atan2(near[1] - hub[1], near[0] - hub[0])
        reach = math.acos(max(-1.0, min(1.0, gap / (2.0 * rim))))
        far = math.hypot(hub[0] - self.cx, hub[1] - self.cy)
        at = math.atan2(hub[1] - self.cy, hub[0] - self.cx)
        edge = []
        for k in range(COVER_SLICES + 1):
            a = towards + reach * (2.0 * k / COVER_SLICES - 1.0)
            # How far this blade reaches along the ray at *a*: to the neighbour's cutting circle,
            # or to the rim if that comes first. Both are one quadratic; neither is a clip.
            side, held = math.sin(a - towards) * gap, math.cos(a - towards) * gap
            stop = held + math.sqrt(max(0.0, rim * rim - side * side))
            side, held = math.sin(a - at) * far, math.cos(a - at) * far
            stop = min(stop, math.sqrt(max(0.0, rim * rim - side * side)) - held)
            edge.append((a, max(rim, stop)))
        for k in range(COVER_SLICES):
            (a0, r0), (a1, r1) = edge[k], edge[k + 1]
            if r0 <= rim and r1 <= rim:
                continue  # this wedge is entirely past the rim, or past the neighbour
            self.d.polygon([(hub[0] + rim * math.cos(a0), hub[1] + rim * math.sin(a0)),
                            (hub[0] + r0 * math.cos(a0), hub[1] + r0 * math.sin(a0)),
                            (hub[0] + r1 * math.cos(a1), hub[1] + r1 * math.sin(a1)),
                            (hub[0] + rim * math.cos(a1), hub[1] + rim * math.sin(a1))],
                           fill=self.steel(self._brushed((a0 + a1) / 2.0, k, shut)))
        self._score(rim, hub, rim * COVER_GROOVE, near,
                    lambda _: self._brushed(towards, 0, shut) * COVER_MACHINED)
        self._score(rim, hub, rim + self.thin, near, lambda _: COVER_EDGE_HELD
                    + (COVER_EDGE - COVER_EDGE_HELD) * self._arrived(shut))
        self._score(rim, hub, rim, near, lambda _: COVER_GAP)
        # ...and his own light on the plate below, brightest at the hub end and gone by the rim.
        self._score(rim, hub, rim - self.thin, near,
                    lambda t: COVER_SEAM * (COVER_SEAM_FADE + (1.0 - COVER_SEAM_FADE) * t),
                    tint=True)

    def _brushed(self, a: float, k: int, shut: float) -> float:
        """How bright the blade is at *a* along its own arc: diffuse, specular, and the grain.

        A brushed surface's value is a function of one angle - the one round its brush marks -
        and here those run along the arc, so this is that angle and nothing else. The blades'
        arcs are centred sixty degrees apart, so the same function gives six different answers
        and the highlight walks the spiral instead of landing on all six at once.

        The specular is held back until the lid is nearly home; see :data:`COVER_MUTED`.
        """
        off = (a - math.radians(COVER_LAMP) + math.pi) % TAU - math.pi
        lit = COVER_DARK + (COVER_LIGHT - COVER_DARK) * 0.5 * (1.0 + math.cos(off))
        lit += COVER_SPEC * self._arrived(shut) * math.exp(-((off / COVER_GLINT) ** 2))
        lit += COVER_GRAIN * (((k + 1) * BLINK_DRIFT % 1.0) - 0.5)
        # Nothing on the steel may go brighter than the chamfer is allowed to be - see
        # :meth:`_arrived`. The band is the one thing here that can reach the top of the tube on
        # its own, and it is held to the same ceiling as the line that is meant to be brightest.
        # After the grain and not before it: a ceiling a jitter can climb over is not one.
        return min(lit, COVER_EDGE_HELD + (COVER_EDGE - COVER_EDGE_HELD) * self._arrived(shut))

    def _arrived(self, shut: float) -> float:
        """How much of its own light the cover is allowed, 0 while he is showing .. 1 once shut.

        The eye is a display and the cover is not, and that is the whole licence for the metal
        being here: a near-white band on the steel while the core is still in shot would make the
        lid the brightest thing on a face that is meant to be the only lit part of the picture.
        Both of the cover's whites wait on this. It is also what a plate swinging into a light
        actually does, so nothing is being fought - and by the time it is at full there is no eye
        left to lose.
        """
        return shut**COVER_MUTED

    def _score(self, rim: float, hub: tuple, rad: float, near: tuple,
               tone: Callable[[float], float], tint: bool = False) -> None:
        """A line along the circle at *hub, rad*, over the part of it that is on this blade.

        Which is the part inside the rim - the neighbour's circle bounds the far end of it, and
        the span here is that. Dropped rather than clamped: a seam pulled onto the rim would draw
        itself along it. *tone* is asked for a value per piece, 0 at the rim end and 1 at the hub
        end, so a seam can fade along its length for the cost of drawing it in pieces.
        """
        span = math.hypot(near[0] - hub[0], near[1] - hub[1])
        if span <= 0.0:
            return
        reach = math.acos(max(-1.0, min(1.0, (rad * rad + span * span - rim * rim)
                                        / (2.0 * rad * span))))
        towards = math.atan2(near[1] - hub[1], near[0] - hub[0])
        run = [p for p in self._arc(hub, rad, towards - reach, towards + reach)
               if math.hypot(p[0] - self.cx, p[1] - self.cy) <= rim]
        if len(run) < 2:
            return
        hair = max(1, round(self.thin))
        step = max(1, len(run) // COVER_BLEED_N)
        for k in range(0, len(run) - 1, step):
            piece = run[k:k + step + 1]
            at = tone(min(1.0, (k + step) / max(1, len(run) - 1)))
            self.d.line(piece, fill=self.shade(at) if tint else self.steel(at), width=hair)

    def _arc(self, hub: tuple, rad: float, a0: float, a1: float) -> list:
        """Points along a circle about *hub* - not about the pen - with angles in radians.

        The cover's own primitive, and the reason it is not :meth:`arc_pts`: every curve on a
        blade is drawn about a point that blade carries rather than about the eye's centre, which
        is the whole of what makes it a rigid shape turning instead of a hole growing.
        """
        n = max(2, round(abs(a1 - a0) / math.radians(COVER_STEP)))
        return [(hub[0] + rad * math.cos(a0 + (a1 - a0) * k / n),
                 hub[1] + rad * math.sin(a0 + (a1 - a0) * k / n)) for k in range(n + 1)]

    def _collar(self, rad: float, corner: float) -> None:
        """The ring the whole assembly is set in: a rim, and pointedly not a mount.

        It only ever lies on the floor plate, never on him - its width is clipped to what the
        floor has reached, so it grows in from the rim with the cover rather than landing on his
        rings at full width the moment the lid starts to move.
        """
        wall = max(1.0, min(rad * (1.0 - COVER_SEAT), rad - corner))
        hair = max(1, round(self.hair))
        # A true circle first, so the eye's own outline is never a polygon, and the grade laid
        # inside it: a thick polyline's outer edge is a chord, and that reads as facets round him.
        self.d.ellipse(self.box(rad), outline=self.steel(COVER_COLLAR_LO),
                       width=max(1, round(wall)))
        inner = max(1.0, wall - hair)
        for k in range(COVER_COLLAR_N):
            a0 = k * 360.0 / COVER_COLLAR_N
            self.d.line(
                # A degree of overlap on each, so the ring has no gaps where its own grade is cut.
                self.arc_pts(rad - hair - inner / 2.0, a0, a0 + 360.0 / COVER_COLLAR_N + 1.0, 4),
                fill=self.steel(COVER_COLLAR_LO + (COVER_COLLAR - COVER_COLLAR_LO) * 0.5 * (
                    1.0 + math.cos(math.radians(a0 + 180.0 / COVER_COLLAR_N - COVER_LAMP)))),
                width=max(1, round(inner)))
        self.d.line(self.arc_pts(rad - wall, 0.0, 360.0, 64), fill=self.steel(COVER_LIP),
                    width=hair)


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
        # ...and the cover, which has its own of all three. It starts where the resting mood
        # leaves it rather than open, so a box that boots asleep boots shut instead of winding
        # itself closed in front of whoever just switched it on.
        self._shut = self._was = self._want = max(0.0, min(1.0, resting.cover))
        self._since = 0.0
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

    def shut(self, mood: Mood, phase: float) -> float:
        """How far the cover has wound across right now: 0 clear of him, 1 shut.

        On its own clock and not the mood crossfade's, which is the whole of why it reads as a
        mechanism. The colour fade is 0.45 s because a face changing its mind should not snap;
        the cover is twice that going one way and half again the other, because it is a lump of
        steel being driven and the two directions are not the same gesture - see COVER_SHUT_S.

        Neither end is linear. `k**bias` before the smoothstep pushes the speed to one end of the
        travel: shutting breaks away slowly and arrives with the speed coming off it, and opening
        breaks hard and spends the tail of its window creeping out of a picture it has already
        left. Both start and end at rest, so nothing steps on the frame the state changes.

        Stateful, unlike :meth:`aperture`, and it has to be: what it needs to know is *when* the
        target last changed, and a mood does not carry that. It is the same three fields
        :meth:`look` keeps for the crossfade, kept for the same reason.
        """
        want = max(0.0, min(1.0, mood.cover))
        if want != self._want:
            self._was, self._want, self._since = self._shut, want, phase
        shutting = want > self._was
        span = COVER_SHUT_S if shutting else COVER_OPEN_S
        k = 1.0 if span <= 0.0 else max(0.0, min(1.0, (phase - self._since) / span))
        k = k ** (COVER_HEFT if shutting else COVER_SPRING)
        self._shut = self._was + (want - self._was) * k * k * (3.0 - 2.0 * k)
        return self._shut

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
        shut = self.shut(mood, phase)
        tile = smoothed(self._size, lambda d: self._draw(d, mood, phase, level, shut))
        img.alpha_composite(tile, (cx - self.r, cy - self.r))

    def _draw(self, d: ImageDraw.ImageDraw, mood: Mood, phase: float, level: float,
              shut: float = 0.0) -> None:
        """The whole eye at the centre of its tile: the socket, then the eye, then the glass.

        ...and then the steel over all three, when there is any. The two halves are exclusive at
        the ends and that is the point: with the cover clear, not a line below runs and he is the
        drawing he has always been; with it shut, the face underneath is not drawn at all, so the
        one state that used to cost a full eye now costs six polygons and a disc. Only the
        crossing pays for both, and it is under a second.
        """
        # `rings` is not clamped at 1: it is how present he is, and the parts below are drawn
        # well short of full on purpose (see the *_LIT block), so "brighter than usual" needs
        # somewhere above 1 to go. That is what the tap acknowledgement uses. `Pen.shade` mixes
        # through `mix`, which clamps, so an absurd value saturates rather than raising.
        pen = Pen(d, self._c, self._c, self.screen, mood.tint,
                  max(0.0, mood.rings), self._stroke, self._thin)
        if shut < 1.0:
            # The gaze first, because the socket is geared to it now - see RING_GEAR. It is still
            # one call: the shell is handed the answer rather than asking for its own.
            gx, gy = gaze_at(phase, mood, self.places)
            self._shell(pen, mood, phase, gx)
            self._optic(pen.shifted(gx * self._c * GAZE_SHIFT, gy * self._c * GAZE_SHIFT),
                        mood, phase, level, gx, gy)
            # The brow last, and on the *unshifted* pen. It is a highlight on the outer glass, and
            # a highlight does not travel with what is under it - which is the whole depth cue,
            # and the reason the eye reads as turning rather than as sliding about.
            pen.band(self._c * BROW_AT, BROW_FROM, BROW_TO, BROW_LIT, self._thin * BROW_W)
        if shut > 0.0:
            pen.cover(self._c, shut)

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
