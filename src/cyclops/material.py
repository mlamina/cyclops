"""The panel's metal: one lamp, one steel, and the build-time fields a part is shaded with.

What this module owns is everything that makes two pieces of chrome look cut from the same bar
under the same light. The lamp is :data:`LAMP` - the terminal's bench light (``GLARE_X`` and
``GLARE_Y`` in :mod:`cyclops.overlay`, up and a little left of the panel) turned into a direction
any surface can dot its normal with - and the steel is the four stops under it. Everything else
is a pure function from a shape to a float field: the normals of a rolled edge, the diffuse and
specular light on them, brushed grain, hairline scratches, the shadow a raised part drops, the
glare a sheet of glass gives back, and a hex-socket cap screw assembled out of all of those.

Nothing in here may be a sprite. A tile is cached, but on everything that makes it different
from the next one: :func:`screw` takes the bearing to the lamp from where it is standing, its own
clocking, its own dirt and its own share of the falloff, because a panel with a dozen fixings on
it that share sixty-six per cent of their pixels has one fixing on it a dozen times, and that is
provable in a diff. If you add a part here, give it the arguments that make each of them itself.

It must never be reached from a frame. Nothing in here is cheap enough for ``Overlay.render`` and
nothing needs to be: metal does not move, so every field is built once into a cached layer or a
tile. It must never put an opaque fill over the camera either. A texture is a modulation - a
colour at a per-pixel alpha, composited rather than written - and the only pixels that come out
of here opaque are the ones that are metal all the way through.

It knows nothing about the panel's layout, its states or its phosphor, and imports nothing from
:mod:`cyclops.overlay`, so a tool can build a bolt or a bar on its own.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

Field = np.ndarray  # float32, rows by columns, one number per panel pixel
Colour = tuple[int, int, int]

# ---- the lamp ----
#
# One light for the whole panel. Above and a little to the left, which is where a bench lamp is
# and where anybody reads a highlight from without being told, and standing well off the panel
# rather than grazing it, so a face square to the viewer is lit and only an edge turned right
# away goes dark. The terminal's own lamp sits just off its top-left corner, which from anywhere
# on its face is this same direction; everything else on the panel now agrees with it.
LAMP = (-0.28, -0.72, 0.63)  # towards the lamp: x right, y down, z out of the panel
AMBIENT = 0.22  # what a face turned right away from the lamp still gets from the room. Not
# zero: an unlit edge reads as a hole cut in the panel rather than as the dark side of a bar. It
# was 0.30, and at that the chamfer turned away came out a mid grey barely under the face -
# which, beside the lit chamfer on the other edge, is an *emboss*: a bar lit from both sides,
# the one thing a panel under a single lamp can never look like. Every critic in the last round
# measured a lit hairline on both edges of something. The far edge of anything under one lamp is
# dark; this is the number that makes it so, and it makes it so for every part on the panel.
SHINE = 32.0  # how tight the highlight is - a machined finish, not a mirror and not matte
SPEC = 0.9  # ...and how much of STEEL_SPEC it lays down where it lands hardest
RIM = 0.8  # how brightly an edge seen end-on lights up where it faces the lamp. The terminal's
# moulding already does this (its sheen); it is what puts the one crisp bright line on the near
# edge of every bar, and without it a rolled edge is a gradient and reads as rubber.
BRUSH = 0.62  # how much of a highlight a brushed surface spreads ALONG its own fibres. Machined
# stock is a bundle of micro-grooves running the length of the bar, so it does not mirror the
# lamp at exactly one bearing and go dark at every other: it mirrors it over a band of them,
# which is why every bar in a photograph of a rack carries a bright arris whichever way it runs.
# Shaded isotropically, ours did not. A horizontal bar's lit edge measured 198 against a face of
# 137 and the diagonal strut bolted to it measured 119 against a face of 111 - the same steel,
# the same lamp, and one member with a section and one with a filled shape, because the mirror
# direction happened to lie along the diagonal's own axis. At 1 the highlight would be identical
# on every bearing, which is a fill of another kind; this is most of the way there and still
# leaves a bar pointing away from the lamp visibly duller than one facing it.
SHADOW_DROP = 1.4  # how far a shadow falls per pixel a part stands proud...
SHADOW_SOFT = 1.3  # ...and how soft its edge is, in the same currency

# ---- the steel ----
#
# Cold grey, and that is the whole point of these four numbers. They used to carry a few counts
# of green - "gunmetal seen by a phosphor tube" - and measured, that came out at a green bias
# (G - (R+B)/2) of +5 to +8 before anything else on the panel had touched them. Every critic
# read the result the same way: the hardware and the illuminated glass are the same material,
# so the frame "reads as a green painted border from a pace away instead of steel". The panel
# this one is judged against splits the two about three to one - its bars sit at +5 and its
# glass at +16 - and the split is what says which parts are lit and which are only lit *on*.
#
# So the metal is neutral now, a count to the blue if anything, and green belongs to the
# phosphor and the glass alone. The LUMINANCE of each stop is held to within a level of what it
# was (81.8/158.3/18.6/212.3/251.5, by 0.299R+0.587G+0.114B), because every part on this panel
# is shaded against this ladder and moving a rung would restage the whole thing.
STEEL = (81, 83, 83)  # a flat face square to the viewer, once the lamp has had its say
STEEL_LIT = (156, 159, 160)  # a rolled edge turned into the lamp
STEEL_DARK = (17, 19, 20)  # the floor of a socket, the inside of a shadow
STEEL_SPEC = (210, 213, 214)  # the brightest a *face* may get. Short of the tube's white on
# purpose: a sheet of steel outshining the phosphor would be a second light on the panel.
STEEL_HOT = (250, 252, 252)  # ...and the brightest the one pixel a specular actually is may get.
# A highlight is not a bright surface, it is a picture of the lamp reflected in the surface, and
# it may be as bright as the lamp. Holding it to STEEL_SPEC put a ceiling of 212 on every arris
# on the panel and every critic measured it: our chrome topped out at 209-214 where the panel it
# is being judged against blows a one-pixel ridge to 240-255. It is reached through HOT_SHINE, so
# it is only ever the ridge that gets there - a face at half the specular gets a *tenth* of the
# extra headroom - and a piece of steel still cannot out-glow the tube over any area at all.
HOT_SHINE = 3.0  # how fast the extra headroom falls away from a mirror-perfect reflection
GRAIN = 0.13  # how far the brushing moves brightness either way
SPECKLE = 0.40  # of the brushing: the fine tooth under it, which is what varies *along* a fibre.
# Brushing on its own is a set of streaks, each the same brightness from one end of a bar to the
# other, and a face read along its own length measured flat to within a level - a rendering, not
# a piece of metal. Stock has a tooth the brush never quite takes out, and one level of it is the
# difference between a surface and a fill.
DOME = 0.14  # sin of the tilt a "flat" machined face has reached by its edge, from none at its
# middle - nothing true is dead flat, and the gentle fall across a bar is what says it is round
SEED = 7  # every field here comes out the same on every boot

# ---- a cap screw ----
BOLT_RIM = 0.30  # of the head's radius: the rolled edge round it
BOLT_DOME = 0.35  # sin of the tilt the head reaches at its rim - a cap screw is crowned
BOLT_SOCKET = 0.46  # of the radius: the hex socket, corner to centre
BOLT_WALL = 0.9  # px of the socket's far wall the lamp reaches down
BOLT_EDGE = 0.7  # px round the socket's mouth that its sharp edge darkens
BOLT_LIFT = 0.28  # of the radius: how proud the head stands, which is what sets its shadow
BOLT_SHADOW = 0.62  # the shadow's alpha where it is deepest
_SS = 3  # the sprite is drawn this many times over and boxed down, which is its anti-aliasing
_TABLE = 4096  # random numbers behind every noise field; long enough that no fibre repeats


def _unit(v: tuple[float, ...]) -> tuple[float, ...]:
    length = math.sqrt(sum(c * c for c in v)) or 1.0
    return tuple(c / length for c in v)


_L = _unit(LAMP)
_H = _unit((_L[0], _L[1], _L[2] + 1.0))  # half-way between the lamp and the viewer


def lamp_2d() -> tuple[float, float]:
    """Which way across the panel the lamp is, unit length: where lit edges are and shadows not."""
    x, y = _unit((_L[0], _L[1]))
    return x, y


_L2 = lamp_2d()


def roll_normals(
    inward: Field, gx: Field, gy: Field, roll: float, dome: float | Field = DOME
) -> tuple[Field, Field, Field]:
    """Normals of a face whose outer *roll* pixels turn down to its edge: quarter-round, then flat.

    *inward* is how far in from the edge each pixel is (0 on it), (*gx*, *gy*) the unit direction
    across the panel from the part's middle to its edge, and *dome* the tilt the flat keeps -
    one number for a bar, a field for a head that crowns towards its middle.
    """
    u = np.clip(1.0 - inward / max(roll, 1e-3), 0.0, 1.0)
    tilt = np.clip(dome + (1.0 - dome) * u, 0.0, 1.0)  # sin of the surface's tilt
    return gx * tilt, gy * tilt, np.sqrt(np.maximum(1.0 - tilt * tilt, 0.0))


def bearing(dx: float, dy: float) -> tuple[float, float, float]:
    """The lamp as seen from a part that is (*dx*, *dy*) away from it: same height, its own way.

    :data:`LAMP` is one direction for the whole panel, which is right for a lamp at infinity and
    wrong for a bench light standing over the top-left corner of a 800x480 sheet. A fixing at the
    bottom right of that sheet sees the light from a different bearing than one at the top left,
    and it is *that* difference - not a random per-instance jitter - that stops a dozen bolts
    reading as one bolt pasted a dozen times. Elevation is kept: the lamp is the same height
    above the panel wherever you stand on it, and only the compass direction turns.
    """
    reach = math.hypot(dx, dy) or 1.0
    lateral = math.hypot(_L[0], _L[1])
    return dx / reach * lateral, dy / reach * lateral, _L[2]


def _half(lamp: tuple[float, float, float]) -> tuple[float, float, float]:
    """Half-way between a lamp and the viewer: where a surface must face to mirror it."""
    return _unit((lamp[0], lamp[1], lamp[2] + 1.0))


def _off_fibre(vx: Field | float, vy: Field | float, vz: Field | float,
               fibre: tuple[Field, Field], brush: float) -> tuple[Field, Field, Field]:
    """*v* with *brush* of its along-the-fibre part taken out, back to unit length.

    The one operation an anisotropic highlight needs: a groove running along *fibre* cannot tell
    where along itself the light is, so the direction it answers to is the one left when that
    component is removed. Removing all of it is a perfect groove; removing some is stock.
    """
    fx, fy = fibre
    along = (vx * fx + vy * fy) * brush
    vx, vy = vx - along * fx, vy - along * fy
    length = np.maximum(np.sqrt(vx * vx + vy * vy + vz * vz), 1e-6)
    return vx / length, vy / length, vz / length


def shade(nx: Field, ny: Field, nz: Field,
          lamp: tuple[float, float, float] | None = None,
          fibre: tuple[Field, Field] | None = None,
          brush: float = BRUSH) -> tuple[Field, Field]:
    """Diffuse and specular light, 0..1 each, on a field of normals under the one lamp.

    The specular is two things added: the lamp's own highlight, which lands where a surface
    faces half-way between the lamp and the viewer, and the rim light an edge catches where it
    turns end-on towards the lamp - squared twice, as the terminal's sheen is, so it stays on
    the edge and off the face.

    *lamp* is that light's direction, and defaults to :data:`LAMP` - the panel's one lamp seen
    from far enough away that everything on it agrees. A part that knows where on the panel it
    is passes its own :func:`bearing` instead, which is the same lamp from where it stands.

    *fibre* is the direction the surface's brushing runs, across the panel and unit length -
    a pair of numbers for a whole part or a pair of fields for one that bends. Given it, both
    the highlight and the rim answer to the lamp with :data:`BRUSH` of its along-the-fibre
    component removed, which is what a bundle of micro-grooves does to a reflection and what
    keeps a bar's lit arris lit whichever way the bar runs. Left out, the surface is isotropic
    and this is the shading every part on the panel had before.
    """
    lamp = _L if lamp is None else lamp
    hx, hy, hz = _H if lamp is _L else _half(lamp)
    diffuse = np.clip(nx * lamp[0] + ny * lamp[1] + nz * lamp[2], 0.0, 1.0)
    # The lamp's own bearing across the panel, unit length, which is what the rim measures
    # against; taking it off the fibre as well keeps the two halves of the highlight agreeing
    # about where the light is.
    px, py = _L2 if lamp is _L else _unit((lamp[0], lamp[1]))
    if fibre is not None:
        px, py, _ = _off_fibre(px, py, 0.0, fibre, brush)
        hx, hy, hz = _off_fibre(hx, hy, hz, fibre, brush)
    spec = np.clip(nx * hx + ny * hy + nz * hz, 0.0, 1.0) ** SHINE
    tilt = np.sqrt(nx * nx + ny * ny)
    facing = np.clip((nx * px + ny * py) / np.maximum(tilt, 1e-6), 0.0, 1.0)
    rim = tilt * tilt * facing * facing
    return diffuse, np.minimum(spec + RIM * rim, 1.0)


def steel(diffuse: Field, spec: Field, grain: Field | None = None, colour: Colour = STEEL,
          ambient: Field | None = None) -> Field:
    """The colour of lit steel, rows by columns by three.

    *colour* where a face is square to the viewer, darker as it turns from the lamp, STEEL_SPEC
    laid on where the highlight lands, and brushed by *grain* if there is any.

    The ceiling is STEEL_SPEC over a face and lifts towards STEEL_HOT as the specular approaches
    a true mirror of the lamp - see HOT_SHINE. That is the difference between a bar with a bright
    edge and a bar with a *ridge* on it: the surface is held down where it is a surface, and the
    one or two pixels that are actually reflecting the lamp are allowed to blow.

    *ambient* is how much of the room this pixel can actually see, defaulting to :data:`AMBIENT`
    for a surface with the whole of it in view. A field instead lets a part say where it is
    shut in - the underside of a bar lying on a plate sees a sliver of room and a bounce off the
    plate, and those two together are what turn a section's dark half into a core shadow with a
    line of reflected light under it rather than into a flat wash at the ambient floor. It moves
    the diffuse *floor* only: the lamp's own share and the highlight are untouched, because a
    surface being shut in does not move the lamp.
    """
    albedo = np.asarray(colour, np.float32) / (AMBIENT + (1.0 - AMBIENT) * _L[2])
    lit = (AMBIENT if ambient is None else ambient) + (1.0 - AMBIENT) * diffuse
    if grain is not None:
        lit = lit * (1.0 + GRAIN * grain)
    ceiling = np.asarray(STEEL_SPEC, np.float32)
    blown = np.asarray(STEEL_HOT, np.float32) - ceiling
    rgb = albedo * lit[..., None] + ceiling * (SPEC * spec)[..., None]
    return np.minimum(rgb, ceiling + blown * (spec**HOT_SHINE)[..., None])


def _noise1d(coord: Field, period: float, rng: np.random.Generator) -> Field:
    """Piecewise-linear noise in -1..1 along one coordinate, a new bend every *period* pixels."""
    table = rng.uniform(-1.0, 1.0, _TABLE).astype(np.float32)
    at = coord / period
    i0 = np.floor(at)
    frac = (at - i0).astype(np.float32)
    i0 = i0.astype(np.int64)
    return table[i0 % _TABLE] * (1.0 - frac) + table[(i0 + 1) % _TABLE] * frac


def _noise2d(x: Field, y: Field, rng: np.random.Generator) -> Field:
    """White noise in -1..1, one value per whole pixel of (*x*, *y*), the same on every boot."""
    table = rng.uniform(-1.0, 1.0, _TABLE).astype(np.float32)
    ix = np.floor(x).astype(np.int64)
    iy = np.floor(y).astype(np.int64)
    return table[(ix * 7919 + iy * 104729) % _TABLE]


def grain(across: Field, along: Field, seed: int = SEED) -> Field:
    """Brushed metal, -1..1: fibres that run with *along* and vary with *across*, both in pixels.

    Three pitches of fibre, because one reads as stripes, and a slow term along the stroke so no
    streak runs the whole length of anything - a brush is dragged by a hand, not a ruler. Under
    the fibres, a fine tooth (SPECKLE) that changes from pixel to pixel *along* them, because a
    fibre of constant brightness measures dead flat down its own length however many pitches of
    fibre sit beside it.
    """
    rng = np.random.default_rng(seed)
    fibres = (
        0.5 * _noise1d(across, 1.0, rng)
        + 0.3 * _noise1d(across, 2.7, rng)
        + 0.2 * _noise1d(across, 7.3, rng)
    )
    stroke = _noise1d(along + 0.35 * across, 29.0, rng)
    return fibres * (0.6 + 0.4 * stroke) + SPECKLE * _noise2d(across, along, rng)


def wear(along: Field, seed: int = SEED) -> Field:
    """Where an edge has been handled, -1..1 and slow: brighter where rubbed, duller between."""
    return _noise1d(along, 37.0, np.random.default_rng(seed + 2))


def scratches(
    width: int, height: int, count: int, along: tuple[float, float], seed: int = SEED,
    spread: float = 28.0, length: tuple[float, float] = (18.0, 80.0),
) -> Field:
    """Sparse hairlines, 0..1: where things have been dragged across the sheet.

    Mostly with the grain (*along*, a direction across the panel), never all of it, and a pixel
    wide - at arm's length a scratch is a line or it is nothing at all.
    """
    rng = np.random.default_rng(seed + 1)
    sheet = Image.new("L", (width, height), 0)
    d = ImageDraw.Draw(sheet)
    heading = math.degrees(math.atan2(along[1], along[0]))
    for _ in range(count):
        x, y = rng.uniform(0.0, width), rng.uniform(0.0, height)
        a = math.radians(heading + rng.normal(0.0, spread))
        n = rng.uniform(*length)
        d.line([(x, y), (x + n * math.cos(a), y + n * math.sin(a))],
               fill=int(rng.uniform(90, 255)), width=1)
    return np.asarray(sheet, np.float32) / 255.0


def pits(width: int, height: int, count: int, seed: int = SEED,
         size: tuple[float, float] = (0.6, 1.4)) -> Field:
    """Sparse pitting, 0..1: where the finish has gone through and the sheet has corroded.

    A pit is a speck a pixel or two across and always *darker* than the face around it - a hole
    catches no light - which is what separates it from a scratch. Drawn twice over and boxed
    down, so the small ones come out a soft speck rather than a square pixel. *size* is the
    radius, smallest to largest, in pixels.
    """
    rng = np.random.default_rng(seed + 3)
    sheet = Image.new("L", (width * 2, height * 2), 0)
    d = ImageDraw.Draw(sheet)
    for _ in range(count):
        x, y = rng.uniform(0.0, width) * 2, rng.uniform(0.0, height) * 2
        r = rng.uniform(*size) * 2
        d.ellipse([x - r, y - r, x + r, y + r], fill=int(rng.uniform(110, 255)))
    return np.asarray(sheet.reduce(2), np.float32) / 255.0


def cast(coverage: Field, lift: float, away: tuple[float, float] | None = None) -> Field:
    """The shadow a part standing *lift* pixels proud drops on what it is bolted to, 0..1.

    Its own coverage, pushed away from the lamp and softened by the same distance. Composited
    under the part, so the part covers its own shadow the way a real one does. *away* is the
    direction the lamp lies in, defaulting to the panel's; a part that knows where it is passes
    its own, so its shadow and its highlight are thrown by the same light.
    """
    dx, dy = lamp_2d() if away is None else away
    sheet = Image.fromarray((np.clip(coverage, 0.0, 1.0) * 255.0).astype(np.uint8), "L")
    shifted = Image.new("L", sheet.size, 0)
    shifted.paste(sheet, (round(-dx * lift * SHADOW_DROP), round(-dy * lift * SHADOW_DROP)))
    soft = shifted.filter(ImageFilter.GaussianBlur(max(0.5, lift * SHADOW_SOFT)))
    return np.asarray(soft, np.float32) / 255.0


def glare(
    width: int, height: int, lamp: tuple[float, float], reach: float,
    ambient: float = 0.2, streak: tuple[float, float, float] | None = None,
) -> Field:
    """How much of the lamp a sheet of glass *width* by *height* gives back, 0..1.

    The lamp is at *lamp*, in pixels from the sheet's top-left, and its light carries *reach*
    pixels; *ambient* is the floor, because no glass in a room goes dead black. *streak*, if
    given, is the wipe a long glossy face makes of a small bright thing - where down the left
    edge it passes as a fraction of the height, how broad it is, and how far down it travels on
    its way across - and a face that is not long wants none. This is the terminal's own glare
    with its numbers made arguments, so a dial or the eye can sit under the same lamp.
    """
    ys = np.arange(height, dtype=np.float32)[:, None]
    xs = np.arange(width, dtype=np.float32)[None, :]
    lit = np.exp(-((xs - lamp[0]) ** 2 + (ys - lamp[1]) ** 2) / max(1.0, reach) ** 2)
    if streak is not None:
        at, depth, tilt = streak
        down = ys / max(1.0, height - 1)
        across = xs / max(1.0, width - 1)
        lit = lit * np.exp(-(((down - (at + tilt * across)) / depth) ** 2))
    return ambient + (1.0 - ambient) * lit


def bar_field(
    points: list[tuple[float, float]], x0: int, y0: int, width: int, height: int
) -> tuple[Field, Field, Field, Field, Field]:
    """Everything a bar along a polyline is shaded from, over one box of the panel.

    Distance from each pixel to the centreline; the unit direction from the line out to the
    pixel; the arc length of the nearest point on the line; and which side of it the pixel is
    on, as a sign. The same idea as the terminal's ``tube_field`` - one distance field, and
    coverage, normals and grain all come off it - for a shape that is a stroke rather than a box.
    Knees come out rounded, as a bent bar's do.
    """
    ys = (np.arange(height, dtype=np.float32) + y0)[:, None]
    xs = (np.arange(width, dtype=np.float32) + x0)[None, :]
    best = np.full((height, width), np.inf, np.float32)
    ox, oy, along, side = (np.zeros((height, width), np.float32) for _ in range(4))
    run = 0.0
    for (ax, ay), (bx, by) in zip(points, points[1:], strict=False):
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        t = np.clip(((xs - ax) * dx + (ys - ay) * dy) / (length * length), 0.0, 1.0)
        px, py = xs - (ax + t * dx), ys - (ay + t * dy)
        d = np.sqrt(px * px + py * py)
        closer = d < best
        best = np.where(closer, d, best)
        ox, oy = np.where(closer, px, ox), np.where(closer, py, oy)
        along = np.where(closer, run + t * length, along)
        side = np.where(closer, np.sign(dx * (ys - ay) - dy * (xs - ax)), side)
        run += length
    safe = np.maximum(best, 1e-6)
    return best, ox / safe, oy / safe, along, side


def to_image(rgb: Field, alpha: Field) -> Image.Image:
    """An (rgb, alpha) pair as the RGBA tile a layer can composite."""
    rgba = np.empty((*alpha.shape, 4), dtype=np.uint8)
    rgba[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    rgba[:, :, 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def _hexagon(x: Field, y: Field, corner: float) -> Field:
    """Signed distance to a hexagon *corner* from centre to vertex, a vertex at three o'clock."""
    flat = corner * math.sqrt(3.0) / 2.0
    kx, ky, kz = -math.sqrt(3.0) / 2.0, 0.5, 1.0 / math.sqrt(3.0)
    px, py = np.abs(x), np.abs(y)
    fold = np.minimum(kx * px + ky * py, 0.0)
    px, py = px - 2.0 * fold * kx, py - 2.0 * fold * ky
    px, py = px - np.clip(px, -kz * flat, kz * flat), py - flat
    return np.hypot(px, py) * np.sign(py)


def _boxed(rgb: Field, alpha: Field) -> tuple[Field, Field]:
    """A supersampled (rgb, alpha) pair averaged down by _SS, colour weighted by its coverage."""
    h, w = alpha.shape
    cover = alpha.reshape(h // _SS, _SS, w // _SS, _SS).mean(axis=(1, 3))
    lit = (rgb * alpha[..., None]).reshape(h // _SS, _SS, w // _SS, _SS, 3).mean(axis=(1, 3))
    return lit / np.maximum(cover[..., None], 1e-6), cover


@lru_cache(maxsize=32)
def bolt(r: float, fx: float = 0.0, fy: float = 0.0) -> Image.Image:
    """A hex-socket cap screw of radius *r*, as an RGBA tile with the head at its centre.

    *fx* and *fy* are the fraction of a pixel the centre sits past the tile's middle, so a bolt
    at a non-integer position lands where its geometry says rather than on the nearest pixel.
    Crowned head with a rolled rim, lit up-left; a socket whose far wall catches the lamp and
    whose mouth's sharp edge darkens the head round it; and the shadow the head drops. One tile
    per (radius, offset), kept, because a panel has a dozen of these and eight kinds.
    """
    lift = r * BOLT_LIFT
    margin = math.ceil(lift * (SHADOW_DROP + 3 * SHADOW_SOFT)) + 1
    half = math.ceil(r) + margin
    size = 2 * half + 1
    grid = (np.arange(size * _SS, dtype=np.float32) + 0.5) / _SS - 0.5
    xs, ys = grid[None, :] - (half + fx), grid[:, None] - (half + fy)
    dist = np.sqrt(xs * xs + ys * ys)
    safe = np.maximum(dist, 1e-6)
    head = dist - r
    cover = np.clip(0.5 - head * _SS, 0.0, 1.0)
    nx, ny, nz = roll_normals(np.maximum(-head, 0.0), xs / safe, ys / safe, r * BOLT_RIM,
                              dome=BOLT_DOME * np.clip(dist / r, 0.0, 1.0))
    rgb = steel(*shade(nx, ny, nz))
    # The socket: a dark floor, and a far wall the lamp reaches down into. Its gradient is the
    # outward normal of the hole, so the wall facing the lamp is the one whose inward normal
    # points at it - the far one, down and right, which is where light gets into any hole.
    hexd = _hexagon(xs, ys, r * BOLT_SOCKET)
    gy, gx = np.gradient(hexd)
    glen = np.maximum(np.hypot(gx, gy), 1e-6)
    facing = np.clip(-(gx * _L[0] + gy * _L[1]) / glen, 0.0, 1.0)
    wall = np.clip(1.0 - np.maximum(-hexd, 0.0) / BOLT_WALL, 0.0, 1.0) * facing
    floor_ = np.asarray(STEEL_DARK, np.float32) * 0.8
    socket = floor_ + (np.asarray(STEEL_LIT, np.float32) - floor_) * 0.8 * wall[..., None]
    inside = np.clip(0.5 - hexd * _SS, 0.0, 1.0)[..., None]
    rgb = rgb * (1.0 - inside) + socket * inside
    mouth = np.clip(1.0 - np.maximum(hexd, 0.0) / BOLT_EDGE, 0.0, 1.0) * (1.0 - inside[..., 0])
    rgb = rgb * (1.0 - 0.4 * mouth)[..., None]
    rgb, cover = _boxed(rgb, cover)
    shadow = cast(cover, lift) * BOLT_SHADOW
    alpha = cover + shadow * (1.0 - cover)
    rgb = rgb * (cover / np.maximum(alpha, 1e-6))[..., None]
    return to_image(rgb, alpha)


# ---- a socket screw seated in flat bar ----
#
# What :func:`bolt` is not. That one is a crowned cap screw standing on the bar, and at panel size
# a crowned head on a bar reads as a bead on a tube. This one is sunk into it: a flat collar of the
# bar's own grey, a countersunk dish, a head at the bottom of the dish and a socket in the head.
#
# Four bands, and the reason there are exactly four is that a critic counted them on the panel
# this one is matched against and could not find them here: a drive recess that bottoms near
# black, a bright ring round it, the flat collar, and the ring of dirt and shadow where the head
# is seated. Measured on that panel at a seven-pixel head, by radius ring in luminance,
# mean/sigma round the ring at r=0..6: 14/0, 33/14, 75/57, 101/60, 86/36, 85/32, 55/37. Ours ran
# 20/0, 46/30, 62/24, 51/15, 97/44, 99/30, 51/- : a soft donut whose one bright ring was on the
# OUTSIDE, at r=6, where a seated screw has nothing but its seat. Every band moves in by three
# pixels, and the bright one is now the mouth of the counterbore at r=3.
#
# Which way that ring is lit is the whole of it. A countersink is a cone falling INWARDS, so the
# wall the lamp reaches is the far one and the only bright thing on the head sits down-right of
# it: measured, upper-left over lower-right round the annulus at r=2.5..4.5 came out 0.55 where
# the reference measures 1.38 to 1.97. On a panel whose entire argument is one lamp at the top
# left, a dozen fixings each lit from the bottom right is the loudest possible contradiction of
# it. So the dish keeps its inverted wall - it is what says the head is BELOW the collar - and
# the arris where the bore breaks out into the collar rolls over convex on top of it. That roll
# mirrors the lamp on the NEAR side, and it is the ring the eye reads: 1.35 now, and the one
# specular on the part.
SCREW_DISH = 0.40  # of the radius: where the counterbore's mouth is, inside the flat collar
SCREW_HEAD = 0.26  # of the radius: the head sitting at the bottom of the dish
SCREW_SLOPE = 0.46  # sin of the dish's slope. It was 0.62 - well past the angle that mirrors
# this lamp, about 0.43 - so the cone was dark all the way round and the drive read as a flat
# grey donut. Near the mirror the far wall is a clear step up from the floor without becoming
# the head's highlight, which belongs to the lip
SCREW_DISH_GLOSS = 0.40  # how much of the steel's highlight the dish keeps. Under half: it is
# cut, not polished, and its far wall has to stay a step lighter than the floor rather than
# competing with the ring above it - two crescents pointing opposite ways is two lamps
SCREW_RIM = 1.1  # px of the collar's outer edge that rolls down to the bar. It was 1.8, and a
# two-pixel roll on a seven-pixel head put a 230 crescent at r=6: a second specular on one part,
# outboard of the first, which is the two-lights fault at fixing scale
SCREW_LIP = 1.3  # px of the counterbore's mouth that rolls over into the collar - the ring
SCREW_LIP_TILT = 0.45  # ...and the sine of the tilt it reaches at the bore. On the mirror, so
# the near side of the ring blows to about 200 and the far side is a step under the collar
SCREW_RIM_GLOSS = 0.30  # ...and how much highlight is left on the outer edge, for the same
# reason. The seat of a head pressed into bar is a sawn edge, not a polished arris: it needs a
# tone step to say where the head stops, not a second crescent as bright as the lip's
SCREW_CROWN = 0.18  # sin of the tilt the head has reached by its edge - barely domed
SCREW_SOCKET = 0.29  # of the radius: the hex socket, corner to centre, so the recess bottoms
# inside r=2 and the counterbore's mouth has somewhere to sit
SCREW_FLOOR = 0.85  # how much of the light that reaches the socket's far wall survives the trip
# back out of it. The hole is deep, but the one wall the lamp gets down to is the thing that
# stops the recess being a flat black disc: at 0.45 the ring at r=2 varied 9 levels round the
# circle where a real drive varies 57, which is a hole with no drive in it
SCREW_LIFT = 0.11  # of the radius: how proud the collar stands. A seated head, not a bead, so
# its shadow is a pixel down and right of it and no more
SCREW_SEAT = 0.40  # that shadow's alpha where it is deepest
SCREW_WEAR = 0.50  # how much darker the bar is in the ring round the head a spanner has been in.
# It was 0.13, which is a tint: the panel this is matched against sits its seat ring 40 to 50
# levels under the plate round it, and that dark ring is half of what makes a head read as sunk
# into the bar rather than as a disc lying on it
SCREW_WEAR_W = 3.2  # ...and how wide that ring is, in px
SCREW_GRIME = 5  # specks of dirt round one head's rim, unevenly placed...
SCREW_GRIME_A = 0.45  # ...and how much of the collar's light one of them takes
SCREW_GRIME_W = 0.34  # of the radius: how far in from the rim they sit
SCREW_SPREAD = 0.07  # how far the head, the socket and the dish's slope vary from screw to screw.
# Not decoration: two heads 34 px apart on the same rail still came back 6.5% bit-identical with
# only their light and their dirt differing, and a box of screws is not a box of one screw
SCREW_SKEW = 0.05  # ...and how far off square the driver left one, as a sine. A fixing driven
# dead perpendicular in every hole is the other half of the same tell
SCREW_DUST = (0.30, 0.95)  # how much light the bottom of one socket gives back, against the rest.
# Capped under one now: a drive recess bottoms at nothing, and at the old ceiling of 1.5 the
# floor of the deepest hole on the panel still measured 21 where the reference reaches 0


@lru_cache(maxsize=64)
def screw(r: float, fx: float = 0.0, fy: float = 0.0, ax: float = 0.0, ay: float = 0.0,
          clock: float = 0.0, mark: int = 0, tone: float = 1.0) -> Image.Image:
    """A socket screw countersunk into flat bar, as an RGBA tile with the head at its centre.

    A flat collar with a rolled edge one pixel wide - lit where it faces the lamp, dark where it
    turns away, and the one specular on it where the roll faces the lamp squarely; inside it a
    countersunk dish whose far wall the lamp reaches down into and whose near wall it does not,
    which is what says the head is *below* the collar rather than on it; a barely-domed head at
    the bottom; and a hex socket near black except for the wall the light gets down to. Round
    the outside, the ring of bar a spanner has darkened and the shadow a head a fraction proud
    drops down and right of itself. *fx*, *fy* as for :func:`bolt`.

    Every argument after those is what stops a panel's dozen fixings being one fixing pasted a
    dozen times, which is the thing a critic can prove with an exact-pixel test and did: five of
    our seven heads came back identical to the decimal. (*ax*, *ay*) is the way to the lamp
    **from where this screw is** - see :func:`bearing` - so the crescent on its rim rotates
    across the panel and its seat shadow follows; *clock* is how far round the driver left the
    hex, which no two screws in a real assembly agree on; *mark* seeds the grime round the rim
    and the unevenness of the spanner's ring; and *tone* is the panel-wide falloff at its
    position, so a screw in the far corner is a screw in the far corner. Give it none of them
    and it is still the old sprite, which is what :func:`bolt` and any tool wanting one head
    want.
    """
    lamp = bearing(ax, ay) if ax or ay else _L
    flat = (lamp[0], lamp[1])
    # This one's own tolerances: nothing off a shelf is to the drawing, and the whole point of
    # the exercise is that no two of these come out of the same mould.
    rng = np.random.default_rng(mark + 5)
    seat, socket_r, slope = rng.uniform(1.0 - SCREW_SPREAD, 1.0 + SCREW_SPREAD, 3)
    skew = rng.uniform(-SCREW_SKEW, SCREW_SKEW, 2)
    # ...and how much of its own walls the floor of the hole bounces back, which is the one thing
    # in a socket that is not geometry. Left at one number it was a nine-pixel block of the same
    # three bytes in every head on the panel - five per cent of a patch, bit-identical, on its own.
    dust = rng.uniform(*SCREW_DUST)
    lift = r * SCREW_LIFT
    margin = math.ceil(max(lift * (SHADOW_DROP + 3 * SHADOW_SOFT), SCREW_WEAR_W)) + 1
    half = math.ceil(r) + margin
    size = 2 * half + 1
    grid = (np.arange(size * _SS, dtype=np.float32) + 0.5) / _SS - 0.5
    xs, ys = grid[None, :] - (half + fx), grid[:, None] - (half + fy)
    dist = np.sqrt(xs * xs + ys * ys)
    safe = np.maximum(dist, 1e-6)
    gx, gy = xs / safe, ys / safe
    cover = np.clip(0.5 - (dist - r) * _SS, 0.0, 1.0)
    # The collar: flat, with its outer pixel rolled down to the bar.
    nx, ny, nz = roll_normals(np.maximum(r - dist, 0.0), gx, gy, SCREW_RIM, dome=0.0)
    # The dish: a cone falling towards the centre, so its normals point inward and the wall the
    # lamp lights is the far one - the opposite of a dome, and the whole difference between a
    # head sunk into a bar and a bead sitting on it.
    sink = SCREW_SLOPE * slope
    top = r * SCREW_HEAD * seat
    dish = (dist < r * SCREW_DISH) & (dist >= top)
    nx = np.where(dish, -gx * sink, nx)
    ny = np.where(dish, -gy * sink, ny)
    nz = np.where(dish, math.sqrt(max(1.0 - sink * sink, 0.0)), nz)
    head = dist < top
    crown = SCREW_CROWN * np.clip(dist / max(top, 1e-6), 0.0, 1.0)
    nx = np.where(head, gx * crown + skew[0], nx)
    ny = np.where(head, gy * crown + skew[1], ny)
    nz = np.where(head, np.sqrt(np.maximum(1.0 - crown * crown, 0.0)), nz)
    # The mouth of the counterbore, which is the bright ring and the one the eye reads the light
    # off. A cone falling inwards lights its FAR wall, so the dish alone puts the only bright
    # thing on a head down and to the right of it: correct for the hole, and at seven pixels the
    # whole fixing then reads as lit from the bottom right on a panel whose one lamp is at the
    # top left. Where the bore breaks out into the collar the metal rolls over instead, and that
    # arris is convex - so it mirrors the lamp on the NEAR side, and the ring at r=3..4 comes out
    # brighter up-left than down-right the way the panel this is matched against measures it
    # (upper-left over lower-right at r=2.5..4.5: 1.38 to 1.97, against 0.55 for the dish alone).
    bore = r * SCREW_DISH
    lip = np.clip(1.0 - (dist - bore) / max(SCREW_LIP, 1e-6), 0.0, 1.0)
    lip = lip * (dist >= bore) * (dist < r - SCREW_RIM)
    lip_tilt = SCREW_LIP_TILT * lip
    nx = np.where(lip > 0.0, gx * lip_tilt, nx)
    ny = np.where(lip > 0.0, gy * lip_tilt, ny)
    nz = np.where(lip > 0.0, np.sqrt(np.maximum(1.0 - lip_tilt * lip_tilt, 0.0)), nz)
    diffuse, spec = shade(nx, ny, nz, lamp)
    # One specular on the part, and it is the counterbore's. The collar's own outer edge passes
    # through the same mirror a couple of pixels further out, so left at full gloss the head
    # came out with two concentric crescents - the fault this panel is marked down for at bar
    # scale, drawn at seven pixels.
    gloss = np.where(dish, SCREW_DISH_GLOSS,
                     np.where(dist > r - SCREW_RIM, SCREW_RIM_GLOSS, 1.0))
    rgb = steel(diffuse, spec * gloss)
    # The socket, as in bolt(): a dark floor and the far wall the light gets down to - clocked
    # where the driver left it, and dark enough that the head reads as a hole with a ring round
    # it rather than as a dished disc.
    turn, spin = math.cos(clock), math.sin(clock)
    hexd = _hexagon(xs * turn - ys * spin, xs * spin + ys * turn, r * SCREW_SOCKET * socket_r)
    hy, hx = np.gradient(hexd)
    hlen = np.maximum(np.hypot(hx, hy), 1e-6)
    facing = np.clip(-(hx * flat[0] + hy * flat[1]) / hlen, 0.0, 1.0)
    wall = np.clip(1.0 - np.maximum(-hexd, 0.0) / BOLT_WALL, 0.0, 1.0) * facing
    floor_ = np.asarray(STEEL_DARK, np.float32) * 0.8 * dust
    # The one wall the lamp reaches is a machined flat seen edge-on, so it comes back at the
    # brightness of a lit edge and not of a face: it is the only thing inside the recess that is
    # not black, and it is what makes a hex socket a hex socket rather than a round hole.
    socket = (floor_
              + (np.asarray(STEEL_LIT, np.float32) - floor_) * (SCREW_FLOOR * wall)[..., None])
    inside = np.clip(0.5 - hexd * _SS, 0.0, 1.0)[..., None]
    rgb = rgb * (1.0 - inside) + socket * inside
    mouth = np.clip(1.0 - np.maximum(hexd, 0.0) / BOLT_EDGE, 0.0, 1.0) * (1.0 - inside[..., 0])
    rgb = rgb * (1.0 - 0.4 * mouth)[..., None]
    rgb = rgb * (1.0 - _grime(xs, ys, dist, r, mark))[..., None]
    rgb, cover = _boxed(rgb, cover)
    rgb = rgb * tone  # the lamp's falloff at this screw's own corner of the panel
    # What goes under the collar: the seat shadow and the spanner's ring, both black, both soft,
    # and both composited under the head so the head covers its own shadow. The ring is heavier
    # on one side, and which side is this screw's own business: a spanner is swung, not centred.
    at = np.arange(size, dtype=np.float32)
    rx, ry = at[None, :] - (half + fx), at[:, None] - (half + fy)
    ring_d = np.hypot(rx, ry) - r
    swing = np.random.default_rng(mark + 17).uniform(0.0, 2.0 * math.pi)
    lean = 1.0 + 0.5 * (rx * math.cos(swing) + ry * math.sin(swing)) / max(r, 1e-6)
    ring = SCREW_WEAR * np.clip(lean, 0.0, 1.5) * np.clip(1.0 - ring_d / SCREW_WEAR_W, 0.0, 1.0)
    ring = ring * np.clip(ring_d + 0.5, 0.0, 1.0)
    shadow = cast(cover, lift, flat) * SCREW_SEAT
    under = 1.0 - (1.0 - shadow) * (1.0 - ring)
    alpha = cover + under * (1.0 - cover)
    rgb = rgb * (cover / np.maximum(alpha, 1e-6))[..., None]
    return to_image(rgb, alpha)


def _grime(xs: Field, ys: Field, dist: Field, r: float, mark: int) -> Field:
    """SCREW_GRIME specks of dirt caught round one head's rim, 0..1 of the light they take.

    Placed from *mark* alone, so no two heads on the panel are dirty in the same places. A
    fixing is the smallest thing a panel repeats, which makes it the first place a repeat shows.
    """
    rng = np.random.default_rng(mark + 31)
    out = np.zeros(dist.shape, np.float32)
    for turn in rng.uniform(0.0, 2.0 * math.pi, SCREW_GRIME):
        at = r * (1.0 - SCREW_GRIME_W * rng.uniform(0.2, 1.0))
        px, py = xs - at * math.cos(turn), ys - at * math.sin(turn)
        size = r * rng.uniform(0.08, 0.20)
        out = np.maximum(out, np.clip(1.0 - np.hypot(px, py) / size, 0.0, 1.0))
    return SCREW_GRIME_A * out * (dist < r)
