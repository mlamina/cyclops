"""The panel's metal: one lamp, one steel, and the build-time fields a part is shaded with.

What this module owns is everything that makes two pieces of chrome look cut from the same bar
under the same light. The lamp is :data:`LAMP` - the terminal's bench light (``GLARE_X`` and
``GLARE_Y`` in :mod:`cyclops.overlay`, up and a little left of the panel) turned into a direction
any surface can dot its normal with - and the steel is the four stops under it. Everything else
is a pure function from a shape to a float field: the normals of a rolled edge, the diffuse and
specular light on them, brushed grain, hairline scratches, the shadow a raised part drops, the
glare a sheet of glass gives back, and a hex-socket cap screw assembled out of all of those.

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
AMBIENT = 0.30  # what a face turned right away from the lamp still gets from the room. Not
# zero: an unlit edge reads as a hole cut in the panel rather than as the dark side of a bar.
SHINE = 32.0  # how tight the highlight is - a machined finish, not a mirror and not matte
SPEC = 0.9  # ...and how much of STEEL_SPEC it lays down where it lands hardest
RIM = 0.8  # how brightly an edge seen end-on lights up where it faces the lamp. The terminal's
# moulding already does this (its sheen); it is what puts the one crisp bright line on the near
# edge of every bar, and without it a rolled edge is a gradient and reads as rubber.
SHADOW_DROP = 1.4  # how far a shadow falls per pixel a part stands proud...
SHADOW_SOFT = 1.3  # ...and how soft its edge is, in the same currency

# ---- the steel ----
#
# Desaturated, with the faintest green in it: gunmetal seen by a phosphor tube. The phosphor is
# the only light on this panel and the metal has none of its own - it is grey, and it borrows.
STEEL = (78, 84, 80)  # a flat face square to the viewer, once the lamp has had its say
STEEL_LIT = (152, 162, 156)  # a rolled edge turned into the lamp
STEEL_DARK = (16, 20, 18)  # the floor of a socket, the inside of a shadow
STEEL_SPEC = (206, 216, 210)  # the brightest a highlight may get. Short of the tube's white on
# purpose: a piece of steel outshining the phosphor would be a second light on the panel.
GRAIN = 0.13  # how far the brushing moves brightness either way
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


def shade(nx: Field, ny: Field, nz: Field) -> tuple[Field, Field]:
    """Diffuse and specular light, 0..1 each, on a field of normals under the one lamp.

    The specular is two things added: the lamp's own highlight, which lands where a surface
    faces half-way between the lamp and the viewer, and the rim light an edge catches where it
    turns end-on towards the lamp - squared twice, as the terminal's sheen is, so it stays on
    the edge and off the face.
    """
    diffuse = np.clip(nx * _L[0] + ny * _L[1] + nz * _L[2], 0.0, 1.0)
    spec = np.clip(nx * _H[0] + ny * _H[1] + nz * _H[2], 0.0, 1.0) ** SHINE
    tilt = np.sqrt(nx * nx + ny * ny)
    facing = np.clip((nx * _L2[0] + ny * _L2[1]) / np.maximum(tilt, 1e-6), 0.0, 1.0)
    rim = tilt * tilt * facing * facing
    return diffuse, np.minimum(spec + RIM * rim, 1.0)


def steel(diffuse: Field, spec: Field, grain: Field | None = None, colour: Colour = STEEL) -> Field:
    """The colour of lit steel, rows by columns by three.

    *colour* where a face is square to the viewer, darker as it turns from the lamp, STEEL_SPEC
    laid on where the highlight lands, and brushed by *grain* if there is any. Never brighter
    than STEEL_SPEC in any channel, however the terms add up.
    """
    albedo = np.asarray(colour, np.float32) / (AMBIENT + (1.0 - AMBIENT) * _L[2])
    lit = AMBIENT + (1.0 - AMBIENT) * diffuse
    if grain is not None:
        lit = lit * (1.0 + GRAIN * grain)
    ceiling = np.asarray(STEEL_SPEC, np.float32)
    rgb = albedo * lit[..., None] + ceiling * (SPEC * spec)[..., None]
    return np.minimum(rgb, ceiling)


def _noise1d(coord: Field, period: float, rng: np.random.Generator) -> Field:
    """Piecewise-linear noise in -1..1 along one coordinate, a new bend every *period* pixels."""
    table = rng.uniform(-1.0, 1.0, _TABLE).astype(np.float32)
    at = coord / period
    i0 = np.floor(at)
    frac = (at - i0).astype(np.float32)
    i0 = i0.astype(np.int64)
    return table[i0 % _TABLE] * (1.0 - frac) + table[(i0 + 1) % _TABLE] * frac


def grain(across: Field, along: Field, seed: int = SEED) -> Field:
    """Brushed metal, -1..1: fibres that run with *along* and vary with *across*, both in pixels.

    Three pitches of fibre, because one reads as stripes, and a slow term along the stroke so no
    streak runs the whole length of anything - a brush is dragged by a hand, not a ruler.
    """
    rng = np.random.default_rng(seed)
    fibres = (
        0.5 * _noise1d(across, 1.0, rng)
        + 0.3 * _noise1d(across, 2.7, rng)
        + 0.2 * _noise1d(across, 7.3, rng)
    )
    stroke = _noise1d(along + 0.35 * across, 29.0, rng)
    return fibres * (0.6 + 0.4 * stroke)


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


def cast(coverage: Field, lift: float) -> Field:
    """The shadow a part standing *lift* pixels proud drops on what it is bolted to, 0..1.

    Its own coverage, pushed away from the lamp and softened by the same distance. Composited
    under the part, so the part covers its own shadow the way a real one does.
    """
    dx, dy = lamp_2d()
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
