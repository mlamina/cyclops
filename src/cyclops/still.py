"""What the page has on the panel, as one frame - so a recording is not blind while it is up.

A session recording the screen samples whatever the kiosk last painted (:class:`cyclops.record.
PanelSource`). While a diagram or a photo is up, the kiosk paints nothing at all: the panel
belongs to the browser behind its window, and the render loop is asleep. It publishes black for
those minutes on purpose - a frozen halo over a running timer watches back as a hung encoder -
but black is what you get exactly where the interesting thing happened, and "here is the wiring"
is the one part of a session worth watching twice.

So the frame is rebuilt from what the page was handed, on the way past. Nothing here reads the
screen: the panel cannot be photographed from this side (the kiosk's window is not what is on
it), and polling a compositor fifteen times a second for a picture that does not change would
cost more than the recording is worth. The handshake in :mod:`cyclops.panel`
already carries the picture, and this is the reader of it: whatever is on the glass rides in
``PANEL_FILE`` as a data URL, and it is the same JPEG the page decodes.

There is a second case, and it is the same shape wearing the opposite trade. A scratchpad is
markup the model wrote, so the offer file holds no pixels for it and there is nothing here to
decode. It used to be black for that reason and this module used to argue for it: turning one back
into pixels sounded like a headless browser per scratchpad on a box that throttles at 85 C. But the
browser that has to be asked is already running and has already drawn it - so it draws it once more
onto a canvas and posts the JPEG back (:func:`cyclops.panel.keep_still`), and this reads it out of
``PANEL_STILL_FILE``. No new process, no ffmpeg, one canvas draw per scratchpad and none per frame.

Which is what the diagram branch that used to live here did too, before diagrams became images: the
page was the only thing that knew what it looked like, so the page said. An ``<iframe sandbox="">``
has an origin of its own and cannot be read even by the page around it, so that is true of a
scratchpad in the strongest possible way.

The result is fitted the way the page fits it - ``object-fit: contain`` on the screen's own
green-black - so the frame in the video is the picture that was on the glass, letterbox and all.


Never raises, and returns None for every kind of failure - a missing file, a payload that is not
one, an ffmpeg that cannot read SVG. The caller falls back to black, which is where it started.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import numpy as np

from .config import PANEL_FILE, PANEL_STILL_FILE

# The green-black the page draws on: --screen in admin/static/base.css, in BGR. The letterbox
# either side of a contained picture is the screen's own colour there rather than a second one,
# and a recording that used black instead would invent an edge the panel does not have.
SCREEN_BGR = (10, 15, 5)

# One picture, once, on a box that has just spent ten seconds drawing it. Long enough that a
# loaded Pi still finishes, short enough that a wedged ffmpeg costs a frame rather than a session.
_JPEG_URL = "data:image/jpeg;base64,"


def of_panel(
    width: int,
    height: int,
    *,
    payload_path: Path = PANEL_FILE,
    still_path: Path = PANEL_STILL_FILE,
) -> np.ndarray | None:
    """The picture the page is showing, as a ``width`` x ``height`` BGR frame, or None.

    None means there is nothing to draw and the caller should go on publishing black: nobody has
    offered the panel anything (the admin page uncovers with the offer withdrawn, so tapping the
    eye lands here), what was offered cannot be turned back into pixels, or it is a scratchpad
    the page has not posted its picture of yet.

    A scratchpad (``cyclops.panel.offer_scratchpad``) has no pixels in the offer, so it comes from
    the still the page left beside it instead - and only if that still names this same offer. A
    still for some earlier one is ignored: black is where this started, and the previous scratchpad
    is a worse answer than black for the reason ``kiosk._restill`` gives.
    """
    offered = _read(payload_path)
    if offered is None:
        return None
    url = offered.get("image")
    if not isinstance(url, str):
        url = _kept(still_path, offered)
    picture = _decode(url) if isinstance(url, str) else None
    if picture is None:
        return None
    return contain(picture, width, height)


def _kept(path: Path, offered: dict) -> str | None:
    """The page's own picture of the scratchpad in ``offered``, or None.

    None for everything except a scratchpad whose still is on the card and names it: an offer that
    is not a scratchpad has no business here, and a still that names another one is a picture of
    something already put away.
    """
    if not isinstance(offered.get("scratchpad"), str):
        return None
    kept = _read(path)
    if kept is None or kept.get("id") != offered.get("id"):
        return None
    url = kept.get("image")
    return url if isinstance(url, str) else None


def contain(picture: np.ndarray, width: int, height: int) -> np.ndarray:
    """Fit a picture inside the panel without cropping it, centred on the screen's own colour.

    The page uses ``object-fit: contain`` and says why (``panel.css``): cropping a picture
    somebody asked to see is a small lie. A recording that cropped what the panel did not would
    be a larger one, so the two agree.
    """
    have_h, have_w = picture.shape[:2]
    if not have_w or not have_h:
        return np.full((height, width, 3), SCREEN_BGR, dtype=np.uint8)
    scale = min(width / have_w, height / have_h)
    fit_w, fit_h = max(1, round(have_w * scale)), max(1, round(have_h * scale))
    # INTER_AREA down, INTER_CUBIC up - the same split, and the same reasoning, as
    # overlay.fit_to_window. Sharpening is not repeated here: a drawing rasterised straight to
    # this size has nothing to rescue, and a photograph reaching the panel has already been
    # through imagine.for_panel.
    shrinking = scale < 1.0
    fitted = cv2.resize(
        picture, (fit_w, fit_h), interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
    )
    canvas = np.full((height, width, 3), SCREEN_BGR, dtype=np.uint8)
    x0, y0 = (width - fit_w) // 2, (height - fit_h) // 2
    canvas[y0 : y0 + fit_h, x0 : x0 + fit_w] = fitted
    return canvas


def _read(path: Path) -> dict | None:
    """One of this module's two JSON files as a dict, or None for every way that can fail.

    The offer (:func:`cyclops.panel.offer_image`) and the still beside it
    (:func:`cyclops.panel.keep_still`) are read the same way and fail the same way.
    """
    try:
        found = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return found if isinstance(found, dict) else None


def _decode(url: str) -> np.ndarray | None:
    """The JPEG out of a data URL, as BGR. None if it is not one, or will not decode."""
    if not url.startswith(_JPEG_URL):
        return None
    try:
        blob = base64.b64decode(url[len(_JPEG_URL) :], validate=True)
    except (ValueError, TypeError):
        return None
    picture = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
    return picture if picture is not None and picture.size else None
