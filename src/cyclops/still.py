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
cost more than the recording is worth. Both halves of the handshake in :mod:`cyclops.diagram`
already carry the picture, and this is the reader of them:

* **a photo** rides in ``DIAGRAM_FILE`` as a data URL, and is the same JPEG the page decodes;
* **a drawing** is laid out by the page and by nothing else, so the page is the only thing that
  can say what it looks like. It already hands the SVG back to be kept beside the session
  (``/diagram/shown``), and the admin service drops a copy at ``PANEL_SVG_FILE`` for this.

Which of the two it is comes from the payload rather than from a caller, for the same reason the
page decides it that way: there is one file, and what is in it is the answer.

The result is fitted the way the page fits it - ``object-fit: contain`` on the screen's own
green-black - so the frame in the video is the picture that was on the glass, letterbox and all.

The SVG is rasterised by ffmpeg, which recording already cannot run without and which is built
against librsvg on the Pi. That is one process per picture shown, next to the ten seconds the
drawing itself took, and none at all for a photo.

Never raises, and returns None for every kind of failure - a missing file, a payload that is not
one, an ffmpeg that cannot read SVG. The caller falls back to black, which is where it started.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from .config import DIAGRAM_FILE, PANEL_SVG_FILE

# The green-black the page draws on: --screen in admin/static/base.css, in BGR. The letterbox
# either side of a contained picture is the screen's own colour there rather than a second one,
# and a recording that used black instead would invent an edge the panel does not have.
SCREEN_BGR = (10, 15, 5)

# One picture, once, on a box that has just spent ten seconds drawing it. Long enough that a
# loaded Pi still finishes, short enough that a wedged ffmpeg costs a frame rather than a session.
RASTER_TIMEOUT_S = 5.0

_JPEG_URL = "data:image/jpeg;base64,"


def of_panel(
    width: int,
    height: int,
    *,
    payload_path: Path = DIAGRAM_FILE,
    svg_path: Path = PANEL_SVG_FILE,
) -> np.ndarray | None:
    """The picture the page is showing, as a ``width`` x ``height`` BGR frame, or None.

    None means there is nothing to draw and the caller should go on publishing black: nobody has
    offered the panel anything (the admin page uncovers with the offer withdrawn, so tapping the
    eye lands here), or what was offered cannot be turned back into pixels.
    """
    offered = _offered(payload_path)
    if offered is None:
        return None
    url = offered.get("image")
    picture = _decode(url) if isinstance(url, str) else _raster(svg_path)
    if picture is None:
        return None
    return contain(picture, width, height)


def contain(picture: np.ndarray, width: int, height: int) -> np.ndarray:
    """Fit a picture inside the panel without cropping it, centred on the screen's own colour.

    The page uses ``object-fit: contain`` and says why (``diagram.css``): cropping a picture
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


def _offered(path: Path) -> dict | None:
    """What is waiting for the panel, or None. The same file the page reads - see diagram.offer."""
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


def _raster(path: Path) -> np.ndarray | None:
    """The drawing the page handed back, rendered. None if there is none, or ffmpeg cannot.

    ``rgb24`` rather than letting the PNG carry an alpha channel: the page's own copy already has
    the screen behind it (``serialize`` in diagram.js puts it there so the file is worth opening),
    and flattening here means the one that reaches the encoder cannot arrive transparent either.
    """
    if not path.is_file() or path.stat().st_size == 0:
        return None
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(path),
        "-frames:v", "1", "-pix_fmt", "rgb24", "-c:v", "png", "-f", "image2pipe", "-",
    ]
    try:
        done = subprocess.run(  # noqa: S603 - the command is ours, and the path is our own cache
            command, capture_output=True, timeout=RASTER_TIMEOUT_S, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"· [still] could not draw the panel's picture ({exc})", file=sys.stderr, flush=True)
        return None
    if done.returncode != 0 or not done.stdout:
        # An ffmpeg without librsvg is the likely one, and it is not fatal: the recording loses
        # the drawing and keeps everything else, which is what it did before this existed.
        detail = done.stderr.decode(errors="replace").strip().splitlines()
        print(
            f"· [still] no picture for the recording ({detail[-1] if detail else 'ffmpeg failed'})",
            file=sys.stderr,
            flush=True,
        )
        return None
    picture = cv2.imdecode(np.frombuffer(done.stdout, dtype=np.uint8), cv2.IMREAD_COLOR)
    return picture if picture is not None and picture.size else None
