"""What a recording sees while the page has the panel.

The hole this fills is a silent one: a diagram or a photo held the screen for minutes, the kiosk
painted nothing while it did, and the video of that session was black over exactly the part worth
watching twice. So what is guarded here is that an offered picture comes back as a frame of the
panel's own size, that it is fitted rather than cropped or stretched, and that every way of
having nothing to show is still an honest None - which the kiosk answers with black, as before.

No panel and no browser: the offer is a file, and the frame is arithmetic on what is in it.
"""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from cyclops import still

PANEL = (800, 480)  # the official 7" panel, and what a screen recording is

SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="480">'
    '<rect width="100%" height="100%" fill="#050f0a"/>'
    '<rect x="40" y="40" width="200" height="80" fill="#ff0000"/></svg>'
)


def offer(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "diagram.json"
    path.write_text(json.dumps({"id": "abc123", **payload}))
    return path


def jpeg_url(width: int, height: int, colour: tuple[int, int, int]) -> str:
    picture = np.full((height, width, 3), colour, dtype=np.uint8)
    ok, blob = cv2.imencode(".jpg", picture)
    assert ok
    return "data:image/jpeg;base64," + base64.b64encode(blob.tobytes()).decode("ascii")


def has_librsvg() -> bool:
    """Whether the ffmpeg on this machine can read an SVG at all - the Pi's can; a Mac's may not."""
    if shutil.which("ffmpeg") is None:
        return False
    done = subprocess.run(
        ["ffmpeg", "-hide_banner", "-decoders"], capture_output=True, check=False
    )
    return b"librsvg" in done.stdout


# ---- a photo ----


def test_a_photo_comes_back_at_the_panels_size(tmp_path: Path) -> None:
    path = offer(tmp_path, {"title": "bench", "image": jpeg_url(*PANEL, (20, 30, 200))})
    frame = still.of_panel(*PANEL, payload_path=path, svg_path=tmp_path / "none.svg")
    assert frame is not None
    assert frame.shape == (PANEL[1], PANEL[0], 3)  # what the encoder was opened at
    assert frame.dtype == np.uint8


def test_a_photo_is_letterboxed_and_not_stretched(tmp_path: Path) -> None:
    """The page fits it with object-fit: contain, so this does too - see diagram.css."""
    path = offer(tmp_path, {"image": jpeg_url(480, 480, (20, 30, 200))})  # square, on a 5:3 panel
    frame = still.of_panel(*PANEL, payload_path=path, svg_path=tmp_path / "none.svg")
    assert frame is not None
    # 480x480 fits the panel's height, so it lands 480 wide with 160 either side of it.
    assert tuple(int(v) for v in frame[240, 10]) == still.SCREEN_BGR  # the letterbox, not black
    assert tuple(int(v) for v in frame[240, 790]) == still.SCREEN_BGR
    middle = frame[240, 400]
    assert abs(int(middle[0]) - 20) < 12 and abs(int(middle[2]) - 200) < 12  # the picture, jpeg'd


def test_a_photo_larger_than_the_panel_keeps_its_shape(tmp_path: Path) -> None:
    path = offer(tmp_path, {"image": jpeg_url(1024, 768, (200, 200, 200))})
    frame = still.of_panel(*PANEL, payload_path=path, svg_path=tmp_path / "none.svg")
    assert frame is not None
    # 4:3 into 5:3: 640x480, so the bars are 80 wide and the top row is picture, not letterbox.
    assert tuple(int(v) for v in frame[240, 5]) == still.SCREEN_BGR
    assert tuple(int(v) for v in frame[0, 400]) != still.SCREEN_BGR


# ---- a drawing ----


@pytest.mark.skipif(not has_librsvg(), reason="this ffmpeg cannot read an SVG (the Pi's can)")
def test_a_drawing_is_rasterised_from_what_the_page_handed_back(tmp_path: Path) -> None:
    svg = tmp_path / "panel.svg"
    svg.write_text(SVG)
    path = offer(tmp_path, {"title": "relay", "spec": {"nodes": []}})
    frame = still.of_panel(*PANEL, payload_path=path, svg_path=svg)
    assert frame is not None
    assert frame.shape == (PANEL[1], PANEL[0], 3)
    assert tuple(int(v) for v in frame[80, 140]) == (0, 0, 255)  # the red rect, in BGR
    assert tuple(int(v) for v in frame[400, 400]) == still.SCREEN_BGR  # the page's own ground


def test_a_drawing_that_never_arrived_is_black_and_not_the_last_one(tmp_path: Path) -> None:
    """The kiosk clears the file before it asks; this is what that clearing has to mean."""
    path = offer(tmp_path, {"title": "relay", "spec": {"nodes": []}})
    assert still.of_panel(*PANEL, payload_path=path, svg_path=tmp_path / "gone.svg") is None
    empty = tmp_path / "panel.svg"
    empty.write_bytes(b"")
    assert still.of_panel(*PANEL, payload_path=path, svg_path=empty) is None


# ---- nothing to show ----


def test_no_offer_at_all_is_none(tmp_path: Path) -> None:
    """The admin page withdraws the offer before it uncovers, so tapping the eye lands here."""
    assert still.of_panel(*PANEL, payload_path=tmp_path / "gone.json") is None


def test_a_payload_that_is_not_one_is_none(tmp_path: Path) -> None:
    bad = tmp_path / "diagram.json"
    bad.write_text("half a fi")
    assert still.of_panel(*PANEL, payload_path=bad) is None
    bad.write_text('["not an object"]')
    assert still.of_panel(*PANEL, payload_path=bad) is None


def test_an_image_that_will_not_decode_is_none(tmp_path: Path) -> None:
    path = offer(tmp_path, {"image": "data:image/jpeg;base64,bm90IGEgamJlZw=="})
    assert still.of_panel(*PANEL, payload_path=path, svg_path=tmp_path / "none.svg") is None
    path = offer(tmp_path, {"image": "https://example.invalid/photo.jpg"})
    assert still.of_panel(*PANEL, payload_path=path, svg_path=tmp_path / "none.svg") is None
