"""Editing a photo: the size arithmetic, the response shape, and what lands on the card.

Everything here is pure - no camera, no key, no network - which is only possible because
`imagine.py` takes a `Path` and hands back bytes, and because the one piece of API-shape
knowledge in it lives in `decode()` rather than inside `edit()`.

The three things worth failing over: a size the model would reject or that crops somebody's
photo; a response shape that changed under us; and the `_edit` suffix, which is the contract
`session._render_photo` and `projects/deps.py` both key off.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
from datetime import datetime
from pathlib import Path

import pytest
from PIL import Image

from cyclops import agent, diagram, imagine
from cyclops.config import Settings


def jpeg(width: int, height: int) -> bytes:
    """A real JPEG of a given size - PIL is the only thing here that has to be honest."""
    out = io.BytesIO()
    Image.new("RGB", (width, height), (90, 120, 90)).save(out, format="JPEG")
    return out.getvalue()


class FakeResponse:
    def __init__(self, data) -> None:
        self.data = data


class FakeImage:
    def __init__(self, b64) -> None:
        self.b64_json = b64


# ---------------------------------------------------------------- the size


SOURCES = [(1024, 576), (640, 480), (1920, 1080), (1024, 768), (100, 4000), (2000, 90), (48, 48)]


@pytest.mark.parametrize("source", SOURCES)
def test_every_size_is_one_the_model_will_take(source) -> None:
    """Multiples of 16, an aspect within 1:3 to 3:1, and - the undocumented one that 400s on a
    real C920 capture - never under the minimum pixel budget."""
    width, height = (int(n) for n in imagine.size_for(*source).split("x"))
    assert width % imagine.STEP == 0 and height % imagine.STEP == 0
    assert max(width, height) <= imagine.MAX_EDGE
    assert min(width, height) >= imagine.MIN_EDGE
    assert max(width / height, height / width) <= imagine.MAX_ASPECT + 0.1
    assert width * height >= imagine.PIXEL_BUDGET, "rounding is upward for exactly this reason"


@pytest.mark.parametrize("source", SOURCES)
def test_the_output_keeps_the_photo_it_came_from(source) -> None:
    """The framing survives. A crop of somebody's own photo is a different picture, and the
    aspect is the only part of the framing a size can carry."""
    want = min(max(source[0] / source[1], 1 / imagine.MAX_ASPECT), imagine.MAX_ASPECT)
    width, height = (int(n) for n in imagine.size_for(*source).split("x"))
    assert width / height == pytest.approx(want, rel=0.06), "within one 16-pixel rounding step"


def test_a_c920_capture_clears_the_floor_that_the_obvious_size_does_not() -> None:
    """1024x576 is exactly what webcam.py encodes a 16:9 capture as, and the API refuses it."""
    width, height = (int(n) for n in imagine.size_for(1024, 576).split("x"))
    assert (width, height) != (1024, 576)
    assert width * height > 1024 * 576


def test_a_moved_floor_is_answered_by_asking_for_more_pixels() -> None:
    """What edit()'s one retry does. Bigger, and still the same shape."""
    first = imagine.size_for(1024, 576)
    again = imagine.size_for(1024, 576, budget=int(imagine.PIXEL_BUDGET * imagine.BUDGET_GROWTH))
    pixels = [w * h for w, h in ((int(n) for n in s.split("x")) for s in (first, again))]
    assert pixels[1] > pixels[0]


# ---------------------------------------------------------------- the response


def test_decode_pulls_the_picture_out_of_a_response() -> None:
    picture = b"\xff\xd8\xff\xe0 not really a jpeg"
    response = FakeResponse([FakeImage(base64.b64encode(picture).decode("ascii"))])
    assert imagine.decode(response) == picture


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse(None),
        FakeResponse([]),
        FakeResponse([FakeImage("")]),
        FakeResponse([FakeImage("not base64!")]),
        object(),
    ],
)
def test_decode_refuses_anything_that_is_not_a_picture(response) -> None:
    with pytest.raises(imagine.ImagineError):
        imagine.decode(response)


# ---------------------------------------------------------------- the network guards


def test_an_empty_request_never_reaches_the_network(tmp_path) -> None:
    source = tmp_path / "14-32-40_you.jpg"
    source.write_bytes(jpeg(64, 64))
    with pytest.raises(imagine.ImagineError):
        asyncio.run(imagine.edit(source, "   ", Settings(api_key="")))


def test_a_missing_photo_is_a_sayable_error_and_not_an_OSError(tmp_path) -> None:
    with pytest.raises(imagine.ImagineError) as raised:
        asyncio.run(imagine.edit(tmp_path / "gone.jpg", "paint it black", Settings(api_key="")))
    assert "card" in str(raised.value), "the message is read out loud, so it has to be a sentence"


def test_a_png_off_the_card_reaches_the_model_as_a_jpeg(tmp_path, monkeypatch) -> None:
    """Whatever is on the panel is what gets edited, and that is no longer always a camera JPEG.

    A picture dropped into a project folder is very often a PNG, and `edit` posts what it reads
    as `image/jpeg`. Sending PNG bytes under that name is a lie the API is entitled to reject,
    so the source is normalised on the way past.
    """
    out = io.BytesIO()
    Image.new("RGBA", (320, 180), (90, 120, 90, 255)).save(out, format="PNG")
    source = tmp_path / "dropped.png"
    source.write_bytes(out.getvalue())

    handed: list[bytes] = []

    class FakeImages:
        async def edit(self, **kwargs):
            handed.append(kwargs["image"][1])
            return FakeResponse([FakeImage(base64.b64encode(jpeg(64, 48)).decode())])

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            self.images = FakeImages()

        async def close(self) -> None:
            return None

    monkeypatch.setattr(imagine, "AsyncOpenAI", FakeClient)
    asyncio.run(imagine.edit(source, "paint it black", Settings(api_key="")))

    assert handed, "the request never went"
    assert handed[0][:3] == b"\xff\xd8\xff", "a JPEG, whatever the card was holding"


# ---------------------------------------------------------------- what lands


def test_an_edit_lands_private_and_named_for_the_clock_and_what_made_it(tmp_path) -> None:
    picture = jpeg(320, 180)
    kept = imagine.write(
        picture, "paint the doors matt black", tmp_path, when=datetime(2026, 9, 2, 14, 33, 5)
    )
    assert kept.path == tmp_path / "14-33-05_edit.jpg"
    assert kept.path.read_bytes() == picture
    assert kept.path.stat().st_mode & 0o777 == 0o600, "a picture of somebody's room"
    assert kept.ident == "14-33-05_edit"
    assert kept.bytes == len(picture)
    # The suffix is a contract: session._render_photo and projects/deps.py both read it back.
    assert kept.path.name.endswith("_edit.jpg")


def test_the_panel_copy_is_smaller_and_a_small_one_is_left_alone() -> None:
    big = jpeg(2000, 1400)
    small = imagine.for_panel(big)
    with Image.open(io.BytesIO(small)) as shrunk:
        assert max(shrunk.size) <= imagine.PANEL_MAX_EDGE
    assert len(small) < len(big)

    already = jpeg(400, 300)
    assert imagine.for_panel(already) is already


def test_a_picture_that_will_not_shrink_is_still_shown() -> None:
    """Never raise on the way to the panel: a picture nobody can resize is still a picture."""
    assert imagine.for_panel(b"not an image at all") == b"not an image at all"


# ---------------------------------------------------------------- the handoff to the panel


@pytest.fixture
def panel_file(tmp_path, monkeypatch):
    path = tmp_path / "cache" / "diagram.json"
    monkeypatch.setattr(diagram, "DIAGRAM_FILE", path)
    return path


def test_offer_image_writes_a_payload_the_page_can_read(panel_file) -> None:
    assert diagram.offer_image(jpeg(64, 48), "matt black doors") is True
    payload = json.loads(panel_file.read_text())
    assert payload["image"].startswith("data:image/jpeg;base64,")
    assert payload["title"] == "matt black doors"
    assert payload[diagram.SPEC_NAME] is None, "the page branches on this to pick a renderer"
    assert payload["svg"] is None, "there is no picture to send back; it came as one"
    assert payload["id"]


def test_two_offers_never_share_an_id(panel_file) -> None:
    """A stable id would make showing the same picture twice silently do nothing the second time."""
    diagram.offer_image(jpeg(64, 48), "one")
    first = json.loads(panel_file.read_text())["id"]
    diagram.offer_image(jpeg(64, 48), "one")
    assert json.loads(panel_file.read_text())["id"] != first


def test_withdrawing_leaves_the_page_nothing_to_paint(panel_file) -> None:
    """The panel's browser is warm and polling, so an offer nobody showed is not inert.

    One left behind by a process that exited without tidying up is what the page paints, and
    keeps painting, until something takes it back - which is how tapping the eye came to uncover
    a picture from an hour earlier instead of the dashboard it promises.
    """
    diagram.offer_image(jpeg(64, 48), "matt black doors")
    assert panel_file.exists()
    diagram.withdraw()
    assert not panel_file.exists()
    diagram.withdraw()  # idempotent: nothing waiting is the state it is trying to reach


def test_withdrawing_what_cannot_be_removed_is_not_a_traceback(tmp_path, monkeypatch) -> None:
    """A payload that will not go is never a reason to refuse the tap that wanted it gone."""
    blocked = tmp_path / "wall"
    blocked.write_text("not a directory")
    monkeypatch.setattr(diagram, "DIAGRAM_FILE", blocked / "diagram.json")
    diagram.withdraw()


def test_a_card_that_will_not_write_is_false_and_not_a_traceback(tmp_path, monkeypatch) -> None:
    blocked = tmp_path / "wall"
    blocked.write_text("not a directory")
    monkeypatch.setattr(diagram, "DIAGRAM_FILE", blocked / "diagram.json")
    assert diagram.offer_image(jpeg(64, 48), "matt black") is False


# ---------------------------------------------------------------- showing it to the model


@pytest.fixture
def voice(monkeypatch):
    """A VoiceAgent with the socket faked out, recording what it would have sent."""
    made = agent.VoiceAgent(Settings(api_key=""))
    made._conn = object()  # `connected` asks only whether this is set
    sent: list[dict] = []

    async def _send_item(item):
        sent.append(item)

    monkeypatch.setattr(made, "_send_item", _send_item)
    monkeypatch.setattr(made, "_log", lambda *a, **k: None)
    return made, sent


def test_the_model_is_shown_the_picture_it_had_made(voice) -> None:
    made, sent = voice
    asyncio.run(made.add_edit(jpeg(320, 180), "paint the doors matt black"))

    assert len(sent) == 1
    content = sent[0]["content"]
    assert sent[0]["role"] == "user", "an image cannot ride in a function_call_output"
    image = next(part for part in content if part["type"] == "input_image")
    assert image["image_url"].startswith("data:image/jpeg;base64,")
    label = next(part for part in content if part["type"] == "input_text")["text"]
    assert "paint the doors matt black" in label
    assert "illustration" in label, "the nearest text to the image says what it is"
    assert label.startswith("[") and label.endswith("]"), "flat, so it is not read out verbatim"


def test_showing_the_model_an_edit_is_not_what_records_it(voice) -> None:
    """A second change now carries on from the first - but add_edit is not where that is said.

    This method is handed bytes and has no path, and it runs whether or not the picture reached
    the card. `_run_edit_photo` records the edit on the loop, from the path it was written to.
    Pinning it here so the bookkeeping cannot quietly migrate back into the method that only
    knows how to show something.
    """
    made, _ = voice
    before = agent.Panel(Path("/cyclops/photos/12-00-00_you.jpg"), "photo")
    made._on_panel = before
    asyncio.run(made.add_edit(jpeg(320, 180), "paint the doors matt black"))
    assert made._on_panel is before


# ---------------------------------------------------------------- what gets edited


class Called:
    """As much of a Realtime function call as `_run_edit_photo` reads."""

    def __init__(self, arguments: str) -> None:
        self.name = "edit_photo"
        self.call_id = "call_1"
        self.arguments = arguments


def edit_result(made) -> dict:
    """Drive `_run_edit_photo` and hand back the tool output it sent."""
    sent: list[dict] = []

    async def _send_tool_output(call_id, output):
        sent.append(output)

    made._send_tool_output = _send_tool_output
    made._request_response = _nothing
    asyncio.run(made._run_edit_photo(Called('{"request": "paint the doors matt black"}')))
    return sent[0]


async def _nothing() -> None:
    return None


def test_nothing_shown_yet_still_asks_for_the_shutter(voice) -> None:
    made, _ = voice
    assert made._on_panel is None, "a session starts with nothing in front of them"
    output = edit_result(made)
    assert output["ok"] is False
    assert "SNAP" in output["note"], "the one thing they can do about it"


def test_a_drawing_on_the_panel_is_not_edited_as_if_it_were_a_photo(voice) -> None:
    """The failure this slot exists to prevent.

    A diagram takes the panel through the same show() a photo does. If it left the previous
    photograph in the slot, "change that" would redraw a picture nobody is looking at and put the
    result up as though it had answered - and it would be an image model painting over a drawing
    whose whole value is that it was checked against a schema.
    """
    made, _ = voice
    made._on_panel = agent.Panel(Path("/cyclops/photos/12-00-00_you.jpg"), "photo")
    made._note_panel({"ok": True, "title": "the fuse box", "shown": True})
    assert made._on_panel == agent.Panel(None, "diagram")

    output = edit_result(made)
    assert output["ok"] is False
    assert "draw_diagram" in output["note"], "it names the tool that can change it"


def test_a_drawing_nobody_could_see_leaves_the_photo_where_it_was(voice) -> None:
    """No panel free means no drawing on it, so the photograph is still the thing in front of
    them - and still the thing to edit."""
    made, _ = voice
    photo = agent.Panel(Path("/cyclops/photos/12-00-00_you.jpg"), "photo")
    made._on_panel = photo
    made._note_panel({"ok": True, "title": "the fuse box", "shown": False})
    assert made._on_panel is photo


def test_nothing_is_sent_once_the_socket_has_gone(voice) -> None:
    made, sent = voice
    made._conn = None
    asyncio.run(made.add_edit(jpeg(320, 180), "paint the doors matt black"))
    assert sent == []
