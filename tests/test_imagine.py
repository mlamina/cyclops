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

from cyclops import agent, imagine, panel, tasks
from cyclops.config import Settings
from cyclops.webcam import Capture


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


def test_a_diagram_with_nothing_described_never_reaches_the_network() -> None:
    """The same guard as an edit's, and it matters more: a drawing costs half a minute."""
    with pytest.raises(imagine.ImagineError):
        asyncio.run(imagine.draw("   ", "a wiring diagram", Settings(api_key="")))


def test_a_drawing_asked_for_at_the_panel_s_own_shape() -> None:
    """The size is fixed rather than computed, so this is the one place it is written down twice.

    Both edges must divide by 16 and the area must clear the image model's minimum pixel
    budget - the two rules that make `size_for` non-obvious for edits. A drawing has no source
    photo to take a shape from, so a typo here would simply be a 400 from the API half a minute
    later.
    """
    width, height = (int(n) for n in imagine.PANEL_SIZE.split("x"))
    assert width % 16 == 0 and height % 16 == 0
    assert width * height >= imagine.PIXEL_BUDGET
    assert round(width / height, 2) == round(800 / 480, 2), "the panel's own 5:3, or it letterboxes"


def test_a_drawing_with_no_style_note_still_asks_for_something() -> None:
    """The tool makes `style` required, but a missing adjective must not cost the whole picture."""
    prompt = imagine.DRAW_PROMPT.format(request="a relay", style="" or imagine.DEFAULT_STYLE)
    assert imagine.DEFAULT_STYLE in prompt


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
    path = tmp_path / "cache" / "panel.json"
    monkeypatch.setattr(panel, "PANEL_FILE", path)
    return path


def test_offer_image_writes_a_payload_the_page_can_read(panel_file) -> None:
    assert panel.offer_image(jpeg(64, 48), "matt black doors") is True
    payload = json.loads(panel_file.read_text())
    assert payload["image"].startswith("data:image/jpeg;base64,")
    assert payload["title"] == "matt black doors"
    assert "drawn" not in payload, "one kind of picture; a press anywhere puts any of it away"
    assert payload["id"]


def test_the_panel_is_told_which_arrivals_are_worth_a_sound(panel_file) -> None:
    """What the cue on arrival keys off, and the only thing that still varies between pictures.

    This is the test that stands where ``is_picture()`` used to. That asked whether the picture
    was a photograph rather than a drawing, which stopped being a question the moment drawings
    became photographs - it would have answered yes to everything and the panel would have gone
    silent for good. The caller now says what it wants.

    The sound is all it says. ``announce`` used to be ``drawn`` and rode into the payload, where
    it also decided how the picture was dismissed; that went on 2026-09-05 and the flag stayed
    behind here, on this side of the file, where it only ever meant "make a noise".
    """
    panel.offer_image(jpeg(64, 48), "18-08-39_you")
    assert panel.announces() is False, "the shutter already said so a beat ago"

    panel.offer_image(jpeg(64, 48), "the fuse box", announce=True)
    assert panel.announces() is True, "a drawing had nothing else to announce it"
    assert "drawn" not in json.loads(panel_file.read_text()), "and the page is told nothing"

    panel.offer_image(jpeg(64, 48), "paint the doors matt black")
    assert panel.announces() is False, "...and a photo back over that is quiet again"


def test_show_without_a_panel_is_false_not_an_error(monkeypatch) -> None:
    """``uv run cyclops`` has a conversation and no screen. That is ordinary, not a failure."""
    monkeypatch.setattr(panel, "_kiosk", None)
    assert panel.show() is False


def test_two_offers_never_share_an_id(panel_file) -> None:
    """A stable id would make showing the same picture twice silently do nothing the second time."""
    panel.offer_image(jpeg(64, 48), "one")
    first = json.loads(panel_file.read_text())["id"]
    panel.offer_image(jpeg(64, 48), "one")
    assert json.loads(panel_file.read_text())["id"] != first


def test_withdrawing_leaves_the_page_nothing_to_paint(panel_file) -> None:
    """The panel's browser is warm and polling, so an offer nobody showed is not inert.

    One left behind by a process that exited without tidying up is what the page paints, and
    keeps painting, until something takes it back - which is how tapping the eye came to uncover
    a picture from an hour earlier instead of the dashboard it promises.
    """
    panel.offer_image(jpeg(64, 48), "matt black doors")
    assert panel_file.exists()
    panel.withdraw()
    assert not panel_file.exists()
    panel.withdraw()  # idempotent: nothing waiting is the state it is trying to reach


def test_withdrawing_what_cannot_be_removed_is_not_a_traceback(tmp_path, monkeypatch) -> None:
    """A payload that will not go is never a reason to refuse the tap that wanted it gone."""
    blocked = tmp_path / "wall"
    blocked.write_text("not a directory")
    monkeypatch.setattr(panel, "PANEL_FILE", blocked / "panel.json")
    panel.withdraw()


def test_a_card_that_will_not_write_is_false_and_not_a_traceback(tmp_path, monkeypatch) -> None:
    blocked = tmp_path / "wall"
    blocked.write_text("not a directory")
    monkeypatch.setattr(panel, "PANEL_FILE", blocked / "panel.json")
    assert panel.offer_image(jpeg(64, 48), "matt black") is False


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


def edit_result(made, arguments: str = '{"request": "paint the doors matt black"}') -> dict:
    """Drive `_run_edit_photo` and hand back the tool output it sent.

    `_spawn` is stubbed out rather than left to run: the handler answers the call and hands the
    half-minute of drawing to a background task, and this suite has no key and no network to do
    that with. Closing the coroutine keeps the loop `asyncio.run` tears down from complaining
    about a task nobody awaited.
    """
    sent: list[dict] = []

    async def _send_tool_output(call_id, output):
        sent.append(output)

    made._send_tool_output = _send_tool_output
    made._request_response = _nothing
    made._spawn = lambda coro: coro.close()
    asyncio.run(made._run_edit_photo(Called(arguments)))
    return sent[0]


def edited(made, arguments: str) -> tuple[dict, list[Path]]:
    """The tool output, and whichever picture it actually handed to the half minute of drawing.

    `_edit` is replaced rather than left to run, for the reason `edit_result`'s docstring gives.
    The path is captured as the coroutine is built, because `_spawn` only closes what it is
    handed and the body never runs.
    """
    picked: list[Path] = []
    made._edit = lambda task, source, request, turn, made_from="": (
        picked.append(source) or _nothing()
    )
    return edit_result(made, arguments), picked


def shot(name: str, kind: str = "photo") -> agent.Panel:
    """A picture in play, named the way the shutter and `imagine.write` name their files."""
    return agent.Panel(Path(f"/cyclops/photos/{name}.jpg"), kind, name)


async def _nothing() -> None:
    return None


def test_an_earlier_picture_is_edited_by_its_name(voice) -> None:
    """The whole point: three photos in, "make the ball red" is the ball and not the desk."""
    made, _ = voice
    made._pictures = [shot("12-00-00_you"), shot("12-01-00_you"), shot("12-02-00_you")]
    made._on_panel = made._pictures[-1]
    output, picked = edited(made, '{"request": "make it red", "picture": "12-00-00_you"}')

    assert output["ok"] is True
    assert picked == [made._pictures[0].path]


def test_naming_no_picture_still_means_the_one_in_play(voice) -> None:
    """The ordinary call, which is nearly all of them: "change that" is what they can see."""
    made, _ = voice
    made._pictures = [shot("12-00-00_you"), shot("12-02-00_you")]
    made._on_panel = made._pictures[-1]
    _, picked = edited(made, '{"request": "make it red"}')

    assert picked == [made._pictures[-1].path]


def test_a_name_nobody_was_given_draws_nothing(voice) -> None:
    """The half minute this saves. A name the model chose to send is a picture it meant, so it
    must not fall through to whatever happens to be newest - that is a wrong photo redrawn, on
    the panel, half a minute later, with nothing to say it went wrong."""
    made, _ = voice
    made._pictures = [shot("12-00-00_you")]
    made._on_panel = made._pictures[-1]
    output, picked = edited(made, '{"request": "make it red", "picture": "the ball one"}')

    assert output["ok"] is False
    assert picked == [], "nothing was spent"


def test_a_refusal_hands_back_the_names_it_would_have_taken(voice) -> None:
    """How it recovers inside one turn: the model can only ask again if it is told the names."""
    made, _ = voice
    made._pictures = [shot("12-00-00_you"), shot("12-01-00_drawn", "drawn")]
    made._on_panel = made._pictures[-1]
    output = edit_result(made, '{"request": "make it red", "picture": "nonsense"}')

    assert [one["name"] for one in output["pictures"]] == ["12-01-00_drawn", "12-00-00_you"]


def test_a_photo_arrives_with_the_name_it_can_be_edited_by(voice, tmp_path) -> None:
    """The caption is the only place the model ever learns a name, so the name in the text and
    the name on the record have to be the same one - and they are written either side of a send
    that could fail between them."""
    made, sent = voice
    ball = Capture("data:image/jpeg;base64,x", tmp_path / "ball.jpg", 4, 3, 9, 0)
    asyncio.run(made.add_photo(ball))
    label = next(part for part in sent[0]["content"] if part["type"] == "input_text")["text"]

    assert made._on_panel is not None
    assert made._on_panel.name in label
    assert made._picture_named(made._on_panel.name) is made._on_panel


def test_two_pictures_with_one_stem_do_not_answer_to_one_name(voice) -> None:
    """A recall can bring back another session's picture named exactly like today's."""
    made, _ = voice
    made._put_on_panel(Path("/cyclops/photos/12-00-00_you.jpg"), "photo")
    made._put_on_panel(Path("/june/photos/12-00-00_you.jpg"), "found")

    assert len({one.name for one in made._pictures}) == 2
    assert made._picture_named(made._pictures[0].name) is made._pictures[0]


def test_nothing_shown_yet_still_asks_for_the_shutter(voice) -> None:
    made, _ = voice
    assert made._on_panel is None, "a session starts with nothing in front of them"
    output = edit_result(made)
    assert output["ok"] is False
    assert "SNAP" in output["note"], "the one thing they can do about it"


def test_a_drawing_on_the_panel_is_edited_like_any_other_picture(voice) -> None:
    """The opposite of what this slot used to assert, and the change is the point.

    A drawing was once a JSON spec with no file behind it, so the slot held ``Panel(None,
    "diagram")`` and an edit had nothing to send: it refused, and named draw_diagram. A drawing
    is a jpg in photos/ now, so "make that clearer, drop the status LED" is an ordinary edit of
    an ordinary picture.
    """
    made, _ = voice
    made._on_panel = agent.Panel(Path("/cyclops/photos/12-00-00_drawn.jpg"), "drawn")
    output = edit_result(made)
    assert output["ok"] is True, "it has a file, so it is sent rather than refused"
    assert output["started"] is True, "the call is answered before the picture exists"


def test_a_picture_that_was_never_written_down_is_not_edited(voice) -> None:
    """What is left of the guard: the slot can still hold something with no file behind it -
    a picture shown with no session running - and there is nothing to send for that."""
    made, _ = voice
    made._on_panel = agent.Panel(None, "drawn")
    output = edit_result(made)
    assert output["ok"] is False
    assert "SNAP" in output["note"], "the one thing they can do about it"


def test_nothing_is_sent_once_the_socket_has_gone(voice) -> None:
    made, sent = voice
    made._conn = None
    asyncio.run(made.add_edit(jpeg(320, 180), "paint the doors matt black"))
    assert sent == []


# ---------------------------------------------------------------- work that outlives its call


class Asked:
    """A draw_diagram call, as much of one as the handler reads."""

    def __init__(self, arguments: str) -> None:
        self.name = "draw_diagram"
        self.call_id = "call_2"
        self.arguments = arguments


def tool_result(made, call) -> dict:
    """Drive one image tool and hand back the output it sent, without doing the work.

    `_spawn` is stubbed for the reason `edit_result`'s docstring gives: the half of these that
    talks to the image model belongs to `cyclops-smoke` and the Pi, not to a suite with no key.
    """
    sent: list[dict] = []

    async def _send_tool_output(call_id, output):
        sent.append(output)

    made._send_tool_output = _send_tool_output
    made._request_response = _nothing
    made._spawn = lambda coro: coro.close()
    asyncio.run(made._run_draw_diagram(call))
    return sent[0]


def test_a_drawing_is_answered_before_it_has_been_drawn(voice) -> None:
    """The change this whole mechanism is for.

    The tool used to await the image model here, so the call stayed open for the whole draw and
    the model could not say another word - while its own description told it to say what it was
    doing and carry on. Now it answers at once and is told separately when the picture lands.
    """
    made, _ = voice
    output = tool_result(made, Asked('{"request": "wire a relay to GPIO 17", "style": "manual"}'))

    assert output["ok"] is True and output["started"] is True
    assert "told" in output["note"], "the model is told that it will be told"


def test_a_drawing_says_on_the_panel_what_it_is_drawing(voice) -> None:
    """The task and the caption say the same sentence, because they come from the same place -
    `_activity_line`, which the panel has always used for this call."""
    made, _ = voice
    tool_result(made, Asked('{"request": "wire a relay to GPIO 17", "style": "manual"}'))

    (row,) = tasks.read()
    assert row.state == tasks.RUNNING
    assert row.what == tasks.line() == "drawing wire a relay to GPIO 17…"
    assert made.drawing_active, "the panel is in the drawing state before the call is answered"


def test_nothing_described_is_refused_without_opening_a_task(voice) -> None:
    """A guard clause is answered in the moment and there is nothing to watch."""
    made, _ = voice
    output = tool_result(made, Asked('{"request": "", "style": "manual"}'))

    assert output["ok"] is False
    assert tasks.read() == [], "no work started, so no row"
    assert not made.drawing_active


def test_a_finished_job_reaches_the_conversation_on_its_own(voice) -> None:
    """There is no tool call left to answer by the time a picture lands, so the arrival is a
    synthetic user turn - the shape `add_photo` uses, and for the same reason."""
    made, sent = voice
    asked: list[bool] = []
    made._request_response = lambda: asked.append(True) or _nothing()

    asyncio.run(made.announce("[The diagram you were drawing is now up on their screen.]"))

    assert len(sent) == 1 and sent[0]["role"] == "user"
    text = sent[0]["content"][0]["text"]
    assert text.startswith("[") and text.endswith("]"), "flat, so it is not read out verbatim"
    assert asked, "unlike add_edit, this one asks for a reply - nothing else is going to"


def test_a_session_that_ended_mid_drawing_is_not_talked_to(voice) -> None:
    made, sent = voice
    made._conn = None
    asyncio.run(made.announce("[The diagram you were drawing is now up on their screen.]"))
    assert sent == []
