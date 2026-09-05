"""Cyclops writing on his own screen: the offer, the video's account of it, and the log line.

The half of this feature that Python can see. The other half is a document rendered by a browser
inside an iframe, which lives in ``tests/render_check.mjs`` and needs Chromium and a running admin
service - so what is here is the part that can run anywhere in a second, which is the part that
actually runs.

Three things worth failing over, and none of them is the happy path:

* **The cue.** ``announces()`` is a module global rather than a read of the offer file, so it is
  the one piece of state in the panel handshake that can get out of step with the glass. A drawing
  arms it; nothing else may leave it armed behind them.
* **What the recording says.** ``still.of_panel`` cannot rasterise a scratchpad and answers None,
and
  ``Kiosk._restill`` used to read None as "keep the frame you have" - which would put the previous
  photograph in ``video.mp4`` for as long as the scratchpad was up. Black is the honest answer and
  this pins it.
* **The two strings in panel.js** that stop a runaway scratchpad, which are exactly the sort of
thing
  that gets tidied out of a file nobody tests.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import numpy as np
import pytest

from cyclops import agent, kiosk, panel, session, still
from cyclops.config import Settings
from cyclops.kiosk import Kiosk

PANEL_JS = Path(__file__).resolve().parents[1] / "src/cyclops/admin/static/panel.js"
PANEL_CSS = Path(__file__).resolve().parents[1] / "src/cyclops/admin/static/panel.css"


@pytest.fixture
def panel_file(tmp_path, monkeypatch):
    path = tmp_path / "cache" / "panel.json"
    monkeypatch.setattr(panel, "PANEL_FILE", path)
    return path


# ---------------------------------------------------------------- the offer


def test_a_scratchpad_is_offered_as_markup_and_not_as_a_picture(panel_file) -> None:
    assert panel.offer_scratchpad("<h1>25 Nm</h1>") is True
    payload = json.loads(panel_file.read_text())
    assert payload["scratchpad"] == "<h1>25 Nm</h1>"
    assert "image" not in payload, "the page branches on which key is there"
    assert payload["id"]


def test_an_offer_carries_only_what_the_page_branches_on(panel_file) -> None:
    """One kind of picture, one way out - see ``show()`` in panel.js.

    There was a ``drawn`` key here until 2026-09-05 that chose between a corner square and a
    press anywhere, so a diagram went away differently from a photograph. Everything the panel
    shows is a picture now and a press anywhere puts any of it away, so the payload carries the
    thing itself and its id and nothing to branch on but which door it came through.
    """
    panel.offer_scratchpad("<h1>25 Nm</h1>")
    assert set(json.loads(panel_file.read_text())) == {"id", "scratchpad"}

    panel.offer_image(_jpeg(), "the fuse box")
    assert set(json.loads(panel_file.read_text())) == {"id", "title", "image"}


def test_a_scratchpad_lands_quietly_and_leaves_the_cue_where_it_found_it(panel_file) -> None:
    """A drawing arrives silently after ninety seconds; a scratchpad arrives while he is talking."""
    panel.offer_scratchpad("<h1>25 Nm</h1>")
    assert panel.announces() is False, "he is still speaking over it; a cue would be one too many"

    panel.offer_image(_jpeg(), "the fuse box", announce=True)
    assert panel.announces() is True

    panel.offer_scratchpad("<h1>25 Nm</h1>")
    assert panel.announces() is False, "a scratchpad over a drawing must disarm the cue again"


def test_two_scratchpads_never_share_an_id(panel_file) -> None:
    """The id is the whole repaint trigger, so showing the same thing twice must not go quiet."""
    panel.offer_scratchpad("<h1>25 Nm</h1>")
    first = json.loads(panel_file.read_text())["id"]
    panel.offer_scratchpad("<h1>25 Nm</h1>")
    assert json.loads(panel_file.read_text())["id"] != first


def test_a_card_that_will_not_write_is_false_and_not_a_traceback(tmp_path, monkeypatch) -> None:
    blocked = tmp_path / "wall"
    blocked.write_text("not a directory")
    monkeypatch.setattr(panel, "PANEL_FILE", blocked / "panel.json")
    assert panel.offer_scratchpad("<h1>25 Nm</h1>") is False


# ---------------------------------------------------------------- what the recording is told


def test_a_scratchpad_cannot_be_rasterised_and_says_so(tmp_path) -> None:
    """Deliberate: rendering it would mean a headless browser per scratchpad on a Pi that
    throttles."""
    payload = tmp_path / "panel.json"
    payload.write_text(json.dumps({"id": "abc", "scratchpad": "<h1>25 Nm</h1>"}))
    assert still.of_panel(800, 480, payload_path=payload) is None


def test_a_scratchpad_over_a_photo_blacks_the_recording(monkeypatch) -> None:
    """The bug this exists to prevent, and the reason ``_restill`` no longer trusts None.

    ``of_panel`` answers None for anything it cannot turn into pixels, and ``_restill`` used to
    read that as "keep the frame you have". Swap a scratchpad over a photograph and the video
    would hold the photograph for as long as the scratchpad was up - a picture that is not on
    the screen,
    presented as the screen, which is worse than the black the reveal already publishes.
    """
    published: list[np.ndarray] = []

    class Panel:
        _panel_showing = threading.Event()
        _page_still = np.full((480, 800, 3), 200, dtype=np.uint8)  # the photograph, still held
        panel = type("Sink", (), {"publish": staticmethod(published.append)})()

        def _panel_size(self):
            return 800, 480

        _restill = Kiosk._restill

    monkeypatch.setattr(kiosk.still, "of_panel", lambda *a, **k: None)
    stand_in = Panel()
    stand_in._panel_showing.set()
    stand_in._restill()

    assert len(published) == 1
    assert not published[0].any(), "black, not the photograph that was up a moment ago"
    assert stand_in._page_still is published[0], "and the recorder samples it from here too"


# ---------------------------------------------------------------- the tool


def test_the_tool_is_left_out_rather_than_refused() -> None:
    """Every gate here works this way: a tool the model can see is a tool it will try."""
    on = Settings(api_key="k", scratchpad=True)
    assert agent._scratchpad_tools(on) == [agent.SCRATCHPAD_TOOL]
    assert agent._scratchpad_tools(Settings(api_key="k", scratchpad=False)) == []


def test_the_activity_line_ends_in_an_ellipsis() -> None:
    """How ``overlay.BUSY_MARK`` tells work in flight from a state, and so blinks a cursor."""
    call = type("Call", (), {"name": "write_on_scratchpad", "arguments": "{}"})()
    assert agent._activity_line(call).endswith("…")


def test_the_scratchpad_is_capped_at_a_screenful() -> None:
    """The argument is the latency: nothing appears until the model has finished writing it."""
    huge = json.dumps({"html": "<li>x</li>" * 1000})
    kept = agent._tool_string(huge, "html", agent.MAX_SCRATCHPAD_CHARS)
    assert len(kept) == agent.MAX_SCRATCHPAD_CHARS


# ---------------------------------------------------------------- the line in the log


def test_the_log_says_what_was_on_the_screen_in_its_own_words() -> None:
    """Read out of the markup rather than asked for alongside it - see ``_scratchpad_gist``."""
    line = session._render_record(
        {"type": "screen", "t": 0, "html": "<h1>25 Nm</h1><p>on the crank bolt</p>"}
    )
    assert "25 Nm on the crank bolt" in line
    assert "<h1>" not in line, "a transcript is for reading"


def test_a_long_scratchpad_is_trimmed_to_a_line() -> None:
    line = session._render_record({"type": "screen", "t": 0, "html": "<li>step</li>" * 40})
    assert line.endswith("…")
    assert len(line) < 140


def test_a_scratchpad_with_no_words_in_it_still_gets_a_line() -> None:
    """An SVG drawing is the ordinary case of this, and a blank line in the page is not a record."""
    line = session._render_record(
        {"type": "screen", "t": 0, "html": '<svg viewBox="0 0 8 8"><circle r="4"/></svg>'}
    )
    assert line.strip() and "characters of markup" in line


def test_entities_come_back_as_the_characters_they_stood_for() -> None:
    record = {"type": "screen", "t": 0, "html": "<h1>8&nbsp;&amp;&nbsp;10</h1>"}
    assert "8 & 10" in session._render_record(record)


# ---------------------------------------------------------------- what holds the line in the page


def test_the_page_still_refuses_to_run_what_the_model_wrote() -> None:
    """Two strings, and between them the reason a scratchpad cannot pin a core or stall a paint.

    ``sandbox`` gives the scratchpad's document an origin of its own and no script at all. The
    CSP is
    the one that is easy to mistake for belt-and-braces and is not: the iframe's ``load`` event
    waits on subresources, so one ``<img src="https://...">`` the model invented would stall the
    paint handshake for as long as DNS and TCP take, on a panel whose whole promise is that this
    appears while he is still talking.

    In the spirit of ``test_panel_css.py``: a textual pin on the part of the interface that has no
    other test and would fail silently, on a screen nobody is watching start.
    """
    source = PANEL_JS.read_text(encoding="utf-8")
    assert "'sandbox', ''" in source
    assert "default-src 'none'" in source


def test_the_scratchpads_stylesheet_is_still_a_string() -> None:
    """SCRATCHPAD_HEAD is a JS template literal, so one backtick in it ends the page.

    Written after doing exactly that: a CSS comment saying `color` is set... closed the literal,
    panel.js stopped parsing, and the panel served a page with no paint loop at all. Nothing on
    the Python side noticed, and the browser check that did notice only said "timed out waiting
    for body.drawing", which is a long way from the cause.

    A millisecond here, in the spirit of ``test_panel_css.py``: the thing that fails silently on a
    screen nobody is watching start is the thing worth a cheap test.
    """
    source = PANEL_JS.read_text(encoding="utf-8")
    opened = source.index("const SCRATCHPAD_HEAD = `")
    body = source[opened + len("const SCRATCHPAD_HEAD = `") :]
    assert "`" in body, "the literal must be closed at all"
    assert "`" not in body[: body.index("`")], "no backtick may appear inside it"
    assert body[: body.index("`")].count("${") == 0, "and nothing may interpolate into it"


def test_a_scratchpad_does_not_eat_the_press_that_dismisses_it() -> None:
    """The press belongs to the iframe unless this says otherwise, and then there is no way out.

    The listener that puts a scratchpad away is on #stage, one level above the frame. Without this
    declaration it never hears a thing, and Cyclops has covered the eye and the way out of the
    session with something that cannot be dismissed.
    """
    rules = PANEL_CSS.read_text(encoding="utf-8")
    block = re.search(r"\.stage \.scratchpad \{[^}]*\}", rules, re.S)
    assert block, ".stage .scratchpad is what the page names the frame"
    assert "pointer-events: none" in block.group(0)


def _jpeg() -> bytes:
    """A real JPEG, small - the cue test needs a picture to offer, not a good one."""
    import io

    from PIL import Image

    out = io.BytesIO()
    Image.new("RGB", (8, 8), (90, 120, 90)).save(out, format="JPEG")
    return out.getvalue()
