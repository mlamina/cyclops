"""Cyclops writing on his own screen: the offer, the cue, and the log line.

The half of this feature that Python can see. The other half is a document rendered by a browser
inside an iframe, which lives in ``tests/render_check.mjs`` and needs Chromium and a running admin
service - so what is here is the part that can run anywhere in a second, which is the part that
actually runs.

Two things worth failing over, and neither is the happy path:

* **The cue.** ``announces()`` is a module global rather than a read of the offer file, so it is
  the one piece of state in the panel handshake that can get out of step with the glass. A drawing
  arms it; nothing else may leave it armed behind them.
* **The string in panel.js** that holds the scratchpad's stylesheet, which is exactly the sort of
  thing that gets tidied out of a file nobody tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cyclops import agent, panel, session
from cyclops.config import Settings

PANEL_JS = Path(__file__).resolve().parents[1] / "src/cyclops/admin/static/panel.js"


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
    """A drawing arrives silently half a minute later; a scratchpad arrives while he is talking."""
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
    assert line.strip(), "a record with no words in it is still a record"


def test_entities_come_back_as_the_characters_they_stood_for() -> None:
    record = {"type": "screen", "t": 0, "html": "<h1>8&nbsp;&amp;&nbsp;10</h1>"}
    assert "8 & 10" in session._render_record(record)


# ---------------------------------------------------------------- what holds the line in the page


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



def _jpeg() -> bytes:
    """A real JPEG, small - the cue test needs a picture to offer, not a good one."""
    import io

    from PIL import Image

    out = io.BytesIO()
    Image.new("RGB", (8, 8), (90, 120, 90)).save(out, format="JPEG")
    return out.getvalue()
