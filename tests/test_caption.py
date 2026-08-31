"""The bottom line of the panel: what it says, and how it moves while it says it.

Three silent failures live here, and none of them raises. A tool that reaches the panel through
no phrase at all is simply never mentioned - which is the bug this whole line was built to fix,
so a ninth tool added without one has to fail something. A job that finishes in five milliseconds
under a job that has not finished takes the caption down with it if the hand-back is wrong, and
the panel goes blank mid-search. And dots that are drawn without their width being reserved make
the slab behind them breathe in and out four times a second, which no unit test would notice and
nobody could look at.

Everything here is pure: no camera, no key, no window. `Overlay` is PIL and numpy only.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np
import pytest

from cyclops import agent, overlay, ui
from cyclops.config import Settings


@dataclass
class Call:
    """As much of a Realtime function call as :func:`agent._activity_line` looks at."""

    name: str
    arguments: str | None


# ---------------------------------------------------------------- the animation


def test_dots_walk_up_and_start_again() -> None:
    step = overlay.DOT_PERIOD_S / (overlay.CAPTION_DOTS + 1)
    counts = [
        overlay.caption_pulse(i * step + step / 2)[1] for i in range(overlay.CAPTION_DOTS + 1)
    ]
    assert counts == [0, 1, 2, 3], "the dots are supposed to fill in one after another"
    assert overlay.caption_pulse(overlay.DOT_PERIOD_S + step / 2)[1] == 0, "and then start over"


def test_the_dots_never_stall_however_long_the_panel_has_been_up() -> None:
    # A Pi's monotonic clock is its uptime, and this panel is left running for weeks. What has to
    # hold at a million seconds is not that the dots are at any particular place - that is
    # meaningless - but that they still advance exactly one step per step, with no stall and no
    # skip. Which phase the sequence happens to start on is the clock's business.
    step = overlay.DOT_PERIOD_S / (overlay.CAPTION_DOTS + 1)
    for base in (0.0, 86_400.0, 1_000_000.0, 5_000_000.0):
        seen = [overlay.caption_pulse(base + i * step + step / 2)[1] for i in range(13)]
        gaps = {(b - a) % (overlay.CAPTION_DOTS + 1) for a, b in zip(seen, seen[1:], strict=False)}
        assert gaps == {1}, f"the dots stumbled at {base:,.0f}s of uptime: {seen}"


def test_the_breath_comes_back_round() -> None:
    for t in (0.15, 0.37, 1.9, 86_400.15, 1_000_000.4):
        assert overlay.caption_pulse(t + overlay.BREATH_PERIOD_S)[0] == pytest.approx(
            overlay.caption_pulse(t)[0], abs=1e-6
        )


def test_the_breath_stays_inside_its_depth() -> None:
    assert overlay.caption_pulse(0.0)[0] == 0.0, "brightest at the top of a breath"
    sunk = [overlay.caption_pulse(i * overlay.BREATH_PERIOD_S / 64)[0] for i in range(64)]
    assert min(sunk) >= 0.0
    assert max(sunk) <= overlay.BREATH_DEPTH
    # A raised cosine, not a square wave: a panel that snapped between two brightnesses would
    # read as a fault light rather than as work being done.
    assert max(sunk) == pytest.approx(overlay.BREATH_DEPTH, rel=1e-3)


# ---------------------------------------------------------------- the vocabulary


def test_every_tool_the_model_is_offered_has_something_to_say() -> None:
    settings = Settings(api_key="")  # diagrams and projects both default on
    offered = {
        tool["name"]
        for tool in [
            agent.WEB_SEARCH_TOOL,
            *agent._diagram_tools(settings),
            *agent._project_tools(settings),
        ]
    }
    assert len(offered) == 8, "the tool list changed; the caption table probably needs to as well"
    for name in sorted(offered):
        line = agent._activity_line(Call(name, "{}"))
        assert line != "working…", f"{name} falls through to the line meant for invented tools"
        assert line.endswith(overlay.BUSY_MARK), f"{name} would not animate: {line!r}"


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        (Call("web_search", '{"query": "M8 torque"}'), "searching for M8 torque…"),
        (Call("draw_diagram", '{"request": "the fuse box"}'), "drawing the fuse box…"),
        (
            Call("find_diagram", '{"query": "the fuse box"}'),
            "looking for a drawing of the fuse box…",
        ),
        (Call("open_project", '{"name": "Kitchen Tap"}'), "opening Kitchen Tap…"),
        (Call("track_project", '{"name": "Kitchen Tap"}'), "starting to track Kitchen Tap…"),
        (Call("find_data", '{"query": "flow rate"}'), "looking up flow rate…"),
        (Call("forget_data", '{"key": "flow rate"}'), "forgetting flow rate…"),
        (Call("save_data", '{"entries": [{"key": "a"}, {"key": "b"}]}'), "writing down 2 values…"),
        (Call("save_data", '{"entries": [{"key": "a"}]}'), "writing down 1 value…"),
        (Call("nonsense", "{}"), "working…"),
    ],
)
def test_a_line_names_the_subject_not_the_tool(call: Call, expected: str) -> None:
    assert agent._activity_line(call) == expected


def test_a_line_survives_arguments_the_model_got_wrong() -> None:
    # Never raise on the way to a caption: a tool called with rubbish still has to be answered,
    # so it still has to be describable.
    for arguments in (None, "", "not json", "[]", '{"query": null}'):
        line = agent._activity_line(Call("web_search", arguments))
        assert line == "searching the web…", arguments


def test_a_long_subject_is_cut_to_a_phrase_and_not_left_mid_word() -> None:
    query = "the correct torque specification for an M8 stainless bolt into aluminium"
    line = agent._activity_line(Call("web_search", f'{{"query": "{query}"}}'))
    assert len(line) < len(query)
    assert query.startswith(line[len("searching for ") : -1])
    assert not line[:-1].endswith(" "), "cut on a word boundary, with nothing left dangling"


# ---------------------------------------------------------------- the channel


@pytest.fixture
def voice() -> agent.VoiceAgent:
    return agent.VoiceAgent(Settings(api_key=""))


def test_a_job_that_finished_instantly_was_still_readable(voice: agent.VoiceAgent) -> None:
    job = voice._start_doing("writing down 6 values…")
    voice._done_doing(job)
    assert voice.activity == "writing down 6 values…", "a five-millisecond tool must still show"


def test_and_it_does_eventually_go(
    voice: agent.VoiceAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(agent, "ACTIVITY_HOLD_S", 0.0)
    voice._done_doing(voice._start_doing("writing down 6 values…"))
    assert voice.activity == ""


def test_a_quick_job_hands_the_line_back_to_the_slow_one(
    voice: agent.VoiceAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    # One response can call two tools, and _on_response_done spawns a task per call. A find_data
    # that lands while a fourteen-second search is still running must not blank the search.
    monkeypatch.setattr(agent, "ACTIVITY_HOLD_S", 0.0)
    search = voice._start_doing("searching for M8 torque…")
    quick = voice._start_doing("looking up flow rate…")
    voice._done_doing(quick)
    assert voice.activity == "searching for M8 torque…"
    voice._done_doing(search)
    assert voice.activity == ""


def test_a_tool_with_nothing_to_say_leaves_the_line_alone(voice: agent.VoiceAgent) -> None:
    voice._start_doing("searching for M8 torque…")
    assert voice._start_doing("") is None
    assert voice.activity == "searching for M8 torque…"


def test_the_chain_does_not_grow_with_the_session(
    voice: agent.VoiceAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(agent, "ACTIVITY_HOLD_S", 0.0)
    for _ in range(50):
        voice._done_doing(voice._start_doing("looking up flow rate…"))
    assert len(voice._doing) <= 1, "finished jobs are pruned on the way in, not left to pile up"


# ---------------------------------------------------------------- who gets to speak


def test_the_controller_prefers_the_agent_then_the_phase(voice: agent.VoiceAgent) -> None:
    control = ui.SessionController(Settings(api_key=""), frames=None, entrypoint="test")
    assert control._detail(ui.LISTENING) == "", "nothing to add: the panel's own line is better"

    control._phase = "saving the video…"
    assert control._detail(ui.LISTENING) == "saving the video…"

    control._agent = voice
    voice._start_doing("searching for M8 torque…")
    assert control._detail(ui.LISTENING) == "searching for M8 torque…", "the live job wins"

    control._error = "OpenAI rejected the API key"
    assert control._detail(ui.ERROR) == "OpenAI rejected the API key", "and a fault wins outright"


# ---------------------------------------------------------------- on the panel


def _slab(frame: np.ndarray, ov: overlay.Overlay) -> tuple[int, int]:
    """The caption slab's left and right edge, read back off a rendered frame.

    The picture band's backdrop alpha is zeroed (see Overlay.__init__), so inside these rows the
    only opaque thing is the slab, its text, and the two corner ticks well outside it.
    """
    height = round(24 * ov.scale)
    y = ov.footer.y - round(12 * ov.scale) - height / 2
    band = frame[int(y - height / 2) + 1 : int(y + height / 2) - 1, :, 3].max(axis=0)
    left = ov.viewport.x + ov.pad + round(34 * ov.scale)
    lit = [c for c in np.where(band >= overlay.PLATE_ALPHA)[0] if c >= left]
    break_at = np.where(np.diff(lit) > 1)[0]
    return left, int(lit[break_at[0]] if len(break_at) else lit[-1])


def test_the_slab_does_not_breathe_with_the_dots() -> None:
    # The one thing here that a unit test would never catch and nobody could stand to look at:
    # the slab is sized to its text, so unreserved dots make a dark rectangle grow and shrink
    # four times a second behind the words.
    ov = overlay.Overlay(800, 480)
    step = overlay.DOT_PERIOD_S / (overlay.CAPTION_DOTS + 1)
    edges = {
        _slab(
            ov.render(
                state=overlay.SEARCHING,
                level=0.0,
                detail="searching for M8 torque…",
                phase=i * step + step / 2,
            ),
            ov,
        )
        for i in range(overlay.CAPTION_DOTS + 1)
    }
    assert len(edges) == 1, f"the slab moved with the dots: {sorted(edges)}"


def test_a_resting_caption_reserves_no_room_for_dots() -> None:
    ov = overlay.Overlay(800, 480)
    at_rest = dict(state=overlay.LISTENING, level=0.0, phase=0.0)
    busy = _slab(ov.render(detail="doing a thing…", **at_rest), ov)
    rest = _slab(ov.render(detail="doing a thing", **at_rest), ov)
    assert busy[1] - rest[1] == pytest.approx(ov._dots_w, abs=2), (
        "a line about work in flight is exactly the dots wider than the same line at rest"
    )


def test_the_dots_actually_land_on_the_panel() -> None:
    ov = overlay.Overlay(800, 480)
    step = overlay.DOT_PERIOD_S / (overlay.CAPTION_DOTS + 1)
    shown = dict(state=overlay.SEARCHING, level=0.0, detail="searching…")
    bare = ov.render(phase=step / 2, **shown)
    full = ov.render(phase=3 * step + step / 2, **shown)
    assert not np.array_equal(bare, full), "nothing moved between no dots and three"


def test_the_wrapper_always_lets_the_line_go(
    voice: agent.VoiceAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    # _run_tool wraps the whole dispatch, so a tool that raises must not leave the panel stuck
    # saying what it was doing for the rest of the session - nor swallow the exception, which
    # is what stops the model waiting forever for a result nobody sent.
    monkeypatch.setattr(agent, "ACTIVITY_HOLD_S", 0.0)
    seen: list[str] = []

    async def boom(call: Call) -> None:
        seen.append(voice.activity)
        raise RuntimeError("the card fell out")

    monkeypatch.setattr(voice, "_dispatch_tool", boom)
    with pytest.raises(RuntimeError, match="the card fell out"):
        asyncio.run(voice._run_tool(Call("open_project", '{"name": "Kitchen Tap"}')))
    assert seen == ["opening Kitchen Tap…"], "the line has to be up while the tool runs"
    assert voice.activity == "", "and gone once it is not"
