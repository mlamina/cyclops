"""The bottom line of the panel: what it says, and how it moves while it says it.

Three silent failures live here, and none of them raises. A tool that reaches the panel through
no phrase at all is simply never mentioned - which is the bug this whole line was built to fix,
so a tenth tool added without one has to fail something. A job that finishes in five milliseconds
under a job that has not finished takes the caption down with it if the hand-back is wrong, and
the panel goes blank mid-search. And a cursor drawn without its width being reserved sits under
the left-hand bracket half the time on any line that fills the screen, which no unit test would
notice and nobody could look at.

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


def test_the_cursor_is_on_for_half_of_every_blink() -> None:
    half = overlay.CURSOR_PERIOD_S / 2
    assert overlay.caption_pulse(half / 2)[1], "a cursor starts a period showing"
    assert not overlay.caption_pulse(half * 1.5)[1], "...and spends the other half of it dark"
    # Square, unlike the breath beside it: a cursor is a thing being switched, not one being
    # dimmed, and a terminal has never faded one in.
    on = [overlay.caption_pulse(i * overlay.CURSOR_PERIOD_S / 64)[1] for i in range(64)]
    assert sum(on) == 32, f"the blink is not half on and half off: {sum(on)}/64"


def test_the_cursor_never_stalls_however_long_the_panel_has_been_up() -> None:
    # A Pi's monotonic clock is its uptime, and this panel is left running for weeks. What has to
    # hold at a million seconds is not that the cursor is in any particular state - that is
    # meaningless - but that it is still turning over once per period, with no stall and no skip.
    half = overlay.CURSOR_PERIOD_S / 2
    for base in (0.0, 86_400.0, 1_000_000.0, 5_000_000.0):
        seen = [overlay.caption_pulse(base + i * half + half / 2)[1] for i in range(13)]
        assert all(a != b for a, b in zip(seen, seen[1:], strict=False)), (
            f"the cursor stumbled at {base:,.0f}s of uptime: {seen}"
        )


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
            *agent._imagine_tools(settings),
            *agent._project_tools(settings),
            *agent._recall_tools(settings),
        ]
    }
    assert len(offered) == 9, "the tool list changed; the caption table probably needs to as well"
    for name in sorted(offered):
        line = agent._activity_line(Call(name, "{}"))
        assert line != "working…", f"{name} falls through to the line meant for invented tools"
        assert line.endswith(overlay.BUSY_MARK), f"{name} would not animate: {line!r}"


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        (Call("web_search", '{"query": "M8 torque"}'), "searching for M8 torque…"),
        (
            Call("draw_diagram", '{"request": "the fuse box", "style": "a wiring diagram"}'),
            "drawing the fuse box…",
        ),
        (
            Call("edit_photo", '{"request": "paint the doors matt black"}'),
            "editing the picture to paint the doors matt black…",
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


def _ink(frame: np.ndarray, ov: overlay.Overlay, line: int = 0) -> tuple[int, int] | None:
    """Where the lit text starts and stops on one line of the terminal's screen, or None.

    Read between the two mounts rather than across the whole screen. Both of the glass's side
    edges are buried under a rail, and a rail is the brightest thing on this panel - measure out
    to the glass's own corners and every line comes back the full width of the bay.
    """
    top = int(ov.caption_top + line * ov.caption_h)
    band = frame[top + 2 : top + ov.caption_h - 2, ov.caption_left - 2 : ov.caption_right + 2]
    on = (band[:, :, :3].astype(int).sum(axis=2) > 300) & (band[:, :, 3] > 150)
    lit = np.where(on.any(axis=0))[0]
    return (int(lit.min()) + ov.caption_left - 2, int(lit.max()) + ov.caption_left - 2) if (
        lit.size
    ) else None


def test_the_cursor_trails_the_sentence_rather_than_moving_it() -> None:
    # What the bubble could not do. It was sized to its own sentence, so an unreserved mark at the
    # end of a line made a dark rectangle grow and shrink behind the words - a thing no unit test
    # would catch and nobody could stand to look at. The glass is a fixed size now, so the only
    # thing left that a blink may move is the far end of its own line.
    #
    # Asleep, where the breath is off and the cursor is not: awake, every letter on the line is
    # also being dimmed and lifted by caption_pulse, and this is asking about position.
    ov = overlay.Overlay(800, 480)
    half = overlay.CURSOR_PERIOD_S / 2
    shown = dict(state=overlay.IDLE, level=0.0, detail="searching for M8 torque…")
    on = _ink(ov.render(phase=half / 2, **shown), ov)
    off = _ink(ov.render(phase=half * 1.5, **shown), ov)
    assert on[0] == off[0], f"the sentence shuffled with the cursor: {on} then {off}"
    assert on[1] > off[1], "the cursor is supposed to appear off the end of the line, not in it"
    assert on[1] - off[1] == pytest.approx(ov._cursor_w, abs=3), (
        "and to be exactly one cursor wide when it does"
    )


def test_the_cursor_never_runs_off_the_glass() -> None:
    # What reserving its width still buys, now that there is no slab edge for it to shove. The
    # wrap is done against a limit the cursor is already subtracted from, so a line that fills the
    # screen at rest still has room for it - and without that it lands under the bracket.
    ov = overlay.Overlay(800, 480)
    half = overlay.CURSOR_PERIOD_S / 2
    for detail in ("searching for an M8 stainless bolt…", "x" * 200 + "…", "wwwwww wwwwwwww w…"):
        for phase in (half / 2, half * 1.5):
            frame = ov.render(state=overlay.SEARCHING, level=0.0, detail=detail, phase=phase)
            for line in range(overlay.CAPTION_LINES):
                edges = _ink(frame, ov, line)
                assert edges is None or edges[1] <= ov.caption_right, (
                    f"{detail[:20]!r} line {line} ran to {edges} past {ov.caption_right}"
                )


def test_the_screen_is_the_same_size_whatever_is_on_it() -> None:
    # The whole of why this stopped being a bubble. A slab sized to its sentence has to move when
    # the sentence does; a terminal is a thing, and a thing that changed shape with what it was
    # printing would be a dialogue box with a bezel drawn on.
    ov = overlay.Overlay(800, 480)
    at_rest = dict(state=overlay.LISTENING, level=0.0, phase=0.0)
    short = ov.render(detail="doing a thing", **at_rest)
    long_ = ov.render(detail="doing a considerably longer thing than the other one", **at_rest)
    band = slice(int(ov.caption_top), int(ov.caption_top + overlay.CAPTION_LINES * ov.caption_h))
    moved = np.argwhere(np.any(short != long_, axis=2))
    assert moved.size, "the two captions rendered identically"
    assert moved[:, 0].min() >= band.start and moved[:, 0].max() < band.stop, (
        "the screen changed shape with its sentence"
    )
    assert _ink(short, ov)[0] == _ink(long_, ov)[0], (
        "the text is printed from the screen's own left edge, not laid out from its middle"
    )
    assert _ink(short, ov)[0] - ov.caption_left < ov.caption_h, "and from near enough that edge"


def test_the_cursor_actually_lands_on_the_panel() -> None:
    ov = overlay.Overlay(800, 480)
    half = overlay.CURSOR_PERIOD_S / 2
    shown = dict(state=overlay.SEARCHING, level=0.0, detail="searching…")
    assert not np.array_equal(ov.render(phase=half / 2, **shown),
                              ov.render(phase=half * 1.5, **shown)), "nothing blinked"


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
