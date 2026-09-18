"""A walkthrough: the three tools, the state they move, and the record they leave.

No key, no model, no session with anybody in it. What the model does with the notes is measured
by ``tools/talk_probe.py --script tutorial`` against the real thing; what the bar looks like is
tests/test_caption.py's, beside the rest of the terminal.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import pytest

from cyclops import agent, card, library, session, tutorial, ui
from cyclops.config import Settings

SEVEN = [f"step {n}" for n in range(1, 8)]


@dataclass
class Call:
    """As much of a Realtime function call as the dispatch looks at."""

    name: str
    arguments: str | None = "{}"
    call_id: str = "call_1"


@pytest.fixture(scope="module")
def settings() -> Settings:
    return Settings(api_key="")


@pytest.fixture
def voice(settings: Settings) -> agent.VoiceAgent:
    return agent.VoiceAgent(settings)


def _call(voice: agent.VoiceAgent, name: str, arguments: str = "{}") -> dict:
    """Drive one tool through the real dispatch and hand back what it sent the model."""
    sent: list[dict] = []

    async def _send_tool_output(call_id: str, output: dict) -> None:
        sent.append(output)

    async def _nothing() -> None:
        return None

    voice._send_tool_output = _send_tool_output  # type: ignore[method-assign]
    voice._request_response = _nothing  # type: ignore[method-assign]
    asyncio.run(voice._dispatch_tool(Call(name, arguments)))  # type: ignore[arg-type]
    assert len(sent) == 1, "every call is answered exactly once"
    return sent[0]


def test_the_three_tools_go_through_the_dispatch(voice: agent.VoiceAgent) -> None:
    started = _call(voice, "start_tutorial", json.dumps({"steps": SEVEN}))
    assert started["ok"] and (started["step"], started["of"]) == (1, 7)
    assert voice.tutorial is not None and voice.tutorial.current == "step 1"

    moved = _call(voice, "advance_tutorial")
    assert (moved["step"], voice.tutorial.number) == (2, 2)

    assert _call(voice, "end_tutorial")["ok"]
    assert voice.tutorial is None


def test_advance_walks_the_step_and_its_label_together(voice: agent.VoiceAgent) -> None:
    voice._start_tutorial(SEVEN)
    for number in range(2, 8):
        output = voice._advance_tutorial()
        assert voice.tutorial is not None
        assert (voice.tutorial.number, output["step"]) == (number, number)
        assert voice.tutorial.current == output["now"] == SEVEN[number - 1]


def test_running_off_the_end_and_ending_both_clear_it(voice: agent.VoiceAgent) -> None:
    voice._start_tutorial(SEVEN[:3])
    voice._advance_tutorial()
    voice._advance_tutorial()
    assert voice.tutorial is not None, "the last step is still a step"
    assert voice._advance_tutorial()["finished"]
    assert voice.tutorial is None

    voice._start_tutorial(SEVEN)
    voice._advance_tutorial()
    assert voice._end_tutorial()["ended"]
    assert voice.tutorial is None


def test_a_move_with_nothing_running_is_answered_and_changes_nothing(
    voice: agent.VoiceAgent,
) -> None:
    for move in (voice._advance_tutorial, voice._end_tutorial):
        output = move()
        assert not output["ok"] and output["note"]
        assert voice.tutorial is None


@pytest.mark.parametrize("count", [0, 1, tutorial.MAX_STEPS + 1, 20])
def test_a_list_it_cannot_draw_is_refused_and_nothing_goes_up(
    voice: agent.VoiceAgent, count: int
) -> None:
    output = voice._start_tutorial([f"step {n}" for n in range(count)])
    assert not output["ok"] and output["note"], "refused, with something to do about it"
    assert voice.tutorial is None


def test_a_refusal_leaves_a_running_walkthrough_where_it_was(voice: agent.VoiceAgent) -> None:
    voice._start_tutorial(SEVEN)
    voice._advance_tutorial()
    voice._start_tutorial([f"step {n}" for n in range(tutorial.MAX_STEPS + 1)])
    assert voice.tutorial is not None and voice.tutorial.number == 2


def test_rubbish_in_the_steps_is_dropped_not_raised() -> None:
    for arguments in (None, "", "not json", "[]", '{"steps": "one, two"}', '{"steps": null}'):
        assert agent._tool_steps(arguments) == [], arguments
    got = agent._tool_steps(json.dumps({"steps": ["  loosen   it ", "", None, 3, "x" * 500]}))
    assert got[:2] == ["loosen it", "3"]
    assert len(got) == 3 and len(got[2]) == tutorial.MAX_STEP_CHARS


def test_the_panel_is_handed_the_walkthrough(voice: agent.VoiceAgent, settings: Settings) -> None:
    control = ui.SessionController(settings, frames=None, entrypoint="test")
    assert control.status()["tutorial"] is None, "no session, no bar"
    control._agent = voice
    voice._start_tutorial(SEVEN)
    assert control.status()["tutorial"] is voice.tutorial


# ------------------------------------------------------------------ the record


@pytest.fixture
def log(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "_file_the_card", lambda settings: None)
    settings = Settings(api_key="", sessions_dir=tmp_path / "sessions", slug=False, projects=False)

    class FakeAgent:
        on_event = None

    return session.SessionLog(settings, FakeAgent(), entrypoint="cli")


def test_a_walkthrough_is_in_the_log_the_page_and_the_web_transcript(
    log, voice: agent.VoiceAgent
) -> None:
    # Three places, and one out of three is silent: a record note() writes but the page has no
    # branch for renders as nothing, and one missing from SPOKEN never reaches the companion.
    with log:
        log.event("you", text="walk me through swapping the cartridge")  # or nothing is kept
        voice._start_tutorial(["Turn the water off", "Pull the handle", "Swap the cartridge"])
        voice._advance_tutorial()
        voice._end_tutorial()
        voice._start_tutorial(SEVEN[:2])
        voice._advance_tutorial()
        voice._advance_tutorial()

    records = [r for r in card.read_log(log.dir / card.LOG_NAME)[0] if r["type"] == "tutorial"]
    assert [r["action"] for r in records] == [
        "started", "advanced", "ended", "started", "advanced", "finished",
    ]
    assert all(session._render_record(r) for r in records), "every one renders a line"
    page = (log.dir / card.PAGE_NAME).read_text()
    assert "Swap the cartridge" in page and "Pull the handle" in page
    shown = library.records(log.dir.parent, log.dir.name)
    assert sum(r["type"] == "tutorial" for r in shown) == len(records)
