"""The story crew: the gates that stop it, the frame that vetoes a beat, the retelling that passes.

Every model here is a :class:`FunctionModel` answering out of a list, so this suite opens no socket
and needs no key. What is deliberately not tested is whether the prompts are any good - that is a
judgement, and it is made by watching a clip on the Pi. What is here is the machinery around them,
which is where a bug would silently publish a clip nobody can follow.

One import of ``pydantic_ai`` for the whole file, which is the second it costs.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RunUsage

from cyclops import stories
from cyclops.config import Settings
from cyclops.stories import agents
from cyclops.stories import deps as deps_module
from cyclops.stories.models import Arc, Beat, Told

# ------------------------------------------------------------------ a crew that answers off a list


def answering(*replies):
    """A model that returns each of ``replies`` in turn as its structured answer."""
    left = list(replies)

    def reply(messages, info: AgentInfo) -> ModelResponse:
        payload = left.pop(0) if len(left) > 1 else left[0]
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, payload)])

    return FunctionModel(reply)


def crew(*, reader=None, director=None, looker=None, editor=None) -> agents.Models:
    return agents.Models(
        client=None,
        reader=reader or answering({"stories": [], "why_not": "nothing in it"}),
        director=director or answering({"stories": [], "why": "nothing in it"}),
        looker=looker or answering({"saw": "drawing", "what": "a wiring diagram"}),
        editor=editor or answering(A_GOOD_RETELLING),
    )


A_GOOD_RETELLING = {
    "asked": "He wanted the wiring drawn out.",
    "did": "Cyclops drew it on the panel.",
    "ended": "He said that was the one.",
    "clear": True,
    "missing": "",
}
A_BAD_RETELLING = {
    "asked": "",
    "did": "Something appeared.",
    "ended": "",
    "clear": False,
    "missing": "no ask: the clip opens on Cyclops answering",
}

RECORDS = [
    {"t": 10.0, "type": "you", "text": "draw me the wiring for that light", "dur": 3.0},
    {"t": 16.0, "type": "cyclops", "text": "here it is, coming up on the panel"},
    {"t": 40.0, "type": "sketch", "code": "x"},
    {"t": 50.0, "type": "you", "text": "that is the one, thanks", "dur": 2.0},
]


def a_story(title="The wiring", *, turn=(40.0, 46.0), payoff=(50.0, 54.0)) -> dict:
    return {
        "line": "He asks for the wiring and the diagram lands.",
        "title": title,
        "setup": {"start": 10.0, "end": 14.0, "what": "he asks"},
        "turn": {"start": turn[0], "end": turn[1], "what": "the diagram lands"},
        "payoff": {"start": payoff[0], "end": payoff[1], "what": "he likes it"},
    }


def told_of(**over) -> Told:
    if isinstance(over.get("turn"), tuple):
        start, end = over.pop("turn")
        over["turn"] = Beat(start=start, end=end, what="the diagram lands")
    return Told(**(a_story() | over))


def a_telling(tmp_path, models) -> deps_module.Telling:
    telling = deps_module.Telling(
        settings=Settings(api_key="k", sessions_dir=tmp_path),
        folder=tmp_path,
        records=RECORDS,
        timeline="[10.0] YOU: draw me the wiring",
        brief="A light",
        seconds=120.0,
        models=models,
    )
    return telling


@pytest.fixture(autouse=True)
def a_frame(monkeypatch):
    """Every test here has a recording it can pull a frame out of. None of them has ffmpeg."""
    monkeypatch.setattr(deps_module.Telling, "frame", lambda self, at: b"\xff\xd8jpeg")


# ------------------------------------------------------------------ the tiers


def test_a_reader_that_found_nothing_never_calls_the_director(tmp_path, monkeypatch) -> None:
    """Tier 2 is the gate. About eighteen sessions a card stop here for about a cent each."""
    def refuse(*a, **kw):
        raise AssertionError("the Director must not run when there is no candidate")

    monkeypatch.setattr(agents, "direct", refuse)
    monkeypatch.setattr(agents.Models, "build", lambda settings: crew())
    monkeypatch.setattr(agents.Models, "close", _nothing)
    outcome = stories.tell(tmp_path, RECORDS, "a timeline", "a brief", 120.0,
                           Settings(api_key="k", sessions_dir=tmp_path))
    assert outcome.tier == stories.READER
    assert outcome.stories == () and outcome.asked
    assert outcome.why == "nothing in it"


def test_a_candidate_reaches_the_crew_and_comes_back_a_story(tmp_path, monkeypatch) -> None:
    reader = answering({"stories": [a_story()], "why_not": ""})
    director = answering({"stories": [a_story()], "why": ""})
    monkeypatch.setattr(agents.Models, "build", lambda settings: crew(reader=reader,
                                                                     director=director))
    monkeypatch.setattr(agents.Models, "close", _nothing)
    outcome = stories.tell(tmp_path, RECORDS, "a timeline", "a brief", 120.0,
                           Settings(api_key="k", sessions_dir=tmp_path))
    assert outcome.tier == stories.CREW
    assert len(outcome.stories) == 1
    story = outcome.stories[0]
    assert [shot.kind for shot in story.shots] == ["setup", "turn", "payoff"]
    assert "wanted" in story.retelling
    assert story.shots[1].saw.startswith("drawing")


def test_a_session_with_no_key_is_never_asked(tmp_path) -> None:
    assert not stories.tell(tmp_path, RECORDS, "a timeline", "", 120.0,
                            Settings(api_key="", sessions_dir=tmp_path)).asked


# ------------------------------------------------------------------ the Looker's veto


def test_a_turn_beat_on_a_black_frame_is_rejected(tmp_path) -> None:
    """The recording is the panel, which shows Cyclops's own face when nothing is up."""
    telling = a_telling(tmp_path, crew(looker=answering({"saw": "black", "what": "nothing"})))
    assert "TURN beat holds nothing" in _why_not(telling, told_of())


def test_a_turn_beat_on_the_bare_eye_is_rejected(tmp_path) -> None:
    telling = a_telling(tmp_path, crew(looker=answering({"saw": "eye", "what": "the eye"})))
    assert "TURN beat holds nothing" in _why_not(telling, told_of())


def test_a_turn_beat_on_a_picture_passes(tmp_path) -> None:
    assert _why_not(a_telling(tmp_path, crew()), told_of()) == ""


def test_the_same_second_is_never_looked_at_twice(tmp_path) -> None:
    """Two Looker frames per candidate is the cap, and the cache is what keeps it one apiece."""
    seen = []

    def count(messages, info: AgentInfo) -> ModelResponse:
        seen.append(1)
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, {"saw": "photo", "what": "a bench"})]
        )

    telling = a_telling(tmp_path, crew(looker=FunctionModel(count)))
    _why_not(telling, told_of())
    _why_not(telling, told_of())
    assert len(seen) == 2, "the turn and the payoff, once each"


# ------------------------------------------------------------------ the Editor's veto


def test_a_story_a_stranger_cannot_follow_is_rejected(tmp_path) -> None:
    telling = a_telling(tmp_path, crew(editor=answering(A_BAD_RETELLING)))
    assert "no ask" in _why_not(telling, told_of())


def test_a_failing_retelling_is_repaired_once_and_then_dropped(tmp_path) -> None:
    """One repair. The Director is handed the missing beat, and if it sends the same thing back
    the story is dropped rather than published - and a story that did pass is still kept."""
    asked = []

    def twice(messages, info: AgentInfo) -> ModelResponse:
        asked.append(1)
        stories_out = [a_story("The wiring")] if len(asked) == 1 else [
            a_story("The wiring"), a_story("The torque", turn=(20.0, 26.0), payoff=(30.0, 34.0))
        ]
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, {"stories": stories_out, "why": ""})]
        )

    read = []

    def one_bad_one_good(messages, info: AgentInfo) -> ModelResponse:
        # The Editor is called once per *distinct* story, because its answer is cached on the
        # beats: the first story comes back unchanged, so it is never watched a second time.
        read.append(1)
        answer = A_BAD_RETELLING if len(read) == 1 else A_GOOD_RETELLING
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, answer)])

    telling = a_telling(
        tmp_path, crew(director=FunctionModel(twice), editor=FunctionModel(one_bad_one_good))
    )
    out = asyncio.run(agents.direct(telling, [Arc(**a_story())], RunUsage()))
    assert len(asked) == 2, "one repair, and only one"
    assert len(read) == 2, "and the story that came back unchanged was not watched again"
    assert [one.title for one in out.stories] == ["The torque"]
    assert [told.title for told, _ in telling.passed] == ["The torque"]


# ------------------------------------------------------------------ the shape check


def test_beats_out_of_order_come_back_to_the_director(tmp_path) -> None:
    telling = a_telling(tmp_path, crew())
    out_of_order = told_of(setup=Beat(start=50.0, end=54.0, what="x"))
    assert "out of order" in _why_not(telling, out_of_order)


def test_a_turn_too_short_to_see_comes_back_to_the_director(tmp_path) -> None:
    telling = a_telling(tmp_path, crew())
    assert "hold the new picture" in _why_not(telling, told_of(turn=(40.0, 41.0)))


def test_a_story_spanning_more_than_a_story_can_comes_back(tmp_path) -> None:
    telling = a_telling(tmp_path, crew())
    stretched = told_of(payoff=Beat(start=100.0, end=106.0, what="x"))
    assert "spans" in _why_not(telling, stretched)


def test_a_payoff_past_the_end_of_the_recording_comes_back(tmp_path) -> None:
    telling = a_telling(tmp_path, crew())
    over = told_of(turn=(40.0, 46.0), payoff=Beat(start=118.0, end=140.0, what="x"))
    assert "past the end" in _why_not(telling, over)


# ------------------------------------------------------------------ the budget


def test_running_out_of_budget_keeps_the_stories_that_already_passed(tmp_path, monkeypatch) -> None:
    """Over budget is not a failure. The work already finished is finished."""
    reader = answering({"stories": [a_story()], "why_not": ""})

    monkeypatch.setattr(agents.Models, "build", lambda settings: crew(reader=reader))
    monkeypatch.setattr(agents.Models, "close", _nothing)

    async def half(telling, arcs, usage):
        telling.passed.append((told_of(), agents.Retelling(**A_GOOD_RETELLING)))
        raise UsageLimitExceeded("out of money")

    monkeypatch.setattr(agents, "direct", half)
    outcome = stories.tell(tmp_path, RECORDS, "a timeline", "a brief", 120.0,
                           Settings(api_key="k", sessions_dir=tmp_path))
    assert outcome.asked and len(outcome.stories) == 1
    assert outcome.stories[0].title == "The wiring"


# ------------------------------------------------------------------ what the Editor is shown


def test_the_editor_never_sees_the_summary_or_the_timeline(tmp_path) -> None:
    """The blind test only tests the clip if it is actually blind."""
    telling = a_telling(tmp_path, crew())
    watched = telling.as_watched(told_of(setup=Beat(start=10.0, end=20.0, what="he asks")))
    assert "A light" not in watched, "the summary is the Director's brief and never the Editor's"
    assert "the wiring for that light" in watched
    assert "here it is" in watched, "Cyclops's line inside a beat is part of the clip"


def test_the_editor_is_shown_only_what_is_inside_the_beats(tmp_path) -> None:
    telling = a_telling(tmp_path, crew())
    telling.records = [*RECORDS, {"t": 95.0, "type": "you", "text": "unrelated later", "dur": 1.0}]
    assert "unrelated later" not in telling.as_watched(told_of())


def _why_not(telling: deps_module.Telling, told: Told) -> str:
    """Run the crew's own gate over one story, outside a Director run."""

    class Ctx:
        deps = telling
        usage = RunUsage()

    return asyncio.run(agents._cannot_stand_alone(Ctx(), told))  # noqa: SLF001


async def _nothing(self) -> None:
    return None
