"""What the sketch surface has to get right to be worth having.

Nothing here needs a browser, a panel or a Pi: the whole point of compiling on this side is that
the thing under test is a function from a string to a dict.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from cyclops import agent, arguments, sketch
from cyclops.config import Settings

WHOLE = """with PrefabApp() as app:
    with Column(gap=3):
        Heading("Shelf brackets")
        Metric(label="M8 bolt", value="25 Nm")
"""


def _leaves(wire: dict | None) -> int:
    """How many components are in a frame, at any depth."""
    if wire is None:
        return 0

    def walk(node: object) -> int:
        if not isinstance(node, dict):
            return 0
        return 1 + sum(walk(child) for child in node.get("children", []) or [])

    return walk(wire.get("view"))


def test_a_half_written_argument_gives_up_the_code_so_far():
    typed = '{"code": "with PrefabApp() as app:\\n    Heading(\\"Sh'
    code, closed = arguments.value(typed, "code")
    assert not closed
    assert code == 'with PrefabApp() as app:\n    Heading("Sh'


def test_a_program_grows_on_the_glass_as_it_is_typed():
    """The claim the whole feature rests on: a prefix draws less of the same thing, not nothing."""
    sizes = {_leaves(sketch.compile(WHOLE[:i])) for i in range(1, len(WHOLE) + 1)}
    assert len(sizes - {0}) > 1, "every prefix that compiled drew the same thing"
    assert max(sizes) == _leaves(sketch.compile(WHOLE))


def test_code_that_does_not_run_draws_nothing():
    assert sketch.compile("with PrefabApp() as app:\n    Heading(") is None
    assert sketch.compile("import os\nos.listdir()") is None
    assert sketch.compile("   ") is None


def test_a_broken_program_does_not_poison_the_next_one():
    """A prefix that dies inside a `with` never runs the __exit__ that would pop the stack.

    Without a fresh context per compile, the Column left open here is still the current parent
    when the next program runs, and everything it draws lands inside a container nobody wrote.
    """
    sketch.compile("with PrefabApp() as app:\n    with Column():\n        Heading(1 / 0)")
    after = sketch.compile('with PrefabApp() as app:\n    Heading("one")')
    assert json.dumps(after).count("Column") == 0


def test_a_frame_is_a_white_sheet_and_carries_a_view():
    """White is not taste: the charts default to a light page, and on black a chart's axes and
    gridlines come out dark grey on near-black."""
    wire = sketch.compile(WHOLE)
    assert wire["mode"] == "light"
    assert wire["view"]


def test_more_than_a_screenful_is_cut_off():
    long = 'with PrefabApp() as app:\n' + '    Heading("x")\n' * 400
    assert len(long) > sketch.MAX_SKETCH_CHARS
    assert _leaves(sketch.compile(long)) < long.count("Heading")


def test_a_watcher_gets_what_is_up_and_what_comes_next():
    sketch.push(None)
    first = sketch.compile(WHOLE)
    sketch.push(first)
    with sketch.listening() as box:
        assert box.get_nowait() == first
        sketch.push(None)
        assert box.get_nowait() is None
    sketch.push(first)  # nobody is listening now; this must not raise


# ---------------------------------------------------------------- while it is being typed
#
# The claim the tool description makes to the model - "it draws as you write it" - is made true
# by two handlers on the agent and a timer between them. These drive them the way the wire does,
# which is the part the first version of these tests got wrong: they fed deltas one at a time in
# a synchronous loop, and a synchronous loop is the one arrival pattern the wire never produces.


class _Item:
    """A function_call item off ``response.output_item.added``."""

    def __init__(self, name: str, call_id: str = "call_1") -> None:
        self.type = "function_call"
        self.name = name
        self.call_id = call_id


@pytest.fixture
def voice(monkeypatch) -> agent.VoiceAgent:
    monkeypatch.setattr(sketch, "THROTTLE_S", 0.01)  # the suite has ten seconds for everything
    made = agent.VoiceAgent(Settings(api_key="", sketch=True))
    made._spawn = lambda coro: coro.close()  # no panel here; the reveal is its own test
    sketch.push(None)
    return made


def _burst(voice: agent.VoiceAgent, code: str, *, size: int = 8) -> None:
    """Deliver a whole call's arguments the way the wire does: all at once.

    Measured against gpt-realtime-2.1, a 170-character program arrives as 52 deltas inside
    386 ms. Nothing here sleeps between them, because nothing there does either.
    """
    voice._on_output_item(_Item("sketch"))
    wire = json.dumps({"code": code})
    for i in range(0, len(wire), size):
        voice._on_sketch_delta("call_1", wire[i : i + size])


async def _settle() -> None:
    await asyncio.sleep(sketch.THROTTLE_S * 4)


def test_a_burst_of_deltas_still_draws(voice: agent.VoiceAgent):
    """The one that matters. Deltas arrive faster than the throttle; a throttle that drops
    instead of deferring compiles ``{"`` and nothing else, and the panel stays empty."""

    async def go():
        _burst(voice, WHOLE)
        await _settle()

    asyncio.run(go())
    assert "Shelf brackets" in json.dumps(sketch.current())


def test_nothing_is_lost_at_the_end_of_the_stream(voice: agent.VoiceAgent):
    """The last delta must leave a compile behind it, or the final line never lands."""

    async def go():
        _burst(voice, WHOLE)
        await _settle()

    asyncio.run(go())
    assert "M8 bolt" in json.dumps(sketch.current()), "the tail of the program was dropped"


def test_the_panel_is_asked_for_once_and_not_before_something_compiled(voice: agent.VoiceAgent):
    asked = []
    voice._spawn = lambda coro: (coro.close(), asked.append(1))

    async def go():
        voice._on_output_item(_Item("sketch"))
        voice._on_sketch_delta("call_1", '{"code": "with PrefabApp() as ')
        await _settle()
        assert not asked, "the browser was uncovered onto nothing"
        for chunk in ('app:\\n    Heading(\\"hi\\")', '"}'):
            voice._on_sketch_delta("call_1", chunk)
        await _settle()

    asyncio.run(go())
    assert len(asked) == 1, "the panel was asked for once per frame instead of once per sketch"


def test_another_tool_being_typed_draws_nothing(voice: agent.VoiceAgent):
    async def go():
        voice._on_output_item(_Item("point_at"))
        voice._on_sketch_delta("call_1", json.dumps({"code": WHOLE}))
        await _settle()

    asyncio.run(go())
    assert sketch.current() is None


# ---------------------------------------------------------------- no drawing on the scratchpad
#
# Drawing goes to the image tool. A Mermaid chart or hand-typed SVG on the scratchpad never once
# came out more useful than a picture, so neither is a name the model's code can reach.


@pytest.mark.parametrize(
    "call",
    ["Mermaid('graph LR\\n  B[Battery] --> F[Fuse]')", "Svg('<svg viewBox=\"0 0 8 8\"/>')"],
)
def test_a_sketch_that_draws_compiles_to_nothing(call):
    assert sketch.compile(f"with PrefabApp() as app:\n    {call}\n") is None
