"""Device extensions: what a plugged-in device adds to the prompt and the tools, and takes away.

The bus is faked at :func:`cyclops.devices.connected`, as the prompt tests in test_devices.py do,
and the tool half is driven by ``tests/fake_extensions`` - a synth with one tool - because the one
extension that ships carries none.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest

from cyclops import agent, devices, extensions, webcam
from cyclops.config import Settings

FAKES = Path(__file__).parent / "fake_extensions"
RULES = Path(__file__).parent.parent / "deploy" / "99-useeplus-camera.rules"

SCOPE = devices.Device("2ce3", "3828", "Endoscope", devices.CAMERA)
SIBLING = devices.Device("0329", "2022", "Endoscope", devices.CAMERA)
SYNTH = devices.Device("f00d", "0303", "Bench Synth", devices.MUSIC)
C920 = devices.Device("046d", "08e5", "HD Pro Webcam C920", devices.CAMERA)
UP = SimpleNamespace(type="session.updated")


class _Conn:
    """test_devices.py's fake conn, plus the session.update a re-arm sends."""

    def __init__(self) -> None:
        self.responses: list[dict] = []
        self.items: list[dict] = []
        self.updates: list[dict] = []
        self.response = SimpleNamespace(create=self._response)
        self.conversation = SimpleNamespace(item=SimpleNamespace(create=self._item))
        self.session = SimpleNamespace(update=self._update)

    async def _response(self, **kwargs) -> None:
        self.responses.append(kwargs)

    async def _item(self, **kwargs) -> None:
        self.items.append(kwargs)

    async def _update(self, **kwargs) -> None:
        self.updates.append(kwargs["session"])


def on_bus(monkeypatch, *plugged: devices.Device) -> None:
    monkeypatch.setattr(agent.devices, "connected", lambda root=None: tuple(plugged))


@pytest.fixture
def with_synth(monkeypatch) -> None:
    """The fake synth installed alongside what ships, the way talk_probe puts it in."""
    monkeypatch.setattr(agent.extensions, "load", partial(extensions.load, extra=FAKES))


def opened(monkeypatch, *plugged: devices.Device) -> agent.VoiceAgent:
    """A session that opened with *plugged* on the bus and has been told it is ready."""
    on_bus(monkeypatch, *plugged)
    made = agent.VoiceAgent(Settings(api_key="", sounds=False))
    monkeypatch.setattr(made, "_log", lambda *a, **k: None)
    made._conn = _Conn()
    made.session_config()
    made.ready.set()
    return made


def names(tools) -> list[str]:
    return [tool["name"] for tool in tools]


# ---------------------------------------------------------------- what the prompt carries

def scope_text() -> str:
    (scope,) = [one for one in extensions.load() if one.name == "Endoscope"]
    return scope.instructions


def test_the_endoscope_is_in_the_prompt_only_while_it_is_plugged_in(monkeypatch) -> None:
    on_bus(monkeypatch, SCOPE, C920)
    assert scope_text() in agent.build_instructions(Settings(api_key=""))
    on_bus(monkeypatch, C920)
    written = agent.build_instructions(Settings(api_key=""))
    assert scope_text() not in written and extensions.HEADER not in written


def test_a_kind_of_device_matches_one_nobody_wrote_an_id_for() -> None:
    any_audio = extensions.Extension(name="Audio in", instructions="x", categories=("music",))
    interface = devices.Device("1397", "0509", "UMC204HD", devices.MUSIC)
    matched = extensions.matching((C920, interface), (any_audio,))
    assert [(m.extension, m.device) for m in matched] == [(any_audio, interface)]


def test_two_endoscopes_are_one_paragraph(monkeypatch) -> None:
    on_bus(monkeypatch, SCOPE, SIBLING)
    assert agent.build_instructions(Settings(api_key="")).count(scope_text()) == 1


def test_a_module_that_raises_on_import_costs_only_itself(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "broken.py").write_text("raise RuntimeError('a driver that is not installed')\n")
    (tmp_path / "synth.py").write_text((FAKES / "synth.py").read_text())
    loaded = extensions.load(extra=tmp_path)
    assert [one.name for one in loaded] == [one.name for one in extensions.load()] + ["Bench Synth"]
    monkeypatch.setattr(agent.extensions, "load", partial(extensions.load, extra=tmp_path))
    made = opened(monkeypatch, SCOPE, SYNTH)
    config = made.session_config()
    assert scope_text() in config["instructions"] and "read_patch" in names(config["tools"])


def test_finding_and_matching_fits_in_the_greetings_margin() -> None:
    """The same bar test_devices sets the bus read: it is on the path of a session opening."""
    crowd = [devices.Device("046d", f"08{n:02x}", f"Thing {n}", devices.CAMERA) for n in range(12)]
    extensions.load(extra=FAKES)  # the imports, which a session pays once per process
    start = time.perf_counter()
    for _ in range(10):
        extensions.matching((*crowd, SCOPE), extensions.load(extra=FAKES))
    assert (time.perf_counter() - start) / 10 * 1000 < 5.0


# ---------------------------------------------------------------- the device's own table

def test_the_endoscope_ids_agree_everywhere_they_are_written() -> None:
    """The extension, the camera opener and the udev rules each carry them; none may drift."""
    (scope,) = [one for one in extensions.load() if one.name == "Endoscope"]
    ruled = re.findall(r'ATTR\{idVendor\}=="(\w+)", ATTR\{idProduct\}=="(\w+)"', RULES.read_text())
    assert set(scope.usb_ids) == set(webcam.USEEPLUS_IDS) == {f"{v}:{p}" for v, p in ruled}
    assert {ident: devices.known()[ident] for ident in scope.usb_ids} == {
        ident: (devices.CAMERA, "Endoscope") for ident in scope.usb_ids}


# ---------------------------------------------------------------- the tools

def test_a_tool_is_offered_only_while_its_device_is_in(monkeypatch, with_synth) -> None:
    assert "read_patch" in names(opened(monkeypatch, SYNTH).session_config()["tools"])
    assert "read_patch" not in names(opened(monkeypatch, C920).session_config()["tools"])


def call(name: str = "read_patch") -> SimpleNamespace:
    return SimpleNamespace(name=name, arguments="{}", call_id="call-1")


def answered(made: agent.VoiceAgent) -> dict:
    (item,) = [one["item"] for one in made.conn.items]
    assert item["type"] == "function_call_output" and item["call_id"] == "call-1"
    return json.loads(item["output"])


def test_a_call_reaches_the_handler_and_its_answer_goes_back(monkeypatch, with_synth) -> None:
    made = opened(monkeypatch, SYNTH)
    asyncio.run(made._dispatch_tool(call()))
    output = answered(made)
    assert output["ok"] and output["synth"] == "Bench Synth" and len(output["steps"]) == 16
    assert len(made.conn.responses) == 1


def test_a_handler_that_raises_still_answers(monkeypatch) -> None:
    def boom(args: dict, device: devices.Device) -> dict:
        raise OSError("the synth stopped answering")

    schema = {"type": "function", "name": "read_patch", "parameters": {"type": "object"}}
    broken = extensions.Extension(name="Bench Synth", instructions="x", usb_ids=(SYNTH.ident,),
                                  tools=(extensions.Tool(schema, boom, "reading…"),))
    monkeypatch.setattr(agent.extensions, "load", lambda: (broken,))
    made = opened(monkeypatch, SYNTH)
    asyncio.run(made._dispatch_tool(call()))
    assert answered(made)["ok"] is False and len(made.conn.responses) == 1


def test_the_panel_says_the_tools_own_caption(monkeypatch, with_synth) -> None:
    made = opened(monkeypatch, SYNTH)
    line = agent._activity_line(call(), made._matched)
    assert line == made._matched[0].extension.tools[0].caption


# ---------------------------------------------------------------- plugged in mid-session

def rearm(made: agent.VoiceAgent, arrived=(), left=()) -> None:
    async def go() -> None:
        await made.add_bus_change(tuple(arrived), tuple(left))
        await made._handle_event(UP)  # the server's answer to the update
        await asyncio.gather(*made._background)

    asyncio.run(go())


def test_a_plug_mid_session_rearms_once_and_greets_nobody(monkeypatch, with_synth) -> None:
    made = opened(monkeypatch)
    heard = []
    monkeypatch.setattr(made.cues, "play", lambda name, **_: heard.append(name) or 0.0)
    rearm(made, arrived=[SYNTH])
    (update,) = made.conn.updates
    assert "read_patch" in names(update["tools"])
    assert "audio" not in update, "a voice cannot be changed once it has spoken"
    assert heard == [] and len(made.conn.responses) == 1, "one line about the plug, nothing more"


def test_an_unplug_takes_its_tools_and_its_paragraph(monkeypatch, with_synth) -> None:
    made = opened(monkeypatch, SYNTH, SCOPE)
    rearm(made, left=[SYNTH])
    (update,) = made.conn.updates
    assert "read_patch" not in names(update["tools"])
    assert "Bench Synth:" not in update["instructions"] and scope_text() in update["instructions"]
    assert made.conn.responses == [], "being told it is out asks for nothing"


def test_the_rearmed_prompt_is_the_opened_one_bar_the_block(monkeypatch, with_synth) -> None:
    """The clock belongs to the greeting: a plug an hour in must not restate the hour."""
    clock = ["WHEN IT IS\nTen past nine."]
    monkeypatch.setattr(agent, "_standing", lambda settings, found: ("RULES", clock[0]))
    made = opened(monkeypatch)
    was = made.session_config()["instructions"]
    clock[0] = "WHEN IT IS\nTen past ten."
    rearm(made, arrived=[SYNTH])
    (update,) = made.conn.updates
    block = extensions.block(made._matched)
    assert update["instructions"].replace(block + "\n", "", 1) == was
