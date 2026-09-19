"""Cyclops speaks first: one greeting asked for as the session comes up, and never over the lid."""

from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace

import pytest

from cyclops import agent, sfx
from cyclops.config import Settings

UP = SimpleNamespace(type="session.updated")
LID_S = 0.05  # the lid sound, shortened: the suite has ten seconds for everything


class FakeConn:
    """Records what the agent asks the server for, and answers nothing."""

    def __init__(self) -> None:
        self.responses: list[dict] = []
        self.items: list[dict] = []
        self.response = SimpleNamespace(create=self._response)
        self.conversation = SimpleNamespace(item=SimpleNamespace(create=self._item))

    async def _response(self, **kwargs) -> None:
        self.responses.append(kwargs)

    async def _item(self, **kwargs) -> None:
        self.items.append(kwargs)


class FakeSpeaker:
    def __init__(self) -> None:
        self.fed: list[bytes] = []

    def begin_item(self) -> None:
        pass

    def feed(self, pcm: bytes) -> None:
        self.fed.append(pcm)


@pytest.fixture
def voice(monkeypatch) -> agent.VoiceAgent:
    made = agent.VoiceAgent(Settings(api_key="", sounds=False))
    made._conn = FakeConn()
    monkeypatch.setattr(made, "_log", lambda *a, **k: None)
    return made


async def _settle(made: agent.VoiceAgent) -> None:
    await asyncio.gather(*made._background)


def test_one_greeting_is_asked_for_and_a_second_ready_asks_for_nothing(voice):
    async def go():
        await voice._handle_event(UP)
        await _settle(voice)
        await voice._handle_event(UP)
        await _settle(voice)

    asyncio.run(go())
    assert len(voice.conn.responses) == 1
    assert voice.conn.items == [], "nobody said anything: there is no user turn in front of it"


def test_the_greeting_cannot_call_a_tool(voice):
    async def go():
        await voice._handle_event(UP)
        await _settle(voice)

    asyncio.run(go())
    (asked,) = voice.conn.responses
    assert asked["response"]["tool_choice"] == "none"


def test_voice_waits_for_the_lid_and_then_plays_at_once(voice, monkeypatch):
    speaker = voice.speaker = FakeSpeaker()
    monkeypatch.setattr(voice.cues, "play", lambda name, **_: LID_S)
    monkeypatch.setattr(sfx, "SETTLE_S", 0.0)

    def say(pcm: bytes) -> None:
        voice._play("greeting", base64.b64encode(pcm).decode("ascii"))

    async def go():
        await voice._handle_event(UP)
        say(b"early")
        assert speaker.fed == [], "the lid is still sounding"
        await asyncio.sleep(LID_S * 2)
        assert speaker.fed == [b"early"]
        say(b"late")
        assert speaker.fed == [b"early", b"late"]
        await _settle(voice)

    asyncio.run(go())
