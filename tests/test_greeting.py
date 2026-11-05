"""A session coming up says nothing and sounds nothing: he should not notice it (hq#572)."""

from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace

import pytest

from cyclops import agent
from cyclops.config import Settings

UP = SimpleNamespace(type="session.updated")


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


def test_a_session_coming_up_asks_for_nothing_and_sounds_nothing(voice, monkeypatch):
    played: list[str] = []
    monkeypatch.setattr(voice.cues, "play", lambda name, **_: played.append(name) or 0.0)

    async def go():
        await voice._handle_event(UP)
        await _settle(voice)
        await voice._handle_event(UP)
        await _settle(voice)

    asyncio.run(go())
    assert voice.conn.responses == []
    assert voice.conn.items == []
    assert played == []


def test_voice_plays_at_once(voice):
    speaker = voice.speaker = FakeSpeaker()

    async def go():
        await voice._handle_event(UP)
        voice._play("first", base64.b64encode(b"early").decode("ascii"))
        assert speaker.fed == [b"early"]
        await _settle(voice)

    asyncio.run(go())
