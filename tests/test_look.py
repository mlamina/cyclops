"""take_a_look: Cyclops takes the photo when they ask it to look, and answers off it."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from cyclops import agent, panel
from cyclops.config import Settings
from cyclops.webcam import Capture, WebcamError


class Called:
    name = "take_a_look"
    call_id = "call_1"
    arguments = "{}"


@pytest.fixture
def look(monkeypatch):
    """Run take_a_look with the socket, camera and panel faked. Returns (agent, log, shutters)."""
    made = agent.VoiceAgent(Settings(api_key=""))
    made._conn = object()
    log: list = []  # every item sent, and "response" where one was asked for, in order
    shutters: list = []

    async def _send_item(item):
        log.append(item)

    async def _request_response():
        log.append("response")

    monkeypatch.setattr(made, "_send_item", _send_item)
    monkeypatch.setattr(made, "_request_response", _request_response)
    monkeypatch.setattr(made, "_log", lambda *a, **k: None)
    monkeypatch.setattr(panel, "shutter", lambda: shutters.append(1) or True)

    def run(capture: Capture | Exception) -> None:
        async def fake_capture(*_a, **_k):
            if isinstance(capture, Exception):
                raise capture
            return capture

        monkeypatch.setattr(agent, "capture_image_async", fake_capture)
        asyncio.run(made._dispatch_tool(Called()))

    return made, log, shutters, run


def test_the_tool_is_left_out_when_switched_off() -> None:
    assert agent._look_tools(Settings(api_key="", look=False)) == []


def test_a_look_answers_the_call_then_shows_the_photo_then_asks_once(look) -> None:
    made, log, shutters, run = look
    run(Capture("data:image/jpeg;base64,x", Path("/p/12-00-00_cyclops.jpg"), 640, 480, 1024, 0))

    output, image, response = log
    assert output["type"] == "function_call_output"
    assert json.loads(output["output"])["ok"] is True
    assert any(part["type"] == "input_image" for part in image["content"])
    assert response == "response"
    assert shutters == [1]
    assert made._picture_named("12-00-00_cyclops") is not None, "point_at can find it"


def test_a_camera_that_fails_is_said_and_neither_flashes_nor_shows(look) -> None:
    _, log, shutters, run = look
    run(WebcamError("no recent frame"))

    output, response = log
    assert json.loads(output["output"])["ok"] is False
    assert response == "response"
    assert shutters == []
