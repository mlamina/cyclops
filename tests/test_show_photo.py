"""The shutter's handoff into the session thread, and how the page describes what it produced.

This is the part of the change with no natural coverage: `show_photo` is called from the
kiosk's snap thread and reaches an event loop it does not own, and every one of its guards
protects against a window that only exists for a second or two during setup or teardown.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from cyclops import session
from cyclops.config import Settings
from cyclops.ui import SessionController


class FakeAgent:
    """Just the three things `show_photo` asks an agent about, plus a record of the handoff."""

    def __init__(self, *, ready: bool = True, connected: bool = True) -> None:
        self.ready = asyncio.Event()
        self.ready_at: float | None = None
        if ready:
            self.ready_at = time.monotonic()
            self.ready.set()
        self.connected = connected
        self.activity = ""  # the detail line asks every agent this, including a fake one
        self.tutorial = None  # ...and so does the status line's bar
        self.photos: list[object] = []

    def queue_photo(self, capture: object) -> None:
        self.photos.append(capture)


@pytest.fixture
def controller(tmp_path):
    settings = Settings(api_key="", sessions_dir=tmp_path / "sessions", slug=False, projects=False)
    return SessionController(settings, frames=None, entrypoint="test")


@pytest.fixture
def running_loop():
    """A real loop on a real thread, the way a session runs one."""
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def run() -> None:
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    ready.wait(5)
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(5)
    loop.close()


def test_the_clock_does_not_start_before_the_session_does(controller, running_loop):
    """What the panel counts is a conversation, not the wait for one.

    It used to start on the tap and run through the connect, so the clock beside REC was already
    seconds into a session by the time his lid finished opening - and the lid is now the thing
    that says the session began. The recording's own clock still starts on the tap, because what
    that measures really does.
    """
    assert controller.status()["elapsed"] is None, "nothing running, nothing to count"

    controller._loop, controller._agent = running_loop, FakeAgent(ready=False)
    assert controller.status()["elapsed"] is None, "asked for, not yet arrived"

    controller._agent = FakeAgent()
    assert controller.status()["elapsed"] is not None


def test_no_session_means_no_handoff(controller):
    assert controller.show_photo(object()) is False


def test_a_live_session_gets_the_photo(controller, running_loop):
    agent = FakeAgent()
    controller._loop, controller._agent = running_loop, agent
    shot = object()

    assert controller.show_photo(shot) is True
    for _ in range(100):  # the callback runs on the loop thread, not this one
        if agent.photos:
            break
        threading.Event().wait(0.01)
    assert agent.photos == [shot]


def test_a_connecting_session_is_refused(controller, running_loop):
    """`ready` is not set yet: the config has not been accepted, so there is nothing to tell."""
    controller._loop, controller._agent = running_loop, FakeAgent(ready=False)
    assert controller.show_photo(object()) is False


def test_a_disconnected_agent_is_refused(controller, running_loop):
    """The window this exists for: `ready` is a latch and stays set long after the socket goes.

    Between `agent.run()` dropping the connection and the controller clearing `_agent`, the
    session folder is still being finished - which takes seconds. Asking `ready` alone here
    would sail through and hand a photo to a closed socket.
    """
    controller._loop, controller._agent = running_loop, FakeAgent(connected=False)
    assert controller.show_photo(object()) is False


def test_a_closed_loop_is_refused_not_raised(controller):
    """A tap landing in the instant the session tore down must not kill the snap thread."""
    loop = asyncio.new_event_loop()
    loop.close()
    controller._loop, controller._agent = loop, FakeAgent()
    assert controller.show_photo(object()) is False


def test_a_stopped_but_unclosed_loop_is_refused(controller):
    """Not running means the callback would be queued and never run."""
    loop = asyncio.new_event_loop()
    controller._loop, controller._agent = loop, FakeAgent()
    try:
        assert controller.show_photo(object()) is False
    finally:
        loop.close()


def test_stop_survives_a_closed_loop(controller):
    """Same window, but this one is called from the render thread and used to take the kiosk out."""
    loop = asyncio.new_event_loop()
    loop.close()
    controller._loop, controller._task = loop, asyncio.Future
    controller.stop()  # must not raise


# ---- how the page describes the result ----


def test_every_kind_of_photo_record_reads_as_its_own_thing():
    """One line per way a picture reaches the card, and no two of them alike.

    What has to hold is that the arms are told apart and that what was asked for survives into
    the page - not the sentence each arm happens to be written in. A record with no ``shown`` key
    is an old one, from before the shutter fed the conversation, and has to read as a photo
    nobody saw rather than fall off the end of the table.
    """
    kinds = {
        "shown": {"by": "you", "shown": True, "file": "photos/a.jpg"},
        "unshown": {"by": "you", "shown": False, "file": "photos/a.jpg"},
        "legacy": {"by": "you", "file": "photos/a.jpg"},
        "drawn": {"by": "drawn", "request": "the relay wiring",
                  "file": "photos/14-35-01_drawn.jpg"},
        "failed": {"by": "drawn", "request": "the relay wiring", "error": "the drawing failed"},
        "cyclops": {"by": "cyclops", "focus": "the mitre joint", "file": "photos/a.jpg"},
    }
    said = {name: session._render_photo(record, "0:05") for name, record in kinds.items()}

    assert said["legacy"] == said["unshown"], "an old record reads as one nobody saw"
    apart = {name: line for name, line in said.items() if name != "legacy"}
    assert len(set(apart.values())) == len(apart), f"two arms read the same: {apart}"

    assert "the relay wiring" in said["drawn"] and "the relay wiring" in said["failed"]
    assert "the mitre joint" in said["cyclops"]
    assert "![Drew, " in said["drawn"], "the picture itself, embedded the way an edit's is"
    assert "![" not in said["failed"], "a drawing that failed has no picture to show"


# ---- the CLI's shutter ----


def test_a_non_tty_stdin_is_never_watched(monkeypatch):
    """The spin trap: under systemd or a pipe, stdin is at EOF and therefore always readable.

    An unguarded reader on that fd fires its callback in a tight loop forever, so the isatty
    gate is the whole safety mechanism - not a nicety.
    """
    from cyclops import app

    monkeypatch.setattr(app.sys.stdin, "isatty", lambda: False)

    class LoudLoop:
        def add_reader(self, *_):
            raise AssertionError("a non-terminal stdin must never get a reader")

    teardown = app.watch_stdin(LoudLoop(), lambda: None)
    teardown()  # must be callable and harmless


# ---- what the handoff does at the far end ----


def test_a_snapped_photo_arrives_without_asking_for_a_reply(monkeypatch, tmp_path):
    """The shutter is not a question. The image goes into the conversation and nothing is asked
    of the model, so the next thing it says is an answer to whatever they eventually say."""
    from cyclops.agent import VoiceAgent
    from cyclops.webcam import Capture

    made = VoiceAgent(Settings(api_key=""))
    made._conn = object()  # `connected` asks only whether this is set
    sent: list[dict] = []
    asked: list[bool] = []

    async def _send_item(item):
        sent.append(item)

    async def _request_response():
        asked.append(True)

    monkeypatch.setattr(made, "_send_item", _send_item)
    monkeypatch.setattr(made, "_request_response", _request_response)
    monkeypatch.setattr(made, "_log", lambda *a, **k: None)

    shot = Capture("data:image/jpeg;base64,x", tmp_path / "a.jpg", 640, 480, 1024, 0)
    asyncio.run(made.add_photo(shot))

    assert len(sent) == 1 and sent[0]["role"] == "user"
    assert any(part["type"] == "input_image" for part in sent[0]["content"])
    assert not asked, "a photo on its own must not start a turn"
