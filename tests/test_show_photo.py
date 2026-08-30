"""The shutter's handoff into the session thread, and how the page describes what it produced.

This is the part of the change with no natural coverage: `show_photo` is called from the
kiosk's snap thread and reaches an event loop it does not own, and every one of its guards
protects against a window that only exists for a second or two during setup or teardown.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from cyclops import session
from cyclops.config import Settings
from cyclops.ui import SessionController


class FakeAgent:
    """Just the three things `show_photo` asks an agent about, plus a record of the handoff."""

    def __init__(self, *, ready: bool = True, connected: bool = True) -> None:
        self.ready = asyncio.Event()
        if ready:
            self.ready.set()
        self.connected = connected
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


def test_a_shown_photo_says_cyclops_looked_at_it():
    rendered = session._render_photo({"by": "you", "shown": True, "file": "photos/a.jpg"}, "0:05")
    assert "looked at it" in rendered


def test_a_photo_with_no_session_says_it_was_never_seen():
    rendered = session._render_photo({"by": "you", "shown": False, "file": "photos/a.jpg"}, "0:05")
    assert "never saw this one" in rendered


def test_an_old_record_with_no_shown_key_still_reads_correctly():
    """Records written before the shutter fed the conversation. --fix re-renders these."""
    rendered = session._render_photo({"by": "you", "file": "photos/a.jpg"}, "0:05")
    assert "never saw this one" in rendered


def test_an_old_cyclops_photo_keeps_its_caption_and_focus():
    """The model no longer holds the shutter, but a card full of old sessions still says it did."""
    rendered = session._render_photo(
        {"by": "cyclops", "focus": "the mitre joint", "file": "photos/a.jpg"}, "0:05"
    )
    assert "Cyclops took a photo" in rendered
    assert "the mitre joint" in rendered


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
