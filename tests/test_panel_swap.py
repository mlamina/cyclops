"""Putting a second picture on a panel that is already showing one.

Everything Cyclops can put on the glass goes through one door - ``panel.show()``, which asks
the kiosk - and that door used to answer "no" whenever anything at all had the panel. That was
right while a picture was a rare event. It stopped being right when the shutter started putting
photos up, because the ordinary shape of the thing is now: a photo goes up, and a minute later
the edit somebody asked for lands while it is still there.

Refusing that second picture is not a small wrongness. ``show()`` returning False is what makes
the model say out loud that there is no screen free - about a picture the page would have painted
inside 400 ms, because it polls the offer file and repaints on any new id.

No camera and no panel here: the method under test is three branches over three flags, so it is
exercised against a stand-in rather than a real ``Kiosk``, which would want a webcam to build.
"""

from __future__ import annotations

import threading

import pytest

from cyclops import kiosk
from cyclops.kiosk import Kiosk


class Cues:
    """The sound the panel makes when something lands on it, counted rather than played."""

    def __init__(self) -> None:
        self.played: list[str] = []

    def play(self, name: str) -> None:
        self.played.append(name)


class Panel:
    """As much of a kiosk as ``show_picture`` touches."""

    def __init__(self) -> None:
        self._panel_showing = threading.Event()
        self._page_busy = threading.Event()
        self._cues = Cues()
        self.threads: list[str] = []

    # The two things the real method starts. Recorded rather than run: one would open a browser
    # and the other would shell out to ffmpeg.
    def _restill(self) -> None:
        self.threads.append("restill")

    def _picture_session(self) -> None:
        self.threads.append("session")

    show_picture = Kiosk.show_picture


class Now:
    """A thread that is simply the call, so what would have been spawned is on the list."""

    def __init__(self, target, name=None, daemon=None) -> None:
        self._target = target

    def start(self) -> None:
        self._target()


@pytest.fixture
def panel(monkeypatch):
    """A stand-in kiosk whose threads run inline, so there is nothing to wait for."""
    monkeypatch.setattr(kiosk.threading, "Thread", Now)
    return Panel()


def test_an_empty_panel_takes_the_picture_the_long_way(panel) -> None:
    assert panel.show_picture() is True
    assert panel.threads == ["session"], "a page has to be uncovered before anything is visible"
    assert panel._page_busy.is_set()


def test_a_picture_already_up_is_swapped_rather_than_refused(panel) -> None:
    """The whole point. The page polls, and the caller has already written the new payload."""
    panel._page_busy.set()
    panel._panel_showing.set()

    answer = panel.show_picture()
    assert answer is True, "saying False would deny a picture that is about to be on the glass"
    assert panel.threads == ["restill"], "no second session; the first one still owns the teardown"


def test_a_photograph_going_up_makes_no_sound(panel, monkeypatch) -> None:
    """The shutter already sounded a beat ago. A second cue on top of it is one too many."""
    monkeypatch.setattr(kiosk.panel, "announces", lambda: False)
    panel._page_busy.set()
    panel._panel_showing.set()

    assert panel.show_picture() is True
    assert panel._cues.played == [], "the picture is its own announcement"


def test_a_drawing_replacing_one_still_says_so(panel, monkeypatch) -> None:
    """Nothing else announces a drawing: it took half a minute and arrived silently."""
    monkeypatch.setattr(kiosk.panel, "announces", lambda: True)
    panel._page_busy.set()
    panel._panel_showing.set()

    assert panel.show_picture() is True
    assert panel._cues.played == ["shown"]


def test_the_admin_page_still_has_no_room(panel) -> None:
    """Busy but not showing a picture of ours: that is the dashboard, and it is not ours to
    paint over. It is also the window where a picture has been offered and the page has not
    painted it yet - answering True there would claim a panel that is still covered."""
    panel._page_busy.set()

    assert panel.show_picture() is False
    assert panel.threads == [], "nothing is started for a panel that cannot take it"
    assert panel._cues.played == [], "and nothing is announced"
