"""Companion mode: a second screen, tethered.

A phone or an iPad on the LAN opens the same page the panel does and behaves as an extension of
the box rather than as a web copy of it. Three things make that true, and each of them is a
sentence that can be wrong quietly:

* **The picture is on both screens.** That half is a browser fact - a data URL fetched, decoded
  and painted - and it lives in ``tests/render_check.mjs``, which drives a real Chromium from a
  real LAN address. Nothing here can see it.
* **A press on either screen puts it away on both.** The LAN's half of that is one gate in
  :func:`cyclops.admin.views.close_browser`, and it is the thing in this feature most worth being
  paranoid about: the note it leaves is read by three separate paths in the kiosk, two of which
  move a window. That gate is the first half of this file.
* **The conversation arrives as it is spoken.** Which session is running, and the lines of it we
  have not seen yet. The second half.
"""

from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "cyclops.admin.settings")

import django  # noqa: E402

django.setup()

from django.test import RequestFactory  # noqa: E402

from cyclops import card, library, recall  # noqa: E402
from cyclops.admin import views  # noqa: E402
from cyclops.config import Settings  # noqa: E402

PANEL = "127.0.0.1"
PHONE = "192.168.1.44"


@pytest.fixture
def flags(tmp_path, monkeypatch):
    """The two notes the kiosk and the page leave each other, pointed somewhere throwaway."""
    monkeypatch.setattr(views, "PICTURE_UP_FLAG", tmp_path / "picture-up")
    monkeypatch.setattr(views, "BROWSER_CLOSE_FLAG", tmp_path / "browser-close")
    return tmp_path


def post(where: str) -> int:
    return views.close_browser(RequestFactory().post("/close", REMOTE_ADDR=where)).status_code


# ------------------------------------------------------------------ putting a picture away


def test_a_companion_may_put_away_a_picture_that_is_on_the_glass(flags) -> None:
    (flags / "picture-up").touch()
    assert post(PHONE) == 204
    assert (flags / "browser-close").exists(), "the kiosk was never told, so the panel keeps it"


def test_a_companion_may_not_close_the_panel_s_own_page(flags) -> None:
    """The gate, and the whole reason it is a flag rather than a look at the offer file.

    ``BROWSER_CLOSE_FLAG`` is read by three paths in the kiosk. Two of them move a window:
    ``_watch_page`` ends whatever session put a page up - including the *admin* page somebody
    tapped the eye for - and ``_sync_stranded`` rebuilds the OpenCV window outright. Neither is
    something a phone on the network is entitled to cause, so with nothing on the glass the note
    is never written at all.
    """
    assert not (flags / "picture-up").exists()
    assert post(PHONE) == 403
    assert not (flags / "browser-close").exists()


def test_the_panel_s_own_browser_still_closes_its_own_page(flags) -> None:
    """It is the one client that has a page to close, picture or no picture."""
    assert post(PANEL) == 204
    assert (flags / "browser-close").exists()


# ------------------------------------------------------------------ the conversation, live


def written(folder, records: list[dict]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / card.LOG_NAME).write_text(
        "".join(json.dumps(one) + "\n" for one in records), encoding="utf-8"
    )


@pytest.fixture
def sessions(tmp_path, monkeypatch):
    """A card with sessions on it, and views pointed at it."""
    root = tmp_path / "sessions"
    root.mkdir()
    monkeypatch.setattr(views, "_settings", lambda: Settings(api_key="", sessions_dir=root))
    return root


SAID = [
    {"t": 0.0, "type": "session", "entrypoint": "kiosk"},
    {"t": 2.1, "type": "you", "text": "what torque does the crank bolt take?"},
    {"t": 4.0, "type": "cyclops", "text": "Forty to fifty newton metres."},
    {"t": 6.2, "type": "screen", "html": "<h1>45 Nm</h1>"},
]


def live(**query) -> dict:
    request = RequestFactory().get("/api/live", query, REMOTE_ADDR=PHONE)
    return json.loads(views.live(request).content)


def test_a_folder_nobody_is_writing_to_is_not_live(sessions) -> None:
    written(sessions / "2026-09-05_14-00-00", SAID)
    assert library.live(sessions) is None, "the log is closed; that session is over"
    assert live()["name"] is None


def test_the_session_being_recorded_right_now_is_the_locked_one(sessions) -> None:
    """The same question ``card.triage`` answers with ``verdict == "live"``, asked without the
    log read that goes with it - and asked of the newest few folders rather than only the newest,
    so the ordering of the names is not load-bearing."""
    written(sessions / "2026-09-05_14-00-00", SAID)
    written(sessions / "2026-09-05_15-00-00", SAID)
    with (sessions / "2026-09-05_14-00-00" / card.LOG_NAME).open("a") as handle:
        assert card.claim(handle)
        assert library.live(sessions) == "2026-09-05_14-00-00"


def test_a_companion_is_handed_only_the_lines_it_has_not_seen(sessions) -> None:
    """What makes this pollable. The bookkeeping ``session`` record is not one of them - the page
    shows what was said, and ``library.SPOKEN`` is the one list that decides which is which."""
    folder = sessions / "2026-09-05_14-00-00"
    written(folder, SAID)
    with (folder / card.LOG_NAME).open("a") as handle:
        card.claim(handle)
        whole = live()
        assert whole["name"] == folder.name
        assert whole["n"] == 3, "the session record is bookkeeping, not conversation"
        assert [r["type"] for r in whole["records"]] == ["you", "cyclops", "screen"]

        tail = live(name=folder.name, since=2)
        assert tail["n"] == 3
        assert [r["type"] for r in tail["records"]] == ["screen"]


def test_a_since_from_another_session_is_ignored_rather_than_trusted(sessions) -> None:
    """``since`` counts lines of one particular conversation, so it only means anything alongside
    the name of that conversation. A companion that asks about the wrong one - it just ended, or
    another just started - is handed the whole thing and starts again."""
    folder = sessions / "2026-09-05_14-00-00"
    written(folder, SAID)
    with (folder / card.LOG_NAME).open("a") as handle:
        card.claim(handle)
        assert len(live(name="2026-09-04_09-00-00", since=2)["records"]) == 3
        assert len(live(since=2)["records"]) == 3, "a since with no name is not a since"


def test_a_log_still_being_written_reads_back_clean(sessions) -> None:
    """The line-by-line read this whole feature rests on. ``SessionLog`` flushes each record as it
    lands and fsyncs once at the close, so a poll can arrive mid-write; ``card.read_log`` drops a
    half-written trailing line and everything before it is still a conversation."""
    folder = sessions / "2026-09-05_14-00-00"
    written(folder, SAID)
    with (folder / card.LOG_NAME).open("a") as handle:
        card.claim(handle)
        handle.write('{"t": 9.0, "type": "you", "text": "half a li')
        handle.flush()
        assert live()["n"] == 3, "the torn line is dropped, not counted"

        handle.write('ne"}\n')
        handle.flush()
        assert live()["n"] == 4, "...and counted once it lands whole"


# ------------------------------------------------------------------ a picture he found


def found(path: str, scope: str, kind: str = "photo") -> recall.Item:
    return recall.Item(kind=kind, path=path, scope=scope, title="Crank arm", text="", mtime_ns=0,
                       size=0)


def test_a_recalled_picture_says_where_it_is_and_not_only_what_it_is_called() -> None:
    """The filename alone was never enough. A recalled picture belongs to some *other* session or
    to a project, so the folder it is in is the half that was missing - and without it the line
    that found the picture was the one line in a transcript with no picture under it."""
    item = found("/home/cyclops/cyclops/sessions/2026-09-04_16-17-13_bmw/photos/16-18-02_you.jpg",
                 "session:2026-09-04_16-17-13_bmw")
    assert item.within == "photos/16-18-02_you.jpg"

    deep = found("/home/cyclops/cyclops/projects/BMW R80RT/Photos/tank/rust.png",
                 "project:BMW R80RT", kind="image")
    assert deep.within == "Photos/tank/rust.png", "a project picture can be nested"


def test_a_recalled_picture_is_fetched_from_the_route_its_scope_belongs_to() -> None:
    assert library.recalled_url(
        {"what": "photo", "scope": "session:2026-09-04_16-17-13", "file": "photos/a.jpg"}
    ) == "/media/2026-09-04_16-17-13/photos/a.jpg"
    assert library.recalled_url(
        {"what": "image", "scope": "project:BMW R80RT", "file": "Photos/a.png"}
    ) == "/project-media/BMW R80RT/Photos/a.png"


@pytest.mark.parametrize(
    "record",
    [
        {"what": "photo", "file": "16-18-02_you.jpg"},  # a log from before the scope was written
        {"what": "entry", "scope": "project:BMW", "file": "Log.md"},  # not a picture at all
        {"what": "photo", "scope": "session:x"},  # found nothing
    ],
)
def test_a_recall_with_nothing_to_show_shows_nothing(record) -> None:
    """All three have always rendered as a line on its own, and still do - the transcript never
    gets an <img> pointed at a URL that will 404."""
    assert library.recalled_url(record) == ""


def test_an_old_log_still_reads_back_without_a_url(sessions) -> None:
    """The whole card is full of these, and a missing key must not become a broken thumbnail."""
    folder = sessions / "2026-09-05_14-00-00"
    written(folder, [{"t": 1.0, "type": "recall", "query": "the crank", "title": "Crank arm",
                      "what": "photo", "file": "16-18-02_you.jpg", "shown": True}])
    line = library.records(sessions, folder.name)[0]
    assert "url" not in line
