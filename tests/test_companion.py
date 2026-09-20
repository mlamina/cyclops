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
* **So do the room and his voice.** The two streams off the kiosk's own port
  (:mod:`cyclops.companion`), which must arrive as they happen and cost nothing when nobody is
  looking. The pure half of that is the third; the socket, the camera and the phone are on the
  box itself.
"""

from __future__ import annotations

import json
import os
import pathlib
import re

import pytest

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "cyclops.admin.settings")

import django  # noqa: E402

django.setup()

from django.test import RequestFactory  # noqa: E402

from cyclops import card, companion, library, recall  # noqa: E402
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


# ------------------------------------------------------------------ the files the page asks for


def test_every_file_the_page_asks_for_is_one_the_server_will_hand_over() -> None:
    """The failure this catches is silent, and the idle screen is why it now matters.

    ``views.static_file`` serves from a flat allow-list, so a file that is not in ``OURS`` is a
    404 - no error anywhere, just a stylesheet that never arrives or, on the screen this feature
    added, a mark that never paints. The panel has no address bar and no console, and the kiosk
    caches ``immutable``, so nobody finds out until somebody walks over to the bench.

    Both directions: nothing is referenced that is not served, and nothing is served that is not
    on disk (``_asset_version`` hashes the list at import, and a missing file quietly hashes as
    nothing).
    """
    static = pathlib.Path(views.STATIC_DIR)
    missing = [name for name in views.OURS if not (static / name).is_file()]
    assert not missing, f"in OURS but not on disk: {missing}"

    page = (
        pathlib.Path(views.__file__).parent / "templates/cyclops/dashboard.html"
    ).read_text()
    css = "\n".join((static / name).read_text() for name in views.OURS if name.endswith(".css"))
    # A name with an extension, so the prose in these files - which says "admin/static/." and
    # "static/*.css" while explaining itself - is not read as a request for a file.
    asked = set(re.findall(r"/static/([A-Za-z0-9_-]+\.[A-Za-z0-9]+)", page + css))
    assert asked, "nothing matched - the reference shape changed and this test stopped looking"
    assert not (asked - set(views.OURS)), f"asked for but never served: {asked - set(views.OURS)}"


# ---------------------------------------------------------------- the room, and the voice
#
# The two streams the same phone pulls off the kiosk's own port (:mod:`cyclops.companion`), which
# is a fourth sentence that can be wrong quietly: *what he sees and what he says arrive as they
# happen, and cost nothing when nobody is looking.*
#
# All of it is the pure half. There is no socket here and no Speaker: the tap takes bytes and
# hands back bytes, the encoder takes a frame and hands back a decision. The half that needs a
# camera, a port and a phone is in the plan's verification list and on the box itself.


def test_a_stream_nobody_is_watching_costs_a_load_and_a_test() -> None:
    """The gate, and the whole argument for the port living inside the kiosk.

    There is no note to read and no flag to leave: the TCP connection *is* the subscription, so
    with the live screen shut the audio thread's tap returns having done nothing at all and the
    encoder thread does not exist to be woken.
    """
    tap = companion._Voice()
    tap.on_block(b"\x01\x02\x03\x04")

    assert tap.listeners == 0
    assert companion._Preview()._thread is None, "an encoder for nobody"


def test_a_listener_is_handed_exactly_what_the_speaker_played() -> None:
    tap = companion._Voice()
    sink = tap.join()

    tap.on_block(b"ab")
    tap.on_block(b"cd")

    assert tap.drain(sink) == b"abcd"


def test_silence_is_sized_by_the_clock_and_not_by_the_loop() -> None:
    """A flat 48 KB/s whether or not a session is running - the page builds its audio graph once
    instead of tearing it down every time he stops - and it has to be *exactly* flat.

    A block sized to the sleep was the first version: an iteration is a sleep plus a write, so it
    always takes a shade longer than it asks for and always delivered a shade less than real
    time. The listener's buffer drained at the difference and the sound broke up.
    """
    a_second = companion._silence(1.0)

    assert len(a_second) == companion.SAMPLE_RATE * companion.BYTES_PER_FRAME
    assert set(a_second) == {0}
    assert len(companion._silence(0.05)) == 1200 * companion.BYTES_PER_FRAME
    assert companion._silence(-1.0) == b"", "a clock that went backwards is not negative audio"


def test_nothing_said_is_nothing_drained() -> None:
    tap = companion._Voice()
    assert tap.drain(tap.join()) == b""


def test_a_listener_that_falls_behind_is_skipped_on_rather_than_grown() -> None:
    """A phone on bad wifi must cost a second of audio, not a megabyte of it - and what it keeps
    is the newest, because in a conversation happening in the room old audio is the wrong audio."""
    tap = companion._Voice()
    sink = tap.join()

    for mark in range(400):  # eight seconds of 20 ms blocks into a one-second cap
        tap.on_block(bytes([mark % 251, 0]) * 480)

    assert len(sink) == companion.VOICE_LAG_MAX, "the buffer grew with the backlog"
    assert bytes(sink[-2:]) == bytes([399 % 251, 0]), "it kept the stale end and dropped the live"


def test_one_listener_leaving_does_not_take_the_others_stream() -> None:
    tap = companion._Voice()
    stays, goes = tap.join(), tap.join()

    tap.leave(goes)
    tap.on_block(b"ab")

    assert tap.listeners == 1
    assert tap.drain(stays) == b"ab"


def test_a_camera_that_has_stopped_is_a_picture_that_says_so() -> None:
    """An <img> holds the last part it was sent for ever, so a stream that simply stopped would
    leave a picture of a room nobody is watching any more. The words and the rule are the panel's
    own (kiosk.py) - a device that is enumerated but silent is a different fault from one that is
    not there, and a phone deserves the distinction the person at the glass gets."""
    assert companion._reason(None, 100.0, connected=False) == companion.NO_CAMERA
    assert companion._reason(None, 100.0, connected=True) == companion.CAMERA_STALLED
    stale = 100.0 - companion.STALE_AFTER_S - 0.1
    assert companion._reason((object(), stale), 100.0, connected=True) == companion.CAMERA_STALLED
    assert companion._reason((object(), 99.9), 100.0, connected=True) == ""


def test_a_frame_is_published_once_however_many_are_watching() -> None:
    """One producer, one slot, a serial. The regression this guards is the obvious one: a viewer
    list that gets encoded into, which would put the cost of the picture on the number of phones."""
    preview = companion._Preview()
    preview._publish(b"jpeg")

    first = preview.latest(-1, 0.0)
    second = preview.latest(-1, 0.0)

    assert first == second == (b"jpeg", 1)
    assert preview.latest(1, 0.0) is None, "a serial already seen came back as news"


def test_somebody_watching_counts_as_company() -> None:
    """The one line this adds to Kiosk._sleeping. Sleeping releases the camera, which is the one
    thing a companion is here for, and there is nobody at the glass to tap it back."""
    assert not companion.watching()

    sink = companion.voice.join()
    try:
        assert companion.watching()
    finally:
        companion.voice.leave(sink)

    assert not companion.watching()


def test_an_open_socket_is_not_somebody_listening(monkeypatch) -> None:
    """The bug this was written for, and it silenced the whole box.

    A curl left running on another machine held /voice.pcm open, the panel handed its voice to
    it, and there was no sound anywhere: not on the panel, which had given the voice away, and
    not on the phone, which was not the thing holding the socket. Nothing on screen said so and
    nothing timed out - a peer that has gone holds a socket for as long as TCP takes to notice.

    So the claim is a heartbeat that has to be renewed, and the panel keeps his voice by default.
    """
    monkeypatch.setattr(companion, "_heard_at", 0.0)
    sink = companion.voice.join()
    try:
        assert not companion.listening(), "an open socket took the voice off the panel"
    finally:
        companion.voice.leave(sink)


def test_a_claim_lapses_rather_than_having_to_be_withdrawn(monkeypatch) -> None:
    """A locked phone sends no "stop". It simply stops asking, and that has to be enough."""
    now = 1000.0
    monkeypatch.setattr(companion.time, "monotonic", lambda: now)
    monkeypatch.setattr(companion, "_heard_at", now)
    assert companion.listening()

    now += companion.LISTEN_FRESH_S + 0.1
    assert not companion.listening(), "his voice stayed on a device that stopped asking for it"


def test_the_page_renews_before_the_kiosk_gives_up() -> None:
    """Two beats may be missed. One interval that crept past the other would make the panel snatch
    his voice back mid-sentence, on a phone that is doing nothing wrong."""
    assert companion.LISTEN_BEAT_S * 2 < companion.LISTEN_FRESH_S


# ---------------------------------------------------------------- which of the two it shows
#
# The picture is the glass now, not the sensor. These are the three claims that makes, and the
# third is the one that inverts a rule: a captured frame is never a fault for being old.


class Source:
    """A frame source, as either the screen or the camera: ``connected`` and ``latest()``."""

    def __init__(self, *, connected: bool, stamp: float | None = None) -> None:
        self.connected = connected
        self.frame = object()
        self._stamp = stamp

    def latest(self):
        return None if self._stamp is None else (self.frame, self._stamp)


def test_the_picture_is_the_panel_while_the_panel_can_be_captured() -> None:
    screen = Source(connected=True, stamp=100.0)
    camera = Source(connected=True, stamp=100.0)

    got, why = companion._sampled(screen, camera, 100.0)

    assert got[0] is screen.frame and not why


def test_a_panel_that_cannot_be_captured_falls_back_to_the_camera() -> None:
    """The session's own rule, so a phone and the recording never disagree about the source."""
    screen = Source(connected=False)
    camera = Source(connected=True, stamp=100.0)

    got, why = companion._sampled(screen, camera, 100.0)

    assert got[0] is camera.frame and not why


def test_a_panel_that_has_not_changed_is_not_a_fault() -> None:
    """Asleep, nothing on the panel moves at all - so every frame is old and all of them true.

    The camera's rule read onto the screen would put "stopped responding" over a screen that is
    working perfectly, all night.
    """
    old = 100.0 - companion.STALE_AFTER_S - 10
    screen = Source(connected=True, stamp=old)

    got, why = companion._sampled(screen, Source(connected=False), 100.0)

    assert got[0] is screen.frame and not why, "a still panel was reported as a fault"
