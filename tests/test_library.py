"""What the web page is told is on the card.

Imports ``cyclops.library`` and ``cyclops.card`` only - between them they have no dependency
beyond the standard library, which is the whole point of the module under test: the admin service
lists a directory without loading OpenCV to do it.
"""

from __future__ import annotations

import json

import pytest

from cyclops import card, library

# ------------------------------------------------------------------ building a card to read


def make(root, name, *, log=None, summary=None, page="done", video=None, receipt=None,
         photos=()):
    """One session folder, in whatever state the test needs it."""
    folder = root / name
    folder.mkdir(parents=True)
    if log is not None:
        (folder / card.LOG_NAME).write_text(log, encoding="utf-8")
    if summary is not None:
        (folder / card.SUMMARY_NAME).write_text(summary, encoding="utf-8")
    if page is not None:
        (folder / card.PAGE_NAME).write_text(page, encoding="utf-8")
    if video is not None:
        (folder / card.VIDEO).write_bytes(video)
    if receipt is not None:
        (folder / card.RECEIPT_NAME).write_text(receipt, encoding="utf-8")
    for shot in photos:
        (folder / card.PHOTOS).mkdir(exist_ok=True)
        (folder / card.PHOTOS / shot).write_bytes(b"\xff\xd8jpeg")
    return folder


LOG = (
    '{"t": 0.0, "type": "session", "uuid": "u", "entrypoint": "kiosk", "model": "m"}\n'
    '{"t": 15.0, "type": "you", "text": "how tight should this bolt be?"}\n'
    '{"t": 18.0, "type": "cyclops", "text": "about 25 newton metres."}\n'
    '{"t": 30.0, "type": "photo", "by": "you", "file": "photos/14-32-30_you.jpg"}\n'
    '{"t": 84.4, "type": "video", "file": "video.mp4", "seconds": 84.4}\n'
    '{"t": 84.4, "type": "end", "reason": "stopped", "seconds": 84.4, "slug": "bolt",'
    ' "photos": 1}\n'
)

SUMMARY = "# How tight the bolt goes\n\nTwenty-five newton metres, and why.\n"


@pytest.fixture(autouse=True)
def _forget():
    """The entry cache is keyed on mtime, and tmp_path reuses none - but say so out loud."""
    library._cache.clear()
    yield
    library._cache.clear()


# ------------------------------------------------------------------ one session, read back


def test_entry_reads_the_folder(tmp_path):
    make(tmp_path, "2026-08-26_14-32-05_bolt", log=LOG, summary=SUMMARY, video=b"mp4-ish",
         photos=["14-32-30_you.jpg"])
    found = library.entry(tmp_path, "2026-08-26_14-32-05_bolt")
    assert found.title == "How tight the bolt goes"
    assert found.summary == "Twenty-five newton metres, and why."
    assert found.started == "2026-08-26T14:32:05"
    assert found.seconds == 84.4
    assert found.entrypoint == "kiosk"
    assert found.video is True
    assert found.video_bytes == len(b"mp4-ish")
    assert found.photos == 1
    assert found.verdict == "finished"


def test_a_session_with_no_summary_is_named_by_its_slug(tmp_path):
    make(tmp_path, "2026-08-26_14-32-05_lego-falcon", log=LOG)
    assert library.entry(tmp_path, "2026-08-26_14-32-05_lego-falcon").title == "Lego Falcon"


def test_a_session_nobody_named_is_still_titled(tmp_path):
    make(tmp_path, "2026-08-26_14-32-05", log=LOG)
    assert library.entry(tmp_path, "2026-08-26_14-32-05").title == "Session 2026-08-26 14:32"


def test_a_session_with_no_end_record_falls_back_to_its_last_moment(tmp_path):
    """Interrupted, or running right now. Either way the last thing that happened is the length."""
    make(tmp_path, "2026-08-26_14-32-05", page=None,
         log='{"t": 0.0, "type": "session"}\n{"t": 41.5, "type": "you", "text": "hi"}\n')
    found = library.entry(tmp_path, "2026-08-26_14-32-05")
    assert found.seconds == 41.5
    assert found.verdict == "unfinished"


def test_a_zero_byte_page_is_not_a_finished_session(tmp_path):
    """card.written, not is_file - the bug the whole card module exists for."""
    folder = make(tmp_path, "2026-08-26_14-32-05", log=LOG)
    (folder / card.PAGE_NAME).write_text("")
    assert library.entry(tmp_path, "2026-08-26_14-32-05").verdict == "unfinished"


def test_the_project_it_was_filed_under_comes_back(tmp_path):
    make(tmp_path, "2026-08-26_14-32-05", log=LOG,
         receipt="---\nproject: Cyclops\nkey: cyclops\n---\n\n# Filed under Cyclops\n")
    assert library.entry(tmp_path, "2026-08-26_14-32-05").filed == "Cyclops"


def test_an_unfiled_session_names_no_project(tmp_path):
    make(tmp_path, "2026-08-26_14-32-05", log=LOG)
    assert library.entry(tmp_path, "2026-08-26_14-32-05").filed == ""


# ------------------------------------------------------------------ the whole card


def test_entries_come_back_newest_first(tmp_path):
    for name in ("2026-08-24_09-00-00", "2026-08-26_14-32-05", "2026-08-25_20-12-34"):
        make(tmp_path, name, log=LOG)
    assert [e.name for e in library.entries(tmp_path)] == [
        "2026-08-26_14-32-05", "2026-08-25_20-12-34", "2026-08-24_09-00-00"]


def test_a_missing_sessions_directory_is_no_sessions_and_not_a_crash(tmp_path):
    assert library.entries(tmp_path / "nothing-here") == []


def test_a_session_photos_folder_is_not_itself_a_session(tmp_path):
    make(tmp_path, "2026-08-26_14-32-05", log=LOG, photos=["14-32-30_you.jpg"])
    assert [e.name for e in library.entries(tmp_path)] == ["2026-08-26_14-32-05"]


# ------------------------------------------------------------------ nothing outside the tree


@pytest.mark.parametrize("name", [
    "..", "../..", "../etc", "/etc", "2026-08-26_14-32-05/photos", "", ".",
    "2026-08-26_14-32-05/../../etc", "nope",
])
def test_resolve_refuses_anything_that_is_not_a_session_folder(tmp_path, name):
    make(tmp_path, "2026-08-26_14-32-05", log=LOG, photos=["14-32-30_you.jpg"])
    assert library.resolve(tmp_path, name) is None


def test_resolve_finds_a_real_session(tmp_path):
    make(tmp_path, "2026-08-26_14-32-05", log=LOG)
    assert library.resolve(tmp_path, "2026-08-26_14-32-05").name == "2026-08-26_14-32-05"


def test_a_symlink_out_of_the_tree_is_not_a_session(tmp_path):
    make(tmp_path, "2026-08-26_14-32-05", log=LOG)
    outside = tmp_path.parent / "elsewhere"
    outside.mkdir(exist_ok=True)
    (tmp_path / "escape").symlink_to(outside, target_is_directory=True)
    assert library.resolve(tmp_path, "escape") is None


# ------------------------------------------------------------------ the transcript


def test_records_are_the_conversation_and_not_the_bookkeeping(tmp_path):
    make(tmp_path, "2026-08-26_14-32-05", log=LOG)
    kinds = [r["type"] for r in library.records(tmp_path, "2026-08-26_14-32-05")]
    assert kinds == ["you", "cyclops", "photo"]  # no session, no video, no end


def test_a_photo_record_carries_the_url_that_serves_it(tmp_path):
    make(tmp_path, "2026-08-26_14-32-05", log=LOG, photos=["14-32-30_you.jpg"])
    shot = [r for r in library.records(tmp_path, "2026-08-26_14-32-05") if r["type"] == "photo"][0]
    assert shot["url"] == "/media/2026-08-26_14-32-05/photos/14-32-30_you.jpg"


def test_records_of_a_session_that_is_not_there(tmp_path):
    assert library.records(tmp_path, "nope") == []


# ------------------------------------------------------------------ the picture stream


def test_the_stream_runs_across_every_session_newest_first(tmp_path):
    make(tmp_path, "2026-08-25_09-00-00", log=LOG, photos=["09-01-00_you.jpg"])
    make(tmp_path, "2026-08-26_14-32-05", log=LOG,
         photos=["14-33-00_you.jpg", "14-32-30_you.jpg"])
    got = library.stream(tmp_path)
    assert [i.when for i in got] == [
        "2026-08-26T14:33:00", "2026-08-26T14:32:30", "2026-08-25T09:01:00"]


def test_a_picture_taken_after_midnight_belongs_to_the_next_day(tmp_path):
    make(tmp_path, "2026-08-26_23-50-00", log=LOG, photos=["00-05-00_you.jpg"])
    assert library.stream(tmp_path)[0].when == "2026-08-27T00:05:00"


def test_the_stream_stops_at_its_limit(tmp_path):
    make(tmp_path, "2026-08-26_14-32-05", log=LOG,
         photos=[f"14-33-0{i}_you.jpg" for i in range(5)])
    assert len(library.stream(tmp_path, limit=3)) == 3


def test_an_empty_card_streams_nothing(tmp_path):
    assert library.stream(tmp_path) == []


# ------------------------------------------------------------------ the cache


def test_a_folder_that_changed_is_read_again(tmp_path):
    folder = make(tmp_path, "2026-08-26_14-32-05", log=LOG)
    assert library.entry(tmp_path, "2026-08-26_14-32-05").photos == 0
    (folder / card.PHOTOS).mkdir()
    (folder / card.PHOTOS / "14-32-30_you.jpg").write_bytes(b"\xff\xd8jpeg")
    assert library.entry(tmp_path, "2026-08-26_14-32-05").photos == 1


def test_a_live_session_is_never_cached(tmp_path):
    """Its mtime is about to move again; caching it would be caching a moving thing."""
    folder = make(tmp_path, "2026-08-26_14-32-05", log=LOG, page=None)
    handle = (folder / card.LOG_NAME).open("a")
    try:
        assert card.claim(handle)
        assert library.entry(tmp_path, "2026-08-26_14-32-05").verdict == "live"
        assert folder not in library._cache
    finally:
        handle.close()
