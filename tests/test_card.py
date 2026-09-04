"""What a session folder amounts to, and that a write cannot land half-done.

Imports only ``cyclops.card`` on purpose - that module is dependency-free so this suite runs
in a second on any box, with no camera, no key and no OpenCV.
"""

from __future__ import annotations

import os

import pytest

from cyclops import card

# ------------------------------------------------------------------ building folders to judge


def make(root, name, *, log=None, page=None, summary=None, receipt=None, video=None, photos=0,
         parts=False, extra=None):
    """One session folder in whatever state of repair the test needs."""
    folder = root / name
    folder.mkdir(parents=True)
    if log is not None:
        (folder / card.LOG_NAME).write_text(log, encoding="utf-8")
    if page is not None:
        (folder / card.PAGE_NAME).write_text(page, encoding="utf-8")
    if summary is not None:
        (folder / card.SUMMARY_NAME).write_text(summary, encoding="utf-8")
    if receipt is not None:
        (folder / card.RECEIPT_NAME).write_text(receipt, encoding="utf-8")
    if video is not None:
        (folder / card.VIDEO).write_bytes(video)
    if photos:
        (folder / card.PHOTOS).mkdir()
        for i in range(photos):
            (folder / card.PHOTOS / f"14-32-0{i}_you.jpg").write_bytes(b"\xff\xd8jpeg")
    if parts:
        (folder / card.PARTS).mkdir()
        (folder / card.PARTS / "video-raw.mp4").write_bytes(b"raw")
    if extra:
        (folder / extra).write_text("a note someone left", encoding="utf-8")
    return folder


ONE_RECORD = '{"t": 0.0, "type": "session", "uuid": "u"}\n'


# ------------------------------------------------------------------ written()


def test_written_distinguishes_a_husk_from_a_file(tmp_path):
    empty, real, missing = tmp_path / "e", tmp_path / "r", tmp_path / "m"
    empty.write_text("")
    real.write_text("x")
    assert card.written(real)
    assert not card.written(empty), "a zero-byte file is the whole bug; it must not read as there"
    assert not card.written(missing)


def test_written_is_false_for_a_directory(tmp_path):
    (tmp_path / "d").mkdir()
    assert not card.written(tmp_path / "d")


# ------------------------------------------------------------------ triage()


def test_the_incident_reads_as_unfinished(tmp_path):
    """2026-08-28_18-34-03: a good log and video, a zero-byte page and summary, unnamed.

    The regression that matters. Under the old `.is_file()` rule this folder read as *finished*,
    which is why --fix and --name both skipped it and it stayed broken.
    """
    folder = make(tmp_path, "2026-08-28_18-34-03", log=ONE_RECORD, page="", summary="",
                  receipt="---\n", video=b"mp4 bytes")
    state = card.triage(folder)
    assert state.verdict == "unfinished"
    assert not state.page and not state.summary
    assert state.records == 1
    assert state.video
    assert not state.named
    assert state.salvage


def test_the_husk_reads_as_empty(tmp_path):
    """2026-08-28_16-07-12: a zero-byte log and nothing else. The only shape we ever delete."""
    folder = make(tmp_path, "2026-08-28_16-07-12", log="")
    state = card.triage(folder)
    assert state.verdict == "empty"
    assert not state.salvage
    assert state.records == 0


def test_a_complete_session_reads_as_finished(tmp_path):
    folder = make(tmp_path, "2026-08-26_16-48-20_pelican", log=ONE_RECORD, page="# p",
                  summary="# s", receipt="---\n", video=b"mp4", photos=2)
    state = card.triage(folder)
    assert state.verdict == "finished"
    assert state.named and state.filed and state.summary
    assert state.photos == 2


def test_leftover_parts_is_unfinished_even_with_a_page_and_a_video(tmp_path):
    """parts/ surviving means the mux never returned zero - the video may be truncated."""
    folder = make(tmp_path, "2026-08-26_14-53-06", log=ONE_RECORD, page="# p", video=b"trunc",
                  parts=True)
    assert card.triage(folder).verdict == "unfinished"


def test_an_unnamed_but_complete_session_is_finished_and_unnamed(tmp_path):
    folder = make(tmp_path, "2026-08-26_17-13-09", log=ONE_RECORD, page="# p", summary="# s")
    state = card.triage(folder)
    assert state.verdict == "finished"
    assert not state.named


def test_a_torn_last_line_is_counted_not_swallowed(tmp_path):
    folder = make(tmp_path, "2026-08-26_10-00-00", log=ONE_RECORD + '{"t": 1.0, "type": "en')
    state = card.triage(folder)
    assert state.records == 1
    assert state.dropped == 1
    assert state.verdict == "unfinished"


def test_photos_alone_are_salvage(tmp_path):
    """No log at all, but two photos. Never deleted: nothing else can reconstruct them."""
    folder = make(tmp_path, "2026-08-26_11-00-00", photos=2)
    state = card.triage(folder)
    assert state.verdict == "unfinished"
    assert state.salvage


# ------------------------------------------------------------------ what may be deleted


@pytest.mark.parametrize(
    "kwargs",
    [
        {"log": ONE_RECORD},
        {"photos": 1},
        {"video": b"mp4"},
        {"parts": True},
        {"page": "# p"},
        {"summary": "# s"},
    ],
)
def test_anything_with_salvage_is_never_empty(tmp_path, kwargs):
    folder = make(tmp_path, "2026-08-26_12-00-00", **kwargs)
    assert card.triage(folder).verdict != "empty"


def test_a_husk_holding_a_strangers_file_is_not_removable(tmp_path):
    """Triage says empty; the folder still keeps its note, and therefore keeps itself."""
    folder = make(tmp_path, "2026-08-26_13-00-00", log="", extra="notes.txt")
    assert card.triage(folder).verdict == "empty"
    assert card.surprises(folder) == ["notes.txt"]


def test_a_plain_husk_has_no_surprises(tmp_path):
    folder = make(tmp_path, "2026-08-26_13-00-00", log="")
    assert card.surprises(folder) == []


# ------------------------------------------------------------------ writing


def test_write_text_round_trips_and_leaves_no_scratch(tmp_path):
    target = tmp_path / card.PAGE_NAME
    card.write_text(target, "# a page\n")
    assert target.read_text() == "# a page\n"
    assert card.strays(tmp_path) == []


def test_write_text_replaces_existing_content(tmp_path):
    target = tmp_path / card.PAGE_NAME
    card.write_text(target, "old")
    card.write_text(target, "new and longer")
    assert target.read_text() == "new and longer"


def test_a_failed_write_leaves_the_old_file_untouched(tmp_path, monkeypatch):
    """The claim this whole module makes: whole or not at all, never a half of either."""
    target = tmp_path / card.PAGE_NAME
    card.write_text(target, "the good page")

    def boom(*args, **kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(card.os, "replace", boom)
    with pytest.raises(OSError, match="no space"):
        card.write_text(target, "the page that never landed")

    assert target.read_text() == "the good page", "a failed write must not damage what was there"
    assert card.strays(tmp_path) == [], "and it must not leave its scratch file behind"


def test_write_bytes_keeps_a_private_mode(tmp_path):
    target = tmp_path / "14-32-40_you.jpg"
    card.write_bytes(target, b"\xff\xd8jpeg", mode=0o600)
    assert target.read_bytes() == b"\xff\xd8jpeg"
    assert oct(target.stat().st_mode)[-3:] == "600"


def test_write_makes_the_parent_directory(tmp_path):
    target = tmp_path / "2026-08-26_14-00-00" / card.PAGE_NAME
    card.write_text(target, "# p")
    assert target.is_file()


def test_strays_finds_what_a_kill_left_behind(tmp_path):
    """write_bytes cleans up after an exception. It cannot clean up after SIGKILL."""
    (tmp_path / f".{card.PAGE_NAME}.tmp").write_text("half a page")
    assert card.strays(tmp_path) == [tmp_path / f".{card.PAGE_NAME}.tmp"]


def test_the_scratch_name_is_hidden_from_every_scan(tmp_path):
    """Directory scans in this codebase filter is_dir() or glob *.jpg. A dotfile is invisible."""
    assert card.tmp_for(tmp_path / card.PAGE_NAME).name.startswith(".")
    assert not card.tmp_for(tmp_path / "a.jpg").name.endswith(".jpg")


# ------------------------------------------------------------------ the live-session lock


def test_a_folder_nobody_holds_is_not_locked(tmp_path):
    folder = make(tmp_path, "2026-08-26_15-00-00", log=ONE_RECORD)
    assert not card.locked(folder)


def test_a_held_folder_reads_as_live(tmp_path):
    folder = make(tmp_path, "2026-08-26_15-00-00", log=ONE_RECORD)
    handle = (folder / card.LOG_NAME).open("a")
    try:
        assert card.claim(handle)
        assert card.locked(folder)
        assert card.triage(folder).verdict == "live", "a live session is never judged or touched"
    finally:
        handle.close()
    assert not card.locked(folder), "closing the handle drops the lock, as a crash would"


def test_a_missing_log_is_not_locked(tmp_path):
    folder = make(tmp_path, "2026-08-26_15-00-00")
    assert not card.locked(folder)


def test_read_log_on_a_zero_byte_file(tmp_path):
    (tmp_path / card.LOG_NAME).write_text("")
    assert card.read_log(tmp_path / card.LOG_NAME) == ([], 0)


def test_sync_dir_survives_a_path_it_cannot_open(tmp_path):
    card.sync_dir(tmp_path / "not-there")  # best-effort, must never raise


def test_write_bytes_lands_the_data_durably(tmp_path):
    """Not a crash test - just proof the fd is fsynced rather than merely closed."""
    calls = []
    real = os.fsync
    card.os.fsync = lambda fd: (calls.append(fd), real(fd))[1]
    try:
        card.write_text(tmp_path / card.PAGE_NAME, "# p")
    finally:
        card.os.fsync = real
    assert len(calls) >= 2, "the file and its directory both need syncing"


# ------------------------------------------------------------------ bytes that arrive in pieces


def test_write_stream_lands_a_file_whole(tmp_path):
    target = tmp_path / "brake.jpg"
    data = b"\xff\xd8jpeg" * 300
    pieces = (data[i : i + 7] for i in range(0, len(data), 7))
    landed = card.write_stream(target, pieces, limit=99999)
    assert landed == len(data)
    assert target.read_bytes() == data
    assert card.strays(tmp_path) == []


def test_write_stream_replaces_a_file_whole(tmp_path):
    target = tmp_path / "spec.txt"
    card.write_stream(target, iter([b"first draft"]), limit=999)
    card.write_stream(target, iter([b"the ", b"corrected ", b"one"]), limit=999)
    assert target.read_text(encoding="utf-8") == "the corrected one"
    assert card.strays(tmp_path) == []


def test_write_stream_leaves_nothing_behind_when_the_stream_gives_up(tmp_path):
    """The promise the whole scratch-then-rename dance is for, on the path that arrives in pieces.

    An upload that dies halfway across the LAN must leave the folder exactly as it found it: no
    truncated file on the target, and no scratch name for ``strays`` to find and for the browser
    to have to hide.
    """

    def dies():
        yield b"the first half"
        raise OSError("the laptop went away")

    target = tmp_path / "half.bin"
    with pytest.raises(OSError):
        card.write_stream(target, dies(), limit=9999)
    assert not target.exists()
    assert card.strays(tmp_path) == []


def test_write_stream_refuses_rather_than_truncates_at_the_limit(tmp_path):
    """A file cut off at the ceiling would land whole, look whole, and be broken in what opened it.

    So the ceiling raises. What makes this worth pinning is the second assertion: the refusal has
    to leave the target absent, not present-and-short, or the cap becomes a way to plant a corrupt
    file rather than a way to refuse a large one.
    """
    target = tmp_path / "huge.bin"
    with pytest.raises(ValueError):
        card.write_stream(target, (b"x" * 64 for _ in range(20)), limit=100)
    assert not target.exists()
    assert card.strays(tmp_path) == []


def test_write_stream_makes_the_folder_it_is_pointed_at(tmp_path):
    """Same as write_bytes: a project may not have the sub-folder an upload is aimed into yet."""
    target = tmp_path / "Datasheets" / "part.pdf"
    card.write_stream(target, iter([b"%PDF-1.4"]), limit=999)
    assert target.read_bytes() == b"%PDF-1.4"
