"""Deleting a session from the panel: everything in it goes, or nothing does.

The gesture lives in ``tests/render_check.mjs``; this is the half behind it - the route, its
gates, and :func:`cyclops.session.erase`.
"""

from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "cyclops.admin.settings")

import django  # noqa: E402

django.setup()

from django.test import RequestFactory  # noqa: E402

from cyclops import card  # noqa: E402
from cyclops.admin import views  # noqa: E402
from cyclops.config import Settings  # noqa: E402

PANEL = "127.0.0.1"
LAPTOP = "192.168.1.44"
NAME = "2026-09-20_10-00-00_fork-seal"


@pytest.fixture
def folder(tmp_path, monkeypatch):
    """One session holding one of everything a session can hold."""
    root = tmp_path / "sessions"
    one = root / NAME
    for sub in (card.PHOTOS, card.CLIPS, card.VIDEOS, card.PARTS):
        (one / sub).mkdir(parents=True)
    (one / card.LOG_NAME).write_text(json.dumps({"t": 0, "type": "you", "text": "hi"}) + "\n")
    for name in (card.PAGE_NAME, card.SUMMARY_NAME, card.RECEIPT_NAME):
        (one / name).write_text("x")
    (one / card.VIDEO).write_bytes(b"v")
    (one / card.PHOTOS / "14-00-00_you.jpg").write_bytes(b"j")
    (one / card.PHOTOS / card.CAPTIONS_NAME).write_text("{}")
    (one / card.PHOTOS / ".14-00-01_you.jpg.tmp").write_bytes(b"j")
    (one / card.CLIPS / card.CLIP_PLAN).write_text("{}")
    (one / card.CLIPS / "1.mp4").write_bytes(b"c")
    (one / card.VIDEOS / card.VIDEOS_NAME).write_text("[]")
    (one / card.VIDEOS / "abc.jpg").write_bytes(b"t")
    (one / card.tmp_for(one / card.VIDEO).name).write_bytes(b"v")
    monkeypatch.setattr(views, "_settings", lambda: Settings(api_key="", sessions_dir=root))
    return one


def delete(where: str = PANEL):
    request = RequestFactory().post(f"/api/session/{NAME}/delete", REMOTE_ADDR=where)
    return views.delete_session(request, NAME)


def test_delete_takes_the_folder_and_everything_in_it(folder) -> None:
    answer = delete()
    assert answer.status_code == 200
    assert not folder.exists()
    assert json.loads(answer.content)["sessions"] == []


def test_a_file_we_did_not_write_keeps_the_session_and_says_why(folder) -> None:
    (folder / "notes.txt").write_text("mine")
    answer = delete()
    assert answer.status_code == 400
    assert "notes.txt" in answer.content.decode()
    assert (folder / card.VIDEO).exists()


def test_a_session_still_recording_is_refused(folder) -> None:
    with (folder / card.LOG_NAME).open("a") as handle:
        assert card.claim(handle)
        assert delete().status_code == 400
    assert folder.exists()


def test_only_the_panel_may_delete(folder) -> None:
    assert delete(LAPTOP).status_code == 403
    assert folder.exists()
    assert delete(PANEL).status_code == 200


def test_a_session_the_clipper_is_working_on_is_refused(folder, tmp_path, monkeypatch) -> None:
    import fcntl

    from cyclops import cut

    monkeypatch.setattr(cut, "CUT_LOCK", tmp_path / "cut.lock")
    (folder / card.CLIPS / card.CLIP_PLAN).unlink()  # no plan yet: the clipper's next unit
    with cut.CUT_LOCK.open("w") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert delete().status_code == 400
    assert (folder / card.VIDEO).exists()


def test_a_delete_that_cannot_finish_leaves_the_folder_whole(folder, monkeypatch) -> None:
    from pathlib import Path

    real = Path.rmdir

    def late_render(self):
        if self.name == card.CLIPS:
            (self / "1.mp4").write_bytes(b"c")  # ffmpeg landing a clip after it was emptied
        real(self)

    monkeypatch.setattr(Path, "rmdir", late_render)
    assert delete().status_code == 400
    for kept in (card.LOG_NAME, card.PAGE_NAME, card.VIDEO, f"{card.PHOTOS}/14-00-00_you.jpg"):
        assert (folder / kept).exists()
