"""One session, start to finish, with no camera, no key and no model.

Covers the teardown ordering, which is where the crash-safety actually lives: the log is
fsynced before anything is derived from it, the folder's flock is held across the whole of the
rename, and every file that lands has bytes in it.
"""

from __future__ import annotations

import pytest

from cyclops import card, session
from cyclops.config import Settings


class FakeAgent:
    """All SessionLog wants from an agent is somewhere to hang `on_event`."""

    on_event = None


@pytest.fixture
def log(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "_file_the_card", lambda settings: None)
    settings = Settings(api_key="", sessions_dir=tmp_path / "sessions", slug=False, projects=False)
    return session.SessionLog(settings, FakeAgent(), entrypoint="cli")


def test_a_session_leaves_a_finished_folder(log):
    with log:
        log.event("you", text="how deep should this go?")
        log.event("cyclops", text="about 40 mm.")

    state = card.triage(log.dir)
    assert state.verdict == "finished"
    assert state.records == 4  # session, two turns, end
    assert state.page, "session.md must have bytes in it, which is what 'finished' now means"
    assert "how deep should this go?" in (log.dir / card.PAGE_NAME).read_text()


def test_nothing_it_writes_is_ever_zero_bytes(log):
    with log:
        log.event("you", text="hello")

    empty = [p.name for p in log.dir.rglob("*") if p.is_file() and p.stat().st_size == 0]
    assert empty == [], "the whole point: a file this wrote either has content or is not there"


def test_no_scratch_files_survive_a_clean_session(log):
    with log:
        log.event("you", text="hello")
    assert card.strays(log.dir) == []


def test_the_folder_is_locked_for_the_whole_session_and_released_after(log):
    with log:
        assert card.locked(log.dir), "a live session must be visible to a recovery sweep"
        assert card.triage(log.dir).verdict == "live"
    assert not card.locked(log.dir), "and invisible to it once the folder is finished"


def test_the_lock_is_still_held_while_the_page_is_written(log, monkeypatch):
    """The teardown reorder. The log used to be closed - dropping the lock - before the page,
    the summary and the rename, leaving all three racing anything that started meanwhile."""
    held_during = []
    real = session.SessionLog._write_page

    def watched(self):
        held_during.append(card.locked(self.dir))
        return real(self)

    monkeypatch.setattr(session.SessionLog, "_write_page", watched)
    with log:
        log.event("you", text="hello")

    assert held_during == [True], "the page is written while this folder still claims itself"


def test_the_log_is_synced_before_the_page_is_derived_from_it(log, monkeypatch):
    """A page that outlived its own log is the one inconsistency this format cannot repair."""
    order = []
    monkeypatch.setattr(session.SessionLog, "_sync_log",
                        lambda self: order.append("sync"))
    real = session.SessionLog._write_page
    monkeypatch.setattr(session.SessionLog, "_write_page",
                        lambda self: (order.append("page"), real(self))[1])

    with log:
        log.event("you", text="hello")

    assert order == ["sync", "page"]


def test_a_session_that_raised_still_leaves_a_readable_folder(log):
    with pytest.raises(RuntimeError), log:
        log.event("you", text="mid-sentence")
        raise RuntimeError("the agent fell over")

    records, dropped = card.read_log(log.dir / card.LOG_NAME)
    assert dropped == 0
    tail = records[-1]
    assert tail["type"] == "end" and tail["reason"] == "error"
    assert card.written(log.dir / card.PAGE_NAME), "a crashed session is still a finished folder"


def test_recovery_would_leave_this_session_alone_while_it_runs(log, tmp_path, capsys):
    """The two halves meeting: a real live SessionLog, and a real recovery sweep beside it."""
    settings = Settings(api_key="", sessions_dir=tmp_path / "sessions")
    with log:
        log.event("you", text="still talking")
        session._recover(settings)
        out = capsys.readouterr().out

    assert "a session is writing here; left alone" in out
    assert log.dir.is_dir(), "and above all, it was not renamed or removed out from under us"
