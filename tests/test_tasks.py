"""The background-task ledger: what a row says, and what it survives.

Pure logic, which is what this suite is for - no key, no network, no camera. Everything here
runs against the throwaway file `conftest.ledger` points cyclops.tasks at.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta

import pytest

from cyclops import card, tasks


def test_a_task_is_running_the_moment_it_is_opened(ledger) -> None:
    tasks.start("drawing the relay wiring…")

    (row,) = tasks.read()
    assert row.state == tasks.RUNNING
    assert row.what == "drawing the relay wiring…"
    assert row.ended == "" and row.result == "", "nothing has happened to it yet"
    assert ledger.exists(), "the file is the whole point; it is written, not held in memory"


def test_finishing_closes_the_row_it_was_told_about_and_no_other() -> None:
    """Two in flight at once is the ordinary case: a drawing outlives the turn that asked for it,
    and the next thing they say can start another."""
    first = tasks.start("drawing the relay wiring…")
    tasks.start("editing the picture to paint the doors matt black…")

    tasks.finish(first)

    by_id = {row.id: row for row in tasks.read()}
    assert by_id[first].state == tasks.DONE
    assert by_id[first].ended, "a closed row says when"
    assert len(tasks.running()) == 1, "the other one is still going"


def test_a_failure_keeps_what_went_wrong() -> None:
    task = tasks.start("drawing the relay wiring…")
    tasks.fail(task, "gpt-image-2 refused: content policy")

    (row,) = tasks.read()
    assert row.state == tasks.FAILED
    assert row.result == "gpt-image-2 refused: content policy"


def test_the_newest_running_task_is_what_the_panel_says() -> None:
    """`line()` is read once per rendered frame and is the caption's last fallback."""
    tasks.start("naming the last session…")
    newer = tasks.start("filing it under a project…")

    assert tasks.line() == "filing it under a project…"

    tasks.finish(newer)
    assert tasks.line() == "naming the last session…", "it falls back to what is still going"


def test_nothing_going_on_is_an_empty_line_and_not_a_crash() -> None:
    assert tasks.line() == ""
    assert tasks.read() == []
    assert tasks.running() == []


def test_the_answer_changes_when_the_file_does() -> None:
    """`line()` memoises on the file's stat so the kiosk can ask every frame. The memo must not
    outlive the file it was taken from - which is the whole risk of having one."""
    assert tasks.line() == ""
    tasks.start("drawing the relay wiring…")
    assert tasks.line() == "drawing the relay wiring…", "a memo that stuck here would say nothing"


def test_a_row_that_has_been_running_too_long_is_not_believed() -> None:
    """Nothing writes "the process that owned this died" - it is dead. Age is the only evidence,
    and a panel still claiming to draw a picture an hour later is the failure this prevents."""
    stale = datetime.now().astimezone() - timedelta(seconds=tasks.STALE_AFTER_S + 60)
    card.write_text(
        tasks.TASKS_FILE,
        "tasks:\n"
        "- id: abc123\n"
        "  what: drawing the relay wiring…\n"
        "  state: running\n"
        f"  started: '{stale.strftime(tasks.STAMP)}'\n",
    )

    assert tasks.read(), "it is still in the file - nothing was rewritten behind anyone's back"
    assert tasks.running() == [], "but nobody believes it"
    assert tasks.line() == ""


def test_a_row_with_an_unreadable_stamp_is_left_alone() -> None:
    """Somebody edited the file by hand, or the row predates the offset in STAMP. Generous is
    right for both: a row whose age cannot be worked out should not be swept up for being old,
    and it is what makes changing that format cost nothing to clean up after."""
    card.write_text(
        tasks.TASKS_FILE,
        "tasks:\n- id: abc123\n  what: doing something…\n  state: running\n  started: yesterday\n",
    )

    assert tasks.line() == "doing something…"


def test_only_the_last_few_rows_survive() -> None:
    """A ledger, not a log. What happened an hour ago is on the card, in the session it was part
    of; this file answers "is anything going on right now"."""
    for n in range(tasks.KEEP + 10):
        tasks.finish(tasks.start(f"job {n}…"))

    rows = tasks.read()
    assert len(rows) == tasks.KEEP
    assert rows[0].what == f"job {tasks.KEEP + 9}…", "newest first, and the oldest fall off"


def test_a_line_too_long_for_the_caption_is_cut_rather_than_wrapped() -> None:
    task = tasks.start("drawing " + "a very long thing " * 20)
    (row,) = tasks.read()
    assert len(row.what) <= tasks.MAX_LINE
    assert row.what.endswith("…")
    tasks.fail(task, "boom\nwith a second line")
    assert "\n" not in tasks.read()[0].result, "never a newline inside a scalar"


# ---------------------------------------------------------------- what it survives


def test_two_writers_at_once_lose_nothing() -> None:
    """The reason for the lock. Without it each thread writes back the half of the file it read,
    and whichever lands second silently drops the other's row."""
    started: list[str] = []
    barrier = threading.Barrier(8)

    def open_one(n: int) -> None:
        barrier.wait()  # all eight inside _edit's critical section as close together as we can
        started.append(tasks.start(f"job {n}…"))

    threads = [threading.Thread(target=open_one, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len({row.id for row in tasks.read()}) == 8, "every one of them is in the file"
    assert sorted(row.id for row in tasks.read()) == sorted(started)


def test_a_file_that_makes_no_sense_reads_as_nothing_going_on() -> None:
    """The same answer a missing summary gets in cyclops.session, and it means the same thing:
    there is nothing here to trust, and the next write rebuilds it."""
    for rubbish in ("}{ not yaml at all", "tasks: a string\n", "- a list\n- not a mapping\n", ""):
        card.write_text(tasks.TASKS_FILE, rubbish)
        assert tasks.read() == []
        assert tasks.line() == ""


def test_a_row_somebody_broke_by_hand_does_not_take_the_file_with_it() -> None:
    card.write_text(
        tasks.TASKS_FILE,
        "tasks:\n- what: no id at all…\n  state: running\n"
        "- id: abc123\n  what: drawing the relay wiring…\n  state: running\n"
        f"  started: '{datetime.now().astimezone().strftime(tasks.STAMP)}'\n",
    )

    assert [row.id for row in tasks.read()] == ["abc123"], "the bad row is skipped, not fatal"
    assert tasks.line() == "drawing the relay wiring…"


def test_a_card_that_will_not_write_is_not_a_traceback(monkeypatch, capsys) -> None:
    """A ledger that throws takes down the work it was only watching, which would make this
    module strictly worse than not having it."""
    def refuse(*args, **kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(card, "write_text", refuse)

    task = tasks.start("drawing the relay wiring…")
    tasks.finish(task)

    assert task, "an id comes back regardless, so finish() always has something to be called with"
    assert "could not write down" in capsys.readouterr().out


# ---------------------------------------------------------------- the context manager


def test_a_block_that_returns_closes_its_task() -> None:
    with tasks.run("naming the last session…") as task:
        assert tasks.line() == "naming the last session…", "it is open while the block runs"

    (row,) = tasks.read()
    assert row.id == task and row.state == tasks.DONE
    assert tasks.line() == ""


def test_a_block_that_raises_is_written_down_and_still_raises() -> None:
    """This records what happened; it does not decide what happens next. cyclops.after's `_step`
    is what catches these, and it must go on seeing them."""
    with pytest.raises(ValueError):
        with tasks.run("filing it under a project…"):
            raise ValueError("nothing to file")

    (row,) = tasks.read()
    assert row.state == tasks.FAILED
    assert "ValueError: nothing to file" in row.result


def test_an_empty_answer_from_a_job_is_a_done_task_and_not_a_failed_one() -> None:
    """cyclops.after's three jobs return early when there is nothing to do. Having nothing to
    remember is a result, not a failure."""
    with tasks.run("remembering what that was about…"):
        pass

    assert tasks.read()[0].state == tasks.DONE


def test_the_file_is_readable_by_a_person() -> None:
    """The one thing this feature is for: `cat ~/.cache/cyclops/tasks.yaml` over ssh."""
    tasks.start("drawing the relay wiring…")
    text = tasks.TASKS_FILE.read_text()

    assert "drawing the relay wiring…" in text, "the ellipsis as itself, not as \\u2026"
    assert text.startswith("tasks:")
    assert "\n  what: " in text, "one row per task, block style, not a flow mapping"


def test_the_clock_it_writes_is_the_one_it_reads_back() -> None:
    task = tasks.start("drawing the relay wiring…")
    time.sleep(0.01)
    (row,) = tasks.read()
    assert row.id == task
    assert row.age_s < tasks.STALE_AFTER_S
    assert datetime.strptime(row.started, tasks.STAMP), "the stamp parses with our own format"


def test_a_reader_in_another_timezone_sees_the_same_row(monkeypatch) -> None:
    """The stamp carries its offset, and this is why.

    The admin service is one of the readers here, and Django sets ``os.environ["TZ"]`` from its
    own TIME_ZONE and calls ``time.tzset()`` while it starts up. With a naive stamp every running
    row read two hours old on that side and `running()` dropped the lot as stale - so the page
    said nothing was going on while a drawing was in flight. Measured, on 2026-09-05.
    """
    tasks.start("drawing the relay wiring...")
    assert tasks.line() == "drawing the relay wiring..."

    for zone in ("America/Chicago", "Pacific/Auckland", "UTC"):
        monkeypatch.setenv("TZ", zone)
        time.tzset()
        monkeypatch.setattr(tasks, "_memo", None)
        assert tasks.line() == "drawing the relay wiring...", f"a reader in {zone} sees it too"
    monkeypatch.undo()
    time.tzset()  # put the process back where it was; monkeypatch cannot do this half
