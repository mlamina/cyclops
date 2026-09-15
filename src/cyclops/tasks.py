"""What is happening in the background, in one file anything on the box can write to.

A *task* is a piece of work that outlives the call that started it. Drawing a diagram takes about
half a minute; naming and filing a finished session takes the better part of a minute. Both
used to happen with nothing on the panel and nothing on the card to say so, which is the whole
reason this module exists: **the only purpose of the ledger is so that somebody can see whether
anything is going on.** It is not a scheduler, nothing pulls work off it, and no task here was ever
queued - the work is already running by the time a row appears.

Cyclops never writes a row himself. He calls a tool, and the tool spawns the work and opens the
task for it (:meth:`cyclops.agent.VoiceAgent._run_draw_diagram`); the detached child that tidies up
after a session opens three of its own (:mod:`cyclops.after`). Any process and any thread may
write here, which is why the file is what it is rather than a table in somebody's memory: the
agent, the kiosk, the admin service and a child that outlives all three share no memory at all.

The file is YAML because it is meant to be read - ``cat ~/.cache/cyclops/tasks.yaml`` over ssh is
the plainest answer to "is it doing anything?" there is. It lives under ``~/.cache`` rather than on
the card for the reason :data:`cyclops.config.RECALL_FILE` gives: sessions/ and projects/ are a
tree meant to be browsed in Finder and copied off whole, and a note about what this machine is
doing this minute is neither.

Four rules, and each one is somebody else's rule first:

* **Every write lands whole**, through :func:`cyclops.card.write_text` - see the incident in
  ``card.py``'s docstring. A reader sees the old file or the new one, never a half of either.
* **A read-modify-write is serialised by ``flock``**, the shape ``store.LOCK_FILE`` and
  :data:`cyclops.config.RECALL_LOCK` already use: the kernel drops the lock when the holder dies,
  so a box that loses power mid-write comes back with nothing stale to detect. Reads take no lock,
  because the write above is atomic and there is nothing for them to catch half of.
* **Nothing here raises.** A ledger that throws takes down the work it was only watching, which
  would make this module strictly worse than not having it. Every function below swallows what it
  can go wrong with and returns a safe answer, the way :func:`cyclops.panel.withdraw` does.
* **A running row goes stale.** Nothing writes "this process died" - it is dead. So a row that has
  said ``running`` for longer than anything here can honestly take is dropped by :func:`running`,
  and the panel stops claiming to draw a picture nobody is drawing. That is the only stale-state
  mechanism, deliberately: a boot hook would need an owner and would still be wrong for a process
  killed while the box stayed up.

Standard library, ``yaml`` and :mod:`cyclops.card`, and nothing else - no OpenCV, no openai, not
even :mod:`cyclops.config`. :mod:`cyclops.after` runs this on the teardown path and
``cyclops.admin.views`` imports it into a request, and both of those are places where an import
that costs a second and a half is a bug.
"""

from __future__ import annotations

import fcntl
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import yaml

from . import card

TASKS_FILE = Path.home() / ".cache" / "cyclops" / "tasks.yaml"
# Beside it rather than on it: flock on the ledger itself would mean opening the file to write it,
# and card.write_text lands a *different* inode over the top - so the lock would follow the file
# that was just replaced and protect nothing. A path of its own has no such problem.
TASKS_LOCK = Path.home() / ".cache" / "cyclops" / "tasks.lock"

RUNNING, DONE, FAILED = "running", "done", "failed"

# How many rows survive a write. Enough to see what the last few minutes did and nowhere near
# enough to be a log: what happened yesterday is on the card, in the session it happened in.
KEEP = 20

# When a row that still says "running" stops being believed. Well past DRAW_TIMEOUT_S (180 s),
# which is the longest anything here can honestly take, so nothing real is ever hidden by it.
STALE_AFTER_S = 900.0

# The clock, with its offset on it. The offset is not decoration and this is not cyclops.after's
# format, though it started as one: a naive "2026-09-05 13:54:01" means one instant to the process
# that wrote it and a different one to any process whose TZ differs - and one of the readers here
# is the admin service, where Django sets os.environ["TZ"] from its own TIME_ZONE and calls
# time.tzset() during setup. Measured on 2026-09-05: every running row read two hours old on that
# side and was dropped by `running()` as stale, so the page said nothing was going on while a
# drawing was in flight. An offset makes the stamp mean one instant to everybody, and costs six
# characters of a file that is still read with `cat`.
STAMP = "%Y-%m-%d %H:%M:%S%z"

# The longest a line may be. These are written for the caption bubble on the panel, which wraps to
# two lines and elides past that - so a task whose sentence ran to a paragraph would be cut there
# anyway, and the ledger may as well hold what is actually shown.
MAX_LINE = 120


@dataclass(frozen=True)
class Task:
    """One piece of background work, as it stands right now."""

    id: str
    what: str  # one sentence for a person, in the panel's idiom and ending in an ellipsis
    state: str  # RUNNING, DONE or FAILED
    started: str
    ended: str = ""
    result: str = ""  # a short note where there is one, and the error when it failed

    @property
    def is_running(self) -> bool:
        return self.state == RUNNING

    @property
    def age_s(self) -> float:
        """Seconds since it opened, or 0.0 when the stamp is unreadable.

        Unreadable means somebody edited the file by hand, or a row survives from before the
        offset was written - and the generous answer is right for both: a row whose age cannot be
        established should not be swept up for being old. That is also what makes the format
        change above free, with no migration and nothing to clean out.
        """
        try:
            return max(0.0, time.time() - datetime.strptime(self.started, STAMP).timestamp())
        except (ValueError, TypeError):
            return 0.0


def _now() -> str:
    """Now, with this machine's offset on it. ``astimezone()`` is what puts the ``%z`` there."""
    return datetime.now().astimezone().strftime(STAMP)


def _clip(text: str) -> str:
    """One line, collapsed and cut. Never a newline: this is a scalar somebody has to read."""
    text = " ".join(str(text).split())
    return text if len(text) <= MAX_LINE else text[: MAX_LINE - 1].rstrip() + "…"


# ------------------------------------------------------------------ the file


def read() -> list[Task]:
    """Every row in the file, newest first. Empty when there is no file or it makes no sense.

    Empty is a real answer and the commonest one - most of the time nothing is going on. It is
    also what a corrupt file gets, for the reason :func:`cyclops.session.read_summary` gives about
    a missing summary: there is nothing here to trust, and the next write rebuilds it.
    """
    try:
        text = TASKS_FILE.read_text()
    except OSError:
        return []
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return []
    rows = (data or {}).get("tasks") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return []
    out: list[Task] = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("id"):
            continue  # somebody hand-edited it; skip the row rather than lose the file
        out.append(
            Task(
                id=str(row.get("id", "")),
                what=str(row.get("what", "")),
                state=str(row.get("state", RUNNING)),
                started=str(row.get("started", "")),
                ended=str(row.get("ended", "")),
                result=str(row.get("result", "")),
            )
        )
    return out


def running() -> list[Task]:
    """The rows still worth believing, newest first.

    Stale ones are dropped rather than repaired. See the module docstring: nothing writes "this
    process died", so age is the only evidence there is, and a panel that goes on saying "drawing…"
    an hour after the drawing failed to happen is the one outcome this module must not produce.
    """
    return [task for task in read() if task.is_running and task.age_s < STALE_AFTER_S]


def line() -> str:
    """The newest running task's sentence, or "" when nothing is going on.

    What the panel's caption falls back to, so it is called once per rendered frame - 25 times a
    second on the kiosk. Hence the memo: the answer only changes when the file does, and a
    ``stat`` is cheap where parsing YAML at that rate is not.
    """
    try:
        stat = TASKS_FILE.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        return ""
    global _memo
    if _memo is not None and _memo[0] == stamp:
        return _memo[1]
    tasks = running()
    answer = tasks[0].what if tasks else ""
    _memo = (stamp, answer)
    return answer


_memo: tuple[tuple[int, int], str] | None = None  # (stat stamp, answer); see line()


# ------------------------------------------------------------------ opening and closing one


def start(what: str) -> str:
    """Open a task saying *what*, and hand back its id. Never raises, and never returns "".

    The id is minted here rather than by the caller and is handed back even when the write failed,
    so that :func:`finish` and :func:`fail` always have something to be called with: a box whose
    cache directory has gone read-only should lose its ledger, not its drawing.
    """
    task_id = uuid.uuid4().hex[:6]
    row = {"id": task_id, "what": _clip(what), "state": RUNNING, "started": _now()}
    _edit(lambda rows: [row, *rows])
    return task_id


def finish(task_id: str, result: str = "") -> None:
    """That task is over and it worked."""
    _close(task_id, DONE, result)


def fail(task_id: str, error: str) -> None:
    """That task is over and it did not."""
    _close(task_id, FAILED, error)


@contextmanager
def run(what: str) -> Iterator[str]:
    """Open a task around a block, and close it however the block ends.

    For the callers that are ordinary blocking code - :mod:`cyclops.after`'s three jobs. The two
    image tools do not use it: their work is an asyncio task that outlives the function that
    spawned it, so they call :func:`start` in one place and :func:`finish` in another.

    The exception is re-raised. This records what happened; it does not decide what happens next.
    """
    task_id = start(what)
    try:
        yield task_id
    except BaseException as exc:  # noqa: BLE001 - re-raised below; we only want it written down
        fail(task_id, f"{type(exc).__name__}: {exc}")
        raise
    finish(task_id)


def _close(task_id: str, state: str, result: str) -> None:
    ended, note = _now(), _clip(result)

    def shut(rows: list[dict]) -> list[dict]:
        for row in rows:
            if row.get("id") == task_id:
                row["state"] = state
                row["ended"] = ended
                if note:
                    row["result"] = note
        return rows

    _edit(shut)


# ------------------------------------------------------------------ writing it down


def _edit(mutate: Callable[[list[dict]], list[dict]]) -> None:
    """Read the file, apply *mutate*, and land the result. Never raises.

    The lock is held across the whole of it, which is the point: two processes closing two tasks in
    the same instant would otherwise each write back the half of the file they had read, and one of
    them would lose a row. It is taken blocking rather than ``LOCK_NB`` because there is nothing
    else to do if somebody else has it - the section is one read and one write of a two-kilobyte
    file, and waiting it out is measured in microseconds.
    """
    try:
        TASKS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with TASKS_LOCK.open("a") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                rows = [asdict(task) for task in read()]
                kept = mutate(rows)[:KEEP]
                card.write_text(TASKS_FILE, _render(kept))
            finally:
                # Explicitly, and before the close: the close would drop it anyway, but a lock
                # released where it is taken is one a reader can follow.
                with suppress(OSError):
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError as exc:
        print(f"· could not write down what is going on ({exc})", flush=True)


def _render(rows: list[dict]) -> str:
    """The file, with the empty fields left out so a running row is three lines and not five."""
    tidy = [{name: value for name, value in row.items() if value} for row in rows]
    return yaml.safe_dump(
        {"tasks": tidy},
        sort_keys=False,  # id, what, state, started, ended, result - the order they are read in
        allow_unicode=True,  # the ellipsis every line ends in, as itself rather than as …
        default_flow_style=False,
        # Never fold a line: this file is read with `cat`, and a sentence wrapped at column
        # 80 with a hanging indent is harder to scan than one long line.
        width=1000,
    )
