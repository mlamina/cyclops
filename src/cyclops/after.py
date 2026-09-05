"""Everything that happens to a session once it has ended, in a process that outlives it.

Three jobs want a model after a conversation is over:

* **what this session was** - a name for the folder and the paragraph inside ``summary.md``
  (:mod:`cyclops.slug`),
* **what it said about the person** - ``about-you.md`` (:mod:`cyclops.about`),
* **where it belongs** - an entry under a project (:mod:`cyclops.projects`).

Between them that is three round trips and, on a slow night, the better part of a minute. None
of it belongs in the teardown. The tap that ends a session is somebody saying they are done, and
what should follow is the panel going dark - not "summarising…" while a Pi holds a socket open
to talk about a conversation that is already over. So :meth:`cyclops.session.SessionLog.__exit__`
writes what it can from the records it already has, spawns this, and stops.

A detached process rather than a thread, for the reason the filing sweep always gave: a deploy
pkills the kiosk, and a thread dies with the process that owns it while a session leader of its
own does not.

Two of the three jobs sweep the whole card rather than visiting the one folder they were handed,
which is what makes the retry story free - the queue is the filesystem. A folder with no name is
one still to name; one with no ``project.md`` is one still to file. So a child that never
started, or died halfway, or ran with the wifi down, costs nothing but time: the next session to
end sweeps the card again, and so does every boot (``cyclops-sessions --recover``).

Remembering is the exception, and knowing why is the price of reading this file. There is no
mark on the card that says a session has been read for facts - ``about-you.md`` is rewritten
rather than appended to, so there is nothing to compare it against - which means there is no
queue to sweep and a session whose child never ran is simply not remembered. For a list that
comes back unchanged after most conversations anyway, that is the cheapest of the three failures
and not worth a ledger to avoid.

The order below is the other thing worth knowing: remembering runs **first**, because naming
*renames* the folder this process was handed and the path on its own command line would stop
existing underneath it.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

from . import tasks
from .config import ConfigError, Settings, load_settings

__all__ = ["main", "spawn", "wanted"]

# Where the child says how it went, in full. The panel now says *that* it is happening - each of
# the three jobs opens a row in cyclops.tasks, so the caption reads "naming the last session…"
# instead of the snore while this runs - but a caption is one sentence and this is the
# traceback. Beside the projects lock
# in ~/.cache rather than on the card, for the reason store.py gives about both: a lock and a
# logbook are facts about this machine, and the card is a tree whose whole purpose is to be
# browsable. The name still says "projects" because that is what it used to carry and what
# fingers still type; the naming and the remembering write here too now.
#
# Here rather than in :mod:`cyclops.projects.store`, where it lived when filing was the only job:
# importing that package costs 1.5 s of pydantic on a Pi 5, and :func:`spawn` runs on the teardown
# path. Paying it there made the shutdown a second longer to hand a child a file handle.
LOG_FILE = Path.home() / ".cache" / "cyclops" / "projects.log"

USAGE = """\
usage: python -m cyclops.after [SESSION FOLDER]

  Names and summarises every session on the card that has neither, notes anything the session
  at SESSION FOLDER said about the person, and files everything still unfiled. This is what a
  session end spawns; running it by hand is the same thing, and safe to repeat."""


# ------------------------------------------------------------------ the detached spawn


def wanted(settings: Settings) -> bool:
    """Is there anything for a child to do? Asked before one is started.

    Every job here is a model call, so a box with no key has nothing to hand off - and a box with
    all three switched off has nobody to hand it to.
    """
    if not settings.api_key:
        return False
    return bool(settings.slug or settings.remember or settings.projects)


def spawn(folder: Path, settings: Settings) -> None:
    """Start the child and let go of it. Never waits, never raises."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    handle = LOG_FILE.open("ab")
    try:
        started = f"{datetime.now().astimezone():%Y-%m-%d %H:%M:%S}"
        handle.write(f"\n=== {started} after {folder.name}\n".encode())
        handle.flush()
        subprocess.Popen(  # noqa: S603 - the argv is ours; nothing here came from a model
            [sys.executable, "-m", "cyclops.after", str(folder)],
            # setsid, for two things: a Ctrl+C in the terminal that ran `cyclops` goes to the
            # foreground process group and would otherwise take this with it, and the child gets
            # a command line of its own that cannot match start-kiosk.sh's `pkill -f
            # bin/cyclops-kiosk`, however that pattern is later edited.
            start_new_session=True,
            # The working directory is not incidental. sessions_dir and projects_dir are
            # CWD-relative and deliberately unresolved, and load_settings finds .env by walking up
            # from the CWD - start the child anywhere else and it works on the wrong tree with no
            # error on either side. Passed explicitly so it reads as a decision.
            cwd=Path.cwd(),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=handle,
        )
    finally:
        handle.close()  # the child holds its own dup; ours would leak one per session


# ------------------------------------------------------------------ the three jobs


def _remember(folder: Path | None, settings: Settings) -> None:
    """What this one conversation said about the person - see :mod:`cyclops.about`.

    This one session and no other, for the reason the module docstring gives: there is nothing
    on the card that says which sessions have already been read for facts.
    """
    if folder is None or not settings.remember:
        return
    from .about import read, remember, write
    from .card import LOG_NAME, read_log
    from .session import transcript_text

    # The task opens after the guard, so a box with CYCLOPS_REMEMBER=0 leaves no row for work it
    # was never going to do. It closes as "done" on the two empty answers below, which is honest:
    # having nothing to remember is a result, not a failure.
    with tasks.run("remembering what that was about…"):
        records, _ = read_log(folder / LOG_NAME)
        text = transcript_text(records)
        if not text:
            return  # nothing was said, so there is nothing this could have taught us
        known = read(settings)
        facts = remember(text, known, settings)
        if not facts or facts == known:
            return  # an empty answer means leave the list alone, never empty it
        write(settings, facts)
        print(f"· remembered: {len(facts)} thing(s) about you (was {len(known)})", flush=True)


def _name(settings: Settings) -> None:
    """A name and a ``summary.md`` for everything on the card that has neither."""
    if not settings.slug:
        return
    from .session import describe_pending

    with tasks.run("naming the last session…"):
        describe_pending(settings)


def _file(settings: Settings) -> None:
    """Everything still unfiled, oldest first. Runs last, because it is the long one.

    After the naming, deliberately: ``summary.md`` is the brief the filing agents are handed, and
    filing a session before it has one files it worse.
    """
    if not settings.projects:
        return
    import asyncio

    from . import projects

    with tasks.run("filing it under a project…"):
        asyncio.run(projects.sweep(settings))


def _step(what: str, job) -> None:
    """Run one job. One that falls over never takes the other two down with it."""
    try:
        job()
    except Exception as exc:  # noqa: BLE001 - three independent jobs, and this is the last stop
        said = f"· could not finish {what}: {type(exc).__name__}: {exc}"
        print(said, file=sys.stderr, flush=True)


def main() -> None:
    """``python -m cyclops.after <session folder>`` - what a session end spawns."""
    import os

    args = sys.argv[1:]
    if "--help" in args or "-h" in args:
        print(USAGE)
        return
    if flags := [a for a in args if a.startswith("-")]:
        raise SystemExit(f"error: unknown argument {flags[0]!r}\n{USAGE}")

    try:
        settings = load_settings(require_api_key=False)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    if not settings.api_key:
        print("· no OPENAI_API_KEY, so there is nothing to finish", flush=True)
        return

    # Deprioritised the way cyclops-admin.service is, and for the same reason: this starts while
    # the next session may already be feeding x264 on the same four cores, and a conversation
    # must never wait on the tidying up after the last one.
    try:
        os.nice(10)
    except OSError:
        pass

    folder = Path(args[0]).expanduser() if args else None
    _step("remembering", lambda: _remember(folder, settings))
    _step("naming", lambda: _name(settings))
    _step("filing", lambda: _file(settings))


if __name__ == "__main__":
    main()
