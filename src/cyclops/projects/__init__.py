"""Filing finished sessions into the projects on the card.

A session ends, its folder is complete, and a moment later a detached process reads it and adds
it to whichever project it advanced. Nothing here decides that a project should exist - that
happens in the conversation, through the voice agent's ``track_project`` tool - so this only ever
chooses between projects a person already agreed to, or files nothing at all.

The whole thing is a sweep rather than a job queue, and the queue is the filesystem: a session
with a ``session.md`` and no ``project.md`` is one that still needs reading. That makes the retry
story free. Nothing is filed twice because the receipt says so; nothing is lost because a failure
leaves the receipt unwritten; a power cut costs one re-read and nothing else.

Deliberately importable with no key, no network and no ``pydantic_ai``: :func:`spawn` is called
from ``SessionLog.__exit__``, and a second of import time on that path would be a second of
kiosk shutdown. The agents are imported inside :func:`file_session`, after the key check.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from ..config import ConfigError, Settings, load_settings
from ..session import PAGE_NAME
from . import store
from .deps import Filing, read_session
from .store import Busy, Exists, Project, Unfilable, catalog, create

__all__ = ["Busy", "Exists", "Project", "Unfilable", "catalog", "create", "main", "spawn", "sweep"]

FILE_PROMPT = "File this session. It is at {folder}."
SWEEP_BUDGET_S = 900.0  # whatever is left over simply waits for the next session to end


# ------------------------------------------------------------------ the detached spawn


def spawn(folder: Path, settings: Settings) -> None:
    """Start a detached sweep and let go of it. Never waits, never raises.

    A process rather than a thread, for one reason: on a deploy ``start-kiosk.sh`` pkills the
    kiosk and starts a new one two seconds later, and a thread dies with the process that owns it.
    Filing takes model calls and some file copies - seconds, sometimes tens of them - so it has to
    outlive whatever spawned it.

    It sweeps rather than filing the one folder it was handed. That is what makes this
    self-healing with no timer anywhere: the sweep files the session that just ended *and*
    anything an earlier power cut left behind, oldest first.
    """
    handle = store.open_log()
    try:
        started = f"{datetime.now().astimezone():%Y-%m-%d %H:%M:%S}"
        handle.write(f"\n=== {started} after {folder.name}\n".encode())
        handle.flush()
        subprocess.Popen(  # noqa: S603 - the argv is ours; nothing here came from a model
            [sys.executable, "-m", "cyclops.projects", "--sweep"],
            # setsid, for two things: a Ctrl+C in the terminal that ran `cyclops` goes to the
            # foreground process group and would otherwise take this with it, and the child gets
            # a command line of its own that cannot match start-kiosk.sh's `pkill -f
            # bin/cyclops-kiosk`, however that pattern is later edited.
            start_new_session=True,
            # The working directory is not incidental. sessions_dir and projects_dir are
            # CWD-relative and deliberately unresolved, and load_settings finds .env by walking up
            # from the CWD - start the child anywhere else and it files into the wrong tree with
            # no error on either side. Passed explicitly so it reads as a decision.
            cwd=Path.cwd(),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=handle,
        )
    finally:
        handle.close()  # the child holds its own dup; ours would leak one per session


# ------------------------------------------------------------------ what needs doing


def _folders(settings: Settings) -> list[Path]:
    """Every finished session on the card, oldest first."""
    root = settings.sessions_dir.expanduser()
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / PAGE_NAME).is_file())


def unfiled(settings: Settings, *, again: bool = False) -> list[Path]:
    """The sessions still to read, oldest first.

    Oldest first is load-bearing rather than tidy: a project's page is rewritten against the state
    the session before it left behind, so filing them out of order would produce a page that does
    not know about work that already happened, and then a log entry underneath it that does.
    """
    return [f for f in _folders(settings) if again or not store.is_filed(f)]


# ------------------------------------------------------------------ filing one session


async def file_session(
    folder: Path, settings: Settings, models, usage, *, dry_run: bool = False
) -> str:
    """Read one session and put it where it belongs. Returns one line about what happened.

    Never raises for anything a card can do to you. The model calls happen here, outside the
    lock; only the writes at the bottom are inside it, so a sweep of fifty sessions never holds a
    global lock across fifty network round trips.
    """
    from .agents import RUN_LIMITS, orchestrator  # here, not at import: pydantic_ai costs a second

    session = read_session(folder)
    projects = {p.key: p for p in store.index(settings.projects_dir)}
    deps = Filing(settings=settings, session=session, projects=projects, models=models)

    result = await orchestrator.run(
        FILE_PROMPT.format(folder=folder.name),
        model=models.orchestrator,
        deps=deps,
        usage=usage,
        usage_limits=RUN_LIMITS,
    )
    outcome = result.output

    from .models import Belongs

    chosen = deps.projects.get(deps.verdict.key) if isinstance(deps.verdict, Belongs) else None
    if not outcome.filed or chosen is None or deps.scribed is None:
        why = outcome.why or "not work on anything being tracked"
        if not dry_run:
            store.write_receipt(folder, None, why)
        return f"{folder.name}: not filed - {why}"

    # Belt and braces on the one step that is unconditional: if there are photos and a project to
    # put them in, they get chosen. Leaving it to the orchestrator to remember made it optional,
    # and the first real session it ran on quietly kept none of its only photo.
    if deps.picks is None and session.photos and settings.project_photos > 0:
        from .agents import choose_photos_for

        await choose_photos_for(deps, models, usage)

    if dry_run:
        entry = deps.scribed.entry
        picked = len(deps.picks.picks) if deps.picks else 0
        return f"{folder.name}: would file under {chosen.name!r} - {entry.title} ({picked} photo)"
    return _apply(folder, session, chosen, deps, settings)


def _apply(folder: Path, session, project: Project, deps, settings: Settings) -> str:
    """Everything that touches the card, in the one order that survives a power cut.

    Photos, then the log, then the receipt, then the page. Each step is safe to repeat, and the
    receipt goes before the page deliberately: the page is derived and free to rebuild, while a
    receipt written after it would leave a window where the log has an entry, the page shows it,
    and the next sweep files it all over again.
    """
    scribed = deps.scribed
    picks = [(p.file, p.caption) for p in (deps.picks.picks if deps.picks else [])]
    with store.held(announce=True):
        # Both checks again, inside the lock: another sweep may have filed this while we were
        # waiting on the model, and the ledger is what catches a crash between log and receipt.
        if store.is_filed(folder):
            return f"{folder.name}: already filed while we were thinking"
        fresh = next((p for p in store.index(settings.projects_dir) if p.key == project.key), None)
        if fresh is None:
            return f"{folder.name}: {project.name!r} went away while we were thinking"
        project = fresh

        photos = store.copy_photos(project, folder, picks, limit=settings.project_photos)
        if session.uuid and session.uuid.lower() in store.filed_uuids(project):
            note = "already in the log"
        else:
            store.append_entry(
                project,
                title=scribed.entry.title,
                body=scribed.entry.body,
                when=_when(session),
                uuid=session.uuid,
                stamp=folder.name,
                span=session.span,
                photos=photos,
            )
            note = "filed"

        # The first filing sets the date outright rather than taking a max: a project created
        # this morning and then handed a session from last month was worked on last month, and a
        # shelf ordered by "last touched" should say so.
        first = project.sessions == 0
        project.sessions = store.entry_count(project)
        project.updated = session.date if first else max(project.updated, session.date)
        # The earliest session wins, which is what makes filing an old card sensible: a project
        # created today and then handed six sessions from last month started last month.
        project.started = min(project.started or session.date, session.date)
        project.photos = store.photo_count(project)
        project.last = session.uuid or project.last
        project.aliases = _merge(project.aliases, scribed.aliases, store.MAX_ALIASES)
        project.keywords = _merge(project.keywords, scribed.keywords, store.MAX_KEYWORDS)
        store.rewrite_readme(project, store.render_page(project, scribed.page.model_dump(), photos))
        store.write_receipt(folder, project)
    shot = f", {len(photos)} photo(s)" if photos else ""
    return f"{folder.name}: {note} under {project.name!r}{shot}"


def _merge(existing: list[str], added: list[str], limit: int) -> list[str]:
    """Old names first, new ones after, no repeats. Nothing a project has been called is dropped."""
    from ..slug import fold

    out, seen = [], set()
    for name in [*existing, *added]:
        name = " ".join(str(name).replace(",", " ").split())
        key = fold(name)
        if name and key and key not in seen:
            seen.add(key)
            out.append(name)
    return out[:limit]


def _when(session) -> datetime:
    """When the session happened, off its own record and falling back to its folder name."""
    try:
        return datetime.fromisoformat(session.started)
    except ValueError:
        try:
            return datetime.strptime(session.date, "%Y-%m-%d")
        except ValueError:
            return datetime.now().astimezone()


# ------------------------------------------------------------------ the sweep


async def sweep(
    settings: Settings, *, again: bool = False, limit: int = 0, dry_run: bool = False
) -> None:
    """Read everything that still needs reading, oldest first."""
    import asyncio

    from pydantic_ai.usage import RunUsage

    from .agents import Models

    folders = unfiled(settings, again=again)
    if limit > 0:
        folders = folders[:limit]
    if not folders:
        print("· nothing to file", flush=True)
        return

    models = Models.build(settings)
    spent, requests = Decimal(0), 0
    try:
        async with asyncio.timeout(SWEEP_BUDGET_S):
            for folder in folders:
                # A fresh budget per session, deliberately. usage_limits are checked against
                # whatever usage object they are handed, so one shared across a card would mean
                # the tenth session had to fit in whatever the first nine left - which is not a
                # budget, it is a queue that starves.
                usage = RunUsage()
                try:
                    said = await file_session(folder, settings, models, usage, dry_run=dry_run)
                    print(f"· {said}", flush=True)
                except Busy as exc:
                    print(f"· {folder.name}: {exc}", flush=True)
                except Exception as exc:  # noqa: BLE001 - one bad session never stops the sweep
                    print(f"· {folder.name}: {type(exc).__name__}: {exc}", flush=True)
                finally:
                    requests += usage.requests
                    spent += usage.cost or Decimal(0)
    except TimeoutError:
        print("· out of time for this sweep; the rest wait for the next one", flush=True)
    finally:
        await models.close()
    print(f"· {requests} model request(s), about ${spent:.3f}", flush=True)


# ------------------------------------------------------------------ looking, without a key


def check(settings: Settings) -> None:
    """Offline: what is on the card, and anything that drifted. Never writes, never calls out."""
    projects = store.index(settings.projects_dir)
    if not projects:
        print(f"· no projects in {settings.projects_dir.expanduser()}")
    for project in projects:
        entries = store.entry_count(project)
        flags = []
        if entries != project.sessions:
            flags.append(f"README says {project.sessions} sessions, the log has {entries}")
        if store.torn(project):
            flags.append("the log ends mid-entry - the power probably went")
        if not project.readme.is_file():
            flags.append("no README.md")
        line = f"{project.name:<40}{project.status:>8}{entries:>4} entries"
        print(f"{line}   {'; '.join(flags)}".rstrip())

    waiting = unfiled(settings)
    total = len(_folders(settings))
    print(f"· {total - len(waiting)}/{total} finished sessions filed, {len(waiting)} waiting")


def _usage() -> str:
    return (
        "usage: cyclops-projects [--check] [--sweep [--again] [--limit N] [--dry-run]]\n"
        "                        [--session PATH]"
    )


def main() -> None:
    """``cyclops-projects`` - what is on the card, and the filing of anything not on it yet.

    Shaped like ``cyclops-sessions`` on purpose: bare lists, ``--check`` is the offline half that
    needs nothing, and ``--sweep`` is the half that needs a key and a network.
    """
    import asyncio
    import os

    args = sys.argv[1:]
    flags = {"--check", "--sweep", "--again", "--dry-run"}
    known = flags | {"--limit", "--session"}
    limit, session_arg = 0, ""
    rest = list(args)
    while rest:
        arg = rest.pop(0)
        if arg == "--limit":
            limit = int(rest.pop(0)) if rest else 0
        elif arg == "--session":
            session_arg = rest.pop(0) if rest else ""
        elif arg not in known:
            raise SystemExit(f"error: unknown argument {arg!r}\n{_usage()}")

    try:
        settings = load_settings(require_api_key=False)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    if not settings.projects:
        print("· CYCLOPS_PROJECTS=0 - projects are turned off")
        return

    sweeping = "--sweep" in args or bool(session_arg)
    if not sweeping or "--check" in args:
        check(settings)
        if not sweeping:
            return

    if not settings.api_key:
        print("· no OPENAI_API_KEY, so nothing can be filed (--check works without one)")
        raise SystemExit(1)

    # Deprioritised the way cyclops-admin.service is, and for the same reason: this starts while
    # the next session may already be feeding x264 on the same four cores, and a conversation
    # must never wait on a filing.
    try:
        os.nice(10)
    except OSError:
        pass

    if session_arg:
        folder = Path(session_arg).expanduser()
        if not (folder / PAGE_NAME).is_file():
            raise SystemExit(f"error: {folder} has no {PAGE_NAME} - it is not a finished session")
        asyncio.run(_one(folder, settings, dry_run="--dry-run" in args))
        return
    asyncio.run(
        sweep(settings, again="--again" in args, limit=limit, dry_run="--dry-run" in args)
    )


async def _one(folder: Path, settings: Settings, *, dry_run: bool) -> None:
    from pydantic_ai.usage import RunUsage

    from .agents import Models

    models = Models.build(settings)
    try:
        print(f"· {await file_session(folder, settings, models, RunUsage(), dry_run=dry_run)}")
    finally:
        await models.close()


if __name__ == "__main__":
    main()
