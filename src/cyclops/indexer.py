"""The service that keeps the index true: watch the card, caption what is new, reconcile.

The write half of recall. :mod:`cyclops.recall` is the read half and the voice agent only ever
touches that one - it loads a file, embeds a query and takes a dot product. Everything expensive
about keeping search working happens here instead, in a niced background process, at the moment a
file changes rather than at the moment somebody asks a question.

That split is the whole design. Indexing inside the kiosk would tie it to a process a deploy
pkills; indexing inside the admin service would put model calls in a request handler; indexing
lazily inside the voice agent would make the first recall after a busy afternoon the slowest one.
So it is its own unit, `cyclops-index`, alongside `cyclops-admin`.

**One rule about captions**, and it is why captioning lives here rather than on the teardown path
where it started out::

    An image with no caption gets one, whatever put it there.

There are three ways a picture reaches this card - the shutter button, the filing curator copying
a hero shot into a project, and you dragging a file onto the admin page or scp'ing one in - and a
hook on the end of a session catches the first of those. A watcher on the tree catches all three
for free, because to inotify they are the same event. See :mod:`cyclops.captions`.

**Reconcile, never rebuild.** Every wake walks the card, diffs ``(path, mtime, size)`` against
what the index says, and touches only the difference. An unchanged card costs a stat walk and no
network at all, which is what makes a 15-minute safety-net sweep affordable on a box that
overheats.

**Nothing here is load-bearing for correctness.** The index is derived; the card holds every fact.
A missed event costs staleness until the next sweep, a crash costs nothing, and deleting the file
costs one rebuild. That is deliberate, and it is what lets this run unattended.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import fcntl
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

from . import around, captions, card, cut, recall
from .config import RECALL_FILE, RECALL_LOCK, ConfigError, Settings, load_settings

# How long the card must be quiet before a burst of events is treated as finished. Filing one
# session writes a dozen files in a second or two - a log entry, two photos, a README, a receipt -
# and reconciling once at the end of that beats reconciling twelve times during it.
QUIET_S = 5.0
# Unless events keep coming, in which case reconcile anyway. A long video mux or a big upload can
# keep a directory busy for minutes, and waiting for silence that never arrives is how a watcher
# stops watching.
MAX_WAIT_S = 60.0
# The safety net, for an event inotify dropped or a change made while this was down. Nearly free:
# an unchanged card is one stat walk of a few dozen files.
SWEEP_EVERY_S = 900.0
# What a single reconcile is allowed before it is abandoned. Above a first run captioning forty
# photos one at a time; a sweep that exceeds this is wedged, not slow.
RECONCILE_BUDGET_S = 1800.0
# And what one unit of clipping is allowed - deciding, or one encode. Comfortably above
# cut.RENDER_TIMEOUT_S, which is the budget that should actually stop a wedged encode: this is
# the outer one, and an outer budget that can fire first would cancel the coroutine and leave
# the ffmpeg it was waiting on still running.
CUT_BUDGET_S = 420.0


def _say(message: str, *, error: bool = False) -> None:
    """One line to the journal. Flushed, because systemd reads a pipe."""
    print(f"· {message}", file=sys.stderr if error else sys.stdout, flush=True)


# ------------------------------------------------------------------ the lock


class Busy(RuntimeError):
    """Another indexer holds the lock."""


def held() -> object:
    """Take the index lock, or raise :class:`Busy`. The handle must be kept alive to keep it.

    ``flock`` for the reason ``store.held`` gives: the kernel drops it when the holder dies, so a
    box that loses power mid-reconcile comes back with no lock at all - nothing stale to detect
    and nobody to unwedge it. Non-blocking, because the two callers are a running service and
    somebody typing ``--once`` at it, and the right answer for the second is "it is already
    running", not a wait.
    """
    RECALL_LOCK.parent.mkdir(parents=True, exist_ok=True)
    handle = RECALL_LOCK.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise Busy(f"another indexer holds {RECALL_LOCK}") from None
    return handle


# ------------------------------------------------------------------ one reconcile


def _known_captions(folders: list[Path]) -> dict[str, str]:
    """Every caption already written, keyed by what the picture *contains* rather than its name.

    This is what stops one photograph being described twice. The filing curator copies a hero shot
    into its project under a new name, so the identical JPEG is on the card in two folders - and
    captioning each of them separately did not merely cost two calls, it produced two *different*
    readings of the same picture. Measured on 2026-09-04 against one manual page: one pass read
    the oil drain plug as "30 and 80" and the other as "N·m 10; in-lb 88". At least one of those
    is wrong, and nothing downstream could tell which.
    """
    known: dict[str, str] = {}
    for folder in folders:
        for name, text in captions.read(folder).items():
            mark = recall.content_hash(folder / name)
            if mark and mark not in known:
                known[mark] = text
    return known


async def caption_pass(settings: Settings, *, again: bool = False) -> int:
    """Give every picture on the card that has none a caption. Returns how many were written.

    Runs before the embedding pass so that a photo captioned on this sweep is embedded on this
    sweep rather than the next one - which is the difference between a photo you just uploaded
    being findable in ten seconds and in fifteen minutes.
    """
    # Every folder holding pictures, not only those with uncaptioned ones: `fill` also prunes
    # captions for pictures that have been deleted, and a folder where that is the only work is
    # exactly a folder with nothing uncaptioned in it.
    folders = recall.image_folders(settings)
    if not folders:
        return 0
    # Folders with a session behind them first. `image_folders` follows `corpus`, which walks
    # projects before sessions - and the filing sweep copies a hero shot into its project about
    # twenty seconds after a session ends, so on the first sweep after a conversation both copies
    # of that photo are undescribed. `_known_captions` keys on content, so whichever is described
    # first wins for both. Projects-first means the copy is described with no session behind it
    # and the original inherits those words - silently losing the context on exactly the two or
    # three pictures anybody kept.
    folders.sort(key=lambda folder: around.session_of(folder) is None)
    # A fresh map on a recaption run, or every picture would reuse the words it already has.
    known: dict[str, str] = {} if again else _known_captions(folders)
    total = 0
    client = captions.client_for(settings) if settings.api_key else None
    try:
        for folder in folders:
            made = await captions.fill(folder, settings, client, known=known, again=again)
            if made:
                total += made
                _say(f"captioned {made} in {folder}")
    finally:
        if client is not None:
            with contextlib.suppress(Exception):
                await client.close()
    return total


async def reconcile(settings: Settings, *, rebuild: bool = False) -> bool:
    """Bring the index into line with the card. True if anything changed.

    The diff is by ``(kind, path, title)`` for identity and ``(mtime_ns, size)`` for freshness -
    the same pair ``library._cache`` keys on, and for the same reason: it moves exactly when what
    we said about a file stopped being true.

    Only what is new or changed is embedded. What vanished is dropped without a call. An unchanged
    card writes nothing at all, so the file's own mtime is a truthful record of when the index
    last actually moved.
    """
    await caption_pass(settings)

    wanted = recall.corpus(settings)
    current = recall.empty() if rebuild else recall.load(RECALL_FILE)
    known = {item.key: (n, item) for n, item in enumerate(current.items)}

    keep: list[tuple[recall.Item, np.ndarray]] = []
    fresh: list[recall.Item] = []
    for item in wanted:
        found = known.get(item.key)
        # And the text, not only the stamp. Half of what a picture is embedded under lives
        # outside the JPEG - in captions.json, in a project's Log.md, in the session log - so
        # for a picture the stamp has always been a partial answer to "did what we said about
        # this stop being true". Rewrite a caption and the file it describes does not move.
        if found is not None and found[1].stamp == item.stamp and found[1].text == item.text:
            keep.append((item, current.vectors[found[0]]))
        else:
            fresh.append(item)

    gone = len(current.items) - len(keep)
    if not fresh and not gone and not rebuild:
        return False

    vectors: list[np.ndarray] = [row for _, row in keep]
    items: list[recall.Item] = [item for item, _ in keep]
    if fresh:
        if not settings.api_key:
            _say("no API key, so nothing new can be embedded", error=True)
            return False
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=settings.api_key, timeout=recall.EMBED_TIMEOUT_S, max_retries=2
        )
        try:
            made = await recall.embed([item.text for item in fresh], client)
        finally:
            with contextlib.suppress(Exception):
                await client.close()
        items.extend(fresh)
        vectors.extend(made[n] for n in range(len(fresh)))

    stacked = (
        np.vstack(vectors).astype(np.float32)
        if vectors
        else np.zeros((0, recall.EMBED_DIMS), dtype=np.float32)
    )
    index = recall.Index(items=items, vectors=stacked)
    RECALL_FILE.parent.mkdir(parents=True, exist_ok=True)
    card.write_bytes(RECALL_FILE, recall.dump(index))
    _say(f"index: {len(items)} items ({len(fresh)} new or changed, {gone} dropped)")
    return True


async def _guarded(settings: Settings, *, rebuild: bool = False) -> None:
    """One reconcile and one video, neither of which can take the service down."""
    try:
        async with asyncio.timeout(RECONCILE_BUDGET_S):
            await reconcile(settings, rebuild=rebuild)
    except TimeoutError:
        _say(f"reconcile gave up after {RECONCILE_BUDGET_S:.0f}s", error=True)
    except Exception as exc:  # noqa: BLE001 - a bad file must never stop the watching
        _say(f"reconcile failed: {type(exc).__name__}: {exc}", error=True)

    # A video somebody asked for, if there is one waiting. Beside the reconcile rather than
    # inside it, and that placement is the whole of why this is safe to host here: reconcile
    # holds RECALL_LOCK, and an encode running under it would make every search on the box wait
    # on x264. Out here it holds only cut's own lock, and the bell goes on ringing throughout.
    #
    # Its own budget, because a cut must never spend the index's. Its own except, because a
    # render that goes wrong must not cost the sweep that was already finished above.
    #
    # to_thread twice over: cyclops.cut is synchronous on purpose, and ruff's ASYNC rules - which
    # this project selects - are right that subprocess.run has no business in a coroutine.
    try:
        async with asyncio.timeout(CUT_BUDGET_S):
            said = await asyncio.to_thread(cut.one, settings)
        if said:
            _say(said)
    except TimeoutError:
        _say(f"the cut gave up after {CUT_BUDGET_S:.0f}s", error=True)
    except Exception as exc:  # noqa: BLE001 - the same last stop, for the same reason
        _say(f"cut failed: {type(exc).__name__}: {exc}", error=True)


# ------------------------------------------------------------------ watching the card


class _Bell:
    """Something changed. Thread-safe, because the watch reads inotify on its own thread.

    Deliberately not a queue of events: what a reconcile does is walk the whole card, so *which*
    file changed tells it nothing it does not find out anyway. All that has to cross the thread
    boundary is the fact that something did - which is also why the watch below never parses an
    event, only notices that one arrived.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self.rung = asyncio.Event()

    def ring(self) -> None:
        self._loop.call_soon_threadsafe(self.rung.set)


# inotify, by hand. ``watchdog`` was here first and did the same job in three lines, but measured
# on the Pi on 2026-09-04 it cost 0.85% of a core *doing nothing at all* - six threads waking on
# short timeouts - which made this the most expensive idle process on the box, four times
# cyclops-admin, for a service that is asleep almost always. Raising its timeouts changed nothing
# (1 s, 30 s and 300 s all measured the same), so the polling is inside it and not reachable from
# out here.
#
# What is below is one blocking ``os.read`` on one file descriptor. The kernel wakes it when
# something happens and never otherwise, which is the whole point: idle costs exactly zero, and
# the measured lag from a file appearing to the bell ringing stays about a millisecond.
IN_CREATE = 0x00000100
IN_DELETE = 0x00000200
IN_CLOSE_WRITE = 0x00000008  # rather than IN_MODIFY, which fires per write() - a video mux would
IN_MOVED_FROM = 0x00000040  # ring the bell hundreds of times where this rings it once, at close
IN_MOVED_TO = 0x00000080
IN_DELETE_SELF = 0x00000400
IN_MOVE_SELF = 0x00000800
WATCH_MASK = (
    IN_CREATE
    | IN_DELETE
    | IN_CLOSE_WRITE
    | IN_MOVED_FROM
    | IN_MOVED_TO
    | IN_DELETE_SELF
    | IN_MOVE_SELF
)


class Watch:
    """Every directory under the two roots, watched. Rings ``bell`` when any of them changes.

    Never raises, and a failure is survivable rather than fatal: with no watch at all the periodic
    sweep still keeps the index correct, just less promptly. That is why every error path here
    logs and carries on instead of refusing to start.
    """

    def __init__(self, bell: _Bell) -> None:
        self._bell = bell
        self._fd = -1
        self._watched: set[str] = set()

    def start(self) -> bool:
        """Open the descriptor and start reading it. False if inotify is not available at all."""
        import ctypes
        import ctypes.util

        try:
            self._libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
            self._fd = self._libc.inotify_init1(0o2000000)  # IN_CLOEXEC
        except (OSError, AttributeError) as exc:
            _say(f"no inotify ({exc}); falling back to the periodic sweep", error=True)
            return False
        if self._fd < 0:
            _say("no inotify; falling back to the periodic sweep", error=True)
            return False
        threading.Thread(target=self._read, name="cyclops-inotify", daemon=True).start()
        return True

    def add(self, folders: list[Path]) -> None:
        """Watch these directories, skipping any already watched.

        Called again after every reconcile rather than only at startup, because a directory
        created *after* the watch was set up has no watch of its own - a new project, or the
        ``photos/`` a session makes the first time somebody hits the shutter. The reconcile walks
        the card anyway, so refreshing here costs one syscall per genuinely new folder.
        """
        if self._fd < 0:
            return
        for folder in folders:
            key = str(folder)
            if key in self._watched:
                continue
            try:
                handle = self._libc.inotify_add_watch(self._fd, key.encode(), WATCH_MASK)
            except OSError:
                continue
            if handle < 0:
                # Almost always ENOSPC: /proc/sys/fs/inotify/max_user_watches. Worth saying once
                # per folder rather than silently watching less of the card than we claim to.
                _say(f"could not watch {folder}", error=True)
                continue
            self._watched.add(key)

    def close(self) -> None:
        """Drop the descriptor, which ends the reading thread. Never raises."""
        fd, self._fd = self._fd, -1
        if fd >= 0:
            with contextlib.suppress(OSError):
                os.close(fd)

    def _read(self) -> None:
        """Block until the kernel says something happened, then ring. Forever."""
        while True:
            try:
                if not os.read(self._fd, 8192):
                    return
            except OSError:
                return  # the descriptor went; the periodic sweep carries the service from here
            self._bell.ring()


def _folders_to_watch(settings: Settings) -> list[Path]:
    """The two roots and every directory under them.

    Every directory, not only the ones holding indexable files: a picture arrives in a folder that
    may not have existed a moment ago, and the way to hear about that is to be watching its parent.
    """
    out: list[Path] = []
    for root in (settings.projects_dir, settings.sessions_dir):
        folder = root.expanduser()
        if not folder.is_dir():
            continue
        out.append(folder)
        try:
            out.extend(p for p in folder.rglob("*") if p.is_dir())
        except OSError:
            continue
    return out


async def serve(settings: Settings) -> None:
    """Watch the card and reconcile when it changes, until killed.

    Three things wake this, and the reasons differ. Startup catches everything that happened while
    it was down - which is what makes a missed event survivable rather than fatal, and what
    backfills a card that has never been indexed. A filesystem event catches the ordinary case,
    after a pause for the burst to finish. And a timer catches what inotify dropped.
    """
    bell = _Bell(asyncio.get_running_loop())
    watch = Watch(bell)
    watching = watch.start()
    watch.add(_folders_to_watch(settings))
    how = "watching" if watching else "sweeping (no watch)"
    _say(f"{how} {settings.projects_dir} and {settings.sessions_dir}")

    await _guarded(settings)  # the catch-up sweep
    watch.add(_folders_to_watch(settings))

    try:
        while True:
            try:
                await asyncio.wait_for(bell.rung.wait(), timeout=SWEEP_EVERY_S)
            except TimeoutError:
                await _guarded(settings)  # the safety net
                watch.add(_folders_to_watch(settings))
                continue

            # Something moved. Wait for the burst to finish rather than reconciling per file.
            deadline = time.monotonic() + MAX_WAIT_S
            while time.monotonic() < deadline:
                bell.rung.clear()
                try:
                    await asyncio.wait_for(bell.rung.wait(), timeout=QUIET_S)
                except TimeoutError:
                    break  # QUIET_S of calm; the writer has finished
            bell.rung.clear()
            await _guarded(settings)
            # A new project or a session's first photos/ folder did not exist when the watch was
            # set up. Catching it here is what keeps the *next* change in it prompt.
            watch.add(_folders_to_watch(settings))
    finally:
        watch.close()


# ------------------------------------------------------------------ the command


async def _search(settings: Settings, query: str, limit: int) -> int:
    """Print what a recall would find, without a panel. How this gets checked over ssh."""
    index = recall.load(RECALL_FILE)
    if not len(index):
        _say("the index is empty; run cyclops-index --once first", error=True)
        return 1
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.api_key, timeout=recall.EMBED_TIMEOUT_S, max_retries=1)
    try:
        vector = await recall.embed([query], client)
    finally:
        with contextlib.suppress(Exception):
            await client.close()
    hits = recall.rank(index, vector[0], scopes=None, limit=limit)
    if not hits:
        print("nothing matched")
        return 0
    for hit in hits:
        mark = "*" if hit.score >= recall.MIN_SCORE else " "
        print(f"{mark} {hit.score:.3f}  {hit.item.kind:8} {hit.item.scope}")
        print(f"           {hit.item.title}")
        print(f"           {hit.item.path}")
    return 0


async def _recaption(settings: Settings) -> None:
    """Describe every picture on the card again, then index what came back.

    For a change to how pictures are described, which nothing else reaches: a caption is only
    ever written for a picture that has none, so words already on the card stay there forever.

    Over the top rather than by emptying the sidecars first. If the wifi drops halfway through a
    run that began by deleting them, the card is left with no captions at all and every picture
    falls back to its path - so each one is replaced only when a real answer comes back for it.
    """
    await caption_pass(settings, again=True)
    await _guarded(settings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cyclops-index",
        description="Watch the card and keep the recall index in step with it.",
    )
    parser.add_argument("--once", action="store_true", help="reconcile once and exit")
    parser.add_argument("--rebuild", action="store_true", help="discard the index and start over")
    parser.add_argument(
        "--recaption",
        action="store_true",
        help="describe every picture again, with the words said around it",
    )
    parser.add_argument("--search", metavar="QUERY", help="print what a recall would find")
    parser.add_argument("--limit", type=int, default=5, help="how many hits --search prints")
    args = parser.parse_args(argv)

    try:
        settings = load_settings()
    except ConfigError as exc:
        _say(str(exc), error=True)
        return 2

    if args.search:
        return asyncio.run(_search(settings, args.search, args.limit))

    # Never compete with the x264 encoder or the realtime audio thread. The same number after.py
    # and the projects sweep both use, for the same reason.
    with contextlib.suppress(OSError):
        os.nice(10)

    try:
        handle = held()
    except Busy as exc:
        _say(str(exc), error=True)
        return 1
    try:
        if args.recaption:
            asyncio.run(_recaption(settings))
            return 0
        if args.once or args.rebuild:
            asyncio.run(_guarded(settings, rebuild=args.rebuild))
            return 0
        asyncio.run(serve(settings))
    except KeyboardInterrupt:
        pass
    finally:
        handle.close()  # closing releases the flock, whichever way we left
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
