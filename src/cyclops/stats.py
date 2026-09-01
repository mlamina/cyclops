"""The numbers the admin page shows: temperature, memory, disk, recorded sessions.

Stdlib only - no ``psutil``. Everything here is a read of ``/proc``, ``/sys`` or ``statvfs``,
so a poll costs microseconds and never writes to the card. On macOS the Linux-only reads return
``None`` rather than raising, so the page still renders while you develop against a laptop.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .card import LOG_NAME, PAGE_NAME, written
from .config import Settings

THERMAL = Path("/sys/class/thermal")
MEMINFO = Path("/proc/meminfo")
UPTIME = Path("/proc/uptime")

# A Pi 5 starts capping its clock at 80 C (the "soft temperature limit") and gets serious at 85.
# Those are the board's own numbers, not taste, so they are what the tile colours off.
WARN_C = 70.0
HOT_C = 80.0
THROTTLE_C = 85.0  # where the board stops being polite about it
# How far back down the board has to come before the panel's lamp lets a step go. A Pi being
# throttled does not cross its limit briskly and carry on past it - it is *held* there, and the
# reading wanders a degree either side of the line while it is (84.2 to 85.3, measured on this
# one). Without this the lamp changes colour every time it is read, which looks like a fault in
# the lamp rather than a fact about the board.
HYSTERESIS_C = 1.5


@dataclass(frozen=True)
class SystemStats:
    """One sample of how the box is doing. ``None`` means "this platform can't tell us"."""

    temp_c: float | None
    temp_band: str  # "ok" | "warn" | "hot" | "unknown" - drives the tile colour
    mem_used: int | None
    mem_total: int | None
    disk_used: int | None
    disk_total: int | None
    disk_free: int | None
    sessions: int  # session folders that finished and wrote their page
    sessions_in_progress: int  # folders a session is still filling, or died holding
    uptime_s: float | None
    load1: float | None


def _cpu_zones() -> list[Path]:
    """Thermal zones with the CPU's first - a Pi 5 has exactly one, other boards have several."""
    if not THERMAL.is_dir():
        return []

    def cpu_last(zone: Path) -> int:
        try:
            return 0 if "cpu" in (zone / "type").read_text() else 1
        except OSError:
            return 1

    return sorted(sorted(THERMAL.glob("thermal_zone*")), key=cpu_last)


def cpu_temp_c() -> float | None:
    """CPU temperature in degrees C, straight from sysfs (millidegrees), or None off Linux.

    Deliberately not ``vcgencmd measure_temp``: that is a subprocess per poll for the same number.
    """
    for zone in _cpu_zones():
        try:
            millidegrees = int((zone / "temp").read_text().strip())
        except (OSError, ValueError):
            continue
        if millidegrees > 0:
            return millidegrees / 1000.0
    return None


def temp_band(temp_c: float | None) -> str:
    """Which of green / orange / red this reading falls in."""
    if temp_c is None:
        return "unknown"
    if temp_c >= HOT_C:
        return "hot"
    if temp_c >= WARN_C:
        return "warn"
    return "ok"


def heat_alarm(temp_c: float | None, was: str = "") -> str:
    """What the panel's heat lamp should show: ``""``, ``"hot"`` or ``"throttled"``.

    A second reading of the same number as :func:`temp_band`, and deliberately a stricter one.
    The admin page's tile is a gauge someone chose to go and look at, so it can afford to go
    amber at WARN_C - a temperature a Pi 5 reaches doing ordinary work. A lamp on the panel is
    a different promise: it interrupts you whether or not you asked, so it is only worth
    lighting once the board is actually taking something away. That is HOT_C, where the clock
    starts being capped, and THROTTLE_C, where it is capped in earnest and the camera and the
    audio devices start missing their deadlines.

    ``was`` is what the lamp is showing now, and it is what makes the answer stable: a step only
    releases once the board is HYSTERESIS_C below the line it went up on. Pass it and the lamp
    holds; leave it out and this is a plain comparison, which is right for a one-off reading and
    wrong for anything watching. See :data:`HYSTERESIS_C`.
    """
    if temp_c is None:
        return ""
    hot = HOT_C - HYSTERESIS_C if was in ("hot", "throttled") else HOT_C
    throttled = THROTTLE_C - HYSTERESIS_C if was == "throttled" else THROTTLE_C
    if temp_c >= throttled:
        return "throttled"
    if temp_c >= hot:
        return "hot"
    return ""


def memory() -> tuple[int, int] | None:
    """``(used, total)`` bytes from /proc/meminfo, or None off Linux.

    Used is ``MemTotal - MemAvailable``, which is the honest figure: it matches what ``free``
    calls available and does not count reclaimable page cache against you.
    """
    try:
        text = MEMINFO.read_text()
    except OSError:
        return None
    values: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            values[parts[0].rstrip(":")] = int(parts[1]) * 1024  # meminfo counts in kB
    total, available = values.get("MemTotal"), values.get("MemAvailable")
    if total is None or available is None:
        return None
    return max(0, total - available), total


def disk(path: Path) -> tuple[int, int, int] | None:
    """``(used, total, free)`` bytes of the filesystem holding *path*, or its nearest parent.

    The sessions directory is what we care about - that is the thing that grows - but nothing
    creates it until the first session runs, so walk up until something exists rather than
    reporting an unrelated ``/``.

    Note ``used + free < total``: free is the unprivileged figure and root's reserve is in
    neither, which is why the percentage is taken over ``used + free``. That is how ``df``
    computes Use%, so the page and the shell agree instead of the page reading two low.
    """
    probe = path.expanduser().resolve()
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            return None
        probe = parent
    try:
        usage = shutil.disk_usage(probe)
    except OSError:
        return None
    return usage.used, usage.total, usage.free


# From cyclops.card, not cyclops.session: session.py pulls in record.py and with it OpenCV, a
# heavyweight import for a status page that only wants to count folders. These used to be typed
# out again here for exactly that reason, which is how two modules quietly stop agreeing about
# what a finished session looks like. card.py exists to be the copy they can share.


def session_counts(sessions_dir: Path) -> tuple[int, int]:
    """``(finished, in progress)`` - session folders, by the file that marks one done.

    ``session.md`` is written last, after the mux and just before the folder is renamed, so a
    page *with bytes in it* is exactly "this one finished" - :func:`cyclops.card.written`, not
    ``is_file()``, because a power cut used to leave a zero-byte one and this tile counted it as
    a finished session. A folder with a log and no page is either running right now or was
    interrupted, which mean the same thing to someone reading the tile. It is also the only rule
    that works for all three entry points: only the kiosk ever produces an mp4, so counting
    those would under-report by two thirds.

    ``iterdir`` does not descend, so a session's own ``photos/`` and ``parts/`` are never
    mistaken for sessions themselves.
    """
    directory = sessions_dir.expanduser()
    if not directory.is_dir():
        return 0, 0
    finished = in_progress = 0
    for entry in directory.iterdir():
        if not entry.is_dir():
            continue
        if written(entry / PAGE_NAME):
            finished += 1
        elif (entry / LOG_NAME).is_file():
            in_progress += 1
    return finished, in_progress


def uptime_s() -> float | None:
    """How long the box has been up, in seconds, or None off Linux."""
    try:
        return float(UPTIME.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def load1() -> float | None:
    """One-minute load average, or None where the platform has no such thing."""
    try:
        return os.getloadavg()[0]
    except (OSError, AttributeError):
        return None


def collect(settings: Settings) -> SystemStats:
    """One sample of everything, for a page render or an API poll."""
    temp = cpu_temp_c()
    mem_used, mem_total = memory() or (None, None)
    disk_used, disk_total, disk_free = disk(settings.sessions_dir) or (None, None, None)
    finished, running = session_counts(settings.sessions_dir)
    return SystemStats(
        temp_c=None if temp is None else round(temp, 1),
        temp_band=temp_band(temp),
        mem_used=mem_used,
        mem_total=mem_total,
        disk_used=disk_used,
        disk_total=disk_total,
        disk_free=disk_free,
        sessions=finished,
        sessions_in_progress=running,
        uptime_s=uptime_s(),
        load1=load1(),
    )
