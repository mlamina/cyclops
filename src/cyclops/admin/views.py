"""The page, the JSON behind it, and the two things the panel can actually change.

Both the first render and every poll go through :func:`_payload`, so the template and the
JavaScript are looking at exactly the same fields and the formatting lives in one place.
"""

from __future__ import annotations

import math
from dataclasses import asdict

from django.http import (
    HttpRequest,
    HttpResponse,
    HttpResponseBadRequest,
    HttpResponseForbidden,
    JsonResponse,
)
from django.shortcuts import render
from django.views.decorators.http import require_POST

from .. import mixer, stats
from ..config import BROWSER_CLOSE_FLAG, ConfigError, Settings, load_settings

LOOPBACK = {"127.0.0.1", "::1"}
GIB = 1024**3  # what df -h means by "G", so the page and the shell agree

_settings_cache: Settings | None = None


def _settings() -> Settings:
    """Config, read once - it comes from the environment, which does not change under us.

    A typo in ``.env`` must not take the status page down with it, so a bad value falls back
    to the defaults rather than raising: knowing the box is at 82 C matters more than knowing
    ``CYCLOPS_RECORD_FPS`` is misspelt. Edits to ``.env`` need a restart to be seen.
    """
    global _settings_cache
    if _settings_cache is None:
        try:
            _settings_cache = load_settings(require_api_key=False)
        except ConfigError as exc:
            print(f"· ignoring bad .env ({exc}); using defaults", flush=True)
            _settings_cache = Settings(api_key="")
    return _settings_cache


def _is_local(request: HttpRequest) -> bool:
    """Is this the kiosk's own browser, rather than a laptop somewhere on the LAN?"""
    return request.META.get("REMOTE_ADDR") in LOOPBACK


def _percent(used: int | None, whole: int | None) -> int | None:
    if used is None or not whole:
        return None
    return round(100 * used / whole)


def _gib(value: int | None) -> str:
    return "—" if value is None else f"{value / GIB:.1f}"


def _duration(seconds: float | None) -> str:
    """A rough, readable uptime: days and hours, or hours and minutes, or just minutes."""
    if seconds is None:
        return "—"
    minutes = int(seconds // 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    if days:
        return f"up {days}d {hours}h"
    if hours:
        return f"up {hours}h {minutes}m"
    return f"up {minutes}m"


def _payload(request: HttpRequest) -> dict:
    """Every number the page shows, raw and pre-formatted, in one dict."""
    sample = stats.collect(_settings())
    mem_percent = _percent(sample.mem_used, sample.mem_total)
    # df's Use% exactly: over used + free (root's reserve belongs to neither), rounded *up*.
    # Anything else and the page and `df -h` disagree by a point, which only invites the
    # question of which one is lying.
    usable = None if sample.disk_used is None else sample.disk_used + (sample.disk_free or 0)
    disk_percent = (
        None if not usable else math.ceil(100 * (sample.disk_used or 0) / usable)
    )

    data = asdict(sample)
    data.update(
        mem_percent=mem_percent,
        disk_percent=disk_percent,
        temp_text="—" if sample.temp_c is None else f"{sample.temp_c:.1f}°",
        temp_detail="—" if sample.load1 is None else f"load {sample.load1:.2f}",
        mem_text="—" if mem_percent is None else f"{mem_percent}%",
        mem_detail=f"{_gib(sample.mem_used)} / {_gib(sample.mem_total)} GB",
        disk_text="—" if disk_percent is None else f"{disk_percent}%",
        disk_detail=f"{_gib(sample.disk_free)} GB free",
        sessions_text=str(sample.sessions),
        sessions_detail=(
            f"{sample.sessions_in_progress} in progress"
            if sample.sessions_in_progress
            else _duration(sample.uptime_s)
        ),
        local=_is_local(request),
        volume=mixer.requested(),
        # Absolute, so "0 sessions" is self-diagnosing: the count is relative to the CWD the
        # service was started in (see WorkingDirectory in deploy/cyclops-admin.service).
        sessions_dir=str(_settings().sessions_dir.expanduser().resolve()),
    )
    return data


def dashboard(request: HttpRequest) -> HttpResponse:
    """The whole interface: four tiles and, on the kiosk only, a volume bar and a close bar."""
    return render(request, "cyclops/dashboard.html", _payload(request))


def status(request: HttpRequest) -> JsonResponse:
    """The same numbers as JSON, for the page's poll and for anything you want to script."""
    return JsonResponse(_payload(request))


@require_POST
def close_browser(request: HttpRequest) -> HttpResponse:
    """Ask the kiosk to close the browser it opened - see ``BROWSER_CLOSE_FLAG``.

    Only the kiosk's own browser has anything to close, so anything off-box is refused rather
    than left as a way for a stranger on the LAN to poke at the panel.
    """
    if not _is_local(request):
        return HttpResponseForbidden("only the kiosk can close its own browser")
    BROWSER_CLOSE_FLAG.parent.mkdir(parents=True, exist_ok=True)
    BROWSER_CLOSE_FLAG.touch()
    return HttpResponse(status=204)


@require_POST
def set_volume(request: HttpRequest) -> HttpResponse:
    """Ask the kiosk to set the speaker's volume - see :mod:`cyclops.mixer`.

    This service cannot set it itself: PrivateDevices=yes leaves it with no /dev/snd, and it
    starts at boot with no user session to reach PipeWire through. So it writes the level down
    and the kiosk applies it, exactly as with the close button.

    Loopback only, like /close, and for a second reason beyond the usual one: the panel is
    meant to be the *only* place the volume is set, and an endpoint the LAN could POST to
    would quietly make it the second.
    """
    if not _is_local(request):
        return HttpResponseForbidden("the volume is set from the panel")
    try:
        wanted = int(request.POST.get("level", ""))
    except ValueError:
        return HttpResponseBadRequest("level must be a whole percent")
    return JsonResponse({"volume": mixer.request(wanted)})  # clamped; echo what actually landed
