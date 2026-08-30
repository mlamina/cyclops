"""The page, the JSON behind it, and the two things the panel can actually change.

Both the first render and every poll go through :func:`_payload`, so the template and the
JavaScript are looking at exactly the same fields and the formatting lives in one place.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

from django.http import (
    Http404,
    HttpRequest,
    HttpResponse,
    HttpResponseBadRequest,
    HttpResponseForbidden,
    JsonResponse,
)
from django.shortcuts import render
from django.views.decorators.http import require_POST

from .. import card, mixer, stats
from ..config import (
    BROWSER_CLOSE_FLAG,
    DIAGRAM_FILE,
    DIAGRAM_SHOWN_FLAG,
    PAGE_SERVED_FLAG,
    ConfigError,
    Settings,
    load_settings,
)

LOOPBACK = {"127.0.0.1", "::1"}
GIB = 1024**3  # what df -h means by "G", so the page and the shell agree

STATIC_DIR = Path(__file__).resolve().parent / "static"
# The three vendored bundles, by the only names that will be served. A allow-list rather than a
# path check because there is no argument to be had about what a suffixed, slashed or dotted name
# resolves to if the set of legal answers is written out in full.
STATIC_FILES = {
    "joint.min.js": "text/javascript",
    "dagre.min.js": "text/javascript",
    "directed-graph.min.js": "text/javascript",
}
MAX_SVG_BYTES = 4 * 1024 * 1024  # a 40-pin pinout is ~90 KB; this is a ceiling, not a budget

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
    page = render(request, "cyclops/dashboard.html", _payload(request))
    if _is_local(request):
        _note_served()
    return page


def _note_served() -> None:
    """Leave word that a browser on this box has just been given the page - see PAGE_SERVED_FLAG.

    The kiosk starts its admin browser at boot and keeps it behind its own window, and this is
    how it learns that the browser has the page and can be uncovered. Loopback renders only: a
    laptop opening the page over the LAN must never answer for the kiosk's own browser.
    """
    try:
        PAGE_SERVED_FLAG.parent.mkdir(parents=True, exist_ok=True)
        PAGE_SERVED_FLAG.touch()
    except OSError as exc:  # a page that renders matters more than a note the kiosk can wait out
        print(f"· could not leave the page-served note ({exc})", flush=True)


def status(request: HttpRequest) -> JsonResponse:
    """The same numbers as JSON, for the page's poll and for anything you want to script."""
    return JsonResponse(_payload(request))


@require_POST
def close_browser(request: HttpRequest) -> HttpResponse:
    """Ask the kiosk to take its panel back from this page - see ``BROWSER_CLOSE_FLAG``.

    Only the kiosk's own browser is covering anything, so anything off-box is refused rather
    than left as a way for a stranger on the LAN to poke at the panel.
    """
    if not _is_local(request):
        return HttpResponseForbidden("only the kiosk can close its own browser")
    BROWSER_CLOSE_FLAG.parent.mkdir(parents=True, exist_ok=True)
    BROWSER_CLOSE_FLAG.touch()
    return HttpResponse(status=204)


# ------------------------------------------------------------------ diagrams


def _pending() -> dict | None:
    """The diagram waiting to be shown, or None - see ``DIAGRAM_FILE``.

    Never raises. A half-written file is not possible (they go through ``card.write_text``) but a
    truncated one from an older build, or none at all, both mean the same thing to the page: show
    the dashboard.
    """
    try:
        found = json.loads(DIAGRAM_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return found if isinstance(found, dict) and found.get("id") else None


def panel(request: HttpRequest) -> JsonResponse:
    """What the panel should be showing. Polled fast, so it stays one ``read`` and nothing else.

    Deliberately not folded into :func:`status`: that one collects temperatures, walks the
    sessions directory and formats a dozen strings, and this is asked several times a second.
    """
    found = _pending()
    return JsonResponse({"diagram": found["id"] if found else None})


def diagram(request: HttpRequest, ident: str) -> JsonResponse:
    """The drawing itself, by id, for the page to render.

    The id is checked against the pending one rather than used to look anything up, so there is
    no path here for a caller to name and nothing to escape out of.
    """
    found = _pending()
    if found is None or found["id"] != ident:
        raise Http404("no such diagram is waiting")
    return JsonResponse(found)


@require_POST
def diagram_shown(request: HttpRequest) -> HttpResponse:
    """The page has painted the diagram: keep the picture, and tell the kiosk it may uncover.

    Two jobs in one request on purpose. The kiosk is waiting on ``DIAGRAM_SHOWN_FLAG`` before it
    drops its window, and this is also the only moment the rendered SVG exists anywhere - the
    panel draws it, so the panel is the only thing that can hand it back for the card.

    The destination comes from our own pending file and never from the request. The body is one
    anonymous blob of bytes; letting it choose where those bytes land would make this the one
    endpoint on the box worth attacking.
    """
    if not _is_local(request):
        return HttpResponseForbidden("only the kiosk's own browser paints the panel")
    found = _pending()
    if found is None:
        return HttpResponseBadRequest("no diagram is waiting")

    svg = request.body[:MAX_SVG_BYTES]
    target = found.get("svg")
    if svg and target:
        try:
            card.write_bytes(Path(target), svg)
        except OSError as exc:  # the drawing is on the panel either way, which is the point
            print(f"· could not keep the diagram picture ({exc})", flush=True)
    try:
        DIAGRAM_SHOWN_FLAG.parent.mkdir(parents=True, exist_ok=True)
        DIAGRAM_SHOWN_FLAG.touch()
    except OSError as exc:
        print(f"· could not leave the diagram-shown note ({exc})", flush=True)
    return HttpResponse(status=204)


def static_file(request: HttpRequest, name: str) -> HttpResponse:
    """Serve one of the vendored bundles - see ``static/NOTICE.md``.

    There is no ``staticfiles`` app here and no INSTALLED_APPS to add one to, which for four
    files is the smaller thing rather than the missing thing. They never change between deploys,
    so they are handed out with a long cache lifetime and read straight off the card.
    """
    kind = STATIC_FILES.get(name)
    if kind is None:
        raise Http404("no such file")
    try:
        body = (STATIC_DIR / name).read_bytes()
    except OSError as exc:
        raise Http404("no such file") from exc
    response = HttpResponse(body, content_type=kind)
    response["Cache-Control"] = "public, max-age=31536000, immutable"
    return response


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
