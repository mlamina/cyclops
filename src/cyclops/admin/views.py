"""The page, the JSON behind it, and the two things the panel can actually change.

Both the first render and every poll go through :func:`_payload`, so the template and the
JavaScript are looking at exactly the same fields and the formatting lives in one place.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict
from pathlib import Path

from django.http import (
    Http404,
    HttpRequest,
    HttpResponse,
    HttpResponseBadRequest,
    HttpResponseForbidden,
    JsonResponse,
    StreamingHttpResponse,
)
from django.shortcuts import render
from django.views.decorators.http import require_POST

from .. import about, barge, card, filming, library, mixer, shelf, stats
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

# What may come out of a session folder, and as what. An allow-list by suffix for the same reason
# STATIC_FILES is one: the set of legal answers is short enough to write down, and writing it down
# is the end of every argument about what some other name might resolve to.
MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
}
# The only sub-directories of a session a browser is ever given. video.mp4 sits in the root, which
# is the third case and the reason this is a set of names rather than a single one.
MEDIA_DIRS = frozenset({card.PHOTOS, card.DIAGRAMS})
RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")
CHUNK = 64 * 1024

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
        # Each bar's own 0-100. Three of these are percentages already; a temperature is not, and
        # the scale it goes on is the board's, so stats.py decides it rather than this page.
        mem_percent=mem_percent,
        disk_percent=disk_percent,
        temp_percent=stats.temp_percent(sample.temp_c),
        # The word beside each bar says what the bar cannot. For memory and disk that is the
        # figure in bytes - the bar is already the percentage, and printing it twice says nothing
        # the second time. cpu_percent itself rides along in asdict(sample).
        temp_text="—" if sample.temp_c is None else f"{sample.temp_c:.1f}°",
        cpu_text="—" if sample.cpu_percent is None else f"{sample.cpu_percent}%",
        mem_text=f"{_gib(sample.mem_used)} / {_gib(sample.mem_total)} GB",
        disk_text=f"{_gib(sample.disk_free)} GB free",
        local=_is_local(request),
        volume=mixer.requested(),
        barge_in=barge.enabled(_settings()),
        record_screen=filming.on_screen(_settings()),
        # Absolute, so "0 sessions" is self-diagnosing: the count is relative to the CWD the
        # service was started in (see WorkingDirectory in deploy/cyclops-admin.service).
        sessions_dir=str(_settings().sessions_dir.expanduser().resolve()),
    )
    return data


def dashboard(request: HttpRequest) -> HttpResponse:
    """The whole interface: four readings and, on the kiosk only, the settings and a way out."""
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


@require_POST
def set_barge_in(request: HttpRequest) -> HttpResponse:
    """Say whether Cyclops may be talked over - see :mod:`cyclops.barge`.

    Loopback only, like the volume, and for the plainer of that route's two reasons: everything
    this page shows the LAN is a copy of what is on the card, and the screen that can change how
    the box behaves is the one bolted to it.

    The note is all this does. The kiosk reads it a couple of times a second and hands it to the
    session holding the microphone, which is the only thing that can actually stop listening.
    """
    if not _is_local(request):
        return HttpResponseForbidden("barge-in is set from the panel")
    wanted = request.POST.get("on", "")
    if wanted not in {"0", "1"}:
        return HttpResponseBadRequest("on must be 0 or 1")
    return JsonResponse({"barge_in": barge.request(wanted == "1")})


@require_POST
def set_record_source(request: HttpRequest) -> HttpResponse:
    """Say what a session's video should be of - the panel, or the camera alone.

    Loopback only, like the volume and the interrupt switch, and for the same reason as the
    latter: everything this page shows the LAN is a copy of what is already on the card, and
    the screen that can change how the box behaves is the one bolted to it.

    The note is all this does, and unlike barge-in nobody reads it until the next tap on WAKE
    UP: a recording already running was given its source when its encoder was opened, and
    cannot be handed another one halfway through. See :mod:`cyclops.filming`.
    """
    if not _is_local(request):
        return HttpResponseForbidden("what is recorded is set from the panel")
    wanted = request.POST.get("source", "")
    if wanted not in {filming.CAMERA, filming.SCREEN}:
        return HttpResponseBadRequest("source must be 'camera' or 'screen'")
    return JsonResponse({"record_source": filming.request(wanted)})


# ------------------------------------------------------------------ what is on the card


def _entries() -> list[library.Entry]:
    return library.entries(_settings().sessions_dir)


def sessions(request: HttpRequest) -> JsonResponse:
    """Every session on the card, newest first, and what it knows about you over them.

    The facts ride along with the listing rather than with the five-second status poll for the
    plainest reason: they change once a session, and this is the one request the screen that
    shows them already makes. Fetched when a view opens, not on a poll.
    """
    return JsonResponse(
        {"sessions": library.as_dicts(_entries()), "facts": about.read(_settings())}
    )


def session(request: HttpRequest, name: str) -> JsonResponse:
    """One session, without its transcript - which the panel never asks for."""
    found = library.entry(_settings().sessions_dir, name)
    if found is None:
        raise Http404("no such session")
    return JsonResponse(asdict(found))


def session_records(request: HttpRequest, name: str) -> JsonResponse:
    """One session's transcript. A separate route because the narrow layout does not show one.

    Splitting it off the entry above is the whole of "everything on desktop, only the video on the
    panel": the panel does not hide the transcript, it never asks for it, and a session that ran
    for an hour costs it nothing.
    """
    if library.entry(_settings().sessions_dir, name) is None:
        raise Http404("no such session")
    return JsonResponse({"records": library.records(_settings().sessions_dir, name)})


def media_stream(request: HttpRequest) -> JsonResponse:
    """Every picture and every drawing on the card, newest first, as one run."""
    return JsonResponse({"items": library.as_dicts(library.stream(_settings().sessions_dir))})


def _media_file(name: str, relative: str) -> tuple[Path, str]:
    """One file inside one session, or 404 - and never a file outside the sessions directory.

    Two resolutions, each ending in a comparison rather than in an inspection of the string. The
    folder must be a direct child of the sessions directory (``library.resolve``), and the file
    must sit either in that folder or in one of the two sub-directories a session is allowed to
    have. ``..``, an absolute name, a symlink out of the tree and a nested path all fail the same
    comparison, so there is no ordering of checks to get wrong.
    """
    folder = library.resolve(_settings().sessions_dir, name)
    if folder is None:
        raise Http404("no such session")
    try:
        found = (folder / relative).resolve()
    except OSError as exc:
        raise Http404("no such file") from exc
    parent = found.parent
    if parent != folder and not (parent.parent == folder and parent.name in MEDIA_DIRS):
        raise Http404("no such file")
    kind = MEDIA_TYPES.get(found.suffix.lower())
    if kind is None or not found.is_file():
        raise Http404("no such file")
    return found, kind


def _chunks(path: Path, start: int, length: int):
    """The bytes, a block at a time, so a 27 MB recording is never a 27 MB string."""
    with path.open("rb") as handle:
        handle.seek(start)
        left = length
        while left > 0:
            block = handle.read(min(CHUNK, left))
            if not block:
                break
            left -= len(block)
            yield block


def media(request: HttpRequest, name: str, relative: str) -> HttpResponse:
    """A recording, a photo or a drawing out of one session."""
    path, kind = _media_file(name, relative)
    return _serve(request, path, kind)


def _serve(request: HttpRequest, path: Path, kind: str) -> HttpResponse:
    """One file down the wire, with byte ranges - which video is not optional about.

    Django serves no ranges of its own (there is no ``HTTP_RANGE`` anywhere in it), and without a
    206 Safari will not start an ``<video>`` at all and nothing anywhere can seek in one. It also
    keeps this off gunicorn's 30 s worker timeout: scrubbing becomes a run of short requests
    rather than one long transfer held open by a paused player.

    Which file it is has already been decided by the caller. That split is the whole security
    story of this function: it never sees a name anybody sent, only a path some resolver already
    proved is inside the root it belongs to.
    """
    size = path.stat().st_size

    start, end = 0, size - 1
    partial = False
    asked = RANGE.match(request.META.get("HTTP_RANGE", "").strip())
    if asked:
        first, last = asked.groups()
        if first:
            start = int(first)
            end = min(int(last), size - 1) if last else size - 1
        elif last:  # bytes=-500: the *final* 500, which is a suffix length and not an offset
            start = max(0, size - int(last))
        else:
            asked = None
        if asked:
            if start > end or start >= size:
                refused = HttpResponse(status=416)
                refused["Content-Range"] = f"bytes */{size}"
                return refused
            partial = True

    length = end - start + 1
    # A HEAD asks what the file is, not for it. Django does not strip the body for one and
    # gunicorn drops it at the socket with a warning per request, having made us read 27 MB off
    # the card first - which is the whole cost of the request and none of the answer.
    body = () if request.method == "HEAD" else _chunks(path, start, length)
    response = StreamingHttpResponse(body, content_type=kind, status=206 if partial else 200)
    response["Content-Length"] = str(length)
    response["Accept-Ranges"] = "bytes"
    if partial:
        response["Content-Range"] = f"bytes {start}-{end}/{size}"
    # Not `immutable` like the vendored bundles: a diagram's picture can land after its spec, and
    # a session being recorded right now grows. An hour is long enough to scrub a video without
    # re-fetching it and short enough that nothing goes stale for a day.
    response["Cache-Control"] = "private, max-age=3600"
    response["X-Content-Type-Options"] = "nosniff"
    if kind == "image/svg+xml":
        # Our own panel drew it, from our own spec, on a private LAN - and it is still the one
        # thing here that a browser would happily execute. It is rendered through <img>, which
        # never runs script in one; this is the belt to that's braces.
        response["Content-Security-Policy"] = "default-src 'none'; style-src 'unsafe-inline'"
    return response


# ------------------------------------------------------------------ what is in the projects


def _project(name: str) -> Path:
    """The project folder called ``name``, or 404. Never anything outside ``projects_dir``."""
    folder = shelf.resolve(_settings().projects_dir, name)
    if folder is None:
        raise Http404("no such project")
    return folder


def projects(request: HttpRequest) -> JsonResponse:
    """Every project on the card, most recently worked on first."""
    return JsonResponse({"projects": shelf.projects(_settings().projects_dir)})


def project_files(request: HttpRequest, name: str) -> JsonResponse:
    """One directory inside one project. ``?path=`` is relative to the project, "" for its root."""
    found = shelf.listing(_project(name), request.GET.get("path", ""))
    if found is None:
        raise Http404("no such folder")
    return JsonResponse(found)


def project_file(request: HttpRequest, name: str) -> JsonResponse:
    """One file inside one project, in whatever shape it is worth reading in."""
    relative = request.GET.get("path", "")
    found = shelf.view(name, _project(name), relative)
    if found is None:
        raise Http404("no such file")
    return JsonResponse(found)


def project_media(request: HttpRequest, name: str, relative: str) -> HttpResponse:
    """A photo, a drawing or a recording out of a project folder.

    The session route's ``MEDIA_DIRS`` allow-list deliberately does not apply here. A session has
    exactly two sub-directories and both are ours; a project folder is a place a person keeps
    their own things, and telling them which folders of their own they may open would be a
    strange thing for a file browser to do. Containment does the work instead - `shelf.inside`
    resolves first and then requires the result to be under the project - and the suffix list
    still decides what may be handed over as bytes.
    """
    folder = _project(name)
    found = shelf.inside(folder, relative)
    if found is None or not found.is_file():
        raise Http404("no such file")
    kind = shelf.MEDIA_TYPES.get(found.suffix.lower())
    if kind is None:
        raise Http404("no such file")
    return _serve(request, found, kind)
