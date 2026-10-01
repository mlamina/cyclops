"""The page, the JSON behind it, and the two things the panel can actually change.

Both the first render and every poll go through :func:`_payload`, so the template and the
JavaScript are looking at exactly the same fields and the formatting lives in one place.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
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

from .. import (
    barge,
    card,
    cut,
    library,
    manuals,
    mixer,
    shelf,
    stats,
    steady,
    tasks,
    voice,
)
from .. import session as sessionlog
from ..config import (
    BROWSER_CLOSE_FLAG,
    COMPANION_PORT,
    PAGE_ALIVE_FLAG,
    PAGE_SCREEN_FILE,
    PAGE_SERVED_FLAG,
    PANEL_FILE,
    PANEL_PAINTED_FLAG,
    PICTURE_UP_FLAG,
    SAY_VOICE_FILE,
    ConfigError,
    Settings,
    load_settings,
)
from ..projects import store

LOOPBACK = {"127.0.0.1", "::1"}
GIB = 1024**3  # what df -h means by "G", so the page and the shell agree

STATIC_DIR = Path(__file__).resolve().parent / "static"
# Everything that will ever be served from static/, by the only names that will be served. An
# allow-list rather than a path check because there is no argument to be had about what a
# suffixed, slashed or dotted name resolves to if the set of legal answers is written out in
# full.
#
# There was a second dict beside this one until 2026-09-04, holding JointJS, dagre and a graph
# layout - 527 KB of vendored bundles that wanted the opposite caching from ours, because they
# changed twice a year and ours change hourly. They were here to lay out a diagram in the
# browser. Diagrams are drawn as images now, so the panel parses none of it.
#
# charset on all of them: twenty-odd lines of the JS carry an em-dash or a middot, and text/css
# is not optional at all - a standards-mode document *rejects* a stylesheet served as anything
# else, and says so in one console line with an unstyled page as the only other symptom.
OURS = {
    "base.css": "text/css; charset=utf-8",
    "system.css": "text/css; charset=utf-8",
    "panel.css": "text/css; charset=utf-8",
    "views.css": "text/css; charset=utf-8",
    "lan.css": "text/css; charset=utf-8",
    "status.js": "text/javascript; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    "panel.js": "text/javascript; charset=utf-8",
    # The companion's two streams - the picture and the voice - which are not this service's to
    # serve and are only this service's to point at. See cyclops.companion.
    "stream.js": "text/javascript; charset=utf-8",
    # The host half of the MCP Apps handshake with the sketch renderer, which this service
    # does serve (see :func:`renderer`) but whose frames it never sees. See cyclops.sketch.
    "sketch.js": "text/javascript; charset=utf-8",
    # The boot mark, for the one screen that has nothing else to show: a companion waiting for a
    # session to start. Derived from assets/eye.png, which is the same mark drawn white on black -
    # its luminance moved into the alpha channel, so CSS can use it as a mask and paint it in the
    # page's own green rather than in whatever colour the file happened to be.
    "eye.png": "image/png",
}
STATIC_FILES = OURS


def _asset_version() -> str:
    """One short digest over our own CSS and JS, read once at import - the page's ``?v=``.

    The kiosk starts Chromium against a profile on the card that no deploy clears
    (``kiosk.py``), and ``--kiosk`` leaves nobody an address bar or a reload button. So an asset
    handed out ``immutable`` at a fixed URL is handed out *for ever*, and the only cure is ssh.
    Versioning by URL instead means an edited file is a different file as far as the browser is
    concerned, while a deploy that changed nothing reuses everything already parsed on the panel
    - no revalidation request, on a box where CPU is the scarce thing.

    Contents and not mtimes: ``rsync -a`` carries whatever the laptop's clock said, to the
    second, and two edits inside one second is a Tuesday.

    Read at import, so the guarantee is *restart implies fresh* - which is the same contract
    ``DEBUG = False`` already imposes on the template, and which ``deploy/push.sh`` honours by
    restarting this service on every push.
    """
    digest = hashlib.sha256()
    for name in sorted(OURS):
        try:
            digest.update((STATIC_DIR / name).read_bytes())
        except OSError:  # a missing file is a 404 the page shouts about, not a dead service
            digest.update(name.encode())
    return digest.hexdigest()[:8]


ASSET_VERSION = _asset_version()
# The same shape of number for what may be dropped onto the page from a laptop. A datasheet is
# kilobytes and a phone photo is single-digit megabytes; this is a ceiling, not a budget, and it
# is here so that a mis-drag of something enormous is refused rather than written to the card.
MAX_UPLOAD_BYTES = 128 * 1024 * 1024
# What may come out of a session folder, and as what. An allow-list by suffix for the same reason
# STATIC_FILES is one: the set of legal answers is short enough to write down, and writing it down
# is the end of every argument about what some other name might resolve to.
MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}
# The only sub-directories of a session a browser is ever given. video.mp4 sits in the root, which
# is the third case and the reason this is a set of names rather than a single one.
MEDIA_DIRS = frozenset({card.PHOTOS, card.CLIPS})
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
        steady=steady.enabled(),
        voice=voice.chosen(_settings()),
        # Absolute, so "0 sessions" is self-diagnosing: the count is relative to the CWD the
        # service was started in (see WorkingDirectory in deploy/cyclops-admin.service).
        sessions_dir=str(_settings().sessions_dir.expanduser().resolve()),
        # What is going on in the background right now, or "" when nothing is - see
        # :mod:`cyclops.tasks`. One sentence rather than the list: this rides on the five-second
        # poll every client already runs, and the header has room for a line and not a table.
        # `curl cyclops.local/api/status | jq .task` is the same answer from a laptop, which is
        # most of why it is here rather than only on the panel's caption.
        task=tasks.line(),
        # Whether the Make a video button is drawn at all. A switched-off feature should leave no
        # control behind that answers 403 when pressed.
        cut=_settings().cut,
    )
    return data


def dashboard(request: HttpRequest) -> HttpResponse:
    """The whole interface: four readings and, on the kiosk only, the settings and a way out."""
    # ASSET_VERSION rides beside the payload rather than inside it: _payload() is also
    # /api/status, whose keys the page's [data-field] loop iterates and the README documents as
    # something you can curl. A cache-busting token has no business in either.
    # `voices` rides here rather than in _payload for the same reason ASSET_VERSION does: the
    # payload is also /api/status, whose keys the README documents as something you can curl, and
    # the ten names never change between two polls of it. The page needs the whole list once, to
    # step through without asking again.
    page = render(
        request,
        "cyclops/dashboard.html",
        _payload(request)
        | {
            "v": ASSET_VERSION,
            "voices": list(voice.VOICES),
            # Where the live picture and the live voice come from. Beside `v` for the reason
            # written above it: the payload is also /api/status, and the port of another
            # process is not one of the four readings anybody curls that for.
            "companion_port": COMPANION_PORT,
        },
    )
    # The page is the one thing that must never be stale, because it is what names the versions
    # of everything else - a cached copy would go on asking for last week's stylesheet for ever.
    # It is fetched once per browser start, so this costs nothing.
    page["Cache-Control"] = "no-store"
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

    Two callers, and they are asking for different things with the same note. The kiosk's own
    browser is asking to close the page it is showing. A companion on the LAN is asking to put
    away a picture, which it can only do while there is one to put away - anything else off-box
    is refused rather than left as a way for a stranger on the LAN to poke at the panel.
    """
    # A companion on the LAN answers for one gesture and one only: putting away a picture that
    # is on the glass right now. That is the same gesture as pressing the panel, and it lands in
    # the same place, so there is nothing here the person standing at the box could not do.
    #
    # PICTURE_UP_FLAG and not _pending(): an offer can outlive the moment anybody could see it -
    # ``Kiosk.show_picture`` refuses one while the admin page has the panel and leaves the
    # payload where it was - and honouring a note in that state reaches ``_sync_stranded``, which
    # rebuilds the window over a panel that was already ours. The kiosk touches this one inside
    # the latch that reveals a picture, so it means what this gate needs it to mean.
    if not _is_local(request) and not PICTURE_UP_FLAG.exists():
        return HttpResponseForbidden("only the kiosk can close its own browser")
    BROWSER_CLOSE_FLAG.parent.mkdir(parents=True, exist_ok=True)
    BROWSER_CLOSE_FLAG.touch()
    return HttpResponse(status=204)


# ------------------------------------------------------------------ the panel


# The last payload parsed, keyed on the file it came out of. Per gunicorn worker, and it holds
# one picture at most, because there is only ever one offer.
_pending_at: tuple[int, int] | None = None
_pending_was: dict | None = None


def _pending() -> dict | None:
    """The picture waiting to be shown, or None - see ``PANEL_FILE``.

    Never raises. A half-written file is not possible (they go through ``card.write_text``) but a
    truncated one from an older build, or none at all, both mean the same thing to the page: show
    the dashboard.

    Memoised on the file's own ``(mtime_ns, size)``, which is what :func:`panel` is asked for
    two and a half times a second by the kiosk and again by every companion on the LAN. Without
    it each of those polls reads and JSON-parses the *whole* picture - a megabyte of base64 for a
    1024px JPEG - to look at one twelve-character id. ``card.write_text`` renames a fresh inode
    into place, so a new offer can never wear the old key.
    """
    global _pending_at, _pending_was
    try:
        stat = PANEL_FILE.stat()
        key = (stat.st_mtime_ns, stat.st_size)
        if key != _pending_at:
            found = json.loads(PANEL_FILE.read_text(encoding="utf-8"))
            _pending_was = found if isinstance(found, dict) and found.get("id") else None
            _pending_at = key
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        _pending_at, _pending_was = None, None
    return _pending_was


def _note_alive() -> None:
    """Leave word that the page is *running*, not merely served - see PAGE_ALIVE_FLAG.

    Written from the fast poll rather than from the render because that is the difference being
    claimed: bytes handed over prove Django answered, and a poll arriving proves a renderer
    parsed them and is executing script. The kiosk waits on this before it opens its own window
    over the browser, so that Chromium has finished raising itself by the time anything is
    covering it.

    Only when the note is missing, which the kiosk arranges by deleting it before it spawns.
    That keeps the steady-state cost of a poll asked several times a second at one ``exists``,
    and means exactly one write per browser start rather than 150 an hour onto an SD card.
    """
    try:
        if PAGE_ALIVE_FLAG.exists():
            return
        PAGE_ALIVE_FLAG.parent.mkdir(parents=True, exist_ok=True)
        PAGE_ALIVE_FLAG.touch()
    except OSError as exc:  # the panel poll matters more than the note; the kiosk times out
        print(f"· could not leave the page-alive note ({exc})", flush=True)


def panel(request: HttpRequest) -> JsonResponse:
    """What the panel should be showing. Polled fast, so it stays one ``read`` and nothing else.

    Deliberately not folded into :func:`status`: that one collects temperatures, walks the
    sessions directory and formats a dozen strings, and this is asked several times a second.
    """
    if _is_local(request):  # a laptop on the LAN must not answer for the kiosk's own browser
        _note_alive()
    found = _pending()
    return JsonResponse({"picture": found["id"] if found else None, "screen": _screen()})


def _screen() -> str | None:
    """Which screen the panel wants this page on, or None if it has not said - PAGE_SCREEN_FILE.

    An ``exists`` first and a read only when there is something to read, the way :func:`_note_alive`
    keeps its own cost down: the note is written for the half-second either side of a reveal and is
    absent the rest of the day, and this is asked two and a half times a second for as long as the
    browser is warm, which is all of it.
    """
    try:
        if not PAGE_SCREEN_FILE.exists():
            return None
        return PAGE_SCREEN_FILE.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def picture(request: HttpRequest, ident: str) -> JsonResponse:
    """The drawing itself, by id, for the page to render.

    The id is checked against the pending one rather than used to look anything up, so there is
    no path here for a caller to name and nothing to escape out of.
    """
    found = _pending()
    if found is None or found["id"] != ident:
        raise Http404("no such picture is waiting")
    return JsonResponse(found)


@require_POST
def picture_painted(request: HttpRequest) -> HttpResponse:
    """The page has painted the picture: tell the kiosk it may uncover.

    The kiosk is waiting on ``PANEL_PAINTED_FLAG`` before it drops its window, and this is the
    page saying it has something to uncover onto. The body is empty. It used to carry the page's
    own picture of a scratchpad, for a recording that was rebuilt from what the page was handed;
    the recording is taken off the glass now, and sees the scratchpad the way anybody does.
    """
    if not _is_local(request):
        return HttpResponseForbidden("only the kiosk's own browser paints the panel")
    if _pending() is None:
        return HttpResponseBadRequest("nothing is waiting for the panel")
    try:
        PANEL_PAINTED_FLAG.parent.mkdir(parents=True, exist_ok=True)
        PANEL_PAINTED_FLAG.touch()
    except OSError as exc:
        print(f"· could not leave the panel-painted note ({exc})", flush=True)
    return HttpResponse(status=204)


def renderer(request: HttpRequest) -> HttpResponse:
    """Prefab's renderer, as one self-contained page. The frame a sketch is drawn in.

    Served from here rather than from jsDelivr, which is Prefab's default. Six megabytes off the
    card on the first load of a warm browser that then keeps it for the life of the profile beats
    a network round trip at the one moment somebody is waiting - and a panel in a workshop should
    not need the internet to draw a number on itself.

    ``mode="bundled"`` is what makes that true: the CDN stub would still fetch the renderer, and
    its lazy chunks - charts, icons - would each come down the first time a program used
    one, mid-sentence. Bundled inlines the lot.

    It is read on every request rather than held in memory. This is asked for once per browser
    profile, and six megabytes of resident set in each of two gunicorn workers, for ever, to save
    a read that the page cache has already made free, is the wrong way round.
    """
    from prefab_ui import renderer as prefab_renderer

    response = HttpResponse(
        prefab_renderer.get_renderer_html(mode="bundled"),
        content_type="text/html; charset=utf-8",
    )
    # Versioned by the package and not by our digest, so it is not in OURS and cannot ride
    # ASSET_VERSION. A deploy that upgrades prefab-ui has to clear the kiosk profile or bump
    # this URL -
    # which is the trade for not parsing six megabytes of JavaScript on every reveal.
    response["Cache-Control"] = "public, max-age=86400"
    return response


def static_file(request: HttpRequest, name: str) -> HttpResponse:
    """Serve one of the files in ``static/`` - our own CSS and JS, and the vendored bundles.

    There is no ``staticfiles`` app here and no INSTALLED_APPS to add one to, which for eleven
    files is the smaller thing rather than the missing thing.

    Everything is handed out ``immutable``, ours included: our files are versioned in the query
    string instead (see :func:`_asset_version`), and Django resolves the URL before the ``?``, so
    ``/static/base.css?v=a1b2c3d4`` arrives here as ``base.css`` and the flat allow-list above
    never has to think about it.
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
def set_steady(request: HttpRequest) -> HttpResponse:
    """Say whether the live picture is held still - see :mod:`cyclops.steady`.

    Loopback only, like the other switches. The note is all this does; the camera's reader looks
    at it twice a second, so the panel changes while you are still looking at the switch.
    """
    if not _is_local(request):
        return HttpResponseForbidden("stabilizing is set from the panel")
    wanted = request.POST.get("on", "")
    if wanted not in {"0", "1"}:
        return HttpResponseBadRequest("on must be 0 or 1")
    return JsonResponse({"steady": steady.request(wanted == "1")})


@require_POST
def set_voice(request: HttpRequest) -> HttpResponse:
    """Say which of the ten voices should answer, and sound it.

    Loopback only, like the three settings above it. One route for both halves of the stepper
    because they are one gesture: stepping to a voice and hearing it are the same act, and a
    control that changed the setting silently would be a list of names again.

    Two notes, because they are two different kinds of thing. The voice is a setting and lands on
    the next session - the Realtime API takes it in the `session.update` that opens a websocket
    and there is no event that changes it afterwards. The other is a request, consumed and
    deleted by the kiosk, which is the only process here with a speaker (see the volume above,
    and :meth:`cyclops.kiosk.Kiosk._sync_voice`).
    """
    if not _is_local(request):
        return HttpResponseForbidden("the voice is set from the panel")
    wanted = request.POST.get("name", "")
    if wanted not in voice.NAMES:
        return HttpResponseBadRequest("name must be one of the ten voices")
    chosen = voice.request(wanted)
    try:
        SAY_VOICE_FILE.parent.mkdir(parents=True, exist_ok=True)
        SAY_VOICE_FILE.write_text(f"{chosen}\n")
    except OSError as exc:  # a voice that is set silently beats one that is not set at all
        print(f"· could not ask for a voice sample ({exc})", flush=True)
    return JsonResponse({"voice": chosen})


# ------------------------------------------------------------------ what is on the card


def _entries() -> list[library.Entry]:
    return library.entries(_settings().sessions_dir)


def sessions(request: HttpRequest) -> JsonResponse:
    """Every session on the card, newest first. Fetched when a view opens, not on a poll."""
    return JsonResponse({"sessions": library.as_dicts(_entries())})


def session(request: HttpRequest, name: str) -> JsonResponse:
    """One session, without its transcript - which the panel never asks for."""
    settings = _settings()
    found = library.entry(settings.sessions_dir, name)
    if found is None:
        raise Http404("no such session")
    # Hung on here rather than put on library.Entry, because cyclops.cut reads library and a
    # field on Entry would make that a circle. It is also a stat and, where there is a video, one
    # small read - which is affordable for one session and would not be for a listing of seventy.
    folder = library.resolve(settings.sessions_dir, name)
    extra: dict = {"cut": "", "cut_made": 0, "cut_total": 0, "short": "", "short_ranges": []}
    if folder is not None:
        got = cut.progress(folder)
        extra |= {"cut": got.state, "cut_made": got.made, "cut_total": got.total}
        # The shortened video is what the session screen plays, and its segments are how a
        # transcript line's clock - the recording's - finds its place in it.
        plan = cut.read_plan(folder)
        if plan and plan.clips and card.written(cut.clip_path(folder, 1)):
            extra["short"] = f"/media/{folder.name}/{card.CLIPS}/1.mp4"
            extra["short_ranges"] = [list(r) for r in plan.clips[0].ranges]
    return JsonResponse(asdict(found) | extra)


def highlights(request: HttpRequest) -> JsonResponse:
    """Every finished clip, newest first, and one line about what has been looked at.

    Finished only, unlike the listing this replaced: nobody arrives here having just pressed a
    button, so a half-made row is noise. The counts are what keep an empty reel from being a
    mystery - "nobody has looked yet" and "everything was looked at and none of it was
    interesting" are different sentences and the footer says which one it is.
    """
    made, seen = cut.clips(_settings().sessions_dir)
    return JsonResponse({"clips": [asdict(one) for one in made], "seen": seen})


@require_POST
def find_clips(request: HttpRequest, name: str) -> HttpResponse:
    """Throw away what was decided for one session, so the next sweep looks at it again.

    Clips are found by themselves; this is the "you judged that one wrong" button rather than
    the way work is normally started. It answers the LAN as well as the panel, for
    ``project_upload``'s reason: pressing it from the laptop you are watching the reel on is the
    whole point, and a rule that let only the Pi ask the Pi would leave nothing behind. What
    holds the line instead is ``library.resolve``, which proves the folder is a direct child of
    the sessions directory before a byte is touched - the same containment every read on this
    page already runs on.

    Never a redirect and never a bare 202: the button repaints from this response.
    """
    settings = _settings()
    if not settings.cut:
        return HttpResponseForbidden("clipping is switched off")
    folder = library.resolve(settings.sessions_dir, name)
    if folder is None:
        raise Http404("no such session")
    if not card.written(folder / card.VIDEO):
        return HttpResponseBadRequest("that session has no recording to clip")
    if card.locked(folder):
        return HttpResponseBadRequest("that session is still recording")
    try:
        cut.forget(folder)
    except OSError as exc:
        return HttpResponseBadRequest(f"could not ask for that ({exc})")
    return JsonResponse({"name": name, "cut": cut.state(folder)})


@require_POST
def delete_session(request: HttpRequest, name: str) -> HttpResponse:
    """Delete one session - its log, its pages, its video and every picture in it. No undo.

    Written beside :func:`find_clips` and in its shape, with one difference that matters: this
    answers the panel only. The page has no login, and a one-tap irreversible delete from
    anything on the LAN is the wrong side of that line, so a laptop gets 403 - and, because
    ``body.kiosk`` is what draws the gesture, never sees anything to press in the first place.

    Answers with the fresh list so the row can leave without a reload.
    """
    if not _is_local(request):
        return HttpResponseForbidden("only the panel can delete a session")
    settings = _settings()
    folder = library.resolve(settings.sessions_dir, name)
    if folder is None:
        raise Http404("no such session")
    if card.locked(folder):
        return HttpResponseBadRequest("that session is still recording")
    with cut.kept_off(folder) as free:
        if not free:
            return HttpResponseBadRequest("a clip is being made from it - try in a minute")
        try:
            sessionlog.erase(folder)
        except OSError as exc:
            return HttpResponseBadRequest(f"could not delete it ({exc})")
    return JsonResponse({"sessions": library.as_dicts(_entries())})


class _Pipe:
    """A write-only file that hands back whatever was written since the last ``take``.

    zipfile writes to anything with ``write`` and copes with one it cannot seek in, so this is
    the whole of streaming an archive: nothing is built on the card and nothing is held in memory
    beyond one chunk of one file.
    """

    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    def write(self, data: bytes) -> int:
        self.chunks.append(bytes(data))
        return len(data)

    def flush(self) -> None:
        pass

    def take(self) -> bytes:
        out, self.chunks = b"".join(self.chunks), []
        return out


def _zipped(folder: Path):
    pipe = _Pipe()
    # Stored, not deflated: the bulk is an mp4 and jpgs, which do not shrink, and the Pi is hot.
    with zipfile.ZipFile(pipe, "w", zipfile.ZIP_STORED) as archive:
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            with archive.open(f"{folder.name}/{path.relative_to(folder)}", "w") as entry:
                with path.open("rb") as src:
                    while chunk := src.read(1 << 20):
                        entry.write(chunk)
                        yield pipe.take()
            yield pipe.take()
    yield pipe.take()


def download_session(request: HttpRequest, name: str) -> HttpResponse:
    """The whole session folder as one zip - video, transcript, pictures, clips."""
    folder = library.resolve(_settings().sessions_dir, name)
    if folder is None:
        raise Http404("no such session")
    if card.locked(folder):
        return HttpResponseBadRequest("that session is still recording")
    response = StreamingHttpResponse(_zipped(folder), content_type="application/zip")
    response["Content-Disposition"] = f'attachment; filename="{folder.name}.zip"'
    return response


def session_records(request: HttpRequest, name: str) -> JsonResponse:
    """One session's transcript. A separate route because the narrow layout does not show one.

    Splitting it off the entry above is the whole of "everything on desktop, only the video on the
    panel": the panel does not hide the transcript, it never asks for it, and a session that ran
    for an hour costs it nothing.
    """
    if library.entry(_settings().sessions_dir, name) is None:
        raise Http404("no such session")
    return JsonResponse({"records": library.records(_settings().sessions_dir, name)})


def live(request: HttpRequest) -> JsonResponse:
    """The conversation happening right now, from ``since`` onwards - companion mode's feed.

    This is the one route here that is polled while somebody watches, so it is built to answer
    almost nothing most of the time. ``name`` is what the caller believes is running and
    ``since`` how many lines of it they already hold; when the two agree, the answer is the
    handful of lines that have landed since. When they do not - a session ended, another started,
    or a log was rewritten under us - the whole thing comes back and the page starts again.

    Reading a session that is still being written is safe by construction rather than by luck.
    ``SessionLog`` appends and flushes each record as it lands and never rewrites one, and
    ``card.read_log`` drops a half-written trailing line, so ``n`` only ever grows and an index
    into it stays pointing at the same record.

    ``library.records`` directly and not :func:`session_records`, which asks ``library.entry``
    first: that is a second full read of the log plus a stat per photo, on the one folder the
    entry cache deliberately never keeps.
    """
    sessions_dir = _settings().sessions_dir
    name = library.live(sessions_dir)
    if name is None:
        return JsonResponse({"name": None, "n": 0, "records": []})
    found = library.records(sessions_dir, name)
    if not found and library.resolve(sessions_dir, name) is None:
        # It ended and was renamed between the two calls above. Say nothing is running rather
        # than describing an empty session the page would read as a fresh one starting.
        return JsonResponse({"name": None, "n": 0, "records": []})
    since = 0
    if request.GET.get("name") == name:
        try:
            since = max(0, min(len(found), int(request.GET.get("since") or 0)))
        except ValueError:
            since = 0
    return JsonResponse({"name": name, "n": len(found), "records": found[since:]})


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
    # Not `immutable`: a session being recorded right now grows. An hour is long enough to scrub
    # a video without re-fetching it and short enough that nothing goes stale for a day.
    response["Cache-Control"] = "private, max-age=3600"
    response["X-Content-Type-Options"] = "nosniff"
    if kind == "image/svg+xml":
        # Only reachable through the project file browser now - somebody's own dropped file, on a
        # private LAN, and still the one thing here a browser would happily execute. It is
        # rendered through <img>, which never runs script in one; this is the belt to that's
        # braces.
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


def project_pictures(request: HttpRequest, name: str) -> JsonResponse:
    """Every picture in one project, newest first - the PHOTOS section of its screen."""
    return JsonResponse({"items": shelf.pictures(name, _project(name))})


def project_youtube(request: HttpRequest, name: str) -> JsonResponse:
    """Every video reference filed onto one project - the YOUTUBE section of its screen."""
    return JsonResponse({"items": shelf.videos(name, _project(name))})


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


# ------------------------------------------------------------------ manuals


def manuals_list(request: HttpRequest) -> JsonResponse:
    """Every manual on the card, and how far through reading it the indexer has got.

    ``read`` against ``pages`` is the whole of the progress display. It comes off the frontmatter
    rather than from any live channel, because the thing doing the work is a different process
    that writes its progress down every few pages - so the page re-fetching this is reading the
    same file the indexer is appending to, which is all the coordination either of them needs.
    """
    settings = _settings()
    out = []
    for one in manuals.catalog(settings):
        out.append(
            {
                "key": one.key,
                "folder": one.path.name,
                "name": one.name,
                "part": one.part,
                "maker": one.maker,
                "revision": one.revision,
                "aliases": one.aliases,
                "pages": one.pages,
                "read": one.read,
                "ready": one.ready,
                "added": one.added,
                "cover": f"/manual-media/{one.path.name}/{manuals.PAGES_DIR}/0001.jpg"
                if (one.pages_dir / "0001.jpg").is_file()
                else "",
            }
        )
    return JsonResponse({"manuals": out, "reading": not all(m["ready"] for m in out)})


@require_POST
def manual_upload(request: HttpRequest) -> JsonResponse:
    """Take in one PDF and give it a folder of its own. The indexer does the rest.

    Nothing is read here and nothing is asked of a model: this answers as soon as the bytes are on
    the card, and ``cyclops-index`` notices the new folder through inotify within seconds. A
    request that waited for a manual to be read would hold a connection open for a minute and
    would have to be retried after a power cut, which is exactly the design the rest of this box
    avoids.
    """
    refusal = _offered(request)
    if refusal is not None:
        return refusal
    chunks = iter(lambda: request.read(CHUNK), b"")
    try:
        landed = manuals.receive(
            _settings().manuals_dir,
            request.GET.get("name", ""),
            chunks,
            limit=MAX_UPLOAD_BYTES,
        )
    except ValueError:  # more bytes arrived than the header promised
        return HttpResponseBadRequest(_too_big())
    except OSError as exc:
        return HttpResponseBadRequest(f"could not write that file ({exc})")
    if landed is None:
        return HttpResponseBadRequest("a manual has to be a PDF with a name in it")
    return manuals_list(request)


def manual_media(request: HttpRequest, name: str, relative: str) -> HttpResponse:
    """One rendered page out of one manual. The same containment every other read here runs on."""
    folder = shelf.resolve(_settings().manuals_dir, name)
    if folder is None:
        raise Http404("no such manual")
    found = shelf.inside(folder, relative)
    if found is None or not found.is_file():
        raise Http404("no such page")
    kind = shelf.MEDIA_TYPES.get(found.suffix.lower())
    if kind is None:
        raise Http404("no such page")
    return _serve(request, found, kind)


# ------------------------------------------------------------------ putting something in

# The two endpoints on this box that answer to somebody who is not the kiosk - see the note on
# MIDDLEWARE in settings.py. Every other mutating view here refuses anything that is not loopback,
# because everything else they do only means something to the panel. These two mean the opposite:
# writing from somewhere that is not the Pi is the entire feature, and a rule that let only the
# Pi upload to the Pi would leave nothing behind.
#
# What is left holding the line is containment, and it is the same containment the read-only
# routes already run on. `shelf.inside` resolves before it compares, so `..`, an absolute name and
# a symlink out of the tree all fail together; `store` sanitizes every name that becomes a path
# segment; and neither of them can be handed a destination outside one project folder.


def _too_big() -> str:
    """What the page is told when a file is past the ceiling. Said from two places, so said once."""
    return f"that file is over {MAX_UPLOAD_BYTES // (1024 * 1024)} MB"


def _offered(request: HttpRequest) -> HttpResponse | None:
    """Refuse a body we cannot size, or one too big. ``None`` means carry on.

    Django sizes the stream it hands us from Content-Length and falls back to *zero* when the
    header is missing (wsgi.py builds a LimitedStream from it). So a chunked body - curl -T, or
    any client that streams - would read as nothing, and write_stream would fsync and rename an
    empty file onto the card: a name with no bytes behind it, landed durably, looking exactly like
    a file that arrived. That is the zero-byte husk this whole card.py dance exists to make
    impossible, and it is only not reachable from our own page because fetch() sets the header.
    Refuse it out loud instead, and take the chance to turn away something enormous before any of
    it has been written rather than 128 MB in.

    ``request.read`` goes straight to the WSGI input and past every ``DATA_UPLOAD_*`` ceiling
    Django would otherwise apply, so the ceiling below is not a second opinion - it is the only
    one there is.
    """
    try:
        offered = int(request.META.get("CONTENT_LENGTH") or 0)
    except ValueError:
        offered = 0
    if offered <= 0:
        return HttpResponse("say how many bytes are coming", status=411)
    if offered > MAX_UPLOAD_BYTES:
        return HttpResponse(_too_big(), status=413)
    return None


def _folder(name: str, relative: str) -> Path:
    """The directory inside one project that a write is aimed at, or 404.

    Deliberately the same two steps in the same order as every read on this page: resolve the
    project, then resolve the path under it. A write does not get its own path logic - that is
    how the two drift and how the stricter one stops being the one that runs.
    """
    here = shelf.inside(_project(name), relative)
    if here is None or not here.is_dir():
        raise Http404("no such folder")
    return here


@require_POST
def project_mkdir(request: HttpRequest, name: str) -> JsonResponse:
    """Make a folder inside a project. ``?path=`` is the folder to make it in, "" for the root."""
    relative = request.GET.get("path", "")
    here = _folder(name, relative)
    try:
        made = store.make_folder(here, request.POST.get("name", ""))
    except OSError as exc:
        return HttpResponseBadRequest(f"could not make that folder ({exc})")
    if made is None:
        return HttpResponseBadRequest("that name has nothing in it a folder can be called")
    return JsonResponse(shelf.listing(_project(name), relative) or {})


@require_POST
def project_upload(request: HttpRequest, name: str) -> JsonResponse:
    """Land one uploaded file in one folder of one project, and answer with the folder again.

    The body is the file itself and not a multipart form, which is two savings on a Pi rather than
    one. Django's upload handling spools anything over 2.5 MB to a temporary file first, and
    ``cyclops-admin.service`` deliberately leaves ``PrivateTmp`` unset - so every large upload
    would be written to the card once before we wrote it to the card again. Reading the stream
    ourselves also skips the multipart parse, on the box whose design note is that CPU matters.

    ``request.read`` goes straight to the WSGI input and past every ``DATA_UPLOAD_*`` ceiling
    Django would otherwise apply, so ``MAX_UPLOAD_BYTES`` below is not a second opinion - it is
    the only one there is.

    Answering with the fresh listing rather than 204 is what lets the page repaint from this
    response. It has just changed the directory it is showing; making it ask again what it already
    caused would be a second round trip over the LAN for something we are holding.
    """
    # Django sizes the stream it hands us from Content-Length and falls back to *zero* when the
    # header is missing (wsgi.py builds a LimitedStream from it). So a chunked body - curl -T, or
    # any client that streams - would read as nothing, and write_stream would fsync and rename an
    # empty file into the project: a name with no bytes behind it, landed durably, looking exactly
    # like a file that arrived. That is the zero-byte husk this whole card.py dance exists to make
    # impossible, and it is only not reachable from our own page because fetch() sets the header.
    # Refuse it out loud instead, and take the chance to turn away something enormous before any
    # of it has been written to the card rather than 128 MB in.
    refusal = _offered(request)
    if refusal is not None:
        return refusal

    relative = request.GET.get("path", "")
    here = _folder(name, relative)
    chunks = iter(lambda: request.read(CHUNK), b"")
    try:
        landed = store.receive(here, request.GET.get("name", ""), chunks, limit=MAX_UPLOAD_BYTES)
    except ValueError:  # more bytes arrived than the header promised
        return HttpResponseBadRequest(_too_big())
    except OSError as exc:
        return HttpResponseBadRequest(f"could not write that file ({exc})")
    if landed is None:
        return HttpResponseBadRequest("that name has nothing in it a file can be called")
    return JsonResponse(shelf.listing(_project(name), relative) or {})
