"""Settings loaded from the environment (and a local ``.env`` file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# Where the kiosk's admin browser is told to close. The admin service drops this file and the
# kiosk - which owns the Chromium process and the panel it covers - sees it and takes the panel
# back, so the process that put the page on screen is the one that takes it off. Deliberately
# not under /tmp: a ``PrivateTmp=`` on the service would silently give the two sides different
# views of it.
BROWSER_CLOSE_FLAG = Path.home() / ".cache" / "cyclops" / "browser-close"

# And where the service says it has just handed the dashboard to a browser on this box. The
# kiosk keeps a Chromium warmed up behind its own window so that tapping the gear only has to
# uncover it, and this note is the only way it can tell when that browser is ready: Chromium
# announces nothing, and nothing can be asked what is currently on the panel.
PAGE_SERVED_FLAG = Path.home() / ".cache" / "cyclops" / "page-served"

# And where the *page* says so, which is a different claim and the one that matters at startup.
# PAGE_SERVED_FLAG is written when Django hands over the bytes; this one is written when the
# browser has parsed them, run the script and begun polling, which is what Chromium's habit of
# raising its own window a second time is tied to. The kiosk clears it before it spawns a
# browser, so its reappearance can only mean *that* browser is alive - see
# :meth:`cyclops.kiosk.Kiosk.warm_browser`.
PAGE_ALIVE_FLAG = Path.home() / ".cache" / "cyclops" / "page-alive"

# And where whatever the panel is being asked to show waits for it to notice. Everything that
# goes on the glass is a jpeg riding in this payload under "image" - a photo off the shutter, one
# recalled from the card, an edit, a diagram (cyclops.imagine). It was diagram.json until
# 2026-09-04, when a diagram stopped being the special case that had named it. This holds
# the thing itself rather than being an empty note, because the page has to render it and the two
# processes share no memory - and because something made with no session running is never written
# to the card, so a path would have nothing to point at. Written by whoever made it, read by the
# admin service, removed by the kiosk when it takes the panel back: absent means "show the
# dashboard".
PANEL_FILE = Path.home() / ".cache" / "cyclops" / "panel.json"

# And beside it, the page's own picture of the scratchpad it is showing - the one thing on the
# panel that arrives as markup rather than as pixels, and so the one thing a recording cannot get
# from the offer above. The page draws it a second time onto a canvas and posts the JPEG back; see
# :func:`cyclops.panel.keep_still` and :mod:`cyclops.still`.
#
# Beside the offer and not inside it, deliberately. PANEL_FILE has exactly one writer, which
# replaces it whole and mints a new id each time. A raster merged into it would make the admin
# process a second writer doing a read-modify-write on the file that decides what is on the glass:
# one landing after a withdraw would put a dismissed picture back up, and one landing after a
# newer offer would revert its id - and the id is the whole repaint trigger, so the page would
# never learn about the new picture at all. A picture *of* an offer is not an offer.
#
# The id says which offer it is a picture of. A stale one is ignored, which costs a black frame
# in the recording and nothing on the glass. Removed with the offer, by ``panel.withdraw``.
PANEL_STILL_FILE = Path.home() / ".cache" / "cyclops" / "panel-still.json"

# And where the page says it has actually painted that picture. The kiosk waits for this before
# uncovering the browser, exactly as it waits on PAGE_SERVED_FLAG at startup - without it the
# panel would show the dashboard for as long as the picture takes to decode, which is the one
# moment somebody is watching.
PANEL_PAINTED_FLAG = Path.home() / ".cache" / "cyclops" / "panel-painted"

# And where the kiosk says a picture is *on the glass* right now - not merely offered. The two
# are not the same question, and only one of them is safe to hand to the LAN. PANEL_FILE can hold
# an offer nothing is showing: ``Kiosk.show_picture`` refuses one while the admin page has the
# panel and leaves the payload where it was, so "something is pending" outlives the moment a
# person could be looking at it. This note is touched inside the latch that reveals a picture and
# removed in the same ``finally`` that withdraws it, so it means exactly "somebody can see this".
#
# It exists for the companion: a phone or an iPad on the LAN may put away a picture that is up,
# and may never close the panel's own page. That gate is this file - see
# ``cyclops.admin.views.close_browser``.
PICTURE_UP_FLAG = Path.home() / ".cache" / "cyclops" / "picture-up"

# And where it leaves the output volume it wants. Same reason it cannot just set it itself:
# the service runs with PrivateDevices=yes and has no /dev/snd, and at boot there is no user
# session to reach PipeWire through. The kiosk, which has both, reads this and applies it.
VOLUME_FILE = Path.home() / ".cache" / "cyclops" / "volume"

# And where it leaves the answer to "may I talk over you?". The same note-and-pick-up shape as
# the volume, for a different reason: the page is not the process holding the microphone, and
# the one that is may be halfway through a sentence. Absent means nobody has ever said, and
# CYCLOPS_BARGE_IN_DB is left to decide - see :mod:`cyclops.barge`.
BARGE_IN_FILE = Path.home() / ".cache" / "cyclops" / "barge-in"

# And where it leaves the answer to "what is a session's video a recording of?" - the panel, or
# the camera on its own. The same note-and-pick-up shape again, for the plainest reason of the
# three: the page is not the process holding either one. Absent means nobody has ever said, and
# CYCLOPS_RECORD_SOURCE is left to decide - see :mod:`cyclops.filming`.
RECORD_SOURCE_FILE = Path.home() / ".cache" / "cyclops" / "record-source"

# And where it leaves the answer to "who should be answering me?" - one of the ten voices the
# Realtime API offers. The same note-and-pick-up shape as the three above, and for the same
# reason as the record source: the page is not the process that opens the conversation, and the
# voice can only be set in the ``session.update`` that opens one. Absent means nobody has ever
# said, and CYCLOPS_VOICE is left to decide - see :mod:`cyclops.voice`.
VOICE_FILE = Path.home() / ".cache" / "cyclops" / "voice"

# And beside it, the one note here that is a request rather than a state: play that voice now, so
# whoever is standing at the panel can hear what they are choosing between. Written by the page,
# consumed *and deleted* by the kiosk - the only process on this box with a /dev/snd - which is
# the same shape as PANEL_FILE and for the same reason: a thing that has been done is not a
# setting, and leaving it lying about would sound it again on the next poll.
SAY_VOICE_FILE = Path.home() / ".cache" / "cyclops" / "say-voice"

# And where the index service leaves what it has read off the card, for the voice agent to search.
# One file rather than two, because the service writes it while a session reads it and there is no
# lock between them: vectors and their metadata in separate files could be torn apart across a
# rewrite, and a reader would rank one generation's numbers against the next generation's names.
# An .npz built in memory and landed through card.write_bytes cannot do that.
#
# Under ~/.cache with the locks and the panel flags, and this is a choice about *kind* rather than
# about disks - on a Pi there is one SD card and everything is on it. sessions/ and projects/ are
# a tree meant to be browsed in Finder and copied off whole; a quarter-megabyte of float32 is not
# something to find in there, nor to have travel with a card image. It is also derived: delete it
# and the service rebuilds it from the card, which is still the only thing that holds a fact.
RECALL_FILE = Path.home() / ".cache" / "cyclops" / "recall.npz"

# And the lock that says one process is reconciling it. Same reasoning as store.LOCK_FILE: flock,
# so the kernel drops it when the holder dies and a box that loses power mid-sweep comes back with
# nothing stale to detect. This is what stops a hand-run `cyclops-index --once` writing over the
# service that is already running.
RECALL_LOCK = Path.home() / ".cache" / "cyclops" / "recall.lock"

# And the lock that says one process is rendering a video - see cyclops.cut. A second one rather
# than sharing RECALL_LOCK, because they answer different questions: a reconcile holds that one
# for a second or two on almost every sweep, and "is an encode running?" has to be answerable
# without an ordinary indexing pass making the answer yes. Same flock discipline either way, so a
# render killed by a deploy or a power cut leaves nothing behind to unwedge.
CUT_LOCK = Path.home() / ".cache" / "cyclops" / "cut.lock"

# The one note that goes the other way: which screen the panel wants the page it is about to
# uncover to be on. The warm browser is loaded once at boot and never navigated, and there are two
# things on the panel that open it - his face, which promises the sessions, and the heat gauge,
# which promises the numbers behind itself - so somebody has to say which. It is answered on
# /api/panel, the poll the page already runs several times a second, and it exists only while a
# reveal is being set up: the kiosk writes it, the page routes on it, the kiosk deletes it.
PAGE_SCREEN_FILE = Path.home() / ".cache" / "cyclops" / "page-screen"

# The one thing the two processes share that is not a file: the port the kiosk hands the live
# picture and the live voice out on (:mod:`cyclops.companion`). It is here rather than in that
# module because the *admin* service is what tells the page the number, and reaching for it
# there would mean importing OpenCV and the overlay's fonts into a web worker to read one int.
# The host is deliberately not here: a phone arrived at the page by cyclops.local or by an
# address and only the page knows which, so the script builds the URL from location.hostname.
COMPANION_PORT = 8081  # unprivileged, so nothing has to grant a capability to bind it


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    api_key: str = field(repr=False)
    model: str = "gpt-realtime-2.1"  # newest GA speech-to-speech model (Jul 2026)
    voice: str = "marin"
    volume: float = 1.0  # output gain 0.0-1.0 (CYCLOPS_VOLUME is a percent 0-100)
    sounds: bool = True  # cues: the box booting, the link opening and closing, and the shutter
    reasoning_effort: str | None = None  # None: "low" on reasoning models, omitted otherwise
    camera_index: int | None = None  # None: probe cameras and remember the one that works
    half_duplex: bool | None = None  # None: auto-detect from the output device
    barge_in_db: float | None = 8.0  # how much louder than the echo you must be; None = off
    transcribe_lang: str | None = "en"  # ISO-639-1 hint for the input transcriber; None = auto
    mic_gain_db: float = 0.0  # fixed gain on the way in, before compression; for a quiet mic
    mic_compress: bool = True  # even out how loud you are - see :class:`cyclops.audio.Compressor`
    input_device: str | None = None  # sounddevice mic: index or name substring; None = default
    output_device: str | None = None  # sounddevice speaker: index or name substring; None = default
    sessions_dir: Path = Path("sessions")  # one folder per session; everything it produced
    captures_dir: Path = Path("captures")  # where a photo goes when no session is running
    projects_dir: Path = Path("projects")  # one folder per project; README, Log, Photos
    manuals_dir: Path = Path("manuals")  # one folder per manual; the PDF, its pages, its identity
    about_file: Path = Path("about-you.md")  # the standing facts about whoever it works with
    slug: bool = True  # name each finished session from its transcript (cyclops.slug)
    remember: bool = True  # keep about-you.md up to date from what is said (cyclops.about)
    projects: bool = True  # keep projects/ up to date, and offer the two project tools
    diagrams: bool = True  # offer draw_diagram, and keep what it draws in photos/
    recall: bool = True  # offer the recall tool, and index what is on the card (cyclops.indexer)
    # Read uploaded PDFs into searchable pages (cyclops.reading). Separate from `recall` rather
    # than folded into it: a manual costs one vision call per page the moment it lands, which is
    # the one thing on this box that spends real money without anybody asking it to.
    manuals: bool = True
    video: bool = True  # offer watch_video, which plays a YouTube video full-screen on the panel
    imagine: bool = True  # offer edit_photo, and keep what it makes (cyclops.imagine)
    scratchpad: bool = True  # offer write_on_scratchpad, so he can write on the panel himself
    cut: bool = True  # offer the Make a video button, and render what it asks for (cyclops.cut)
    pointing: bool = True  # offer point_at, so he can mark their photo instead of describing it
    look: bool = True  # offer take_a_look, so asking it to look takes the photo - no SNAP needed
    # offer sketch, which draws while he is still writing it - and takes the scratchpad's
    # place rather than sitting beside it, because two doors onto the same glass is a choice
    # the model would have to make mid-sentence, every time.
    sketch: bool = False
    record: bool = True  # record the session to its folder (needs a camera, or a panel)
    record_source: str = "screen"  # "screen": the panel, UI and all; "camera": the raw picture
    record_fps: int = 15  # video sampling rate; ~5% of a Pi 5 core at 800x480
    record_width: int = 0  # cap the recorded width, never upscaling; 0 keeps the source's own
    sleep_after_s: int = 60  # untouched for this long the panel goes dark; 0 keeps it awake
    button_pin: int | None = 17  # BCM17 (physical 11): the shutter button beside the panel
    button_led_pin: int | None = 27  # BCM27 (physical 13): the ring inside it; see cyclops.button
    admin_host: str = "0.0.0.0"  # the admin page is meant to be read from the LAN, not just here
    admin_port: int = 80  # so it opens with a bare hostname


def _env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def _tristate(value: str | None) -> bool | None:
    """'1/true/yes/on' -> True, '0/false/no/off' -> False, unset/'auto' -> None."""
    normalized = (value or "").lower()
    if normalized in {"", "auto"}:
        return None
    return normalized in {"1", "true", "yes", "on"}


def _flag(name: str, default: bool) -> bool:
    """Like :func:`_tristate`, but an unset variable means the given default rather than None."""
    value = _tristate(_env(name))
    return default if value is None else value


def _int(name: str) -> int | None:
    raw = _env(name)
    if raw is None or raw.lower() == "auto":
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer or 'auto', got {raw!r}") from exc


def _count(name: str, default: int) -> int:
    """A non-negative count, where 0 is a real answer and not "unset".

    ``_int(name) or default`` reads well and is wrong for any setting whose zero means "off":
    it silently hands back the default. That is survivable for a frame rate, which is why
    ``record_fps`` still does it, but ``CYCLOPS_SLEEP_AFTER_S=0`` has to mean never blank
    rather than blank after the default.
    """
    value = _int(name)
    if value is None:
        return default
    if value < 0:
        raise ConfigError(f"{name} must be zero or more, got {value}")
    return value


def _pin(name: str, default: int | None) -> int | None:
    """A BCM pin number, or None where the thing is not wired at all.

    Unset means the default rather than None, unlike :func:`_int`, whose None means "work it
    out": there is nothing to work out about a pin. Either it is the one the panel was built
    with or somebody has moved it, and the numbers live here rather than inline because the
    panel layout is still settling. ``off`` is how you say the button is not there.
    """
    raw = _env(name)
    if raw is None:
        return default
    if raw.lower() in {"off", "none", "no"}:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a BCM pin number or 'off', got {raw!r}") from exc


def _barge_in_db(name: str) -> float | None:
    raw = _env(name)
    if raw is None:
        return Settings.barge_in_db
    if raw.lower() in {"off", "0", "false", "no"}:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number of dB or 'off', got {raw!r}") from exc


def _decibels(name: str, default: float) -> float:
    """A number of dB, plus or minus. Unclamped on purpose: a lav can want a lot, and the
    compressor's ceiling is what stops any of it clipping."""
    raw = _env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number of dB, got {raw!r}") from exc


def _volume(name: str) -> float:
    raw = _env(name)
    if raw is None:
        return Settings.volume
    try:
        percent = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number 0-100, got {raw!r}") from exc
    return max(0.0, min(1.0, percent / 100.0))


def _record_source(name: str) -> str:
    """Which of the two things a session's video is of. See :mod:`cyclops.filming`.

    Spelled out rather than made a flag because "record the screen: off" does not say what you
    get instead, and the two words are the whole feature.
    """
    raw = _env(name)
    if raw is None:
        return Settings.record_source
    if raw.lower() not in {"camera", "screen"}:
        raise ConfigError(f"{name} must be 'camera' or 'screen', got {raw!r}")
    return raw.lower()


def _voice(name: str) -> str:
    """One of the ten the Realtime API will accept. See :mod:`cyclops.voice`.

    Validated here rather than left to OpenAI for the reason ``_record_source`` is: a box that
    will not do what you meant should say so while you are watching, and the alternative is a
    session that opens, is refused, and closes again with the reason three layers down in a
    websocket error.

    The names are written out here rather than imported from ``cyclops.voice``, which imports
    this module - ``_record_source`` has the same duplication for the same reason, and
    ``tests/test_voice.py`` is what holds the two lists to each other.
    """
    raw = _env(name)
    if raw is None:
        return Settings.voice
    wanted = raw.lower()
    legal = ("marin", "cedar", "alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer",
             "verse")
    if wanted not in legal:
        raise ConfigError(f"{name} must be one of {', '.join(legal)}, got {raw!r}")
    return wanted


def _lang(name: str) -> str | None:
    raw = _env(name)
    if raw is None:
        return Settings.transcribe_lang
    return None if raw.lower() == "auto" else raw.lower()


def load_settings(*, require_api_key: bool = True) -> Settings:
    """Load the nearest ``.env`` (searching upward from the CWD) and build Settings.

    ``require_api_key=False`` is for ``cyclops-admin``, which never talks to OpenAI and has no
    business holding a key - it only wants the paths and the port.
    """
    if dotenv_path := find_dotenv(usecwd=True):
        load_dotenv(dotenv_path)

    api_key = _env("OPENAI_API_KEY")
    if not api_key and require_api_key:
        raise ConfigError(
            "OPENAI_API_KEY is not set. Put it in a .env file next to pyproject.toml "
            "(see .env.example) or export it in your shell."
        )
    return Settings(
        api_key=api_key or "",
        model=_env("CYCLOPS_MODEL") or Settings.model,
        voice=_voice("CYCLOPS_VOICE"),
        volume=_volume("CYCLOPS_VOLUME"),
        sounds=_flag("CYCLOPS_SOUNDS", Settings.sounds),
        reasoning_effort=_env("CYCLOPS_REASONING_EFFORT"),
        camera_index=_int("CYCLOPS_CAMERA_INDEX"),
        half_duplex=_tristate(_env("CYCLOPS_HALF_DUPLEX")),
        barge_in_db=_barge_in_db("CYCLOPS_BARGE_IN_DB"),
        transcribe_lang=_lang("CYCLOPS_LANG"),
        mic_gain_db=_decibels("CYCLOPS_MIC_GAIN_DB", Settings.mic_gain_db),
        mic_compress=_flag("CYCLOPS_MIC_COMPRESS", Settings.mic_compress),
        input_device=_env("CYCLOPS_INPUT_DEVICE"),
        output_device=_env("CYCLOPS_OUTPUT_DEVICE"),
        sessions_dir=Path(_env("CYCLOPS_SESSIONS_DIR") or Settings.sessions_dir).expanduser(),
        captures_dir=Path(_env("CYCLOPS_CAPTURES_DIR") or Settings.captures_dir).expanduser(),
        projects_dir=Path(_env("CYCLOPS_PROJECTS_DIR") or Settings.projects_dir).expanduser(),
        manuals_dir=Path(_env("CYCLOPS_MANUALS_DIR") or Settings.manuals_dir).expanduser(),
        about_file=Path(_env("CYCLOPS_ABOUT_FILE") or Settings.about_file).expanduser(),
        slug=_flag("CYCLOPS_SLUG", Settings.slug),
        remember=_flag("CYCLOPS_REMEMBER", Settings.remember),
        projects=_flag("CYCLOPS_PROJECTS", Settings.projects),
        diagrams=_flag("CYCLOPS_DIAGRAMS", Settings.diagrams),
        recall=_flag("CYCLOPS_RECALL", Settings.recall),
        manuals=_flag("CYCLOPS_MANUALS", Settings.manuals),
        video=_flag("CYCLOPS_VIDEO", Settings.video),
        imagine=_flag("CYCLOPS_IMAGINE", Settings.imagine),
        scratchpad=_flag("CYCLOPS_SCRATCHPAD", Settings.scratchpad),
        cut=_flag("CYCLOPS_CUT", Settings.cut),
        pointing=_flag("CYCLOPS_POINTING", Settings.pointing),
        look=_flag("CYCLOPS_LOOK", Settings.look),
        sketch=_flag("CYCLOPS_SKETCH", Settings.sketch),
        record=_flag("CYCLOPS_RECORD", Settings.record),
        record_source=_record_source("CYCLOPS_RECORD_SOURCE"),
        record_fps=_int("CYCLOPS_RECORD_FPS") or Settings.record_fps,
        # _count, not `or`: 0 means "whatever the source is", which `or` would read as unset.
        record_width=_count("CYCLOPS_RECORD_WIDTH", Settings.record_width),
        # _count, not `or`: 0 means "never blank", which `or` would read as unset. See _count.
        sleep_after_s=_count("CYCLOPS_SLEEP_AFTER_S", Settings.sleep_after_s),
        button_pin=_pin("CYCLOPS_BUTTON_PIN", Settings.button_pin),
        button_led_pin=_pin("CYCLOPS_BUTTON_LED_PIN", Settings.button_led_pin),
        admin_host=_env("CYCLOPS_ADMIN_HOST") or Settings.admin_host,
        admin_port=_int("CYCLOPS_ADMIN_PORT") or Settings.admin_port,
    )
