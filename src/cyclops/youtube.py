"""Everything this box knows how to ask YouTube, and nothing about what to do with it.

Three questions, one library. What videos match a phrase (:func:`search`), what one video
is made of (:func:`details`), and what is said in it minute by minute (:func:`windows`).
:mod:`cyclops.watch` decides; this only fetches.

**What must never leave here unaccompanied.** :func:`windows` returns a video's whole
transcript. It is thousands of tokens, and it exists for exactly one purpose: a side-car
model call that reads it once and answers with a number. It must never reach the Realtime
session - not in a ``function_call_output``, not in a note, not summarised. A transcript
that lands in that conversation is paid for again on every turn that follows it, for the
rest of the conversation. Everything else here is small enough to say out loud.

All of it blocks, and some of it for seconds. Callers are on an event loop that is carrying
audio, so every one of these goes through ``asyncio.to_thread``.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field

from yt_dlp import YoutubeDL

# A search is titles, and a title is short. Six is enough for a model to have a real choice
# without paying for a page of near-duplicates - YouTube's first three results for a
# how-to are frequently the same channel.
SEARCH_RESULTS = 6
# Half a minute is about one step of a procedure, and it is what makes the transcript
# readable as a list of moments rather than a wall of speech. Smaller windows cost tokens
# for no extra resolution: nobody asks for a video "at 4:07 rather than 4:15".
WINDOW_S = 30
MAX_WINDOW_CHARS = 600  # one window of babble cannot crowd out the rest of the video

# Progressive MP4s, best first: one URL carrying both tracks, which is the only shape a bare
# <video> can play. 18 is 360p and 22 is 720p, both H.264+AAC - the same decoder path the
# panel already uses for session recordings, and 360p is the right size for an 800x480
# screen on a box with no hardware video decode. Everything else YouTube offers is DASH:
# separate audio and video URLs that would have to be muxed, which is a download pipeline.
PROGRESSIVE = ("18", "22")
MAX_HEIGHT = 720

# The player client is load-bearing, and this is the least obvious line in the file.
# YouTube serves a different menu of formats to each of its own clients, and only some of
# them still include a progressive MP4 at all; the rest are DASH-only, which a bare <video>
# cannot play. Measured 2026-09-11 against yt-dlp 2026.8.19 over seven videos: the default
# client set returned no progressive format for any of them, and "android" returned 18 for
# all seven. So this is pinned rather than left to the default, and pinned in one place.
#
# It is also the thing most likely to break. If playback stops working and the extract still
# succeeds, look here first: check whether another client still serves format 18 before
# assuming the video is at fault.
_CLIENT = {"youtube": {"player_client": ["android"]}}
_QUIET = {
    "quiet": True,
    "no_warnings": True,
    "skip_download": True,
    "noplaylist": True,
    "extractor_args": _CLIENT,
}
_FETCH_TIMEOUT_S = 10.0
# yt-dlp signs the caption URLs it hands back for the client it pretended to be. Asking for
# them with a bare urllib User-Agent gets an empty document rather than an error.
_UA = "Mozilla/5.0"


@dataclass(frozen=True)
class Candidate:
    """One search result, as much as a flat listing knows: no chapters, no formats."""

    id: str
    title: str
    channel: str
    duration: int


@dataclass(frozen=True)
class Video:
    """One video, looked at properly. ``stream`` is empty when nothing progressive exists."""

    id: str
    title: str
    channel: str
    duration: int
    stream: str
    thumb: str
    chapters: list[tuple[int, str]] = field(default_factory=list)


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _seconds(value: object) -> int:
    return int(value) if isinstance(value, (int, float)) and value > 0 else 0


def search(query: str, limit: int = SEARCH_RESULTS) -> list[Candidate]:
    """Videos matching a phrase, best first. Flat: titles and durations, no per-video fetch.

    Empty rather than raising when YouTube answers with nothing - "I could not find one" is
    an answer the caller has to be able to give anyway, and a search that finds nothing is
    not an error.
    """
    with YoutubeDL({**_QUIET, "extract_flat": True}) as ydl:
        found = ydl.extract_info(f"ytsearch{limit}:{query}", download=False)
    out = []
    for entry in (found or {}).get("entries") or []:
        ident = _text(entry.get("id"))
        # A live stream has no fixed transcript and no moment to jump to, and a
        # zero-duration entry is a channel or a premiere that has not aired.
        if not ident or entry.get("live_status") in {"is_live", "is_upcoming"}:
            continue
        duration = _seconds(entry.get("duration"))
        if not duration:
            continue
        out.append(
            Candidate(
                id=ident,
                title=_text(entry.get("title")),
                channel=_text(entry.get("channel")) or _text(entry.get("uploader")),
                duration=duration,
            )
        )
    return out


def _stream_of(info: dict) -> str:
    """The best progressive URL, or "" if this video has none.

    Always the URL yt-dlp hands back, never one rebuilt from its parts: it applies the
    `n`-parameter transform, and without that googlevideo throttles the stream to a crawl a
    few seconds in - which looks like buffering, not like a bug here.
    """
    formats = {_text(f.get("format_id")): f for f in info.get("formats") or []}
    for wanted in PROGRESSIVE:
        found = formats.get(wanted)
        if found and _text(found.get("url")):
            return _text(found["url"])
    for found in info.get("formats") or []:
        muxed = found.get("vcodec") not in (None, "none") and found.get("acodec") not in (
            None,
            "none",
        )
        if muxed and 0 < (found.get("height") or 0) <= MAX_HEIGHT and _text(found.get("url")):
            return _text(found["url"])
    return ""


def _thumb_of(info: dict) -> str:
    """The largest thumbnail that is still smaller than the panel."""
    sized = [t for t in info.get("thumbnails") or [] if 0 < (t.get("width") or 0) <= 800]
    if not sized:
        return _text(info.get("thumbnail"))
    return _text(max(sized, key=lambda t: t["width"]).get("url"))


def details(video_id: str) -> Video | None:
    """One video in full: its chapters, a playable URL and a picture of it.

    None when YouTube will not say - a pulled video, a private one, a network that is gone.
    A video with no progressive format comes back with an empty ``stream`` rather than as
    None, so the caller can tell "there is no such video" from "I cannot play this one".
    """
    try:
        with YoutubeDL(_QUIET) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
    except Exception:  # noqa: BLE001 - every failure here is the same answer: ask for another
        return None
    if not info:
        return None
    chapters = [
        (_seconds(c.get("start_time")), _text(c.get("title")))
        for c in info.get("chapters") or []
        if _text(c.get("title"))
    ]
    return Video(
        id=_text(info.get("id")) or video_id,
        title=_text(info.get("title")),
        channel=_text(info.get("channel")) or _text(info.get("uploader")),
        duration=_seconds(info.get("duration")),
        stream=_stream_of(info),
        thumb=_thumb_of(info),
        chapters=chapters,
    )


def _caption_url(info: dict) -> str:
    """The English caption track, written ones before automatic ones, as json3."""
    for source in (info.get("subtitles") or {}, info.get("automatic_captions") or {}):
        for lang in ("en", "en-orig", "en-US", "en-GB"):
            for track in source.get(lang) or []:
                if track.get("ext") == "json3" and _text(track.get("url")):
                    return _text(track["url"])
    return ""


def buckets(events: list[dict]) -> list[tuple[int, str]]:
    """Caption cues folded into :data:`WINDOW_S` windows. Pure, so it is the testable half.

    Takes the ``events`` of a json3 document, which is what YouTube serves and what a test
    can write by hand.
    """
    held: dict[int, list[str]] = {}
    for event in events:
        segments = event.get("segs")
        if not segments:
            continue
        # Cues carry hard line breaks from however the captioner wrapped them. A window is
        # one line of a prompt, so they are folded out here rather than by every reader.
        said = " ".join("".join(str(s.get("utf8", "")) for s in segments).split())
        if said:
            held.setdefault(int(event.get("tStartMs", 0)) // 1000 // WINDOW_S, []).append(said)
    return [(at * WINDOW_S, " ".join(v)[:MAX_WINDOW_CHARS]) for at, v in sorted(held.items())]


def windows(video_id: str) -> list[tuple[int, str]]:
    """What is said in a video, as (second, text) windows. Empty when it has no captions.

    Read the warning in this module's docstring before you pass the result anywhere.
    """
    try:
        with YoutubeDL({**_QUIET, "writeautomaticsub": True}) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
        url = _caption_url(info or {})
        if not url:
            return []
        request = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(request, timeout=_FETCH_TIMEOUT_S) as response:
            document = json.loads(response.read())
    except Exception:  # noqa: BLE001 - no captions is a shrug, not a failure; chapters remain
        return []
    return buckets(document.get("events") or [])


def fetch(url: str) -> bytes:
    """One picture off the web, as bytes. Empty when it could not be had.

    Here rather than in a caller because it is the same signed-URL etiquette as the caption
    fetch above, and because nothing else in this package reaches off-box for an image.
    """
    try:
        request = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(request, timeout=_FETCH_TIMEOUT_S) as response:
            return response.read()
    except Exception:  # noqa: BLE001 - a missing picture costs a title card, nothing more
        return b""
