"""The panel handshake: one thing at a time, offered to the page and taken back afterwards.

Everything Cyclops can put on the glass comes through here - a photo off the shutter, a picture
recalled from the card, an edit, a diagram, a scratchpad of HTML the model wrote itself - and it is
one small file plus a flag. What is to be shown is written to :data:`cyclops.config.PANEL_FILE`;
the panel's page polls ``/api/panel`` every 400 ms, sees an id it has not drawn, fetches the
payload and paints it; :func:`show` asks the kiosk to uncover the browser once it has.

There are four doors, and they differ only in what they leave in the file. :func:`offer_image`
carries a JPEG, :func:`offer_scratchpad` carries markup, :func:`offer_sketch` carries nothing at
all - its content arrives over the companion stream instead, because it is still being written
when the offer is made - and :func:`offer_video` carries a URL to play and a picture to hold the
screen until it does. The page branches on which key is there and on nothing else; nothing
between here and there - not ``_leave``, not ``/api/picture``, not the kiosk - looks inside the
payload at all. Whatever is up, a press anywhere puts it away.

The scratchpad door has a way back through it, and only that one needs one. What travels out as
markup has to come home as pixels for a recording to see it, so the page posts its own picture
of what it painted and :func:`keep_still` writes it down beside the offer.

The picture travels *in* the file rather than as a path to one, and that is deliberate. The page
is served by the admin process, which shares no memory with whoever made the picture, and a
picture made with no session running was never written to the card at all - there would be
nothing for a path to point at.

**This was ``diagram.py`` until 2026-09-04**, and most of it drew diagrams: a JSON schema of
nodes and wires that a text model emitted and JointJS rendered on the page. An image model draws
a better diagram than a schema we maintain, so that came out, and a diagram is now a jpg in
``photos/`` like any other picture (:func:`cyclops.imagine.draw`).

What was left was the half that had never been about diagrams - the half the shutter button,
``recall`` and ``edit_photo`` all go through - still called ``diagram``. That name was worth
changing rather than keeping: it is how somebody deletes the camera from the panel while tidying
up a feature that no longer exists.
"""

from __future__ import annotations

import base64
import json
import sys
import uuid
from typing import Any

from . import card
from .config import PANEL_FILE, PANEL_STILL_FILE

# What a picture looks like on the way through here and in the still beside it. One spelling,
# because :func:`keep_still` has to recognise what :func:`offer_image` writes.
JPEG_URL = "data:image/jpeg;base64,"

_kiosk: object | None = None  # a cyclops.kiosk.Kiosk while one is running; see set_kiosk


def set_kiosk(kiosk: object | None) -> None:
    """Register the running kiosk, so a finished picture can find a screen to appear on.

    Same shape and same reason as :func:`cyclops.webcam.set_live_source` and
    :data:`cyclops.session._live`: what wants the panel is the agent's tool coroutine, which sits
    behind a :class:`~cyclops.ui.SessionController` built with settings and a camera and no way
    back to the kiosk. Threading a reference through would mean a new argument on
    ``SessionController`` and another on ``VoiceAgent``, for something there is only ever one of.

    It lives here rather than in :mod:`cyclops.kiosk` so that the caller does not have to import
    the kiosk to reach it - that module owns OpenCV and a Qt window, and ``uv run cyclops`` has
    neither and wants neither.
    """
    global _kiosk
    _kiosk = kiosk


def show() -> bool:
    """Ask the panel to show whatever :func:`offer_image` last left. False if there is no panel.

    False is the ordinary answer under ``uv run cyclops``, where there is a conversation and no
    screen. The caller says so out loud rather than treating it as a failure.
    """
    kiosk = _kiosk
    return bool(kiosk is not None and kiosk.show_picture())  # type: ignore[attr-defined]


_announce = False  # whether the picture waiting for the panel should make a sound; see announces
_hold_s = 0.0  # how long the panel should keep what is waiting; see hold_s


def announces() -> bool:
    """True when what is waiting for the panel should sound the "shown" cue as it lands.

    The panel asks so that it can stay quiet for some arrivals and not others, and the rule is
    whether anything else already said so. A photograph had the shutter a beat earlier and a
    second sound on top of it is one too many. A drawing had nothing: it was asked for out loud,
    it takes about half a minute to arrive, and in between there is only a strip of text on a
    screen the person may not be looking at.

    This used to be ``is_picture()``, and it read the question the other way round - was this a
    photograph rather than a drawing - because a drawing was then the only thing that was not a
    photograph. Once diagrams became photographs too that phrasing answered "yes" to everything
    and the panel would have gone silent for good. So the caller says what it wants rather than
    what it is holding.

    A module global rather than a read of the offer file, which carries the picture inline and
    would cost a megabyte of JSON parsed to answer one question. Whoever offers and whoever shows
    are the same process - :func:`show` reaches the kiosk through ``_kiosk`` - so there is
    nothing here that can get out of step with what is on the glass.
    """
    return _announce


def hold_s() -> float:
    """How long the kiosk should leave what is waiting on the glass, or 0 for the usual.

    A module global read by the kiosk, exactly as :func:`announces` is and for the argument
    it makes: whoever offers and whoever shows are the same process, so there is nothing here
    that can get out of step with what is on the glass - and reading it out of the offer file
    instead would mean something between here and the page looking inside the payload, which
    is the one thing this module promises nothing does.

    Zero from every door but :func:`offer_video`. A picture has nothing to run out.
    """
    return _hold_s


def offer_image(jpeg: bytes, title: str, *, announce: bool = False) -> bool:
    """Leave a picture where the panel's page will find it. False if it could not be left.

    ``announce`` asks the kiosk to sound the "shown" cue as this one lands, and that is all it
    does - see :func:`announces` for when it is worth it. It defaults to False because three of
    the four callers are photographs, which the shutter already announced.

    **There is one kind of picture.** This used to carry a second flag, ``drawn``, which rode all
    the way to the page and made a diagram behave differently from a photograph: a press anywhere
    put a photograph away, while a diagram kept a small square in the corner, on the argument that
    a diagram is a thing you point at while you talk about it. Two dismissals for one gesture, and
    the corner square was the one nobody could find on a second screen. A photo, a picture
    recalled from the card, an edit and a diagram are all now simply pictures, and a press
    anywhere puts any of them away. The scratchpad is the only other thing the panel shows, and
    it is a different door (:func:`offer_scratchpad`) rather than a flag on this one.

    Downscale it first. The caller does that, because the caller knows what the full-size copy is
    for - see :func:`cyclops.imagine.for_panel`.
    """
    url = JPEG_URL + base64.b64encode(jpeg).decode("ascii")
    global _announce, _hold_s
    _announce = announce
    _hold_s = 0.0
    return _leave({"title": title, "image": url})


def offer_scratchpad(html: str) -> bool:
    """Leave a scratchpad of HTML where the panel's page will find it. False if it could not.

    The other door onto the glass, and the cheap one. Everything above arrives as pixels somebody
    made - a camera, or half a minute of the image model - and this arrives as markup the
    model wrote itself in the time it takes to say a sentence. A number, a short list, a word worth
    reading rather than hearing, a small SVG.

    It is silent. :func:`announces` is about whether *anything else* said this had arrived, and a
    scratchpad is the one case where something did: the model is still talking when it lands. A
    drawing sounds because it took half a minute and nothing else marked it.

    No ``drawn`` key, and that is the whole of how it is dismissed. The page reads ``drawn`` to
    decide between a corner square and a press anywhere, and a scratchpad gets the press: it is one
    thing to look at and be done with rather than a diagram to point at while you argue about it.

    No title either, unlike :func:`offer_image`. Nothing has ever read that field - not the page,
    not :mod:`cyclops.still` - and here it would be a second argument the model has to write before
    anything can appear, which is the one cost this feature exists to avoid. The session log names
    it from the words in what he wrote (see ``cyclops.session._render_record``).
    """
    global _announce, _hold_s
    _announce = False
    _hold_s = 0.0
    return _leave({"scratchpad": html})


def offer_sketch() -> bool:
    """Tell the panel's page a sketch is coming. False if it could not be left.

    The third door, and the only one that carries nothing. A picture travels in the file and a
    scratchpad travels in the file, because both are one finished thing and the page's 400 ms
    poll is the cheapest way to hand it over. A sketch is not finished when it is offered - it is
    a program the model is still typing - so what travels here is a marker, and the frames
    themselves go straight from :mod:`cyclops.sketch` to the page over the companion stream,
    arriving every hundred milliseconds instead of every four hundred.

    Which means the poll is doing the one job it is good at: getting the browser uncovered. By
    the time the page has seen this and posted that it painted, several frames have already
    landed in the renderer sitting behind our window, so what appears is a half-built interface
    rather than a spinner.

    Silent, for the same reason a scratchpad is: the model is still talking when it lands.
    """
    global _announce, _hold_s
    _announce = False
    _hold_s = 0.0
    return _leave({"sketch": True})


def offer_video(url: str, title: str, start_s: int, thumb: bytes, hold: float) -> bool:
    """Leave a video for the panel to play, from ``start_s`` seconds in. False if it could not.

    The fourth door, and the only one that makes a noise for a while after it is opened.

    ``thumb`` rides along under ``image``, the same key :func:`offer_image` uses, and that is
    not a convenience - it is what keeps a video out of the session's recording as a black
    stretch. The kiosk paints nothing while the browser is uncovered, so the recorder asks
    :func:`cyclops.still.of_panel` what is on the glass, and that reads ``image`` and nothing
    else. Without a picture here a ten-minute video is ten minutes of black in ``video.mp4``.
    The page therefore has to check ``video`` *before* ``image``, or it paints the title card
    and never starts anything: see the branch order in ``admin/static/panel.js``.

    ``hold`` is read back by the kiosk through :func:`hold_s`, not by anything in between.
    The panel normally takes itself back after fifteen minutes, which is right for a picture
    nobody dismissed and wrong for a twenty-six-minute video: it would go dark in the middle.

    It announces. A video was asked for out loud and took several seconds to arrive, which is
    the same case a drawing makes - see :func:`announces`.
    """
    global _announce, _hold_s
    _announce = True
    _hold_s = hold
    picture = JPEG_URL + base64.b64encode(thumb).decode("ascii") if thumb else ""
    return _leave({"title": title, "video": url, "start": max(0, int(start_s)), "image": picture})


def keep_still(url: str, ident: str) -> bool:
    """Keep the page's own picture of the scratchpad it just painted. False if it could not.

    The way back for the one thing on the panel that is markup rather than pixels. A recording of
    the screen samples what the kiosk paints, and the kiosk paints nothing while the browser has
    the glass - so everything else on the panel reaches the video out of the offer file, and a
    scratchpad had nothing there to reach it with. The page draws it a second time onto a canvas
    and posts the result here on its way past; :mod:`cyclops.still` is the reader.

    Nothing here rasterises anything, and nothing here can. An ``<iframe sandbox="">`` has an
    origin of its own, so the page around it cannot read what it drew either - what it re-renders
    is the same markup it was handed. This end only writes it down.

    ``ident`` is the offer this is a picture of, and it is read back before the picture is used.
    A still that names a different offer is one the page drew for something already put away.

    Never raises. A still that cannot be written costs a black stretch of recording, which is
    exactly where this started.
    """
    if not url.startswith(JPEG_URL):
        return False
    try:
        card.write_text(PANEL_STILL_FILE, json.dumps({"id": ident, "image": url}))
    except OSError as exc:
        print(f"· could not keep the panel's own picture ({exc})", file=sys.stderr, flush=True)
        return False
    return True


def withdraw() -> None:
    """Take back whatever was last offered, so the page falls back to the dashboard.

    The panel's browser is warm and polling, so an offer left lying about is not inert: the page
    paints it and keeps it up, behind our window, until something says otherwise. Anything that
    is about to hand the panel to that browser for some *other* reason therefore has to clear
    this first, or it uncovers onto a picture from an hour ago - which is exactly what tapping
    the eye did on 2026-09-02, after a throwaway script offered a picture to a panel that was
    never going to show it and exited without tidying up.

    Never raises. A payload that cannot be removed is not a reason to refuse a tap.
    """
    # The offer first and its still second: the still is only ever read against an offer, so an
    # offer that is gone already makes it unreachable. The reverse order would leave a window in
    # which a live offer has no picture of itself.
    for path in (PANEL_FILE, PANEL_STILL_FILE):
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            print(
                f"· could not take the panel's last picture back ({exc})",
                file=sys.stderr,
                flush=True,
            )


def _leave(payload: dict[str, Any]) -> bool:
    """One thing for the panel to show, written where the page is looking. False if it could not.

    The id is minted here rather than by the caller: all it has to do is differ from whatever the
    page last drew, and a stable one would make showing the same thing twice in a row silently do
    nothing the second time.
    """
    try:
        card.write_text(PANEL_FILE, json.dumps({"id": uuid.uuid4().hex[:12], **payload}))
    except OSError as exc:
        print(f"· could not offer the panel something to show ({exc})", file=sys.stderr, flush=True)
        return False
    return True
