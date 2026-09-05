"""The panel handshake: one thing at a time, offered to the page and taken back afterwards.

Everything Cyclops can put on the glass comes through here - a photo off the shutter, a picture
recalled from the card, an edit, a diagram, a scratchpad of HTML the model wrote itself - and it is
one small file plus a flag. What is to be shown is written to :data:`cyclops.config.PANEL_FILE`;
the panel's page polls ``/api/panel`` every 400 ms, sees an id it has not drawn, fetches the
payload and paints it; :func:`show` asks the kiosk to uncover the browser once it has.

There are two doors, and they differ only in what they leave in the file. :func:`offer_image`
carries a JPEG, and :func:`offer_scratchpad` carries markup. The page branches on which key is
there and on nothing else; nothing between here and there - not ``_leave``, not ``/api/picture``,
not the kiosk - looks inside the payload at all. Whatever is up, a press anywhere puts it away.

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
from .config import PANEL_FILE

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


def announces() -> bool:
    """True when what is waiting for the panel should sound the "shown" cue as it lands.

    The panel asks so that it can stay quiet for some arrivals and not others, and the rule is
    whether anything else already said so. A photograph had the shutter a beat earlier and a
    second sound on top of it is one too many. A drawing had nothing: it was asked for out loud,
    it takes a minute and a half to arrive, and in between there is only a strip of text on a
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
    url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
    global _announce
    _announce = announce
    return _leave({"title": title, "image": url})


def offer_scratchpad(html: str) -> bool:
    """Leave a scratchpad of HTML where the panel's page will find it. False if it could not.

    The other door onto the glass, and the cheap one. Everything above arrives as pixels somebody
    made - a camera, or a minute and a half of ``gpt-image-2`` - and this arrives as markup the
    model wrote itself in the time it takes to say a sentence. A number, a short list, a word worth
    reading rather than hearing, a small SVG.

    It is silent. :func:`announces` is about whether *anything else* said this had arrived, and a
    scratchpad is the one case where something did: the model is still talking when it lands. A
    drawing sounds because it took ninety seconds and nothing else marked it.

    No ``drawn`` key, and that is the whole of how it is dismissed. The page reads ``drawn`` to
    decide between a corner square and a press anywhere, and a scratchpad gets the press: it is one
    thing to look at and be done with rather than a diagram to point at while you argue about it.

    No title either, unlike :func:`offer_image`. Nothing has ever read that field - not the page,
    not :mod:`cyclops.still` - and here it would be a second argument the model has to write before
    anything can appear, which is the one cost this feature exists to avoid. The session log names
    it from the words in what he wrote (see ``cyclops.session._render_record``).
    """
    global _announce
    _announce = False
    return _leave({"scratchpad": html})


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
    try:
        PANEL_FILE.unlink(missing_ok=True)
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
