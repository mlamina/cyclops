"""The shutter button beside the panel, and the ring in it.

Two things worth pinning down, neither of which needs a pin. The button is a second way into a
path that already existed, so what matters is that it lands on exactly that path and nowhere
else - and that the three cases the touchscreen already distinguishes (asleep, menu up, plain
press) come out the same way when the press arrives on a thread of its own instead.

And the object itself has to survive having no hardware under it, because that is the normal
case everywhere except the Pi: no gpiozero on a Mac, no free pin on a box where the kiosk is
already running. It answers that by doing nothing, and a test says so out loud.

All of it is pure: no GPIO, no camera, no window.
"""

from __future__ import annotations

import time

import pytest

from cyclops import kiosk as kiosk_module
from cyclops import overlay
from cyclops.button import RING_ACTIVE, RING_ERROR, RING_IDLE, ShutterButton


def _panel(monkeypatch: pytest.MonkeyPatch) -> kiosk_module.Kiosk:
    """A kiosk with just enough of itself to answer a press. No pin, no camera, no window."""
    kiosk = object.__new__(kiosk_module.Kiosk)
    kiosk._menu = False
    kiosk._asleep = False
    kiosk._touched_at = 0.0
    kiosk._pressed = None
    kiosk._press_until = 0.0
    kiosk._shutter_error_until = 0.0
    kiosk.did: list[str] = []
    monkeypatch.setattr(kiosk, "_snap", lambda: kiosk.did.append("snap"))
    monkeypatch.setattr(kiosk, "_wake", lambda: kiosk.did.append("wake"))
    return kiosk


# ---------------------------------------------------------------- the press


def test_a_press_takes_a_photo(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole point of it, and the aperture's own path: the same _snap the screen calls."""
    kiosk = _panel(monkeypatch)
    kiosk.shutter_pressed()
    assert kiosk.did == ["snap"]
    assert kiosk._pressed == "shutter", "the aperture on screen should look pressed too"


def test_a_press_on_a_dark_panel_is_spent_waking_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sleeping hands the camera back, so a photo taken now would be of nothing at all.

    The touchscreen refuses for a different reason - you cannot aim at what you cannot see -
    and it does not matter that a button you can find by feel escapes that one. The camera is
    gone either way, and the second press, on a panel that is now awake, is the one that works.
    """
    kiosk = _panel(monkeypatch)
    kiosk._asleep = True
    kiosk.shutter_pressed()
    assert kiosk.did == ["wake"], "a press on a dark panel photographed a released camera"

    kiosk._asleep = False
    kiosk.shutter_pressed()
    assert kiosk.did == ["wake", "snap"]


def test_the_power_menu_swallows_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """It is modal, and two of its three rows end the box. A photo is not an answer to it."""
    kiosk = _panel(monkeypatch)
    kiosk._menu = True
    kiosk.shutter_pressed()
    assert kiosk.did == []


# ---------------------------------------------------------------- the ring


def test_the_ring_says_what_the_box_is_doing(monkeypatch: pytest.MonkeyPatch) -> None:
    kiosk = _panel(monkeypatch)
    assert kiosk._ring_state(overlay.IDLE) == RING_IDLE
    assert kiosk._ring_state(overlay.LISTENING) == RING_ACTIVE
    assert kiosk._ring_state(overlay.STARTING) == RING_ACTIVE, "the tap counts before the session"


def test_a_photo_that_did_not_happen_reaches_the_ring(monkeypatch: pytest.MonkeyPatch) -> None:
    """The failure the shutter can really have, on the half of the panel you need not look at.

    A caption saying the camera stalled is no use to somebody whose head is under a bench with
    their hand on the button - which is the posture the button exists for.
    """
    kiosk = _panel(monkeypatch)
    kiosk._shutter_error_until = time.monotonic() + kiosk_module.NOTICE_S
    assert kiosk._ring_state(overlay.IDLE) == RING_ERROR
    assert kiosk._ring_state(overlay.LISTENING) == RING_ERROR, "the failure outranks the session"

    kiosk._shutter_error_until = 0.0
    assert kiosk._ring_state(overlay.IDLE) == RING_IDLE


# ---------------------------------------------------------------- no hardware under it


def test_no_pin_is_a_button_that_does_nothing() -> None:
    """The normal case off the Pi. Every method still answers, and the kiosk never asks."""
    pressed: list[int] = []
    button = ShutterButton(pin=None, led_pin=None, on_press=lambda: pressed.append(1))
    assert not button.available
    button.show(RING_ACTIVE)  # no ring to light, and no exception either
    button.close()
    assert pressed == []


def test_a_handler_that_raises_cannot_kill_the_thread() -> None:
    """gpiozero calls this on one thread and never starts another.

    An exception let out of here would take the button out for the rest of the run, silently,
    and the only symptom anybody would ever see is a button that used to work.
    """

    def explode() -> None:
        raise RuntimeError("the application action went wrong")

    button = ShutterButton(pin=None, led_pin=None, on_press=explode)
    button._pressed()  # what gpiozero would call; it must come back rather than propagate
