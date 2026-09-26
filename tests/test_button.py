"""The button beside the panel: its two gestures, and the ring in it.

Three things worth pinning down, none of which needs a pin. The button is a second way into two
paths that already existed - the aperture on a tap, the microphone switch on a hold - so what
matters is that each lands on exactly its own path and nowhere else, and that the cases the
touchscreen already distinguishes (asleep, menu up, plain press) come out right when the press
arrives on a thread of its own instead. The one place the two deliberately part company is a
dark panel, and that has a test to itself.

Then the latch, which is the only real logic the module gained: a press that became a hold must
not also arrive as a tap when the finger finally comes off. Beside it now, the grace: a press
answers a finger only once that finger has been down long enough to mean something, so a click
is a shutter and nothing else.

And the three sounds of a long press, which the module and the kiosk own one edge of each: the
rise starts at the grace, ends at the hold boundary rather than at the lift, and the gears take
over on that same edge.

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


class Cues:
    """The speaker, written down instead of made. `stop_if` has to be the real rule rather than
    a counter: what it is for is a stop that lands only while its own cue is still the one
    sounding, and a fake that always stops cannot fail the way the real one can."""

    def __init__(self) -> None:
        self.played: list[str] = []
        self.stopped = 0

    def play(self, name: str, *, loop: bool = False) -> float:
        self.played.append(name)
        return 0.0

    def stop(self) -> None:
        self.played.append(None)  # nothing is sounding now, which stop_if has to be able to see
        self.stopped += 1

    def stop_if(self, name: str) -> None:
        if self.played and self.played[-1] == name:
            self.stop()

    @property
    def heard(self) -> list[str]:
        return [name for name in self.played if name is not None]


class Controller:
    """Just enough session controller for _toggle_session to get through it."""

    settings = None

    def __init__(self, state: str = "idle") -> None:
        self.state = state
        self.started = 0

    def status(self) -> dict[str, object]:
        return {"state": self.state}

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        pass


def _panel(monkeypatch: pytest.MonkeyPatch, *, real_toggle: bool = False) -> kiosk_module.Kiosk:
    """A kiosk with just enough of itself to answer a press. No pin, no camera, no window."""
    kiosk = object.__new__(kiosk_module.Kiosk)
    kiosk._menu = False
    kiosk._asleep = False
    kiosk._touched_at = 0.0
    kiosk._pressed = None
    kiosk._press_until = 0.0
    kiosk._shutter_error_until = 0.0
    kiosk._pending = None
    kiosk._pending_at = 0.0
    kiosk._cues = Cues()
    kiosk.did: list[str] = []
    monkeypatch.setattr(kiosk, "_snap", lambda: kiosk.did.append("snap"))
    monkeypatch.setattr(kiosk, "_wake", lambda: kiosk.did.append("wake"))
    if real_toggle:
        # The gears live inside _toggle_session, in the branch that knows a session is starting,
        # so the one test about them has to let the real thing run.
        kiosk.controller = Controller()
    else:
        monkeypatch.setattr(kiosk, "_toggle_session", lambda: kiosk.did.append("toggle"))
    return kiosk


def _pinless(**handlers: object) -> ShutterButton:
    """A ShutterButton with no pin under it, so its callbacks can be driven by hand."""
    return ShutterButton(pin=None, led_pin=None, hold_s=0.7, **handlers)


GRACE_S = 0.01  # the real one is 0.2; this test only cares that there is one


def _graced(**handlers: object) -> ShutterButton:
    return ShutterButton(pin=None, led_pin=None, hold_s=0.7, press_after_s=GRACE_S, **handlers)


# ------------------------------------------------------------------ the tap


def test_a_press_takes_a_photo(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole point of it, and now the only path: the glass has no aperture on it any more.

    Nothing is lit on the panel to say so, and nothing needs to be. The flash and the click
    belong to _snap, they are what a shutter has always answered with, and the two discs that
    used to invert under a thumb are a volume knob and a heat gauge - neither of which has
    anything to say about a photograph.
    """
    kiosk = _panel(monkeypatch)
    kiosk.shutter_pressed()
    assert kiosk.did == ["snap"]
    assert kiosk._pressed is None, "nothing on the glass is this button's twin any more"


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
    """It is modal, and two of its rows end the box. A photo is not an answer to it."""
    kiosk = _panel(monkeypatch)
    kiosk._menu = True
    kiosk.shutter_pressed()
    assert kiosk.did == []


# ----------------------------------------------------------------- the hold


def test_a_hold_wakes_him_and_puts_him_back_to_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """The only way into a session from the panel now, and still the one _toggle_session.

    What says it landed is the ring in the button, the halo round the picture and the strip
    saying STARTING - all of which _pending has true before this returns. The microphone that
    used to light up beside the aperture is a heat gauge.
    """
    kiosk = _panel(monkeypatch)
    kiosk.button_held()
    assert kiosk.did == ["toggle"]
    assert kiosk._pressed is None, "nothing on the glass is this button's twin any more"


def test_a_hold_on_a_dark_panel_lands_as_well_as_lighting_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one place the two gestures part company, and the reason is not symmetry.

    A tap under a black screen is ambiguous and is spent waking it. A hold cannot be: it can
    only have meant one thing. It has to light the glass itself, though - _sleeping never clears
    _asleep on its own, so the session would otherwise run its whole length behind a dark panel.
    """
    kiosk = _panel(monkeypatch)
    kiosk._asleep = True
    kiosk.button_held()
    assert kiosk.did == ["wake", "toggle"], "a session started behind a panel nobody lit"


def test_the_power_menu_swallows_a_hold_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """Modal is modal. Two of its rows end the box; starting a session is no answer."""
    kiosk = _panel(monkeypatch)
    kiosk._menu = True
    kiosk.button_held()
    assert kiosk.did == []


# ---------------------------------------------------------------- the latch


def test_a_hold_does_not_also_arrive_as_a_tap() -> None:
    """What the release edge costs, and the only real logic in the module.

    Letting go is what ends both gestures, so without the latch every hold would be followed by
    the photo it was not: press, hold, release, snap.
    """
    did: list[str] = []
    button = _pinless(on_tap=lambda: did.append("tap"), on_hold=lambda: did.append("hold"))

    button._down()
    button._held()
    button._up()
    assert did == ["hold"]

    # ...and the next press is a tap again: the latch is cleared on the way down, not up, so a
    # release that never arrived cannot go on swallowing taps for the rest of the run.
    button._down()
    button._up()
    assert did == ["hold", "tap"]


# ---------------------------------------------------------------- the grace, and the rise


def test_a_click_never_reaches_the_press_handler() -> None:
    """What the grace is for. A photo should be a shutter and nothing else, and the rise that
    fills the wait for a hold cannot start on the press edge without putting a smear of itself
    in front of every picture anybody takes."""
    did: list[str] = []
    button = _graced(on_tap=lambda: None, on_hold=lambda: None,
                     on_press=lambda: did.append("press"))
    button._down()
    button._up()
    time.sleep(4 * GRACE_S)
    assert did == []


def test_a_finger_that_settles_does_reach_it() -> None:
    did: list[str] = []
    button = _graced(on_tap=lambda: None, on_hold=lambda: None,
                     on_press=lambda: did.append("press"))
    button._down()
    time.sleep(4 * GRACE_S)
    assert did == ["press"]
    button._up()


def test_a_hold_ends_the_rise_and_sounds_the_gears(monkeypatch: pytest.MonkeyPatch) -> None:
    """The three sounds of a long press, and the two edges between them.

    The rise stops because the boundary arrived, not because the finger came off: it used to
    run until the lift, which is a cue outliving the thing it was counting. And the gears are
    sounded from this thread, on this edge - they used to come off the session's own thread once
    that had imported an agent and opened PortAudio both ways, half a second to two seconds
    later and never twice the same.
    """
    kiosk = _panel(monkeypatch, real_toggle=True)
    kiosk.button_down()
    kiosk.button_held()
    assert kiosk._cues.heard == ["button_pressed", "gears"]
    assert kiosk._cues.stopped == 1, "the rise ended on the boundary"
    assert kiosk.controller.started == 1

    # ...and the lift that follows has nothing left to stop, which is the whole of why the stop
    # is conditional: by now the gears are what is sounding and they are not the finger's.
    kiosk.button_up()
    assert kiosk._cues.stopped == 1


def test_the_press_that_ends_a_session_is_silent_until_it_lands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rise promises a machine about to run, and this gesture stops one.

    Same button, same hold, and the same gears on the boundary - what tells the two apart is the
    lid that follows, not the sound under the finger. Putting a rise here would be the box
    getting more eager as you switch it off.
    """
    kiosk = _panel(monkeypatch, real_toggle=True)
    kiosk.controller.state = "listening"
    kiosk.button_down()
    assert kiosk._cues.heard == [], "nothing is winding up"
    kiosk.button_held()
    assert kiosk._cues.heard == ["gears"]


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
    did: list[str] = []
    button = _pinless(on_tap=lambda: did.append("tap"), on_hold=lambda: did.append("hold"))
    assert not button.available
    button.show(RING_ACTIVE)  # no ring to light, and no exception either
    button.close()
    assert did == []


def test_a_handler_that_raises_cannot_kill_the_thread() -> None:
    """gpiozero has one thread for edges and one for holds, and never starts another of either.

    An exception let out of either would take half the button out for the rest of the run,
    silently, and the only symptom anybody would ever see is a gesture that used to work.
    """

    def explode() -> None:
        raise RuntimeError("the application action went wrong")

    button = _pinless(on_tap=explode, on_hold=explode)
    # What gpiozero would call. Both must come back rather than propagate.
    button._down()
    button._up()
    button._down()
    button._held()
