"""The panel's physical shutter, and the switch that wakes him: two gestures on one button.

The aperture in the corner of the screen already takes a photo, and on a bench that is the
control a screen is worst at offering. The moment you want a picture of a thing is usually the
moment both hands are busy holding the thing, and a 12 mm target on glass wants to be looked at
before it can be pressed. A button with travel can be found by feel, in the dark, with a knuckle.

The same argument applies twice over to the microphone switch beside it, which is the control
that starts and ends a conversation: the moment you want to start talking to him is the same
moment both hands are full. So the button carries both. A tap lands on
:meth:`cyclops.kiosk.Kiosk.shutter_pressed` and a hold on :meth:`cyclops.kiosk.Kiosk.button_held`,
which are the aperture's own path and the microphone's own path with the coordinates taken out -
the same flash, the same click, the same cues, the same one-at-a-time guards. Neither is a second
implementation of anything, because anything this module added of its own would be a way for the
button and the glass to start drifting apart.

What telling them apart costs is the press edge. A hold and a tap are the same event until
somebody lets go, so the photo now fires on the *release* - it is delayed by however long you
lean on the button. That is the bargain the panel already makes for his eye, which is a tap and a
hold on one target for the same reason, and it is the bargain every phone makes.

Where there is no GPIO to talk to - a Mac, a Pi without the library, a pin something else got to
first - every call here does nothing and the kiosk runs exactly as it did before. That is the
bargain :class:`cyclops.backlight.Backlight` already makes with a panel it cannot dim, and it is
what lets the kiosk build one of these without first asking whether it should.

The wiring it expects (see the hardware brief): the switch between **BCM17** (physical 11) and a
ground pin, read with the Pi's internal pull-up so a press is a pull *down*; the ring's cathode
on **BCM27** (physical 13) and its anode on **3.3 V**, physical 1. The 3.3 V rail matters and is
not interchangeable with the 5 V one beside it: the Pi 5's 5 V header pins feed the PMIC and
stay live after a halt, so a ring wired there would go on glowing on a box you had shut down.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

# How long the switch is given to stop chattering. Cheap metal buttons bounce enough to
# double-fire without it, and 50 ms measured comfortable on the bench: ten deliberate presses,
# no doubles, and the shortest gap between a release and the next press was still 330 ms.
BOUNCE_S = 0.05

# The ring sinks - cathode on the GPIO, anode on 3.3 V - so the pin pulls *down* to light it.
# That is what the brief specifies and it has not been confirmed against the soldered article
# yet: if the ring turns out to be lit whenever it should be dark, this is the single line to
# flip. Nothing else here knows which way round it is.
LED_ACTIVE_HIGH = False

# What the ring says. Three states about the box rather than one about your finger: the press
# already answers itself, with a white flash and a shutter click, and a light that only repeats
# that is a light saying nothing during all the time nobody is touching it.
RING_IDLE = "idle"  # the box is up and nothing is running: dark
RING_ACTIVE = "active"  # a session is live and he is listening: lit
RING_ERROR = "error"  # the last press did not produce a photo: blinking


class ShutterButton:
    """A momentary button on a GPIO pin, and the ring inside it.

    There is one of these whether or not any of it exists. :attr:`available` says whether a real
    pin was claimed and :attr:`note` says why not; every method is safe to call when none was.
    """

    def __init__(
        self,
        *,
        pin: int | None,
        led_pin: int | None,
        hold_s: float,
        on_tap: Callable[[], None],
        on_hold: Callable[[], None],
        on_press: Callable[[], None] | None = None,
        on_release: Callable[[], None] | None = None,
    ) -> None:
        self._on_tap = on_tap
        self._on_hold = on_hold
        # The edges themselves, as opposed to what the press turns out to have meant. A tap and
        # a hold are both decided later - the first on the way up, the second on a clock - and
        # neither can answer a finger at the moment it lands. These two are for what belongs to
        # the finger rather than to the gesture, which so far is the sound under it.
        self._on_press = on_press
        self._on_release = on_release
        # Whether the hold on *this* press already fired. Cleared on the way down rather than on
        # the way up, so a release that never arrives - a wedged switch, a missed edge - cannot
        # leave the latch set and swallow the next tap.
        self._landed = False
        self._button = None
        self._led = None
        self._shown: str | None = None
        self.note = "not configured"
        if pin is None:
            return
        try:
            # Imported here rather than at the top of the module because this is the one file in
            # the package that will not import on a Mac, and the kiosk has to.
            from gpiozero import LED, Button
        except ImportError as exc:
            self.note = f"no gpiozero ({exc})"
            return
        try:
            # gpiozero keeps the hold clock, on a thread of its own, and fires once per press
            # while hold_repeat stays off. Timing it here would mean a second timer thread doing
            # the same arithmetic slightly differently.
            self._button = Button(pin, pull_up=True, bounce_time=BOUNCE_S, hold_time=hold_s)
            if led_pin is not None:
                self._led = LED(led_pin, active_high=LED_ACTIVE_HIGH)
        except Exception as exc:  # noqa: BLE001 - a pin that will not come is a note, not a stop
            self.note = f"BCM{pin} would not open: {exc}"
            self.close()
            return
        # Last, and only once everything above it worked: a handler on a half-built object would
        # be answering presses with an exception.
        self._button.when_pressed = self._down
        self._button.when_held = self._held
        self._button.when_released = self._up
        self.show(RING_IDLE)  # a known state to start from, whatever the last process left
        self.note = f"BCM{pin}" + ("" if self._led is None else f", ring on BCM{led_pin}")

    @property
    def available(self) -> bool:
        return self._button is not None

    def _down(self) -> None:
        """The switch closing. It decides nothing yet - only which press the latch is about."""
        self._landed = False
        if self._on_press is not None:
            self._fire(self._on_press)

    def _held(self) -> None:
        """The press has lasted. Fire the hold now, under the finger, not when it comes off.

        The panel's own long press works the same way: the power menu opens at the threshold
        while his eye is still held down, so the gesture is answered while you can still feel
        yourself making it.
        """
        self._landed = True
        self._fire(self._on_hold)

    def _up(self) -> None:
        """The switch opening. A tap, unless the hold above already had this press.

        The release goes first and unconditionally. It is the end of the finger being down,
        which is true however the press is about to be read - and the sound it stops is one that
        should not outlive the finger by the length of a tap handler.
        """
        if self._on_release is not None:
            self._fire(self._on_release)
        if not self._landed:
            self._fire(self._on_tap)

    def _fire(self, action: Callable[[], None]) -> None:
        """Call one of them, wrapped so nothing raised inside can kill the thread it came on.

        gpiozero has one thread for edges and one for holds, and neither is ever restarted, so
        an exception escaping here would take half the button out for the rest of the run -
        silently, with the only symptom being a gesture that used to work. The kiosk's own
        failures already have somewhere to be said: the caption, and the ring.
        """
        try:
            action()
        except Exception as exc:  # noqa: BLE001 - nothing downstream is worth the thread
            print(f"[button] press handler raised: {exc!r}", file=sys.stderr, flush=True)

    def show(self, state: str) -> None:
        """Put the ring in one of the three states above. Any thread, and free when unchanged.

        The render loop hands this every frame, which is why the comparison comes first: nothing
        should be writing a pin thirty times a second to go on saying the same thing.
        """
        if self._led is None or state == self._shown:
            return
        self._shown = state
        try:
            if state == RING_ACTIVE:
                self._led.on()
            elif state == RING_ERROR:
                # Fast enough to read as an alarm rather than as a slow pulse, and gpiozero
                # keeps time for it on its own thread - the render loop is not asked to.
                self._led.blink(on_time=0.15, off_time=0.15)
            else:
                self._led.off()
        except Exception as exc:  # noqa: BLE001 - a ring that will not light is not a dead panel
            print(f"[button] ring: {exc!r}", file=sys.stderr, flush=True)

    def close(self) -> None:
        """Put the ring out and hand the pins back.

        The ring first: dark is the state this leaves behind, and once the device is closed
        there is nothing left here that could drive the pin either way. A known state on a clean
        shutdown is asked for, and this is the only moment there is one to set.
        """
        for device in (self._led, self._button):
            if device is None:
                continue
            try:
                if device is self._led:
                    device.off()
                device.close()
            except Exception as exc:  # noqa: BLE001 - we are on the way out either way
                print(f"[button] closing: {exc!r}", file=sys.stderr, flush=True)
        self._led = None
        self._button = None
        self._shown = None
        self._landed = False
