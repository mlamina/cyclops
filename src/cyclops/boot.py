"""The fanfare that says the box is up, sounded from the user session itself.

This does not live in :func:`cyclops.kiosk.main` and it used to. The kiosk is the last thing to
start on this box: labwc has to be up before it is launched at all, and then it spends a second
or two importing OpenCV and the OpenAI client before it reaches its first line. Measured on the
Pi, that put "the box is up" some forty seconds after the kernel started - long after anyone
watching had concluded nothing was going to happen.

The user session is up at about sixteen seconds and PipeWire with it, which is the earliest
moment this machine can make any sound at all, and it is also the honest one: what the cue says
is that Linux is running, not that the panel is ready. ``cyclops_boot_sequence_finished`` is the
one that answers for the panel, and the kiosk still sounds that itself.

Earlier than this would mean going around PipeWire to the ALSA device directly, which is
possible - the USB speaker enumerates at one second - and is a bad trade: a fourteen-second
sound holding the card is a sound that can still be playing when PipeWire comes looking for it,
and losing that race costs the box its audio for the whole session rather than costing it a
fanfare.
"""

from __future__ import annotations

import sys
import time

from . import mixer, sfx
from .audio import SAMPLE_RATE, resolve_device
from .config import ConfigError, load_settings

SINK_WAIT_S = 25.0  # how long to keep asking; past this the box simply came up quietly
SINK_POLL_S = 0.25


def wait_for_sink(timeout: float = SINK_WAIT_S) -> bool:
    """Block until PipeWire can name a default sink. False if it never does.

    Ordering after ``pipewire.service`` says the daemon has started, not that WirePlumber has
    finished finding the speaker - and a cue played into that gap goes nowhere without failing,
    which is the one way this could be broken and look fine. :func:`cyclops.mixer.level` is
    already the careful way this project asks PipeWire a question, so it is what does the
    asking here.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if mixer.level() is not None:
            return True
        time.sleep(SINK_POLL_S)
    return False


def main() -> None:
    """Sound the boot cue, once, and stay alive for as long as it lasts.

    ``sd.play`` returns immediately, so leaving here would cut the fanfare off in its first
    millisecond - the sleep *is* the playback. Nothing in here is worth failing a unit over: a
    box with no settings, no speaker or no sound files still boots, it just boots quietly.
    """
    try:
        settings = load_settings(require_api_key=False)  # a fanfare has no business with a key
    except ConfigError as exc:
        print(f"· no settings, so no fanfare: {exc}", file=sys.stderr, flush=True)
        return
    if not settings.sounds:
        return
    if not wait_for_sink():
        note = f"· no sink in {SINK_WAIT_S:g}s; the box came up quietly"
        print(note, file=sys.stderr, flush=True)
        return
    seconds = sfx.play("booted", rate=SAMPLE_RATE, device=resolve_device(settings.output_device))
    print(f"· the box is up ({seconds:.1f}s of saying so)", flush=True)
    time.sleep(seconds + sfx.SETTLE_S)


if __name__ == "__main__":
    main()
