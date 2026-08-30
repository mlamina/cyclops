"""Synthesized sound cues: the audio half of the kiosk's chrome.

The panel says what the session is doing in colour - amber while the link opens, green once it
is up, a white flash for the shutter (:mod:`cyclops.overlay`) - which is no use at all to
someone with their head under a bench. These are the same events, said out loud: a ping while
we connect, a chime when the agent is listening, a click on the shutter, and for the end of a
session a falling pair on the press, ticks while it winds down, and one low note when it is
really over.

The cues are generated rather than shipped as ``.wav`` files. A few lines of numpy beats binary
assets in the repo, needs no path to resolve on the Pi, and lets a cue be retuned by editing a
number. Adding one is a single entry in :data:`CUES`.

They play on their own output stream rather than through :class:`cyclops.audio.Speaker`, which
is the whole reason this module is short. That buffer is entangled with four other things: the
:class:`~cyclops.audio.EchoGuard` calibrates its echo estimate from whatever it sees playing,
and a pure tone couples through a room on completely different terms than a voice; the
begin_item/flush accounting behind ``conversation.item.truncate`` counts every frame fed to it;
``is_audible`` is what makes the panel say SPEAKING; and its playback clock stops advancing on
underrun, so marking cue frames within it invites a latch that would leave the kiosk deaf.
Sharing that buffer means defending all four. Sitting beside it means none of them can see us -
``sounddevice.stop()`` is documented to have "no influence on streams created with ...
RawOutputStream", and a Speaker is exactly that.

What we give up by sitting beside it is the recording: :class:`cyclops.record.SessionRecorder`
taps the Speaker, so cues are not in the session's video. That is the trade, and it is the
right way round - a recording of the conversation is worth more than a recording of the beeps.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from functools import cache

import numpy as np
import sounddevice as sd

# A sine has ~3 dB of crest factor against speech's ~12, so a full-scale tone lands far louder
# than the voice it sits beside. A quarter of full scale puts them in the same room.
PEAK = 0.25
FADE_MS = 5.0  # ramp in and out of every tone; a hard edge clicks through a hardware speaker
SETTLE_S = 0.1  # PortAudio takes a moment to open a stream: a caller waiting a cue out adds this


def _envelope(n: int, rate: int, fade_ms: float = FADE_MS) -> np.ndarray:
    """A raised-cosine ramp at each end. Linear would still leave a corner in the first
    derivative, which a small speaker turns into a thump rather than a fade."""
    env = np.ones(n, dtype=np.float32)
    edge = min(int(rate * fade_ms / 1000), n // 2)
    if edge:
        ramp = (1 - np.cos(np.linspace(0, np.pi, edge, dtype=np.float32))) / 2
        env[:edge] = ramp
        env[-edge:] = ramp[::-1]
    return env


def tone(
    freq: float, ms: float, *, rate: int, amp: float = PEAK, fade_ms: float = FADE_MS
) -> np.ndarray:
    t = np.arange(int(rate * ms / 1000), dtype=np.float32) / rate
    return np.sin(2 * np.pi * freq * t) * amp * _envelope(len(t), rate, fade_ms)


def silence(ms: float, *, rate: int) -> np.ndarray:
    return np.zeros(int(rate * ms / 1000), dtype=np.float32)


def knock(ms: float, *, rate: int, amp: float = PEAK, seed: int = 7) -> np.ndarray:
    """A noise burst under a fast decay: a mechanical click, which no sine can be.

    The seed is fixed, so a cue built from this renders identically on every machine and every
    run - it is cached and compared like any other, and a shutter that came out differently
    each time would be a strange thing to own.
    """
    n = int(rate * ms / 1000)
    grain = np.random.default_rng(seed).standard_normal(n).astype(np.float32)
    decay = np.exp(-np.linspace(0.0, 6.0, n, dtype=np.float32))
    return np.clip(grain * decay, -1.0, 1.0) * amp * _envelope(n, rate)


def join(*parts: np.ndarray) -> np.ndarray:
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def repeat(part: np.ndarray, times: int) -> np.ndarray:
    return np.tile(part, times)


# The chrome is a Pip-Boy terminal, one phosphor hue and no ornament, so the cues are the same
# instrument: oscillators, no samples, no reverb. A handshake, not a notification.
#
# Add a cue by adding a line. It is a function of the sample rate and nothing else.
CUES: dict[str, Callable[[int], np.ndarray]] = {
    # Two blips and a long gap, looped: the gap is what keeps a cue you are meant to ignore for
    # several seconds from becoming a drone in a small workshop.
    "connecting": lambda rate: join(
        tone(660, 60, rate=rate),
        silence(90, rate=rate),
        tone(660, 60, rate=rate),
        silence(890, rate=rate),
    ),
    # A rising fifth: the resolution the pinging was asking for.
    "ready": lambda rate: join(tone(740, 130, rate=rate), tone(1110, 130, rate=rate)),
    # Ending a session is three things heard as one gesture, and this cue is the first two of
    # them: a falling pair the instant you press stop, then soft ticks while the link winds
    # down. The ticks run far longer than the ~2.3 s teardown usually takes, because they are a
    # budget rather than a duration - "ended" cuts them off the moment the session is really
    # over (one cue at a time; see Cues), so the length only decides how long a slow teardown
    # keeps saying something rather than falling silent.
    "closing": lambda rate: join(
        tone(740, 90, rate=rate),
        tone(494, 110, rate=rate),
        silence(120, rate=rate),
        repeat(join(tone(392, 40, rate=rate, amp=PEAK * 0.5), silence(360, rate=rate)), 15),
    ),
    # ...and the third: one low note, the lowest thing here, with a slow release. The machine
    # has gone quiet. Distinct in register from both the falling pair and the ticks above it,
    # because its whole job is to be recognised as the end rather than as more of the middle.
    "ended": lambda rate: join(tone(247, 300, rate=rate, amp=PEAK * 0.7, fade_ms=90)),
    # Click-clack, the second knock softer, the way a mechanical shutter actually sounds. It
    # is the flash the kiosk paints at the same moment (cyclops.kiosk._snap), said out loud.
    "shutter": lambda rate: join(
        knock(16, rate=rate),
        silence(38, rate=rate),
        knock(26, rate=rate, amp=PEAK * 0.6, seed=11),
    ),
}


@cache
def render(name: str, rate: int) -> np.ndarray:
    """The cue as int16 mono at ``rate``, built once per process.

    Clipping happens here, before the cast: ``astype(np.int16)`` wraps on overshoot, which
    turns a tone that is slightly too loud into a full-scale square burst.
    """
    if (build := CUES.get(name)) is None:
        raise KeyError(f"no such cue: {name!r} (have {', '.join(sorted(CUES))})")
    pcm = (np.clip(build(rate), -1.0, 1.0) * 32767.0).astype(np.int16)
    pcm.setflags(write=False)  # cached and handed out: nobody gets to edit it in place
    return pcm


def play(name: str, *, rate: int, device: int | str | None = None, loop: bool = False) -> float:
    """Sound a cue on its own stream. Returns how long it will sound, in seconds.

    ``sd.play`` stops whatever it was playing first, so a one-shot also ends a looping cue -
    "stop the ping and sound the chime" is this call and nothing else. A cue that cannot play
    is not worth taking a session down for, so a device that refuses is reported and shrugged
    off; the caller reads that as zero seconds of sound.
    """
    pcm = render(name, rate)
    try:
        sd.play(pcm, samplerate=rate, device=device, loop=loop)
    except Exception as exc:  # noqa: BLE001 - any PortAudio trouble here is a missing beep
        print(f"[sfx] {name} did not play: {exc}", file=sys.stderr, flush=True)
        return 0.0
    return len(pcm) / rate


def stop() -> None:
    """End whatever cue is playing. Cannot touch the voice: sounddevice's own docs say this
    "has no influence on streams created with ... RawOutputStream", which is what a Speaker
    is. It already swallows its own errors."""
    sd.stop()


class Cues:
    """Somewhere to sound cues from: the device and the on/off switch, decided once.

    Whoever holds one of these can make a noise without knowing where the speaker is or
    whether the user wanted to hear it - which is the point, because the two things that make
    a sound (the agent, for the link; the kiosk, for the shutter) have nothing else in common.

    **One cue sounds at a time.** A new one replaces whatever was playing, so snapping a photo
    while the connecting ping is still looping ends the ping early. With a single small speaker
    that is the better failure: two beeps at once sound like a fault, not like two events.
    """

    def __init__(self, *, rate: int, device: int | str | None = None, enabled: bool = True):
        self._rate = rate
        self._device = device
        self.enabled = enabled

    def play(self, name: str, *, loop: bool = False) -> float:
        """Sound a cue, and say how long it will sound. Zero when sounds are off, which is
        what lets a caller wait a cue out without asking whether there was one."""
        if not self.enabled:
            return 0.0
        return play(name, rate=self._rate, device=self._device, loop=loop)

    def stop(self) -> None:
        if self.enabled:
            stop()
