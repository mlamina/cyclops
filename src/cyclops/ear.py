"""Listening for somebody to start talking, while no session is up.

Plugged in should mean ready to talk: no press first. So while the session is down the kiosk
keeps one of these open on the mic and runs a small speech detector over it (WebRTC's VAD -
free, local, a few microseconds a block on the Pi). When it hears about a third of a second of
speech inside half a second, it says so, and the kiosk starts a session the way the long press
does - except the ear is handed to it, so what was said before and during the connect reaches
the model as the first turn and nothing has to be repeated.

What it hears stays here. The last :data:`PREROLL_S` live in memory, overwritten block by block;
nothing is written to the card, logged or sent anywhere until a session has started. A session
that never starts takes it all with it when the ear is closed.

It cannot tell *who* is talking or to whom - that would be a wake word. What it can do is not
be fooled by the box itself: the kiosk holds it deaf while any cue is sounding, for a second
after one stops and for a few after a session ends (:attr:`Ear.deaf`). A clang, a sander or
music with nobody singing does not have the shape of speech it waits for.
"""

from __future__ import annotations

import sys
import threading
from collections import deque
from collections.abc import Callable

import numpy as np
import sounddevice as sd
import webrtcvad

from .audio import BLOCK_FRAMES, CHANNELS, DTYPE, SAMPLE_RATE, Compressor, _rms, pin_input_source

VAD_RATE = 16_000  # one of the four rates WebRTC's VAD takes; 24 kHz is not one of them
VAD_FRAMES = BLOCK_FRAMES * VAD_RATE // SAMPLE_RATE  # 20 ms at 16 kHz: a length it takes
AGGRESSIVENESS = 3  # its strictest: a missed "hey" costs a repeat, a false start costs a session
WINDOW_BLOCKS = 25  # half a second of 20 ms blocks...
SPEECH_BLOCKS = 15  # ...of which this many must be speech: 0.3 s of it
# Below this a block is not counted as speech whatever the VAD says: a voice from across the
# shop, or the next room, is not somebody talking to it. Levelled int16 RMS, the same scale as
# Microphone.level; normal talking at arm's length is in the thousands.
SPEECH_FLOOR = 300.0
# WebRTC's VAD calls loud broadband noise speech - a sander, a hiss of air. Noise like that is
# flat across the spectrum and a voice is not, so a block flatter than this is not counted.
# Measured on 20 ms blocks, 200 Hz - 7.5 kHz: a voice's median is near 0.003 and its 90th
# percentile 0.13; white noise sits around 0.56 and pink around 0.36. Low rumble (brown noise)
# is as peaked as a voice and still gets through - see the PR for arc-eng/hq#561.
MAX_FLATNESS = 0.25
PREROLL_S = 2.0  # how much of what came before the trigger is kept, in memory only
PREROLL_BLOCKS = int(PREROLL_S * SAMPLE_RATE / BLOCK_FRAMES)
HANDOVER_MAX_S = 15.0  # a connect slower than this is a failed one; stop collecting
HANDOVER_BLOCKS = int(HANDOVER_MAX_S * SAMPLE_RATE / BLOCK_FRAMES)

_WINDOW = np.hanning(BLOCK_FRAMES).astype(np.float32)
_BAND = slice(4, 150)  # rfft bins of a 480-sample block at 24 kHz: 50 Hz apart, so 200 Hz - 7.5 kHz
_SOURCE = np.arange(BLOCK_FRAMES, dtype=np.float32)
_TARGET = np.linspace(0, BLOCK_FRAMES - 1, VAD_FRAMES, dtype=np.float32)


def to_vad_rate(block: bytes) -> bytes:
    """A 24 kHz block as 16 kHz, by straight interpolation - plenty for deciding *is it speech*."""
    samples = np.frombuffer(block, dtype=np.int16).astype(np.float32)
    return np.interp(_TARGET, _SOURCE, samples).astype(np.int16).tobytes()


def flatness(block: bytes) -> float:
    """How noise-like a block is: 1 for a flat spectrum, near 0 for a voice's peaked one."""
    samples = np.frombuffer(block, dtype=np.int16).astype(np.float32) * _WINDOW
    power = np.abs(np.fft.rfft(samples))[_BAND] ** 2 + 1e-9
    return float(np.exp(np.mean(np.log(power))) / np.mean(power))


class SpeechGate:
    """Says when a run of 24 kHz blocks has started to sound like somebody talking.

    No audio device here, so it is the part the tests drive.
    """

    def __init__(self, *, aggressiveness: int = AGGRESSIVENESS) -> None:
        self._vad = webrtcvad.Vad(aggressiveness)
        self._window: deque[bool] = deque(maxlen=WINDOW_BLOCKS)

    def feed(self, block: bytes) -> bool:
        """Judge one block. True once the window holds enough speech to start on."""
        if len(block) != BLOCK_FRAMES * 2:
            return False  # a short block from a stream starting or stopping; not worth judging
        voiced = (
            _rms(block) >= SPEECH_FLOOR
            and flatness(block) <= MAX_FLATNESS
            and self._vad.is_speech(to_vad_rate(block), VAD_RATE)
        )
        self._window.append(voiced)
        return sum(self._window) >= SPEECH_BLOCKS

    def reset(self) -> None:
        self._window.clear()


class Ear:
    """The mic, open while no session is, and the gate run over it on the audio thread.

    Built and closed by the kiosk; once it has heard somebody it is handed to the session, whose
    agent takes what it collected (:meth:`take`) the moment the model can hear, and then closes
    it. Levelled the way :class:`cyclops.audio.Microphone` levels, so the first turn sounds like
    the rest of the session.
    """

    def __init__(
        self,
        *,
        device: int | str | None = None,
        gain_db: float = 0.0,
        compress: bool = True,
    ) -> None:
        self._compressor = (
            Compressor(gain_db=gain_db, compress=compress) if gain_db or compress else None
        )
        self._gate = SpeechGate()
        self._lock = threading.Lock()
        self._before: deque[bytes] = deque(maxlen=PREROLL_BLOCKS)
        self._after: list[bytes] = []
        self._taken = False
        self._closed = False
        # Asked on the audio thread, every block: is the box itself making a sound, or just has?
        # A callable rather than a flag the kiosk sets each frame, because a frame is a quarter
        # of a second behind a dark panel, and most of a cue fits in that.
        self.deaf: Callable[[], bool] = lambda: False
        self.heard = threading.Event()  # set on the audio thread, read on the render thread
        self.source = pin_input_source(device)
        self._stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_FRAMES,
            channels=CHANNELS,
            dtype=DTYPE,
            device=device,
            callback=self._callback,
        )

    def _callback(self, indata, frames: int, time_info, status) -> None:
        block = bytes(indata)  # copy: PortAudio reuses the buffer
        if self._compressor is not None:
            block = self._compressor.process(block)
        with self._lock:
            if self._taken:
                return
            if self.heard.is_set():
                if len(self._after) < HANDOVER_BLOCKS:
                    self._after.append(block)
                return
            self._before.append(block)
        if self.deaf():
            self._gate.reset()
            return
        if self._gate.feed(block):
            self.heard.set()

    def start(self) -> None:
        self._stream.start()

    def take(self) -> bytes:
        """Everything kept from before the trigger up to now, and stop keeping any more.

        Cut under the lock and the stream left to :meth:`close`: whoever takes this is about to
        drain the session's own mic, and the nearer those two moments are the less of the room
        arrives twice or not at all.
        """
        with self._lock:
            self._taken = True
            pcm = b"".join([*self._before, *self._after])
            self._before.clear()
            self._after.clear()
        return pcm

    def close(self) -> None:
        """Stop the stream and drop whatever was kept. Safe to call twice, from any thread."""
        with self._lock:
            if self._closed:
                return
            self._closed = self._taken = True
            self._before.clear()
            self._after.clear()
        try:
            self._stream.stop()
            self._stream.close()
        except Exception as exc:  # noqa: BLE001 - a device already gone is still closed
            print(f"[ear] closing the mic: {exc}", file=sys.stderr, flush=True)
