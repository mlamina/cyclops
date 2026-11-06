"""Listening for somebody to start talking, while no session is up.

Plugged in should mean ready to talk: no press first. So while the session is down the kiosk
keeps one of these open on the mic and runs a small speech detector over it (WebRTC's VAD -
free, local, a few microseconds a block on the Pi). That detector calls most sound speech - 90%
of a set of recorded noises (arc-eng/hq#609) - so all it does is wake a second check: a small
neural speech detector (Silero, :class:`SpeechCheck`), also local, which has to hear steady
speech before anything happens. Only then does the kiosk start a session, the way the long press
does - except the ear is handed to it, so what was said before and during the connect reaches
the model as the first turn and nothing has to be repeated.

What it hears stays here. The last :data:`PREROLL_S` live in memory, overwritten block by block;
nothing is written to the card, logged or sent anywhere until a session has started. A session
that never starts takes it all with it when the ear is closed.

It cannot tell *who* is talking or to whom - that would be a wake word. What it can do is not
be fooled by the box itself: the kiosk holds it deaf while any cue is sounding, for a second
after one stops and for half of one after a session ends (:attr:`Ear.deaf`).
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

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

# The second check. Silero takes 512 samples at 16 kHz, 32 ms: about 0.3 ms of a core on the Pi.
CHECK_MODEL = Path(__file__).parent / "assets" / "silero_vad.onnx"
CHECK_FRAME = 512
CHECK_THRESHOLD = 0.7  # how sure a frame must be; with the run below, 1.3% of noise clips got by
CHECK_RUN = 12  # this many frames in a row, 0.38 s. Longer starts losing one-word starts.
CHECK_S = 1.5  # after this long with no steady speech it was a noise: back to the first check
CHECK_BLOCKS = int(CHECK_S * SAMPLE_RATE / BLOCK_FRAMES)
NOISE_LOG_EVERY_S = 60.0  # a noisy room would otherwise fill the log, and a count says as much

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


@lru_cache(maxsize=1)
def _check_model():
    """The detector, loaded once for the life of the process. Importing its runtime takes over a
    second on the Pi, so this only ever runs on an ear's worker thread, never a frame's."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1  # CPU matters: it's a fan
    return ort.InferenceSession(str(CHECK_MODEL), options, providers=["CPUExecutionProvider"])


class SpeechCheck:
    """Says when a run of 24 kHz blocks holds steady speech, by Silero's reckoning.

    One per attempt: the model carries state from frame to frame, and an attempt starts clean.
    """

    def __init__(self) -> None:
        self._model = _check_model()
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(64, dtype=np.float32)  # the model expects the last frame's tail
        self._pending = np.zeros(0, dtype=np.float32)
        self._run = 0

    def feed(self, block: bytes) -> bool:
        """Judge one block. True once enough frames in a row were speech."""
        samples = np.frombuffer(to_vad_rate(block), dtype=np.int16).astype(np.float32) / 32768
        self._pending = np.concatenate([self._pending, samples])
        passed = False
        while self._pending.size >= CHECK_FRAME:
            frame, self._pending = self._pending[:CHECK_FRAME], self._pending[CHECK_FRAME:]
            out, self._state = self._model.run(None, {
                "input": np.concatenate([self._context, frame])[None, :],
                "state": self._state,
                "sr": np.array(VAD_RATE, dtype=np.int64),
            })
            self._context = frame[-64:]
            self._run = self._run + 1 if float(out[0][0]) >= CHECK_THRESHOLD else 0
            passed = passed or self._run >= CHECK_RUN
        return passed


class Ear:
    """The mic, open while no session is, and the gate run over it on the audio thread.

    The gate wakes the second check, which runs on a thread of its own: the audio thread has only
    a few milliseconds a block, and it hands over two seconds of sound at once. It keeps what it
    heard while the check decides, so a start that comes later loses none of the first words.

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
        self._before: deque[bytes] = deque()  # PREROLL_BLOCKS, and whatever a check has added
        self._after: list[bytes] = []
        self._taken = False
        self._closed = False
        self._attempt = 0  # which check the worker's verdicts count for; 0 is none running
        self._attempts = 0
        self._attempt_blocks = 0
        self._noises = 0  # checks that came to nothing, since the last line about them
        self._noise_logged_at = time.monotonic()
        self._jobs: queue.Queue[tuple[int, list[bytes]] | None] = queue.Queue()
        # Asked on the audio thread, every block: is the box itself making a sound, or just has?
        # A callable rather than a flag the kiosk sets each frame, because a frame is a quarter
        # of a second behind a dark panel, and most of a cue fits in that.
        self.deaf: Callable[[], bool] = lambda: False
        # Is its own voice sounding right now? What's heard then isn't kept either: with a short
        # hold, it would otherwise reach the model as the start of his first turn.
        self.loud: Callable[[], bool] = lambda: False
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
            if self.loud():
                self._before.clear()
            else:
                self._before.append(block)
                if not self._attempt:
                    while len(self._before) > PREROLL_BLOCKS:
                        self._before.popleft()
            checking = bool(self._attempt)
        if self.deaf():
            self._gate.reset()
            if checking:
                self._end_attempt()
            return
        if checking:
            self._attempt_blocks += 1
            if self._attempt_blocks > CHECK_BLOCKS:
                self._end_attempt(noise=True)
            else:
                self._jobs.put((self._attempt, [block]))
            return
        if self._gate.feed(block):
            with self._lock:
                self._attempts += 1
                self._attempt = self._attempts
                self._attempt_blocks = 0
                self._jobs.put((self._attempt, list(self._before)))

    def _end_attempt(self, *, noise: bool = False) -> None:
        self._gate.reset()
        with self._lock:
            self._attempt = 0
        if noise:
            self._noises += 1
            now = time.monotonic()
            if now - self._noise_logged_at >= NOISE_LOG_EVERY_S:
                print(f"· sound but no speech, not starting: {self._noises} since the last line",
                      flush=True)
                self._noises, self._noise_logged_at = 0, now

    def _check_worker(self) -> None:
        """Runs the second check, one attempt at a time, off the audio thread.

        If the detector can't run, hands-free still has to work: the first check alone decides,
        as it did before there was a second, and the log says so. The tap needs neither.
        """
        attempt, check = 0, None
        broken = False
        try:
            _check_model()
        except Exception as exc:  # noqa: BLE001 - a missing or broken model must not deafen it
            print(f"· speech check not available ({exc}); any speech-like sound will start a "
                  f"session", file=sys.stderr, flush=True)
            broken = True
        while True:
            job = self._jobs.get()
            try:
                if job is None:
                    return
                number, blocks = job
                if number != attempt:
                    attempt, check = number, None if broken else SpeechCheck()
                passed = broken
                if not broken:
                    try:
                        passed = any([check.feed(b) for b in blocks if len(b) == BLOCK_FRAMES * 2])
                    except Exception as exc:  # noqa: BLE001 - same again, mid-run
                        print(f"· speech check failed ({exc}); any speech-like sound will start "
                              f"a session", file=sys.stderr, flush=True)
                        broken = passed = True
                if passed:
                    with self._lock:
                        # A verdict from an attempt that has since ended counts for nothing.
                        if number == self._attempt and not self._closed:
                            self._trim_to_attempt()
                            self.heard.set()
            finally:
                self._jobs.task_done()

    def _trim_to_attempt(self) -> None:
        """Two seconds from before the first check fired, and everything since. Under the lock."""
        while len(self._before) > PREROLL_BLOCKS + self._attempt_blocks:
            self._before.popleft()

    def start(self) -> None:
        self._stream.start()
        # Here, once the mic has opened, so a mic that won't open leaves no thread behind; and not
        # at the first sound, so the model is loaded before anybody speaks.
        threading.Thread(target=self._check_worker, name="ear-check", daemon=True).start()

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
        self._jobs.put(None)
        try:
            self._stream.stop()
            self._stream.close()
        except Exception as exc:  # noqa: BLE001 - a device already gone is still closed
            print(f"[ear] closing the mic: {exc}", file=sys.stderr, flush=True)
