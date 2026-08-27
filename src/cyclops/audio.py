"""Microphone capture and speaker playback for 24 kHz / 16-bit / mono PCM.

PortAudio invokes stream callbacks on its own real-time thread, so:

* the microphone callback hands bytes to the asyncio loop with ``call_soon_threadsafe``;
* the speaker callback pulls from a lock-protected byte buffer that the loop appends to;
* neither callback does I/O - status flags are recorded and reported from the loop thread.
"""

from __future__ import annotations

import asyncio
import math
import os
import sys
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Callable

import numpy as np
import sounddevice as sd

from .mixer import pactl

SAMPLE_RATE = 24_000
CHANNELS = 1
DTYPE = "int16"
BYTES_PER_FRAME = 2  # int16 mono
FRAMES_PER_MS = SAMPLE_RATE // 1000
BLOCK_FRAMES = 480  # 20 ms per callback: small enough for snappy barge-in, large enough to be cheap
MIC_QUEUE_MAX = 250  # ~5 s of audio before we start dropping (loop stalled)
AUDIBLE_TAIL_S = 0.3  # audio already handed to CoreAudio keeps sounding after our buffer empties
OUTPUT_HISTORY_S = 1.0  # how long the speaker remembers its output level (for echo estimation)


def _rms(pcm: bytes) -> float:
    if not pcm:
        return 0.0
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    return math.sqrt(float(np.mean(samples * samples)))


# Output-device name fragments that mean "this plays out loud and the mic will hear it".
_LOUDSPEAKER_HINTS = ("speaker", "hdmi", "loudspeaker", "display", "tv")


def resolve_device(device: str | None) -> int | str | None:
    """A device is an index or a name substring; turn an all-digit string into an int index."""
    if device is None:
        return None
    return int(device) if device.isdigit() else device


def default_output_name(device: int | str | None = None) -> str:
    if device is not None:
        return str(sd.query_devices(device, "output")["name"])
    return str(sd.query_devices(kind="output")["name"])


def output_is_speaker(name: str | None = None) -> bool:
    """Heuristic: loudspeakers (built-in, external, HDMI/TV) feed back into the mic and want
    half-duplex; headphones/earbuds/headsets don't. Ambiguous names default to headphones -
    set CYCLOPS_HALF_DUPLEX=1 to force speaker mode."""
    low = (name if name is not None else default_output_name()).lower()
    if any(h in low for h in ("headphone", "headset", "earbud", "airpod")):
        return False
    return any(h in low for h in _LOUDSPEAKER_HINTS)


# ---- which microphone ----
#
# Two mics are plugged into the Pi: the camera's own, behind the housing and pointed at the room,
# and a lavalier a hand's width from the mouth. PortAudio cannot choose between them - through
# PipeWire it sees one "pulse" device and listens to whatever the default source happens to be,
# which is decided by enumeration order and changes when things are replugged. So we choose, by
# naming a source before the stream is opened. The lav is there because someone clipped it on.

# Sources that belong to a camera. Anything else is a mic that was plugged in on purpose.
_WEBCAM_MIC_HINTS = ("webcam", "camera", "_cam", "c920")

# Input settings that mean "go through PipeWire", and can therefore be steered by source name.
# Anything else is a specific ALSA device that someone pinned by hand; leave it alone.
_THROUGH_PIPEWIRE = (None, "pulse", "default")

# A choice made before we started - PULSE_SOURCE in .env or in the shell - outranks ours, and is
# how to force the camera's mic. Read at import so that our own pinning never looks like one.
_PINNED_BY_HAND = os.environ.get("PULSE_SOURCE") or None


def capture_sources() -> list[str]:
    """PipeWire's capture sources, in the order it lists them; empty where there is no PipeWire.

    Monitors are dropped - those are loopbacks of an output, not microphones.
    """
    listing = pactl("list", "short", "sources")
    if listing is None:
        return []
    names = [line.split("\t")[1] for line in listing.splitlines() if "\t" in line]
    return [name for name in names if not name.endswith(".monitor")]


def preferred_source(sources: list[str]) -> str | None:
    """The mic to listen through: one someone plugged in beats the camera's own.

    ``None`` means "leave PipeWire's default alone" - either there is no PipeWire to ask, or the
    camera's mic is the only mic there is. Where two plugged-in mics are present, the first
    PipeWire lists wins.
    """
    for name in sources:
        if not any(hint in name.lower() for hint in _WEBCAM_MIC_HINTS):
            return name
    return None


def pin_input_source(device: int | str | None = None) -> str | None:
    """Point this process's capture at that mic. Returns the source it will now listen through.

    The ALSA-pulse plugin reads ``PULSE_SOURCE`` at the moment PortAudio opens the device, so
    this has to run before the stream is created - and may be run again afterwards, which is what
    lets a mic plugged in between two sessions be picked up without restarting anything. Naming a
    source that has since been unplugged is not an error: libpulse falls back to the default.
    """
    if _PINNED_BY_HAND is not None:
        return _PINNED_BY_HAND
    if device not in _THROUGH_PIPEWIRE:
        return None
    chosen = preferred_source(capture_sources())
    if chosen is None:
        os.environ.pop("PULSE_SOURCE", None)  # the mic we pinned last time is gone
    else:
        os.environ["PULSE_SOURCE"] = chosen
    return chosen


def list_devices() -> str:
    """Human-readable table of audio devices, for `cyclops-devices`."""
    return str(sd.query_devices())


class Microphone:
    """Streams raw PCM16 chunks from the chosen input device into an asyncio queue."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        *,
        guard: EchoGuard | None = None,
        device: int | str | None = None,
    ) -> None:
        self._loop = loop
        self._guard = guard  # decides which blocks get through while the speaker is audible
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=MIC_QUEUE_MAX)
        self._status = ""  # last PortAudio status flags, written on the audio thread
        self._warned = False
        self.level = 0.0  # smoothed RMS of the latest mic block (for the UI meter)
        # Optional tap, called on the audio thread with (raw block, how many the guard admitted).
        # Must not do I/O. Set by the recorder; see cyclops.record.
        self.on_block: Callable[[bytes, int], None] | None = None
        # Which physical mic this is, decided here rather than once at startup: the kiosk process
        # outlives any number of sessions, so a lav clipped on between two of them is picked up by
        # the next one. Must precede the stream: that is when PULSE_SOURCE is read.
        self.source = pin_input_source(device)
        self._stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_FRAMES,
            channels=CHANNELS,
            dtype=DTYPE,
            latency="low",
            device=device,
            callback=self._callback,
        )

    def _callback(self, indata, frames: int, time_info, status: sd.CallbackFlags) -> None:
        if status:
            self._status = str(status)
        block = bytes(indata)  # copy: PortAudio reuses the buffer
        self.level = 0.6 * self.level + 0.4 * _rms(block)
        blocks = self._guard.admit(block) if self._guard is not None else [block]
        if self.on_block is not None:
            self.on_block(block, len(blocks))
        for admitted in blocks:
            self._loop.call_soon_threadsafe(self._enqueue, admitted)

    def _enqueue(self, data: bytes) -> None:
        try:
            self._queue.put_nowait(data)
        except asyncio.QueueFull:
            self._warn("event loop is falling behind; dropping mic audio")
        if self._status:
            self._warn(f"PortAudio reported: {self._status}")

    def _warn(self, message: str) -> None:
        if not self._warned:
            self._warned = True
            print(f"[mic] {message}", file=sys.stderr)

    def drain(self) -> None:
        """Discard everything captured so far (e.g. audio from before the session was ready)."""
        while not self._queue.empty():
            self._queue.get_nowait()

    async def chunks(self) -> AsyncIterator[bytes]:
        while True:
            yield await self._queue.get()

    def start(self) -> None:
        self._stream.start()

    def stop(self) -> None:
        self._stream.stop()
        self._stream.close()


class Speaker:
    """Plays PCM16 from a byte buffer; zero-fills on underrun; can be flushed instantly.

    Tracks how much of the current assistant audio item has actually reached the speaker
    (needed to truncate the item on barge-in).
    """

    def __init__(self, *, device: int | str | None = None) -> None:
        self._lock = threading.Lock()
        self._buffer = bytearray()
        self._played_frames = 0  # total frames ever handed to PortAudio
        self._item_start_frame = 0  # playback position at which the current item begins
        self._last_output_at = 0.0  # monotonic time we last handed real audio to PortAudio
        self._output_levels: deque[tuple[float, float]] = deque(maxlen=int(OUTPUT_HISTORY_S * 50))
        self.item_serial = 0  # incremented by begin_item(); lets the EchoGuard notice new speech
        self.volume = 1.0  # output gain 0.0-1.0, applied as we buffer audio
        # Optional tap, called on the audio thread with each block handed to PortAudio - already
        # zero-filled, so it is a continuous record of the output. Must not do I/O.
        self.on_block: Callable[[bytes], None] | None = None
        self._status = ""
        self._warned = False
        self._stream = sd.RawOutputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_FRAMES,
            channels=CHANNELS,
            dtype=DTYPE,
            latency="low",
            device=device,
            callback=self._callback,
        )

    def _callback(self, outdata, frames: int, time_info, status: sd.CallbackFlags) -> None:
        if status:
            self._status = str(status)
        needed = frames * BYTES_PER_FRAME
        with self._lock:
            chunk = bytes(self._buffer[:needed])
            del self._buffer[:needed]
            self._played_frames += len(chunk) // BYTES_PER_FRAME
            if chunk:
                now = time.monotonic()
                self._last_output_at = now
                self._output_levels.append((now, _rms(chunk)))
        if len(chunk) < needed:
            chunk += b"\x00" * (needed - len(chunk))
        outdata[:] = chunk
        if self.on_block is not None:
            self.on_block(chunk)

    def begin_item(self) -> None:
        """Mark the start of a new assistant audio item: it begins after everything buffered."""
        with self._lock:
            self._item_start_frame = self._played_frames + len(self._buffer) // BYTES_PER_FRAME
            self.item_serial += 1

    def recent_output_level(self, window_s: float) -> float:
        """Loudest block (int16 RMS) we sent to the speaker in the last ``window_s`` seconds."""
        cutoff = time.monotonic() - window_s
        with self._lock:
            return max((rms for at, rms in self._output_levels if at >= cutoff), default=0.0)

    def output_level(self) -> float:
        """Recent playback loudness (int16 RMS), for the UI meter."""
        return self.recent_output_level(0.12)

    def feed(self, pcm: bytes) -> None:
        if self._status and not self._warned:
            self._warned = True
            print(f"[speaker] PortAudio reported: {self._status}", file=sys.stderr)
        pcm = pcm[: len(pcm) // BYTES_PER_FRAME * BYTES_PER_FRAME]  # keep sample alignment
        vol = self.volume
        if vol != 1.0 and pcm:  # scale here so _output_levels/EchoGuard see the real output level
            samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) * vol
            pcm = np.clip(samples, -32768, 32767).astype(np.int16).tobytes()
        with self._lock:
            self._buffer.extend(pcm)

    def flush(self) -> int:
        """Drop everything not yet played. Returns ms of the current item that *was* played."""
        with self._lock:
            played_ms = max(0, self._played_frames - self._item_start_frame) // FRAMES_PER_MS
            self._buffer.clear()
            self._item_start_frame = self._played_frames
        return played_ms

    @property
    def has_unplayed_audio(self) -> bool:
        with self._lock:
            return len(self._buffer) > 0

    @property
    def is_audible(self) -> bool:
        """True while audio is buffered or was output within the last AUDIBLE_TAIL_S."""
        with self._lock:
            if self._buffer:
                return True
            return time.monotonic() - self._last_output_at < AUDIBLE_TAIL_S

    def start(self) -> None:
        self._stream.start()

    def stop(self) -> None:
        self._stream.stop()
        self._stream.close()


class EchoGuard:
    """Half-duplex with barge-in, for open speakers.

    While the speaker is audible the mic is muted - except when the mic is clearly louder than
    the *echo* of what we are playing, which means you are talking over the assistant. Then
    playback is cut on the spot, the last few mic blocks are released so the start of your
    sentence isn't lost, and the mic stays open until the assistant starts a new utterance.

    The echo is predicted as ``k * playback level``, where ``k`` is the acoustic coupling of this
    room and volume. Because the echo lags the playback and its level swings block to block,
    ``k`` tracks the *upper envelope* of the observed echo/playback ratio (fast attack, slow
    decay) - a conservative estimate that avoids false cuts. ``margin_db`` is how much louder
    than that envelope the mic must be; ``None`` disables barge-in (plain half-duplex gating).
    """

    WARMUP_BLOCKS = 10  # ignore the first 200 ms of playback: the echo hasn't reached the mic yet
    LEARN_BLOCKS = 25  # then calibrate k on 500 ms of clean echo before arming
    CONSEC_BLOCKS = 6  # 120 ms above threshold to trigger (a syllable, not a click)
    PREROLL_BLOCKS = 15  # release 300 ms of history on trigger so the word onset is kept
    REF_WINDOW_S = 0.7  # playback lookback, must cover output + acoustic + input latency
    REF_FLOOR = 150.0  # int16 RMS; quieter playback counts as silence
    MIC_FLOOR = 400.0  # int16 RMS; quieter mic input is never speech
    DECAY = 0.995  # envelope decay per 20 ms block (~4 s time constant)

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        speaker: Speaker,
        *,
        margin_db: float | None,
        on_barge_in: Callable[[int, float], None] | None = None,
    ) -> None:
        self._loop = loop
        self._speaker = speaker
        self._margin = None if margin_db is None else 10 ** (margin_db / 20)
        self.on_barge_in = on_barge_in  # called on the loop thread with (played_ms, strength)
        self.k: float | None = None  # upper envelope of echo rms / playback rms
        self.triggers = 0
        self.peak_ratio = 0.0  # loudest mic/(k*playback) seen without triggering (for tuning)
        self._audible_blocks = 0
        self._learn_peak = 0.0
        self._learn_n = 0
        self._consec = 0
        self._preroll: deque[bytes] = deque(maxlen=self.PREROLL_BLOCKS)
        self._open_serial = -1  # speaker.item_serial during which the mic is held open
        self._trigger_ratio = 0.0

    @property
    def barge_in_enabled(self) -> bool:
        return self._margin is not None

    def admit(self, block: bytes) -> list[bytes]:
        """Audio-thread hook: return the mic blocks to forward for this 20 ms input block."""
        speaker = self._speaker
        if not speaker.is_audible or speaker.item_serial == self._open_serial:
            self._audible_blocks = 0
            self._consec = 0
            return [block]
        if self._margin is None:
            return []  # plain half-duplex

        self._audible_blocks += 1
        if self._audible_blocks <= self.WARMUP_BLOCKS:
            return []
        mic = _rms(block)
        playback = speaker.recent_output_level(self.REF_WINDOW_S)
        ref = max(playback, self.REF_FLOOR)
        ratio = mic / ref
        self._preroll.append(block)

        if self.k is None:  # calibrating on clean echo
            if playback > self.REF_FLOOR:
                self._learn_peak = max(self._learn_peak, ratio)
                self._learn_n += 1
                if self._learn_n >= self.LEARN_BLOCKS:
                    self.k = max(self._learn_peak, 1e-3)
            return []

        if mic > self.MIC_FLOOR and ratio > self.k * self._margin:
            self._consec += 1
            if self._consec >= self.CONSEC_BLOCKS:
                self._trigger(ratio)
                released = list(self._preroll)
                self._preroll.clear()
                return released
            return []

        self._consec = 0
        if playback > self.REF_FLOOR:
            self.peak_ratio = max(self.peak_ratio, ratio / self.k)
            # Track the echo envelope DOWNWARD only. A block louder than the current estimate is
            # exactly what a barge-in looks like, so it must never be absorbed into k (that is the
            # double-talk leak that desensitises the detector); k only rises via reject().
            if ratio < self.k:
                self.k = max(self.k * self.DECAY, ratio)
        return []

    def _trigger(self, ratio: float) -> None:
        self._consec = 0
        self._open_serial = self._speaker.item_serial
        self._trigger_ratio = ratio
        self.triggers += 1
        strength = ratio / self.k if self.k else 0.0  # how many times over the predicted echo
        played_ms = self._speaker.flush()  # cut the echo right here, on the audio thread
        if self.on_barge_in is not None:
            self._loop.call_soon_threadsafe(self.on_barge_in, played_ms, strength)

    def confirm(self) -> None:
        """The server heard speech after our trigger: the calibration was right."""

    def reject(self) -> None:
        """No speech followed the trigger: that loudness was echo, so expect it from now on."""
        if self.k is not None:
            self.k = max(self.k, self._trigger_ratio)
