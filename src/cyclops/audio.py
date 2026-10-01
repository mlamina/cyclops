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
# How long his voice goes on sounding in the room *after* this box has finished playing it,
# when a companion on the LAN is the speaker (Speaker.on_air False). The phone holds a jitter
# buffer of its own - 0.45 s at rest, trimmed back whenever it passes DEEP in
# admin/static/stream.js - then a 4096-frame callback (170 ms) and whatever its own output
# costs. Every clock the EchoGuard owns is the local one, and there is no reference signal at
# all for audio we did not play, so the only honest answer is to hold the mic shut until the
# phone has certainly finished. That makes this pure dead air after every sentence, which is
# why DEEP is 0.7 s rather than the 1.5 it was: this number is *set by* that one, and shrinking
# the queue is the only way to shrink the wait. Measured against a session that fed Cyclops his
# own voice for 25 minutes on 2026-09-19.
COMPANION_LAG_S = 1.0


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


# ---- making a quiet microphone usable ----
#
# A lavalier clipped to a shirt hears you 20-30 dB quieter than a mic held at the mouth, and what
# it does hear swings: the same sentence is loud facing the panel and thin turned towards the
# bench. Nothing downstream fixes that - the realtime API takes whatever level it is given, and a
# recording is whatever reached the disk - so the levelling happens here, on the way in, before
# anything else sees a block: the guard, the queue, the meter and the recorder all get the same
# audio, and there is one place to look when it sounds wrong.


class Compressor:
    """Input gain, then compression with makeup: quiet speech comes up, loud speech stays put.

    Downward compression above ``THRESHOLD_DBFS`` plus a fixed ``MAKEUP_DB`` is the whole idea.
    The loudest speech is pushed back down by roughly what the makeup adds, so it lands about
    where it already was, while anything under the threshold is simply lifted - which closes the
    gap between the word you said into the panel and the one you said over your shoulder.

    Block-rate rather than per-sample, which is what makes it cheap enough for the Pi: one RMS,
    one peak and one interpolated ramp per 20 ms, instead of an envelope follower running at
    24 kHz. The ramp is not decoration - a gain that stepped at the block boundary would put a
    click in the waveform fifty times a second.

    Attack is one block and release is ``RELEASE_S``, so a shout is caught at once and the gain
    crawls back afterwards rather than pumping between syllables. Whatever the compressor asks
    for, the block's own peak has the last word: the gain is capped so nothing clips.

    Both halves are optional and independent - ``gain_db`` alone is a plain input gain, and
    compression alone is the usual case, because the makeup *is* the gain most rooms need.

    :class:`EchoGuard` sees the levelled block too, which is deliberate but worth knowing. Its
    echo estimate is a *ratio* learnt from the same signal it is later tested against, so it
    stays self-consistent; its one absolute number, ``MIC_FLOOR``, does not, and a room this
    quiet now clears it easily. If barge-in starts firing on nothing, that is the first thing
    to raise - or raise ``CYCLOPS_BARGE_IN_DB``, which is the knob already in the environment.
    """

    # Set against a real session rather than a rule of thumb. Measured over the speaking blocks
    # of a recorded conversation on the Pi's lavalier, a 20 ms block runs -54 dBFS at the fifth
    # percentile to -22 at the ninety-ninth, with the median at -41 - so the threshold sits down
    # among ordinary speech, not up at shouting. A threshold placed at the loud end would only
    # ever catch the loudest syllable of the loudest sentence, and the quiet half of what you
    # said would come back exactly as quiet as it went in, which is the complaint.
    THRESHOLD_DBFS = -45.0  # above this the compressor starts pulling down
    RATIO = 4.0  # 4 dB in over the threshold buys 1 dB out
    MAKEUP_DB = 20.0  # what everything gains afterwards; the point of the exercise
    RELEASE_S = 0.25  # how long the gain takes to crawl back after a loud passage
    CEILING = 0.98  # of full scale: the last word, so a block never clips on our account
    FULL_SCALE = 32768.0

    def __init__(self, *, gain_db: float = 0.0, compress: bool = True) -> None:
        self.gain_db = gain_db
        self.compress = compress
        self._gain = 10 ** (gain_db / 20)
        self._release = 1 - math.exp(-(BLOCK_FRAMES / SAMPLE_RATE) / self.RELEASE_S)
        self._envelope = self.THRESHOLD_DBFS  # dBFS, smoothed across blocks
        self._applied = self._gain  # gain at the end of the last block, so the ramp is continuous

    def process(self, block: bytes) -> bytes:
        """One block in, the same block levelled. Runs on the audio thread: no I/O, no locks."""
        samples = np.frombuffer(block, dtype=np.int16).astype(np.float32)
        if not samples.size:
            return block
        target = self._gain * self._factor(samples * self._gain)
        # Ramped on the way up, immediate on the way down. Sliding a rising gain across the block
        # is what keeps a release from clicking; sliding a *falling* one would leave the first
        # samples of the block still carrying the old, higher gain - and the reason the gain is
        # falling is that this block has a peak in it that the old gain would clip. So a cut
        # lands at once, which is what a limiter's attack is, and the step it makes is under the
        # transient that caused it.
        start = self._applied if target >= self._applied else target
        self._applied = target
        ramp = np.linspace(start, target, samples.size, dtype=np.float32)
        return np.clip(samples * ramp, -32768.0, 32767.0).astype(np.int16).tobytes()

    def _factor(self, hot: np.ndarray) -> float:
        """What to multiply the already-gained block by: makeup, less any compression."""
        if not self.compress:
            return 1.0
        rms = math.sqrt(float(np.mean(hot * hot)))
        level = 20 * math.log10(max(rms, 1.0) / self.FULL_SCALE)  # floored at one LSB, not -inf
        if level > self._envelope:
            self._envelope = level  # attack: a loud block is caught immediately
        else:
            self._envelope += (level - self._envelope) * self._release
        over = self._envelope - self.THRESHOLD_DBFS
        reduction = over * (1 - 1 / self.RATIO) if over > 0 else 0.0
        factor = 10 ** ((self.MAKEUP_DB - reduction) / 20)
        peak = float(np.max(np.abs(hot)))
        ceiling = self.CEILING * self.FULL_SCALE
        return ceiling / peak if peak * factor > ceiling and peak > 0 else factor


class Microphone:
    """Streams PCM16 chunks from the chosen input device into an asyncio queue.

    Levelled on the way through unless that is turned off - see :class:`Compressor`. That is
    upstream of everything: the echo guard judges the levelled block, the recorder's tap gets
    it, and it is what goes to the model, so there is never a version of a session's audio that
    only one of them heard.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        *,
        guard: EchoGuard | None = None,
        device: int | str | None = None,
        gain_db: float = 0.0,
        compress: bool = True,
    ) -> None:
        self._loop = loop
        self._guard = guard  # decides which blocks get through while the speaker is audible
        # Levelling comes first, so everything below sees one version of the audio. Skipped
        # entirely when there is nothing to do, rather than run as an expensive no-op.
        self._compressor = (
            Compressor(gain_db=gain_db, compress=compress) if gain_db or compress else None
        )
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=MIC_QUEUE_MAX)
        self._status = ""  # last PortAudio status flags, written on the audio thread
        self._warned = False
        self.level = 0.0  # smoothed RMS of the latest mic block (for the UI meter)
        # Optional tap, called on the audio thread with every block, whatever the guard then
        # does with it - the recording keeps the room, not the guard's verdict on it. Must not
        # do I/O. Set by the recorder; see cyclops.record.
        self.on_block: Callable[[bytes], None] | None = None
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
        if self._compressor is not None:
            block = self._compressor.process(block)
        self.level = 0.6 * self.level + 0.4 * _rms(block)
        if self.on_block is not None:
            self.on_block(block)
        blocks = self._guard.admit(block) if self._guard is not None else [block]
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

    @property
    def levelling(self) -> str:
        """How the input is being treated, for the startup line; empty when it is left alone."""
        shaping = self._compressor
        if shaping is None:
            return ""
        said = [f"{shaping.gain_db:+g} dB"] if shaping.gain_db else []
        if shaping.compress:
            said.append(f"compressed +{Compressor.MAKEUP_DB:g} dB")
        return f" ({', '.join(said)})" if said else ""

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
        # And the second one, which is not the recorder's. A recording is one consumer of the
        # output and a phone being used as the speaker is another, and they must not be able to
        # clobber each other: `SessionLog._start_recorder` assigns on_block unconditionally, so
        # anything else reaching for that name would silently take the agent track out of every
        # session video. Two names, two owners. Same rule: must not do I/O.
        self.on_monitor: Callable[[bytes], None] | None = None
        # Whether his voice reaches this box's own amplifier. False while a companion is being
        # used as the speaker, and it silences *his voice only* - the sound cues play on their
        # own stream and are sent to the companion by cyclops.sfx itself (sfx.divert).
        self.on_air = True
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
        # The taps are handed the real block whatever the amp is doing with it: a session's video
        # is a record of what he said, not of which speaker happened to play it, and the companion
        # holding his voice is the whole reason this box might not be playing it.
        outdata[:] = chunk if self.on_air else bytes(needed)
        if self.on_block is not None:
            self.on_block(chunk)  # the recording first: it is the durable one
        if self.on_monitor is not None:
            self.on_monitor(chunk)

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
        """True while audio is buffered or was output within the last AUDIBLE_TAIL_S.

        Plus COMPANION_LAG_S while a phone is the speaker, because then the sound leaves the
        room on somebody else's clock and this buffer emptying says nothing about it.
        """
        with self._lock:
            tail = AUDIBLE_TAIL_S if self.on_air else AUDIBLE_TAIL_S + COMPANION_LAG_S
            if self._buffer:
                return True
            return time.monotonic() - self._last_output_at < tail

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

    ``half_duplex=False`` is headphones: nothing of the output reaches the mic, so with barge-in
    on there is nothing to guard against and every block goes through. It still gates when
    barge-in is turned off, because with nothing holding the mic shut there would be no way to
    honour that at all - so the guard is built for every session rather than only for speakers,
    and :meth:`set_barge_in` can flip it in the middle of one.
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
        half_duplex: bool = True,
        on_barge_in: Callable[[int, float], None] | None = None,
    ) -> None:
        self._loop = loop
        self._speaker = speaker
        self._half = half_duplex
        self._margin = _margin(margin_db)
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

    def set_barge_in(self, margin_db: float | None) -> None:
        """Turn barge-in on or off in a session already running - see :mod:`cyclops.barge`.

        Called from whichever thread the settings screen reaches us on, and read by the audio
        thread on its next block. That is safe because it is one rebinding of one name and
        there is no state to unwind: turning it off holds the mic shut for the rest of whatever
        is being said, and turning it on re-arms against an echo estimate this room has already
        earned. Only the trigger's run of loud blocks is dropped, so a count that had built up
        while the switch was off cannot fire on the first block after it comes back.
        """
        margin = _margin(margin_db)
        if margin == self._margin:
            return
        self._margin = margin
        self._consec = 0

    def cut(self) -> int:
        """Stop playback on somebody's say-so rather than the room's. Returns ms already heard.

        Deliberately *not* :meth:`_trigger`: it drops the buffer and stops there, leaving the
        mic to the ordinary ``is_audible`` gate below. _trigger can afford to force the mic open
        with ``_open_serial`` because of what it took to get there - the room was measured at
        several times the predicted echo, so somebody is provably talking over the speaker and
        the next block is theirs.

        A tap proves nothing of the kind. It says only that a finger touched glass, and
        :meth:`flush` cannot recall the audio already handed to the device: that goes on sounding
        for AUDIBLE_TAIL_S, which is the whole reason that constant exists. Forcing the mic open
        across it feeds Cyclops his own last words back as if they were yours, and he answers his
        own echo - which is exactly what he did the first time this shipped. So the gate opens
        when the room is actually quiet, a third of a second later, and not a moment sooner.
        """
        self._consec = 0  # any run of loud blocks belonged to the sentence just ended
        return self._speaker.flush()

    def admit(self, block: bytes) -> list[bytes]:
        """Audio-thread hook: return the mic blocks to forward for this 20 ms input block."""
        speaker = self._speaker
        # A phone is the speaker. Both ways out of the gate below assume the room is hearing
        # what this box is playing, and neither holds: headphones say nothing about a speaker
        # in somebody's hand, and the echo ratio is measured against a reference that ran
        # COMPANION_LAG_S ahead of the sound - so once our own buffer drains the predicted
        # echo is zero and every word he says reads as you talking over him. Half-duplex on
        # the lengthened tail is the only claim left that is true.
        elsewhere = not speaker.on_air
        if not self._half and self._margin is not None and not elsewhere:
            return [block]  # headphones, barge-in on: there is no echo to guard against
        if not speaker.is_audible or speaker.item_serial == self._open_serial:
            self._audible_blocks = 0
            self._consec = 0
            return [block]
        if self._margin is None or elsewhere:
            return []  # plain half-duplex; k is left alone, so barge-in resumes calibrated

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


def _margin(margin_db: float | None) -> float | None:
    """dB over the predicted echo as a plain ratio, or None for "never interrupt"."""
    return None if margin_db is None else 10 ** (margin_db / 20)
