"""Sessions that start on speech and end on their own (arc-eng/hq#561).

The ear's gate judged on real speech and on things that are not speech; what it keeps and hands
over; the agent answering the first turn instead of greeting; the quiet timer and the goodbye
ending a session once; and the kiosk opening, deafening and handing over the ear.
"""

from __future__ import annotations

import asyncio
import base64
import wave
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cyclops import agent, ear
from cyclops import kiosk as kiosk_module
from cyclops.audio import BLOCK_FRAMES, SAMPLE_RATE
from cyclops.config import Settings
from cyclops.ui import CONNECTING, ERROR, IDLE, LISTENING

SOUNDS = Path(agent.__file__).parent / "assets" / "sounds"
BLOCK_BYTES = BLOCK_FRAMES * 2
SILENCE = b"\0" * BLOCK_BYTES


def _blocks(samples: np.ndarray) -> list[bytes]:
    pcm = samples.astype(np.int16).tobytes()
    return [pcm[i:i + BLOCK_BYTES] for i in range(0, len(pcm) - BLOCK_BYTES + 1, BLOCK_BYTES)]


@cache
def _speech() -> tuple[bytes, ...]:
    """A real voice (one of the voice previews), at 24 kHz and talking volume."""
    with wave.open(str(SOUNDS / "voice_cedar.wav")) as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2
        rate = w.getframerate()
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    at_24k = np.interp(np.arange(0, len(raw), rate / SAMPLE_RATE), np.arange(len(raw)), raw)
    peak = float(np.max(np.abs(at_24k))) or 1.0
    return tuple(_blocks(at_24k * (12_000 / peak)))


def _pink(rng: np.random.Generator, n: int) -> np.ndarray:
    spectrum = np.fft.rfft(rng.normal(0, 1, n)) / np.sqrt(np.arange(1, n // 2 + 2))
    pink = np.fft.irfft(spectrum, n)
    return pink * (3000 / pink.std())


def _triggers(blocks) -> bool:
    gate = ear.SpeechGate()
    return any(gate.feed(block) for block in blocks)


# ------------------------------------------------------------------ the gate


def test_speech_starts_one() -> None:
    assert _triggers(_speech())


@pytest.mark.parametrize("name", ["silence", "hum", "noise", "pink", "clang"])
def test_what_is_not_speech_does_not(name: str) -> None:
    rng = np.random.default_rng(7)
    t = np.arange(SAMPLE_RATE * 2) / SAMPLE_RATE
    signal = {
        "silence": np.zeros_like(t),
        "hum": 6000 * np.sin(2 * np.pi * 120 * t),  # a motor, a fan
        "noise": rng.normal(0, 3000, t.size),  # a hiss of air
        "pink": _pink(rng, t.size),  # a sander, a fan at full
        # a hammer on metal, every half second: loud, short, ringing
        "clang": 15000 * np.sin(2 * np.pi * 2200 * t) * np.exp(-30 * (t % 0.5)),
    }[name]
    assert not _triggers(_blocks(signal))


def test_speech_from_across_the_room_does_not() -> None:
    quiet = [(np.frombuffer(b, dtype=np.int16) * 0.01).astype(np.int16).tobytes()
             for b in _speech()]
    assert not _triggers(quiet)


# ------------------------------------------------------------------ the ear


class FakeStream:
    def __init__(self, **kwargs) -> None:
        self.stopped = self.closed = False

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def an_ear(monkeypatch) -> ear.Ear:
    monkeypatch.setattr(ear.sd, "RawInputStream", FakeStream)
    monkeypatch.setattr(ear, "pin_input_source", lambda device: None)
    return ear.Ear(compress=False)


def _hear(made: ear.Ear, blocks) -> None:
    for block in blocks:
        made._callback(block, BLOCK_FRAMES, None, None)


def test_before_a_start_only_the_last_two_seconds_are_kept(an_ear) -> None:
    _hear(an_ear, [SILENCE] * (ear.PREROLL_BLOCKS * 3))
    assert not an_ear.heard.is_set()
    assert len(an_ear.take()) == ear.PREROLL_BLOCKS * BLOCK_BYTES


def test_what_was_said_before_and_after_the_start_is_handed_over(an_ear) -> None:
    speech = list(_speech())
    _hear(an_ear, speech)
    assert an_ear.heard.is_set()
    _hear(an_ear, [SILENCE] * 10)  # the connect, still being collected
    taken = an_ear.take()
    assert taken.endswith(SILENCE * 10)
    assert speech[-1] in taken
    _hear(an_ear, [SILENCE] * 10)  # after the take nothing more is kept
    assert an_ear.take() == b""


def test_a_deaf_ear_does_not_start_one(an_ear) -> None:
    an_ear.deaf = True
    _hear(an_ear, _speech())
    assert not an_ear.heard.is_set()


def test_closing_drops_what_it_kept(an_ear) -> None:
    _hear(an_ear, _speech())
    an_ear.close()
    an_ear.close()  # twice is fine
    assert an_ear.take() == b""
    assert an_ear._stream.closed


# ------------------------------------------------------------------ the agent

UP = SimpleNamespace(type="session.updated")


class FakeConn:
    def __init__(self) -> None:
        self.responses: list[dict] = []
        self.appended: list[bytes] = []
        self.response = SimpleNamespace(create=self._response)
        self.input_audio_buffer = SimpleNamespace(append=self._append)

    async def _response(self, **kwargs) -> None:
        self.responses.append(kwargs)

    async def _append(self, audio: str) -> None:
        self.appended.append(base64.b64decode(audio))


class FakeMic:
    def __init__(self, queued: list[bytes]) -> None:
        self.queued = list(queued)

    def drain(self) -> None:
        self.queued.clear()

    async def chunks(self):
        for chunk in self.queued:
            yield chunk


class FakeEar:
    def __init__(self, pcm: bytes) -> None:
        self.pcm = pcm
        self.closed = False

    def take(self) -> bytes:
        pcm, self.pcm = self.pcm, b""
        return pcm

    def close(self) -> None:
        self.closed = True


def _agent(monkeypatch, *, ear_pcm: bytes | None = None, quiet_end_s: int = 0, mic=None):
    made = agent.VoiceAgent(
        Settings(api_key="", sounds=False, quiet_end_s=quiet_end_s),
        mic=mic,
        ear=FakeEar(ear_pcm) if ear_pcm is not None else None,
    )
    made._conn = FakeConn()
    monkeypatch.setattr(made, "_log", lambda *a, **k: None)
    monkeypatch.setattr(agent.session, "note", lambda *a, **k: None)
    return made


def test_a_voiced_session_answers_what_was_said_instead_of_greeting(monkeypatch) -> None:
    said = b"\1\2" * (SAMPLE_RATE * 3)  # three seconds he said before the model could hear
    mic = FakeMic([b"stale" * 10])  # the same stretch of room, heard by the session's own mic
    made = _agent(monkeypatch, ear_pcm=said, mic=mic)
    held = made._ear

    async def go():
        await made._handle_event(UP)
        await asyncio.gather(*made._background)

    asyncio.run(go())
    assert made._conn.responses == []  # no greeting
    assert b"".join(made._conn.appended) == said  # all of it, and not the drained mic
    assert held.closed


def test_a_button_session_still_greets(monkeypatch) -> None:
    made = _agent(monkeypatch)

    async def go():
        await made._handle_event(UP)
        await asyncio.gather(*made._background)

    asyncio.run(go())
    assert len(made._conn.responses) == 1


def _ending(made: agent.VoiceAgent) -> list[str]:
    asked: list[str] = []
    made.on_end = lambda: asked.append(made.ended_by or "")
    return asked


def test_two_quiet_minutes_end_it(monkeypatch) -> None:
    monkeypatch.setattr(agent, "QUIET_CHECK_S", 0.001)
    made = _agent(monkeypatch, quiet_end_s=120)
    asked = _ending(made)
    made._heard_at = agent.time.monotonic() - 121

    asyncio.run(asyncio.wait_for(made._end_when_quiet(), 1))
    assert asked == ["quiet"]


def test_quiet_does_not_count_while_cyclops_is_busy(monkeypatch) -> None:
    monkeypatch.setattr(agent, "QUIET_CHECK_S", 0.001)
    made = _agent(monkeypatch, quiet_end_s=120)
    asked = _ending(made)
    made._heard_at = agent.time.monotonic() - 121
    made._tools_running = 1  # a lookup still running

    async def go():
        task = asyncio.create_task(made._end_when_quiet())
        await asyncio.sleep(0.02)
        assert not task.done()
        task.cancel()

    asyncio.run(go())
    assert asked == []
    assert agent.time.monotonic() - made._heard_at < 1  # and the count starts again after


def test_somebody_speaking_restarts_the_count(monkeypatch) -> None:
    made = _agent(monkeypatch)
    made._heard_at = 0.0
    asyncio.run(made._handle_event(SimpleNamespace(type="input_audio_buffer.speech_stopped")))
    assert agent.time.monotonic() - made._heard_at < 1


def test_a_goodbye_ends_it_once(monkeypatch) -> None:
    made = _agent(monkeypatch)
    asked = _ending(made)
    call = SimpleNamespace(name="end_session", call_id="c1", arguments="{}")
    asyncio.run(made._dispatch_tool(call))
    made.end("quiet")  # anything after the first ask is ignored
    assert asked == ["goodbye"]
    assert made.ended_by == "goodbye"


def test_end_session_is_offered() -> None:
    names = {tool["name"] for tool in agent._offered_tools(Settings(api_key=""))}
    assert "end_session" in names


# ------------------------------------------------------------------ the kiosk


class Controller:
    def __init__(self, voice_wake: bool = True) -> None:
        self.settings = Settings(api_key="", voice_wake=voice_wake)
        self.closing = False
        self.started_with: list = []

    def start(self, *, ear=None) -> None:
        self.started_with.append(ear)


class Cues:
    def __init__(self) -> None:
        self.left = 0.0

    def waiting(self) -> float:
        return self.left


class StubEar:
    def __init__(self, **kwargs) -> None:
        self.deaf = False
        self.heard = ear.threading.Event()
        self.closed = False

    def start(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def _panel(monkeypatch, *, voice_wake: bool = True) -> kiosk_module.Kiosk:
    monkeypatch.setattr(kiosk_module.ear, "Ear", StubEar)
    kiosk = object.__new__(kiosk_module.Kiosk)
    kiosk.controller = Controller(voice_wake)
    kiosk._cues = Cues()
    kiosk._menu = False
    kiosk._wifi = None
    kiosk._online = True
    kiosk._asleep = True
    kiosk._touched_at = 0.0
    kiosk._pending = None
    kiosk._pending_at = 0.0
    kiosk._ear = None
    kiosk._ear_quiet_until = 0.0
    kiosk._ear_retry_at = 0.0
    kiosk._ear_state = ""
    kiosk.woke = []
    monkeypatch.setattr(kiosk, "_wake", lambda: kiosk.woke.append(True))
    return kiosk


def test_an_idle_kiosk_listens_and_a_voice_starts_a_session(monkeypatch) -> None:
    kiosk = _panel(monkeypatch)
    kiosk._tend_ear(IDLE, 100.0)
    listening = kiosk._ear
    assert listening is not None and not listening.deaf
    listening.heard.set()
    kiosk._tend_ear(IDLE, 100.1)
    assert kiosk.controller.started_with == [listening]  # handed over, not closed
    assert not listening.closed and kiosk._ear is None
    assert kiosk._pending == "start" and kiosk.woke == [True]


def test_no_ear_during_a_session_and_deaf_just_after(monkeypatch) -> None:
    kiosk = _panel(monkeypatch)
    kiosk._tend_ear(IDLE, 100.0)
    first = kiosk._ear
    kiosk._tend_ear(LISTENING, 101.0)
    assert first.closed and kiosk._ear is None
    kiosk._tend_ear(IDLE, 101.5)
    assert kiosk._ear.deaf  # its last sounds cannot start the next one
    kiosk._tend_ear(IDLE, 101.0 + kiosk_module.EAR_HOLD_S + 0.1)
    assert not kiosk._ear.deaf


def test_deaf_while_the_box_makes_a_sound(monkeypatch) -> None:
    kiosk = _panel(monkeypatch)
    kiosk._cues.left = 0.5
    kiosk._tend_ear(IDLE, 100.0)
    assert kiosk._ear.deaf


def test_a_failed_start_is_not_retried_on_the_next_word(monkeypatch) -> None:
    kiosk = _panel(monkeypatch)
    kiosk._tend_ear(CONNECTING, 100.0)
    kiosk._tend_ear(ERROR, 101.0)
    kiosk._tend_ear(ERROR, 110.0)
    assert kiosk._ear.deaf
    kiosk._tend_ear(ERROR, 101.0 + kiosk_module.EAR_FAILED_HOLD_S + 0.1)
    assert not kiosk._ear.deaf


def test_the_switch_off_leaves_the_mic_closed(monkeypatch) -> None:
    kiosk = _panel(monkeypatch, voice_wake=False)
    kiosk._tend_ear(IDLE, 100.0)
    assert kiosk._ear is None


def test_no_network_no_ear(monkeypatch) -> None:
    kiosk = _panel(monkeypatch)
    kiosk._online = False
    kiosk._tend_ear(IDLE, 100.0)
    assert kiosk._ear is None
