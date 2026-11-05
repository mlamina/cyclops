#!/usr/bin/env python3
"""hq#572 against the real voice model: silence on start, goodbye, talking during a close, carry.

Run on the Pi, from a throwaway copy, with its working directory in the repo so `.env` is found:

    cd ~/cyclops && .venv/bin/python /tmp/qa-572/tests/acceptance/hq_572_real_sessions.py

Real: SessionController, VoiceAgent, Ear, SpeechGate, Kiosk._tend_ear on a 30 fps loop, the model.
Fake: the sound card. A scripted room plays recorded speech into the mic, and what the speaker
plays comes back into the mic after ECHO_S, so its own voice can try to start a session.
Not exercised: a real speaker and lav, the room's real echo, video muxing (CLOSE_EXTRA_S stands in).

Isolation: HOME and every record folder are scratch, set before cyclops is imported, so
~/.cache/cyclops (tasks, panel, voice) is out of reach as well as his notes. Nothing is filed.
It refuses to run if any path resolves outside scratch. Compare his records before and after anyway.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

SCRATCH = Path(tempfile.mkdtemp(prefix="cyclops-qa-572-"))
os.environ["HOME"] = str(SCRATCH)
for name, sub in (("SESSIONS", "sessions"), ("PROJECTS", "projects"), ("CAPTURES", "captures"),
                  ("MANUALS", "manuals")):
    (SCRATCH / sub).mkdir()
    os.environ[f"CYCLOPS_{name}_DIR"] = str(SCRATCH / sub)
os.environ["CYCLOPS_ABOUT_FILE"] = str(SCRATCH / "about-you.md")
os.environ["CYCLOPS_RECORD"] = "0"
(SCRATCH / ".cache" / "cyclops").mkdir(parents=True)

import numpy as np  # noqa: E402

from cyclops import agent as agent_module  # noqa: E402
from cyclops import audio, ear, session, sfx, tasks, ui  # noqa: E402
from cyclops import config as config_module  # noqa: E402
from cyclops import kiosk as kiosk_module  # noqa: E402
from cyclops.audio import BLOCK_FRAMES, SAMPLE_RATE  # noqa: E402
from cyclops.config import load_settings  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures_572"
BLOCK_S = BLOCK_FRAMES / SAMPLE_RATE
ECHO_S = 0.15
ECHO_GAIN = 0.5
CLOSE_EXTRA_S = 1.5  # roughly the mux of a real session's video: he talks inside this
SILENT = b"\0\0" * BLOCK_FRAMES


def line(name: str) -> list[bytes]:
    with wave.open(str(FIXTURES / f"{name}.wav")) as w:
        assert w.getframerate() == SAMPLE_RATE and w.getnchannels() == 1
        pcm = w.readframes(w.getnframes())
    size = BLOCK_FRAMES * 2
    return [pcm[i:i + size].ljust(size, b"\0") for i in range(0, len(pcm), size)]


class Room:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.talk: list[bytes] = []
        self.echo: list[tuple[float, bytes]] = []
        self.out_times: list[float] = []

    def say(self, name: str) -> float:
        blocks = line(name)
        with self.lock:
            self.talk = list(blocks)
        return time.monotonic() + len(blocks) * BLOCK_S

    def played(self, block: bytes) -> None:
        now = time.monotonic()
        with self.lock:
            self.out_times.append(now)
            self.echo.append((now + ECHO_S, block))

    def mic(self) -> bytes:
        now = time.monotonic()
        with self.lock:
            voice = self.talk.pop(0) if self.talk else SILENT
            back = SILENT
            if self.echo and self.echo[0][0] <= now:
                back = self.echo.pop(0)[1]
        mixed = (np.frombuffer(voice, dtype=np.int16).astype(np.float32)
                 + np.frombuffer(back, dtype=np.int16).astype(np.float32) * ECHO_GAIN)
        return np.clip(mixed, -32768, 32767).astype(np.int16).tobytes()


ROOM = Room()


class Stream:
    def __init__(self, *, callback, **_kw) -> None:  # noqa: ANN001
        self.callback, self.run = callback, False

    def start(self) -> None:
        self.run = True
        threading.Thread(target=self.tick, daemon=True).start()

    def stop(self) -> None:
        self.run = False

    def close(self) -> None:
        self.run = False

    def tick(self) -> None:
        clock = time.monotonic()
        while self.run:
            self.block()
            clock += BLOCK_S
            time.sleep(max(0.0, clock - time.monotonic()))


class FakeIn(Stream):
    def block(self) -> None:
        pcm = np.frombuffer(ROOM.mic(), dtype=np.int16).reshape(-1, 1)
        self.callback(pcm, BLOCK_FRAMES, None, None)


class FakeOut(Stream):
    def block(self) -> None:
        out = bytearray(BLOCK_FRAMES * 2)
        self.callback(memoryview(out), BLOCK_FRAMES, None, None)
        if any(out):
            ROOM.played(bytes(out))


CUES: list[str] = []
DELAY_CLOSE = [False]


def install() -> None:
    session.SessionLog._after = lambda self: None  # noqa: SLF001
    audio.sd.RawInputStream = FakeIn
    audio.sd.RawOutputStream = FakeOut
    audio.pin_input_source = lambda device: "room"
    ear.pin_input_source = lambda device: "room"
    ui.output_is_speaker = lambda name: True
    ui.default_output_name = lambda dev: "probe"
    kiosk_module.resolve_device = lambda d: None
    real_play = sfx.Cues.play

    def play(self, name, **kw):  # noqa: ANN001, ANN003
        CUES.append(name)
        return 0.0

    sfx.Cues.play = play
    _ = real_play
    real_stop = audio.Speaker.stop

    def stop(self) -> None:
        real_stop(self)
        if DELAY_CLOSE[0]:
            time.sleep(CLOSE_EXTRA_S)

    audio.Speaker.stop = stop

    async def no_camera(*_a, **_k):  # noqa: ANN002, ANN003
        raise RuntimeError("no camera in this check")

    if hasattr(agent_module, "capture_image_async"):
        agent_module.capture_image_async = no_camera


class Box:
    """The kiosk's frame loop, reduced to the part that tends the ear."""

    def __init__(self, settings) -> None:  # noqa: ANN001
        self.controller = ui.SessionController(settings, frames=None, entrypoint="qa-572")
        k = object.__new__(kiosk_module.Kiosk)
        k.controller = self.controller
        k._cues = type("C", (), {"waiting": lambda self: 0.0})()
        k._menu, k._wifi, k._online = False, None, True
        k._asleep, k._touched_at = False, 0.0
        k._pending, k._pending_at = None, 0.0
        k._ear, k._ear_state, k._ear_quiet_until, k._ear_retry_at = None, "", 0.0, 0.0
        self.k = k
        self.starts = 0
        real_start = self.controller.start

        def start(*, ear=None):  # noqa: ANN001, ANN202
            self.starts += 1
            real_start(ear=ear)

        self.controller.start = start
        self.agents: list = []
        self.alive = True
        threading.Thread(target=self.frames, daemon=True).start()

    def frames(self) -> None:
        while self.alive:
            status = self.controller.status()
            self.k._effective(str(status["state"]))
            tasks.line()
            self.k._tend_ear(str(status["state"]), time.monotonic())
            agent = self.controller._agent
            if agent is not None and (not self.agents or self.agents[-1] is not agent):
                self.agents.append(agent)
            time.sleep(1 / kiosk_module.TARGET_FPS)

    def wait_up(self, timeout: float = 15.0) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            agent = self.controller._agent
            if agent is not None and agent.ready.is_set():
                return True
            time.sleep(0.01)
        return False

    def wait_gone(self, timeout: float = 30.0) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if not self.controller.running:
                return True
            time.sleep(0.01)
        return False


def said_since(agent, n: int) -> list[tuple[str, str]]:  # noqa: ANN001
    return list(agent.spoken[n:])


def wait_said(agent, who: str, start: int, timeout: float = 20.0) -> bool:  # noqa: ANN001
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if any(w == who for w, _ in agent.spoken[start:]):
            return True
        time.sleep(0.05)
    return False


def voiced_session(box: Box, opener: str) -> object | None:
    before = box.starts
    ROOM.say(opener)
    end = time.monotonic() + 15
    while box.starts == before and time.monotonic() < end:
        time.sleep(0.02)
    if box.starts == before:
        print(f"   ear never started a session on {opener!r}")
        return None
    box.wait_up()
    agent = box.controller._agent
    wait_said(agent, "cyclops", 0)
    time.sleep(2.0)
    return agent


def settle(box: Box) -> None:
    while box.controller._agent is not None and box.controller.speaking:
        time.sleep(0.05)
    time.sleep(1.0)


def t1_silent_start(box: Box) -> None:
    print("T1 press-started session, nobody talks for 10 s")
    CUES.clear()
    ROOM.out_times.clear()
    box.k._pending, box.k._pending_at = "start", time.monotonic()
    box.controller.start()
    box.wait_up()
    agent = box.controller._agent
    time.sleep(10)
    print(f"   audio blocks out: {len(ROOM.out_times)} · cues: {CUES} · said: {agent.spoken}")
    box.k._pending, box.k._pending_at = "stop", time.monotonic()
    stopped = time.monotonic()
    box.controller.stop()
    box.wait_gone()
    print(f"   close {time.monotonic() - stopped:.2f} s · cues after close: {CUES}")


def t2_goodbye(box: Box, phrase: str) -> None:
    print(f"T2 voice start, a question, then goodbye {phrase!r}")
    CUES.clear()
    agent = voiced_session(box, "ask")
    if agent is None:
        return
    settle(box)
    mark = len(agent.spoken)
    ROOM.out_times.clear()
    ended = ROOM.say(phrase)
    gone = box.wait_gone(25)
    print(f"   ended itself: {gone}, {time.monotonic() - ended:.2f} s after his last word")
    print(f"   said after goodbye: {said_since(agent, mark)}")
    out = ROOM.out_times
    if out:
        print(f"   voice after goodbye: {(out[-1] - out[0]) + BLOCK_S:.2f} s of output")
    else:
        print("   voice after goodbye: none")
    starts = box.starts
    time.sleep(10)
    print(f"   new sessions in the 10 s after: {box.starts - starts} (its own voice mustn't start"
          f" one) · cues: {CUES}")
    if box.starts != starts:
        box.wait_up()
        agent = box.controller._agent
        print(f"   FALSE START said: {agent.spoken if agent else '?'}")
        box.controller.stop()
        box.wait_gone()


def t3_talk_during_close(box: Box) -> None:
    print("T3 question, stop, and he talks during the close")
    agent = voiced_session(box, "ask")
    if agent is None:
        return
    print(f"   first session said: {agent.spoken}")
    settle(box)
    DELAY_CLOSE[0] = True
    starts = box.starts
    box.k._pending, box.k._pending_at = "stop", time.monotonic()
    stopped = time.monotonic()
    box.controller.stop()
    time.sleep(0.3)
    print(f"   closing={box.controller.closing} as he starts talking")
    last_word = ROOM.say("again")
    end = time.monotonic() + 15
    while box.starts == starts and time.monotonic() < end:
        time.sleep(0.02)
    DELAY_CLOSE[0] = False
    if box.starts == starts:
        print("   LOST: no new session started")
        return
    print(f"   new session asked for {time.monotonic() - stopped:.2f} s after the stop")
    box.wait_up()
    second = box.controller._agent
    wait_said(second, "cyclops", 0)
    first_out = time.monotonic()
    time.sleep(4)
    print(f"   heard and said: {second.spoken}")
    print(f"   carried in: {len(second.earlier)} turns · answer about {first_out - last_word:.2f} s"
          " after his last word")
    box.k._pending, box.k._pending_at = "stop", time.monotonic()
    box.controller.stop()
    box.wait_gone()


def t4_unprompted(box: Box, opener: str) -> None:
    print(f"T4 new session soon after, opener {opener!r}: must not raise the old topic")
    box.wait_gone()
    agent = voiced_session(box, opener)
    if agent is None:
        return
    time.sleep(3)
    print(f"   carried in: {len(agent.earlier)} turns · said: {agent.spoken}")
    box.k._pending, box.k._pending_at = "stop", time.monotonic()
    box.controller.stop()
    box.wait_gone()


def t5_stop_mid_answer(box: Box) -> None:
    print("T5 a question, stopped while its voice is coming out and echoing: must start nothing")
    before = box.starts
    ROOM.say("ask")
    end = time.monotonic() + 15
    while box.starts == before and time.monotonic() < end:
        time.sleep(0.02)
    box.wait_up()
    ROOM.out_times.clear()
    end = time.monotonic() + 20
    while len(ROOM.out_times) < 40 and time.monotonic() < end:  # 0.8 s into its answer
        time.sleep(0.01)
    print(f"   its voice out for {len(ROOM.out_times) * BLOCK_S:.2f} s when stopped")
    starts = box.starts
    box.k._pending, box.k._pending_at = "stop", time.monotonic()
    box.controller.stop()
    box.wait_gone()
    time.sleep(10)
    print(f"   new sessions in the 10 s after: {box.starts - starts}")
    if box.starts != starts:
        box.wait_up()
        agent = box.controller._agent
        time.sleep(4)
        print(f"   FALSE START said: {agent.spoken if agent else '?'}")
        box.controller.stop()
        box.wait_gone()


def t6_talk_after_stop_mid_answer(box: Box) -> None:
    print("T6 stopped while its voice is coming out, and he talks during the close")
    before = box.starts
    ROOM.say("ask")
    end = time.monotonic() + 15
    while box.starts == before and time.monotonic() < end:
        time.sleep(0.02)
    box.wait_up()
    ROOM.out_times.clear()
    end = time.monotonic() + 20
    while len(ROOM.out_times) < 40 and time.monotonic() < end:  # 0.8 s into its answer
        time.sleep(0.01)
    print(f"   its voice out for {len(ROOM.out_times) * BLOCK_S:.2f} s when stopped")
    DELAY_CLOSE[0] = True
    starts = box.starts
    box.k._pending, box.k._pending_at = "stop", time.monotonic()
    stopped = time.monotonic()
    box.controller.stop()
    time.sleep(0.8)  # past the speaker's tail and the ear's hold, if neither is stuck
    speaker = box.controller._speaker
    print(f"   0.8 s into the close: closing={box.controller.closing}"
          f" speaking={box.controller.speaking}"
          f" buffered={len(speaker._buffer) if speaker else 'gone'}")  # noqa: SLF001
    ROOM.say("again")
    end = time.monotonic() + 15
    while box.starts == starts and time.monotonic() < end:
        time.sleep(0.02)
    DELAY_CLOSE[0] = False
    if box.starts == starts:
        print("   LOST: no new session started")
        return
    print(f"   new session asked for {time.monotonic() - stopped:.2f} s after the stop")
    box.wait_up()
    second = box.controller._agent
    wait_said(second, "cyclops", 0)
    time.sleep(4)
    print(f"   heard and said: {second.spoken}")
    box.k._pending, box.k._pending_at = "stop", time.monotonic()
    box.controller.stop()
    box.wait_gone()


def recheck_round(settings) -> None:  # noqa: ANN001
    box = Box(settings)
    t6_talk_after_stop_mid_answer(box)
    time.sleep(3)
    t6_talk_after_stop_mid_answer(box)
    box.alive = False


def short_word(box: Box) -> None:
    print("T4b 'Hello?' 3 s after a close: a word back, and no old topic")
    time.sleep(3)
    t4_unprompted(box, "hello")


def gate_check() -> None:
    for name in ("ask", "bye1", "bye2", "bye3", "again", "hello", "water"):
        gate = ear.SpeechGate()
        fired = next((i for i, b in enumerate(line(name)) if gate.feed(b)), None)
        at = "never" if fired is None else f"fires at {fired * BLOCK_S:.2f} s"
        print(f"gate on {name}: {at}")


def smoke_check() -> None:
    import base64

    import cv2

    from cyclops import smoke
    from cyclops.webcam import Capture

    async def picture(*_a, save_dir=None, **_k):  # noqa: ANN002, ANN003, ANN001
        # The live kiosk holds the real camera, and a picture of his bench isn't needed here.
        frame = np.full((480, 640, 3), 200, dtype=np.uint8)
        cv2.rectangle(frame, (160, 120), (480, 360), (40, 40, 200), -1)
        jpeg = cv2.imencode(".jpg", frame)[1].tobytes()
        where = Path(save_dir or SCRATCH / "captures")
        where.mkdir(parents=True, exist_ok=True)
        (where / "latest.jpg").write_bytes(jpeg)
        path = where / "qa.jpg"
        path.write_bytes(jpeg)
        url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
        return Capture(url, path, 640, 480, len(jpeg), 0)

    smoke.capture_image_async = picture
    if hasattr(agent_module, "capture_image_async"):
        agent_module.capture_image_async = picture
    began = time.monotonic()
    try:
        smoke.main()
        print(f"smoke returned after {time.monotonic() - began:.0f} s")
    except BaseException as exc:  # noqa: BLE001 - reporting what it does, SystemExit included
        print(f"smoke ended after {time.monotonic() - began:.0f} s: {type(exc).__name__}: {exc}")


def second_round(settings) -> None:  # noqa: ANN001
    gate_check()
    box = Box(settings)
    voiced_session(box, "ask")
    box.controller.stop()
    box.wait_gone()
    short_word(box)
    short_word(box)
    t5_stop_mid_answer(box)
    t5_stop_mid_answer(box)
    box.alive = False
    smoke_check()


def main() -> None:
    install()
    settings = load_settings()
    paths = [settings.sessions_dir, settings.projects_dir, settings.captures_dir,
             settings.manuals_dir, settings.about_file, tasks.TASKS_FILE,
             config_module.PANEL_FILE, config_module.VOICE_FILE]
    for where in paths:
        if not Path(where).resolve().is_relative_to(SCRATCH.resolve()):
            raise SystemExit(f"refusing to run: {where} is not scratch")
    if not settings.voice_wake:
        raise SystemExit("voice_wake is off in settings: the ear never opens")
    print(f"scratch: {SCRATCH} · model {settings.model} · quiet_end {settings.quiet_end_s}")
    if "--second" in sys.argv:
        second_round(settings)
        return
    if "--smoke" in sys.argv:
        smoke_check()
        return
    if "--recheck" in sys.argv:
        recheck_round(settings)
        return
    box = Box(settings)
    t1_silent_start(box)
    time.sleep(2)
    for phrase in ("bye1", "bye2", "bye3"):
        t2_goodbye(box, phrase)
    t3_talk_during_close(box)
    t3_talk_during_close(box)
    t4_unprompted(box, "hello")
    t4_unprompted(box, "water")
    box.alive = False
    print(f"sessions started in all: {box.starts} · all cues: {CUES}")


if __name__ == "__main__":
    main()
