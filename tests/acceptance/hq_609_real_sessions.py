#!/usr/bin/env python3
"""hq#609 against the real voice model and the real Ear, on the kiosk's real start path.

Run on the Pi by hq_609_pi.sh, from a throwaway copy, with its working directory in the repo so
`.env` is found. Usage: hq_609_real_sessions.py <clips folder> <blocks of A/B runs>

Real: Kiosk._tend_ear on a 30 fps loop, Ear, SpeechGate, SpeechCheck, SessionController, VoiceAgent,
the model. Fake: the sound card. Recordings go into the Ear's callback; nothing is played aloud and
the real mic isn't opened, so the live kiosk can't hear any of this. Not covered: his mic and room.

Isolation (hq#573): hq_572_real_sessions sets HOME and every record folder to scratch before cyclops
is imported, and this checks them again. Recording is off, and filing is replaced with a no-op.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hq_572_real_sessions as h  # noqa: E402  (sets HOME and the record folders to scratch)

from cyclops import ear, tasks  # noqa: E402
from cyclops import config as config_module  # noqa: E402
from cyclops.audio import BLOCK_FRAMES, SAMPLE_RATE  # noqa: E402
from cyclops.config import load_settings  # noqa: E402

h.FIXTURES = Path(__file__).resolve().parent / "fixtures_609"
h.ECHO_GAIN = 0.0  # the real room's echo is for Marco's check; see hq_599_real_sessions
RESULTS: list[bool] = []
SIZE = BLOCK_FRAMES * 2
NOISES = (
    "keyboard_typing",
    "door_wood_knock",
    "coughing",
    "glass_breaking",
    "can_opening",
    "clapping",
    "mouse_click",
    "rain",
    "vacuum_cleaner",
)
SESSIONS_BEFORE = 0


def check(name: str, ok: bool, evidence: object = "") -> None:
    RESULTS.append(bool(ok))
    tail = f" · {str(evidence)[:320]}" if evidence != "" else ""
    print(f"   {'PASS' if ok else 'FAIL'} {name}{tail}", flush=True)


def guard() -> None:
    with urllib.request.urlopen("http://localhost/api/live", timeout=3) as r:  # noqa: S310
        name = json.load(r).get("name")
    if name is not None:
        raise SystemExit(f"ABORT: a real session is live: {name}")


def load(path: Path) -> list[bytes]:
    pcm = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "s16le",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return [pcm[i : i + SIZE].ljust(SIZE, b"\0") for i in range(0, len(pcm), SIZE)]


def play(blocks: list[bytes]) -> float:
    with h.ROOM.lock:
        h.ROOM.talk = list(blocks)
    return time.monotonic()


def mix(voice: list[bytes], noise: list[bytes], noise_gain: float) -> list[bytes]:
    import numpy as np

    out = []
    for i, block in enumerate(voice):
        a = np.frombuffer(block, dtype=np.int16).astype(np.float32)
        b = np.frombuffer(noise[i % len(noise)], dtype=np.int16).astype(np.float32) * noise_gain
        out.append(np.clip(a + b, -32768, 32767).astype(np.int16).tobytes())
    return out


def end_session(box: h.Box) -> None:
    box.k._pending, box.k._pending_at = "stop", time.monotonic()
    box.controller.stop()
    box.wait_gone()
    time.sleep(2.0)  # past the ear's hold, and the next ear is open
    assert box.k._ear is not None or not box.alive, "the ear did not come back"


def you_said(agent) -> str:  # noqa: ANN001
    return " ".join(text for who, text in agent.spoken if who == "you")


def started_by(box: h.Box, before: int, timeout: float) -> bool:
    end = time.monotonic() + timeout
    while box.starts == before and time.monotonic() < end:
        time.sleep(0.02)
    return box.starts != before


def heard_whole(box: h.Box, label: str, blocks: list[bytes], want: tuple[str, ...]) -> None:
    guard()
    before = box.starts
    t0 = play(blocks)
    if not started_by(box, before, 12):
        check(f"{label}: starts a session", False, "the ear never started one")
        return
    took = time.monotonic() - t0
    box.wait_up()
    agent = box.controller._agent
    h.wait_said(agent, "you", 0, timeout=25)
    h.wait_said(agent, "cyclops", 0, timeout=25)
    time.sleep(1.5)
    said = you_said(agent).lower()
    check(f"{label}: starts a session", True, f"{took:.2f} s after the sound began")
    check(f"{label}: first words whole", all(w in said for w in want), repr(said))
    print(f"      cyclops: {[t for w, t in agent.spoken if w == 'cyclops'][:1]}", flush=True)
    end_session(box)


def first_words(box: h.Box, clips: Path) -> None:
    print("=== first words, from a real start", flush=True)
    for name, want in (
        ("torque_samantha", ("hey", "torque", "bolt")),
        ("torque_daniel", ("hey", "torque", "bolt")),
        ("time_karen", ("what time",)),
        ("okay_samantha", ("okay", "oil")),
        ("torque_quiet", ("hey", "torque", "bolt")),
    ):
        heard_whole(box, name, h.line(name), want)
    typing = load(sorted((clips / "keyboard_typing").iterdir())[0])
    vacuum = load(sorted((clips / "vacuum_cleaner").iterdir())[0])
    heard_whole(
        box,
        "typing, then straight into speech",
        typing + h.line("torque_samantha"),
        ("hey", "torque"),
    )
    heard_whole(box, "over a vacuum", mix(h.line("torque_samantha"), vacuum, 0.35), ("torque",))


def noise(box: h.Box, clips: Path) -> None:
    print("=== noise, through the kiosk's real start path, 3 clips each", flush=True)
    total = 0
    for name in (*NOISES, "music", "laughing"):
        started = 0
        files = sorted((clips / name).iterdir())
        for path in files[:: max(1, len(files) // 3)][:3]:
            guard()
            before = box.starts
            blocks = load(path)
            play(blocks)
            if started_by(box, before, len(blocks) * h.BLOCK_S + 2.5):
                started += 1
                box.wait_up()
                print(f"      started on {name}/{path.name}", flush=True)
                end_session(box)
        total += started
        if name in NOISES:
            check(f"{name}: no session", started == 0, f"{started} of 3")
        else:
            print(f"   INFO {name}: {started} of 3 (swe's limits say it can start one)", flush=True)
    filed = len(list(Path(box.controller.settings.sessions_dir).iterdir()))
    print(f"   sessions started by noise: {total}; session folders in scratch: {filed}", flush=True)


def tap_during_check(box: h.Box, clips: Path) -> None:
    print("=== the hold's path, in the middle of a check", flush=True)
    guard()
    before = box.starts
    made = box.k._ear
    play(load(sorted((clips / "keyboard_typing").iterdir())[0]))
    end = time.monotonic() + 4
    while made is not None and not made._attempt and time.monotonic() < end:
        time.sleep(0.005)
    busy = bool(made and made._attempt)
    box.k._toggle_session()  # what button_held calls, and the on-screen switch
    box.wait_up()
    check(
        "it starts at once, with a check running",
        box.starts == before + 1,
        f"check running: {busy}",
    )
    time.sleep(3)
    check("and only once", box.starts == before + 1, f"{box.starts - before}")
    check("the ear it interrupted is closed", made is None or made._closed)
    end_session(box)


def delay(box: h.Box, runs: int) -> None:
    print("=== delay: same real Ear, second check on and off, from the first word", flush=True)
    real_feed = ear.SpeechCheck.feed
    numbers: dict[str, list[float]] = {"on": [], "off": []}
    starts: dict[str, list[float]] = {"on": [], "off": []}
    for block in range(runs):
        for side in ("off", "on") if block % 2 else ("on", "off"):
            for _ in range(3):
                guard()
                ear.SpeechCheck.feed = real_feed if side == "on" else (lambda self, b: True)
                before = box.starts
                h.ROOM.out_times.clear()
                t0 = play(h.line("time_karen"))
                if not started_by(box, before, 12):
                    print(f"      {side}: never started", flush=True)
                    continue
                starts[side].append(time.monotonic() - t0)
                end = time.monotonic() + 25
                while not h.ROOM.out_times and time.monotonic() < end:
                    time.sleep(0.005)
                if h.ROOM.out_times:
                    numbers[side].append(h.ROOM.out_times[0] - t0)
                h.wait_said(box.controller._agent, "cyclops", 0, timeout=10)
                time.sleep(1)
                end_session(box)
    ear.SpeechCheck.feed = real_feed
    for side in ("off", "on"):
        print(
            f"   second check {side}: to session start, median "
            f"{statistics.median(starts[side]):.2f} s; to first audio, median "
            f"{statistics.median(numbers[side]):.2f} s, range {min(numbers[side]):.2f} to "
            f"{max(numbers[side]):.2f} (n={len(numbers[side])}) {[round(x, 2) for x in numbers[side]]}",
            flush=True,
        )
    added = statistics.median(numbers["on"]) - statistics.median(numbers["off"])
    check(
        "adds no more than about half a second to the first reply", added <= 0.5, f"{added:+.2f} s"
    )


class Timeline:
    """One room, heard whole by every open stream, as a real sound card would. Room.talk pops: with
    the ear and the session's mic both open during a connect they split the speech between them, and
    each hears a chopped version. That garbled transcripts here until this replaced it."""

    def __init__(self) -> None:
        self.blocks: list[bytes] = []
        self.t0 = 0.0

    def install(self) -> None:
        room = self

        class Each(h.Stream):
            def block(self) -> None:
                index = int((time.monotonic() - room.t0) / h.BLOCK_S)
                pcm = room.blocks[index] if 0 <= index < len(room.blocks) else h.SILENT
                self.callback(
                    h.np.frombuffer(pcm, dtype=h.np.int16).reshape(-1, 1), BLOCK_FRAMES, None, None
                )

        h.audio.sd.RawInputStream = Each
        ear.sd.RawInputStream = Each

    def say(self, blocks: list[bytes]) -> float:
        self.blocks, self.t0 = list(blocks), time.monotonic()
        return self.t0


def span_s(pcm: bytes) -> float:
    """Seconds from the first to the last sound in what reached the model."""
    import numpy as np

    data = np.abs(np.frombuffer(pcm, dtype=np.int16))
    size = BLOCK_FRAMES
    loud = [i for i in range(0, len(data) - size, size) if data[i : i + size].max() > 200]
    return (loud[-1] - loud[0] + size) / SAMPLE_RATE if loud else 0.0


def handover(box: h.Box, rounds: int) -> None:
    """What the model is sent, against the recording, for speech that runs across the connect."""
    from cyclops import agent as agent_module

    print("=== what reaches the model, speech running across the connect", flush=True)
    timeline = Timeline()
    timeline.install()
    box.k._close_ear()  # it was opened on the old stream; the frame loop opens a new one
    time.sleep(1.0)
    assert box.k._ear is not None, "the ear did not reopen"
    sent: list[bytes] = []
    real_opening = agent_module.VoiceAgent._send_opening

    async def opening(self, pcm):  # noqa: ANN001, ANN202
        real_append = self.conn.input_audio_buffer.append

        async def append(*, audio):  # noqa: ANN001, ANN202
            sent.append(h_b64(audio))
            return await real_append(audio=audio)

        self.conn.input_audio_buffer.append = append
        return await real_opening(self, pcm)

    agent_module.VoiceAgent._send_opening = opening
    real_feed = ear.SpeechCheck.feed
    h.FIXTURES = Path(__file__).resolve().parent / "fixtures_572"
    blocks = h.line("ask")
    length = len(blocks) * h.BLOCK_S
    results: dict[tuple[str, float], list[str]] = {}
    for block in range(rounds):
        for side in ("off", "on") if block % 2 else ("on", "off"):
            ear.SpeechCheck.feed = real_feed if side == "on" else (lambda self, b: True)
            for wait in (2.5, 0.7):
                guard()
                before = box.starts
                sent.clear()
                timeline.say(blocks)
                if not started_by(box, before, 8):
                    results.setdefault((side, wait), []).append("NEVER STARTED")
                    time.sleep(wait)
                    continue
                box.wait_up()
                agent = box.controller._agent
                h.wait_said(agent, "you", 0, timeout=15)
                time.sleep(max(0.0, length + 1 - (time.monotonic() - timeline.t0)))
                span = span_s(b"".join(sent))
                results.setdefault((side, wait), []).append(f"{span:.2f}s {you_said(agent)[:48]!r}")
                box.k._pending, box.k._pending_at = "stop", time.monotonic()
                box.controller.stop()
                box.wait_gone()
                time.sleep(wait)
    ear.SpeechCheck.feed = real_feed
    print(f"   the recording is {length:.2f} s long", flush=True)
    for key in sorted(results):
        print(f"   check {key[0]}, {key[1]} s after the last session: {results[key]}", flush=True)


def h_b64(audio: str) -> bytes:
    import base64

    return base64.b64decode(audio)


def words(box: h.Box, rounds: int) -> None:
    """A one-word start and a question, soon after a session ended: is any loss the second check's?"""
    print("=== one word and first words, second check on and off", flush=True)
    real_feed = ear.SpeechCheck.feed
    fixtures = Path(__file__).resolve().parent / "fixtures_572"
    h.FIXTURES = fixtures
    tally: dict[tuple[str, str, str], list[str]] = {}
    for block in range(rounds):
        for side in ("off", "on") if block % 2 else ("on", "off"):
            ear.SpeechCheck.feed = real_feed if side == "on" else (lambda self, b: True)
            for name, wait in (("hello", 2.5), ("hello", 0.7), ("ask", 2.5), ("ask", 0.7)):
                guard()
                before = box.starts
                play(h.line(name))
                if not started_by(box, before, 8):
                    tally.setdefault((name, f"{wait}", side), []).append("NEVER STARTED")
                    time.sleep(wait)
                    continue
                box.wait_up()
                agent = box.controller._agent
                h.wait_said(agent, "you", 0, timeout=15)
                said = you_said(agent)[:40]
                tally.setdefault((name, f"{wait}", side), []).append(said or "(nothing heard)")
                box.k._pending, box.k._pending_at = "stop", time.monotonic()
                box.controller.stop()
                box.wait_gone()
                time.sleep(wait)
    ear.SpeechCheck.feed = real_feed
    for key in sorted(tally):
        print(
            f"   {key[0]}, {key[1]} s after the last session, check {key[2]}: {tally[key]}",
            flush=True,
        )


def main() -> None:
    clips, runs = Path(sys.argv[1]), int(sys.argv[2])
    h.install()
    settings = load_settings()
    paths = [
        settings.sessions_dir,
        settings.projects_dir,
        settings.captures_dir,
        settings.manuals_dir,
        settings.about_file,
        tasks.TASKS_FILE,
        config_module.PANEL_FILE,
        config_module.VOICE_FILE,
    ]
    for where in paths:
        if not Path(where).resolve().is_relative_to(h.SCRATCH.resolve()):
            raise SystemExit(f"refusing to run: {where} is not scratch")
    if not settings.voice_wake:
        raise SystemExit("voice_wake is off in settings: the ear never opens")
    print(f"scratch: {h.SCRATCH} · code {ear.__file__}", flush=True)
    box = h.Box(settings)
    time.sleep(4)  # the model has loaded
    assert box.k._ear is not None, "the ear never opened"
    if len(sys.argv) > 3:
        {"words": words, "handover": handover}[sys.argv[3]](box, runs)
        box.alive = False
        return
    noise(box, clips)
    tap_during_check(box, clips)
    first_words(box, clips)
    delay(box, runs)
    box.alive = False
    print(
        f"SUMMARY {sum(RESULTS)} of {len(RESULTS)} passed; sessions started: {box.starts}",
        flush=True,
    )


if __name__ == "__main__":
    main()
