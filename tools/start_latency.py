#!/usr/bin/env python3
"""How long the first answer takes when a voice starts a session, next to an answer mid-session.

    uv run python tools/start_latency.py --runs 6            # on the Pi, from the repo

The kiosk's own path, end to end - ``SessionController.start(ear=...)``, the audio devices, the
session log, the echo guard, the agent - with only the sound card replaced: a scripted room plays
recorded speech (``tests/fixtures/speech``) into the mic, 20 ms a block in real time, and the
speaker's output is swallowed in real time. Nothing is said out loud and the real mic isn't used.

Each run:

1. He starts talking. ``--gate`` seconds in (what the ear takes to decide it's a voice), the
   session is started the way the kiosk starts one, with an ear holding the last 2 s of the room.
2. He finishes his question; the room stays quiet. The first answer's first audio is timed from
   his last word.
3. Once the answer has played out, he asks a second question, timed the same way. Then the
   session is stopped the way the button stops it, and the next run starts.

The questions swap places every other run. Sessions, projects and the about file are scratch and
nothing is filed afterwards, so a run leaves his records alone.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

# Everything a session can write goes to scratch, and nothing is filed afterwards: this probe once
# ran with only the sessions folder redirected, and the filer wrote its made-up questions into a
# real project's notes (arc-eng/hq#572). The scratch folders are created empty, so the prompt is
# smaller than on a used box - fine for a comparison, not for an absolute number.
_SCRATCH = Path(tempfile.mkdtemp(prefix="cyclops-latency-"))
_DIRS = (("SESSIONS", "sessions"), ("PROJECTS", "projects"), ("CAPTURES", "captures"),
         ("MANUALS", "manuals"))  # no manuals, so no lookup can put a page on his panel
for _name, _sub in _DIRS:
    (_SCRATCH / _sub).mkdir()
    os.environ[f"CYCLOPS_{_name}_DIR"] = str(_SCRATCH / _sub)
os.environ["CYCLOPS_ABOUT_FILE"] = str(_SCRATCH / "about-you.md")
os.environ["CYCLOPS_RECORD"] = "0"
os.environ["HOME"] = str(_SCRATCH)  # ~/.cache/cyclops: the task list, the panel file, the index

import numpy as np  # noqa: E402

from cyclops import audio, session, ui  # noqa: E402
from cyclops.audio import BLOCK_FRAMES, SAMPLE_RATE  # noqa: E402
from cyclops.config import load_settings  # noqa: E402
from cyclops.ear import PREROLL_BLOCKS  # noqa: E402

HERE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "speech"
BLOCK_S = BLOCK_FRAMES / SAMPLE_RATE
SILENT = b"\0\0" * BLOCK_FRAMES
ANSWER_MAX_S = 25.0


def blocks(path: Path) -> list[bytes]:
    with wave.open(str(path)) as w:
        assert w.getframerate() == SAMPLE_RATE and w.getnchannels() == 1 and w.getsampwidth() == 2
        pcm = w.readframes(w.getnframes())
    size = BLOCK_FRAMES * 2
    return [pcm[i:i + size].ljust(size, b"\0") for i in range(0, len(pcm), size)]


class Room:
    """What the mic hears: silence, except while a line is being said. One clock for everything."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._line: list[bytes] = []
        self._at = 0.0  # when the current line started
        self.heard: list[bytes] = []  # the last PREROLL_BLOCKS blocks, for the ear

    def say(self, line: list[bytes]) -> float:
        """Start saying ``line`` now; returns when its last word will be out."""
        with self._lock:
            self._line, self._at = line, time.monotonic()
            return self._at + len(line) * BLOCK_S

    def block(self) -> bytes:
        with self._lock:
            i = int((time.monotonic() - self._at) / BLOCK_S)
            out = self._line[i] if 0 <= i < len(self._line) else SILENT
            self.heard = (self.heard + [out])[-PREROLL_BLOCKS:]
            return out


ROOM = Room()
FIRST_AUDIO: list[float] = []
STOPPED: list[float] = []
PRIME = False  # when the server's VAD said he stopped, per turn


def watch(agent) -> None:  # noqa: ANN001
    """Write down when the server hears him stop: splits an answer's wait into noticing he's done
    and the model answering."""
    inner = agent._handle_event  # noqa: SLF001

    async def handle(event):  # noqa: ANN001
        if event.type == "input_audio_buffer.speech_stopped" and not STOPPED:
            STOPPED.append(time.monotonic())
        await inner(event)

    agent._handle_event = handle  # noqa: SLF001


class _Stream:
    """sounddevice's Raw*Stream, ticking its callback every 20 ms on its own thread."""

    def __init__(self, *, callback, **_kw) -> None:  # noqa: ANN001
        self._callback = callback
        self._run = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._run = True
        self._thread = threading.Thread(target=self._tick, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._run = False

    def close(self) -> None:
        self._run = False

    def _tick(self) -> None:
        clock = time.monotonic()
        while self._run:
            self._block()
            clock += BLOCK_S
            time.sleep(max(0.0, clock - time.monotonic()))


class FakeInput(_Stream):
    def _block(self) -> None:
        block = np.frombuffer(ROOM.block(), dtype=np.int16).reshape(-1, 1)
        self._callback(block, BLOCK_FRAMES, None, None)


class FakeOutput(_Stream):
    def _block(self) -> None:
        out = bytearray(BLOCK_FRAMES * 2)
        self._callback(memoryview(out), BLOCK_FRAMES, None, None)
        if any(out) and not FIRST_AUDIO:
            FIRST_AUDIO.append(time.monotonic())


class Ear:
    """The ear that heard him: what it kept is the room's last 2 s, at the moment it is taken."""

    def take(self) -> bytes:
        return b"".join(ROOM.heard)

    def close(self) -> None:
        pass


def install() -> None:
    session.SessionLog._after = lambda self: None  # noqa: SLF001 - never name, remember or file
    audio.sd.RawInputStream = FakeInput  # type: ignore[assignment]
    audio.sd.RawOutputStream = FakeOutput  # type: ignore[assignment]
    audio.pin_input_source = lambda device: "the probe's room"  # type: ignore[assignment]
    ui.output_is_speaker = lambda name: True  # the Pi's own speaker: half-duplex, as there
    ui.default_output_name = lambda dev: "probe"


def wait_answer(ended: float) -> float:
    while not FIRST_AUDIO and time.monotonic() - ended < ANSWER_MAX_S:
        time.sleep(0.01)
    if not FIRST_AUDIO:
        raise RuntimeError("no answer")
    return FIRST_AUDIO[0] - ended


def one_run(
    controller: ui.SessionController, first: list[bytes], second: list[bytes], gate_s: float
) -> dict:
    FIRST_AUDIO.clear()
    ended = ROOM.say(first)
    time.sleep(gate_s)
    asked = time.monotonic()
    controller.start(ear=Ear())
    while controller._agent is None or not controller._agent.ready.is_set():  # noqa: SLF001
        if controller._error:  # noqa: SLF001
            raise RuntimeError(controller._error)  # noqa: SLF001
        time.sleep(0.005)
    ready = time.monotonic()
    STOPPED.clear()
    watch(controller._agent)  # noqa: SLF001
    first_s = wait_answer(ended)
    first_heard = STOPPED[0] - ended if STOPPED else float("nan")
    agent = controller._agent  # noqa: SLF001
    while agent.speaker is not None and (agent.speaker.is_audible or agent._response_active):  # noqa: SLF001
        time.sleep(0.05)
    time.sleep(1.0)
    FIRST_AUDIO.clear()
    STOPPED.clear()
    ended = ROOM.say(second)
    mid_s = wait_answer(ended)
    mid_heard = STOPPED[0] - ended if STOPPED else float("nan")
    stopping = time.monotonic()
    controller.stop()
    while controller._running:  # noqa: SLF001
        time.sleep(0.05)
    return {"start_s": ready - asked, "first_s": first_s, "mid_s": mid_s,
            "close_s": time.monotonic() - stopping,
            "first_heard_s": first_heard, "mid_heard_s": mid_heard}


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--runs", type=int, default=6)
    p.add_argument("--gate", type=float, default=0.5,
                   help="seconds of speech before the ear starts a session")
    p.add_argument("--short", action="store_true",
                   help="open with a short line (about 1 s), the case where the start can't hide")
    args = p.parse_args()
    install()
    settings = load_settings()
    for where in (settings.sessions_dir, settings.projects_dir, settings.captures_dir,
                  settings.manuals_dir, settings.about_file):
        if not where.resolve().is_relative_to(_SCRATCH.resolve()):
            raise SystemExit(f"refusing to run: {where} is not scratch")
    controller = ui.SessionController(settings, frames=None, entrypoint="latency-probe")
    q = [blocks(HERE / f"{'s' if args.short else 'q'}{i}.wav") for i in (1, 2)]
    rows = []
    for n in range(args.runs):
        a, b = (q[0], q[1]) if n % 2 == 0 else (q[1], q[0])
        r = one_run(controller, a, b, args.gate)
        rows.append(r)
        print(f"run {n + 1}: start {r['start_s']:.2f} s · first answer {r['first_s']:.2f} s"
              f" · mid-session {r['mid_s']:.2f} s  (he was heard to stop after"
              f" {r['first_heard_s']:.2f} / {r['mid_heard_s']:.2f} s; close {r['close_s']:.2f} s)",
              file=sys.stderr, flush=True)
        time.sleep(2.0)
    med = {k: statistics.median(r[k] for r in rows) for k in rows[0]}
    print(f"median of {len(rows)}: start {med['start_s']:.2f} s"
          f" · first answer {med['first_s']:.2f} s · mid-session {med['mid_s']:.2f} s"
          f" · close {med['close_s']:.2f} s",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
