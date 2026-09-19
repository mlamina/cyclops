"""Headless smoke test: ``uv run cyclops-smoke``.

No microphone or speakers. Sends typed turns through the very same session config the voice
app uses, and checks that (1) it greets unprompted as the session comes up, with audio and a
transcript, (2) a photo handed to the model
the way the shutter hands it one comes back described, (3) the model calls the web_search tool
and answers from what it found, and (4) it draws a diagram that lands on the card.

The diagram turn is here rather than in ``tests/`` because it is the only check that runs the
whole tool - the drawing model, the validation, the write, and the record the session log keeps.
A TypeError in that record once threw *after* the picture was already on the panel, which cost
the log line and left the model waiting for a tool result that never came; nothing offline saw
it, because offline nothing calls the tool.
"""

from __future__ import annotations

import asyncio
import base64
import sys
import time

from openai.types.realtime import RealtimeServerEvent

from .agent import VoiceAgent, function_calls
from .audio import BYTES_PER_FRAME, SAMPLE_RATE
from .config import PANEL_FILE, ConfigError, load_settings
from .webcam import capture_image_async

READY_TIMEOUT_S = 30.0
TURN_TIMEOUT_S = 90.0
# How long turn 4 then waits for the picture itself. draw_diagram returns as soon as it is called
# and the image lands about half a minute later (imagine.DRAW_QUALITY), so the turn is over long
# before the drawing is - and checking the offer file the moment the turn ends would always find
# it empty. Comfortably past imagine.DRAW_TIMEOUT_S, so a slow draw fails on its own timeout with
# a message rather than on this one without.
DRAW_DEADLINE_S = 210.0


class TurnObserver:
    """Collects what came back for the current typed turn."""

    def __init__(self) -> None:
        self.done = asyncio.Event()
        self.audio_bytes = 0
        self.transcripts: list[str] = []
        self.tool_calls: list[str] = []

    def reset(self) -> None:
        self.done.clear()
        self.audio_bytes = 0
        self.transcripts = []
        self.tool_calls = []

    def __call__(self, event: RealtimeServerEvent) -> None:
        match event.type:
            case "response.output_audio.delta":
                self.audio_bytes += len(base64.b64decode(event.delta))
            case "response.output_audio_transcript.done":
                self.transcripts.append(event.transcript)
            case "response.done":
                calls = [call.name for call in function_calls(event.response)]
                self.tool_calls.extend(calls)
                if not calls:  # a tool-call response is followed by the real answer
                    self.done.set()

    @property
    def seconds_of_audio(self) -> float:
        return self.audio_bytes / (SAMPLE_RATE * BYTES_PER_FRAME)

    @property
    def has_transcript(self) -> bool:
        return any(t.strip() for t in self.transcripts)


def _offered_at() -> float:
    """When a picture was last handed to the panel, or 0.0 - see ``config.PANEL_FILE``.

    This is what a headless run can check. Smoke keeps no session (the photo turn uses
    ``captures/`` for the same reason), so a drawing has nowhere on the card to be written. But
    the offer is made either way, and a file that appears is proof an image came back whole.
    """
    try:
        return PANEL_FILE.stat().st_mtime
    except OSError:
        return 0.0


async def _wait_for_offer(since: float, limit_s: float) -> bool:
    """Wait for a picture to reach the panel's offer file. False if none does in time.

    Polled rather than watched: it is one stat() a second against a drawing that takes ninety,
    and a smoke run has nothing better to do while it waits.
    """
    deadline = time.monotonic() + limit_s
    while time.monotonic() < deadline:
        if _offered_at() > since:
            return True
        await asyncio.sleep(1.0)
    return False


async def _await_or_fail(agent_task: asyncio.Task, event: asyncio.Event, limit_s: float) -> None:
    """Wait for ``event``; fail fast if the session dies first or ``limit_s`` elapses."""
    waiter = asyncio.ensure_future(event.wait())
    done, _ = await asyncio.wait(
        {agent_task, waiter}, timeout=limit_s, return_when="FIRST_COMPLETED"
    )
    waiter.cancel()
    if waiter in done:
        return
    if agent_task in done:
        agent_task.result()  # raises the connection error, if any
        raise RuntimeError("session closed before the turn finished")
    raise TimeoutError(f"no response within {limit_s:.0f}s")


async def _main() -> int:
    settings = load_settings()
    agent = VoiceAgent(settings)
    turn = TurnObserver()
    agent.on_event = turn
    agent_task = asyncio.create_task(agent.run())
    failures: list[str] = []
    try:
        await _await_or_fail(agent_task, agent.ready, READY_TIMEOUT_S)
        print("· session ready")

        # Nothing typed: the greeting is asked for by the session coming up, and `turn` has been
        # listening since before the connect, so it is already collecting it.
        await _await_or_fail(agent_task, turn.done, TURN_TIMEOUT_S)
        print(f"· turn 1: {turn.seconds_of_audio:.1f}s audio, transcript={turn.transcripts!r}")
        if turn.audio_bytes == 0:
            failures.append("turn 1 returned no audio")
        if not turn.has_transcript:
            failures.append("turn 1 returned no transcript")

        # The model has no camera of its own any more, so this is the shutter's path exactly:
        # take a photo, hand it over, and see whether the next thing it says is about the photo.
        turn.reset()
        capture = await capture_image_async(
            settings.camera_index, save_dir=settings.captures_dir
        )
        print(f"· photo {capture.width}x{capture.height} → {capture.path}")
        await agent.add_photo(capture)
        # A photo on its own is deliberately silent now, so the question is what makes the turn.
        await agent.send_text("What do you see in that photo? One short sentence.")
        await _await_or_fail(agent_task, turn.done, TURN_TIMEOUT_S)
        print(
            f"· turn 2: {turn.seconds_of_audio:.1f}s audio, transcript={turn.transcripts!r}"
        )
        if turn.audio_bytes == 0:
            failures.append("turn 2 returned no audio for the photo")
        if not turn.has_transcript:
            failures.append("turn 2 returned no transcript")
        latest = settings.captures_dir / "latest.jpg"
        if not latest.exists():
            failures.append(f"{latest} was not written")

        # Asked to look, it takes the photo itself and answers off it in the same turn.
        turn.reset()
        await agent.send_text("Take a look. What do you see? One short sentence.")
        await _await_or_fail(agent_task, turn.done, TURN_TIMEOUT_S)
        print(
            f"· turn 2b: tools={turn.tool_calls}, {turn.seconds_of_audio:.1f}s audio, "
            f"transcript={turn.transcripts!r}"
        )
        if "take_a_look" not in turn.tool_calls:
            failures.append("model did not call take_a_look when asked to look")
        elif turn.audio_bytes == 0:
            failures.append("take_a_look returned no audio")
        turn.reset()
        await agent.send_text(
            "Search the web for the torque spec of a Shimano Hollowtech II crank arm bolt, "
            "then tell me the number in one short sentence."
        )
        await _await_or_fail(agent_task, turn.done, TURN_TIMEOUT_S)
        print(
            f"· turn 3: tools={turn.tool_calls}, {turn.seconds_of_audio:.1f}s audio, "
            f"transcript={turn.transcripts!r}"
        )
        if "web_search" not in turn.tool_calls:
            failures.append("model did not call web_search")
        if not turn.has_transcript:
            failures.append("turn 3 returned no transcript")

        # A diagram, drawn for real. There is no panel and no session here, so the tool answers
        # "shown: false" and writes nothing to the card - both correct. What is under test is the
        # model calling it, an image coming back whole, and the tool answering at all: this turn
        # timing out is what a tool that raises instead of replying looks like from here.
        #
        # Its own deadline, and a long one. The tool returns the moment it is called and the
        # picture lands about half a minute later, so the turn finishes long before the drawing
        # does - and the ordinary TURN_TIMEOUT_S would have this checking for an offer that was
        # never going to be there yet.
        turn.reset()
        before = _offered_at()
        await agent.send_text(
            "Draw me a wiring diagram: a Raspberry Pi 5 GPIO 17 through a 1k resistor into the "
            "IN pin of a 5V relay module, with 5V to VCC and ground to GND."
        )
        await _await_or_fail(agent_task, turn.done, TURN_TIMEOUT_S)
        offered = await _wait_for_offer(before, DRAW_DEADLINE_S)
        print(
            f"· turn 4: tools={turn.tool_calls}, {turn.seconds_of_audio:.1f}s audio, "
            f"picture offered to the panel: {offered}"
        )
        if "draw_diagram" not in turn.tool_calls:
            failures.append("model did not call draw_diagram")
        elif not offered:
            # The tool answered, so nothing is stuck - but no picture reached the panel inside
            # the deadline, which means the drawing failed or is slower than it has ever been.
            failures.append(
                f"draw_diagram was called but nothing was drawn in {DRAW_DEADLINE_S:.0f}s"
            )
        if not turn.has_transcript:
            failures.append("turn 4 returned no transcript")
        # Leave nothing waiting: on the Pi this file is what the panel's page draws, and a
        # smoke run must not leave a picture sitting behind the kiosk window.
        PANEL_FILE.unlink(missing_ok=True)

        if agent.unacked_item_ids:
            failures.append(f"server never acknowledged items {sorted(agent.unacked_item_ids)}")
    finally:
        await agent.close()
        await asyncio.gather(agent_task, return_exceptions=True)

    if failures:
        print("SMOKE FAIL:\n  - " + "\n  - ".join(failures))
        return 1
    print("SMOKE PASS")
    return 0


def main() -> None:
    try:
        sys.exit(asyncio.run(_main()))
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    except (TimeoutError, RuntimeError) as exc:
        print(f"SMOKE FAIL: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
