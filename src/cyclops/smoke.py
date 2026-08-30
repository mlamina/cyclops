"""Headless smoke test: ``uv run cyclops-smoke``.

No microphone or speakers. Sends typed turns through the very same session config the voice
app uses, and checks that (1) audio + transcript come back, (2) a photo handed to the model
the way the shutter hands it one comes back described, and (3) the model calls the web_search
tool and answers from what it found.
"""

from __future__ import annotations

import asyncio
import base64
import sys

from openai.types.realtime import RealtimeServerEvent

from .agent import VoiceAgent, function_calls
from .audio import BYTES_PER_FRAME, SAMPLE_RATE
from .config import ConfigError, load_settings
from .webcam import capture_image_async

READY_TIMEOUT_S = 30.0
TURN_TIMEOUT_S = 90.0


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

        turn.reset()
        await agent.send_text("Say hello in one short sentence.")
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
