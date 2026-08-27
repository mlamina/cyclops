"""Command-line entrypoint: ``uv run cyclops``."""

from __future__ import annotations

import asyncio
import sys
from dataclasses import replace

from openai import OpenAIError
from websockets.exceptions import WebSocketException

from .agent import SessionError, VoiceAgent
from .audio import (
    EchoGuard,
    Microphone,
    Speaker,
    capture_sources,
    default_output_name,
    list_devices,
    output_is_speaker,
    preferred_source,
    resolve_device,
)
from .config import ConfigError, Settings, load_settings
from .session import SessionLog


def resolve_half_duplex(settings: Settings, output_name: str) -> tuple[bool, str]:
    """Decide whether to mute the mic while the assistant talks, and say why."""
    if settings.half_duplex is not None:
        forced = "on" if settings.half_duplex else "off"
        return settings.half_duplex, f"CYCLOPS_HALF_DUPLEX={forced} (output: {output_name})"
    if output_is_speaker(output_name):
        return True, f"output is '{output_name}', which the mic would hear"
    return False, f"output is '{output_name}'"


async def _run(settings: Settings) -> None:
    loop = asyncio.get_running_loop()
    in_dev = resolve_device(settings.input_device)
    out_dev = resolve_device(settings.output_device)
    output_name = default_output_name(out_dev)
    half_duplex, why = resolve_half_duplex(settings, output_name)
    settings = replace(settings, half_duplex=half_duplex)
    speaker = Speaker(device=out_dev)
    speaker.volume = settings.volume
    guard = None
    if half_duplex:
        guard = EchoGuard(loop, speaker, margin_db=settings.barge_in_db)
        if guard.barge_in_enabled:
            print(
                f"· speaker mode: mic muted while Cyclops talks — {why}.\n"
                f"  Talk clearly over it to interrupt (needs to be {settings.barge_in_db:g} dB "
                "above the echo; tune with CYCLOPS_BARGE_IN_DB, headphones give full duplex).",
                flush=True,
            )
        else:
            print(f"· half-duplex: mic muted while Cyclops talks, no barge-in — {why}.", flush=True)
    else:
        print(f"· full duplex with barge-in — {why}", flush=True)
    mic = Microphone(loop, guard=guard, device=in_dev)
    print(f"· mic: {mic.source or 'whatever PipeWire calls the default'}", flush=True)
    agent = VoiceAgent(settings, mic=mic, speaker=speaker, guard=guard)
    if guard is not None:
        guard.on_barge_in = agent.local_barge_in

    speaker.start()
    mic.start()
    # No camera is held open here, so no frames and no video.mp4 - the folder gets its transcript
    # and its photos and nothing else. That is by construction, not by a check.
    with SessionLog(settings, agent, entrypoint="cli", mic=mic, speaker=speaker):
        try:
            await agent.run()
        finally:
            mic.stop()
            speaker.stop()


def main() -> None:
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    try:
        asyncio.run(_run(settings))
    except KeyboardInterrupt:
        print("\n· bye", flush=True)
    except (SessionError, OpenAIError, WebSocketException, OSError) as exc:
        message = str(exc)
        if "invalid_api_key" in message:  # the server closes the socket with this reason
            message = "OpenAI rejected the API key. Check OPENAI_API_KEY in .env."
        print(f"error: {message}", file=sys.stderr)
        sys.exit(1)


def devices() -> None:
    """`cyclops-devices`: list audio input/output devices so you can pick one for the Pi."""
    print(list_devices())
    # PortAudio shows PipeWire as the single "pulse" device, which hides the fact that there is
    # more than one microphone behind it. This is the part you actually want when the wrong one
    # is listening, so it is printed alongside, with the one cyclops will choose marked.
    sources = capture_sources()
    if sources:
        chosen = preferred_source(sources)
        print("\nMicrophones behind the 'pulse' device — cyclops takes the marked one:")
        for name in sources:
            print(f"  {'→' if name == chosen else ' '} {name}")
        print("Set PULSE_SOURCE to one of these to override (e.g. to force the camera's mic).")
    print(
        "\nSet CYCLOPS_INPUT_DEVICE / CYCLOPS_OUTPUT_DEVICE to an index above or a name "
        "substring.\nOn a Raspberry Pi also set CYCLOPS_HALF_DUPLEX=1 if the output is a "
        "loudspeaker."
    )


if __name__ == "__main__":
    main()
