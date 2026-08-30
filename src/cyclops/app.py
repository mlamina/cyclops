"""Command-line entrypoint: ``uv run cyclops``."""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Callable
from dataclasses import replace

from openai import OpenAIError
from websockets.exceptions import WebSocketException

from . import session
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
from .webcam import CAPTURE_TIMEOUT_S, WebcamError, capture_image_async


def resolve_half_duplex(settings: Settings, output_name: str) -> tuple[bool, str]:
    """Decide whether to mute the mic while the assistant talks, and say why."""
    if settings.half_duplex is not None:
        forced = "on" if settings.half_duplex else "off"
        return settings.half_duplex, f"CYCLOPS_HALF_DUPLEX={forced} (output: {output_name})"
    if output_is_speaker(output_name):
        return True, f"output is '{output_name}', which the mic would hear"
    return False, f"output is '{output_name}'"


def watch_stdin(
    loop: asyncio.AbstractEventLoop, on_enter: Callable[[], None]
) -> Callable[[], None]:
    """Call ``on_enter`` when Enter is pressed. Does nothing at all when stdin is not a terminal.

    The terminal is the CLI's shutter button - it has no panel to tap. The isatty gate is what
    makes that safe: under systemd or a pipe, stdin is at EOF and therefore *always* readable,
    so an unguarded reader would spin a core forever. Returns the teardown.

    fd 0 is deliberately left in whatever mode the shell handed over: a TTY in canonical mode
    only becomes readable once a whole line is buffered, so the read never blocks, and the
    process does not leave a non-blocking stdin behind for the shell to trip over.
    """
    try:
        if not sys.stdin.isatty():
            return lambda: None
        fd = sys.stdin.fileno()
    except (ValueError, OSError):
        return lambda: None

    def readable() -> None:
        try:
            data = os.read(fd, 4096)
        except OSError:
            data = b""
        if not data:  # EOF - stop watching rather than spin on a fd that is readable forever
            loop.remove_reader(fd)
            return
        if b"\n" in data or b"\r" in data:
            on_enter()

    loop.add_reader(fd, readable)
    return lambda: loop.remove_reader(fd)


async def snap(agent: VoiceAgent, settings: Settings) -> None:
    """Take a photo and show it to the model - the CLI's version of tapping SNAP.

    No live source is registered here, so this is the slow open-warm-shoot path. It runs on a
    thread of its own (see :func:`cyclops.webcam.capture_image_async`) so the mic keeps
    streaming while the camera wakes up.
    """
    save_dir, keep_as = session.photo_target(settings, by="you")
    try:
        async with asyncio.timeout(CAPTURE_TIMEOUT_S):
            shot = await capture_image_async(
                settings.camera_index, save_dir=save_dir, keep_as=keep_as
            )
    except TimeoutError:
        print(
            f"· snapshot failed: camera did not answer within {CAPTURE_TIMEOUT_S:.0f}s",
            file=sys.stderr,
            flush=True,
        )
        return
    except WebcamError as exc:
        print(f"· snapshot failed: {exc}", file=sys.stderr, flush=True)
        return
    print(f"· snapped {shot.path}", flush=True)
    await agent.add_photo(shot)
    session.note(
        "photo",
        by="you",
        file=f"{session.PHOTOS}/{shot.path.name}",
        width=shot.width,
        height=shot.height,
        bytes=shot.jpeg_bytes,
        shown=True,
    )


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

    # Enter is the shutter. One at a time, and not before the session is up, so a held-down key
    # cannot queue a burst of captures that all serialise behind the camera lock.
    snapping: set[asyncio.Task] = set()

    def on_enter() -> None:
        if snapping or not agent.ready.is_set() or not agent.connected:
            return
        task = asyncio.ensure_future(snap(agent, settings))
        snapping.add(task)
        task.add_done_callback(snapping.discard)

    unwatch = watch_stdin(loop, on_enter)
    if sys.stdin.isatty():
        print("· press ENTER to take a photo and show it to Cyclops", flush=True)
    else:
        print("· stdin is not a terminal, so there is no shutter - this session cannot see",
              flush=True)

    # No camera is held open here, so no frames and no video.mp4 - the folder gets its transcript
    # and its photos and nothing else. That is by construction, not by a check.
    with SessionLog(settings, agent, entrypoint="cli", mic=mic, speaker=speaker):
        try:
            await agent.run()
        finally:
            unwatch()
            for task in snapping:
                task.cancel()
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
