"""The session controller: runs a :class:`~cyclops.agent.VoiceAgent` on a thread of its own.

The kiosk owns the screen, the camera and the main thread. Everything about a *session* -
opening the audio devices, running the agent, logging what happened, and tearing it all down
again - happens here instead, off that thread, and is reported back as one thread-safe status
snapshot per rendered frame.

This used to sit under a small web server that served a page to a browser in kiosk mode
(``cyclops-ui``). The OpenCV kiosk replaced it outright, so the server and its page are gone
and only the controller they were built around is left.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import replace

from .agent import VoiceAgent
from .audio import (
    EchoGuard,
    Microphone,
    Speaker,
    default_output_name,
    output_is_speaker,
    resolve_device,
)
from .config import Settings
from .record import FrameSource
from .session import SessionLog

IDLE, CONNECTING, LISTENING, SPEAKING, LOOKING, SEARCHING, ERROR = (
    "idle",
    "connecting",
    "listening",
    "speaking",
    "looking",
    "searching",
    "error",
)
LEVEL_FULL_SCALE = 3000.0  # int16 RMS that maps to a full meter


class SessionController:
    """Starts/stops a VoiceAgent on its own thread and reports a thread-safe status snapshot.

    ``frames`` is an already-open camera to record the session from - the kiosk's own, which it
    holds open for the preview. Every session is logged either way; ``entrypoint`` is what goes
    in the log, and it is passed rather than inferred from ``frames`` because that would only
    be right by accident.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        frames: FrameSource | None,
        entrypoint: str,
    ) -> None:
        self.settings = settings
        self._frames = frames
        self._entrypoint = entrypoint
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task | None = None
        self._agent: VoiceAgent | None = None
        self._speaker: Speaker | None = None
        self._mic: Microphone | None = None
        self._error = ""
        self._started_at: float | None = None  # monotonic, for the kiosk's session timer

    @property
    def _running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._error = ""
            self._started_at = time.monotonic()  # the clock starts on the tap, not on connect
            self._thread = threading.Thread(target=self._run, name="cyclops-session", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            loop, task = self._loop, self._task
        if loop is not None and task is not None:
            loop.call_soon_threadsafe(task.cancel)

    def join(self, timeout: float) -> None:
        """Wait for a stopping session to finish - it may still be muxing its recording."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def status(self) -> dict[str, object]:
        state, level = self._state_and_level()
        detail = {
            IDLE: "Tap to start",
            CONNECTING: "Connecting…",
            LISTENING: "Listening — talk to me",
            SPEAKING: "Speaking…",
            LOOKING: "Looking…",
            SEARCHING: "Searching the web…",
            ERROR: self._error or "Something went wrong",
        }[state]
        started = self._started_at
        return {
            "state": state,
            "level": round(level, 3),
            "detail": detail,
            "elapsed": None if started is None else round(time.monotonic() - started, 1),
        }

    def _state_and_level(self) -> tuple[str, float]:
        if not self._running:
            return (ERROR if self._error else IDLE), 0.0
        agent, speaker, mic = self._agent, self._speaker, self._mic
        if agent is None or not agent.ready.is_set():
            return CONNECTING, 0.0
        if agent.tool_active:
            return LOOKING, 0.0
        if agent.search_active:
            return SEARCHING, 0.0
        if speaker is not None and speaker.is_audible:
            return SPEAKING, min(1.0, speaker.output_level() / LEVEL_FULL_SCALE)
        level = min(1.0, mic.level / LEVEL_FULL_SCALE) if mic is not None else 0.0
        return LISTENING, level

    # ---- background thread ----

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        task = loop.create_task(self._session(loop))
        with self._lock:
            self._loop, self._task = loop, task
        try:
            loop.run_until_complete(task)
        except asyncio.CancelledError:
            pass  # a normal stop
        except Exception as exc:  # noqa: BLE001 - surface it to the panel, don't crash the kiosk
            message = str(exc)
            if "invalid_api_key" in message:
                message = "OpenAI rejected the API key"
            self._error = message
        finally:
            loop.close()
            with self._lock:
                self._loop = self._task = self._agent = self._speaker = self._mic = None
                self._started_at = None  # the timer disappears with the session

    async def _session(self, loop: asyncio.AbstractEventLoop) -> None:
        s = self.settings
        in_dev = resolve_device(s.input_device)
        out_dev = resolve_device(s.output_device)
        half = s.half_duplex
        if half is None:
            half = output_is_speaker(default_output_name(out_dev))
        speaker = Speaker(device=out_dev)
        speaker.volume = s.volume
        guard = EchoGuard(loop, speaker, margin_db=s.barge_in_db) if half else None
        mic = Microphone(loop, guard=guard, device=in_dev)
        print(f"· mic: {mic.source or 'whatever PipeWire calls the default'}", flush=True)
        agent = VoiceAgent(replace(s, half_duplex=half), mic=mic, speaker=speaker, guard=guard)
        if guard is not None:
            guard.on_barge_in = agent.local_barge_in
        with self._lock:
            self._speaker, self._mic, self._agent = speaker, mic, agent
        speaker.start()
        mic.start()
        # The log is the one thing every entry point shares, so it does its own wiring: it hooks
        # the agent's events, starts the recorder when there is a camera, and finishes the folder
        # on the way out - including when the session dies rather than stops. The inner `finally`
        # still runs first, so the recorder is stopped only once the audio callbacks have ceased.
        with SessionLog(
            s, agent, entrypoint=self._entrypoint, mic=mic, speaker=speaker, frames=self._frames
        ):
            try:
                await agent.run()
            finally:
                mic.stop()  # no more audio callbacks, so the recorder can flush what it has
                speaker.stop()
