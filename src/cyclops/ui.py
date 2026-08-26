"""Touchscreen web UI: a tiny local server + a self-contained HTML page.

`cyclops-ui` serves a full-screen page (tap the eye to start/stop a session) plus status
icons, and drives a :class:`~cyclops.agent.VoiceAgent` on a background thread. It uses only
the standard library - the page is shown in a browser (Chromium kiosk on the Pi touchscreen).
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .agent import VoiceAgent
from .audio import (
    EchoGuard,
    Microphone,
    Speaker,
    default_output_name,
    output_is_speaker,
    resolve_device,
)
from .config import ConfigError, Settings, load_settings
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
UI_HTML = Path(__file__).with_name("ui.html")
EPOCH = str(int(time.time()))  # changes each server start; the page reloads itself when it does


class SessionController:
    """Starts/stops a VoiceAgent on its own thread and reports a thread-safe status snapshot.

    ``frames`` is an already-open camera to record the session from. Only the kiosk has one -
    it holds the device open for its preview - so ``cyclops-ui`` passes nothing and records
    nothing. Every session is logged either way; ``entrypoint`` is what goes in the log, and it
    is passed rather than inferred from ``frames`` because that would only be right by accident.
    """

    def __init__(
        self,
        settings: Settings,
        frames: FrameSource | None = None,
        *,
        entrypoint: str = "ui",
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
        self._volume = settings.volume
        self._error = ""
        self._started_at: float | None = None  # monotonic, for the kiosk's session timer

    def set_volume(self, value: float) -> None:
        value = max(0.0, min(1.0, value))
        with self._lock:
            self._volume = value
            speaker = self._speaker
        if speaker is not None:
            speaker.volume = value

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
            "volume": round(self._volume, 3),
            "elapsed": None if started is None else round(time.monotonic() - started, 1),
            "epoch": EPOCH,
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
        except Exception as exc:  # noqa: BLE001 - surface it to the UI, don't crash the server
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
        speaker.volume = self._volume
        guard = EchoGuard(loop, speaker, margin_db=s.barge_in_db) if half else None
        mic = Microphone(loop, guard=guard, device=in_dev)
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


class _Handler(BaseHTTPRequestHandler):
    controller: SessionController  # set on the class before serving

    def log_message(self, *args: object) -> None:  # silence per-request stderr logging
        pass

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, UI_HTML.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/status":
            body = json.dumps(self.controller.status()).encode()
            self._send(200, body, "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/start":
            self.controller.start()
        elif parsed.path == "/stop":
            self.controller.stop()
        elif parsed.path == "/volume":
            values = parse_qs(parsed.query).get("v")
            if values:
                try:
                    self.controller.set_volume(float(values[0]))
                except ValueError:
                    pass
        else:
            self._send(404, b"not found", "text/plain")
            return
        body = json.dumps(self.controller.status()).encode()
        self._send(200, body, "application/json")


def _launch_kiosk(url: str) -> None:
    for browser in ("chromium-browser", "chromium"):
        try:
            subprocess.Popen(
                [browser, "--kiosk", "--noerrdialogs", "--disable-infobars", url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"· launched {browser} in kiosk mode", flush=True)
            return
        except FileNotFoundError:
            continue
    print("· no chromium found; open the URL above in a browser", file=sys.stderr, flush=True)


def main() -> None:
    port = 8730
    kiosk = False
    for arg in sys.argv[1:]:
        if arg == "--kiosk":
            kiosk = True
        elif arg.startswith("--port="):
            port = int(arg.split("=", 1)[1])
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)

    _Handler.controller = SessionController(settings)
    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    url = f"http://localhost:{port}/"
    print(f"· cyclops UI on {url}  (Ctrl+C to quit)", flush=True)
    if kiosk:
        _launch_kiosk(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n· bye", flush=True)
    finally:
        _Handler.controller.stop()
        server.shutdown()


if __name__ == "__main__":
    main()
