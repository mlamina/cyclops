"""The second port: the room as the camera sees it, and the voice as the speaker plays it.

A phone or an iPad propped against the bench already gets the conversation as text (the LIVE
screen, ``admin/static/app.js``). This is the other two senses - the live picture, and the
answer out loud - handed to that same page over a port of the kiosk's own.

It owns exactly that, and it must never: touch the panel, do I/O on the audio thread, hear the
**microphone** (only the agent's own output passes through here), or cost anything at all while
nobody is connected. The last one is the gate, and it is the reason this lives in the kiosk
rather than in a service of its own: *the TCP connection is the subscription*. With the live
screen shut there are no viewers, so the encoder thread does not exist and the voice tap is a
load and a test per 20 ms block. Nothing is polled, nothing is published on the off chance, and
there is no note another process could leave armed for a kiosk that has restarted under it.

Not on ``cyclops-admin``, deliberately: that is two gunicorn workers of two threads, and one
viewer holding an endless response for the picture and another for the voice is half the server
the panel itself polls four hundred milliseconds at a time. See the note at ``app.js:603``.

Nothing here is load-bearing. A port that will not bind, a client that vanishes mid-frame, an
encode that raises - each costs a stream and nothing else. The panel comes up regardless.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

from .audio import BYTES_PER_FRAME, SAMPLE_RATE
from .camera import STALE_AFTER_S
from .config import COMPANION_PORT
from .overlay import message

BIND = "0.0.0.0"  # the same reach the admin page has: a phone on the LAN, and nothing wider
BOUNDARY = "cyclopsframe"
PREVIEW_FPS = 12  # the panel draws at 30 and the camera delivers 15; a phone wants neither
PREVIEW_WIDTH = 640  # exactly half a 720p frame, which is cv2.resize's fast path
PREVIEW_QUALITY = 55  # 1.51 ms a frame at this size on this box, measured 2026-09-06
PREVIEW_SIZE = (640, 360)  # what a card saying "no camera" is drawn at, when there is no frame
STILL_FPS = 1  # ...and how often that card is re-sent: it is a caption, not a picture
# Both messages are the panel's own words (kiosk.py), repeated rather than imported because
# importing them would mean importing the kiosk - which owns OpenCV's Qt window - into a module
# the CLI loads. Two short strings against that, and a test pins them to each other.
NO_CAMERA = "No camera found"
CAMERA_STALLED = "Camera stopped responding"
PACE_S = 0.05  # how often a listener's socket is written: smaller chunks are a smoother arrival
VOICE_LAG_MAX = SAMPLE_RATE * BYTES_PER_FRAME  # a second; past this a slow reader is skipped on
MAX_VIEWERS = 4  # per stream. A page left open in ten tabs is not a reason to cook the Pi
SEND_TIMEOUT_S = 10.0  # a phone that walks out of range is dropped, not left holding a thread
# How long a device's claim on his voice stands without being renewed, and how often the page
# renews it. An open socket is *not* the claim - see listening(), which is where the reasoning is.
LISTEN_FRESH_S = 8.0
LISTEN_BEAT_S = 3.0  # the page's interval; two may be missed before the panel takes his voice back


# ------------------------------------------------------------------ the voice


class _Voice:
    """The agent's voice, fanned out to whoever is listening - and to nobody, cheaply.

    The tap runs on PortAudio's real-time thread, so it does what :mod:`cyclops.record` does
    with the same blocks: append under a lock and let somebody else do the writing. Each
    listener owns a buffer rather than sharing a queue, because they are at different points in
    the same stream and a slow one must not hold up a fast one.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sinks: list[bytearray] = []

    @property
    def listeners(self) -> int:
        return len(self._sinks)

    def join(self) -> bytearray:
        sink = bytearray()
        with self._lock:
            self._sinks = [*self._sinks, sink]  # replaced, never mutated: see on_block
        return sink

    def leave(self, sink: bytearray) -> None:
        with self._lock:
            self._sinks = [s for s in self._sinks if s is not sink]

    def on_block(self, block: bytes) -> None:
        """The Speaker's second tap. Real-time thread: no I/O, no allocation storms, never raises.

        The empty check is deliberately outside the lock and reads a list that is only ever
        *replaced*, so with nobody listening - which is almost always - this costs one load and
        one test, and never waits on a lock held by a socket.
        """
        if not self._sinks:
            return
        with self._lock:
            for sink in self._sinks:
                sink.extend(block)
                if len(sink) > VOICE_LAG_MAX:
                    del sink[:-VOICE_LAG_MAX]  # the reader fell behind: stay live, lose the past

    def drain(self, sink: bytearray) -> bytes:
        """Whatever has arrived since the last call, empty when he has not said anything."""
        with self._lock:
            if not sink:
                return b""
            out = bytes(sink)
            del sink[:]
        return out


# ------------------------------------------------------------------ the picture


def _reason(got, now: float, connected: bool) -> str:
    """Empty when the newest frame is worth sending, else the line to send instead.

    The decision, split from the acting, because it is the whole of what a test can ask. The
    rule and both words are the panel's (``kiosk.py:1806-1810``): a camera that is enumerated
    but silent is a different fault from one that is not there, and a viewer deserves the same
    distinction the person standing at the box gets.
    """
    if got is not None and now - got[1] <= STALE_AFTER_S:
        return ""
    return CAMERA_STALLED if connected else NO_CAMERA


class _Preview:
    """The camera as JPEG, encoded once however many are watching.

    One producer thread, one slot, a serial. A viewer waits for a serial it has not seen; a slow
    one simply misses frames, which is what you want from video and what a queue would get
    wrong. The thread exists only while somebody is watching.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._jpeg = b""
        self._serial = 0
        self._watchers = 0
        self._camera: object | None = None
        self._thread: threading.Thread | None = None

    def bind(self, camera: object) -> None:
        self._camera = camera

    @property
    def watchers(self) -> int:
        return self._watchers

    def join(self) -> None:
        with self._cond:
            self._watchers += 1
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run, name="kiosk-preview", daemon=True
                )
                self._thread.start()

    def leave(self) -> None:
        with self._cond:
            self._watchers = max(0, self._watchers - 1)
            self._cond.notify_all()  # so the producer notices it has nobody left

    def latest(self, after: int, timeout: float) -> tuple[bytes, int] | None:
        """The first frame with a serial past ``after``, or None if none came in time.

        The loop rather than one ``wait``: a Condition may return early, and the first viewer
        through the door is waiting on a producer thread that has not encoded anything yet.
        """
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._serial <= after or not self._jpeg:
                left = deadline - time.monotonic()
                if left <= 0:
                    return None
                self._cond.wait(left)
            return self._jpeg, self._serial

    def _publish(self, jpeg: bytes) -> None:
        with self._cond:
            self._jpeg = jpeg
            self._serial += 1
            self._cond.notify_all()

    def _run(self) -> None:
        """Encode while anybody is watching. Never raises: a dead stream beats a dead panel."""
        # Per-thread, which is what nice is on Linux: the render loop and the realtime audio
        # thread both outrank this, so a busy box drops preview frames rather than dropping his
        # voice or stuttering the glass.
        with contextlib.suppress(OSError, AttributeError):
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 10)
        sent = 0.0  # the camera stamp of the last real frame we encoded
        said = ""  # the words on the last card, so it is not re-drawn every tick
        card_at = 0.0
        while self._watchers > 0:
            began = time.monotonic()
            try:
                got = self._camera.latest() if self._camera is not None else None
                why = _reason(got, began, bool(getattr(self._camera, "connected", False)))
                if why:
                    # A card, and only when the words change or a second has passed. An <img>
                    # holds the last part it was sent for ever, so saying nothing at all would
                    # leave a picture of a room nobody is watching any more.
                    if why != said or began - card_at > 1.0 / STILL_FPS:
                        said, card_at = why, began
                        self._publish(_encode(message(*PREVIEW_SIZE, why)))
                elif got[1] != sent:  # a frame we have not already sent
                    said, sent = "", got[1]
                    self._publish(_encode(_shrink(got[0])))
            except Exception as exc:  # noqa: BLE001 - one bad frame must not end the stream
                print(f"· companion preview: {exc}", flush=True)
            time.sleep(max(0.0, 1.0 / PREVIEW_FPS - (time.monotonic() - began)))


def _shrink(frame):
    height, width = frame.shape[:2]
    if width <= PREVIEW_WIDTH:
        return frame
    size = (PREVIEW_WIDTH, max(1, round(height * PREVIEW_WIDTH / width)))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def _encode(frame) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_QUALITY])
    return buf.tobytes() if ok else b""


voice = _Voice()
preview = _Preview()
_heard_at = 0.0  # when a device last said it is still being used as the speaker; see listening()


def watching() -> bool:
    """Whether anybody is on the other end of either stream - see ``Kiosk._sleeping``."""
    return preview.watchers > 0 or voice.listeners > 0


def listening() -> bool:
    """Whether a device is being used as the speaker *right now* - see ``Kiosk._sync_handover``.

    A heartbeat and deliberately not ``voice.listeners``, which is the obvious answer and is
    wrong. An open socket is not somebody listening: a browser that died, a phone that slept, a
    curl left running in another room and a connection whose peer has gone without saying so all
    hold one open, and every one of them silenced the panel until something noticed - which for a
    half-open TCP connection is a minute of send buffer, and for a live-but-idle one is never.
    That is a bad way round for a fault to fall. The panel is the speaker somebody is standing
    at, so it keeps his voice unless a device is actively still asking to have it.

    So the claim has to be renewed (``GET /listening``, every ``LISTEN_BEAT_S`` from the page
    while its switch is on) and it lapses on its own. Same shape as every other note between the
    two halves of this box, and the same reason: staleness is the only liveness test that a
    process dying cannot lie about.
    """
    return time.monotonic() - _heard_at < LISTEN_FRESH_S


def _silence(seconds: float) -> bytes:
    """Silence for exactly the time that has passed, and that *exactly* is the whole point.

    A fixed block per loop was the first version and it starved the page: one iteration is a
    sleep plus a write, so it takes a shade longer than the sleep asks for, and a block sized to
    the sleep therefore delivers a shade less than real time, every time, forever. The listener's
    buffer drains at the difference and the sound breaks up. Sized to the clock instead, the
    stream cannot drift no matter what the loop costs. Audio drained from the speaker needs no
    such help - it arrived on the sound card's own clock and is already true.
    """
    return bytes(max(0, int(seconds * SAMPLE_RATE)) * BYTES_PER_FRAME)


def _keepalive(sock) -> None:
    """Notice a peer that has gone, without first filling a socket buffer at 48 KB/s.

    Linux-only options, suppressed elsewhere: this runs on a Mac under ``uv run cyclops`` too,
    where there is no panel to take a voice from and nothing to protect.
    """
    with contextlib.suppress(OSError, AttributeError):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 5)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 3)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 2)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_USER_TIMEOUT, 10_000)


# ------------------------------------------------------------------ the port


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"  # an endless body has no length to declare; close is honest

    def log_message(self, *_args) -> None:
        """Nothing. A phone locking its screen is not journal-worthy."""

    def _open(
        self,
        kind: str,
        status: int = 200,
        length: int | None = None,
        extra: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        if length is not None:
            self.send_header("Content-Length", str(length))
        for name, value in (extra or {}).items():
            self.send_header(name, value)
        # The page is served from port 80 and this is 8081, so to a browser they are different
        # origins. <img> does not care; the fetch() behind the voice does, and a header on some
        # answers and not others is the kind of bug that only shows up on the iPad.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _refuse(self, status: int, why: str) -> None:
        body = (why + "\n").encode()
        self._open("text/plain; charset=utf-8", status, len(body))
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's naming
        self._open("text/plain", 204)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's naming
        route = self.path.split("?", 1)[0]
        try:
            if route == "/":
                self._state()
            elif route == "/camera.jpg":
                self._still()
            elif route == "/camera.mjpg":
                self._stream()
            elif route == "/voice.pcm":
                self._voice()
            elif route == "/listening":
                self._still_listening()
            else:
                self._refuse(404, "try /, /camera.jpg, /camera.mjpg, /voice.pcm or /listening")
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            pass  # somebody closed a tab, or wifi went. Not news.
        except Exception as exc:  # noqa: BLE001 - one bad request must not end the server
            print(f"· companion {route}: {exc}", flush=True)

    def _state(self) -> None:
        """What the page draws a state from - a stream's status line cannot carry this."""
        camera = preview._camera  # noqa: SLF001 - same module; the attribute is the binding
        body = json.dumps(
            {
                "camera": bool(getattr(camera, "connected", False)),
                "watching": {"camera": preview.watchers, "voice": voice.listeners},
                "preview": {"width": PREVIEW_WIDTH, "fps": PREVIEW_FPS},
                "voice": {"rate": SAMPLE_RATE, "channels": 1, "format": "s16le"},
            }
        ).encode()
        self._open("application/json", 200, len(body))
        self.wfile.write(body)

    def _still_listening(self) -> None:
        """A device renewing its claim on his voice. Cheap on purpose: it is asked every 3 s."""
        global _heard_at
        _heard_at = time.monotonic()
        self._open("text/plain", 204)

    def _still(self) -> None:
        """One frame. The insurance policy: every browser can show this, whatever it makes of
        multipart, and a page that polls it at two a second is warmer but never blank."""
        preview.join()
        try:
            got = preview.latest(-1, 2.0)
        finally:
            preview.leave()
        if got is None:
            self._refuse(503, "no frame")
            return
        self._open("image/jpeg", 200, len(got[0]))
        self.wfile.write(got[0])

    def _stream(self) -> None:
        if preview.watchers >= MAX_VIEWERS:
            self._refuse(503, "too many are already watching")
            return
        self.connection.settimeout(SEND_TIMEOUT_S)
        _keepalive(self.connection)
        preview.join()
        first = preview.watchers == 1
        if first:
            print("· companion is watching", flush=True)
        try:
            self._open(f"multipart/x-mixed-replace; boundary={BOUNDARY}")
            seen = -1
            while True:
                got = preview.latest(seen, 5.0)
                if got is None:
                    continue  # nothing new to say; the socket stays open and we ask again
                jpeg, seen = got
                self.wfile.write(
                    f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                )
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        finally:
            preview.leave()
            if preview.watchers == 0:
                print("· companion stopped watching", flush=True)

    def _voice(self) -> None:
        if voice.listeners >= MAX_VIEWERS:
            self._refuse(503, "too many are already listening")
            return
        self.connection.settimeout(SEND_TIMEOUT_S)
        _keepalive(self.connection)
        # Opening the stream is deliberately *not* a claim on his voice, though it was for an
        # afternoon. The page beats the moment its switch goes on, so there is no gap to cover -
        # and leaving the open here would put back exactly the hole the beat was written to close:
        # anything at all that connects would take the sound off the panel, for eight seconds a
        # go, forever, if it kept reconnecting. One route says "I am the speaker", and it is the
        # only one that does.
        sink = voice.join()
        print("· companion opened the voice stream", flush=True)
        try:
            # Self-describing rather than audio/L16, which is registered *big*-endian: claiming
            # it for native little-endian samples would be a lie a decoder could act on.
            self._open(
                "application/octet-stream",
                extra={
                    "X-Sample-Rate": str(SAMPLE_RATE),
                    "X-Channels": "1",
                    "X-Sample-Format": "s16le",
                },
            )
            sent_at = time.monotonic()
            while True:
                time.sleep(PACE_S)
                now = time.monotonic()
                block = voice.drain(sink) or _silence(now - sent_at)
                sent_at = now
                self.wfile.write(block)
        finally:
            voice.leave(sink)
            print("· companion closed the voice stream", flush=True)


def serve(camera: object, port: int = COMPANION_PORT) -> None:
    """Open the port, if it opens. Called once, from the kiosk's ``main``."""
    preview.bind(camera)
    try:
        server = ThreadingHTTPServer((BIND, port), _Handler)
    except OSError as exc:  # a panel is worth more than a stream
        print(f"· no companion stream: port {port} ({exc})", flush=True)
        return
    server.daemon_threads = True  # start-kiosk.sh SIGKILLs at ten seconds; join nobody
    threading.Thread(target=server.serve_forever, name="kiosk-stream", daemon=True).start()
    print(f"· companion stream on :{port}", flush=True)
