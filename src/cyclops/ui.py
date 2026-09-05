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

from . import barge, sfx, voice
from .agent import VoiceAgent
from .audio import (
    SAMPLE_RATE,
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
from .webcam import Capture

IDLE, CONNECTING, LISTENING, SPEAKING, LOOKING, SEARCHING, DRAWING, ERROR = (
    "idle",
    "connecting",
    "listening",
    "speaking",
    "looking",
    "searching",
    "drawing",
    "error",
)
LEVEL_FULL_SCALE = 3000.0  # int16 RMS that maps to a full meter


class SessionController:
    """Starts/stops a VoiceAgent on its own thread and reports a thread-safe status snapshot.

    ``frames`` is what the session's video is a recording of, and only the kiosk has one: either
    the camera it is already holding open for the preview, or the panel that preview ends up on
    (see :mod:`cyclops.filming`). It is settable, because which of the two is a switch on the
    settings screen - see :meth:`set_record_source`. Every session is logged either way;
    ``entrypoint`` is what goes in the log, and it is passed rather than inferred from ``frames``
    because that would only be right by accident.
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
        self._guard: EchoGuard | None = None
        self._error = ""
        # What the session is doing when there is no agent to ask - which is both ends of it:
        # the audio devices going up before one exists, and the whole teardown after its socket
        # has gone. A plain string, written from this thread and from the kiosk's, read from the
        # kiosk's: one rebinding of one name, which is the same bargain the flags on the agent
        # have always made. See :meth:`_detail`.
        self._phase = ""
        self._closing = False  # a teardown is under way; see the property below
        self._started_at: float | None = None  # monotonic, for the kiosk's session timer
        # This controller is the only thing that knows when a session is *finished* rather than
        # merely cancelled - the socket, the audio devices and the recording all outlive the
        # agent - so the sound that says so is sounded from here.
        self._cues = sfx.Cues(
            rate=SAMPLE_RATE,
            device=resolve_device(settings.output_device),
            enabled=settings.sounds,
        )

    @property
    def _running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._error = ""
            self._phase = "waking up…"  # said on the tap; the thread does not exist yet
            self._closing = False
            self._started_at = time.monotonic()  # the clock starts on the tap, not on connect
            self._thread = threading.Thread(target=self._run, name="cyclops-session", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            loop, task = self._loop, self._task
        if loop is None or task is None:
            return
        # On the tap, like the closing cue in the kiosk, and for a reason beyond symmetry: for
        # the couple of seconds it takes the task to notice it was cancelled this controller
        # still honestly reports LISTENING, so without a phrase here the caption would sit
        # saying "listening - talk to me" underneath a strip that already says SLEEPING.
        self._phase = "going to sleep…"
        self._closing = True
        try:
            loop.call_soon_threadsafe(task.cancel)
        except RuntimeError:
            pass  # it finished on its own between the read and the call; nothing left to cancel

    def set_barge_in(self, margin_db: float | None) -> None:
        """Let the session that is running now be interrupted, or not - see :mod:`cyclops.barge`.

        Called from the kiosk's thread, which is where the settings screen's note is noticed.
        Nothing to dispatch to the loop: the guard is read on the audio thread and this is one
        store, so a switch flipped while Cyclops is mid-sentence lands on the next 20 ms block
        rather than at the end of the turn. No session means nothing to tell - the next one
        reads the same note when it opens its devices.
        """
        with self._lock:
            guard = self._guard
        if guard is not None:
            guard.set_barge_in(margin_db)

    def set_record_source(self, frames: FrameSource | None) -> None:
        """Say what the *next* session's video should be of - see :mod:`cyclops.filming`.

        Unlike :meth:`set_barge_in` this deliberately does not reach into a running session, and
        could not usefully: the encoder is opened once, at the frame size its first frame had, so
        a source swapped mid-recording would at best be stretched to the shape of the other one.
        The kiosk calls this on the tap that starts a session, which is the moment the answer is
        needed and the last moment it can still be changed for free.

        One store of one name, from the kiosk's thread, read by the session thread a beat later
        in :meth:`_session` - the same bargain the rest of this class makes.
        """
        self._frames = frames

    def show_photo(self, capture: Capture) -> bool:
        """Hand a photo to the running agent. False when there is nothing live to hand it to.

        Called from the kiosk's snap thread and from nothing that can afford to wait, so it
        dispatches and returns - the same shape as :meth:`stop`, and the same shape the audio
        callbacks use to reach this loop. All it can honestly report is that it handed the photo
        to a live, connected session; whether the model got it is answered on the other side.
        """
        with self._lock:
            loop, agent = self._loop, self._agent
        if loop is None or agent is None or loop.is_closed() or not loop.is_running():
            return False
        if not agent.ready.is_set() or not agent.connected:
            return False
        try:
            loop.call_soon_threadsafe(agent.queue_photo, capture)
        except RuntimeError:
            return False  # the loop closed in the moment between the guard and the call
        return True

    def join(self, timeout: float) -> None:
        """Wait for a stopping session to finish - it may still be muxing its recording."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def status(self) -> dict[str, object]:
        state, level = self._state_and_level()
        started = self._started_at
        return {
            "state": state,
            "level": round(level, 3),
            "detail": self._detail(state),
            "elapsed": None if started is None else round(time.monotonic() - started, 1),
        }

    @property
    def closing(self) -> bool:
        """Is a teardown under way - the stretch between the tap that stops it and the last file?

        Asked by the kiosk, which shows an optimistic CLOSING on the tap and needs to know how
        long to keep believing it. This controller goes on honestly reporting LISTENING the
        whole time, because the state machine describes the *agent* and the agent is the first
        thing to go; the mux that follows it is not a state, it is a queue. (The naming and the
        summarising used to be in that queue too. They are in another process now - see
        :mod:`cyclops.after` - which is most of why this stretch is short.)
        """
        return self._closing

    def _say_phase(self, phase: str) -> None:
        """What the teardown is doing, handed to :class:`~cyclops.session.SessionLog`.

        It runs inside :meth:`_session`, so the agent is still published while it works - but
        the agent's socket is already down and it has nothing left to narrate, which is exactly
        why the last seconds of a session need a channel that is not the agent.
        """
        self._phase = phase

    def _detail(self, state: str) -> str:
        """The live line for the bottom of the panel: what is going on, when we know.

        Empty is a real answer and the commonest one. It means nothing is happening beyond the
        state itself, and the overlay's own resting sentence for that state is the better thing
        to say - which is why this used to be a sentence per state here and is not any more.
        Those were the same nine strings the overlay already had, and one of them ("Searching
        the web…") a vaguer version of what the agent can now name outright.
        """
        if state == ERROR:
            return self._error or "something went wrong"
        agent = self._agent  # snapshot: _run()'s finally clears it, from the session thread
        return (agent.activity if agent is not None else "") or self._phase

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
        if agent.drawing_active:
            return DRAWING, 0.0
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
            # Cleared before the close, not after: anything holding this controller reads these
            # under the lock, and a loop that is closed but still published is one they will
            # call into and get RuntimeError from.
            with self._lock:
                self._loop = self._task = self._agent = self._speaker = self._mic = None
                self._guard = None
                self._started_at = None  # the timer disappears with the session
            self._phase = ""  # the folder is written; there is nothing left to report
            self._closing = False
            loop.close()
            # Now it is over: the folder is written and the panel is about to go back to
            # ASLEEP. This cuts off the closing ticks the kiosk started on the tap, however
            # long or short the teardown turned out to be.
            self._cues.play("ended")

    async def _session(self, loop: asyncio.AbstractEventLoop) -> None:
        self._phase = "opening the audio devices…"
        s = self.settings
        in_dev = resolve_device(s.input_device)
        out_dev = resolve_device(s.output_device)
        half = s.half_duplex
        if half is None:
            half = output_is_speaker(default_output_name(out_dev))
        speaker = Speaker(device=out_dev)
        speaker.volume = s.volume
        # A guard for every session, headphones included: with barge-in switched off it is the
        # thing that holds the mic shut until Cyclops has finished, and that switch is on the
        # settings screen, which can be reached in the middle of a session. See cyclops.barge.
        guard = EchoGuard(loop, speaker, half_duplex=half, margin_db=barge.margin_db(s))
        mic = Microphone(
            loop, guard=guard, device=in_dev, gain_db=s.mic_gain_db, compress=s.mic_compress
        )
        source = mic.source or "whatever PipeWire calls the default"
        print(f"· mic: {source}{mic.levelling}", flush=True)
        # What this session actually runs as, which is not quite what .env says: half-duplex may
        # have been worked out from the output device just now, and the voice may have been
        # chosen on the panel since the kiosk started. Built once and given to both the agent and
        # the log, so the folder records the voice you were answered in rather than the one the
        # variable holds - see cyclops.voice.
        live = replace(s, half_duplex=half, voice=voice.chosen(s))
        agent = VoiceAgent(live, mic=mic, speaker=speaker, guard=guard)
        guard.on_barge_in = agent.local_barge_in
        with self._lock:
            self._speaker, self._mic, self._agent = speaker, mic, agent
            self._guard = guard
        self._phase = ""  # there is an agent now, and it narrates itself from here
        speaker.start()
        mic.start()
        # The log is the one thing every entry point shares, so it does its own wiring: it hooks
        # the agent's events, starts the recorder when there is a camera, and finishes the folder
        # on the way out - including when the session dies rather than stops. The inner `finally`
        # still runs first, so the recorder is stopped only once the audio callbacks have ceased.
        with SessionLog(
            live,
            agent,
            entrypoint=self._entrypoint,
            mic=mic,
            speaker=speaker,
            frames=self._frames,
            on_phase=self._say_phase,
        ):
            try:
                await agent.run()
            finally:
                mic.stop()  # no more audio callbacks, so the recorder can flush what it has
                speaker.stop()
