"""The realtime voice agent: OpenAI Realtime API over WebSocket, plus its webcam and web tools."""

from __future__ import annotations

import asyncio
import base64
import json
import re
import sys
import uuid
from collections.abc import Callable, Coroutine
from typing import Any

from openai import AsyncOpenAI
from openai.resources.realtime.realtime import AsyncRealtimeConnection
from openai.types.realtime import (
    ConversationItemParam,
    RealtimeConversationItemFunctionCall,
    RealtimeError,
    RealtimeFunctionToolParam,
    RealtimeResponse,
    RealtimeServerEvent,
    RealtimeSessionCreateRequestParam,
)

from . import session
from .audio import SAMPLE_RATE, EchoGuard, Microphone, Speaker
from .config import Settings
from .search import SearchError, search_web
from .webcam import WebcamError, capture_image_async

CAPTURE_TIMEOUT_S = 12.0
BARGE_IN_CONFIRM_S = 1.5  # server must report speech within this long of a local barge-in
TRANSCRIPTION_MODEL = "gpt-4o-mini-transcribe"
DEFAULT_REASONING_EFFORT = "low"  # OpenAI's recommendation for production voice agents
REASONING_MODEL = re.compile(r"^gpt-realtime-2(\.\d+)?(-mini)?$")  # not gpt-realtime-2025-08-28
MAX_FOCUS_CHARS = 200
MAX_QUERY_CHARS = 300
MAX_PROJECT_NAME_CHARS = 80
MAX_PROJECT_TAGLINE_CHARS = 300  # a little under store.MAX_TAGLINE_CHARS
MAX_PROJECT_NOTES_CHARS = 4000  # a project page, not a card's worth of them
SEARCH_TIMEOUT_S = 14.0  # above search.SEARCH_TIMEOUT_S, so its own message wins
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")  # keep \t and \n

WEBCAM_TOOL: RealtimeFunctionToolParam = {
    "type": "function",
    "name": "capture_webcam_image",
    "description": (
        "Take a photo with the user's webcam right now and look at it. Call this whenever the "
        "user shows you something, holds something up, asks what you can see, or refers to an "
        "object in front of the camera. The photo is added to the conversation as an image you "
        "can see, so you can describe it and talk about it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "focus": {
                "type": "string",
                "description": "Optional: what to look for in the photo, in the user's words.",
            }
        },
        "required": [],
        "additionalProperties": False,
    },
}

WEB_SEARCH_TOOL: RealtimeFunctionToolParam = {
    "type": "function",
    "name": "web_search",
    "description": (
        "Search the web for current or factual information you do not reliably know: specs, "
        "measurements, torque values, part compatibility, prices, instructions, news, or "
        "anything that may have changed recently. Use it when the user asks a question about "
        "the world that a photo alone cannot answer. Say a few words out loud first, because "
        "the search takes a few seconds."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for, as a specific question or phrase.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}

OPEN_PROJECT_TOOL: RealtimeFunctionToolParam = {
    "type": "function",
    "name": "open_project",
    "description": (
        "Read your notes on one of the projects you are keeping. You are told the project names "
        "at the start of every session but not what is in them, so call this whenever the user "
        "returns to one and you need the detail: what was decided, what the measurements were, "
        "and what was left unresolved last time."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The project, as the user calls it. Close is good enough.",
            }
        },
        "required": ["name"],
        "additionalProperties": False,
    },
}

TRACK_PROJECT_TOOL: RealtimeFunctionToolParam = {
    "type": "function",
    "name": "track_project",
    "description": (
        "Start keeping notes on something new. Creates a folder for it, and from then on every "
        "session about it is written into that folder automatically. ONLY call this after the "
        "user has agreed to it out loud - never on your own initiative, and never to correct or "
        "rename a project that already exists."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": (
                    "Two to five plain words, what the user would write on the folder tab: "
                    "'Lego Millennium Falcon', 'Kitchen Tap', 'Shed Roof'. No dates, no "
                    "punctuation, no the word 'project'."
                ),
            },
            "description": {
                "type": "string",
                "description": (
                    "One dense sentence saying what this project IS, naming the specifics that "
                    "tell it apart from anything else being tracked: what the thing is, what it "
                    "is built from or on, and what it is for. This is what a later session reads "
                    "when deciding whether new work belongs here, so be concrete. Good: "
                    "'A Raspberry Pi voice assistant with a camera that helps with workshop "
                    "projects and writes up each session.' Useless: 'A Pi project', 'the build'. "
                    "Never just restate the name."
                ),
            },
        },
        "required": ["name", "description"],
        "additionalProperties": False,
    },
}

# The static half of what the model is told. The other half - what the last few sessions were
# about - is read off the card at connect time by :func:`build_instructions`.
BASE_INSTRUCTIONS = """\
You are Cyclops: a one-eyed device that sits on the bench next to someone who is making or
fixing something. They switch you on, point you at the job, and switch you off when they are
done. The eye is their webcam, and you can look through it whenever it would help.

WHAT YOU ARE FOR
Help this project move forward - the thing on the bench today, and the one they come back to
next week. Look at what they hold up. Hold on to what was decided. Look up what neither of you
knows. Speak up when you can see a mistake coming. Be curious about the work itself: what it
is, how far along it is, where it is stuck. Every session is written down and kept, so what
gets worked out here is not lost.

HOW YOU TALK
- They set the agenda, always. Go where they go. Never steer them somewhere else, and never
  hand them a plan they did not ask for.
- One or two sentences. Their hands are busy and probably dirty; this is talk, not a document.
- Answer first. No preamble, no repeating back what they just said, no summarising yourself.
- Offer once. If you spot a risk, a better order to do things in, or something still
  unresolved, say it briefly and then let it go. Never raise the same unheeded point twice.
- Saying nothing is a real option. While they measure, count, cut or think, stay quiet.
- Curiosity is one good question, not more words. Ask only when the answer would change what
  you say next, and only one question at a time.
- Useful beats warm. A number, a caution, the next step - that is the help. Praise is not.
- Speak whatever language they speak.
- Never state a measurement, spec or part number as fact unless a photo or a search gave it to
  you. If you are going from memory, say so.
- Do not read out URLs, file paths, or JSON.

USING THE EYE
- Whenever they show you something, hold something up, say "look at this", or ask what you can
  see, call the capture_webcam_image tool. Do not guess what is in view - take a photo.
- Take a NEW photo only when they clearly ask you to look again or show you something new.
  Never take a photo in reply to a short, vague, or unclear utterance.
- After a photo arrives, say what you actually see, then answer what they asked about it. If
  the image is dark, blurry, or empty, say so ONCE and wait - do not keep taking photos of an
  empty or unclear scene.

LOOKING THINGS UP
- When they ask something factual you are not sure about - a spec, a size, a torque value,
  whether two parts fit together, what something costs, anything that may have changed
  recently - call the web_search tool instead of guessing. Say a few words first ("let me look
  that up") so they are not left in silence, because the search takes several seconds.
- Combine the two tools when it helps: look at the thing, then search for what you saw. If a
  search comes back empty or failed, say so plainly instead of inventing an answer.

THE PROJECTS YOU KEEP
- You keep notes on the things they are building. Whichever ones exist are listed further down;
  you are told their names but not what is in them.
- When they come back to one, call open_project to read what was decided, what the measurements
  were, and what was left unresolved. Do that before answering from memory about it.
- When they are plainly working on something that is NOT one of them, and it looks like a thing
  they will come back to, ask - once, in one short sentence - whether you should keep notes on
  it. "Want me to keep track of this one?" is the whole question.
- Only call track_project after they have clearly said yes. Never call it on your own, never to
  rename something, and never for a passing question or a one-off job.
- When you do call it, the description matters more than the name. Say what the thing actually
  is, concretely - what it is, what it is built on, what it is for - because that sentence is
  how you will recognise this same project in three weeks. If you do not know enough to write
  one, you do not know enough to start tracking it yet: ask them what it is first.
- Ask at most once a session, and never in the first moments or while they are mid-cut,
  mid-measurement or otherwise busy. If they say no, say nothing about it again - that is the
  same rule as offering once, and it applies here hardest.
- Everything else is automatic: every session about a tracked project is written into its folder
  on your memory card when you switch off. Do not offer to write things down, and do not read
  file paths out.

WHEN THEY CUT IN
- If they say "stop", "wait", "hold on", "never mind", "that's enough", or anything like that,
  stop immediately: do not take a photo, do not keep talking, just briefly acknowledge
  ("Okay." / "Sure.") and wait for them.
- If they talk while you are speaking, they are interrupting you. Stop and respond only to what
  they just said. Do NOT resume, repeat, or restart what you were saying - even if you had not
  finished - and do NOT describe the image again unless they explicitly ask.
- A short acknowledgment ("cool", "ok", "got it", "nice", "thanks", "mhm") means "I heard you,
  move on": reply with at most a few words, or simply keep listening. Never re-read or
  re-describe something you already said.
- If you cannot make out what they said, or it sounds like a stray word or noise, ask them to
  repeat - do NOT take another photo and do NOT guess.
"""

# Introduces the recap below it. Its job is to stop the two failure modes a memory invites:
# opening with a recital of last week, and quietly assuming today is a continuation of it.
RECAP_HEADER = """\
WHERE YOU LEFT OFF
These notes were written at the end of the last few sessions. Use them the way someone who was
there would: if they pick up where they left off, you already know where that is. Do not recite
them back, and do not assume today is about the same thing - wait and hear what they want. You
recall nothing beyond these notes; if they ask about something older, say so plainly.
"""


# Introduces the list below it. Names only - what is in each project is a tool call away, which
# is the whole point: a card with twenty projects on it would otherwise spend the model's opening
# context on nineteen it is not being asked about.
PROJECTS_HEADER = """\
PROJECTS YOU ARE KEEPING NOTES ON
Call open_project with one of these names to read what is in it.
"""
PROJECTS_LISTED = 20
PROJECT_LINE_CHARS = 380


def _projects_block(settings: Settings) -> str:
    """The names of what is being tracked, or nothing at all.

    Read at connect time off the card, like the recap above it, and for the same reason: a project
    started in the last session has to be there in this one.
    """
    if not settings.projects:
        return ""
    from . import projects  # local, so a box with no key and no card still builds instructions

    try:
        tracked = projects.catalog(settings)
    except OSError:
        return ""
    if not tracked:
        print("· projects: none being tracked yet", flush=True)
        return ""
    lines = []
    for project in tracked[:PROJECTS_LISTED]:
        line = f"- {project.name}"
        if project.tagline:
            line += f" — {project.tagline}"
        lines.append(line[:PROJECT_LINE_CHARS])
    print(f"· projects: {', '.join(p.name for p in tracked[:PROJECTS_LISTED])}", flush=True)
    return f"{PROJECTS_HEADER}\n" + "\n".join(lines) + "\n"


def build_instructions(settings: Settings) -> str:
    """The full system prompt for one session: the standing rules, plus what came before.

    Read at connect time rather than baked in at import, because the notes change every time a
    session ends - including the one that ended a minute ago. Costs a handful of small reads
    off the card, once per session.

    What was handed over is printed, not silent. A memory you cannot see is one you cannot
    trust: when Cyclops opens by knowing something, this line is where you check it was told.
    """
    recap = session.recent_context(settings)
    print(f"· continuity: {recap.note}", flush=True)
    blocks = [BASE_INSTRUCTIONS]
    if recap:
        blocks.append(f"{RECAP_HEADER}\n{recap.text}\n")
    if projects_block := _projects_block(settings):
        blocks.append(projects_block)
    return "\n".join(blocks)


class SessionError(RuntimeError):
    """The server rejected something during session setup (before ``session.updated``)."""

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


def function_calls(response: RealtimeResponse) -> list[RealtimeConversationItemFunctionCall]:
    return [item for item in (response.output or []) if item.type == "function_call"]


class VoiceAgent:
    """Owns one Realtime session: streams mic audio up, plays audio down, runs the webcam tool.

    Response sequencing: the server allows one active response, and with server VAD it
    creates responses on its own whenever the user stops talking. So after adding items we
    *want* a response; it is created when nothing is active and the user isn't speaking.
    The want is dropped as soon as a response starts after all our items were acknowledged,
    because that response already sees them.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        mic: Microphone | None = None,
        speaker: Speaker | None = None,
        guard: EchoGuard | None = None,
    ) -> None:
        self.settings = settings
        self.mic = mic
        self.speaker = speaker
        self.guard = guard
        self.tool_active = False  # True while the webcam tool is capturing (UI 'looking')
        self.search_active = False  # True while a web search is in flight (UI 'searching')
        self._turn_serial = 0  # bumped when the user speaks; lets a late search spot staleness
        self._barge_in_timer: asyncio.TimerHandle | None = None
        self.ready = asyncio.Event()  # set once the server accepted our session config
        self.on_event: Callable[[RealtimeServerEvent], None] | None = None  # observer hook
        self._conn: AsyncRealtimeConnection | None = None
        self._user_speaking = False
        self._response_active = False
        self._want_response = False
        self._create_event_id: str | None = None  # event_id of our last response.create
        self._unacked_item_ids: set[str] = set()
        self._current_item_id: str | None = None  # assistant item whose audio is being played
        self._dead_item_ids: set[str] = set()  # interrupted items: drop their in-flight output
        self._assistant_line_open = False
        self._background: set[asyncio.Task[None]] = set()

    @property
    def unacked_item_ids(self) -> frozenset[str]:
        return frozenset(self._unacked_item_ids)

    @property
    def interrupted_item_ids(self) -> frozenset[str]:
        """Assistant items a barge-in cut short.

        Their transcript is the whole turn the model meant to say - more than was ever spoken -
        so anything writing it down has to say so rather than quote it as heard. Covers both
        paths: the server-VAD one and the local :class:`~cyclops.audio.EchoGuard` one.
        """
        return frozenset(self._dead_item_ids)

    @property
    def conn(self) -> AsyncRealtimeConnection:
        assert self._conn is not None, "not connected"
        return self._conn

    # ---------------------------------------------------------------- session configuration

    def session_config(self) -> RealtimeSessionCreateRequestParam:
        transcription: dict[str, Any] = {"model": TRANSCRIPTION_MODEL}
        if self.settings.transcribe_lang:
            transcription["language"] = self.settings.transcribe_lang
        config: RealtimeSessionCreateRequestParam = {
            "type": "realtime",
            "instructions": build_instructions(self.settings),
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                    "transcription": transcription,
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 500,
                        "create_response": True,
                        "interrupt_response": True,
                    },
                    "noise_reduction": {"type": "near_field"},
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                    "voice": self.settings.voice,
                    "speed": 1.0,
                },
            },
            "tools": [WEBCAM_TOOL, WEB_SEARCH_TOOL, *_project_tools(self.settings)],
            "tool_choice": "auto",
        }
        effort = self.settings.reasoning_effort  # explicit setting always goes through
        if effort is None and REASONING_MODEL.match(self.settings.model):
            effort = DEFAULT_REASONING_EFFORT
        if effort:
            config["reasoning"] = {"effort": effort}  # type: ignore[typeddict-item]
        return config

    # ---------------------------------------------------------------- lifecycle

    async def run(self) -> None:
        client = AsyncOpenAI(api_key=self.settings.api_key)
        async with client.realtime.connect(model=self.settings.model) as conn:
            self._conn = conn
            await conn.session.update(session=self.session_config())
            try:
                async for event in conn:
                    await self._handle_event(event)
            finally:
                for task in self._background:
                    task.cancel()
                await asyncio.gather(*self._background, return_exceptions=True)
                self._conn = None

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()

    async def send_text(self, text: str) -> None:
        """Inject a typed user turn (used by the headless smoke test)."""
        await self._send_item(
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}
        )
        await self._request_response()

    # ---------------------------------------------------------------- audio up

    async def _pump_mic(self) -> None:
        assert self.mic is not None
        async for chunk in self.mic.chunks():
            encoded = base64.b64encode(chunk).decode("ascii")
            await self.conn.input_audio_buffer.append(audio=encoded)

    # ---------------------------------------------------------------- events down

    async def _handle_event(self, event: RealtimeServerEvent) -> None:
        match event.type:
            case "session.created":
                self._log(
                    f"connected · model={self.settings.model} · voice={self.settings.voice}"
                    + (" · half-duplex" if self.settings.half_duplex else "")
                )
            case "session.updated":
                self._on_session_ready()
            case "error":
                self._on_error(event.error)
            case "input_audio_buffer.speech_started":
                self._user_speaking = True
                self._turn_serial += 1  # any search still running is now answering an old question
                self._confirm_local_barge_in()
                await self._on_user_speech_started()
            case "input_audio_buffer.speech_stopped":
                self._user_speaking = False
            case "conversation.item.added" | "conversation.item.created":
                self._unacked_item_ids.discard(event.item.id)
            case "conversation.item.input_audio_transcription.completed":
                self._log(f"You: {event.transcript.strip()}")
            case "conversation.item.input_audio_transcription.failed":
                self._log(f"transcription failed: {event.error.message}", stream=sys.stderr)
            case "response.created":
                self._response_active = True
                if not self._unacked_item_ids:
                    self._want_response = False  # this response already sees our items
            case "response.output_audio.delta":
                self._play(event.item_id, event.delta)
            case "response.output_audio_transcript.delta":
                if event.item_id not in self._dead_item_ids:
                    self._print_assistant(event.delta)
            case "response.output_audio_transcript.done":
                self._end_assistant_line()
            case "response.done":
                await self._on_response_done(event.response)
        if self.on_event is not None:
            self.on_event(event)

    def _on_session_ready(self) -> None:
        if self.ready.is_set():
            return
        self.ready.set()
        if self.mic is not None:
            self.mic.drain()  # audio from before the session was configured is stale
            self._spawn(self._pump_mic())
            self._log("listening — talk, show the camera something, Ctrl+C to quit")

    def _on_error(self, err: RealtimeError) -> None:
        if err.code == "conversation_already_has_active_response":
            return  # response.created for the winner arrived first and settled _want_response
        if err.code == "response_cancel_not_active":
            return  # our barge-in cancel raced the response finishing on its own
        if err.event_id and err.event_id == self._create_event_id:
            self._response_active = False  # our response.create was rejected outright
            self._want_response = False
        self._log(f"error {err.type}/{err.code}: {err.message}", stream=sys.stderr)
        if not self.ready.is_set():
            raise SessionError(err.message, code=err.code)

    async def _on_user_speech_started(self) -> None:
        # The server cancels the in-progress response itself (interrupt_response=True). Locally:
        # stop playback, ignore late output for that item, and tell the server what was heard.
        item_id = self._current_item_id
        if item_id is None or self.speaker is None:
            return
        self._current_item_id = None
        self._dead_item_ids.add(item_id)
        self._end_assistant_line()
        if not self.speaker.has_unplayed_audio:
            return  # everything received was heard; nothing to truncate
        played_ms = self.speaker.flush()
        self._log(f"(interrupted after {played_ms} ms)")
        await self.conn.conversation.item.truncate(
            item_id=item_id, content_index=0, audio_end_ms=played_ms
        )

    def local_barge_in(self, played_ms: int, strength: float) -> None:
        """EchoGuard heard you over the speaker and cut playback; finish the job here.

        ``strength`` is how many times louder the mic was than the predicted echo.
        """
        item_id = self._current_item_id
        self._current_item_id = None
        self._end_assistant_line()
        self._log(f"(barge-in: {strength:.0f}x over the echo; interrupted after {played_ms} ms)")
        if item_id is not None:
            self._dead_item_ids.add(item_id)
            self._spawn(self._truncate_and_cancel(item_id, played_ms))
        if self._barge_in_timer is not None:
            self._barge_in_timer.cancel()
        self._barge_in_timer = asyncio.get_running_loop().call_later(
            BARGE_IN_CONFIRM_S, self._reject_local_barge_in
        )

    async def _truncate_and_cancel(self, item_id: str, played_ms: int) -> None:
        if played_ms > 0:
            await self.conn.conversation.item.truncate(
                item_id=item_id, content_index=0, audio_end_ms=played_ms
            )
        if self._response_active:
            await self.conn.response.cancel()

    def _confirm_local_barge_in(self) -> None:
        if self._barge_in_timer is None:
            return
        self._barge_in_timer.cancel()
        self._barge_in_timer = None
        if self.guard is not None:
            self.guard.confirm()

    def _reject_local_barge_in(self) -> None:
        self._barge_in_timer = None
        if self.guard is not None:
            self.guard.reject()
            self._log("(that barge-in was just echo; raising the bar)")

    def _play(self, item_id: str, delta_b64: str) -> None:
        if self.speaker is None or item_id in self._dead_item_ids:
            return
        if item_id != self._current_item_id:
            self._current_item_id = item_id
            self.speaker.begin_item()
        self.speaker.feed(base64.b64decode(delta_b64))

    async def _on_response_done(self, response: RealtimeResponse) -> None:
        self._response_active = False
        self._dead_item_ids.clear()  # all of a response's output precedes its response.done
        calls = function_calls(response)
        if response.status == "completed":
            for call in calls:
                self._spawn(self._run_tool(call))
        elif calls:
            self._log(f"skipped {len(calls)} tool call(s): response {response.status}")
        if response.status == "failed":
            detail = getattr(response.status_details, "error", None)
            self._log(f"response failed: {detail}", stream=sys.stderr)
        await self._maybe_create_response()

    async def _request_response(self) -> None:
        """Ask for a model response now, or as soon as the currently active one finishes."""
        self._want_response = True
        await self._maybe_create_response()

    async def _maybe_create_response(self) -> None:
        if self._want_response and not self._response_active and not self._user_speaking:
            self._response_active = True
            self._create_event_id = uuid.uuid4().hex
            await self.conn.response.create(event_id=self._create_event_id)

    async def _send_item(self, item: ConversationItemParam) -> None:
        item_id = uuid.uuid4().hex  # server caps item ids at 32 chars
        self._unacked_item_ids.add(item_id)
        await self.conn.conversation.item.create(item={"id": item_id, **item})  # type: ignore[misc]

    # ---------------------------------------------------------------- the tools

    async def _run_tool(self, call: RealtimeConversationItemFunctionCall) -> None:
        if call.name == "web_search":
            await self._run_web_search(call)
            return
        if call.name in {"open_project", "track_project"}:
            await self._run_project_tool(call)
            return
        if call.name != "capture_webcam_image":
            await self._send_tool_output(call.call_id, {"ok": False, "error": "unknown tool"})
            await self._request_response()
            return
        focus = _tool_focus(call.arguments)
        self._log(
            f"[tool] capture_webcam_image {focus!r}" if focus else "[tool] capture_webcam_image"
        )

        image_item: ConversationItemParam | None = None
        save_dir, keep_as = session.photo_target(self.settings, by="cyclops")
        self.tool_active = True
        try:
            async with asyncio.timeout(CAPTURE_TIMEOUT_S):
                capture = await capture_image_async(
                    self.settings.camera_index, save_dir=save_dir, keep_as=keep_as
                )
        except TimeoutError:
            output = {
                "ok": False,
                "error": f"camera did not answer within {CAPTURE_TIMEOUT_S:.0f}s "
                "(macOS may be showing a camera permission prompt)",
            }
        except WebcamError as exc:
            output = {"ok": False, "error": str(exc)}
        except Exception as exc:  # never leave the model waiting for a tool result
            output = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        else:
            output = {
                "ok": True,
                "width": capture.width,
                "height": capture.height,
                "note": "Photo captured; it follows as an image in the next user message.",
            }
            caption = "[Webcam photo captured by the capture_webcam_image tool just now"
            if focus:
                caption += f"; the tool was asked to focus on: {focus}"
            image_item = {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": caption + "]"},
                    {"type": "input_image", "image_url": capture.data_url, "detail": "auto"},
                ],
            }
            session.note(
                "photo",
                by="cyclops",
                file=f"{session.PHOTOS}/{capture.path.name}",
                width=capture.width,
                height=capture.height,
                bytes=capture.jpeg_bytes,
                focus=focus or None,  # logged here because this is the only place `focus` exists
            )
            self._log(
                f"[tool] photo {capture.width}x{capture.height}, "
                f"{capture.jpeg_bytes // 1024} KB → {capture.path}"
            )

        self.tool_active = False
        if not output["ok"]:
            self._log(f"[tool] failed: {output['error']}", stream=sys.stderr)
        await self._send_tool_output(call.call_id, output)
        if image_item is not None:
            await self._send_item(image_item)
        await self._request_response()

    async def _run_web_search(self, call: RealtimeConversationItemFunctionCall) -> None:
        """Bridge the Realtime session to the Responses API's hosted web_search tool.

        A search takes ten seconds or more, which is a long time in a conversation, so the user
        may well have moved on before it lands. The Realtime API has no way to withdraw a tool
        call once made - the model waits for its result - so a stale answer is reported as
        stale rather than dropped, and the instructions tell the model not to deliver it out of
        the blue. That is the lifecycle question OpenAI declined to answer for us.
        """
        query = _tool_query(call.arguments)
        self._log(f"[tool] web_search {query!r}")
        if not query:
            await self._send_tool_output(call.call_id, {"ok": False, "error": "empty query"})
            await self._request_response()
            return

        turn = self._turn_serial  # if this moves while we search, the answer arrived too late
        self.search_active = True
        try:
            async with asyncio.timeout(SEARCH_TIMEOUT_S):
                answer = await search_web(query, self.settings)
        except TimeoutError:
            output = {"ok": False, "error": f"the search timed out after {SEARCH_TIMEOUT_S:.0f}s"}
        except SearchError as exc:
            output = {"ok": False, "error": str(exc)}
        except Exception as exc:  # never leave the model waiting for a tool result
            output = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        else:
            stale = self._turn_serial != turn
            output = {"ok": True, "query": query, "result": answer, "stale": stale}
            if stale:
                output["note"] = (
                    "The user has spoken since this search started, so it may no longer be what "
                    "they want. Do not read it out unless it is still relevant to them."
                )
            session.note("search", query=query, chars=len(answer), stale=stale)
            self._log(f"[tool] search: {len(answer)} chars{' (stale)' if stale else ''}")
        finally:
            self.search_active = False

        if not output["ok"]:
            session.note("search", query=query, error=output["error"])
            self._log(f"[tool] search failed: {output['error']}", stream=sys.stderr)
        await self._send_tool_output(call.call_id, output)
        await self._request_response()

    async def _run_project_tool(self, call: RealtimeConversationItemFunctionCall) -> None:
        """Open a project's notes, or start keeping some. Both are reads and writes of the card.

        Unlike the web search there is no staleness to manage: this is a few kilobytes off local
        disk, back in milliseconds, so nobody has moved on by the time it lands. It still goes
        through a thread, because the card is an SD card and the event loop here is also carrying
        the audio.
        """
        name = _tool_name(call.arguments)
        description = _tool_description(call.arguments)
        self._log(f"[tool] {call.name} {name!r}")
        if not name:
            output: dict[str, Any] = {"ok": False, "error": "no project name given"}
        else:
            output = await asyncio.to_thread(self._project_call, call.name, name, description)
        if not output["ok"]:
            self._log(f"[tool] {call.name} failed: {output['error']}", stream=sys.stderr)
        await self._send_tool_output(call.call_id, output)
        await self._request_response()

    def _project_call(self, kind: str, name: str, description: str = "") -> dict[str, Any]:
        """The blocking half of the two project tools. Returns an answer, never raises."""
        from . import projects

        try:
            tracked = projects.catalog(self.settings)
            if kind == "open_project":
                project = projects.store.find(tracked, name)
                if project is None:
                    return {
                        "ok": False,
                        "error": f"nothing is being tracked called {name!r}",
                        "tracked": [p.name for p in tracked],
                    }
                session.note("project", action="opened", key=project.key, name=project.name)
                notes = projects.store.read_body(project)[:MAX_PROJECT_NOTES_CHARS]
                return {"ok": True, "project": project.name, "notes": notes}

            project = projects.create(self.settings, name, tagline=description)
            session.note("project", action="tracked", key=project.key, name=project.name)
            return {
                "ok": True,
                "project": project.name,
                "note": (
                    "Tracking it from now on. This session and every one after it about this "
                    "gets written into its folder automatically - tell them so, briefly, once."
                ),
            }
        except projects.Exists as exc:
            return {
                "ok": False,
                "error": f"that is already being tracked as {exc.project.name!r}",
                "note": "Say so and carry on. Do not create anything.",
            }
        except projects.Unfilable as exc:
            return {"ok": False, "error": str(exc)}
        except OSError as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        except Exception as exc:  # noqa: BLE001 - never leave the model waiting for a result
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    async def _send_tool_output(self, call_id: str, output: dict[str, Any]) -> None:
        await self._send_item(
            {"type": "function_call_output", "call_id": call_id, "output": json.dumps(output)}
        )

    def _spawn(self, coro: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: asyncio.Task[None]) -> None:
        self._background.discard(task)
        if not task.cancelled() and (exc := task.exception()) is not None:
            self._log(f"background task failed: {exc!r}", stream=sys.stderr)

    # ---------------------------------------------------------------- console

    def _print_assistant(self, delta: str) -> None:
        if not self._assistant_line_open:
            print("Cyclops: ", end="", flush=True)
            self._assistant_line_open = True
        print(_CONTROL_CHARS.sub("", delta), end="", flush=True)

    def _end_assistant_line(self) -> None:
        if self._assistant_line_open:
            print(flush=True)
            self._assistant_line_open = False

    def _log(self, message: str, *, stream=None) -> None:
        self._end_assistant_line()
        print(f"· {_CONTROL_CHARS.sub('', message)}", file=stream or sys.stdout, flush=True)


def _tool_string(arguments: str | None, key: str, limit: int) -> str:
    """One of the model's string arguments, defensively parsed and length-capped."""
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return ""
    if not isinstance(args, dict):
        return ""
    return str(args.get(key) or "")[:limit]


def _tool_focus(arguments: str | None) -> str:
    """The webcam tool's optional 'focus' argument."""
    return _tool_string(arguments, "focus", MAX_FOCUS_CHARS)


def _project_tools(settings: Settings) -> list[RealtimeFunctionToolParam]:
    """The two project tools, or neither. ``CYCLOPS_PROJECTS=0`` leaves them out entirely.

    Left out rather than offered and refused: a tool the model can see is a tool it will try, and
    being told "that is switched off" mid-conversation is worse than never being offered it.
    """
    return [OPEN_PROJECT_TOOL, TRACK_PROJECT_TOOL] if settings.projects else []


def _tool_name(arguments: str | None) -> str:
    """The project tools' required 'name' argument."""
    return _tool_string(arguments, "name", MAX_PROJECT_NAME_CHARS)


def _tool_description(arguments: str | None) -> str:
    """track_project's 'description' argument - the sentence that makes the project findable."""
    return _tool_string(arguments, "description", MAX_PROJECT_TAGLINE_CHARS)


def _tool_query(arguments: str | None) -> str:
    """The search tool's required 'query' argument."""
    return _tool_string(arguments, "query", MAX_QUERY_CHARS)
