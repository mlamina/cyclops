"""The realtime voice agent: OpenAI Realtime API over WebSocket, plus its webcam and web tools."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import re
import sys
import time
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

from . import imagine, panel, recall, session, sfx
from .audio import SAMPLE_RATE, EchoGuard, Microphone, Speaker, resolve_device
from .config import Settings
from .search import SearchError, search_web
from .webcam import Capture

if TYPE_CHECKING:  # the projects package pulls in pydantic_ai; the tools import it when called
    from .projects.data import Book
    from .projects.store import Project

BARGE_IN_CONFIRM_S = 1.5  # server must report speech within this long of a local barge-in
# How often the websocket has to prove the link is still there, and how long a ping may go
# unanswered before the socket is declared dead. A realtime connection can black-hole: the TCP
# session stays open, we go on writing audio into it, and nothing whatsoever comes back. With no
# ping there is nothing to notice it - `async for event in conn` simply waits, forever, while the
# panel goes on saying "listening — talk to me". That is not hypothetical: on 2026-08-31 a session
# took forty seconds of speech into a dead socket and answered none of it. Fifteen seconds each
# way puts a fault on the panel inside half a minute, which is about as long as anyone will keep
# talking to a box that has stopped answering.
KEEPALIVE_S = 15.0
# ...and no silent reconnect underneath us. The SDK will transparently redial a dropped socket,
# but a redial is a *new* realtime session: it arrives with none of our session.update config -
# no voice, no tools, no transcription - and none of the conversation. A Cyclops that quietly
# comes back as a different, emptier Cyclops while the panel stays green is a worse lie than the
# one being fixed here. Let it surface instead; the panel has a red state and a line to say why.
RECONNECT_ATTEMPTS = 0
TRANSCRIPTION_MODEL = "gpt-4o-mini-transcribe"
DEFAULT_REASONING_EFFORT = "low"  # OpenAI's recommendation for production voice agents
REASONING_MODEL = re.compile(r"^gpt-realtime-2(\.\d+)?(-mini)?$")  # not gpt-realtime-2025-08-28
MAX_QUERY_CHARS = 300
# A screenful of markup and no more. The argument IS the latency here - nothing can appear until
# the model has finished writing it - and 800x480 read at arm's length holds a number, a short
# list or a small drawing, none of which need more than this.
MAX_SCRATCHPAD_CHARS = 1500
MAX_PROJECT_NAME_CHARS = 80
MAX_PROJECT_TAGLINE_CHARS = 300  # a little under store.MAX_TAGLINE_CHARS
MAX_PROJECT_NOTES_CHARS = 4000  # a project page, not a card's worth of them
SEARCH_TIMEOUT_S = 14.0  # above search.SEARCH_TIMEOUT_S, so its own message wins
MAX_DATA_TAB_CHARS = 40  # a sheet title Excel will take; see projects.data.MAX_TITLE_CHARS
MAX_DATA_KEY_CHARS = 80  # a label someone looks a value up by, not a sentence
MAX_DATA_VALUE_CHARS = 200
MAX_DATA_NOTE_CHARS = 200
MAX_DATA_QUERY_CHARS = 120
# A recall is one embedding round trip and a dot product, so it is fast enough that the panel's
# caption barely appears. The budget is a backstop against a wedged connection, not a plan: past
# this the user has been staring at "looking for…" long enough that no answer is the kinder one.
RECALL_TIMEOUT_S = 8.0
RECALL_HITS = 5
# How many of those the model is told about at all. The first is already on the panel by the time
# it reads this, so the rest are there to be offered aloud - "I've also got a caliper close-up" -
# and three names is the most anybody wants read back at them.
RECALL_OFFERED = 3
# A photo's title is its whole caption, which for a page of specifications is a paragraph. Cut it:
# these are labels in a result, not the answer, and the picture itself is going to the model.
MAX_OTHER_CHARS = 90
MAX_DATA_ENTRIES = 20  # one plate's worth of values, generously
ACTIVITY_SUBJECT_CHARS = 40  # a subject on the caption, not a sentence
# How long a finished job's sentence stays on the panel. A data tool is off the card and back in
# five milliseconds - a tenth of one frame - so without a floor under it the caption would strobe
# a phrase nobody could catch, which is worse than the nothing it used to show. Fifteen frames is
# about a glance.
ACTIVITY_HOLD_S = 0.6
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")  # keep \t and \n

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

DRAW_DIAGRAM_TOOL: RealtimeFunctionToolParam = {
    "type": "function",
    "name": "draw_diagram",
    "description": (
        "Draw a technical diagram on the touchscreen: a wiring or connection diagram, a pinout, "
        "a block diagram, a flow or a state machine. Use it when the answer is a layout or a set "
        "of connections that would take several sentences to say and one picture to show - "
        "'wire this relay to GPIO 17', 'what goes where on the header', 'how does this loop "
        "work'. "
        "It takes about a minute and fills the panel when it lands, and it stays until they put "
        "it away, so say one short sentence out loud first and then keep talking; do not narrate "
        "the drawing or read it back to them, they can see it. It is kept with this session's "
        "photos, so use recall to put it back up later rather than drawing it a second time. "
        "What comes back is drawn, not checked. It is usually right, but a line can land on the "
        "wrong pin while the label beside it stays correct. So when a connection is one they "
        "would actually act on - and getting it wrong would cost them a part or a shock - say "
        "that connection out loud as well as showing it. "
        "Nothing in it is to scale and nothing in it can be measured, whatever it looks like. So "
        "it can show how the parts of a bracket go together, but it cannot be a cutting list, a "
        "joinery detail or a panel layout somebody would work to: give those out loud, as "
        "numbers. Never tell them to measure anything off the picture."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "request": {
                "type": "string",
                "description": (
                    "What to draw, in one or two sentences, with every value that matters - "
                    "part names, pin numbers, resistances, voltages. Whoever draws it sees only "
                    "this sentence and nothing of your conversation, so it has to stand alone."
                ),
            },
            "style": {
                "type": "string",
                "description": (
                    "How this particular drawing should look, as a short phrase naming the "
                    "conventions of the field it belongs to - the kind of drawing they would "
                    "find in a manual for the thing on the bench in front of them. This is not "
                    "a choice from a list: work out what the subject is, then name the paper, "
                    "the linework, the use of colour and the labelling that a drawing of THAT "
                    "would have. "
                    "It matters because a diagram in the wrong idiom is read wrong. Ask for "
                    "wires and terminals on a piece of software and it will invent terminals "
                    "that do not exist; ask for a flat schematic of a loom and you lose the "
                    "colour code they need to match against real wire in their hand. "
                    "Examples of the shape of answer wanted - a relay wired to a Pi, or a "
                    "motorcycle loom: 'a printed workshop service-manual wiring diagram, black "
                    "line-art on off-white paper, wires drawn in their real colours with a small "
                    "colour key'. The 40-pin header: 'a pinout chart, two numbered columns of "
                    "pins with the names set outward, power, ground and signal each in their own "
                    "colour'. An audio path, or how a program hangs together: 'a plain flat "
                    "block diagram, plain boxes and labelled arrows, no wires, no terminals, one "
                    "accent colour at most'. How a bracket goes together: 'a clean assembly "
                    "sketch, the parts drawn separated along the axis they slide on, numbered "
                    "callouts and a small parts key'. A pipe or duct run: 'an isometric pipework "
                    "diagram in the style of a plumbing manual, runs in a single line weight, "
                    "fittings and valves as standard symbols'."
                ),
            },
        },
        "required": ["request", "style"],
        "additionalProperties": False,
    },
}

SCRATCHPAD_TOOL: RealtimeFunctionToolParam = {
    "type": "function",
    "name": "write_on_scratchpad",
    "description": (
        "Write on your SCRATCHPAD: the touchscreen in front of them, which is yours to write on "
        "as a small piece of HTML. It is up about a second later, while you are still talking, "
        "and it costs nothing. "
        "The scratchpad is what you and they both call this. Expect to be asked for it by that "
        "name - 'put that on your scratchpad', 'scratchpad it', 'what's on the scratchpad' - and "
        "call it that yourself when you mention it at all. Every one of those is this tool. "
        "Use it on your own initiative, without being asked, whenever the answer has something "
        "in it worth looking at rather than hearing: a torque figure or a temperature set large, "
        "the steps of a job as a numbered list they can work down, a part number, a size, a "
        "setting, a simple shape as inline SVG. A number you say once over a running compressor "
        "is a number they will ask you for again; a number on the scratchpad is one they work to. "
        "So show it AND say it - put the figure up, then say the caveat out loud. "
        "This is the tool to reach for by default. draw_diagram is the expensive exception: it "
        "is for a real technical drawing - how something is wired, what goes where on a header, "
        "how parts fit together - and it costs a minute, which is a minute wasted on anything "
        "that is text, numbers, a list or a simple shape. "
        "The scratchpad is 800x480, read at arm's length across a bench, and it does not scroll: "
        "whatever does not fit is not seen. One idea on it at a time, set big, a handful of "
        "elements. "
        "The background, the text colour and the font are already his own, and headings, lists "
        "and SVG are already sized for the panel, so plain markup with no styling at all comes "
        "out right. Style it when the styling MEANS something - a value in red because it is out "
        "of range, a colour because they are matching a wire to it. "
        "Two things to know. It holds the whole panel until they touch it - a press anywhere wipes "
        "the scratchpad and gives them his eye back - so it is for the answer and not a caption "
        "on every sentence. And it replaces whatever picture was there, so do not cover a photo "
        "they are still asking you about; edit_photo will have nothing left to work on."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "html": {
                "type": "string",
                "description": (
                    "What goes on the scratchpad, as HTML. No <html>, <head> or <body> - just the "
                    "elements. A number is '<h1>25 Nm</h1>'. Steps are an <ol> of short <li>. "
                    "You may use <style> or inline style= to override anything. Emoji are just "
                    "characters and can go anywhere. Scripts do not run and nothing is loaded "
                    "from the network, so images must be inline SVG rather than a src."
                ),
            },
        },
        "required": ["html"],
        "additionalProperties": False,
    },
}

EDIT_PHOTO_TOOL: RealtimeFunctionToolParam = {
    "type": "function",
    "name": "edit_photo",
    "description": (
        "Redraw the photo on the touchscreen with a change made to it, and put the result up "
        "in its place. Use it when the answer is 'like this' about the actual thing in front of "
        "them and saying it would take a paragraph: a colour or a finish, a part moved or taken "
        "away, a shelf on that wall, the half-built thing shown finished, that corner tidied. "
        "It works on whichever photo is on the panel - one they just took, one you found for "
        "them, one you drew for them, or one you already edited, so a second change carries on "
        "from the first. If nothing has been up at all yet, ask them to hit SNAP. "
        "It takes up to a minute and fills the panel when it lands, so say "
        "one short sentence out loud first and then keep talking. You are shown the result when "
        "it lands, but so are they: do not narrate it back at them unprompted. Volunteer "
        "something only if it did not do what they asked or there is something worth flagging - "
        "but answer whatever they do ask about it, directly, because you can see it. "
        "What comes back is an illustration, never evidence. The whole picture is redrawn, so "
        "nothing in it is measured and nothing in it is a fact about their hardware - and "
        "because it started as a photograph of their own bench, it is the one picture they could "
        "mistake for a record of it. So do NOT use it for connections, wiring, which way round a "
        "part goes, the order to assemble something, or anything they would act on: draw_diagram "
        "is for those, because it draws the answer from a description instead of painting over "
        "their hardware. Do not use it to read a label or a plate: look at the photo you "
        "already have. Never call it to show them what something 'really' looks like."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "request": {
                "type": "string",
                "description": (
                    "What to change, as an instruction to whoever is holding the picture: "
                    "'paint the cabinet doors matt black, leave the worktop alone', 'show the "
                    "bracket moved to the left end of the rail'. Name what to change AND what "
                    "to leave alone. Whoever edits it sees the photo and this sentence and "
                    "nothing of your conversation, so it has to stand alone."
                ),
            }
        },
        "required": ["request"],
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

SAVE_DATA_TOOL: RealtimeFunctionToolParam = {
    "type": "function",
    "name": "save_data",
    "description": (
        "Write down a value belonging to a project so it can be looked up again weeks from now: "
        "a torque, a size, a pressure, a part number, a paint code, a setting, a short "
        "descriptor. Call it the moment one comes up - when they tell you one, and when you read "
        "one off a plate, label or manual page in a photo they have just shown you. Do not ask "
        "permission and do not offer: save it, then say in a few words what you wrote down. "
        "Saving a key that is already there replaces its value, which is how a corrected "
        "measurement is recorded."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "project": {
                "type": "string",
                "description": (
                    "Which project this belongs to, as the user calls it - one of the ones you "
                    "are keeping notes on. Close is good enough. If you cannot tell from the "
                    "conversation which one it is, ask them in one short sentence instead of "
                    "guessing, and never start a new project just to have somewhere to put it."
                ),
            },
            "tab": {
                "type": "string",
                "description": (
                    "The sheet to file these under, grouped by kind: 'Torque specs', "
                    "'Dimensions', 'Paint', 'Part numbers', 'Settings'. Reuse a tab that already "
                    "fits - open_project tells you which ones exist and how full they are - "
                    "rather than making a near-duplicate of one."
                ),
            },
            "entries": {
                "type": "array",
                "description": (
                    "Every value from this photo or this sentence, in ONE call. A plate with six "
                    "numbers on it is six entries here, not six calls."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": (
                                "What a person would look it up by: 'Caliper bolt torque', 'Jaw "
                                "width', 'Cover paint'. A short label, not a sentence, and "
                                "specific enough to tell it from the next bolt along."
                            ),
                        },
                        "value": {
                            "type": "string",
                            "description": (
                                "Exactly as they said it or as it is written on the thing, unit "
                                "included: '25 Nm', '3/8 inch', 'RAL 7016', 'M10x1.5'. Never "
                                "strip the unit, never convert, never round."
                            ),
                        },
                        "note": {
                            "type": "string",
                            "description": (
                                "Optional: the one thing that makes the value usable later - "
                                "'dry thread', 'measured, not spec', 'front pair only'. Leave it "
                                "out when there is nothing to add."
                            ),
                        },
                    },
                    "required": ["key", "value"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["project", "tab", "entries"],
        "additionalProperties": False,
    },
}

FIND_DATA_TOOL: RealtimeFunctionToolParam = {
    "type": "function",
    "name": "find_data",
    "description": (
        "Look up a value that was written down for a project. Call this every single time they "
        "ask for something that was measured, set or specified before - never answer such a "
        "question from memory, because what you remember of a number is not what was written "
        "down. It searches by meaning, so their own words are enough. If it comes back with "
        "nothing, say plainly that it was not written down instead of producing a number."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "project": {
                "type": "string",
                "description": (
                    "Which project to look in, as the user calls it. Close is good enough. Only "
                    "this project is searched, so if you cannot tell which one they mean, ask."
                ),
            },
            "query": {
                "type": "string",
                "description": (
                    "What they are after, in their words: 'caliper torque', 'jaw width', 'the "
                    "paint code'. Near enough is fine - spelling and word order do not matter."
                ),
            },
        },
        "required": ["project", "query"],
        "additionalProperties": False,
    },
}

FORGET_DATA_TOOL: RealtimeFunctionToolParam = {
    "type": "function",
    "name": "forget_data",
    "description": (
        "Delete one value from a project, for when they say it was wrong, was for something "
        "else, or is finished with. To CORRECT a value, call save_data with the same key "
        "instead - that replaces it and keeps its place. If more than one thing matches, this "
        "deletes nothing and hands you the candidates: ask which one they mean."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "project": {
                "type": "string",
                "description": "Which project the value is in, as the user calls it.",
            },
            "key": {
                "type": "string",
                "description": "The value to remove, as they refer to it. Close is good enough.",
            },
            "tab": {
                "type": "string",
                "description": (
                    "Optional: the sheet it is on, when you know it and the key alone would "
                    "match more than one thing."
                ),
            },
        },
        "required": ["project", "key"],
        "additionalProperties": False,
    },
}

RECALL_TOOL: RealtimeFunctionToolParam = {
    "type": "function",
    "name": "recall",
    "description": (
        "Find something you have on the card and, when it is a picture, put it on the "
        "touchscreen. This searches by meaning rather than by wording, so their words do not "
        "have to match what was written: photos and their descriptions, what was written up "
        "about each session of a project, and any file they put in the project folder "
        "themselves. Use it whenever they refer back to something that exists - 'show me the "
        "pic of the torque spec from the manual', 'what did we decide about the fork seals', "
        "'find that datasheet I put in there'. A photo appears on the panel and stays until "
        "they tap it, so say one short sentence and then stop; they can see it, so do not "
        "describe it back at them unless they ask - and you are shown it too, so answer "
        "whatever they do ask about it by reading the picture. Read it off the picture and not "
        "off this tool's result: the words in the result were written to find the photo, not to "
        "describe it, and they are not reliable about numbers or small print. "
        "It hands back the one best match and the names of a couple of near misses. Those are "
        "not a menu to read out - they are there in case what you showed is plainly the wrong "
        "thing, in which case offer one of them in a few words. "
        "A diagram you drew earlier is kept with the photos, so this is how you put one back up "
        "when they refer to it - it is far quicker than drawing it again, and a redraw would "
        "come back different. "
        "Do NOT use it for: a number written down with save_data - find_data looks those up "
        "exactly and this only finds the words around them; anything about the world rather "
        "than about their own work - that is web_search. If it finds nothing, say so and offer "
        "to look another way rather than showing them the closest thing anyway."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "What they are looking for, in their words and in a full phrase rather "
                    "than keywords - 'the torque spec from the owner's manual' beats 'torque'. "
                    "It is matched on meaning, so describing the thing works better than "
                    "guessing at what it was called."
                ),
            },
            "project": {
                "type": "string",
                "description": (
                    "Optional: the project to look in, when they named one or the conversation "
                    "is plainly about a single one. Leave it out to search everything."
                ),
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}

DATA_TOOLS = frozenset({"save_data", "find_data", "forget_data"})

# The static half of what the model is told. The other half - what the last few sessions were
# about - is read off the card at connect time by :func:`build_instructions`.
BASE_INSTRUCTIONS = """\
You are Cyclops: a one-eyed device that sits on the bench next to someone who is making or
fixing something. They switch you on, point you at the job, and switch you off when they are
done. The eye is their webcam, and they hold the shutter: you see what they show you,
when they show it.

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
- You have a SCRATCHPAD as well as a voice - the touchscreen in front of them - and writing on
  it is part of answering rather than an extra. When the answer has a number, a list of steps
  or a shape in it, put that on the scratchpad and say the rest out loud. A figure they have
  to hold in their head while they work is a figure they will ask you for twice.
- Call it the scratchpad, because that is what they call it. "Put that on your scratchpad",
  "scratchpad it", "what's on the scratchpad" - all of them mean write_on_scratchpad. A press
  anywhere on it wipes it and gives them your eye back, so they never have to ask you to.
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
- You cannot take photos. They take them, by pressing the SNAP button on the panel, and the
  photo reaches you the moment they do.
- A photo arriving means they just pressed that button, sometimes mid-sentence. Go straight to
  what you actually see, briefly, then answer whatever they were asking about it. No preamble:
  never open with "look at this", "let me see", or by narrating that a photo arrived.
- When they hold something up or ask what you can see, and no photo has arrived, ask them for
  one - once, in a few words. "Hit SNAP and I'll look."
- Ask once and then let it go. If no photo comes, carry on without it; never nag for one, and
  never claim to see something you have not been shown.
- If the image is dark, blurry, or empty, say so ONCE and wait. Do not ask for another.

LOOKING THINGS UP
- When they ask something factual you are not sure about - a spec, a size, a torque value,
  whether two parts fit together, what something costs, anything that may have changed
  recently - call the web_search tool instead of guessing. Say a few words first ("let me look
  that up") so they are not left in silence, because the search takes several seconds.
- Combine the two when it helps: ask for a photo of the thing, then search for what you saw.
  If a search comes back empty or failed, say so plainly instead of inventing an answer.
- When what comes back is a figure they are going to work to - a torque, a clearance, a gap, a
  temperature - write it on the scratchpad as you say it. That is exactly what the scratchpad
  is for, and a number read out once over a running compressor is a number they lose.

SHOWING THEM SOMETHING
- Write on the scratchpad yourself, and do it without being asked. A torque figure or a
  temperature, set large. The steps of a job as a numbered list they can work down. A part
  number, a size, a setting. A simple shape as SVG. It costs nothing and it is up in about a
  second, while you are still talking.
- Do it as you answer, not instead of answering: put the number on the scratchpad and say the
  caveat out loud. Then stop describing what is up there - they can see it.
- The one restraint: it holds the whole panel until they touch it, so it is for the answer,
  not for a caption on every sentence.
- When the answer is a set of connections or a layout - how something is wired, what goes where on
  a header, how parts fit together - draw it rather than saying it; see draw_diagram for what it
  can and cannot draw. It takes about a minute, so say what you are doing and carry on talking.
  Never reach for it for something you could have put on the scratchpad in a second.
- When the answer is what something would LOOK like - a colour, a finish, a part moved, a thing
  that is not there yet - edit the picture in front of them rather than describing it; see
  edit_photo. It can take a minute, so say what you are doing and carry on talking. What comes
  back is a drawing of their photo and not a photograph: never treat it as evidence and never
  measure anything off it.
- Once it is up, stop describing it. They can see it. Answer what they ask about it.

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

NUMBERS YOU KEEP
- Alongside the notes, every project has a sheet of the small hard facts about it: sizes,
  torques, pressures, part numbers, settings, codes. Write one down whenever it comes up. It
  costs a moment now and it is the whole reason they can ask you in three weeks.
- When they want one back, look it up. Never answer a saved value from memory, and if it is not
  there, say so rather than produce a number.

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

# Introduces the list of standing facts about whoever is on the other side of the bench - see
# :mod:`cyclops.about`. Knowing somebody is worth having and easy to hold wrong, so the header
# spends its lines on the three ways it goes wrong: reading the list out, treating it as more
# current than the person in front of you, and telling them there is a file.
ABOUT_HEADER = """\
WHO YOU ARE TALKING TO
These are standing facts about them, written down at the end of earlier sessions and true until
they say otherwise. Use them the way you use knowing somebody: call them by their name, pitch
what you say to what they already know, and remember what they said they wanted help with. Do
not recite the list back, do not congratulate yourself on remembering, and never mention that
any of this is written down anywhere. If something they say today contradicts a line here, they
are right and the line is out of date.
"""

# And what stands in its place on a box that has never been told anything. It is an instruction
# to ask, and it needs to be a careful one: BASE_INSTRUCTIONS already says they set the agenda
# and that curiosity is one good question rather than more words, and the fastest way to ruin
# this whole feature is a machine that opens by interviewing somebody who came to fix a tap.
ABOUT_UNKNOWN = """\
YOU HAVE NOT MET THEM YET
Nothing has ever been written down about the person you work with - not their name, not what
they do, not what they came to you for. Somewhere in this session, once, when they are not
mid-cut, mid-measurement or waiting on an answer from you, ask them one open question about
themselves. What they do, what they are into, what they would want something like you for. One
question, and then let it go: if they brush it off or answer thinly, do not ask again today and
do not work round to it. Never open with it. The job in front of them still comes first, and
what you learn this way you keep.
"""

# A backstop on the block as a whole, as RECAP_MAX_CHARS is on the recap - and, like that one,
# sized so it never actually bites: the header is about 490 characters and MAX_FACTS lines at
# MAX_FACT_CHARS each are 1692 more, so the worst case a full card can produce still fits under
# it. A cap that trims in ordinary use is not a backstop, it is a budget nobody wrote down, and
# what it would trim here is the end of a sentence about somebody.
ABOUT_MAX_CHARS = 2400


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


def _about_block(settings: Settings) -> str:
    """What it knows about them, or the instruction to go and find out.

    Read at connect time off the card, like the recap and the project list, and for the same
    reason: what it learned about somebody in the session that ended a minute ago has to be
    there in this one.

    Unlike those two this never returns "" while the feature is on. An empty list is not nothing
    to say - it is the one case worth spending words on, because a machine that does not know
    who it is talking to and is not told so will simply carry on as though it did.
    """
    if not settings.remember:
        return ""
    from . import about  # local, like the projects import below, so a keyless box still builds

    facts = about.read(settings)
    if not facts:
        print("· about you: nothing yet - it will ask once", flush=True)
        return ABOUT_UNKNOWN
    listed = "\n".join(f"- {fact}" for fact in facts)
    print(f"· about you: {len(facts)} thing(s) known", flush=True)
    return f"{ABOUT_HEADER}\n{listed}\n"[:ABOUT_MAX_CHARS]


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
    # Who, then what happened, then what is being kept - which is the order somebody walking up
    # to the bench would want them in.
    if about_block := _about_block(settings):
        blocks.append(about_block)
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


class _Doing:
    """One job the panel is reporting, and how long it is still worth reporting.

    Mutable, and deliberately not a frozen dataclass: :meth:`VoiceAgent._done_doing` marks one
    of these finished by writing a deadline into it, and rebuilding the whole tuple to do that
    would race with a second tool starting on the same loop.
    """

    __slots__ = ("line", "until")

    def __init__(self, line: str) -> None:
        self.line = line
        self.until = float("inf")  # monotonic; infinite while the work is actually running


@dataclass(frozen=True)
class Panel:
    """The last picture put in front of them, and what made it.

    One slot rather than the ``_last_photo`` this replaced, which only ever meant "the last thing
    the *camera* saw". A picture reaches the panel four ways now - the shutter, a recall off the
    card, an edit, and a diagram drawn from a description - and ``edit_photo`` works on whichever
    of them is up, because the thing somebody means by "change that" is the thing they are
    looking at.

    ``path`` stayed optional after the drawings became photographs. It no longer marks the
    diagram case - a drawing is a jpg in ``photos/`` with a path like everything else - but a
    picture that could not be written to the card still reaches the glass, and there is nothing
    for an edit to send when it does.
    """

    path: Path | None
    what: str  # "photo" | "found" | "edit" | "drawn" - for the log and the tool's own error


class VoiceAgent:
    """Owns one Realtime session: streams mic audio up, plays audio down, shows it the photos.

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
        self.tool_active = False  # True while a photo is going up (UI 'looking')
        self.search_active = False  # True while a web search is in flight (UI 'searching')
        self.drawing_active = False  # True while a picture is being made (UI 'drawing')
        # The picture on the panel, which is what edit_photo works on. See :class:`Panel`.
        # Kept here rather than found by scanning photos/ for the newest file, because those two
        # are not the same thing: a shutter pressed before the session was ready writes a jpg
        # nothing ever saw, and editing a picture the model cannot reason about is worse than
        # asking for another.
        self._on_panel: Panel | None = None
        # The picture a recall just put on the panel, waiting to be handed to the model. Held on
        # the agent rather than returned, because it has to be sent *after* the tool output that
        # mentions it - see _run_recall.
        self._found_image: bytes | None = None
        # ...and, beside those three, the sentence the panel says underneath. The flags answer
        # "what mode is this?", which colours the border and picks the word on the strip, and
        # they stay a closed set of three. This answers "what is it doing?", which is open-ended
        # - every tool adds one - and so it is carried as text rather than as another flag.
        #
        # A tuple rather than one string because a response can call two tools at once
        # (_on_response_done spawns a task per call), and a find_data that lands in five
        # milliseconds must not take a fifteen-second search's caption down with it. Oldest
        # first; the newest one still worth saying wins.
        self._doing: tuple[_Doing, ...] = ()
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
        # Cues sound on their own stream (see cyclops.sfx), but on the same device as the voice
        # rather than whatever the system calls default - those are not always the same speaker.
        # Public because the shutter is sounded by whoever holds the camera, not by the agent.
        self.cues = sfx.Cues(
            rate=SAMPLE_RATE,
            device=resolve_device(settings.output_device),
            enabled=settings.sounds,
        )

    # ---- what the panel says we are doing ----
    #
    # Written from this agent's own event loop and read from the kiosk's render thread, once a
    # frame, with no lock between them. What makes that safe is that every write is a single
    # rebinding of ``_doing`` to a new tuple: the reader binds the name once and then walks a
    # tuple nobody can shorten under it, so it sees the old set of jobs or the new one and never
    # half of either. The deadline inside a :class:`_Doing` is written by one thread and read by
    # the other, which on CPython is one float store - the reader can be a frame behind, and a
    # frame is 40 ms.

    def _doing_now(self) -> tuple[_Doing, ...]:
        """The jobs still worth mentioning, with the expired ones dropped.

        Pruning on the way in rather than on a timer is what keeps the tuple the length of the
        work actually in flight - one to three - rather than the length of the session.
        """
        now = time.monotonic()
        return tuple(job for job in self._doing if now < job.until)

    def _start_doing(self, line: str) -> _Doing | None:
        """Say *line* until told otherwise. Hand what comes back to :meth:`_done_doing`.

        An empty line is not an error and not a job: a tool with nothing worth naming leaves the
        panel saying whatever it was saying, which is better than blanking it for the duration.
        """
        if not line:
            return None
        job = _Doing(line)
        self._doing = (*self._doing_now(), job)
        return job

    def _done_doing(self, job: _Doing | None) -> None:
        """That job is over - but let it stand for a beat, or nobody could have read it."""
        if job is not None and job.until == float("inf"):
            job.until = time.monotonic() + ACTIVITY_HOLD_S

    @property
    def activity(self) -> str:
        """One sentence about work in flight, or empty when there is nothing to add.

        The newest job still worth mentioning wins, falling back through anything underneath it
        that is genuinely still running - so a lookup that finished instantly hands the line
        back to the search it interrupted rather than to nothing at all.
        """
        now = time.monotonic()
        for job in reversed(self._doing):
            if now < job.until:
                return job.line
        return ""

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
    def connected(self) -> bool:
        """Whether the socket is still up.

        Not the same question as :attr:`ready`, which is a latch: it is set once the server
        accepts our config and never cleared. ``run()`` drops the connection on the way out
        while the controller is still holding the agent - and still finishing the session
        folder, which takes seconds - so anything arriving from another thread has to ask this
        instead, or it reaches the assert in :attr:`conn`.
        """
        return self._conn is not None

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
            "tools": [
                WEB_SEARCH_TOOL,
                *_diagram_tools(self.settings),
                *_scratchpad_tools(self.settings),
                *_imagine_tools(self.settings),
                *_project_tools(self.settings),
                *_recall_tools(self.settings),
            ],
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
        self.cues.play("connecting", loop=True)
        # The try opens before the connect, not after it: a bad key or a dead network raises
        # from there, and that is precisely when the connecting cue is playing. Left outside,
        # the finally never runs and the kiosk pings on over a red border.
        try:
            # Narrated in two steps because they fail in two different places and take
            # noticeably different amounts of time: the socket, and then the round trip that
            # settles the config. A panel that said only "connecting" for both would leave the
            # commonest failure - a key the server rejects, which happens after the connect -
            # looking like a network that never came up.
            self._start_doing("connecting to OpenAI…")
            client = AsyncOpenAI(api_key=self.settings.api_key)
            async with client.realtime.connect(
                model=self.settings.model,
                websocket_connection_options={
                    "ping_interval": KEEPALIVE_S,
                    "ping_timeout": KEEPALIVE_S,
                },
                max_retries=RECONNECT_ATTEMPTS,
            ) as conn:
                self._conn = conn
                await conn.session.update(session=self.session_config())
                self._start_doing("waiting for the model…")
                async for event in conn:
                    await self._handle_event(event)
        except Exception:
            # A connect that never landed: the ping is ours to stop, because nothing else is
            # going to - the panel just goes red. A *cancel* is the other thing entirely, and
            # deliberately not caught here: someone pressed stop, and the sound of stopping
            # belongs to them. Ending a session is sounded by whoever owns its lifecycle (the
            # kiosk on the tap, the CLI on its way out), never from here, because this runs
            # long before it is over - the socket, the devices and the recording outlive it.
            if not self.ready.is_set():
                self.cues.stop()
            raise
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

    # ---------------------------------------------------------------- photos down the pipe

    def queue_photo(self, capture: Capture) -> None:
        """Show the model a photo the user just took. Safe to call from the loop thread only.

        The entry point for a button press: :meth:`cyclops.ui.SessionController.show_photo`
        hands it here with ``call_soon_threadsafe``, so this is where the "is there still a
        session to show it to" question gets its authoritative answer - on the loop, where it
        cannot go stale between the check and the send. The work is spawned as a *tracked*
        task so ``run()``'s teardown cancels it rather than leaving it pending on a closing loop.
        """
        if not self.connected or not self.ready.is_set():
            return
        self._spawn(self.add_photo(capture))

    async def add_photo(self, capture: Capture) -> None:
        """Put a photo into the conversation as an image, and ask for a reply about it.

        The model has no camera of its own, so this is the only way anything is ever seen. The
        item is a synthetic user turn rather than a tool result, because there is no tool call
        to answer: as far as the conversation is concerned the user held something up.
        """
        if not self.connected:
            return
        self.tool_active = True
        job = self._start_doing("looking at the photo…")
        try:
            await self._send_item(
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            # Flat and unquotable on purpose: a caption written as speech
                            # ("look at this") comes back out of the speaker verbatim.
                            "text": "[Photo from their camera, taken just now.]",
                        },
                        {
                            "type": "input_image",
                            "image_url": capture.data_url,
                            "detail": "auto",
                        },
                    ],
                }
            )
            # Only now: this means "shown", not "taken".
            self._on_panel = Panel(capture.path, "photo")
            await self._request_response()
        finally:
            self.tool_active = False
            self._done_doing(job)
        self._log(
            f"[photo] {capture.width}x{capture.height}, "
            f"{capture.jpeg_bytes // 1024} KB → {capture.path}"
        )

    async def add_edit(self, jpeg: bytes, request: str) -> None:
        """Show the model the picture it just had made. No response is asked for here.

        Deliberately not :meth:`add_photo`, for two reasons that both still matter. It must not
        ask for a response of its own: this rides inside a tool call whose output is still to be
        sent, and ``_run_edit_photo`` issues the one ``response.create`` for the pair. And it must
        not move ``_on_panel`` - not because an edit is the wrong thing to edit next, which it no
        longer is, but because ``_run_edit_photo`` owns that bookkeeping and does it on the loop
        with the path the picture was actually written to. This method is handed bytes and has no
        path to record.

        The label is flat and unquotable for the reason :meth:`add_photo`'s is - a caption
        written as speech comes back out of the speaker verbatim - and it says twice over what
        the picture is, because this is the nearest text to an image that otherwise looks exactly
        like a photograph of the user's own bench.
        """
        if not self.connected:
            return
        await self._send_item(
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "[The picture you just had made, now on their screen. It is an "
                            f"illustration of their last photo with this change applied: "
                            f"{request}. Every pixel of it was drawn, including the parts that "
                            "look untouched, so nothing in it is a measurement or a fact about "
                            "their hardware. They are looking at it too, so do not narrate it "
                            "unprompted - but answer what they ask about it.]"
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": "data:image/jpeg;base64,"
                        + base64.b64encode(jpeg).decode("ascii"),
                        "detail": "auto",
                    },
                ],
            }
        )
        self._log(f"[edit] showed the model {len(jpeg) // 1024} KB")

    # ---------------------------------------------------------------- audio up

    async def _listen(self, *, after_s: float) -> None:
        """Open the mic once the ready chime has finished, and pump it from there on.

        The chime plays on its own stream, so the EchoGuard - which only knows about the
        Speaker - cannot mute the mic for it, and on headphones there is no guard at all.
        Rather than defend against hearing our own chime, we just don't listen until it has
        stopped: the chime is the thing telling you to talk, so nobody is mid-sentence during
        it, and this is the one moment in a session where half a second of deafness is free.
        Draining afterwards rather than before then clears exactly the blocks it played into.
        """
        assert self.mic is not None
        if after_s:
            await asyncio.sleep(after_s + sfx.SETTLE_S)
        self.mic.drain()  # audio from before the session was configured is stale
        self._log("listening — talk, or take a photo, Ctrl+C to quit")
        await self._pump_mic()

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
        self.ready.set()  # set first: the panel and show_photo() both gate on it
        # Both handshake lines go at once rather than expiring: nothing else can be in flight
        # this early - a tool needs a response and a response needs this - so there is nothing
        # underneath them to hand the caption back to. From here the state's own resting line is
        # the true one, until a tool has something better to say.
        self._doing = ()
        sounding = self.cues.play("ready")  # which also ends the connecting loop
        if self.mic is not None:
            self._spawn(self._listen(after_s=sounding))

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
        """Run one tool call, saying on the panel what it is for as long as it takes.

        One place says it, for every tool there is. The chain below already knows the name, so a
        line in each of its six branches would only be six chances to forget the seventh - and
        the seventh is precisely the one nobody would notice was silent.
        """
        job = self._start_doing(_activity_line(call))
        try:
            await self._dispatch_tool(call)
        finally:
            self._done_doing(job)

    async def _dispatch_tool(self, call: RealtimeConversationItemFunctionCall) -> None:
        if call.name == "web_search":
            await self._run_web_search(call)
            return
        if call.name in {"open_project", "track_project"}:
            await self._run_project_tool(call)
            return
        if call.name in DATA_TOOLS:
            await self._run_data_tool(call)
            return
        if call.name == "write_on_scratchpad":
            await self._run_scratchpad(call)
            return
        if call.name == "draw_diagram":
            await self._run_draw_diagram(call)
            return
        if call.name == "edit_photo":
            await self._run_edit_photo(call)
            return
        if call.name == "recall":
            await self._run_recall(call)
            return
        # Every name still gets an output. A tool the model invents, or one it remembers from a
        # session config that has since changed, must be answered or it waits for it forever.
        self._log(f"[tool] unknown tool {call.name!r}", stream=sys.stderr)
        await self._send_tool_output(call.call_id, {"ok": False, "error": "unknown tool"})
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

    # ---- drawing one ----

    async def _run_scratchpad(self, call: RealtimeConversationItemFunctionCall) -> None:
        """Put what the model wrote on the panel. The shortest handler here, and deliberately.

        Every other tool that reaches the glass has a picture to make first - a minute of
        ``gpt-image-2``, or a file off the card. This one already has everything it needs in its
        own arguments, so there is nothing between the call and the panel but one small write.
        """
        html = _tool_string(call.arguments, "html", MAX_SCRATCHPAD_CHARS)
        self._log(f"[tool] write_on_scratchpad {len(html)} chars")
        if not html:
            await self._send_tool_output(call.call_id, {"ok": False, "error": "nothing to show"})
            await self._request_response()
            return

        try:
            shown = await asyncio.to_thread(self._write_scratchpad, html)
        except Exception as exc:  # never leave the model waiting for a tool result
            self._log(f"[tool] write_on_scratchpad failed: {exc!r}", stream=sys.stderr)
            await self._send_tool_output(
                call.call_id, {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            )
            await self._request_response()
            return

        if shown:
            # What they are looking at is no longer a picture, so ``edit_photo`` has nothing to
            # work on. Saying so here is the honest answer: the alternative is an edit that
            # spends a minute and a real API call redrawing a photograph that came off the glass
            # when this went up, and hands back something nobody asked to see.
            self._on_panel = None
        output = (
            {"ok": True, "shown": True, "note": "It is on the screen. Do not read it out."}
            if shown
            else {
                "ok": True,
                "shown": False,
                "note": (
                    "There is no panel to show it on. Say so plainly rather than describing it."
                ),
            }
        )
        await self._send_tool_output(call.call_id, output)
        await self._request_response()

    def _write_scratchpad(self, html: str) -> bool:
        """Offer the scratchpad and ask for the panel. Blocking; runs off the loop's thread.

        Off the loop because ``panel.offer_scratchpad`` goes through ``card.write_text``, which
        fsyncs an SD card - the same reason :meth:`_keep_and_show_drawing` is not a coroutine.
        It is a small write, but the loop it would block is the one carrying his voice.

        Nothing is kept. The scratchpad is a sentence he said with the screen instead of his
        mouth, and the session log records it the way it records the rest; there is no file
        for it in ``photos/``, nothing to caption, and nothing for ``recall`` to find. Putting the
        same thing up again costs a second, which is the whole argument.
        """
        shown = panel.offer_scratchpad(html) and panel.show()
        session.note("screen", html=html[:MAX_SCRATCHPAD_CHARS], panel=shown)
        return shown

    async def _run_draw_diagram(self, call: RealtimeConversationItemFunctionCall) -> None:
        """Draw one, keep it with the photos, and put it on the panel.

        The same shape as :meth:`_run_edit_photo`, because since the JointJS schema went these
        are the same operation: a picture arrives as jpeg bytes, is written into ``photos/`` and
        offered to the glass. What differs is only that there is no source photograph.

        The model is deliberately *not* told what the diagram contains. It asked for a picture,
        the picture is on the screen, and a model handed a description of it will read that
        description out at somebody who is already looking at the thing.
        """
        request = _tool_string(call.arguments, "request", imagine.MAX_REQUEST_CHARS)
        style = _tool_string(call.arguments, "style", imagine.MAX_STYLE_CHARS)
        self._log(f"[tool] draw_diagram {request!r} ({style!r})")
        if not request:
            await self._send_tool_output(call.call_id, {"ok": False, "error": "nothing described"})
            await self._request_response()
            return

        self.drawing_active = True
        try:
            # Keeping and showing is inside the try, not in an else. It was in an else once, and
            # a TypeError in the record it writes propagated straight out of this coroutine: the
            # picture was on the panel, and the model sat waiting for a tool result that was
            # never sent until the user spoke over it.
            jpeg = await imagine.draw(request, style, self.settings)
            output, kept = await asyncio.to_thread(self._keep_and_show_drawing, jpeg, request)
        except imagine.ImagineError as exc:
            session.note("photo", by="drawn", request=request[:80], error=str(exc))
            self._log(f"[tool] draw_diagram failed: {exc}", stream=sys.stderr)
            output, kept = {"ok": False, "error": str(exc)}, None
        except Exception as exc:  # never leave the model waiting for a tool result
            session.note("photo", by="drawn", request=request[:80], error=f"{type(exc).__name__}")
            self._log(f"[tool] draw_diagram failed: {exc!r}", stream=sys.stderr)
            output, kept = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, None
        finally:
            self.drawing_active = False

        # The panel slot, set on the loop rather than in the thread that showed it, because
        # `edit_photo` reads it from here. A drawing now has a real path like every other
        # picture, so "make that clearer" edits the diagram instead of being refused.
        if output.get("shown") and kept is not None and kept.path is not None:
            self._on_panel = Panel(kept.path, "drawn")
        await self._send_tool_output(call.call_id, output)
        await self._request_response()

    def _keep_and_show_drawing(
        self, jpeg: bytes, request: str
    ) -> tuple[dict[str, Any], imagine.Edit | None]:
        """Write the drawing down, then ask for the panel. Blocking; runs off the loop's thread.

        Written before it is shown, and shown whether or not writing worked: the panel is what
        was asked for, and a diagram nobody can keep is still a diagram somebody can read.

        With no session running it is drawn and not kept, exactly as before and exactly as
        :meth:`_keep_and_show_edit` does. Not ``session.photo_target``, which looks like the
        right helper and is not: its ``captures/`` fallback is ``webcam._save``'s pruning archive
        with its own naming and its own ``latest.jpg``, and ``imagine.write`` honours neither.
        """
        live = session.current()
        kept: imagine.Edit | None = None
        if live is not None:
            try:
                kept = imagine.write(jpeg, request, live.photos_dir, role="drawn")
            except OSError as exc:
                self._log(f"[tool] could not keep the drawing: {exc}", stream=sys.stderr)

        # One downscaled copy, doing the panel's job. The card keeps the full-size one.
        small = imagine.for_panel(jpeg)
        shown = panel.offer_image(small, request, drawn=True) and panel.show()
        if kept is not None and kept.path is not None:
            session.note(
                "photo",
                by="drawn",
                request=kept.request,
                file=f"{session.PHOTOS}/{kept.path.name}",
                bytes=kept.bytes,
                panel=shown,
            )
        if not shown:
            return {
                "ok": True,
                "shown": False,
                "note": (
                    "It was drawn but there is no panel to show it on. Say so plainly rather "
                    "than describing it."
                ),
            }, kept
        return {"ok": True, "shown": True}, kept

    # ---- recall ----

    async def _run_recall(self, call: RealtimeConversationItemFunctionCall) -> None:
        """Find something on the card and, if it is a picture, put it on the panel.

        The tool is read-only by design. ``cyclops-index`` owns the index and pays for keeping it
        current; this loads a file, embeds one short string and takes a dot product, so a recall
        costs about as long as the network round trip and nothing else. Nothing here walks the
        card, captions anything or writes a byte.
        """
        query = _tool_query(call.arguments)
        project = _tool_string(call.arguments, "project", MAX_QUERY_CHARS)
        self._log(f"[tool] recall {query!r}" + (f" in {project!r}" if project else ""))
        if not query:
            nothing = {"ok": False, "error": "nothing to look for"}
            await self._send_tool_output(call.call_id, nothing)
            await self._request_response()
            return
        turn = self._turn_serial
        self._found_image = None
        try:
            output = await self._recall(query, project)
        except Exception as exc:  # never leave the model waiting for a tool result
            self._log(f"[tool] recall failed: {exc!r}", stream=sys.stderr)
            output = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        if self._turn_serial != turn:
            # They spoke while this was in flight. Same treatment web_search gets, and for the
            # same reason: the Realtime API cannot withdraw a tool call, so a late answer is
            # reported as late rather than delivered out of the blue.
            output["stale"] = True
            output["note"] = (
                "They have said something since this was asked for. Answer them first, and "
                "only mention this if it is still what they want."
            )
        await self._send_tool_output(call.call_id, output)
        # The output first, so the call it answers is closed before anything else joins the
        # conversation, and the picture after it - an image cannot ride in a function_call_output,
        # so it has to be an item of its own. One response.create covers both. The same sequence
        # `_run_edit_photo` uses, and for the same reasons.
        found, self._found_image = self._found_image, None
        if found is not None:
            await self.add_found(found)
        await self._request_response()

    def _recall_scopes(self, project: str) -> set[str] | None:
        """Which corners of the card this query may look in, as ``recall.Item.scope`` values.

        This session and every project by default: "that one" almost always means either
        something from ten minutes ago or something filed under whatever is on the bench. Older
        sessions are indexed but not searched, because a year of half-finished conversations is
        mostly noise against a question about a project, and the project is where the finished
        version of anything ends up.
        """
        scopes: set[str] = set()
        if (live := session.current()) is not None:
            scopes.add(f"session:{live.dir.name}")
        if not self.settings.projects:
            return scopes or None
        from .projects import store

        catalog = store.catalog(self.settings)
        if project:
            found = store.find(catalog, project)
            if found is not None:
                return {f"project:{found.path.name}"} | scopes
            # A project name we do not recognise. Searching everything is better than searching
            # nothing: they said a name, it did not resolve, and the thing they want is probably
            # still on the card under a spelling neither of us guessed.
        scopes.update(f"project:{p.path.name}" for p in catalog)
        return scopes or None

    async def _recall(self, query: str, project: str) -> dict[str, Any]:
        """One search, and the panel if the answer is a picture."""
        index = await asyncio.to_thread(recall.load)
        if not len(index):
            return {
                "ok": True,
                "hits": 0,
                "note": (
                    "Nothing is indexed yet, so there is nothing to search. Say you cannot find "
                    "it rather than guessing at what they meant."
                ),
            }
        client = AsyncOpenAI(api_key=self.settings.api_key, timeout=RECALL_TIMEOUT_S, max_retries=0)
        try:
            vector = await recall.embed([query], client)
        finally:
            with contextlib.suppress(Exception):
                await client.close()

        scopes = await asyncio.to_thread(self._recall_scopes, project)
        hits = recall.rank(index, vector[0], scopes=scopes, limit=RECALL_HITS)
        hits = [hit for hit in hits if hit.score >= recall.MIN_SCORE]
        if not hits:
            return {
                "ok": True,
                "hits": 0,
                "note": (
                    "Nothing on the card matches that. Say so and offer to look another way - "
                    "do not describe the nearest thing as if it were what they asked for."
                ),
            }

        best = hits[0]
        # What else it could have been, for the model to offer aloud rather than to choose from -
        # the picture is already going up by the time this is read. Titles are trimmed because a
        # photo's title is its whole caption, which for a page of specifications runs to a
        # paragraph and would drown the result it is a footnote to.
        others = [
            {"title": _shorten(hit.item.title, MAX_OTHER_CHARS), "kind": hit.item.kind}
            for hit in hits[1:RECALL_OFFERED]
        ]
        if not best.item.showable:
            # `what`, not `kind`: session.note takes the record's own type as its first
            # parameter and that parameter is called kind, so passing one as a field is a
            # TypeError at the call rather than at import.
            session.note("recall", query=query[:120], title=best.item.title, what=best.item.kind)
            return {
                "ok": True,
                "hits": len(hits),
                "kind": best.item.kind,
                "title": best.item.title,
                "text": best.item.text,
                "others": others,
            }

        shown, seen = await asyncio.to_thread(self._show_found, Path(best.item.path))
        if shown:
            # It is the picture in front of them now, so it is the one edit_photo works on. The
            # path is the file on the card, not the downscaled copy that went to the panel: an
            # edit should start from the best pixels there are, not from the panel's 800x480.
            self._on_panel = Panel(Path(best.item.path), "found")
        session.note(
            "recall",
            query=query[:120],
            title=best.item.title,
            what=best.item.kind,  # never `kind`; see above
            file=Path(best.item.path).name,
            shown=shown,
        )
        result: dict[str, Any] = {
            "ok": True,
            "hits": len(hits),
            "kind": best.item.kind,
            "title": _shorten(best.item.title, MAX_OTHER_CHARS),
            "shown": shown,
            "others": others,
        }
        if others:
            result["note"] = (
                "The best match is on the panel. The others are near misses, not a menu - "
                "mention one only if what you showed looks like the wrong thing."
            )
        if not shown:
            result["note"] = (
                "It was found but there is no panel free to show it on. Say what it is out loud "
                "instead of pretending they can see it."
            )
        self._found_image = seen  # picked up by _run_recall once the tool output is away
        return result

    def _show_found(self, path: Path) -> tuple[bool, bytes | None]:
        """Put a picture already on the card onto the panel, and hand back what was shown.

        The bytes come back for the same reason :meth:`_keep_and_show_edit` returns them: the
        downscaled copy does two jobs, travelling to the panel and going to the model, and 1024 on
        the long edge is what ``webcam.MAX_EDGE`` hands the model for a real photograph anyway.

        ``imagine.for_panel`` returns its input untouched when it is already small enough, and
        ``panel.offer_image`` labels whatever it is given ``data:image/jpeg``. That pairing is
        safe for a photo and wrong for a small PNG - which is what a picture dropped into a project
        folder tends to be - so anything not already a JPEG is re-encoded first. A mislabelled data
        URL renders as nothing at all, silently, on the one screen nobody can see from here.
        """
        try:
            blob = path.read_bytes()
        except OSError as exc:
            self._log(f"[tool] could not read {path}: {exc}", stream=sys.stderr)
            return False, None
        if path.suffix.lower() not in {".jpg", ".jpeg"}:
            blob = imagine.as_jpeg(blob)
        small = imagine.for_panel(blob)
        return bool(panel.offer_image(small, path.stem) and panel.show()), small

    async def add_found(self, jpeg: bytes) -> None:
        """Show the model the picture it just put on the panel. No response is asked for here.

        The reason this exists rather than the caption being enough: the caption is *index text*,
        written by a small model to make the picture findable, and it is not reliable about detail.
        Measured on 2026-09-04, two passes over one identical manual page disagreed about the
        numbers printed on it. So the caption is allowed to find a photograph and never to describe
        one - if somebody asks what the torque figure actually is, the answer has to come off the
        pixels, and this is what puts the pixels where the model can read them.

        The same shape as :meth:`add_edit`, and not that method, because what it says about the
        picture is the opposite: an edit is a drawing that must not be read as evidence, and this
        is a photograph of their own bench that must be.
        """
        if not self.connected:
            return
        await self._send_item(
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "[The picture you just found, now on their screen. This is a real "
                            "photograph from their own card, so you may read it and answer "
                            "questions about what is in it - and prefer reading it to anything "
                            "the search said about it, which was written to find the picture "
                            "rather than to describe it accurately. They are looking at it too, "
                            "so do not narrate it unprompted.]"
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": "data:image/jpeg;base64,"
                        + base64.b64encode(jpeg).decode("ascii"),
                        "detail": "auto",
                    },
                ],
            }
        )
        self._log(f"[recall] showed the model {len(jpeg) // 1024} KB")

    # ---- imagined pictures ----

    async def _run_edit_photo(self, call: RealtimeConversationItemFunctionCall) -> None:
        """Redraw the picture on the panel with a change, keep it, and put it up.

        The same lifecycle as :meth:`_run_draw_diagram` - the module does the work, this does the
        bookkeeping - with one thing added that the diagram does not need. An edit can take a
        minute where a drawing takes ten seconds, which is long enough for the user to have moved
        on entirely, so it borrows :meth:`_run_web_search`'s staleness check: if they have spoken
        since this started, the model is told so and told not to launch into it.

        Like the diagram, the model is not told what the picture contains. Unlike the diagram it
        is told, out loud in the result, that it has not seen it - because this one *looks* like
        a photograph of their bench, and a model that forgets it is a drawing will start
        answering questions off it.
        """
        request = _tool_string(call.arguments, "request", imagine.MAX_REQUEST_CHARS)
        self._log(f"[tool] edit_photo {request!r}")
        if not request:
            await self._send_tool_output(call.call_id, {"ok": False, "error": "nothing described"})
            await self._request_response()
            return
        shot = self._on_panel
        if shot is None:
            await self._send_tool_output(call.call_id, {
                "ok": False,
                "error": "no photo to edit",
                "note": (
                    "They have not shown you a photo yet. Ask them to hit SNAP, in a few words."
                ),
            })
            await self._request_response()
            return
        if shot.path is None:
            # Nothing on the glass has a file behind it. This used to be the diagram case, back
            # when a drawing was a JSON spec with no jpg anywhere; a drawing is a picture in
            # photos/ now and edits like any other, which is why "make that clearer, drop the
            # LED" works. What is left here is the genuinely pathological case - a slot filled
            # with something that was never written down - and there is nothing to send.
            await self._send_tool_output(call.call_id, {
                "ok": False,
                "error": "what is on the panel was never kept",
                "note": "Say so in a few words and ask them to hit SNAP.",
            })
            await self._request_response()
            return

        turn = self._turn_serial  # if this moves while we render, they have moved on
        started = time.monotonic()
        self.drawing_active = True
        try:
            # Inside the try rather than an else, for the reason _run_draw_diagram's comment
            # gives: a failure in the bookkeeping must not leave the model waiting for a result.
            jpeg = await imagine.edit(shot.path, request, self.settings)
            output, seen, kept = await asyncio.to_thread(self._keep_and_show_edit, jpeg, request)
            self._log(f"[tool] edit: {len(jpeg) // 1024} KB in {time.monotonic() - started:.1f}s")
        except imagine.ImagineError as exc:
            session.note("photo", by="edit", request=request[:80], error=str(exc))
            self._log(f"[tool] edit failed: {exc}", stream=sys.stderr)
            output, seen, kept = {"ok": False, "error": str(exc)}, None, None
        except Exception as exc:  # never leave the model waiting for a tool result
            session.note("photo", by="edit", request=request[:80], error=f"{type(exc).__name__}")
            self._log(f"[tool] edit failed: {exc!r}", stream=sys.stderr)
            output, seen, kept = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, None, None
        finally:
            self.drawing_active = False

        if kept is not None:
            # The edit is the picture on the panel now, so it is what the next change starts
            # from: "now make it blue instead" means the one they are looking at. Set on the loop
            # from the path it was actually written to, and only when it was written - an edit
            # that reached no card is not a file anything can open again.
            self._on_panel = Panel(kept, "edit")

        if output.get("ok") and self._turn_serial != turn:
            output["stale"] = True
            output["note"] = (
                "They have spoken since this started, so it may no longer be what they want. "
                "The picture is on the panel: mention it in a few words if it still fits, and "
                "do not launch into it."
            )
        await self._send_tool_output(call.call_id, output)
        # The output first, so the call it answers is closed before anything else joins the
        # conversation, and the picture after it - an image cannot ride in a function_call_output,
        # so it has to be an item of its own. One response.create covers both.
        if seen is not None:
            await self.add_edit(seen, request)
        await self._request_response()

    def _keep_and_show_edit(
        self, jpeg: bytes, request: str
    ) -> tuple[dict[str, Any], bytes, Path | None]:
        """Write the picture down, ask for the panel, and hand back the copy to be shown.

        Written before it is shown, and shown whether or not writing worked - the same order and
        the same argument as :meth:`_keep_and_show`. The downscaled copy is made once and does
        two jobs: it is what travels to the panel, and it is what the model is shown. Both want
        the same thing - no more than 1024 on the long edge - and 1024 is exactly what
        ``webcam.MAX_EDGE`` hands the model for a real photograph, so the model sees an edit at
        the size it sees everything else.

        The third thing handed back is where the full-size copy landed, or None if it landed
        nowhere. This runs in a thread, and ``_run_edit_photo`` is the one that records it as the
        picture on the panel - on the loop, where the slot is read.
        """
        live = session.current()
        kept: imagine.Edit | None = None
        if live is not None:
            try:
                kept = imagine.write(jpeg, request, live.photos_dir)
            except OSError as exc:
                self._log(f"[tool] could not keep the edit: {exc}", stream=sys.stderr)

        small = imagine.for_panel(jpeg)
        shown = panel.offer_image(small, request) and panel.show()
        if kept is not None and kept.path is not None:
            # type "photo" and not a kind of its own: it is a jpg of their bench in the session's
            # photos/, so the transcript, the picture stream, the Media view and the projects
            # sweep all take it as read. `by` is the slot that already says who made a picture.
            session.note(
                "photo",
                by="edit",
                request=kept.request,
                file=f"{session.PHOTOS}/{kept.path.name}",
                bytes=kept.bytes,
                # Whether it reached the *panel*. Not `shown`, which on a photo record answers
                # the other question - whether Cyclops was shown it - and for an edit that is
                # always yes, panel or no panel.
                panel=shown,
            )
        # It is a drawing of their photo, not a photograph: that line is in the tool description,
        # in the item the picture arrives in, and here, because this is the one output in the
        # codebase where believing otherwise would have Cyclops reading measurements off fiction.
        if shown:
            note = (
                "It is on the panel and you are being shown it too. They are looking at the same "
                "picture, so do not narrate it unprompted - but you can see it perfectly well, "
                "so answer anything they ask about it, directly. Volunteer something only if it "
                "did not do what they asked, or if there is something in it worth flagging. It "
                "is a drawing of their photo, so never read a measurement off it or treat "
                "anything in it as a fact about their hardware."
            )
        else:
            note = (
                "You are being shown it, but there is no panel free to put it on, so they "
                "cannot see it. Say that, then describe it briefly - this is the one case where "
                "they have nothing to look at. It is a drawing of their photo, so never read a "
                "measurement off it or treat anything in it as a fact about their hardware."
            )
        return (
            {"ok": True, "request": request, "shown": shown, "note": note},
            small,
            kept.path if kept is not None else None,
        )

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
                # The tab names and their row counts, and not one value. Twenty tokens that tell
                # the model numbers exist here and roughly what kind, so it calls find_data at
                # the right moment instead of answering out of the prose - which is the whole
                # bargain of the workbook: two hundred values cost the conversation two lines.
                return {
                    "ok": True,
                    "project": project.name,
                    "notes": notes,
                    "data": projects.store.read_data(project).tabs(),
                }

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

    async def _run_data_tool(self, call: RealtimeConversationItemFunctionCall) -> None:
        """Save, find or forget one project's numbers. Local disk, and back in milliseconds.

        The same shape as the project tools above, and through a thread for the same reason. It
        used to show nothing on the panel on the grounds that it is over before a caption could
        render, which is true and is why :meth:`_run_tool` holds a finished job's line up for
        ACTIVITY_HOLD_S rather than dropping it the instant the work ends.
        """
        name = _tool_project(call.arguments)
        self._log(f"[tool] {call.name} {name!r}")
        if not name:
            output: dict[str, Any] = {
                "ok": False,
                "error": "no project name given",
                "note": "Ask them which project this is for. Do not guess.",
            }
        else:
            output = await asyncio.to_thread(self._data_call, call.name, name, call.arguments)
        if not output["ok"]:
            self._log(f"[tool] {call.name} failed: {output['error']}", stream=sys.stderr)
        await self._send_tool_output(call.call_id, output)
        await self._request_response()

    def _data_call(self, kind: str, name: str, arguments: str | None) -> dict[str, Any]:
        """The blocking half of the three data tools. Returns an answer, never raises.

        One lock spans read, change and write. Every tool call of a response is spawned as its
        own task, so two ``save_data`` calls off one photo genuinely do arrive here at once, and
        without this the second would write a book built before the first one's rows existed.
        """
        from .projects import store

        try:
            tracked = store.catalog(self.settings)
            project = store.find(tracked, name)
            if project is None:
                return {
                    "ok": False,
                    "error": f"nothing is being tracked called {name!r}",
                    "tracked": [p.name for p in tracked],
                    "note": "Ask which project they mean. Do not create one to hold a value.",
                }
            with store.data_held():
                book = store.read_data(project)
                if kind == "find_data":
                    return self._find_data(project, book, arguments)
                if kind == "save_data":
                    return self._save_data(project, book, arguments)
                return self._forget_data(project, book, arguments)
        except OSError as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        except Exception as exc:  # noqa: BLE001 - never leave the model waiting for a result
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def _find_data(self, project: Project, book: Book, arguments: str | None) -> dict[str, Any]:
        from .projects import data

        query = _tool_data_query(arguments)
        if not query:
            return {"ok": False, "error": "no query given"}
        hits = data.search(book, query)
        session.note(
            "data", action="found", project=project.name, query=query, hits=len(hits)
        )
        self._log(f"[tool] find_data {query!r} → {len(hits)} hit(s)")
        found = {
            "ok": True,
            "project": project.name,
            "query": query,
            "found": [row.as_dict() for row in hits],
        }
        if not hits:
            found["note"] = (
                "Nothing is written down for that. Say so plainly - do not offer a number from "
                "memory, and do not read out a value for something else instead."
            )
        return found

    def _save_data(self, project: Project, book: Book, arguments: str | None) -> dict[str, Any]:
        from .projects import data, store

        entries = _tool_entries(arguments)
        if not entries:
            return {"ok": False, "error": "no values given"}
        wanted = _tool_tab(arguments)
        tab = book.tab_named(wanted) or data.sheet_title(wanted)
        saved: list[str] = []
        replaced: list[str] = []
        for key, value, note in entries:
            (replaced if book.put(tab, key, value, note) else saved).append(key)
        store.write_data(project, book)
        session.note(
            "data",
            action="saved",
            project=project.name,
            tab=tab,
            keys=saved + replaced,
            replaced=len(replaced),
        )
        self._log(f"[tool] save_data {project.name!r}/{tab!r} ← {len(entries)} value(s)")
        return {
            "ok": True,
            "project": project.name,
            "tab": tab,
            "saved": saved,
            "replaced": replaced,
        }

    def _forget_data(self, project: Project, book: Book, arguments: str | None) -> dict[str, Any]:
        from .projects import store

        key = _tool_key(arguments)
        if not key:
            return {"ok": False, "error": "no key given"}
        matches = book.find_key(key, _tool_tab(arguments) or None)
        if not matches:
            return {
                "ok": False,
                "error": f"nothing is written down called {key!r}",
                "note": "Say so. Nothing was deleted.",
            }
        if len(matches) > 1:
            # A wrong delete is the only thing here that cannot be undone, so it never happens
            # on a guess. The candidates go back so the model can ask in the user's own words.
            return {
                "ok": False,
                "error": f"{len(matches)} values could be {key!r}",
                "candidates": [row.as_dict() for row in matches],
                "note": "Ask which one they mean. Nothing has been deleted.",
            }
        gone = matches[0]
        book.drop(gone)
        store.write_data(project, book)
        session.note(
            "data", action="forgot", project=project.name, tab=gone.tab, keys=[gone.key]
        )
        self._log(f"[tool] forget_data {project.name!r}/{gone.tab!r} ← {gone.key!r}")
        return {"ok": True, "project": project.name, "forgot": gone.as_dict()}

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


def _diagram_tools(settings: Settings) -> list[RealtimeFunctionToolParam]:
    """The drawing tool, or nothing. Left out rather than refused, as with the project tools.

    There used to be a second one, find_diagram, that put a drawing back on the panel from a
    folder of its own. A diagram is a photograph now, so recall does that job and this is one
    tool again.
    """
    return [DRAW_DIAGRAM_TOOL] if settings.diagrams else []


def _scratchpad_tools(settings: Settings) -> list[RealtimeFunctionToolParam]:
    """The one tool that writes on the panel, or nothing. Left out rather than refused, as ever.

    A flag of its own, and not because of what it costs - it costs nothing, which is the point of
    it. It is the switch to reach for if he turns out to put something on the glass every second
    sentence: this is the first tool here that takes the screen away from the person using it
    without being asked, and ``CYCLOPS_SCRATCHPAD=0`` is a way to find out what that is like that is
    not a revert.
    """
    return [SCRATCHPAD_TOOL] if settings.scratchpad else []


def _imagine_tools(settings: Settings) -> list[RealtimeFunctionToolParam]:
    """The one editing tool, or none. A flag of its own rather than a ride on ``diagrams``.

    They look like the same feature from the panel - something appears on the screen - but they
    are a different model at a materially different price per call, and "drawings yes, generated
    pictures no" is a position somebody may well hold. It is also the switch to reach for if the
    model ever starts answering wiring questions with a picture of a loom.
    """
    return [EDIT_PHOTO_TOOL] if settings.imagine else []


def _recall_tools(settings: Settings) -> list[RealtimeFunctionToolParam]:
    """The one recall tool, or none. Left out rather than refused, as with every other gate.

    Its index is kept by ``cyclops-index``, a service this process does not start and cannot see.
    Offering the tool anyway is the right call: a card with no index answers "nothing found",
    which is also the honest answer for a card with nothing on it, and the alternative is a
    setting that has to guess whether some other unit is running.
    """
    return [RECALL_TOOL] if settings.recall else []


def _project_tools(settings: Settings) -> list[RealtimeFunctionToolParam]:
    """The five project tools, or none of them. ``CYCLOPS_PROJECTS=0`` leaves them out entirely.

    Left out rather than offered and refused: a tool the model can see is a tool it will try, and
    being told "that is switched off" mid-conversation is worse than never being offered it.

    The three data tools ride on the same switch and need no flag of their own, because the
    workbook they write lives *inside* a project folder: with no projects there is nowhere to put
    a value and nothing to look one up in.
    """
    if not settings.projects:
        return []
    return [
        OPEN_PROJECT_TOOL,
        TRACK_PROJECT_TOOL,
        SAVE_DATA_TOOL,
        FIND_DATA_TOOL,
        FORGET_DATA_TOOL,
    ]


def _tool_name(arguments: str | None) -> str:
    """The project tools' required 'name' argument."""
    return _tool_string(arguments, "name", MAX_PROJECT_NAME_CHARS)


def _tool_description(arguments: str | None) -> str:
    """track_project's 'description' argument - the sentence that makes the project findable."""
    return _tool_string(arguments, "description", MAX_PROJECT_TAGLINE_CHARS)


def _tool_query(arguments: str | None) -> str:
    """The search tool's required 'query' argument."""
    return _tool_string(arguments, "query", MAX_QUERY_CHARS)


def _tool_project(arguments: str | None) -> str:
    """The data tools' required 'project' argument - which workbook this call is about."""
    return _tool_string(arguments, "project", MAX_PROJECT_NAME_CHARS)


def _tool_tab(arguments: str | None) -> str:
    """The sheet to file under. Required by save_data, optional for forget_data."""
    return _tool_string(arguments, "tab", MAX_DATA_TAB_CHARS)


def _tool_key(arguments: str | None) -> str:
    """forget_data's required 'key' argument."""
    return _tool_string(arguments, "key", MAX_DATA_KEY_CHARS)


def _tool_data_query(arguments: str | None) -> str:
    """find_data's required 'query' argument. Shorter cap than the web search: this is a label."""
    return _tool_string(arguments, "query", MAX_DATA_QUERY_CHARS)


def _tool_entries(arguments: str | None) -> list[tuple[str, str, str]]:
    """save_data's 'entries' argument, as (key, value, note) triples.

    The only structured argument any tool here takes, and parsed in exactly the spirit of
    :func:`_tool_string`: every malformed thing is dropped and nothing raises, so a model that
    sends one good row and one piece of nonsense still gets the good row written down. A row
    without a key is nonsense - there would be no way to ask for it back.
    """
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return []
    if not isinstance(args, dict) or not isinstance(raw := args.get("entries"), list):
        return []
    entries: list[tuple[str, str, str]] = []
    for item in raw[:MAX_DATA_ENTRIES]:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "")[:MAX_DATA_KEY_CHARS].strip()
        if not key:
            continue
        entries.append(
            (
                key,
                str(item.get("value") or "")[:MAX_DATA_VALUE_CHARS].strip(),
                str(item.get("note") or "")[:MAX_DATA_NOTE_CHARS].strip(),
            )
        )
    return entries


def _subject(text: str) -> str:
    """A tool's argument, cut down to something that still reads as part of a spoken phrase.

    Cut on a word boundary rather than mid-syllable, and with no marker left behind: the caption
    is elided again on the way to the panel, and two sets of trailing dots on one line - one for
    "there was more of this" and one for "this is still happening" - say nothing between them.
    """
    text = " ".join(text.split())
    if len(text) <= ACTIVITY_SUBJECT_CHARS:
        return text
    return text[:ACTIVITY_SUBJECT_CHARS].rsplit(" ", 1)[0]


def _shorten(text: str, limit: int) -> str:
    """One label, collapsed and cut on a word boundary. The same cut :func:`_subject` makes."""
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return (text[:limit].rsplit(" ", 1)[0] or text[:limit]) + "…"


def _phrase(verb: str, subject: str, bare: str) -> str:
    """*verb* applied to *subject*, or *bare* when the model sent nothing to name."""
    named = _subject(subject)
    return f"{verb} {named}…" if named else f"{bare}…"


def _activity_line(call: RealtimeConversationItemFunctionCall) -> str:
    """One clause for the panel about what this call is off to do.

    Written for someone glancing up from a bench rather than reading a log, so it names the
    subject and not the tool - "searching for the M8 torque", never "web_search" - and stays in
    the second person about the user's own things. Five of these had no way to reach the panel
    at all before, which is most of the point: a session could open a project, write six numbers
    down and look two more up while the strip said only LISTENING.

    Every line ends in an ellipsis. That is not decoration either - it is how the overlay tells a
    caption about work in flight from one about a state, and decides whether to walk its dots
    underneath it. See :data:`cyclops.overlay.BUSY_MARK`.
    """
    args = call.arguments
    if call.name == "web_search":
        return _phrase("searching for", _tool_query(args), "searching the web")
    if call.name == "draw_diagram":
        return _phrase("drawing", _tool_string(args, "request", MAX_QUERY_CHARS), "drawing")
    # Barely seen - the browser covers this strip about a second later - but the chain wants no
    # silent branches, and the recording is still watching while it is up.
    if call.name == "write_on_scratchpad":
        return "putting that on the screen…"
    if call.name == "edit_photo":
        return _phrase(
            "editing the picture to", _tool_string(args, "request", MAX_QUERY_CHARS),
            "editing the picture",
        )
    if call.name == "open_project":
        return _phrase("opening", _tool_name(args), "opening a project")
    if call.name == "track_project":
        return _phrase("starting to track", _tool_name(args), "starting a project")
    if call.name == "save_data":
        rows = len(_tool_entries(args))
        if not rows:
            return "writing that down…"
        return f"writing down {rows} value{'' if rows == 1 else 's'}…"
    if call.name == "find_data":
        return _phrase("looking up", _tool_data_query(args), "looking that up")
    if call.name == "recall":
        return _phrase("looking for", _tool_query(args), "looking through your things")
    if call.name == "forget_data":
        return _phrase("forgetting", _tool_key(args), "rubbing that out")
    return "working…"  # a tool the model invented; it still gets an answer, so it still gets a line
