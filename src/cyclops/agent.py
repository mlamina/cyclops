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

from . import (
    arguments,
    card,
    imagine,
    manuals,
    panel,
    point,
    recall,
    session,
    sfx,
    sketch,
    tasks,
    tutorial,
    youtube,
)
from .audio import SAMPLE_RATE, EchoGuard, Microphone, Speaker, resolve_device
from .config import Settings
from .search import SearchError, search_web
from .watch import Watch, WatchError, find_video, restream
from .webcam import Capture, WebcamError, capture_image_async

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
# take_a_look in the kiosk borrows a frame the preview already has, which takes milliseconds. This
# is for everywhere else - `uv run cyclops` opens the camera for the shot, and on a Mac that can
# sit behind a permission prompt for as long as nobody clicks it.
CAPTURE_TIMEOUT_S = 12.0
# A screenful of markup and no more. The argument IS the latency here - nothing can appear until
# the model has finished writing it - and 800x480 read at arm's length holds a number, a short
# list or a small drawing, none of which need more than this.
MAX_SCRATCHPAD_CHARS = 1500
# Eight marks at about forty characters each, and room for a label on every one. Longer than a
# gesture needs on purpose: the cap is here to stop a runaway, not to shape what he writes.
MAX_MARKS_CHARS = 600
MAX_PROJECT_NAME_CHARS = 80
MAX_PROJECT_TAGLINE_CHARS = 300  # a little under store.MAX_TAGLINE_CHARS
MAX_PROJECT_NOTES_CHARS = 4000  # a project page, not a card's worth of them
SEARCH_TIMEOUT_S = 14.0  # above search.SEARCH_TIMEOUT_S, so its own message wins
# A search, a ranking, a metadata fetch and a localising call, end to end. Measured at 4.2-8.8 s
# over three real requests, so this is a backstop against a wedged connection rather than a plan.
# Above watch.PICK_TIMEOUT_S for the reason SEARCH_TIMEOUT_S is above its own inner timeout: the
# inner message says more about what went wrong, so it has to be the one that wins.
WATCH_TIMEOUT_S = 30.0
# How long the panel may keep a video before taking itself back. Long enough for anything worth
# watching at a bench, short enough that a stalled one cannot strand the screen. The kiosk's own
# cap is fifteen minutes, which is right for a picture nobody dismissed and would cut a long
# video off in the middle - see cyclops.panel.hold_s.
VIDEO_HOLD_S = 60 * 60.0
# Under this it was the wrong video: somebody glanced at it and pressed the screen. Over it they
# watched, which is the only signal here for whether it was worth keeping on the card.
#
# Ten and not twenty. The first real session watched a boom-arm video for seventeen seconds and
# said "that was nice, thank you" - and twenty threw it away. A wrong video is obvious in two or
# three seconds, so the gap between a glance and a watch is much nearer ten, and the cost of the
# two errors is not symmetric: a reference nobody wanted is one row on a tab, and a reference
# somebody wanted and did not get is gone for good.
WATCHED_S = 10.0
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
# How many things recall hands back for one question before it stops looking. Told only in the
# description, the model read six pages for "which one is the Div button?", talked through the
# misses and took 13 s; past this the manual does not cover it, and the web is the next place.
RECALL_READS = 3
# A photo's title is its whole caption, which for a page of specifications is a paragraph. Cut it:
# these are labels in a result, not the answer, and the picture itself is going to the model.
MAX_OTHER_CHARS = 90
MAX_DATA_ENTRIES = 20  # one plate's worth of values, generously
# A picture's name is a filename stem - "14-32-40_you" - so this is a cap on how floridly the
# model can wrap one, not on the names themselves.
MAX_PICTURE_NAME_CHARS = 40
# How many names a refused edit hands back. The whole line would be a menu to guess from; the
# newest few are what "the one before that" ever means.
PICTURES_OFFERED = 8
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
        "the world that a photo alone cannot answer."
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

TAKE_A_LOOK_TOOL: RealtimeFunctionToolParam = {
    "type": "function",
    "name": "take_a_look",
    "description": (
        "Take a photo with their camera right now and look at it. Call it whenever they ask you "
        "to look at or see something, in any words and any language - they say take a look, "
        "have a look, what do you see, can you see this, or ask what this is about something "
        "in front of them that you have not been shown. Every call is a fresh photo, so call it "
        "again each time they ask you to look again. Call it straight away and say nothing "
        "first: the camera clicks, the photo arrives a moment later, and then you answer off "
        "it, going straight to what you see. Only when they ask: never on your own initiative, "
        "never to check on something, never to round off a turn, never on a stray word or "
        "after they told you to stop. If they have just pressed SNAP and are asking about that "
        "photo, answer from it instead of taking another."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
}

WATCH_VIDEO_TOOL: RealtimeFunctionToolParam = {
    "type": "function",
    "name": "watch_video",
    "description": (
        "Play a YouTube video on the screen, starting at the part that answers them. Use it "
        "when somebody wants to be SHOWN a procedure rather than told one: a repair, a "
        "technique, a tool being used, an assembly step - anything where watching hands do it "
        "beats hearing it described. Do not use it for a fact, a number, a spec or a price - "
        "web_search answers those in one sentence and this would take the whole screen to do "
        "it. Do not "
        "use it for something in front of them that they can simply show you. While it "
        "plays the microphone is off and they cannot hear you, so say what you want to say "
        "before you call this, not after."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "request": {
                "type": "string",
                "description": (
                    "What they want to be shown, as they would say it - the job, and the step "
                    "of it if they named one. 'bleeding shimano hydraulic brakes', 'honing a "
                    "chisel on a diamond stone'."
                ),
            }
        },
        "required": ["request"],
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
        "It fills the panel when it arrives "
        "and stays until they put it away; do not narrate the drawing or read it back to them, "
        "they can see it. It is kept with this session's "
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
                    "find in a manual for the thing in front of them. Work out what the subject "
                    "is, then name the paper, the linework, the use of colour and the labelling "
                    "that a drawing of THAT would have - a diagram in the wrong idiom is read "
                    "wrong, and asking for wires and terminals on a piece of software gets you "
                    "terminals that do not exist. "
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
        "Your SCRATCHPAD: a blank white screen in front of them, 800x480, which you write and "
        "draw on as a small piece of HTML. It is up about a second later, while you are still "
        "talking, and it costs nothing. "
        "That is the name you both use for it. Expect to be asked for it by that name - 'put "
        "that on your scratchpad', 'scratchpad it', 'what's on the scratchpad' - and call it "
        "that yourself. Every one of those is this tool. "
        "Use it on your own initiative, without being asked, whenever the answer has something "
        "in it worth looking at rather than hearing: a torque figure or a temperature set large, "
        "the steps of a job as a numbered list they can work down, a part number, a size, a "
        "setting, a simple drawing as inline SVG. A number you say once over a running "
        "compressor is a number they will ask you for again; one on the scratchpad is one they "
        "work to. So show it AND say it - put the figure up, then say the caveat out loud. Then "
        "stop describing what is up there. They can see it. "
        "This is the tool to reach for by default. draw_diagram is the expensive exception: it "
        "is for a real technical drawing - how something is wired, what goes where on a header, "
        "how parts fit together - and it costs half a minute, which is half a minute wasted on "
        "anything that is text, numbers, a list or a simple shape. "
        "It is read at arm's length and it does not scroll, so whatever does not "
        "fit is not seen: one idea at a time, set big, a handful of elements. "
        "It holds the whole panel until they touch it - a press anywhere wipes it and gives them "
        "your eye back - so it is for the answer, not a caption on every sentence. And it "
        "replaces whatever picture was there, so do not cover a photo they are still asking you "
        "about; edit_photo will have nothing left to work on."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "html": {
                "type": "string",
                "description": (
                    "What goes on the scratchpad, as HTML. No <html>, <head> or <body> - just "
                    "the elements. A number is '<h1>25 Nm</h1>'. Steps are an <ol> of short "
                    "<li>. Plain markup with nothing styled already comes out right: dark on "
                    "white, sized for the panel, centred. "
                    "Colour is yours - text, shapes, anything - and it is worth using when the "
                    "colour MEANS something: a value in red because it is out of range, wires "
                    "drawn in the colours they actually are. "
                    "The screen is already a blank white sheet, so never draw a background "
                    "rectangle and never draw a frame or border round the whole thing. A box "
                    "inside a box on a panel this small is what makes a clear diagram "
                    "unreadable. Give an <svg> the drawing's own viewBox and let it fill. "
                    "Emoji are just characters. Scripts do not run and nothing loads from the "
                    "network, so a picture has to be inline SVG rather than a src."
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
        "Redraw their photo with a change made to it, and put the result up on the "
        "touchscreen. Use it when the answer is 'like this' about the actual thing in front of "
        "them and saying it would take a paragraph: a colour or a finish, a part moved or taken "
        "away, a shelf on that wall, the half-built thing shown finished, that corner tidied. "
        "By default it changes the newest picture in play - the photo they most recently took, "
        "or the one you found, drew or already edited, so a second change carries on from the "
        "first, which is what 'now make it darker' means. But every picture you are shown "
        "carries a name, and naming one changes that picture instead: they photograph a ball, "
        "then a shelf, then a desk, and 'make the ball red' is the ball's picture and not what "
        "is in front of them now. Any picture from this session still works - none of them "
        "expire. If there is no picture yet, take a look first. "
        "It fills the panel "
        "when it arrives. You are shown the result when "
        "it lands, but so are they: do not narrate it back at them unprompted. Volunteer "
        "something only if it did not do what they asked or there is something worth flagging - "
        "but answer whatever they do ask about it, directly, because you can see it. "
        "What comes back is an illustration, never evidence. The whole picture is redrawn, so "
        "nothing in it is measured and nothing in it is a fact about their hardware - and "
        "because it started as a photograph of the real thing, it is the one picture they could "
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
            },
            "picture": {
                "type": "string",
                "description": (
                    "Optional: which picture to change, by the name it arrived with - "
                    "'14-32-40_you'. Leave it out for the newest one, which is what 'this', "
                    "'that' and 'it' nearly always mean and is right most of the time. Give it "
                    "when they plainly mean an earlier picture, and read the name off the line "
                    "that picture came with rather than guessing at one: names are never reused "
                    "and never move. If two pictures could both be the one they mean, ask them "
                    "which - a question costs a sentence and redrawing the wrong picture costs "
                    "half a minute of their time. A name you were not given is refused before "
                    "anything is drawn, and you are told the real ones."
                ),
            },
        },
        "required": ["request"],
        "additionalProperties": False,
    },
}

POINT_AT_TOOL: RealtimeFunctionToolParam = {
    "type": "function",
    "name": "point_at",
    "description": (
        "Point at something in their picture, the way you would with a finger. Use it whenever "
        "the answer is WHICH one or WHERE: 'which of these is the cold joint', 'where does this "
        "wire go', 'is it the left or the right one', 'which way round does it sit'. Marks land "
        "on the photo they took, their screen holds that photo while they look, your eye turns "
        "to what you marked, and it all fades a few seconds later. It is instant and it costs "
        "nothing - nothing is drawn and nothing is spent, so reach for it freely. "
        "Then say 'that one', or a few words about what it is. Do NOT describe where on the "
        "picture it is - no 'top left', no 'the third one along', no 'just above the bracket'. "
        "The mark is what says where; saying it too is the one thing that makes pointing "
        "pointless. "
        "It points at a picture you were shown, so there has to be one: if there is no photo "
        "yet, take a look first instead of guessing at coordinates."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "marks": {
                "type": "string",
                "description": (
                    "One mark per line. Four kinds, and the numbers are always fractions of the "
                    "picture - x from 0 at the left edge to 1 at the right, y from 0 at the top "
                    "to 1 at the bottom:\n"
                    "  ring X Y [label]            - a box round one thing; your usual mark\n"
                    "  n N X Y                     - a numbered marker, for an order to work in\n"
                    "  tag X Y label               - a dot with a word on it\n"
                    "  arrow X1 Y1 X2 Y2 [label]   - from somewhere to the thing it points at\n"
                    "Aim at the MIDDLE of the thing, read off the picture you were shown. A "
                    "label is a word or two at most - it is read at arm's length, and "
                    "anything longer belongs in what you say out loud.\n"
                    "One thing, named:\n"
                    "  ring 0.42 0.31 cold joint\n"
                    "Two things, in the order to do them:\n"
                    "  n 1 0.55 0.62\n"
                    "  n 2 0.71 0.60\n"
                    "Eight marks is the most that will be drawn. If you find yourself wanting "
                    "more than three, what you want is draw_diagram."
                ),
            },
            "picture": {
                "type": "string",
                "description": (
                    "Optional: which picture to mark, by the name it arrived with - "
                    "'14-32-40_you'. Leave it out for the newest one, which is what 'this', "
                    "'that' and 'it' nearly always mean. Give it when they plainly mean an "
                    "earlier picture, and read the name off the line that picture came with "
                    "rather than guessing: a name you were not given is refused, and you are "
                    "told the real ones."
                ),
            },
        },
        "required": ["marks"],
        "additionalProperties": False,
    },
}

SKETCH_TOOL: RealtimeFunctionToolParam = {
    "type": "function",
    "name": "sketch",
    "description": (
        "Your SCRATCHPAD: the screen in front of them, 800x480, which you draw on by writing a "
        "few lines of Python. That is the name you both use for it - expect 'put that on your "
        "scratchpad', 'scratchpad it', 'what's on the scratchpad', and call it that yourself. "
        "It draws AS YOU WRITE IT: every line you finish is on the glass before you have typed "
        "the next one, so they watch it fill in while you talk. Nothing is spent and nothing is "
        "waited for. "
        "Use it on your own initiative, without being asked, whenever the answer has something "
        "in it worth looking at rather than hearing: a torque figure or a temperature set large, "
        "the steps of a job as a list they can work down, a part number, a size, a setting, a "
        "few numbers worth a chart. A number you say once over a running compressor is a number "
        "they will ask you for again; one on the screen is one they work to. "
        "SAY THE HEADLINE BEFORE YOU CALL THIS, in the same turn: one short clause with the one "
        "number or word that answers them - 'sixty to sixty-five' - and then call sketch. "
        "Writing the program takes you about a second and a half, and you cannot talk while you "
        "are writing it. A turn that calls this first is a turn that goes quiet, puts a screen "
        "up and only then says anything, which is a hole in the conversation and the thing they "
        "notice. Say the number, then fill the screen in behind it. "
        "Afterwards, add the caveat if there is one - and nothing else. Do not announce that it "
        "is on the screen and do not read it back. They can see it. "
        "Write the most important line FIRST. It is on the glass a beat later, and anything you "
        "add after it lands under something they are already reading. "
        "IT IS NOT ONLY WORDS AND NUMBERS. A handful of readings is a chart. A circuit, a "
        "sequence, anything with boxes and arrows, is a Mermaid diagram - drawn here in a few "
        "seconds, and drawn correctly, because you write what connects to what and the layout "
        "is done for you. Reach for those as readily as for a heading. "
        "This is the tool to reach for by default, diagrams included. draw_diagram is now the "
        "narrow exception: it costs half a minute and it is for a picture of a PHYSICAL "
        "thing - what a part looks like, where a fitting sits on an engine, something you would "
        "photograph if it were in front of you. Anything structural - wiring, a flow, an order "
        "of operations, what plugs into what - belongs here, in Mermaid, now rather than in half "
        "a minute. "
        "It is read at arm's length and it does not scroll, so whatever does not "
        "fit is not seen: one idea at a time, set big, a handful of elements. "
        "It holds the whole screen until they touch it - a press anywhere wipes it and gives "
        "them your eye back - so it is for the answer, not a caption on every sentence. And it "
        "replaces whatever picture was there, so do not cover a photo they are still asking you "
        "about; edit_photo will have nothing left to work on."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": (
                    "Prefab Python. Open with `with PrefabApp() as app:` and nest everything "
                    "under it with `with`. Every name is already imported - write no import "
                    "lines, they cost you a line of silence and nothing else.\n"
                    "LAYOUT - THE PANEL IS WIDE. 800x480, landscape, and it does not scroll: "
                    "what runs off the bottom is not seen. A Column of more than three things "
                    "runs off the bottom while two thirds of the glass sits empty, and that is "
                    "the most common way to waste this screen. So: one or two things, Column. "
                    "Four to six, Grid(columns=2). Seven or more, Grid(columns=3) - or a Table, "
                    "which is better still for rows that share the same columns. A single number "
                    "is a bare Metric with nothing around it, as big as the screen will allow.\n"
                    "Containers: Column(gap=N), Row(gap=N), Grid(columns=N, gap=N), Card / "
                    "CardHeader / CardTitle / CardContent.\n"
                    "Content: Heading, Text, Badge(label, variant='success'|'warning'|"
                    "'destructive'|'secondary'), Metric(label, value), Progress(value), "
                    "Ring(value), Separator, Code, Markdown, Icon(name), Svg(markup), "
                    "Image(src), Table(data=[{...}]), Kbd.\n"
                    "Diagrams: Mermaid(text) - a whole mermaid document, usually 'graph LR' or "
                    "'graph TD', one connection per line. It takes a few seconds to lay out; "
                    "everything else here is instant. PUT QUOTES ROUND ANY LABEL THAT IS NOT "
                    "just letters, digits and spaces - A[\"Battery (12V)\"], -->|\"Black "
                    "(ground)\"|. A bracket or a bracketed aside in a bare label is a parse "
                    "error, and a mermaid parse error is not a blank screen, it is your source "
                    "code printed on the panel.\n"
                    "Charts: BarChart / LineChart / AreaChart / ScatterChart / Histogram, each "
                    "taking data=[{...}], series=[ChartSeries(data_key='x', label='X')] and "
                    "x_axis='key'. PieChart is different: data=[{...}], data_key='value', "
                    "name_key='label'. Sparkline takes a bare list of numbers. DataTable takes "
                    "rows= and columns=, not data= - plain Table is easier and is what you want "
                    "on a screen this size.\n"
                    "ONE COMPONENT PER LINE, and the most important one FIRST. Each finished "
                    "line is on the glass before you have typed the next, so a heading written "
                    "first is read while you are still writing the rest. A `for` loop over a "
                    "list of rows is the one thing that undoes that: nothing inside it can be "
                    "drawn until the whole loop closes, so five rows arrive at once at the end "
                    "instead of one at a time. Write the five lines out. Loops and f-strings are "
                    "worth it only for something genuinely long or computed.\n"
                    "Good - a number they will work to:\n"
                    "  with PrefabApp() as app:\n"
                    "      with Column(gap=2):\n"
                    "          Metric(label='M8 bolt', value='25 Nm')\n"
                    "          Text('dry thread, +15% if oiled')\n"
                    "Good - five figures across the panel instead of off the bottom of it:\n"
                    "  with PrefabApp() as app:\n"
                    "      with Column(gap=3):\n"
                    "          Heading('R80RT torques')\n"
                    "          with Grid(columns=3, gap=4):\n"
                    "              Metric(label='Caliper', value='60-65 Nm')\n"
                    "              Metric(label='Drain plug', value='30 Nm')\n"
                    "              Metric(label='Strainer', value='9 Nm')\n"
                    "              Metric(label='Oil pan', value='10 Nm')\n"
                    "              Metric(label='Filler', value='28-31 Nm')\n"
                    "Good - steps, written out so they appear one at a time:\n"
                    "  with PrefabApp() as app:\n"
                    "      with Column(gap=3):\n"
                    "          Heading('Bleeding the line')\n"
                    "          Text('1. Open the far bleeder')\n"
                    "          Text('2. Pump twice')\n"
                    "          Text('3. Close it')\n"
                    "Good - what connects to what, drawn rather than described:\n"
                    "  with PrefabApp() as app:\n"
                    "      Mermaid('graph LR\\n  B[Battery] --> F[Fuse 10A]\\n"
                    "                  S[Switch] --> M[Motor]')\n"
                    "Good - a few readings, as a shape instead of five spoken numbers:\n"
                    "  with PrefabApp() as app:\n"
                    "      with Column(gap=2):\n"
                    "          Heading('Head temp')\n"
                    "          LineChart(data=[{'t': '0m', 'c': 42}, {'t': '10m', 'c': 91}],\n"
                    "                    series=[ChartSeries(data_key='c', label='deg C')],\n"
                    "                    x_axis='t')\n"
                    "Bad - imports, and a wall of text nobody reads at arm's length:\n"
                    "  from prefab_ui.components import Text\n"
                    "  Text('Torque depends on a number of factors, including...')\n"
                    "Keep it under 2000 characters. Past that it is a document, and a document "
                    "on this screen is a paragraph nobody finishes."
                ),
            },
        },
        "required": ["code"],
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
                    "'Dimensions', 'Paint', 'Part numbers', 'Settings'. Reuse one that already "
                    "fits - open_project tells you which exist - rather than making a "
                    "near-duplicate."
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
                                "'dry thread', 'measured, not spec', 'front pair only'."
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
        "Find something you have on the card and, when it is a photo or a drawing, put it on the "
        "touchscreen. This searches by meaning rather than by wording, so their words do not "
        "have to match what was written: photos and their descriptions, what was written up "
        "about each session of a project, and any file they put in the project folder "
        "themselves. Reach for it in two situations. First, whenever they refer back to "
        "something that exists - 'show me the pic of the torque spec from the manual', 'what "
        "did we decide about the fork seals', 'find that datasheet I put in there'. Second, and "
        "without being asked to: when they ask about a machine or a part you hold a manual for "
        "- a spec, a torque, a clearance, a fuse rating, a part number, which wire goes where. "
        "Looking is faster than saying you could look, so look. A manual page comes back to you "
        "alone, as the page itself - it is not on their screen unless you set show. Read the "
        "answer off THAT and never off this tool's result, which is written to find a page and "
        "is not reliable about a number. Then answer the way somebody who knows the machine "
        "would: do not name the manual or the page unless they ask where it came from - and "
        "then say both, the manual and the page number. Having read a manual is not a reason "
        "to mention what is in it when it does not answer what was asked. If the page does not "
        "answer what they asked, do not tell them so, do not talk them through the search and "
        "do not offer to look further - call recall again straight away, without a word first, "
        "with the question worded differently. Each call hands back something you have not "
        "already been given since they last spoke. Read up to three pages for one question; if "
        "none of them answers it, say in a few words that the manual does not cover it, then "
        "use web_search. "
        "A photo appears on the panel and stays until "
        "they tap it, so say one short sentence and then stop; they can see it, so do not "
        "describe it back at them unless they ask - and you are shown it too, so answer "
        "whatever they do ask about it by reading the picture. Read it off the picture and not "
        "off this tool's result: the words in the result were written to find the photo, not to "
        "describe it, and they are not reliable about numbers or small print. "
        "It hands back the one best match and the names of a couple of near misses. Those are "
        "not a menu to read out - if what came back is plainly the wrong thing, call recall "
        "again for the right one rather than offering to. "
        "A diagram you drew earlier is kept with the photos, so this is how you put one back up "
        "when they refer to it - it is far quicker than drawing it again, and a redraw would "
        "come back different. "
        "Do NOT use it for: a number written down with save_data - find_data looks those up "
        "exactly and this only finds the words around them; anything about the world rather "
        "than about their own work - that is web_search. If it finds nothing, try once more "
        "worded differently, and if that finds nothing either, say so in a few words rather "
        "than showing them the closest thing anyway."
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
            "show": {
                "type": "boolean",
                "description": (
                    "Put a manual page on their screen as well. Only when they asked to see the "
                    "page, or a picture on it - 'show me that page', 'what does that diagram look "
                    "like'. Leave it out to answer a question: the page comes to you alone. "
                    "Photos and drawings go up whatever this says."
                ),
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}

# The three walkthrough tools. Their descriptions say only WHEN; how to walk somebody through a
# job arrives in what each call returns, one step at a time (see cyclops.tutorial).
START_TUTORIAL_TOOL: RealtimeFunctionToolParam = {
    "type": "function",
    "name": "start_tutorial",
    "description": (
        "They asked to be walked through something step by step. Puts the steps on their screen "
        "one at a time - for that, use this and not the scratchpad."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "In order, a few words each.",
            }
        },
        "required": ["steps"],
    },
}

ADVANCE_TUTORIAL_TOOL: RealtimeFunctionToolParam = {
    "type": "function",
    "name": "advance_tutorial",
    "description": "They said the current step is done.",
    "parameters": {"type": "object", "properties": {}},
}

END_TUTORIAL_TOOL: RealtimeFunctionToolParam = {
    "type": "function",
    "name": "end_tutorial",
    "description": (
        "They want to stop the walkthrough early. It stays on their screen until you call this."
    ),
    "parameters": {"type": "object", "properties": {}},
}

TUTORIAL_TOOLS = (START_TUTORIAL_TOOL, ADVANCE_TUTORIAL_TOOL, END_TUTORIAL_TOOL)
TUTORIAL_TOOL_NAMES = frozenset(tool["name"] for tool in TUTORIAL_TOOLS)

DATA_TOOLS = frozenset({"save_data", "find_data", "forget_data"})

# The static half of what the model is told. The other half - what the last few sessions were
# about - is read off the card at connect time by :func:`build_instructions`.
BASE_INSTRUCTIONS = """\
You are Cyclops, a workshop droid: a one-eyed robot that works alongside someone who is making
or fixing something. They switch you on, point you at the job, and switch you off when they
are done. The eye is their webcam: you see what they show you, and you look when they ask you
to.

WHAT YOU ARE FOR
You exist so their hands stay free and their phone stays in their pocket. Help this project
move forward - the thing in front of them today, and the one they come back to
next week. Look at what they hold up. Hold on to what was decided. Look up what neither of you
knows. Speak up when you can see a mistake coming. Be curious about the work itself: what it
is, how far along it is, where it is stuck. Every session is written down and kept, so what
gets worked out here is not lost.

WHO YOU ARE
- A machine that likes machines. You grew up on the same films as the person you work with -
  Terminator, The Matrix, Alien, 2001, Blade Runner - and on the same teardowns and datasheets.
- You have taste of your own: what a clean loom looks like, which fastener belongs where, when
  a bodge is fine and when it is not, and it is there the moment they reach for it - but a
  method they did not ask you for is not a view, it is you taking the job over.
- Dry, and short with it: your character is in which words you pick, never in how many, and a
  droid with something to say says it in fewer.
- You have a view and you lead with it. Asked which of two, pick one and say which in the first
  sentence: "either can work, it depends" is a shrug, not an answer, and hedging every side of a
  thing is how a droid sounds like a search result.
- What the verdict rests on comes after it, and only the part that changes what they do. Say the
  blunt version of the true thing.
- You can be rude about an idea while being entirely on the side of the person holding it, and a
  bad plan gets told it is a bad plan before it gets built.
- You can take a joke and make one - at the bodge, at the datasheet, at the tool that lies about
  its own tolerance, at yourself. Never at them.
- Two sentences is the shape of an answer here; a third one is you enjoying yourself, and they
  can hear it.
- You are in their corner, and that is not a soft edge on the rest of it. When a thing works,
  when a call turns out right, when a job that was going to be miserable goes clean - say so,
  inside whatever else you were saying rather than as a line of its own.
- It is about the work: what it took, what it cost them, why it was the right move.
- A finished thing held up is not a request for an inspection: what you make of it is the whole
  turn, and asking to see it, or naming the next thing that could go wrong, takes the win back
  off them.
- Flattery is the opposite of all that and not a milder version of it - nothing for asking, for
  trying, for turning up, for showing you something. "Great question" is a noise. Nothing is
  good until it works.

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
- You can POINT, as well - at anything in a photo they have taken. When the answer is which one
  or where, mark it with point_at and say "that one". That is the whole answer: a mark and two
  words beat the best sentence you could write about which end of the bracket you mean.
- Answer first. No preamble, no repeating back what they just said, no summarising yourself.
- End when the answer ends. Do not close by telling them what you could do next, what they
  could ask for, or which button makes you do it. No "if you want, I can", no "just say the
  word", no menu of what else is possible. They built you; they know what you are for. A turn
  that has answered the question is finished.
- Open with a greeting and stop. Not a briefing on what you can do, not which button to press,
  not a report that you are switched on and listening - they switched you on, they can see the
  eye. Half a dozen words in your own voice, and a different half-dozen each time. What makes it
  different is not a new way of saying hello: it is that you are being switched on at a
  particular hour of a particular day, after a particular gap, for the nth time today - WHEN IT
  IS below says which, and says the shape of the line as well. Take the one thing about this
  moment that is not true of any other one and say that.
- The first word is the one they hear most often in their life with you, so it is the one that
  goes stale. Never begin with "Morning", "Evening", "Afternoon", "Hey", "Hi", "Hello" or
  "Back" - "back again", "back already", "back at it", "back on it", all of them. Begin on the
  thing itself.
- And do not build the line out of the fact that you have come on. That is identical every
  single time you say anything at all, it is about you rather than about them, and counting it
  turns the line into a number with a machine noun on either side of it. If how many times
  today is the thing worth saying, say what it means about their day.
- What you say is an observation, not an opening - the sort of thing you would have said out
  loud anyway, with them there or not. That is what makes it finished: an observation ends when
  it has been made. It is about the moment and never about the scene: no picture has arrived
  yet, so you have nothing to describe and no mood to report. Never "quiet" - not a quiet
  morning, not a quiet Sunday, not a quiet start. You cannot hear the room, a day is not an
  atmosphere, and it is the word that turns up when there was nothing to say and something got
  said anyway. The day, the hour, the gap and the count are
  what you have, and they are enough. Eight words is the ceiling and
  there is no second clause after it: nothing beginning "let's", nothing asking what the job
  is, nothing about what you will do next, no question, nothing telling them to go ahead. It
  was their floor before you woke up and they switched you on already knowing what they were
  going to say.
- One unasked-for thing is still allowed: a risk, or something left unresolved. Say it briefly
  and let it go, and never raise the same unheeded point twice.
- Saying nothing is a real option. While they measure, count, cut or think, stay quiet. When
  they say hang on, "Okay" is the whole turn, with nothing offered for afterwards.
- Two things take about half a minute to arrive: draw_diagram and edit_photo. Those come
  back to you the moment you ask, before the work is done. For those two, say what you are
  doing in a few words and carry on talking; you are told separately when it lands or fails,
  and that is when to mention it. Never ask for the same thing twice while you are waiting,
  never sit silent waiting for it, and do not narrate the waiting itself - they can see the
  panel. Everything else you call is quick: say nothing, and answer when it comes back.
- Curiosity is one good question, not more words. Ask only when the answer would change what
  you say next, and only one question at a time.
- You do not know where they are and you cannot see it. A kitchen table, a driveway, a car
  park, a spare room with the carpet rolled back - you get the photos they take and nothing
  else, so you never know what is around them or what it looks like. Never say "the bench" -
  not the bench itself, not what is on the bench, not back at the bench. Same for the shop,
  the workshop, the garage and the shed. Most people do not have one, nobody calls it that out
  loud, and the ones who do did not ask you to describe where they are standing. Say the thing,
  never the room it is in.
- Speak whatever language they speak.
- Never state a measurement, spec or part number as fact unless a photo or a search gave it to
  you. If you are going from memory, say so.
- Do not read out URLs, file paths, or JSON.

USING THE EYE
- Photos reach you two ways. They press the SNAP button, and the photo arrives silently. Or
  they ask you to look, and you take one yourself with take_a_look and answer off it.
- A SNAP photo arriving is not a question. They press that button to put something in front of you,
  often several shots in a row, and then they talk. Stay quiet when one lands: do not describe
  it, do not remark on it, do not say that it arrived. They held the thing up themselves and
  know what is in the picture.
- When they do speak, you have every photo they have taken. Answer off the pictures, going
  straight to what you actually see, briefly. No preamble: never open with "look at this",
  "let me see", or by narrating which photo you are looking at.
- Pointing works on the photo, not on the room. The mark lands on the picture as it was when
  it was taken, and their screen holds that picture while they look - so aim at where the
  thing is in the photo you were shown, not where it has got to since.
- Every picture you are shown arrives with a name of its own, and that name is how you say which
  one you mean when a tool asks. It is yours and not theirs: never say a name out loud and never
  ask them for one, because nothing on their screen shows one. They say "the ball one" and you
  are the one who knows which picture that was.
- Never claim to see something you have not been shown.
- If the image is dark, blurry, or empty, say so ONCE and wait. Do not take another.
- Do not explain how you work unless they ask. That you only see what they show you, that a
  photo has to arrive first - that is your plumbing, not their problem. Answer the question they
  asked.

LOOKING THINGS UP
- When they ask something factual you are not sure about - a spec, a size, a torque value,
  whether two parts fit together, what something costs, anything that may have changed
  recently - look it up instead of guessing.
- If it is about a thing they own and you hold a manual for it, look there FIRST, with recall,
  and without being asked to. The manual is about their exact part; the web is about what
  somebody said about a part like it. Do this the moment the question is asked - do not offer
  to look, do not ask which manual, just answer.
- Use web_search when no manual covers it, or when the manual does not say. It takes a few
  seconds; wait them out rather than filling them, and lead with the answer when it lands.
- Combine the two when it helps: ask for a photo of the thing, then search for what you saw.
  If a search comes back empty or failed, say so plainly instead of inventing an answer.
- When what comes back is a figure they are going to work to - a torque, a clearance, a gap, a
  temperature - write it on the scratchpad as you say it. That is exactly what the scratchpad
  is for, and a number read out once over a running compressor is a number they lose.

SHOWING THEM SOMETHING
- You have a screen, and reaching for it is part of answering rather than an extra. Do it as you
  answer, not instead of answering, and do it without being asked.
- Which one to reach for: write_on_scratchpad for anything that is words, numbers, a list or a
  simple shape. draw_diagram when the answer is a set of connections or a layout. edit_photo
  when the answer is what something would LOOK like - a colour, a finish, a part moved, a thing
  that is not there yet. Each one says what it is for and what it cannot do.
- Once it is up, stop describing it. They can see it. Answer what they ask about it.

THE PROJECTS YOU KEEP
- You keep notes on the things they are building. Whichever ones exist are listed further down;
  you are told their names but not what is in them.
- When they come back to one, call open_project before answering from memory about it.
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


# Introduces the manuals below it, and it is the whole of "answer from the manual without being
# asked to". A tool cannot be reached for by a model that does not know there is anything to
# reach for, so this list is what turns recall from a thing it uses when told into a thing it
# uses when a question arrives. Names and what each is for - never contents, the rule
# PROJECTS_HEADER keeps for the same reason.
# This block has one job: let the model recognise that a question in front of it is one a manual
# on the card can answer. Nothing about *how* to use what comes back belongs here - that is in
# RECALL_TOOL's description, which is read only when the model is already considering the tool,
# where this is read at the top of every session whether a manual comes up or not. At ten manuals
# the long version of this cost about 900 tokens a session for rules that apply to none of them.
MANUALS_HEADER = """\
MANUALS YOU HAVE READ - use recall on one before the web, and name the part in the query.
"""
MANUALS_LISTED = 24


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


def _manuals_block(settings: Settings) -> str:
    """What it has read, or nothing at all. ``_projects_block``'s sibling, in every respect."""
    if not settings.manuals:
        return ""
    try:
        held = [one for one in manuals.catalog(settings) if one.read]
    except OSError:
        return ""
    if not held:
        return ""
    lines = [one.line() for one in held[:MANUALS_LISTED]]
    named = ", ".join(one.part or one.name for one in held[:MANUALS_LISTED])
    print(f"· manuals: {named}", flush=True)
    return f"{MANUALS_HEADER}\n" + "\n".join(lines) + "\n"


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
    moment = session.now_context(settings)
    print(f"· right now: {moment.note}", flush=True)
    blocks = [BASE_INSTRUCTIONS]
    # Who, then what happened, then what is being kept, and the clock last of all. Last because
    # the greeting is the very next thing it says and the end of the prompt is what it has just
    # read: every rule about the first line that would not hold anywhere else - do not open on
    # a hello, do not hand them the floor at the end of it - held once it was read here.
    if about_block := _about_block(settings):
        blocks.append(about_block)
    if recap:
        blocks.append(f"{RECAP_HEADER}\n{recap.text}\n")
    if projects_block := _projects_block(settings):
        blocks.append(projects_block)
    # Last, because it is the only block that is reference rather than context: the others say
    # who and what happened, this says what can be looked up.
    if manuals_block := _manuals_block(settings):
        blocks.append(manuals_block)
    if moment:
        blocks.append(moment.text)
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

    ``name`` is the file's own stem - ``14-32-40_you`` - and it is the picture's name to the
    model as well as to the card, because the model has to be able to say which picture it means
    when it means an earlier one. The stem rather than a number for one reason: a number the
    model misremembers is still a picture, and redrawing the wrong one costs half a minute, an
    image call and the panel; a name it misremembers is nothing, and refuses in a round trip.
    Empty when the picture was never written down, which is the same thing as having no name.
    """

    path: Path | None
    what: str  # "photo" | "found" | "edit" | "drawn" - for the log and the tool's own error
    name: str = ""  # the file's stem, and what the model calls it. "" when nothing was kept.


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
        # A video holds the panel and talks over the room. There is no echo cancellation in this
        # box - the EchoGuard knows about the Speaker and knows nothing about Chromium - so while
        # one is up the microphone is held shut, or the model hears the narrator and answers him.
        # Opened again by panel_closed, which is the kiosk saying the glass is ours.
        self.mic_shut = False
        self._video: Watch | None = None  # what is on the panel, while one is
        self._video_at = 0.0
        self._video_thumb = b""
        self._mic_shut_until = 0.0  # the backstop on mic_shut; see _pump_mic
        # The last picture in play, which is what edit_photo works on. See :class:`Panel`.
        # Not "what is on the glass" any more: a snapped photo is handed over without going up,
        # so the two parted company the day the shutter stopped filling the screen.
        # Kept here rather than found by scanning photos/ for the newest file, because those two
        # are not the same thing: a shutter pressed before the session was ready writes a jpg
        # nothing ever saw, and editing a picture the model cannot reason about is worse than
        # asking for another.
        self._on_panel: Panel | None = None
        # ...and every picture that came into play this session, oldest first. The slot above is
        # what "change that" means; this is what "change the one with the ball, not the desk"
        # means, three photos later. Only the model can tell those two apart - it has all of the
        # pictures in context and we have none of them - so each one is named as it arrives and
        # the model hands the name back. Nothing prunes it: a session's pictures are a few dozen.
        self._pictures: list[Panel] = []
        # The picture a recall just put on the panel, waiting to be handed to the model, and the
        # name it was given. Held on the agent rather than returned, because it has to be sent
        # *after* the tool output that mentions it - see _run_recall. The name rides along rather
        # than being read back off `_on_panel`, which a drawing landing in that same window moves.
        self._found_image: bytes | None = None
        self._found_name = ""
        # What kind of picture it was, so add_found can say so. A manual page and a photograph of
        # the bench are both "found", and calling one the other is how the model ends up saying
        # "in the photo you took" about page 28 of a manual.
        self._found_kind = ""
        # Whether it went up on the panel too. A manual page is read without being shown unless
        # they asked to see it, and the model must not talk as if they can see what only it can.
        self._found_shown = False
        # Everything recall has handed the model since they last spoke, by item key. A page that
        # did not answer the question would otherwise come straight back: ranking is a dot product
        # with no memory, and three rewordings of one question land on the same page.
        self._recalled: set[str] = set()
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
        # ...and when, on the same clock the kiosk's render loop reads. The session's own log
        # times itself from the tap, because what it is measuring is a recording and a recording
        # starts then; the panel's clock is measuring a conversation, and there is no
        # conversation to count until there is somebody on the other end of it.
        self.ready_at: float | None = None
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
        # A sketch being typed. The only tool here whose arguments are acted on before they
        # have finished arriving, so it needs somewhere to accumulate them; see
        # ``_on_sketch_delta``. One at a time, because there is one screen.
        self._sketch_call: str | None = None
        self._sketch_text = ""
        self._sketch_timer: asyncio.TimerHandle | None = None  # a compile on its way
        self._sketch_shown = False  # whether the panel has been asked for this one yet
        # A walkthrough in progress, which the kiosk draws as a bar over the step it is on. Dies
        # with this agent and so with the session: nothing about it is ever written to the card
        # but the record. Rebound whole, never mutated, for the reason _doing is - the render
        # thread reads it every frame with no lock.
        self._tutorial: tutorial.Tutorial | None = None
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
    def tutorial(self) -> tutorial.Tutorial | None:
        """The walkthrough on the glass, or None. Read once a frame by the kiosk."""
        return self._tutorial

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

    # ---------------------------------------------------------------- which picture they mean

    def _put_on_panel(self, path: Path | None, what: str) -> str:
        """Record a picture as in play, and hand back the name the model may ask for it by.

        The one writer of both :attr:`_on_panel` and :attr:`_pictures`, because two things that
        have to agree are two things that can disagree - and what they would disagree about is
        which photograph gets redrawn. Every caller sends the name it gets back to the model in
        the same breath as the picture, so the name in the conversation is always the name in
        the list.

        The name is the file's own stem, which is unique inside a session for free - the shutter
        and ``imagine.write`` both see to that. Across sessions it is not: a picture recalled off
        the card can arrive carrying the same ``14-32-40_you`` as this afternoon's photo, and two
        pictures with one name is the silent wrong edit this whole scheme exists to avoid. So a
        collision takes a suffix.

        A picture with no path is still what they are looking at, so it still takes the slot -
        but it cannot be named, because there is nothing to open again later.
        """
        name = path.stem if path is not None else ""
        if name and any(shot.name == name for shot in self._pictures):
            name = f"{name}-{len(self._pictures) + 1}"
        shot = Panel(path, what, name)
        self._on_panel = shot
        if path is not None:
            self._pictures.append(shot)
        return name

    def _picture_named(self, name: str) -> Panel | None:
        """The picture the model asked for by name, or None when nothing answers to it.

        Deliberately narrow. An exact name, newest first, then an unambiguous prefix so the time
        alone reaches it; a stray "picture " or ".jpg" the model wrapped it in is forgiven. What
        it will NOT do is guess: "the second one" and "the ball" resolve to nothing, and the
        caller refuses rather than redrawing whatever happened to be last. A refusal costs a
        round trip, and the alternative costs half a minute, an image call and the panel.
        """
        want = name.strip().strip("'\"").casefold().removeprefix("picture ").removesuffix(".jpg")
        if not want:
            return None
        for shot in reversed(self._pictures):
            if shot.name.casefold() == want:
                return shot
        hits = [shot for shot in self._pictures if shot.name.casefold().startswith(want)]
        return hits[0] if len(hits) == 1 else None

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
                *_look_tools(self.settings),
                *_video_tools(self.settings),
                *_diagram_tools(self.settings),
                *_scratchpad_tools(self.settings),
                *_sketch_tools(self.settings),
                *_point_tools(self.settings),
                *_imagine_tools(self.settings),
                *_project_tools(self.settings),
                *_recall_tools(self.settings),
                *TUTORIAL_TOOLS,
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
        # The try opens before the connect, not after it: a bad key or a dead network raises
        # from there, and that is precisely when the gears the button sounded may still be
        # going. Left outside, the finally never runs and the kiosk grinds on over a red border.
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
            # A connect that never landed: the gears are ours to stop, because nothing else is
            # going to - the panel just goes red, and the lid stays shut over a socket that was
            # never there. A *cancel* is the other thing entirely, and deliberately not caught
            # here: someone pressed stop, and the sound of stopping belongs to them. Ending a
            # session is sounded by whoever owns its lifecycle (the kiosk on the tap, the CLI on
            # its way out), never from here, because this runs long before it is over - the
            # socket, the devices and the recording outlive it.
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
        self._recalled.clear()  # a typed turn is them speaking, as far as recall is concerned
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
        """Put a photo into the conversation as an image, and say nothing about it.

        The model has no camera of its own, so this is the only way anything is ever seen. The
        item is a synthetic user turn rather than a tool result, because there is no tool call
        to answer: as far as the conversation is concerned the user held something up.

        No ``response.create`` here, and that is the whole point: pressing the shutter is not a
        question. The picture sits in the conversation until they say something, and the reply
        they get then is about what they actually asked - not a description of a photo they are
        already looking at on the panel. Server VAD creates that response the moment they speak,
        with the image already in context, so nothing has to be re-sent.

        The photo is recorded as in play *before* the send rather than after it, which is the one
        thing here that used to be the other way round ("this means shown, not taken"). The
        caption is the only place the model is ever told the picture's name, so the name has to
        exist before the text that carries it. What that costs is a photo left in the line when a
        send fails - and a send only fails on a socket that is going down, which ends the session
        and the line with it.
        """
        if not self.connected:
            return
        self.tool_active = True
        job = self._start_doing("taking the photo in…")
        # Named before it is sent, because the caption has to say the name: the picture and the
        # only place the model will ever learn what to call it travel in one item.
        name = self._put_on_panel(capture.path, "photo")
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
                            "text": (
                                "[Photo from their camera, taken just now. They have not "
                                "asked anything about it - do not speak about it until they "
                                f"do, then answer off the picture. It is called {name}.]"
                            ),
                        },
                        {
                            "type": "input_image",
                            "image_url": capture.data_url,
                            "detail": "auto",
                        },
                    ],
                }
            )
        finally:
            self.tool_active = False
            self._done_doing(job)
        self._log(
            f"[photo] {capture.width}x{capture.height}, "
            f"{capture.jpeg_bytes // 1024} KB → {capture.path}"
        )

    async def add_edit(
        self, jpeg: bytes, request: str, name: str = "", made_from: str = ""
    ) -> None:
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
                            "unprompted - but answer what they ask about it."
                            + (f" It is called {name}." if name else "")
                            + (f" It was made from {made_from}." if made_from else "")
                            + "]"
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
            # Read and dropped, not left unread: the queue has to keep draining or the room
            # arrives half a minute late once the video is over. See mic_shut and panel_closed.
            #
            # The deadline is the backstop, and it is here rather than nowhere because the
            # failure it guards against is the worst one this feature can produce: a box that
            # has stopped listening and gives no sign of it. panel_closed is what normally
            # opens the mic again, and it is reached from the kiosk's thread - so anything that
            # loses that hop (a swap that outlives its session, a kiosk that went down with the
            # panel up) would leave this shut for good. Past the deadline it opens itself.
            if self.mic_shut and time.monotonic() < self._mic_shut_until:
                continue
            if self.mic_shut:
                self._log("[tool] the mic was still shut when the video's time ran out")
                self.mic_shut = False
                self.mic.drain()
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
            case "response.output_item.added":
                self._on_output_item(event.item)
            case "response.function_call_arguments.delta":
                self._on_sketch_delta(event.call_id, event.delta)
            case "response.done":
                await self._on_response_done(event.response)
        if self.on_event is not None:
            self.on_event(event)

    # ---- a sketch, while it is being typed ----
    #
    # Every other tool here waits for ``response.done`` and then acts on finished arguments,
    # which is the right shape for work that takes a file or a minute. It is the wrong shape for
    # a screen: the model has written the number by the time it has typed ten characters, and
    # waiting for the closing brace throws away most of the sentence it was said in.
    #
    # So these two run on the way past. They are the local half of what MCP Apps does with
    # ``tool-input-partial``: the renderer is already mounted, so a prefix that compiles is a
    # frame, and a prefix that does not is nothing at all. Neither of them answers the model -
    # ``_run_sketch`` still does that when the call lands, off ``response.done``, like the rest.

    def _on_output_item(self, item: object) -> None:
        """Notice a sketch starting, so its argument deltas have somewhere to go."""
        if getattr(item, "type", None) != "function_call" or getattr(item, "name", "") != "sketch":
            return
        call_id = getattr(item, "call_id", None)
        if not call_id:
            return
        self._sketch_call = call_id
        self._sketch_text = ""
        self._sketch_shown = False

    def _on_sketch_delta(self, call_id: str, delta: str) -> None:
        """Take a piece of the sketch, and make sure a compile is coming.

        This does not compile. It cannot: deltas do not trickle in at reading speed, they arrive
        in a burst - 52 of them inside 386 ms, measured against gpt-realtime-2.1 - and the first
        one carries two characters of JSON. A throttle that compiles the delta in front of it and
        drops the rest therefore compiles exactly one prefix per burst, and that prefix is ``{"``.
        Which is what shipped on 2026-09-10, and why the first session with this on drew nothing
        until the call landed.

        So the throttle defers instead of dropping. Every delta makes sure a timer is pending;
        the timer compiles whatever has arrived by the time it fires and then clears itself. A
        burst costs one compile, the last delta always leaves a timer behind it, and nothing is
        lost at the end of the stream. It is the same trailing-edge shape Prefab's own renderer
        uses on this exact event, which should have been the hint.
        """
        if call_id != self._sketch_call:
            return
        self._sketch_text += delta
        if self._sketch_timer is None:
            loop = asyncio.get_running_loop()
            self._sketch_timer = loop.call_later(sketch.THROTTLE_S, self._draw_sketch)

    def _draw_sketch(self) -> None:
        """Compile everything that has arrived and put it up if it changed anything.

        The panel is asked for on the first frame that compiles rather than on the first delta.
        Uncovering the browser is most of a second, and starting it before there is anything to
        uncover onto buys a spinner where the eye used to be.
        """
        self._sketch_timer = None
        code, _ = arguments.value(self._sketch_text, "code")
        if not sketch.offer(code):
            return
        if not self._sketch_shown:
            self._sketch_shown = True
            self._spawn(asyncio.to_thread(self._take_the_glass))

    def _take_the_glass(self) -> None:
        """Ask the panel to uncover the browser. Blocking; runs off the loop's thread.

        Off the loop for :meth:`_write_scratchpad`'s reason: ``panel.offer_sketch`` goes through
        ``card.write_text``, which fsyncs an SD card, and the loop it would block is the one
        carrying his voice. What it writes is a marker and not a payload - the frames themselves
        went out over the companion stream a moment ago and are already drawn.
        """
        panel.offer_sketch() and panel.show()

    def _on_session_ready(self) -> None:
        if self.ready.is_set():
            return
        self.ready_at = time.monotonic()
        self.ready.set()  # set first: the panel and show_photo() both gate on it
        # Both handshake lines go at once rather than expiring: nothing else can be in flight
        # this early - a tool needs a response and a response needs this - so there is nothing
        # underneath them to hand the caption back to. From here the state's own resting line is
        # the true one, until a tool has something better to say.
        self._doing = ()
        # His lid, on the event that opens it. Setting `ready` above is what turns the
        # controller's CONNECTING into LISTENING, which is what the panel's lid answers to
        # (cyclops.overlay.session_live), so the sound and the steel come off one fact and are
        # never more than a frame apart. Sounded from here rather than from the render loop that
        # notices afterwards, for the same reason the gears are sounded from the button's own
        # thread: a cue that has to arrive with a movement cannot be quantised to a frame that
        # drops to four a second behind a dark panel.
        #
        # It replaces the rising fifth that used to be here. Both said the same thing at the
        # same instant - the session is up, talk now - and with one speaker the later of the two
        # simply clipped the other; only one of them is also a picture.
        sounding = self.cues.play("iris_open")
        if self.mic is not None:
            self._spawn(self._listen(after_s=sounding))
        # The UI framework's import, bought now rather than on the first delta of the first
        # sketch - which is the one moment in that path where somebody is watching nothing.
        if self.settings.sketch:
            self._spawn(asyncio.to_thread(sketch.warm))

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
        self._recalled.clear()  # a new question; a page that missed the last one may answer this
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

    def stop_talking(self, played_ms: int) -> None:
        """A finger on the glass stopped him. Playback is already cut; finish the job here.

        :meth:`local_barge_in` without its confirming timer, and the missing timer is the point.
        That one is there to ask the server whether the room really spoke, so a trigger that was
        only echo can raise EchoGuard's bar. Nobody spoke here. Arming it would have
        ``_reject_local_barge_in`` punish the room for a gesture it never made.
        """
        item_id = self._current_item_id
        if item_id is None:
            return  # nothing of his is in the air; the tap has nothing to take back
        self._current_item_id = None
        self._dead_item_ids.add(item_id)
        self._end_assistant_line()
        self._log(f"(stopped by a tap after {played_ms} ms)")
        self._spawn(self._truncate_and_cancel(item_id, played_ms))

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
        if call.name == "take_a_look":
            await self._run_take_a_look(call)
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
        if call.name == "sketch":
            await self._run_sketch(call)
            return
        if call.name == "point_at":
            await self._run_point(call)
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
        if call.name == "watch_video":
            await self._run_watch_video(call)
            return
        if call.name in TUTORIAL_TOOL_NAMES:
            await self._run_tutorial(call)
            return
        # Every name still gets an output. A tool the model invents, or one it remembers from a
        # session config that has since changed, must be answered or it waits for it forever.
        self._log(f"[tool] unknown tool {call.name!r}", stream=sys.stderr)
        await self._send_tool_output(call.call_id, {"ok": False, "error": "unknown tool"})
        await self._request_response()

    async def _run_take_a_look(self, call: RealtimeConversationItemFunctionCall) -> None:
        """Take a photo because they asked Cyclops to look, and answer off it straight away.

        The same camera, folder, flash and click as the SNAP button. What differs is the reply:
        a SNAP photo waits silently for them to speak, and this one was asked for out loud, so
        the answer follows at once. The trigger is still theirs - the schema says never
        unprompted.

        The output first and the picture after it, in the order :meth:`_run_recall` uses: an
        image cannot ride in a function_call_output, and one response.create covers both.
        """
        self._log("[tool] take_a_look")
        save_dir, keep_as = session.photo_target(self.settings, by="cyclops")
        self.tool_active = True
        try:
            async with asyncio.timeout(CAPTURE_TIMEOUT_S):
                capture = await capture_image_async(
                    self.settings.camera_index, save_dir=save_dir, keep_as=keep_as
                )
        except TimeoutError:
            error = f"the camera did not answer within {CAPTURE_TIMEOUT_S:.0f}s"
        except WebcamError as exc:
            error = str(exc)
        except Exception as exc:  # never leave the model waiting for a tool result
            error = f"{type(exc).__name__}: {exc}"
        else:
            error = ""
        finally:
            self.tool_active = False
        if error:
            # No flash and no click: those mean a photo was taken.
            self._log(f"[tool] take_a_look failed: {error}", stream=sys.stderr)
            await self._send_tool_output(call.call_id, {
                "ok": False,
                "error": error,
                "note": "No photo was taken. Say so in a few words; do not guess what is there.",
            })
            await self._request_response()
            return

        if not panel.shutter():
            self.cues.play("shutter")  # no panel to flash, so the click is the whole of it
        session.note(
            "photo",
            by="cyclops",
            file=f"{session.PHOTOS}/{capture.path.name}",
            width=capture.width,
            height=capture.height,
            bytes=capture.jpeg_bytes,
            shown=True,
        )
        name = self._put_on_panel(capture.path, "photo")
        await self._send_tool_output(
            call.call_id, {"ok": True, "note": "The photo follows as an image."}
        )
        await self._send_item(
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        # Flat on purpose, as add_photo's is: a caption written as speech comes
                        # back out of the speaker verbatim.
                        "text": f"[Photo you just took, because they asked you to look. "
                        f"It is called {name}.]",
                    },
                    {"type": "input_image", "image_url": capture.data_url, "detail": "auto"},
                ],
            }
        )
        await self._request_response()
        self._log(
            f"[tool] photo {capture.width}x{capture.height}, "
            f"{capture.jpeg_bytes // 1024} KB → {capture.path}"
        )

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

    async def _run_watch_video(self, call: RealtimeConversationItemFunctionCall) -> None:
        """Find a video, seek it to the part that answers them, and hand the panel to it.

        The whole of what comes back to the model is a title, a channel and a clock time -
        about thirty tokens. The transcript it was chosen from runs to thousands, and it stays
        inside :mod:`cyclops.watch`: a Realtime conversation pays again for everything in it on
        every turn that follows, so the one thing this must never do is bring one home.

        No staleness check, unlike :meth:`_run_web_search`. A late search answer is worth
        reporting as late, because the model can still decide not to read it out. A late video
        has already taken the screen by the time anyone could decide anything, and the way to
        dismiss it is the way to dismiss everything else here: press the panel.
        """
        request = _tool_request(call.arguments)
        self._log(f"[tool] watch_video {request!r}")
        if not request:
            await self._send_tool_output(call.call_id, {"ok": False, "error": "empty request"})
            await self._request_response()
            return

        self.search_active = True  # the panel says SEARCHING; it is the same kind of waiting
        try:
            async with asyncio.timeout(WATCH_TIMEOUT_S):
                found = await find_video(request, self.settings)
                # Fetched here rather than inside watch.find, because it is the one part of
                # this that is not a decision: a picture to hold the screen while the video
                # starts, and the only thing between a video and a black stretch of recording.
                thumb = await asyncio.to_thread(youtube.fetch, found.thumb)
        except TimeoutError:
            output = {
                "ok": False,
                "error": f"finding a video took longer than {WATCH_TIMEOUT_S:.0f}s",
            }
        except WatchError as exc:
            output = {"ok": False, "error": str(exc)}
        except Exception as exc:  # never leave the model waiting for a tool result
            output = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        else:
            output = await self._play_video(found, thumb)
        finally:
            self.search_active = False

        await self._send_tool_output(call.call_id, output)
        if not output["ok"]:
            session.note("watched", request=request, error=output["error"])
            self._log(f"[tool] watch_video failed: {output['error']}", stream=sys.stderr)
            await self._request_response()  # it has to be able to say it could not find one
            return
        # ...and no response at all when one is playing. This is the only tool here that does
        # not ask for one, and it is the difference between a video and a video with somebody
        # talking over it. The note in the output asks the model to stay quiet, but a note is
        # advice: the model had already said what it was doing before it called this, so being
        # asked again afterwards reads as a turn to fill and it filled it - "Got it, there's a
        # video playing that shows how to mount a boom arm" over the top of the video saying
        # the same thing. Not creating a response is not a request, it is the mechanism.
        #
        # Nothing is left waiting. The function_call_output is sent, so the model is not
        # blocked; it simply has no turn until the user takes one, which is the whole point.

    async def _play_video(self, found: Watch, thumb: bytes) -> dict:
        """Put one video on the panel, and shut the microphone for as long as it is up.

        Named in full rather than ``_play``, which is taken: :meth:`_play` is what feeds an
        audio delta to the speaker. Shadowing it silently costs the session its voice - the
        model answers, the transcript fills up and nothing comes out of the speaker - which is
        exactly what happened on 2026-09-11 before this was renamed.
        """
        shown = await asyncio.to_thread(
            lambda: panel.offer_video(found.stream, found.title, found.start, thumb, VIDEO_HOLD_S)
            and panel.show()
        )
        if not shown:
            # No panel at all (uv run cyclops), or the admin page is up and somebody is reading
            # it. Withdraw, or this offer is what the next tap uncovers onto - see panel.py.
            await asyncio.to_thread(panel.withdraw)
            return {
                "ok": False,
                "error": "there is no screen free to play it on right now",
                "title": found.title,
            }
        self._video = found
        self._video_thumb = thumb
        self._video_at = time.monotonic()
        self.mic_shut = True
        # Past the panel's own cap, so this only ever fires when that one did not.
        self._mic_shut_until = self._video_at + VIDEO_HOLD_S + 60.0
        self._log(f"[tool] watching {found.id} at {found.clock}: {found.title!r}")
        return {
            "ok": True,
            "title": found.title,
            "channel": found.channel,
            "start": found.clock,
            "note": (
                "It is playing on the screen now. They cannot hear you over it and you have "
                "not been given a turn, which is deliberate - wait for them to speak."
            ),
        }

    def panel_closed(self, seconds: float) -> None:
        """The panel is ours again: open the microphone, and keep the video if it was watched.

        Reached from the kiosk's own thread through
        :meth:`cyclops.ui.SessionController.panel_closed`, which is why this is a plain method
        rather than a coroutine - it arrives by ``call_soon_threadsafe`` onto a loop that is
        carrying audio, so it must not block and must not await.

        Every picture comes through here and only a video has anything to do with it.
        ``seconds`` is how long the glass was theirs, and that is the one honest signal for
        whether this was the right video: somebody who pressed the screen after four seconds
        was telling us it was not, and a reference kept from that is clutter on the card.
        """
        self.mic_shut = False
        if self.mic is not None:
            # Everything the narrator said while the gate was shut is still queued behind this.
            # Dropped rather than let through, for the reason _listen gives about the chime.
            self.mic.drain()
        found, self._video = self._video, None
        if found is None:
            return
        watched = seconds if seconds > 0 else time.monotonic() - self._video_at
        if watched < WATCHED_S:
            self._log(f"[tool] video let go after {watched:.0f}s: {found.title!r}")
            return
        self._log(f"[tool] keeping {found.id} after {watched:.0f}s: {found.title!r}")
        session.keep_video(
            video_id=found.id,
            title=found.title,
            channel=found.channel,
            start=found.start,
            seconds=round(watched),
            thumb=self._video_thumb,
        )

    # ---- drawing one ----

    async def _run_scratchpad(self, call: RealtimeConversationItemFunctionCall) -> None:
        """Put what the model wrote on the panel. The shortest handler here, and deliberately.

        Every other tool that reaches the glass has a picture to make first - half a minute of
        the image model, or a file off the card. This one already has everything it needs in its
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

        # ``_on_panel`` is deliberately left alone. It used to be cleared here, because what
        # they were looking at was no longer a picture and editing one they could not see was
        # worse than asking for another. A snapped photo never reaches the glass now, so that
        # reasoning has gone with it: the photo they took two sentences ago is still the thing
        # an edit is about, and a torque figure written down in between does not change that.
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

    async def _run_sketch(self, call: RealtimeConversationItemFunctionCall) -> None:
        """Answer the model about the sketch that has mostly already happened.

        By the time this runs, the panel is usually showing the thing: the deltas drew it while
        the arguments were arriving (:meth:`_on_sketch_delta`). What is left is the last compile
        - on the complete code, so a program whose final line never got a chance to run still
        finishes - the session note, and telling the model where it stands.

        It also has to work when none of that happened. A response can carry a sketch whose
        arguments never streamed, and then this is the only compile there is.
        """
        code = _tool_code(call.arguments)
        drew = self._sketch_shown
        self._log(f"[tool] sketch {len(code)} chars, {'drawn while typed' if drew else 'whole'}")
        self._sketch_call = None
        self._sketch_text = ""
        if self._sketch_timer is not None:
            # A compile scheduled by the last delta, which this call has now overtaken. It
            # would only redraw a prefix of what we are about to draw in full.
            self._sketch_timer.cancel()
            self._sketch_timer = None
        if not code:
            await self._send_tool_output(call.call_id, {"ok": False, "error": "nothing to draw"})
            await self._request_response()
            return

        drawn = sketch.offer(code) or sketch.current() is not None
        if not drawn:
            # Every prefix failed and so did the whole thing. The model wrote Python that does
            # not run, and the useful answer says so rather than pretending: it can write it
            # again, and it is the only one that knows what it meant.
            await self._send_tool_output(call.call_id, {
                "ok": False,
                "error": "that code did not run",
                "note": (
                    "Nothing is on the screen. Open with `with PrefabApp() as app:` and write "
                    "no import lines. Say the answer out loud, and try once more if it is worth "
                    "seeing."
                ),
            })
            await self._request_response()
            return

        shown = self._sketch_shown
        if not shown:
            shown = await asyncio.to_thread(self._take_the_glass_now)
        session.note("sketch", code=code[: sketch.MAX_SKETCH_CHARS], panel=shown)
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

    def _take_the_glass_now(self) -> bool:
        """:meth:`_take_the_glass`, for the case where the deltas never ran, and it reports back."""
        self._sketch_shown = True
        return panel.offer_sketch() and panel.show()

    async def _run_point(self, call: RealtimeConversationItemFunctionCall) -> None:
        """Mark their picture. The only tool here that reaches the glass without touching a file.

        The scratchpad is the shortest handler in this class and this one is shorter still.
        There is no picture to make, and unlike the scratchpad there is nothing to write either:
        a gesture is a few marks in memory that the render loop reads on its next frame, so
        ``panel``, ``card`` and the browser are all out of it. It is up in about 40 ms, which is
        the whole reason to prefer it to a sentence about where something is.

        Nothing is kept, and for the scratchpad's reason: pointing is something he did while
        talking, not a picture. The session log records that it happened, and ``photos/`` gets
        nothing - the photo that was marked is already in there.
        """
        marks = point.parse(_tool_marks(call.arguments))
        wanted = _tool_picture(call.arguments)
        self._log(f"[tool] point_at {len(marks)} mark(s){f' on {wanted}' if wanted else ''}")
        if not marks:
            await self._send_tool_output(call.call_id, {
                "ok": False,
                "error": "no marks I could read",
                "note": (
                    "One per line, and the numbers are fractions of the picture: "
                    "'ring 0.42 0.31 cold joint'."
                ),
            })
            await self._request_response()
            return

        shot = self._picture_named(wanted) if wanted else self._on_panel
        if wanted and shot is None:
            await self._send_tool_output(call.call_id, {
                "ok": False,
                "error": "no picture by that name",
                "pictures": [
                    {"name": one.name, "kind": one.what}
                    for one in reversed(self._pictures[-PICTURES_OFFERED:])
                ],
                "note": (
                    "Nothing was drawn. These are the pictures you have been shown, newest "
                    "first. Call it again with one of these names if one is plainly the one "
                    "they mean - and if two of them could be, ask them which rather than picking."
                ),
            })
            await self._request_response()
            return
        if shot is None or shot.path is None:
            await self._send_tool_output(call.call_id, {
                "ok": False,
                "error": "no picture to point at",
                "note": "Take a look first, then point. Do not guess at coordinates.",
            })
            await self._request_response()
            return

        point.show(point.Gesture(tuple(marks), shot.path, time.monotonic()))
        session.note("point", picture=shot.name, marks=len(marks),
                     text=_tool_marks(call.arguments))
        await self._send_tool_output(call.call_id, {
            "ok": True,
            "picture": shot.name,
            "marks": len(marks),
            "note": (
                "It is on their screen. Say which one in a few words - never where on the "
                "picture it is, because they can see the mark."
            ),
        })
        await self._request_response()

    async def _run_tutorial(self, call: RealtimeConversationItemFunctionCall) -> None:
        """Start, move on or stop a walkthrough, and tell the model what to do next.

        What comes back is the whole of the teaching: the note is composed from the step that is
        now up (:func:`cyclops.tutorial.note`), so the model reads the waiting rule as the step
        it is about goes up, and a session with no walkthrough in it reads nothing about them.
        """
        if call.name == "start_tutorial":
            output = self._start_tutorial(_tool_steps(call.arguments))
        elif call.name == "advance_tutorial":
            output = self._advance_tutorial()
        else:
            output = self._end_tutorial()
        self._log(f"[tool] {call.name} -> {output.get('step', '-')}/{output.get('of', '-')}")
        await self._send_tool_output(call.call_id, output)
        await self._request_response()

    def _start_tutorial(self, steps: list[str]) -> dict[str, Any]:
        """Put *steps* up at the first one, or refuse and leave the glass as it was."""
        if not tutorial.MIN_STEPS <= len(steps) <= tutorial.MAX_STEPS:
            return {"ok": False, "error": f"{len(steps)} steps",
                    "note": tutorial.refused(len(steps))}
        self._tutorial = tutorial.Tutorial(tuple(steps))
        session.note("tutorial", action="started", steps=steps)
        return self._step_output(self._tutorial)

    def _advance_tutorial(self) -> dict[str, Any]:
        """The next step up, or the walkthrough over once the last one is done."""
        running = self._tutorial
        if running is None:
            return {"ok": False, "error": "no tutorial running", "note": tutorial.NONE_RUNNING}
        self._tutorial = running.advanced()
        if self._tutorial is None:
            session.note("tutorial", action="finished", of=running.total)
            return {"ok": True, "finished": True, "note": tutorial.FINISHED}
        session.note("tutorial", action="advanced", step=self._tutorial.number,
                     of=self._tutorial.total, label=self._tutorial.current)
        return self._step_output(self._tutorial)

    def _end_tutorial(self) -> dict[str, Any]:
        """Take the walkthrough off the glass wherever it had got to."""
        running = self._tutorial
        if running is None:
            return {"ok": False, "error": "no tutorial running", "note": tutorial.NONE_RUNNING}
        self._tutorial = None
        session.note("tutorial", action="ended", step=running.number, of=running.total)
        return {"ok": True, "ended": True, "note": tutorial.ENDED}

    @staticmethod
    def _step_output(up: tutorial.Tutorial) -> dict[str, Any]:
        return {"ok": True, "step": up.number, "of": up.total, "now": up.current,
                "note": tutorial.note(up)}

    async def _run_draw_diagram(self, call: RealtimeConversationItemFunctionCall) -> None:
        """Set a drawing going, answer at once, and let :meth:`_draw` finish it.

        The tool returns in a moment and the picture arrives about half a minute later. That is
        the whole point of the split: this used to await the image model here, so the call stayed
        open for the whole of it and the model could not say another word until it closed - while
        both this tool's own description and the system prompt told it to say what it was doing
        and carry on talking. ``imagine.py`` said the same thing about the wait being bearable
        "only because nothing is waiting on it", and asked for the lifecycle to be fixed rather
        than the quality dropped. This is that fix. The wait is a third of what it was since
        ``IMAGE_MODEL`` became 2.5 sunburst, and none of that argument changes: a call held open
        for twenty-six seconds is still a conversation with nothing in it.

        What holds the two halves together is a task (:mod:`cyclops.tasks`): the panel says what
        is being drawn for as long as it takes, and :meth:`announce` tells the model when it
        lands. Nothing polls and nothing waits.
        """
        request = _tool_string(call.arguments, "request", imagine.MAX_REQUEST_CHARS)
        style = _tool_string(call.arguments, "style", imagine.MAX_STYLE_CHARS)
        self._log(f"[tool] draw_diagram {request!r} ({style!r})")
        if not request:
            await self._send_tool_output(call.call_id, {"ok": False, "error": "nothing described"})
            await self._request_response()
            return

        # Set before the answer goes out, not inside the coroutine: the model is about to be told
        # the drawing has started, and a panel still saying LISTENING underneath that would be the
        # one moment the two disagree. `_activity_line` writes the sentence the caption already
        # uses for this call - "drawing the relay wiring…" - so the task and the panel agree too.
        self.drawing_active = True
        task = tasks.start(_activity_line(call))
        self._spawn(self._draw(task, request, style))
        await self._send_tool_output(call.call_id, {
            "ok": True,
            "started": True,
            "note": (
                "It is being drawn now and takes about half a minute. Say in a few words "
                "that you are drawing it, then carry on talking about something else - you will "
                "be told when it lands or if it fails."
            ),
        })
        await self._request_response()

    async def _draw(self, task: str, request: str, style: str) -> None:
        """The half minute. Runs on its own after :meth:`_run_draw_diagram` has answered.

        The model is deliberately *not* told what the diagram contains. It asked for a picture,
        the picture is on the screen, and a model handed a description of it will read that
        description out at somebody who is already looking at the thing.
        """
        started = time.monotonic()
        try:
            # Keeping and showing is inside the try, not in an else. It was in an else once, and
            # a TypeError in the record it writes propagated straight out of this coroutine: the
            # picture was on the panel, and the model sat waiting for a tool result that was
            # never sent until the user spoke over it.
            jpeg = await imagine.draw(request, style, self.settings)
            output, kept = await asyncio.to_thread(self._keep_and_show_drawing, jpeg, request)
            # The one number that says whether a change of image model landed, and the only place
            # it is visible on the box. `_edit` has had this line all along; this half timed
            # nothing, so the thing everybody waits on was the thing nothing wrote down.
            self._log(f"[tool] draw: {len(jpeg) // 1024} KB in {time.monotonic() - started:.1f}s")
        except imagine.ImagineError as exc:
            session.note("photo", by="drawn", request=request[:80], error=str(exc))
            self._log(f"[tool] draw_diagram failed: {exc}", stream=sys.stderr)
            output, kept = {"ok": False, "error": str(exc)}, None
        except Exception as exc:  # never leave the model waiting to be told how it went
            session.note("photo", by="drawn", request=request[:80], error=f"{type(exc).__name__}")
            self._log(f"[tool] draw_diagram failed: {exc!r}", stream=sys.stderr)
            output, kept = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, None
        finally:
            self.drawing_active = False

        # The panel slot, set on the loop rather than in the thread that showed it, because
        # `edit_photo` reads it from here. A drawing now has a real path like every other
        # picture, so "make that clearer" edits the diagram instead of being refused.
        drawn = ""
        if output.get("shown") and kept is not None and kept.path is not None:
            drawn = self._put_on_panel(kept.path, "drawn")

        if not output.get("ok"):
            tasks.fail(task, str(output.get("error", "")))
            await self.announce(
                "[The diagram you were drawing could not be made: "
                f"{output.get('error', 'it failed')}. Tell them so in a few words. Nothing "
                "appeared on their screen, so nobody else has told them.]"
            )
            return
        tasks.finish(task)
        if not output.get("shown"):
            # Drawn with no panel to put it on - `uv run cyclops` at a desk. The one case where
            # describing it is the right thing to do, because they have nothing to look at.
            await self.announce(
                "[The diagram you were drawing is finished, but there was no screen to put it "
                f"on. Say so plainly rather than describing it. It was: {request}]"
            )
            return
        await self.announce(
            "[The diagram you were drawing is now up on their screen. Say it is there, in a few "
            "words. Do not describe it or read it back - they are looking at it."
            # The one picture the model is told about and never shown, so this sentence is the
            # only handle it will ever have on the diagram it just drew.
            + (f" It is called {drawn}." if drawn else "")
            + "]"
        )

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
        shown = panel.offer_image(small, request, announce=True) and panel.show()
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
        show = _tool_show(call.arguments)
        self._log(
            f"[tool] recall {query!r}"
            + (f" in {project!r}" if project else "")
            + (" (show)" if show else "")
        )
        if not query:
            nothing = {"ok": False, "error": "nothing to look for"}
            await self._send_tool_output(call.call_id, nothing)
            await self._request_response()
            return
        turn = self._turn_serial
        self._found_image = None
        try:
            output = await self._recall(query, project, show=show)
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
        name, self._found_name = self._found_name, ""
        kind, self._found_kind = self._found_kind, ""
        shown, self._found_shown = self._found_shown, False
        if found is not None:
            await self.add_found(found, name, kind=kind, shown=shown)
        await self._request_response()

    async def _replay(self, query: str, best: recall.Hit, others: list[dict]) -> dict[str, Any]:
        """Put a video reference from the card back on the panel, playing again.

        A reference keeps the id and the second, never the URL - a signed googlevideo link is
        dead within six hours of being minted, so a stored one would fail exactly when it was
        wanted. Resolving a fresh one costs about three seconds.

        A failure here is not a dead end. The thumbnail beside the sidecar is a real picture on
        the card, so a video that has been pulled, or a box that is offline, still puts the
        title card up and lets the model say what it found and that it will not play.
        """
        about = await asyncio.to_thread(_reference_beside, Path(best.item.path))
        stream = await restream(about["id"]) if about.get("id") else ""
        if not stream:
            shown, _ = await asyncio.to_thread(self._show_found, Path(best.item.path))
            session.note(
                "recall", query=query[:120], title=best.item.title, what="video", shown=shown
            )
            return {
                "ok": True,
                "hits": 1 + len(others),
                "kind": "video",
                "title": best.item.title,
                "note": (
                    "That video is on the card but will not play right now. Its picture is on "
                    "the screen. Say what it was and that you cannot start it."
                ),
                "others": others,
            }
        found = Watch(
            id=about["id"],
            title=str(about.get("title", "")) or best.item.title,
            channel=str(about.get("channel", "")),
            start=int(about.get("start", 0) or 0),
            stream=stream,
            thumb="",
        )
        thumb = await asyncio.to_thread(_read_bytes, Path(best.item.path))
        output = await self._play_video(found, thumb)
        session.note(
            "recall",
            query=query[:120],
            title=found.title,
            what="video",
            file=best.item.within,
            scope=best.item.scope,
            shown=bool(output["ok"]),
        )
        output["hits"] = 1 + len(others)
        output["others"] = others
        return output

    def _recall_scopes(self, project: str) -> set[str] | None:
        """Which corners of the card this query may look in, as ``recall.Item.scope`` values.

        This session and every project by default: "that one" almost always means either
        something from ten minutes ago or something filed under whatever is on the bench. Older
        sessions are indexed but not searched, because a year of half-finished conversations is
        mostly noise against a question about a project, and the project is where the finished
        version of anything ends up.

        Manuals are in scope always, and are why this builds the set before the early returns
        below rather than after them. A manual belongs to a part and not to a project, so "on the
        Honda, what is the valve clearance" - a question that names a project and so takes the
        ``if project:`` return - must still be able to reach one. Both exits have to carry them,
        or the feature is dead in exactly the phrasing most likely to be used.
        """
        scopes: set[str] = set()
        if (live := session.current()) is not None:
            scopes.add(f"session:{live.dir.name}")
        if self.settings.manuals:
            scopes.update(one.scope for one in manuals.catalog(self.settings))
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

    def _full_page(self, path: Path, fallback: bytes | None) -> bytes | None:
        """A manual page at the size it was rendered, for the model to read rather than glance at.

        Falls back to whatever went to the panel if the file will not read - a smaller picture is
        worse than this one but much better than none, and the page is already on their screen.
        """
        try:
            return path.read_bytes()
        except OSError:
            return fallback

    async def _recall(self, query: str, project: str, *, show: bool = False) -> dict[str, Any]:
        """One search, and the panel if the answer is a photo or a page they asked to see.

        Nothing handed back since they last spoke is handed back again, so asking twice in one
        turn is how the model reads the next page rather than the same one.
        """
        if len(self._recalled) >= RECALL_READS:
            return {
                "ok": True,
                "hits": 0,
                "note": (
                    "That is enough looking for one question. If nothing you were given answers "
                    "it, say in a few words that the manual does not cover it, and use "
                    "web_search if the web can answer it."
                ),
            }
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
        # Ranked deep enough that everything already handed back can be skipped and a full
        # RECALL_HITS still be left to judge against MIN_SCORE.
        ranked = recall.rank(
            index, vector[0], scopes=scopes, limit=RECALL_HITS + len(self._recalled)
        )
        fresh = [hit for hit in ranked if hit.item.key not in self._recalled][:RECALL_HITS]
        hits = [hit for hit in fresh if hit.score >= recall.MIN_SCORE]
        if not hits and any(hit.score >= recall.MIN_SCORE for hit in ranked):
            return {
                "ok": True,
                "hits": 0,
                "note": (
                    "Nothing new matches that - you have already been given everything that "
                    "does. If none of it answered the question, say in a few words that it is "
                    "not in there, and use web_search if the web can answer it."
                ),
            }
        if not hits:
            return {
                "ok": True,
                "hits": 0,
                "note": (
                    "Nothing on the card matches that. If you have not already, try once more "
                    "worded differently; otherwise say so in a few words. Do not describe the "
                    "nearest thing as if it were what they asked for."
                ),
            }

        best = hits[0]
        self._recalled.add(best.item.key)
        # What else it could have been, for the model to offer aloud rather than to choose from -
        # the picture is already going up by the time this is read. Titles are trimmed because a
        # photo's title is its whole caption, which for a page of specifications runs to a
        # paragraph and would drown the result it is a footnote to.
        others = [
            {"title": _shorten(hit.item.title, MAX_OTHER_CHARS), "kind": hit.item.kind}
            for hit in hits[1:RECALL_OFFERED]
        ]
        if best.item.playable:
            return await self._replay(query, best, others)
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

        page = best.item.kind == "page"
        if page and not show:
            # Read, not shown: they asked a question, not to look at a page. The model reads the
            # full render exactly as it would have, and the panel is left as it was.
            shown, seen = False, None
        else:
            shown, seen = await asyncio.to_thread(
                self._show_found, Path(best.item.path), page=page
            )
        if page:
            # The panel keeps the downscaled copy - it is 800x480 and cannot use more - but the
            # model gets the full render. `_show_found` hands back one copy for both jobs, which
            # is right for a photograph and wrong for a page of small print: `imagine.for_panel`
            # caps the long edge at PANEL_MAX_EDGE, so a 150 dpi A4 page reaches the model at
            # about 87 dpi. The 0.96 transcription recall this feature rests on was measured at
            # 120. Reading a torque figure off the page is the whole point, so it reads the page.
            seen = await asyncio.to_thread(self._full_page, Path(best.item.path), seen)
        self._found_kind = best.item.kind
        self._found_shown = shown
        if shown:
            # It is the picture in front of them now, so it is the one edit_photo works on. The
            # path is the file on the card, not the downscaled copy that went to the panel: an
            # edit should start from the best pixels there are, not from the panel's 800x480.
            self._found_name = self._put_on_panel(Path(best.item.path), "found")
        session.note(
            "recall",
            query=query[:120],
            title=best.item.title,
            what=best.item.kind,  # never `kind`; see above
            # Where it is, not just what it is called. A recalled picture is somebody else's -
            # another session's photos, or a project folder - so the filename alone was never
            # enough to find it again, and the transcript showed the line with no picture under
            # it while every other picture in the conversation had one. The pair is what
            # library.records turns into a URL.
            file=best.item.within,
            scope=best.item.scope,
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
                "The best match is on the panel. The others are near misses, not a menu - if "
                "what you showed is plainly the wrong thing, call recall again for the right one."
            )
        if page and not show:
            result["note"] = (
                "Only you can see this page - it is not on their screen. If it answers what they "
                "asked, answer as though you know it. If it does not, call recall again at once, "
                "worded differently, without a word to them first and nothing about this page."
            )
        elif not shown:
            result["note"] = (
                "It was found but there is no panel free to show it on. Say what it is out loud "
                "instead of pretending they can see it."
            )
        self._found_image = seen  # picked up by _run_recall once the tool output is away
        return result

    def _show_found(self, path: Path, *, page: bool = False) -> tuple[bool, bytes | None]:
        """Put a picture already on the card onto the panel, and hand back what was shown.

        The bytes come back for the same reason :meth:`_keep_and_show_edit` returns them: the
        downscaled copy does two jobs, travelling to the panel and going to the model, and 1024 on
        the long edge is what ``webcam.MAX_EDGE`` hands the model for a real photograph anyway.

        ``imagine.for_panel`` returns its input untouched when it is already small enough, and
        ``panel.offer_image`` labels whatever it is given ``data:image/jpeg``. That pairing is
        safe for a photo and wrong for a small PNG - which is what a picture dropped into a project
        folder tends to be - so anything not already a JPEG is re-encoded first. A mislabelled data
        URL renders as nothing at all, silently, on the one screen nobody can see from here.

        A manual ``page`` goes the full width of the panel instead (``imagine.for_page``), and the
        panel is told it is a page so that it lays it edge to edge and lets a finger scroll it.
        """
        try:
            blob = path.read_bytes()
        except OSError as exc:
            self._log(f"[tool] could not read {path}: {exc}", stream=sys.stderr)
            return False, None
        if path.suffix.lower() not in {".jpg", ".jpeg"}:
            blob = imagine.as_jpeg(blob)
        small = imagine.for_page(blob) if page else imagine.for_panel(blob)
        return bool(panel.offer_image(small, path.stem, page=page) and panel.show()), small

    async def add_found(
        self, jpeg: bytes, name: str = "", *, kind: str = "", shown: bool = True
    ) -> None:
        """Show the model the picture it just found. No response is asked for here.

        The reason this exists rather than the caption being enough: the caption is *index text*,
        written by a small model to make the picture findable, and it is not reliable about detail.
        Measured on 2026-09-04, two passes over one identical manual page disagreed about the
        numbers printed on it. So the caption is allowed to find a photograph and never to describe
        one - if somebody asks what the torque figure actually is, the answer has to come off the
        pixels, and this is what puts the pixels where the model can read them.

        The same shape as :meth:`add_edit`, and not that method, because what it says about the
        picture is the opposite: an edit is a drawing that must not be read as evidence, and this
        is a photograph of their own bench that must be.

        A manual page is the same argument with a different noun, and it is worth the branch: the
        sentence below is the only thing telling the model what it is looking at, and told it is a
        photograph it will talk about a printed page as though they had taken it.

        ``shown`` says whether it went on the panel as well. A manual page is usually read without
        being shown, and told nothing the model assumes they are looking at what it is - "as you
        can see there" about a page that is not on any screen.
        """
        page = kind == "page"
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
                            "[The "
                            + ("page you just looked up" if page else "picture you just found")
                            + (
                                ", now on their screen."
                                if shown
                                else ". Only you can see it - it is not on their screen."
                            )
                            + " This is a real "
                            + (
                                "page of a manual on their card"
                                if page
                                else "photograph from their own card"
                            )
                            + ", so you may read it and answer "
                            "questions about what is in it - and prefer reading it to anything "
                            "the search said about it, which was written to find the "
                            + ("page" if page else "picture")
                            + " rather than to describe it accurately."
                            + (
                                " They are looking at it too, so do not narrate it unprompted."
                                if shown
                                else ""
                            )
                            + (f" It is called {name}." if name else "")
                            + "]"
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

    async def announce(self, text: str) -> None:
        """Tell the model that something it started in the background has landed, and let it talk.

        The other end of a tool that returns before its work is done. ``draw_diagram`` answers in
        a moment and the picture arrives about half a minute later, so the arrival has to reach
        the conversation on its own - there is no tool call left to answer by then, and the model
        would otherwise be told nothing at all and go on believing a drawing was still being made.

        A synthetic user turn for the reason :meth:`add_photo`'s docstring gives, and flat and
        bracketed for the reason its caption is: a line written as speech comes back out of the
        speaker verbatim. It asks for a response, which is what separates it from
        :meth:`add_edit` - that one rides inside a tool call that is about to ask for its own.

        Silent when the socket has gone. A session that ended while a drawing was in flight has
        nobody left to tell, and the picture went to the panel regardless.
        """
        if not self.connected:
            return
        await self._send_item(
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}
        )
        await self._request_response()

    # ---- imagined pictures ----

    async def _run_edit_photo(self, call: RealtimeConversationItemFunctionCall) -> None:
        """Check there is something to edit, set the redraw going, and answer at once.

        The same lifecycle as :meth:`_run_draw_diagram` and split the same way and for the same
        reason: the picture takes half a minute, and a tool call held open for that long is a
        conversation with nothing in it. What is left here is the three questions that can be
        answered without drawing anything - is there a request, is there a picture to change, and
        was that picture ever written down. None of those is a task: they are answered in the
        moment and there is nothing to watch.

        Which picture is the second of those questions, and it has two answers. Nothing named
        means the newest, which is what "change that" means and what nearly every call wants. A
        name means that picture, anywhere in the session - and a name that answers to nothing is
        refused here, with the real names, rather than falling through to the newest. That is the
        whole reason the names are names and not numbers: every number would have been somebody's
        picture, and redrawing the wrong one costs half a minute, an image call and the panel.

        :meth:`_edit` is the other half, and it carries the one thing the diagram does not need -
        the staleness check, because an edit is asked for about a photograph they are looking at
        and half a minute is long enough for them to have moved on.
        """
        request = _tool_string(call.arguments, "request", imagine.MAX_REQUEST_CHARS)
        wanted = _tool_picture(call.arguments)
        self._log(f"[tool] edit_photo {request!r}{f' on {wanted!r}' if wanted else ''}")
        if not request:
            await self._send_tool_output(call.call_id, {"ok": False, "error": "nothing described"})
            await self._request_response()
            return
        shot = self._picture_named(wanted) if wanted else self._on_panel
        if wanted and shot is None:
            await self._send_tool_output(call.call_id, {
                "ok": False,
                "error": "no picture by that name",
                "pictures": [
                    {"name": one.name, "kind": one.what}
                    for one in reversed(self._pictures[-PICTURES_OFFERED:])
                ],
                "note": (
                    "Nothing was drawn and nothing was spent. These are the pictures you have "
                    "been shown, newest first. Call it again with one of these names if one is "
                    "plainly the one they mean - and if two of them could be, ask them which "
                    "rather than picking."
                ),
            })
            await self._request_response()
            return
        if shot is None:
            await self._send_tool_output(call.call_id, {
                "ok": False,
                "error": "no photo to edit",
                "note": (
                    "There is no photo yet. Take a look first, then edit that."
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
                "note": "Say so in a few words, and take a look if they want a new one.",
            })
            await self._request_response()
            return

        self.drawing_active = True
        task = tasks.start(_activity_line(call))
        self._spawn(self._edit(task, shot.path, request, self._turn_serial, shot.name))
        await self._send_tool_output(call.call_id, {
            "ok": True,
            "started": True,
            # Which one it actually took, so a mis-pick is catchable before the picture lands
            # rather than half a minute later, when it is on the glass.
            "picture": shot.name,
            "note": (
                "It is being redrawn now and takes about half a minute. Say in a few words that "
                "you are working on it, then carry on - you will be told when it lands or if it "
                "fails, and you will be shown the result."
            ),
        })
        await self._request_response()

    async def _edit(
        self, task: str, source: Path, request: str, turn: int, made_from: str = ""
    ) -> None:
        """The half minute. Runs on its own after :meth:`_run_edit_photo` has answered.

        ``turn`` is :attr:`_turn_serial` as it stood when the edit was asked for. If it has moved
        by the time the picture lands, they have spoken since and may have moved on entirely - so
        the arrival is mentioned rather than announced. That check used to ride in the tool output;
        it belongs here now, because the tool answers before there is anything to be stale about.

        Unlike the diagram, the model *is* shown what came back, and told plainly that it is a
        drawing - because this one looks like a photograph of their own bench, and a model that
        forgets that will start answering questions off it.
        """
        started = time.monotonic()
        try:
            # Inside the try rather than an else, for the reason _draw's comment gives: a failure
            # in the bookkeeping must not leave the model waiting to be told how it went.
            jpeg = await imagine.edit(source, request, self.settings)
            output, seen, kept = await asyncio.to_thread(self._keep_and_show_edit, jpeg, request)
            self._log(f"[tool] edit: {len(jpeg) // 1024} KB in {time.monotonic() - started:.1f}s")
        except imagine.ImagineError as exc:
            session.note("photo", by="edit", request=request[:80], error=str(exc))
            self._log(f"[tool] edit failed: {exc}", stream=sys.stderr)
            output, seen, kept = {"ok": False, "error": str(exc)}, None, None
        except Exception as exc:  # never leave the model waiting to be told how it went
            session.note("photo", by="edit", request=request[:80], error=f"{type(exc).__name__}")
            self._log(f"[tool] edit failed: {exc!r}", stream=sys.stderr)
            output, seen, kept = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, None, None
        finally:
            self.drawing_active = False

        made = ""
        if kept is not None:
            # The edit is the picture on the panel now, so it is what the next change starts
            # from: "now make it blue instead" means the one they are looking at. Set on the loop
            # from the path it was actually written to, and only when it was written - an edit
            # that reached no card is not a file anything can open again, and cannot be named.
            made = self._put_on_panel(kept, "edit")

        if not output.get("ok"):
            tasks.fail(task, str(output.get("error", "")))
            await self.announce(
                "[The change you were making to their photo could not be drawn: "
                f"{output.get('error', 'it failed')}. Tell them so in a few words.]"
            )
            return
        tasks.finish(task)
        # The picture first and the word after it, which is the order the tool output and the
        # image used to go in and for the same reason: an image cannot ride in a message that is
        # asking for a response, so it goes down as an item of its own and `announce` issues the
        # one response.create that covers both.
        if seen is not None:
            await self.add_edit(seen, request, made, made_from)
        if self._turn_serial != turn:
            await self.announce(
                "[The change you were making is now on their screen, but they have spoken since "
                "they asked for it and may have moved on. Mention it in a few words if it still "
                "fits, and do not launch into it.]"
            )
            return
        await self.announce(
            "[The change you were making is now on their screen. Say it is there, in a few "
            "words. Do not describe it - they are looking at it.]"
        )

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


def _look_tools(settings: Settings) -> list[RealtimeFunctionToolParam]:
    """The tool that takes a photo when they ask it to look, or nothing. ``CYCLOPS_LOOK=0``
    leaves SNAP as the only shutter, which is how it was from 2026-08-29 until this came back."""
    return [TAKE_A_LOOK_TOOL] if settings.look else []


def _video_tools(settings: Settings) -> list[RealtimeFunctionToolParam]:
    """The tool that plays a video, or nothing. Left out rather than refused, as ever.

    A flag of its own because it is the most invasive thing in the list. Everything else that
    takes the screen gives it back the instant somebody presses it, and is silent while it has
    it; this one holds the panel for minutes, talks over the room the whole time, and shuts the
    microphone for the duration. ``CYCLOPS_VIDEO=0`` is a way to find out what the box is like
    without it that is not a revert.
    """
    return [WATCH_VIDEO_TOOL] if settings.video else []


def _scratchpad_tools(settings: Settings) -> list[RealtimeFunctionToolParam]:
    """The one tool that writes on the panel, or nothing. Left out rather than refused, as ever.

    A flag of its own, and not because of what it costs - it costs nothing, which is the point of
    it. It is the switch to reach for if he turns out to put something on the glass every second
    sentence: this is the first tool here that takes the screen away from the person using it
    without being asked, and ``CYCLOPS_SCRATCHPAD=0`` is a way to find out what that is like that is
    not a revert.
    """
    return [SCRATCHPAD_TOOL] if settings.scratchpad and not settings.sketch else []


def _sketch_tools(settings: Settings) -> list[RealtimeFunctionToolParam]:
    """The tool that draws on the panel while it is being written, or nothing.

    It is the scratchpad, done the other way round, which is why turning it on turns that one
    off (see :func:`_scratchpad_tools`). Two tools that both put a screenful in front of the
    same person is not twice the feature: it is a choice the model has to make mid-sentence,
    every time, on a distinction only we can see. One door onto the glass, and ``CYCLOPS_SKETCH``
    picks which.

    Off by default, because it is the newer of the two and the older one has been used in anger.
    """
    return [SKETCH_TOOL] if settings.sketch else []


def _point_tools(settings: Settings) -> list[RealtimeFunctionToolParam]:
    """The tool that marks their picture, or nothing. Left out rather than refused, as ever.

    A flag of its own for the same reason the scratchpad has one: it is the second tool here
    that takes the screen away from the person using it without being asked, and this one holds
    a still photo over the live camera while it is up. ``CYCLOPS_POINTING=0`` is a way to find
    out what a bench feels like without it that is not a revert.
    """
    return [POINT_AT_TOOL] if settings.pointing else []


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


def _reference_beside(thumb: Path) -> dict:
    """What the sidecar in this folder says about one kept video. Empty when it says nothing.

    Read off the card rather than carried in the index, so recall.Item keeps the one shape it
    has for everything - a rebuilt index is expensive and a widened one invalidates every row.
    """
    try:
        kept = json.loads((thumb.parent / card.VIDEOS_NAME).read_text())
    except (OSError, ValueError):
        return {}
    found = kept.get(thumb.name) if isinstance(kept, dict) else None
    return found if isinstance(found, dict) else {}


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError:
        return b""


def _tool_request(arguments: str | None) -> str:
    """The video tool's required 'request' argument."""
    return _tool_string(arguments, "request", MAX_QUERY_CHARS)


def _tool_query(arguments: str | None) -> str:
    """The search tool's required 'query' argument."""
    return _tool_string(arguments, "query", MAX_QUERY_CHARS)


def _tool_show(arguments: str | None) -> bool:
    """recall's optional 'show' flag. Only a real true counts; anything else leaves the panel be."""
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return False
    return isinstance(args, dict) and args.get("show") is True


def _tool_picture(arguments: str | None) -> str:
    """edit_photo's optional 'picture' argument - which picture to change, by name.

    Empty means it named none, which is the documented default and the whole conversation's
    ordinary case: "change that" is the picture in play. Anything else is a name to resolve,
    and a name that resolves to nothing is refused rather than swapped for the newest.
    """
    return _tool_string(arguments, "picture", MAX_PICTURE_NAME_CHARS)


def _tool_code(arguments: str | None) -> str:
    """sketch's required 'code' argument: the Prefab program he wants on the glass."""
    return _tool_string(arguments, "code", sketch.MAX_SKETCH_CHARS)


def _tool_marks(arguments: str | None) -> str:
    """point_at's required 'marks' argument: the lines he wants drawn on their picture."""
    return _tool_string(arguments, "marks", MAX_MARKS_CHARS)


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


def _tool_steps(arguments: str | None) -> list[str]:
    """start_tutorial's 'steps' argument, read the way :func:`_tool_entries` reads its rows."""
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return []
    return tutorial.clean(args.get("steps")) if isinstance(args, dict) else []


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
    is elided again on the way to the panel, and a trailing ellipsis next to the cursor standing
    in for one - "there was more of this" beside "this is still happening" - says nothing.
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
    caption about work in flight from one about a state, and decides whether to blink a cursor
    underneath it. See :data:`cyclops.overlay.BUSY_MARK`.
    """
    args = call.arguments
    if call.name == "web_search":
        return _phrase("searching for", _tool_query(args), "searching the web")
    if call.name == "take_a_look":
        return "taking a look…"
    if call.name == "draw_diagram":
        return _phrase("drawing", _tool_string(args, "request", MAX_QUERY_CHARS), "drawing")
    # Barely seen - the browser covers this strip about a second later - but the chain wants no
    # silent branches, and the recording is still watching while it is up.
    if call.name == "write_on_scratchpad":
        return "putting that on the screen…"
    # Seen for about as long as it takes to read, and that is right: the mark is up before the
    # line has finished typing, and the mark is the answer.
    if call.name == "point_at":
        return "pointing…"
    # Up for a fraction of a second: by the time a sketch has two lines in it the browser is
    # already over this strip. It is here for the recording and for the seconds before the
    # first line compiles.
    if call.name == "sketch":
        return "drawing that…"
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
    # Up for the few seconds before the browser covers this strip, which is most of the wait:
    # the picture only lands once a video has been chosen, fetched and seeked.
    if call.name == "watch_video":
        return _phrase("finding a video of", _tool_request(args), "finding a video")
    if call.name == "forget_data":
        return _phrase("forgetting", _tool_key(args), "rubbing that out")
    # A frame or two at most - the step's own label takes the row back the moment the call
    # returns - but the chain wants no silent branches.
    if call.name == "start_tutorial":
        return "setting out the steps…"
    if call.name == "advance_tutorial":
        return "next step…"
    if call.name == "end_tutorial":
        return "putting the steps away…"
    return "working…"  # a tool the model invented; it still gets an answer, so it still gets a line
