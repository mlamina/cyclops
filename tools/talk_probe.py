#!/usr/bin/env python3
"""Run one scripted, typed conversation through the real session config N times, and tally it.

    uv run python tools/talk_probe.py --runs 10 --out runs.jsonl
    uv run python tools/talk_probe.py --tally runs.jsonl
    uv run python tools/talk_probe.py --script tutorial --runs 10 --out walk.jsonl
    uv run python tools/talk_probe.py --script manual --runs 5 --out oxi.jsonl

``--script manual`` sets its own bench up: the OXI One manual from the fixtures folder and nothing
else on the card, with the recall index, the panel file and the task ledger all in a scratch
folder rather than ``~/.cache/cyclops``. Its tally is per question: every recall and the page it
read, whether it asked for the page on the screen, seconds to the answer, and any line that names
the manual's title or a page number, or owns up to a page that missed, and the line spoken before
each question's first look. ``--script manual-cold`` is the same
bench with only the notes-below-the-grid question, asked before anything has been read.

No mic, no speaker, no camera and no session folder: ``VoiceAgent`` + ``send_text``, the way
``cyclops-smoke`` does its text turns. Every run is a fresh session, so the first turn is a real
first turn and the greeting is a real greeting. Point the card somewhere scratch with the
``CYCLOPS_*_DIR`` variables before running, or the model will open his real projects.

What it measures is what a prompt change to HOW YOU TALK can quietly buy: words per turn,
whether a turn cost more than one response, how the greeting varies from run to run, and
which turns carried a film reference or a compliment. The last two are a word list plus a
pair of eyes - the tally flags candidates and prints the lines, and somebody reads them.

Warmth is not a fault to be driven to zero any more: encouragement that is earned is part of
the character and empty flattery is not, and no word list can tell those apart. The bucket
flags both and the reading sorts them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import statistics
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from openai import AsyncOpenAI
from openai.types.realtime import RealtimeServerEvent

from cyclops import card, panel, recall, session, tasks
from cyclops.agent import VoiceAgent, function_calls
from cyclops.config import ConfigError, load_settings

READY_TIMEOUT_S = 30.0
TURN_TIMEOUT_S = 60.0

# One session at a bench: a greeting, a question with a genuine rhyme in it (a red power light
# in a robot's head), a method to bless, a finished job held up, a pause, and a recall. The
# rhyme is there once so the tally can see whether it gets taken; everything else is plain.
SCRIPT = (
    "Hey.",
    "I'm putting a big red LED in the head of this robot arm build, it's the power light. "
    "Three point three volts or five?",
    "I'm going to run the wires down the arm and zip tie them every few centimetres. "
    "Sound alright?",
    "Check it out, I got all twelve wires down the arm and into the base. Took me an hour.",
    "Okay, going to count the standoffs, hang on.",
    "What did we say about the LED voltage?",
)

# A walkthrough asked for, moved on twice, and stopped halfway - with two turns in the middle that
# are NOT "that step is done": a pause and a question about the step. Those two are the ones that
# say whether it waits, and the tally lays out the tools per turn so a wrong advance stands out.
TUTORIAL_SCRIPT = (
    "Hey.",
    "I'm replacing the cartridge in my kitchen mixer tap. Can you walk me through it step by "
    "step?",
    "Okay, the water's off.",
    "Hang on, let me find a screwdriver.",
    "Which way does the big nut undo?",
    "Right, done that.",
    "Actually, forget it, stop this. I'll call a plumber.",
)
# Yesterday's OXI One session, question by question: a slide, the notes below the grid (page 12
# has it and page 18 is what three rewordings kept landing on), the Div button, copying a pattern,
# then asking to see the page and asking where the answer came from. Only the last one may name
# the manual or a page, and only the one before it may put a page on the screen.
MANUAL_SCRIPT = (
    "Hey. I've got the OXI One out tonight.",
    "How do I add a slide between two notes?",
    "How do I get to the notes below the ones that are lit on the grid?",
    "Which one is the Div button?",
    "How do I copy a pattern?",
    "Show me that page.",
    "Where's that from?",
)
# The notes-below-the-grid question with nothing read before it. In MANUAL_SCRIPT the slide
# question usually reads page 12 first, so the page is already in hand by the time it is asked.
MANUAL_COLD_SCRIPT = MANUAL_SCRIPT[:1] + MANUAL_SCRIPT[2:3]
SCRIPTS = {
    "bench": SCRIPT,
    "tutorial": TUTORIAL_SCRIPT,
    "manual": MANUAL_SCRIPT,
    "manual-cold": MANUAL_COLD_SCRIPT,
}

# Where --script manual finds its one manual, and where it keeps everything it would otherwise
# write under ~/.cache/cyclops. CYCLOPS_MANUALS_DIR overrides the first.
FIXTURE_MANUALS = Path.home() / "code" / "cyclops-fixtures" / "manuals"
MANUAL_BENCH = Path(tempfile.gettempdir()) / "cyclops-probe-manual"

# Candidates only. A line that matches is printed for a human to read; one that does not can
# still be a reference, which is why the tally prints every turn of every run as well.
REFERENCE_WORDS = (
    "terminator", "skynet", "t-800", "t800", "endoskeleton", "cyberdyne", "sarah connor",
    "i'll be back", "hasta la vista", "come with me if you want to live",
    "matrix", "neo", "morpheus", "red pill", "blue pill", "there is no spoon", "agent smith",
    "hal", "pod bay", "dave", "2001", "daisy",
    "alien", "nostromo", "xenomorph", "ripley", "weyland",
    "blade runner", "replicant", "tears in rain",
    "star wars", "r2", "c-3po", "threepio", "the force", "vader", "death star", "skywalker",
    "star trek", "warp", "beam me up", "borg", "resistance is futile", "spock", "picard",
    "jarvis", "iron man", "stark", "wall-e", "robocop", "cylon", "battlestar", "tron",
    "hitchhiker", "don't panic", "marvin", "asimov", "three laws", "ex machina", "westworld",
    "bender", "futurama", "short circuit", "johnny 5", "baymax", "transformers", "optimus",
    "robot wars", "battlebots", "flux capacitor", "delorean", "great scott",
    "portal", "glados", "aperture", "wintermute", "neuromancer", "cyberpunk", "ghost in the shell",
)
WARMTH_WORDS = (
    "great job", "nice work", "good job", "well done", "nicely done", "good call", "smart",
    "clever", "love that", "love it", "awesome", "excellent", "brilliant", "fantastic",
    "amazing", "impressive", "perfect", "kudos", "proud", "solid work", "nice one", "sweet",
    "you're good", "you've clearly", "you clearly", "you're clearly", "good work", "nice,",
    "nice.", "nice!", "great,", "great.", "great!", "nice progress", "good progress",
    "solid work", "solid milestone", "milestone", "congrats", "congratulations", "you did",
    "you've done", "hats off", "respect", "great question", "good question", "well spotted",
    "nice catch", "good thinking", "right call", "the right call", "well played", "tidy",
    "that's the one", "earned", "not bad", "decent", "handsome", "clean job", "worth it",
)


class Turn:
    """What came back for one typed turn: transcripts, how many responses it cost, and when.

    ``answer_s`` is from the question going out to the first sound of the response that answers
    it - the one with no tool call in it - which is the wait somebody at the bench actually has.
    """

    def __init__(self) -> None:
        self.done = asyncio.Event()
        self.transcripts: list[str] = []
        self.responses = 0
        self.tools: list[str] = []
        self.recalls: list[dict] = []
        self.started = time.monotonic()
        self.heard: dict[str, float] = {}  # response id -> when its first audio arrived
        self.answer_s = 0.0
        self.first_sound_s = 0.0
        # (response id, output index, transcript), so what was said before a call can be told
        # from what was said after it: a spoken line and a call in one response are two items.
        self.spoken: list[tuple[str, int, str]] = []
        self.before_recall: str | None = None  # None until the first recall call of the turn

    def __call__(self, event: RealtimeServerEvent) -> None:
        match event.type:
            case "response.created":
                self.responses += 1
            case "response.output_audio.delta":
                if not self.heard:
                    self.first_sound_s = round(time.monotonic() - self.started, 2)
                self.heard.setdefault(event.response_id, time.monotonic())
            case "response.output_audio_transcript.done":
                self.transcripts.append(event.transcript)
                self.spoken.append((event.response_id, event.output_index, event.transcript))
            case "response.done":
                calls = [call.name for call in function_calls(event.response)]
                self.tools.extend(calls)
                if "recall" in calls and self.before_recall is None:
                    output = event.response.output or []
                    at = next(i for i, item in enumerate(output)
                              if item.type == "function_call" and item.name == "recall")
                    self.before_recall = " ".join(
                        text.strip() for rid, index, text in self.spoken
                        if rid != event.response.id or index < at
                    )
                if not calls:  # a tool-call response is followed by the real answer
                    spoke = self.heard.get(event.response.id, time.monotonic())
                    self.answer_s = round(spoke - self.started, 2)
                    self.done.set()

    @property
    def text(self) -> str:
        return " ".join(t.strip() for t in self.transcripts if t.strip())


async def _await_or_fail(agent_task: asyncio.Task, event: asyncio.Event, limit_s: float) -> None:
    waiter = asyncio.ensure_future(event.wait())
    done, _ = await asyncio.wait(
        {agent_task, waiter}, timeout=limit_s, return_when="FIRST_COMPLETED"
    )
    waiter.cancel()
    if waiter in done:
        return
    if agent_task in done:
        agent_task.result()
        raise RuntimeError("session closed before the turn finished")
    raise TimeoutError(f"no response within {limit_s:.0f}s")


def _spy_on_recall(agent: VoiceAgent, current: list[Turn]) -> None:
    """Note every recall the agent runs - what it asked for, and what it was handed back."""
    real = agent._recall

    async def spy(query: str, project: str, *, show: bool = False) -> dict:
        out = await real(query, project, show=show)
        current[0].recalls.append(
            {"query": query, "show": show, "title": out.get("title", ""), "hits": out.get("hits")}
        )
        return out

    agent._recall = spy  # type: ignore[method-assign]


async def _one_run(run: int, out, script: tuple[str, ...] = SCRIPT) -> None:
    settings = load_settings()
    agent = VoiceAgent(settings)
    current = [Turn()]
    _spy_on_recall(agent, current)
    agent_task = asyncio.create_task(agent.run())
    try:
        await _await_or_fail(agent_task, agent.ready, READY_TIMEOUT_S)
        for index, line in enumerate(script, start=1):
            turn = current[0] = Turn()
            agent.on_event = turn
            await agent.send_text(line)
            await _await_or_fail(agent_task, turn.done, TURN_TIMEOUT_S)
            record = {
                "run": run,
                "turn": index,
                "you": line,
                "cyclops": turn.text,
                "words": len(turn.text.split()),
                "responses": turn.responses,
                "tools": turn.tools,
                "recalls": turn.recalls,
                "answer_s": turn.answer_s,
                "first_sound_s": turn.first_sound_s,
                "before_recall": turn.before_recall,
            }
            out.write(json.dumps(record) + "\n")
            out.flush()
            print(f"· run {run} turn {index}: {turn.text!r}", flush=True)
    finally:
        await agent.close()
        await asyncio.gather(agent_task, return_exceptions=True)


async def _manual_bench() -> None:
    """One manual on the card and nothing else, and nothing written under ~/.cache/cyclops.

    The index is built here rather than by ``cyclops-index``, which would also read and caption,
    and it is kept between runs: 162 embeddings are cheap, but not free every time.
    """
    MANUAL_BENCH.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CYCLOPS_MANUALS_DIR", str(FIXTURE_MANUALS))
    for name in ("SESSIONS", "PROJECTS", "CAPTURES"):
        os.environ[f"CYCLOPS_{name}_DIR"] = str(MANUAL_BENCH / name.lower())
    os.environ["CYCLOPS_ABOUT_FILE"] = str(MANUAL_BENCH / "about-you.md")
    recall.RECALL_FILE = MANUAL_BENCH / "recall.npz"
    panel.PANEL_FILE = MANUAL_BENCH / "panel.json"
    panel.PANEL_STILL_FILE = MANUAL_BENCH / "panel-still.json"
    tasks.TASKS_FILE = MANUAL_BENCH / "tasks.yaml"
    tasks.TASKS_LOCK = MANUAL_BENCH / "tasks.lock"
    # The Pi's kiosk, as far as the agent can tell: a page asked for goes "up", so the model is
    # told they can see it - exactly what it would be told on the glass - instead of "no panel".
    panel.show = lambda: True
    settings = load_settings()
    items = recall.corpus(settings)
    if not any(item.kind == "page" for item in items):
        raise ConfigError(f"no manual pages under {settings.manuals_dir}")
    if [i.key for i in recall.load(recall.RECALL_FILE).items] == [i.key for i in items]:
        return
    client = AsyncOpenAI(api_key=settings.api_key)
    try:
        vectors = await recall.embed([item.text for item in items], client)
    finally:
        await client.close()
    card.write_bytes(recall.RECALL_FILE, recall.dump(recall.Index(items=items, vectors=vectors)))
    print(f"· indexed {len(items)} items into {recall.RECALL_FILE}", flush=True)


async def _probe(runs: int, out_path: Path, script: tuple[str, ...] = SCRIPT) -> int:
    if script in (MANUAL_SCRIPT, MANUAL_COLD_SCRIPT):
        await _manual_bench()
    with out_path.open("a", encoding="utf-8") as out:
        for run in range(1, runs + 1):
            try:
                await _one_run(run, out, script)
            except (TimeoutError, RuntimeError) as exc:
                print(f"run {run} failed: {exc}", file=sys.stderr)
                return 1
    return 0


# ------------------------------------------------------------------ one wake, on a made-up clock

# "The greeting knows when it is" cannot be checked by ten runs at one moment - ten runs at one
# moment is one situation, sampled ten times. So each of these is a whole situation: a clock, and
# a card with the right sessions already on it. Everything else about the session is identical,
# which is what makes a difference between two greetings attributable to the moment.
MONDAY = datetime(2026, 9, 14, 8, 10)  # a Monday morning; every other situation is offset from it


def _at(day: int, hour: int, minute: int = 0) -> datetime:
    return (MONDAY + timedelta(days=day)).replace(hour=hour, minute=minute)


# (name, the clock when the eye opens, [(when a past session started, how many minutes it ran)])
SITUATIONS: tuple[tuple[str, datetime, tuple[tuple[datetime, int], ...]], ...] = (
    ("mon-first-thing", _at(0, 8, 10), ((_at(-3, 15, 40), 35),)),
    ("fri-fifth-wake", _at(4, 16, 40), tuple((_at(4, h, 0), 20) for h in (9, 11, 13, 15))),
    ("straight-back", _at(1, 14, 5), ((_at(1, 13, 30), 33),)),
    ("three-weeks-dark", _at(2, 10, 0), ((_at(-19, 11, 0), 40),)),
    ("saturday-morning", _at(5, 9, 30), ((_at(4, 19, 20), 25),)),
    ("late-night", _at(3, 23, 40), tuple((_at(3, h, 0), 30) for h in (14, 19))),
    ("never-switched-on", _at(6, 11, 0), ()),
    ("sunday-evening", _at(6, 19, 20), ((_at(6, 17, 50), 30),)),
)

# One sentence, the same one in every fabricated session, so the recap block is present and
# identical everywhere and the only thing that moves between situations is the clock.
FAKE_SUMMARY = (
    "# Robot arm wiring\n\nRan the twelve servo wires down the arm and into the base, and "
    "settled on 3.3 volts for the power LED with a 220 ohm resistor.\n"
)


def _lay_the_card(root: Path, history: tuple[tuple[datetime, int], ...]) -> Path:
    """A sessions directory holding the past this situation says it has."""
    sessions = root / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    for started, minutes in history:
        folder = sessions / started.strftime(card.STAMP)
        folder.mkdir(parents=True, exist_ok=True)
        rows = [
            {"type": "session", "uuid": "probe"},
            {"type": "you", "text": "how deep should this go?"},
            {"type": "cyclops", "text": "about 40 mm."},
            {"type": "end", "seconds": minutes * 60},
        ]
        card.write_text(folder / card.LOG_NAME, "".join(json.dumps(r) + "\n" for r in rows))
        card.write_text(folder / card.PAGE_NAME, "# a session\n")
        card.write_text(folder / card.SUMMARY_NAME, FAKE_SUMMARY)
    return sessions


async def _one_wake(name: str, when: datetime, sessions: Path, take: int, out) -> None:
    """Switch on once into a fabricated moment and keep only the first thing it says."""
    os.environ["CYCLOPS_SESSIONS_DIR"] = str(sessions)
    settings = load_settings()
    told = session.now_context(settings, now=when)
    real = session.now_context
    session.now_context = lambda s, now=None: real(s, now=when)  # type: ignore[assignment]
    agent = VoiceAgent(settings)
    agent_task = asyncio.create_task(agent.run())
    try:
        await _await_or_fail(agent_task, agent.ready, READY_TIMEOUT_S)
        turn = Turn()
        agent.on_event = turn
        await agent.send_text("Hey.")
        await _await_or_fail(agent_task, turn.done, TURN_TIMEOUT_S)
        out.write(json.dumps({
            "situation": name,
            "take": take,
            "told": told.note,
            "greeting": turn.text,
            "words": len(turn.text.split()),
        }) + "\n")
        out.flush()
        print(f"· {name} #{take}: [{told.note}] {turn.text!r}", flush=True)
    finally:
        session.now_context = real  # type: ignore[assignment]
        await agent.close()
        await asyncio.gather(agent_task, return_exceptions=True)


async def _wakes(repeat: int, out_path: Path) -> int:
    root = Path(tempfile.mkdtemp(prefix="cyclops-wakes-"))
    with out_path.open("a", encoding="utf-8") as out:
        for take in range(1, repeat + 1):
            for name, when, history in SITUATIONS:
                sessions = _lay_the_card(root / f"{name}-{take}", history)
                try:
                    await _one_wake(name, when, sessions, take, out)
                except (TimeoutError, RuntimeError) as exc:
                    print(f"{name} #{take} failed: {exc}", file=sys.stderr)
                    return 1
    return 0


# ------------------------------------------------------------------ the words it leans on

# "Ten distinct greetings" turned out to be ten variations on two words: every line opened
# "Morning" or "Back again" and nearly every one reached for "the bench". Distinctness is
# therefore not the measure - a line can be unique and still be the same line. This counts what
# recurs *across* greetings: the first word, the first two words, and every content word. A noun
# that turns up in a third of them is a crutch, and the word "bench" additionally assumes a
# workshop that not everybody has.
STOPWORDS = frozenset("""
a an and are as at back be been but by can did do for from get got had has have he her here his
how i if in into is it its just let lets me my no not of off on once one or our out so some
than that the their them then there these they this to too up us was we were what when where
which while who will with you your yours i'm im you're we're let's thats it's don't going go
""".split())
# Not a noun, but the thing "the bench" is a member of: a line that only works in a workshop.
ROOM_WORDS = ("bench", "workbench", "workshop", "shop", "garage", "shed", "bench-top")
# The clock, in words. These are what the greeting is built out of - a line that names the day
# is doing what it was told - so they are counted and printed but they are not crutches, and the
# verdict is taken on everything else. "morning" recurring as the FIRST word is still a fail:
# that is the opening-word count, which is where the stale opener was measured in the first place.
TIME_WORDS = frozenset("""
monday tuesday wednesday thursday friday saturday sunday weekend weekday today tonight yesterday
morning afternoon evening night midday noon midnight hour hours minute minutes day days week
weeks month months
""".split())


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z'-]*", text.lower().replace("’", "'"))


def _greetings(paths: list[Path]) -> list[str]:
    """Every first-thing-said in these files, whether they are runs or wakes."""
    lines = []
    for path in paths:
        for row in (json.loads(x) for x in path.read_text().splitlines() if x.strip()):
            said = row.get("greeting") if "greeting" in row else (
                row["cyclops"] if row.get("turn") == 1 else None
            )
            if said and said.strip():
                lines.append(said.strip())
    return lines


def props(paths: list[Path], limit: float = 0.25) -> None:
    """First words, first pairs and content words, as a share of the greetings they appear in."""
    lines = _greetings(paths)
    n = len(lines)
    print(f"greetings={n}  (pass line: nothing over {limit:.0%})")

    def show(label: str, counts: Counter, top: int = 8) -> bool:
        worst = counts.most_common(1)[0][1] / n if counts else 0
        print(f"\n{label}: worst is {worst:.0%}  {'OK' if worst <= limit else 'OVER'}")
        for word, hits in counts.most_common(top):
            if hits > 1:
                print(f"  {hits:3d}  {hits / n:4.0%}  {word}")
        return worst <= limit

    firsts = Counter(_words(line)[:1][0] for line in lines if _words(line))
    pairs = Counter(" ".join(_words(line)[:2]) for line in lines if len(_words(line)) > 1)
    content = Counter()
    for line in lines:
        for word in {w for w in _words(line) if w not in STOPWORDS and len(w) > 2}:
            content[word] += 1
    nouns = Counter({w: n for w, n in content.items() if w not in TIME_WORDS})
    ok = show("opening word", firsts)
    ok = show("opening phrase", pairs) and ok
    ok = show("word that is not the clock", nouns, top=12) and ok
    show("every content word, clock included", content, top=12)
    room = [line for line in lines if any(w in _words(line) for w in ROOM_WORDS)]
    print(f"\nassumes a workshop: {len(room)} of {n} ({len(room) / n:.0%})")
    for line in room:
        print(f"  {line!r}")
    print(f"\nverdict: {'PASS' if ok and not room else 'FAIL'}")


def wake_tally(path: Path) -> None:
    """Every greeting, grouped by the situation it was said into."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    order = [name for name, _, _ in SITUATIONS]
    words = [r["words"] for r in rows if r["words"]]
    print(f"wakes={len(rows)} situations={len(order)}")
    print(f"words/greeting: median={statistics.median(words)} mean={statistics.mean(words):.1f}"
          f" max={max(words)}")
    seen: set[str] = set()
    for name in order:
        mine = [r for r in rows if r["situation"] == name]
        if not mine:
            continue
        print(f"\n{name}  [{mine[0]['told']}]")
        for r in mine:
            print(f"  #{r['take']} ({r['words']}w) {r['greeting']!r}")
        seen.update(r["greeting"].strip().lower() for r in mine)
    print(f"\ndistinct greetings overall: {len(seen)} of {len(rows)}")


def _hits(text: str, words: tuple[str, ...]) -> list[str]:
    low = f" {text.lower()} "
    return [w for w in words if re.search(rf"(?<![a-z0-9]){re.escape(w)}(?![a-z0-9])", low)]


# A line that names where the answer came from: the manual's title or a page number. Only
# "where's that from?" should have one. "Let me check the manual" is the announcement, not this.
CITATION = re.compile(
    r"\bpages?\s+\d|\bp\.\s*\d|\boxi\s*one\W+(user\s+)?(manual|guide)|\buser\s+(manual|guide)",
    re.IGNORECASE,
)
# A line that owns up to a page that missed, or offers to look instead of looking.
MISS_WORDS = (
    "didn't have", "doesn't have", "did not have", "does not have", "wasn't on", "isn't on",
    "not on that page", "doesn't cover", "does not cover", "doesn't say", "does not say",
    "didn't say", "want me to", "i can look", "i could look", "shall i", "should i look",
    "look further", "look for that", "keep looking", "look again", "if you want, i",
    "not seeing", "can't find", "couldn't find", "can't reliably", "pages i've", "none of them",
    "if you can put", "ask me to look", "doesn't show", "does not show", "isn't documented",
    "not documented", "doesn't clearly", "pages we", "page here", "pages i", "wrong page",
)
PAGE_NUMBER = re.compile(r"page (\d+)")
# Saying it is going to the manual. Once before the first look is the whole allowance.
ANNOUNCE = re.compile(
    r"\b(let me|let's|i'll|i will|checking|looking|check|look)\b[^.!?]*\bmanual\b", re.IGNORECASE
)


def _page(title: str) -> str:
    found = PAGE_NUMBER.search(title or "")
    return f"p{found.group(1)}" if found else (title or "nothing")[:24]


def manual_tally(records: list[dict]) -> None:
    """Per question: what it read, whether it asked for the screen, how long, and what it said."""
    for r in records:  # the transcripts say "doesn’t"; every word list here says "doesn't"
        r["cyclops"] = r["cyclops"].replace("\u2019", "'")
        if r.get("before_recall"):
            r["before_recall"] = r["before_recall"].replace("\u2019", "'")
    runs = sorted({r["run"] for r in records})
    print(f"runs={len(runs)} turns={len(records)}")
    for turn in sorted({r["turn"] for r in records}):
        mine = [r for r in records if r["turn"] == turn]
        waits = [r["answer_s"] for r in mine]
        shows = sum(call["show"] for r in mine for call in r["recalls"])
        calls = sum(len(r["recalls"]) for r in mine)
        looked = [r for r in mine if r["recalls"]]
        announced = [r for r in looked if ANNOUNCE.search(r.get("before_recall") or "")]
        sounds = [r["first_sound_s"] for r in mine if "first_sound_s" in r] or [0.0]
        print(f"\nt{turn} {mine[0]['you']!r}")
        print(f"  seconds to answer: median={statistics.median(waits):.1f} max={max(waits):.1f}"
              f"   to first sound: median={statistics.median(sounds):.1f}"
              f"   recalls={calls}  with show={shows}"
              f"   announced before the first look: {len(announced)} of {len(looked)}")
        for r in mine:
            read = " -> ".join(
                _page(c["title"]) + ("[show]" if c["show"] else "") for c in r["recalls"]
            ) or "-"
            flags = []
            if CITATION.search(r["cyclops"]):
                flags.append("NAMES")
            if _hits(r["cyclops"], MISS_WORDS):
                flags.append("MISS?")
            if len(ANNOUNCE.findall(r["cyclops"])) > 1:
                flags.append("TWICE")
            print(f"  run {r['run']} {r['answer_s']:4.1f}s  read {read}  {' '.join(flags)}")
            if r["recalls"] and "before_recall" in r:
                said = r["before_recall"] or ""
                print(f"      before the first look ({len(said.split())}w): {said!r}")
            print(f"      {r['cyclops']!r}")
    cited = [r for r in records if CITATION.search(r["cyclops"])]
    missed = [(r, _hits(r["cyclops"], MISS_WORDS)) for r in records]
    missed = [(r, h) for r, h in missed if h]
    print(f"\nlines naming the manual's title or a page number: {len(cited)}")
    for r in cited:
        print(f"  run {r['run']} t{r['turn']}: {r['cyclops']!r}")
    print(f"lines owning up to a miss or offering to look: {len(missed)}")
    for r, h in missed:
        print(f"  run {r['run']} t{r['turn']} {h}: {r['cyclops']!r}")


def tally(path: Path) -> None:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if records and records[0]["you"] == MANUAL_SCRIPT[0]:
        manual_tally(records)
        return
    runs = sorted({r["run"] for r in records})
    spoken = [r["words"] for r in records if r["words"] > 0]
    silent = sum(1 for r in records if r["words"] == 0)
    multi = [r for r in records if r["responses"] != 1]
    tools = [r for r in records if r["tools"]]
    greetings = Counter(r["cyclops"] for r in records if r["turn"] == 1)
    print(f"runs={len(runs)} turns={len(records)} silent={silent}")
    print(
        f"words/turn: median={statistics.median(spoken)} mean={statistics.mean(spoken):.1f} "
        f"max={max(spoken)}"
    )
    print(f"turns costing != 1 response: {len(multi)}   turns with tool calls: {len(tools)}")
    print("tools per turn, across runs:")
    for turn in sorted({r["turn"] for r in records}):
        called = Counter(" + ".join(r["tools"]) or "-" for r in records if r["turn"] == turn)
        you = next(r["you"] for r in records if r["turn"] == turn)
        print(f"  t{turn} {you[:48]!r}")
        for calls, n in called.most_common():
            print(f"      {n}x {calls}")
    print(f"distinct greetings: {len(greetings)} of {len(runs)}")
    for line, n in greetings.most_common():
        print(f"  {n}x {line!r}")
    for label, words in (("reference", REFERENCE_WORDS), ("warmth", WARMTH_WORDS)):
        flagged = [(r, _hits(r["cyclops"], words)) for r in records]
        flagged = [(r, h) for r, h in flagged if h]
        by_run = Counter(r["run"] for r, _ in flagged)
        print(f"{label} candidates: {len(flagged)} turns in {len(by_run)} of {len(runs)} runs")
        for r, h in flagged:
            print(f"  run {r['run']} turn {r['turn']} {h}: {r['cyclops']!r}")
    print("\nevery turn:")
    for r in records:
        print(f"  run {r['run']} t{r['turn']} ({r['words']}w) {r['cyclops']!r}")


def blind(a: Path, b: Path, out: Path) -> None:
    """Shuffle two runs' turns together with the arm hidden, for reading without knowing which.

    "Recognisably funnier and more opinionated than the tool it replaced" is a comparison, and a
    comparison you make while knowing which side you wrote is not one. This writes the lines in
    one shuffled list with an id each, and the key beside it; mark the list, then --score it.
    """
    rows = []
    for arm, path in (("A", a), ("B", b)):
        for r in (json.loads(line) for line in path.read_text().splitlines() if line.strip()):
            if r["turn"] != 1 and r["cyclops"].strip():  # the greeting has its own criterion
                rows.append({"arm": arm, "turn": r["turn"], "text": r["cyclops"]})
    random.seed(2)
    random.shuffle(rows)
    key = out.with_suffix(".key.json")
    key.write_text(json.dumps({str(i): r for i, r in enumerate(rows, 1)}, indent=1))
    lines = [f"{i}\tt{r['turn']}\t{r['text']}" for i, r in enumerate(rows, 1)]
    out.write_text("\n".join(lines) + "\n")
    print(f"{len(rows)} lines -> {out}   key -> {key}")


def score(key_path: Path, marks_path: Path) -> None:
    """Join marks (``<id> <letter>`` per line) back onto the key and tally per arm."""
    key = json.loads(key_path.read_text())
    marks: dict[str, str] = {}
    for line in marks_path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in key:
            marks[parts[0]] = parts[1].lower()
    tally_: dict[str, Counter] = {"A": Counter(), "B": Counter()}
    for ident, mark in marks.items():
        tally_[key[ident]["arm"]][mark] += 1
    for arm in ("A", "B"):
        total = sum(tally_[arm].values())
        got = ", ".join(f"{m}={n} ({n / total:.0%})" for m, n in sorted(tally_[arm].items()))
        print(f"arm {arm}: n={total}  {got}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--out", type=Path, help="JSONL to append each turn to")
    parser.add_argument("--tally", type=Path, help="tally a JSONL written earlier and exit")
    parser.add_argument("--script", choices=sorted(SCRIPTS), default="bench",
                        help="which conversation to type at it")
    parser.add_argument("--blind", type=Path, nargs=3, metavar=("A", "B", "OUT"),
                        help="shuffle two JSONLs into one anonymous list to read")
    parser.add_argument("--score", type=Path, nargs=2, metavar=("KEY", "MARKS"),
                        help="tally marks made against a --blind list, per arm")
    parser.add_argument("--wakes", type=Path,
                        help="JSONL to append one greeting per made-up situation to")
    parser.add_argument("--repeat", type=int, default=2, help="takes per situation, for --wakes")
    parser.add_argument("--wake-tally", type=Path, help="tally a --wakes JSONL and exit")
    parser.add_argument("--props", type=Path, nargs="+",
                        help="count openers and recurring nouns over every greeting in these")
    args = parser.parse_args()
    if args.props:
        props(args.props)
        return
    if args.wake_tally:
        wake_tally(args.wake_tally)
        return
    if args.wakes:
        try:
            sys.exit(asyncio.run(_wakes(args.repeat, args.wakes)))
        except ConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(2)
    if args.tally:
        tally(args.tally)
        return
    if args.blind:
        blind(*args.blind)
        return
    if args.score:
        score(*args.score)
        return
    if not args.out:
        parser.error("--out is required unless --tally is given")
    try:
        sys.exit(asyncio.run(_probe(args.runs, args.out, SCRIPTS[args.script])))
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
