#!/usr/bin/env python3
"""Run one scripted, typed conversation through the real session config N times, and tally it.

    uv run python tools/talk_probe.py --runs 10 --out runs.jsonl
    uv run python tools/talk_probe.py --tally runs.jsonl

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
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

from openai.types.realtime import RealtimeServerEvent

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
    """What came back for one typed turn: transcripts, and how many responses it cost."""

    def __init__(self) -> None:
        self.done = asyncio.Event()
        self.transcripts: list[str] = []
        self.responses = 0
        self.tools: list[str] = []

    def __call__(self, event: RealtimeServerEvent) -> None:
        match event.type:
            case "response.created":
                self.responses += 1
            case "response.output_audio_transcript.done":
                self.transcripts.append(event.transcript)
            case "response.done":
                calls = [call.name for call in function_calls(event.response)]
                self.tools.extend(calls)
                if not calls:  # a tool-call response is followed by the real answer
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


async def _one_run(run: int, out) -> None:
    settings = load_settings()
    agent = VoiceAgent(settings)
    agent_task = asyncio.create_task(agent.run())
    try:
        await _await_or_fail(agent_task, agent.ready, READY_TIMEOUT_S)
        for index, line in enumerate(SCRIPT, start=1):
            turn = Turn()
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
            }
            out.write(json.dumps(record) + "\n")
            out.flush()
            print(f"· run {run} turn {index}: {turn.text!r}", flush=True)
    finally:
        await agent.close()
        await asyncio.gather(agent_task, return_exceptions=True)


async def _probe(runs: int, out_path: Path) -> int:
    with out_path.open("a", encoding="utf-8") as out:
        for run in range(1, runs + 1):
            try:
                await _one_run(run, out)
            except (TimeoutError, RuntimeError) as exc:
                print(f"run {run} failed: {exc}", file=sys.stderr)
                return 1
    return 0


def _hits(text: str, words: tuple[str, ...]) -> list[str]:
    low = f" {text.lower()} "
    return [w for w in words if re.search(rf"(?<![a-z0-9]){re.escape(w)}(?![a-z0-9])", low)]


def tally(path: Path) -> None:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
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
    parser.add_argument("--blind", type=Path, nargs=3, metavar=("A", "B", "OUT"),
                        help="shuffle two JSONLs into one anonymous list to read")
    parser.add_argument("--score", type=Path, nargs=2, metavar=("KEY", "MARKS"),
                        help="tally marks made against a --blind list, per arm")
    args = parser.parse_args()
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
        sys.exit(asyncio.run(_probe(args.runs, args.out)))
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
