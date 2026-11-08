#!/usr/bin/env python3
"""Which tool the real voice model reaches for, by wording (arc-eng/hq#669). Typed turns, tally out.

    python tools/lapse_voice_probe.py --runs 3 --out lapse-voice.jsonl

The time-lapse is a stand-in that only writes down what it was asked, so nothing is filmed and
nothing is filed; no session folder is opened. Point the ``CYCLOPS_*_DIR`` variables at scratch
before running, as for ``talk_probe.py``. Each script is a fresh session; "later" scripts start
with a run already going, as one that outlived its conversation would.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import talk_probe  # noqa: E402

from cyclops import lapse  # noqa: E402
from cyclops.agent import VoiceAgent  # noqa: E402
from cyclops.config import load_settings  # noqa: E402

SCRIPTS: dict[str, tuple[bool, tuple[str, ...]]] = {
    "start-ask-stop": (
        False, ("Start a time-lapse.", "How long is it so far?", "Stop the time-lapse.")
    ),
    "wording-1": (False, ("Can you do a timelapse of this?",)),
    "wording-2": (False, ("Film this as a time lapse, I'll be a while.",)),
    "wording-3": (False, ("Let's get a time-lapse of the bench going.",)),
    "record-this": (False, ("Record this.",)),
    "later-ask": (True, ("Is the time-lapse still running?",)),
    "later-stop": (True, ("Stop the time-lapse.",)),
    "never-started": (False, ("Is anything recording right now?",)),
    "later-anything": (True, ("Is anything recording right now?",)),
}


class Stand:
    """Answers like the kiosk's recorder and writes down every call."""

    def __init__(self, running: bool) -> None:
        self._running, self.calls, self.frames = running, [], 1800 if running else 0

    def running(self) -> bool:
        return self._running

    def seconds(self) -> float:
        return self.frames / lapse.PLAYBACK_FPS

    def start(self) -> str:
        self.calls.append("start")
        self._running = True
        return ""

    def stop(self, why: str = lapse.ASKED) -> dict:
        self.calls.append("stop")
        self._running = False
        return {"seconds": self.seconds(), "saved": True, "stopped": why, "problem": ""}

    def take_untold(self) -> None:
        return None


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    with args.out.open("w") as out:
        for name, (going, script) in SCRIPTS.items():
            for run in range(1, args.runs + 1):
                stand = Stand(going)
                lapse.install(stand)
                try:
                    records = await _turns(script, VoiceAgent(load_settings()))
                except (TimeoutError, RuntimeError) as err:
                    records = [{"you": "", "error": str(err)}]
                for record in records:
                    record |= {"script": name, "run": run, "lapse_calls": stand.calls}
                    out.write(json.dumps(record) + "\n")
                    said = record.get("cyclops", record.get("error"))
                    print(f"{name} #{run} {record['you']!r} -> {record.get('tools')} "
                          f"{stand.calls}: {said!r}", flush=True)  # fmt: skip
    lapse.install(None)


async def _turns(script: tuple[str, ...], agent: VoiceAgent) -> list[dict]:
    """Typed turns, one after another. Nothing is said first: a session opens silently
    (arc-eng/hq#572), so this waits for ready, not for a greeting as ``talk_probe._turns`` does."""
    records: list[dict] = []
    task = asyncio.create_task(agent.run())
    try:
        await talk_probe._await_or_fail(task, agent.ready, talk_probe.READY_TIMEOUT_S)
        for line in script:
            turn = talk_probe.Turn()
            agent.on_event = turn
            await agent.send_text(line)
            await talk_probe._await_or_fail(task, turn.done, talk_probe.TURN_TIMEOUT_S)
            records.append({"you": line, "cyclops": turn.text, "tools": turn.tools,
                            "responses": turn.responses, "answer_s": turn.answer_s})  # fmt: skip
    finally:
        await agent.close()
        await asyncio.gather(task, return_exceptions=True)
    return records


if __name__ == "__main__":
    asyncio.run(main())
