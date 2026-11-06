#!/usr/bin/env python3
"""hq#599 against the real voice model and the real worker service, one scenario per process.

Run on the Pi via hq_599_pi.sh, from a throwaway copy. Usage: hq_599_real_sessions.py open|ended|fail

Real: SessionController, VoiceAgent, the installed cyclops-worker service, Claude Code on its token,
jobs.py and the voice tool. Fake: the sound card (nothing is played aloud), and a thread here stands
in for the index service's collect, so results land in scratch manuals.

Isolation (hq#573): HOME and every record folder, the job ledger and the tasks ledger are scratch
before cyclops is imported, and checked again here. Naming and filing are replaced with a counter;
slug and remember are off. CYCLOPS_WORKER is on in this process only.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
import urllib.request
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hq_572_real_sessions as h  # noqa: E402  (sets HOME and the record folders to scratch)

os.environ.update(CYCLOPS_WORKER="1", CYCLOPS_SLUG="0", CYCLOPS_REMEMBER="0", CYCLOPS_QUIET_END_S="900")

from cyclops import agent as agent_module  # noqa: E402
from cyclops import indexer, jobs, session  # noqa: E402
from cyclops.config import load_settings  # noqa: E402

h.FIXTURES = Path(__file__).resolve().parent / "fixtures_599"
# The room's echo at 0.5 sent the first run into a loop of the model answering its own voice, so
# there was never a pause to test. Off here; the real room's echo is for Marco's own check.
h.ECHO_GAIN = 0.0
SCRATCH = h.SCRATCH.resolve()
CALLS: list[tuple[str, dict]] = []
AFTER: list[str] = []
RESULTS: list[tuple[str, bool, str]] = []
STOP_COLLECT = threading.Event()


def check(name: str, ok: bool, evidence: str = "") -> None:
    RESULTS.append((name, bool(ok), evidence))
    print(f"   {'PASS' if ok else 'FAIL'} {name}{' · ' + evidence[:380] if evidence else ''}", flush=True)


def guard_live() -> None:
    with urllib.request.urlopen("http://localhost/api/live", timeout=3) as r:  # noqa: S310
        live = json.load(r)
    if live.get("name") is not None:
        raise SystemExit(f"ABORT: a real session is live: {live.get('name')}")


def install() -> None:
    h.install()
    session.SessionLog._after = lambda self: AFTER.append(self.dir.name)  # noqa: SLF001
    real_output = agent_module.VoiceAgent._send_tool_output  # noqa: SLF001
    real_run = agent_module.VoiceAgent._run_background_job  # noqa: SLF001
    pending: dict[str, str] = {}

    async def run(self, call):  # noqa: ANN001, ANN202
        pending[call.call_id] = call.arguments
        await real_run(self, call)

    async def output(self, call_id, value, *a, **k):  # noqa: ANN001, ANN002, ANN003, ANN202
        if call_id in pending:
            CALLS.append((pending.pop(call_id), value))
            print(f"   tool: {CALLS[-1]}", flush=True)
        return await real_output(self, call_id, value, *a, **k)

    agent_module.VoiceAgent._run_background_job = run  # noqa: SLF001
    agent_module.VoiceAgent._send_tool_output = output  # noqa: SLF001


def collecting(settings) -> None:  # noqa: ANN001
    while not STOP_COLLECT.is_set():
        for said in jobs.collect(settings):
            print(f"   collect: {said}", flush=True)
        STOP_COLLECT.wait(2)


def start(box: h.Box):  # noqa: ANN201
    guard_live()
    box.k._pending, box.k._pending_at = "start", time.monotonic()  # noqa: SLF001
    box.controller.start()
    box.wait_up()
    return box.controller._agent  # noqa: SLF001


def stop(box: h.Box) -> None:
    box.k._pending, box.k._pending_at = "stop", time.monotonic()  # noqa: SLF001
    box.controller.stop()
    box.wait_gone(60)


def say(box: h.Box, agent, name: str, timeout: float = 30.0) -> tuple[list, float]:  # noqa: ANN001
    mark = len(agent.spoken)
    done = h.ROOM.say(name)
    while time.monotonic() < done:
        time.sleep(0.05)
    end = time.monotonic() + timeout
    while time.monotonic() < end and not any(w == "cyclops" for w, _ in agent.spoken[mark:]):
        time.sleep(0.05)
    waited = time.monotonic() - done
    h.settle(box)
    said = h.said_since(agent, mark)
    print(f"   {name} ({waited:.1f} s to its first words): {said}", flush=True)
    return said, waited


def settled(timeout: float = 420.0) -> dict | None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        rows = jobs._read()  # noqa: SLF001
        if rows and rows[-1].get("state") != jobs.RUNNING:
            return rows[-1]
        time.sleep(1)
    return None


def told_by(row_id: str, timeout: float = 45.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if any(r.get("id") == row_id and r.get("told") for r in jobs._read()):  # noqa: SLF001
            return True
        time.sleep(0.5)
    return False


def manual_folders(settings) -> list[str]:  # noqa: ANN001
    return sorted(p.name for p in settings.manuals_dir.iterdir() if p.is_dir())


def after_told(box: h.Box, agent, row: dict, mark: int) -> list[str]:  # noqa: ANN001
    ok = told_by(str(row.get("id")))
    h.settle(box)
    words = [s for w, s in agent.spoken[mark:] if w == "cyclops"]
    check("told, and marked told", ok, str(words))
    count = len(agent.spoken)
    time.sleep(15)
    check("told once: nothing more in the next 15 s", len(agent.spoken) == count, str(agent.spoken[count:]))
    print(f"   row: { {k: row.get(k) for k in ('state', 'name', 'pages', 'reason', 'seconds', 'cost_usd')} }",
          flush=True)
    return words


def s_open(settings, box: h.Box) -> None:  # noqa: ANN001
    print("OPEN: the session stays open; second ask; ordinary question; told; answers from the PDF", flush=True)
    agent = start(box)
    said, waited = say(box, agent, "job")
    check("the tool is called and starts", bool(CALLS and CALLS[0][1].get("started")), str(CALLS))
    check("a spoken 'on it'", any(w == "cyclops" for w, _ in said), f"{waited:.1f} s · {said}")
    CALLS.clear()
    said, _ = say(box, agent, "job2")
    check("a second ask gets 'still on the other one'",
          bool(CALLS and CALLS[0][1].get("ok") is False), f"{CALLS} {said}")
    said, waited = say(box, agent, "ask", timeout=40)
    check("an ordinary question still answered during the job",
          any(w == "cyclops" for w, _ in said), f"{waited:.1f} s · {said}")
    mark = len(agent.spoken)
    row = settled()
    check("the job settles", row is not None, str(row))
    if row is None:
        stop(box)
        return
    check("it landed", row.get("state") == jobs.LANDED, str(row.get("reason")))
    after_told(box, agent, row, mark)
    folders = manual_folders(settings)
    check("the PDF is in (scratch) manuals", bool(folders), str(folders))
    print("   reading it the way the index service does", flush=True)
    asyncio.run(indexer.reconcile(settings))
    said, waited = say(box, agent, "question", timeout=60)
    time.sleep(12)
    print(f"   JUDGE answer from the manual: {agent.spoken[-4:]}", flush=True)
    stop(box)


def s_ended(settings, box: h.Box) -> None:  # noqa: ANN001
    print("ENDED: session ends before the job finishes; told once in the next one", flush=True)
    agent = start(box)
    say(box, agent, "job")
    check("the tool started it", bool(CALLS and CALLS[0][1].get("started")), str(CALLS))
    stop(box)
    row = settled()
    check("the job completes with no session open", row is not None and row.get("state") == jobs.LANDED, str(row))
    if row is None:
        return
    agent = start(box)
    time.sleep(10)
    check("the next session opens silent", not agent.spoken, str(agent.spoken))
    check("not told before he speaks", not any(r.get("told") for r in jobs._read()))  # noqa: SLF001
    mark = len(agent.spoken)
    say(box, agent, "ask", timeout=40)
    after_told(box, agent, row, mark)
    stop(box)


def s_fail(settings, box: h.Box) -> None:  # noqa: ANN001
    print("FAIL: nothing to find", flush=True)
    agent = start(box)
    say(box, agent, "nothing")
    mark = len(agent.spoken)
    row = settled()
    check("the job settles", row is not None, str(row))
    if row is None:
        stop(box)
        return
    words = after_told(box, agent, row, mark)
    print(f"   JUDGE state={row.get('state')} reason={row.get('reason')} he hears: {words}", flush=True)
    if row.get("state") == jobs.FAILED:
        check("nothing added to manuals", not manual_folders(settings), str(manual_folders(settings)))
    stop(box)


def main() -> None:
    install()
    settings = load_settings()
    for f in fields(settings):
        if f.name.endswith(("_dir", "_file")) and f.name != "worker_dir":
            value = getattr(settings, f.name)
            if value and not Path(value).resolve().is_relative_to(SCRATCH):
                raise SystemExit(f"refusing to run: {f.name}={value} is not scratch")
    for where in (jobs.LEDGER, h.tasks.TASKS_FILE, h.config_module.RECALL_FILE):
        if not Path(where).resolve().is_relative_to(SCRATCH):
            raise SystemExit(f"refusing to run: {where} is not scratch")
    if not settings.worker or settings.slug or settings.remember:
        raise SystemExit("refusing to run: settings not as asked")
    print(f"scratch: {SCRATCH} · worker folder {settings.worker_dir} · code {agent_module.__file__}", flush=True)
    threading.Thread(target=collecting, args=(settings,), daemon=True).start()
    box = h.Box(settings)
    {"open": s_open, "ended": s_ended, "fail": s_fail}[sys.argv[1]](settings, box)
    STOP_COLLECT.set()
    box.alive = False
    failed = [r for r in RESULTS if not r[1]]
    print(f"RESULT {len(RESULTS) - len(failed)} passed, {len(failed)} failed", flush=True)
    for name, _, why in failed:
        print(f"  FAILED {name}: {why[:300]}", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
