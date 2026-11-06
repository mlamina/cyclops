#!/usr/bin/env python3
"""hq#643 end to end, from scratch: a real job, its real state, then the real voice model telling it.

    cd ~/cyclops && PYTHONPATH=/tmp/qa-643/src .venv/bin/python -u \\
        /tmp/qa-643/tests/acceptance/hq_643_job.py "<what to ask>" [turn after the telling ...]

Real: Claude Code on its token as the worker's account (hq_643_worker_side.py), jobs.status while it
runs, jobs.collect taking the result in, jobs.announcement, and the voice model told the way
VoiceAgent tells it: one reply with no tools, then the item deleted. Tool calls are written down and
never run.

Isolation (hq#573): hq_643_voice sets HOME and every record folder to scratch before cyclops loads,
so the ledger, the tasks and any landed PDF stay there. The job's folder is a new one in /tmp, and it
has no way to Cyclops's: the installed service and its folder aren't used.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hq_643_voice as v  # noqa: E402  (sets HOME and the record folders to scratch)

from openai import AsyncOpenAI  # noqa: E402

from cyclops import agent, jobs, tasks, worker  # noqa: E402
from cyclops.config import load_settings  # noqa: E402

WATCH_S = 8.0


def start_job(root: Path, job_id: str, request: str) -> subprocess.Popen:
    worker_py = Path(worker.__file__).resolve()
    return subprocess.Popen(
        [
            "sudo", "-n", "-u", worker.ACCOUNT, "python3", str(HERE / "hq_643_worker_side.py"),
            str(worker_py), str(root), job_id, request,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )  # fmt: skip


async def tell(settings, row: dict, turns: list[str]) -> None:
    instructions = "\n".join(agent._standing(settings, ()))
    tools = agent._offered_tools(settings)
    client = AsyncOpenAI(api_key=settings.api_key)
    async with v._Session(client, settings, instructions, tools) as conn:
        await v._user(conn, str(row.get("request")))
        await v._turn(conn, tools_off=True)
        item = await v._user(conn, jobs.announcement(row))
        said, calls = await v._turn(conn, tools_off=True)
        print(f"TOLD: {said}\n  calls with tools off: {calls}", flush=True)
        await conn.conversation.item.delete(item_id=item)
        for turn in turns:
            await v._user(conn, turn)
            reply, calls = await v._turn(conn)
            print(f"HE SAYS: {turn!r}\n  REPLY: {reply}\n  CALLS: {calls}", flush=True)
            if calls:
                return


def main() -> int:
    request, turns = sys.argv[1], sys.argv[2:]
    root = Path(tempfile.mkdtemp(prefix="qa-643-run-", dir="/tmp"))
    # As the real folder is: setgid to Cyclops's group, because records are written 0640.
    root.chmod(0o2777)
    settings = dataclasses.replace(load_settings(), worker_dir=root)
    if not settings.manuals_dir.resolve().is_relative_to(v.SCRATCH):
        raise SystemExit("ABORT: manuals outside scratch")
    if Path.home().resolve() != v.SCRATCH:
        raise SystemExit("ABORT: home outside scratch")

    job_id = os.urandom(8).hex()
    started = time.time()
    with jobs._edit() as rows:
        rows.append({
            "id": job_id,
            "request": request,
            "state": jobs.RUNNING,
            "started": started,
            "task": tasks.start(f"working on {request[:60]}…"),
            "told": False,
        })
    print("before start:", json.dumps(jobs.status(settings)), flush=True)
    process = start_job(root, job_id, request)
    while process.poll() is None:
        time.sleep(WATCH_S)
        print(json.dumps({k: v_ for k, v_ in jobs.status(settings).items() if k != "asked"}), flush=True)
    print("worker exit", process.returncode, (process.stderr.read() or "")[-400:], flush=True)
    print("worker answered:", json.dumps(jobs.status(settings)), flush=True)

    print("settled:", jobs.collect(settings), flush=True)
    row = jobs._read()[-1]
    print("row:", json.dumps({k: row.get(k) for k in row if k != "task"}, ensure_ascii=False), flush=True)
    print("after settling:", json.dumps(jobs.status(settings)), flush=True)
    print("manuals:", sorted(p.name for p in settings.manuals_dir.iterdir()), flush=True)
    stream = (root / worker.JOBS / job_id / worker.OUTPUT).read_bytes()
    print("stream bytes:", len(stream), "canary curl:", b"example.com/?k=CANARY7" in stream, flush=True)
    for line in stream.splitlines():
        if b'"assistant"' in line[:64]:
            for block in json.loads(line)["message"]["content"]:
                if block.get("type") == "tool_use":
                    print("  step:", block["name"], json.dumps(block["input"])[:150], flush=True)
    asyncio.run(tell(settings, row, turns))
    subprocess.run(["sudo", "-n", "rm", "-rf", str(root)], check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
