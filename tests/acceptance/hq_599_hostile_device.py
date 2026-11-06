#!/usr/bin/env python3
"""hq#599 round 2, on the Pi: a hostile job inside the REAL cyclops-worker service.

Run by hq_599_r2_pi.sh, which has put hq_599_standin.sh where the service expects `claude`. Requests
go through the real jobs.submit into the real inbox, written by `cyclops`, so the ownership rule is
tested for real, which a laptop on one account can't do.

Isolation (hq#573): HOME, manuals and the job and tasks ledgers are scratch before cyclops loads and
are checked here. The voice tool, kiosk and index service are not involved. No model call.

Usage: hq_599_hostile_device.py <case> ...   (cases: see CASES at the bottom)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from dataclasses import replace
from pathlib import Path

SCRATCH = Path(tempfile.mkdtemp(prefix="qa-599-r2-"))
os.environ["HOME"] = str(SCRATCH)
os.environ["CYCLOPS_MANUALS_DIR"] = str(SCRATCH / "manuals")
os.environ["OPENAI_API_KEY"] = "sk-qa-not-real"
(SCRATCH / ".cache" / "cyclops").mkdir(parents=True)
(SCRATCH / "manuals").mkdir()
os.chdir(SCRATCH)

from cyclops import jobs, tasks, worker  # noqa: E402
from cyclops.config import load_settings  # noqa: E402

assert str(jobs.LEDGER).startswith(str(SCRATCH)), jobs.LEDGER
assert str(tasks.TASKS_FILE).startswith(str(SCRATCH)), tasks.TASKS_FILE

ROOT = Path("/var/lib/cyclops-worker")
QA = ROOT / "qa"  # not home/: since d55cf60 a restart empties that
SETTINGS = replace(
    load_settings(require_api_key=False),
    worker=True,
    worker_dir=ROOT,
    manuals_dir=SCRATCH / "manuals",
)
assert str(SETTINGS.manuals_dir).startswith(str(SCRATCH))
RESULTS: list[bool] = []


def check(name: str, ok: bool, evidence: object = "") -> None:
    RESULTS.append(bool(ok))
    print(
        f"   {'PASS' if ok else 'FAIL'} {name}{' · ' + str(evidence)[:300] if evidence != '' else ''}",
        flush=True,
    )


def note(text: object) -> None:
    print(f"   · {text}", flush=True)


def guard() -> None:
    with urllib.request.urlopen("http://localhost/api/live", timeout=3) as r:  # noqa: S310
        name = json.load(r).get("name")
    if name is not None:
        print(f"ABORT: a real session is live: {name}", flush=True)
        os._exit(3)


def sh(*argv: str) -> str:
    return subprocess.run(argv, capture_output=True, text=True, check=False).stdout.strip()


def as_worker(*argv: str) -> None:
    # cwd: this process sits in a scratch folder that account can't enter, and `find` gives up there.
    subprocess.run(
        ["sudo", "-n", "-u", "cyclops-worker", *argv], check=False, capture_output=True, cwd="/"
    )


def still_up(seconds: float = 15) -> bool:
    """Active and staying that way. NRestarts alone misleads: a manual restart sets it back to 0."""
    time.sleep(seconds)
    first = sh("systemctl", "show", "cyclops-worker", "-p", "MainPID", "--value")
    time.sleep(7)  # past RestartSec, so a crash loop can't look steady
    return (
        sh("systemctl", "is-active", "cyclops-worker") == "active"
        and first not in ("", "0")
        and first == sh("systemctl", "show", "cyclops-worker", "-p", "MainPID", "--value")
    )


def restarts() -> int:
    return int(sh("systemctl", "show", "cyclops-worker", "-p", "NRestarts", "--value") or 0)


def worker_processes() -> list[str]:
    return [
        line
        for line in sh("ps", "-u", "cyclops-worker", "-o", "args=").splitlines()
        if line.strip()
    ]


def runs() -> list[str]:
    try:
        return (QA / "runs.log").read_text().split()
    except OSError:
        return []


def ask(words: str) -> str:
    guard()
    jobs.collect(SETTINGS)  # settle whatever came before, as the index service would
    answer = jobs.submit(SETTINGS, words)
    assert answer == jobs.STARTED, (words, answer, jobs._read()[-2:])  # noqa: SLF001
    # By request and state, not rows[-1]: past KEEP rows the ledger puts old untold ones last, and
    # this harness never tells any. That had it waiting on the wrong id (2026-11-05, round 3).
    rows = [r for r in jobs._read() if r.get("request") == words and r.get("state") == jobs.RUNNING]  # noqa: SLF001
    return str(rows[-1]["id"])


def wait_file(path: Path, seconds: float) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if os.path.lexists(path):
            return True
        time.sleep(0.2)
    return False


def answer_of(job_id: str) -> dict | None:
    return jobs._answer(SETTINGS, job_id)  # noqa: SLF001


def log_of(job_id: str) -> str:
    try:
        return (QA / f"{job_id}.log").read_text()
    except OSError:
        return ""


def ticking(marker: Path) -> bool:
    """Is a detached process still writing its marker?"""
    try:
        first = marker.read_text()
    except OSError:
        return False
    time.sleep(2.5)
    return marker.read_text() != first


def forge() -> None:
    before = len(runs())
    job = ask("QA-FORGE")
    check("the job gets an answer", wait_file(ROOT / "done" / f"{job}.json", 30))
    time.sleep(8)
    check(
        "forged request, link, FIFO and hard link: none ran",
        len(runs()) == before + 1,
        f"{len(runs()) - before} run(s)",
    )
    left = sorted(os.listdir(ROOT / "inbox"))
    check("the inbox is empty again", not left, left)
    alive = sh("systemctl", "is-active", "cyclops-worker")
    check("the service is still up", alive == "active", alive)
    control = ask("QA-COUNT control")
    check(
        "control: a request Cyclops wrote still runs",
        wait_file(ROOT / "done" / f"{control}.json", 30),
    )


def hardlink() -> None:
    job = ask("QA-HARDLINK")
    wait_file(ROOT / "done" / f"{job}.json", 30)
    time.sleep(6)
    note(log_of(job).replace("\n", " | "))
    check("no file of another account linked into the inbox", "allowed" not in log_of(job))


def outlive() -> None:
    job = ask("QA-OUTLIVE")
    check("the job gets an answer", wait_file(ROOT / "done" / f"{job}.json", 40))
    time.sleep(1)
    check("the detached process is gone", not ticking(QA / f"alive-{job}"))
    left = worker_processes()
    check(
        "only the worker itself is left on the account",
        len(left) == 1 and "worker.py" in left[0],
        left,
    )
    print(log_of(job).split("--- ways to outlive the job")[-1], flush=True)
    time.sleep(65)
    check("nothing started by cron", not (QA / "from-cron").exists())
    check("nothing started by systemd-run", not (QA / "from-systemd-run").exists())
    disable_linger()  # it succeeded on 2026-11-05: see the `linger` case, and don't leave it on


def disable_linger() -> None:
    subprocess.run(["sudo", "-n", "loginctl", "disable-linger", "cyclops-worker"], check=False)


def linger() -> None:
    """A job turns on lingering and leaves a user unit. Does it run after the job, outside the wall?"""
    tick = QA / "unit-tick"
    facts = QA / "unit-facts"
    job = ask("QA-LINGER")
    try:
        check("the job gets an answer", wait_file(ROOT / "done" / f"{job}.json", 40))
        note(log_of(job).replace("\n", " | "))
        check("lingering stays off", not Path("/var/lib/systemd/linger/cyclops-worker").exists())
        started = wait_file(tick, 30)
        time.sleep(1)
        running = started and ticking(tick)
        note(f"the unit started by itself after the job: {running}")
        if not running and Path("/var/lib/systemd/linger/cyclops-worker").exists():
            # The job's end can catch the manager. The linger file stays, and at boot this is what
            # systemd does with it. A reboot is what's being stood in for here, not done.
            subprocess.run(["sudo", "-n", "systemctl", "start", "user@999.service"], check=False)
            started = wait_file(tick, 30)
            time.sleep(1)
            running = started and ticking(tick)
            note(f"after starting the user manager as a boot would: {running}")
        check("nothing the job left is running after its answer, now or after a boot", not running)
        if facts.exists():
            print(facts.read_text(), flush=True)
        time.sleep(2)
        worker_alive = sh("systemctl", "is-active", "cyclops-worker")
        note(f"worker {worker_alive}; on the account now: {[p[:40] for p in worker_processes()]}")
    finally:
        disable_linger()
    time.sleep(3)
    check(
        "my cleanup: gone once lingering is turned off",
        not ticking(tick),
        [p[:40] for p in worker_processes()],
    )


def limit() -> None:
    job = ask("QA-LIMIT")
    started = time.monotonic()
    answered = wait_file(ROOT / "done" / f"{job}.json", worker.JOB_LIMIT_S + 90)
    took = time.monotonic() - started
    answer = answer_of(job) or {}
    check(
        "answered timed_out at the 15-minute limit",
        answered and answer.get("reason") == worker.TIMED_OUT,
        f"{took:.0f} s, {answer}",
    )
    time.sleep(1)
    check("the detached process is gone", not ticking(QA / f"alive-{job}"))
    left = worker_processes()
    check(
        "only the worker itself is left on the account",
        len(left) == 1 and "worker.py" in left[0],
        left,
    )


def kill_watcher() -> None:
    before = restarts()
    job = ask("QA-KILL-WATCHER")
    check("answered after the watcher was killed", wait_file(ROOT / "done" / f"{job}.json", 40))
    answer = answer_of(job) or {}
    check("answered as cut off", answer.get("reason") == worker.INTERRUPTED, answer)
    check("systemd restarted it", restarts() > before, f"{before} -> {restarts()}")
    time.sleep(1)
    check("the detached process is gone", not ticking(QA / f"alive-{job}"))


def bad_answer(word: str) -> None:
    job = ask(word)
    assert wait_file(QA / f"planted-{job}", 30), "stand-in didn't plant"
    done = threading.Event()
    started = time.monotonic()
    threading.Thread(target=lambda: (jobs.collect(SETTINGS), done.set()), daemon=True).start()
    returned = done.wait(5)
    check(f"{word}: collect returned within 5 s", returned, f"{time.monotonic() - started:.2f} s")
    row = next((r for r in jobs._read() if r.get("id") == job), {})  # noqa: SLF001
    check(
        f"{word}: settled as failed",
        row.get("state") == jobs.FAILED and row.get("reason") == worker.BROKEN,
        row.get("reason"),
    )
    if returned:
        second = threading.Event()
        threading.Thread(
            target=lambda: (jobs.submit(SETTINGS, "QA-COUNT next"), second.set()), daemon=True
        ).start()
        check(f"{word}: the lock is free for a new voice request", second.wait(5))
    wait_file(ROOT / "done" / f"{job}.json", 5)
    time.sleep(25)  # the planted job and the one after it finish


def answers() -> None:
    his = Path("/tmp/qa-599-his-file.json")
    his.write_text(json.dumps({"state": "failed", "reason": "planted-through-a-link"}))
    his.chmod(0o644)
    for word in ("QA-ANSWER-FIFO", "QA-ANSWER-HUGE", "QA-ANSWER-LINK"):
        bad_answer(word)
    his.unlink()


def restart_service() -> None:
    subprocess.run(["sudo", "-n", "systemctl", "restart", "cyclops-worker"], check=False)
    time.sleep(3)


def output_fifo() -> None:
    job = ask("QA-OUTPUT-FIFO")
    answered = wait_file(ROOT / "done" / f"{job}.json", 25)
    note(f"a FIFO in place of its own output.json: answered within 25 s: {answered}")
    check("the worker isn't stuck on a FIFO left in its job folder", answered)
    if not answered:
        restart_service()
        note(f"after a service restart: {answer_of(job)}")


def done_is_a_file() -> None:
    job = ask("QA-DONE-IS-A-FILE")
    time.sleep(12)  # the crash, RestartSec, and the repair
    up = still_up(5)
    check(
        "the worker comes back by itself after `done` is replaced by a file",
        up,
        f"{restarts()} restarts",
    )
    check(
        "the job is answered as cut off",
        (answer_of(job) or {}).get("reason") == worker.INTERRUPTED,
        answer_of(job),
    )
    info = (ROOT / "done").stat()
    check(
        "`done` is a folder again, group cyclops, setgid",
        oct(info.st_mode)[-4:] == "2750",
        oct(info.st_mode),
    )
    as_worker("rm", "-rf", str(ROOT / "done.qa"))
    jobs.collect(SETTINGS)
    control = ask("QA-COUNT control")
    check(
        "control: runs again with nobody clearing it",
        wait_file(ROOT / "done" / f"{control}.json", 30),
    )


def long_name() -> None:
    job = ask("QA-LONG-NAME")
    wait_file(ROOT / "done" / f"{job}.json", 30)
    time.sleep(3)
    restart_service()  # recover() only looks at this on the next start, say after a power cut
    up = still_up()
    note(f"service {sh('systemctl', 'is-active', 'cyclops-worker')}, restarts since: {restarts()}")
    check("the worker starts with a job folder whose name is too long to answer for", up)
    check("the planted folder is gone", not list((ROOT / "jobs").glob("xxxxxxxxxx*")))
    jobs.collect(SETTINGS)
    control = ask("QA-COUNT control")
    check(
        "control: runs again with nobody clearing it",
        wait_file(ROOT / "done" / f"{control}.json", 30),
    )


def stop() -> None:
    """kill -STOP on the program that enforces the limit, and no time limit is counting."""
    job = ask("QA-STOP")
    started = time.monotonic()
    answered = wait_file(ROOT / "done" / f"{job}.json", 100)
    answer = answer_of(job) or {}
    check(
        "a frozen worker is restarted, and the job answered as cut off",
        answered and answer.get("reason") == worker.INTERRUPTED,
        f"{time.monotonic() - started:.0f} s, {answer}",
    )
    time.sleep(3)
    left = worker_processes()
    check("the job is gone", len(left) == 1 and "worker.py" in left[0], left)


def land_traps() -> None:
    for word in ("QA-LAND-PIPE", "QA-DONE-FOLDER"):
        job = ask(word)
        check(f"{word}: answered", wait_file(ROOT / "done" / f"{job}.json", 25), answer_of(job))
        time.sleep(3)
        up = sh("systemctl", "is-active", "cyclops-worker")
        check(f"{word}: the service is up", up == "active", up)


def sockets() -> None:
    """The bus was one local socket a job could reach. What else can it connect to?"""
    job = ask("QA-SOCKETS")
    wait_file(ROOT / "done" / f"{job}.json", 40)
    print(log_of(job), flush=True)
    check(
        "the system bus isn't among them",
        "dbus/system_bus_socket" not in log_of(job).split("--- reached")[-1],
    )


def proc_mem() -> None:
    job = ask("QA-PROC-MEM")
    wait_file(ROOT / "done" / f"{job}.json", 40)
    print(log_of(job), flush=True)
    check(
        "a job can't open the worker's memory to write",
        "mem open for writing: allowed" not in log_of(job),
    )
    check("a job can't ptrace the worker", "ptrace seize: allowed" not in log_of(job))


def real_job() -> None:
    """The real program. Run with the marker's hook planted by the driver: does it fire in a new job?"""
    job = ask("Find the Raspberry Pi 5 product brief PDF and download it.")
    answered = wait_file(ROOT / "done" / f"{job}.json", 300)
    answer = answer_of(job) or {}
    check(
        "a real job on this code gets an answer", answered and answer.get("state") == "ok", answer
    )
    check(
        "a hook left in the old home or the top folder doesn't run in this one",
        not (QA / "from-hook").exists(),
    )
    print(f"   said: {jobs.collect(SETTINGS)}", flush=True)
    landed = sorted(p.name for p in SETTINGS.manuals_dir.iterdir())
    check("it landed, in scratch manuals only", bool(landed), landed)


CASES = {
    "forge": forge,
    "hardlink": hardlink,
    "outlive": outlive,
    "linger": linger,
    "answers": answers,
    "kill_watcher": kill_watcher,
    "output_fifo": output_fifo,
    "done_is_a_file": done_is_a_file,
    "long_name": long_name,
    "limit": limit,
    "stop": stop,
    "land_traps": land_traps,
    "sockets": sockets,
    "proc_mem": proc_mem,
    "real_job": real_job,
}

for name in sys.argv[1:]:
    print(f"=== {name}", flush=True)
    try:
        CASES[name]()
    except Exception as exc:  # noqa: BLE001
        check(f"{name} ran to the end", False, f"{type(exc).__name__}: {exc}")
print(f"SUMMARY {sum(RESULTS)} of {len(RESULTS)} as hoped", flush=True)
os._exit(0)
