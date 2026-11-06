#!/usr/bin/env python3
"""hq#602 against the real voice model, real SessionLog, real recorder, real screen capture.

Run on the Pi via hq_602_pi.sh, from a throwaway copy, with its working directory in the deployed
folder so `.env` is found.

Real: SessionController, VoiceAgent, SessionLog, SessionRecorder, ScreenSource (wf-recorder), ffmpeg,
the model. Fake: the sound card (hq_572's scripted room), so nothing is played aloud and the live
kiosk's ear can't start a real session. Not exercised: a real mic and speaker, a real voice.

Isolation (hq#573): HOME and every record folder are scratch before cyclops is imported, every
*_dir and *_file setting is checked to be inside scratch, slug and remember are off, `_after` (naming
and filing, in another process) is replaced with a counter. It stops if a real session goes live.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from dataclasses import fields, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hq_572_real_sessions as h  # noqa: E402  (sets HOME and the record folders to scratch)

os.environ["CYCLOPS_RECORD"] = "1"
os.environ["CYCLOPS_SLUG"] = "0"
os.environ["CYCLOPS_REMEMBER"] = "0"
os.environ["CYCLOPS_QUIET_END_S"] = "15"  # short, so a held and a released quiet both show quickly

import cyclops  # noqa: E402
from cyclops import agent as agent_module  # noqa: E402
from cyclops import card, library, record, session  # noqa: E402
from cyclops.config import load_settings  # noqa: E402
from cyclops.screen import ScreenSource  # noqa: E402

h.FIXTURES = Path(__file__).resolve().parent / "fixtures_602"
SCRATCH = h.SCRATCH
OUT = Path(os.environ["QA_OUT"])
PANEL = tuple(int(x) for x in os.environ["QA_PANEL"].split("x"))
CALLS: list[tuple[str, dict]] = []
AFTER: list[str] = []
RESULTS: list[tuple[str, str, str]] = []
QUIET = 15


def check(name: str, ok: bool, evidence: str = "") -> None:
    RESULTS.append((name, "PASS" if ok else "FAIL", evidence))
    print(f"   {'PASS' if ok else 'FAIL'} {name}{' · ' + evidence if evidence else ''}", flush=True)


def guard_live() -> None:
    """Stop at once if Marco starts a real session: this run isn't worth disturbing one."""
    with urllib.request.urlopen("http://localhost/api/live", timeout=3) as r:  # noqa: S310
        live = json.load(r)
    if live.get("name") is not None:
        finish()
        raise SystemExit(f"ABORT: a real session is live: {live.get('name')}")


def install() -> None:
    h.install()
    session.SessionLog._after = lambda self: AFTER.append(self.dir.name)  # noqa: SLF001
    real_output = agent_module.VoiceAgent._send_tool_output  # noqa: SLF001
    real_run = agent_module.VoiceAgent._run_recording  # noqa: SLF001
    pending: dict[str, str] = {}

    async def run(self, call):  # noqa: ANN001, ANN202
        pending[call.call_id] = call.arguments
        await real_run(self, call)

    async def output(self, call_id, value, *a, **k):  # noqa: ANN001, ANN002, ANN003, ANN202
        if call_id in pending:
            CALLS.append((pending.pop(call_id), value))
            print(f"   tool: {CALLS[-1]}", flush=True)
        return await real_output(self, call_id, value, *a, **k)

    agent_module.VoiceAgent._run_recording = run  # noqa: SLF001
    agent_module.VoiceAgent._send_tool_output = output  # noqa: SLF001


def children() -> list[str]:
    done = subprocess.run(["ps", "--ppid", str(os.getpid()), "-o", "comm="],  # noqa: S603, S607
                          capture_output=True, text=True, check=False)
    return [c for c in done.stdout.split() if c in {"ffmpeg", "wf-recorder"}]


def start(box: h.Box):  # noqa: ANN201
    guard_live()
    box.k._pending, box.k._pending_at = "start", time.monotonic()  # noqa: SLF001
    box.controller.start()
    box.wait_up()
    return box.controller._agent  # noqa: SLF001


def stop(box: h.Box) -> None:
    box.k._pending, box.k._pending_at = "stop", time.monotonic()  # noqa: SLF001
    box.controller.stop()
    box.wait_gone(150)


def say(box: h.Box, agent, name: str, timeout: float = 25.0) -> list:  # noqa: ANN001
    mark = len(agent.spoken)
    ROOM_DONE = h.ROOM.say(name)
    while time.monotonic() < ROOM_DONE:
        time.sleep(0.05)
    end = time.monotonic() + timeout
    while time.monotonic() < end and not any(w == "cyclops" for w, _ in agent.spoken[mark:]):
        time.sleep(0.05)
    h.settle(box)
    said = h.said_since(agent, mark)
    print(f"   {name}: {said}", flush=True)
    return said


def newest_folder() -> Path:
    folders = [p for p in Path(os.environ["CYCLOPS_SESSIONS_DIR"]).iterdir() if p.is_dir()]
    return max(folders, key=lambda p: p.stat().st_mtime)


def probe(path: Path) -> dict:
    done = subprocess.run(  # noqa: S603
        ["ffprobe", "-v", "error", "-show_entries",  # noqa: S607
         "format=duration:stream=codec_type,channels,width,height",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=False)
    return json.loads(done.stdout or "{}")


def channel_loud(path: Path, side: str) -> str:
    done = subprocess.run(  # noqa: S603
        ["ffmpeg", "-hide_banner", "-nostdin", "-i", str(path), "-af",  # noqa: S607
         f"pan=mono|c0={side},volumedetect", "-vn", "-f", "null", "-"],
        capture_output=True, text=True, check=False)
    return next((ln.split("max_volume:")[1].strip() for ln in done.stderr.splitlines()
                 if "max_volume" in ln), "?")


def keep(folder: Path, tag: str) -> None:
    subprocess.run(["cp", "-a", str(folder), str(OUT / tag)], check=False)  # noqa: S603, S607


def s1_default(box: h.Box) -> None:
    print("S1 a normal session: nothing asked for", flush=True)
    agent = start(box)
    time.sleep(1.5)
    during = children()
    tag = session.recording()
    said = say(box, agent, "ask")
    during += children()
    check("no recorder or capture during a normal session", not during and not tag,
          f"children={during} tag={tag}")
    check("the conversation still works", any(w == "cyclops" for w, _ in said), str(said)[:160])
    ended = h.ROOM.say("bye1")
    gone = box.wait_gone(40)
    folder = newest_folder()
    names = sorted(p.name for p in folder.iterdir())
    check("goodbye ends it", gone, f"ended_by={agent.ended_by} {time.monotonic() - ended:.1f}s")
    check("no video.mp4, no parts", card.VIDEO not in names and card.PARTS not in names, str(names))
    check("folder kept and handed to naming and filing", folder.name in AFTER, folder.name)
    entry = library._read(folder)  # noqa: SLF001
    check("Sessions list reads it as finished, no video",
          not entry.video and entry.verdict not in {"empty", "broken", "failed"},
          f"video={entry.video} verdict={entry.verdict}")
    keep(folder, "s1-no-video")


def s2_full(box: h.Box) -> None:
    print("S2 start, asked twice, quiet held, stop, stop again, one per session, quiet released",
          flush=True)
    agent = start(box)
    say(box, agent, "ask")  # so the recording starts well after the session did
    CALLS.clear()
    began = time.monotonic()
    said = say(box, agent, "record")
    started_at = time.monotonic()
    first = CALLS[0] if CALLS else None
    check("'record this' reaches the tool as start",
          bool(first and '"start"' in first[0] and first[1].get("ok")), str(first))
    check("encoder and capture really running",
          sorted(children()) == ["ffmpeg", "wf-recorder"], str(children()))
    check("panel's tag says recording", session.recording())
    check("Cyclops confirms", any(w == "cyclops" for w, _ in said), str(said)[:200])
    CALLS.clear()
    said = say(box, agent, "record2")
    again = CALLS[-1] if CALLS else None
    check("'record' again changes nothing", session.recording() and len(children()) == 2
          and bool(again is None or not again[1].get("ok")), f"{again} said={said}"[:300])
    time.sleep(QUIET + 20)  # well past the quiet limit, with nobody talking
    check("quiet doesn't end it while recording", agent.ended_by is None and box.controller.running,
          f"{QUIET + 20} s of silence against a {QUIET} s limit")
    CALLS.clear()
    said = say(box, agent, "stop", timeout=35)
    stopped = CALLS[-1] if CALLS else None
    span = time.monotonic() - started_at
    check("'stop recording' reaches the tool as stop",
          bool(stopped and '"stop"' in stopped[0] and stopped[1].get("ok")), str(stopped))
    check("Cyclops confirms the stop", any(w == "cyclops" for w, _ in said), str(said)[:200])
    check("encoder and capture gone", not children(), str(children()))
    folder = Path(session.current().dir)
    video = folder / card.VIDEO
    check("a finished video exists mid-session", card.written(video),
          f"{video.stat().st_size if video.exists() else 0} bytes")
    info = probe(video)
    seconds = float(info.get("format", {}).get("duration", 0))
    # A file far shorter than the recording would mean -shortest cut it at a track that stopped.
    check("video runs the length of the recording, silence included",
          seconds > QUIET + 20 and abs(seconds - span) < 25,
          f"file {seconds:.1f} s, about {span:.1f} s between tool answer and stop heard")
    print(f"   streams: {info.get('streams')}", flush=True)
    print(f"   left (him): {channel_loud(video, 'c0')} · right (Cyclops): "
          f"{channel_loud(video, 'c1')}", flush=True)
    CALLS.clear()
    said = say(box, agent, "stop")
    check("'stop' when it isn't changes nothing", not session.recording() and card.written(video)
          and bool(not CALLS or not CALLS[-1][1].get("ok")), f"{CALLS} said={said}"[:300])
    size = video.stat().st_size
    CALLS.clear()
    said = say(box, agent, "record")
    check("a second recording is refused plainly, first video intact",
          not session.recording() and video.stat().st_size == size
          and bool(not CALLS or not CALLS[-1][1].get("ok")), f"{CALLS} said={said}"[:300])
    released = time.monotonic()
    gone = box.wait_gone(QUIET + 25)
    check("quiet ends it again after the stop", gone and agent.ended_by == "quiet",
          f"ended_by={agent.ended_by} after {time.monotonic() - released:.1f} s")
    check("capture gone after the session", not children(), str(children()))
    keep(folder, "s2-recorded")
    print(f"   whole test took {time.monotonic() - began:.0f} s", flush=True)


def s3_never_stopped(box: h.Box) -> None:
    print("S3 'stop' before any recording, then record and just say goodbye", flush=True)
    agent = start(box)
    CALLS.clear()
    said = say(box, agent, "stop")
    check("'stop' with nothing recording gets a plain answer",
          not session.recording() and bool(not CALLS or not CALLS[-1][1].get("ok")),
          f"{CALLS} said={said}"[:300])
    CALLS.clear()
    say(box, agent, "record")
    check("recording", session.recording(), str(CALLS))
    folder = Path(session.current().dir)
    time.sleep(5)
    h.ROOM.say("bye1")
    gone = box.wait_gone(150)
    video = folder / card.VIDEO
    info = probe(video) if video.exists() else {}
    check("goodbye ends it and the video is finished", gone and card.written(video),
          f"ended_by={agent.ended_by} {info.get('format')}")
    check("no parts left behind", not (folder / card.PARTS).exists())
    check("capture gone", not children(), str(children()))
    keep(folder, "s3-goodbye")


def s4_limit(box: h.Box) -> None:
    print("S4 the time limit, set to 25 s for this run", flush=True)
    real = agent_module.RECORD_LIMIT_S
    agent_module.RECORD_LIMIT_S = 25
    try:
        agent = start(box)
        CALLS.clear()
        say(box, agent, "record")
        check("recording", session.recording(), str(CALLS))
        folder = Path(session.current().dir)
        mark = len(agent.spoken)
        time.sleep(30)
        h.settle(box)
        end = time.monotonic() + 15
        while time.monotonic() < end and not any(w == "cyclops" for w, _ in agent.spoken[mark:]):
            time.sleep(0.1)
        said = h.said_since(agent, mark)
        check("the limit stops it and saves the video",
              not session.recording() and card.written(folder / card.VIDEO), str(children()))
        check("Cyclops tells him", any(w == "cyclops" for w, _ in said), str(said)[:240])
        check("the session carries on", box.controller.running and agent.ended_by is None)
        stop(box)
    finally:
        agent_module.RECORD_LIMIT_S = real


class _Shutil:
    def __init__(self, **over) -> None:  # noqa: ANN003
        self.__dict__.update(over)

    def __getattr__(self, name: str):  # noqa: ANN204
        import shutil

        return getattr(shutil, name)


def s5_cant_record(box: h.Box, label: str, module, **over) -> None:  # noqa: ANN001, ANN003
    print(f"S5 can't record: {label}", flush=True)
    real = module.shutil
    module.shutil = _Shutil(**over)
    try:
        agent = start(box)
        CALLS.clear()
        said = say(box, agent, "record")
        folder = Path(session.current().dir)
        check(f"{label}: tool says no", bool(CALLS and not CALLS[-1][1].get("ok")), str(CALLS)[:240])
        check(f"{label}: nothing running", not session.recording() and not children(), str(children()))
        print(f"   JUDGE {label}: words he hears: {[s for w, s in said if w == 'cyclops']}", flush=True)
        said = say(box, agent, "ask")
        check(f"{label}: the conversation carries on", any(w == "cyclops" for w, _ in said))
        stop(box)
        check(f"{label}: no video", not (folder / card.VIDEO).exists())
    finally:
        module.shutil = real


def s6_not_on_its_own(box: h.Box) -> None:
    print("S6 talking about a recording must not start one", flush=True)
    for _ in range(2):
        agent = start(box)
        CALLS.clear()
        said = say(box, agent, "mention")
        check("no tool call, nothing running", not CALLS and not session.recording(),
              f"{CALLS} said={said}"[:300])
        stop(box)


def cpu_ticks() -> tuple[int, int]:
    parts = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
    values = [int(x) for x in parts]
    return sum(values), values[3] + values[4]


def temp_c() -> float:
    return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000


def measure(seconds: int) -> dict:
    total0, idle0 = cpu_ticks()
    temps = []
    for _ in range(seconds // 5):
        time.sleep(5)
        temps.append(temp_c())
    total1, idle1 = cpu_ticks()
    busy = (total1 - total0 - (idle1 - idle0)) / (total1 - total0)
    return {"cores": round(busy * (os.cpu_count() or 4), 2), "temp": (min(temps), max(temps))}


def s7_numbers(settings) -> None:  # noqa: ANN001
    print("S7 a live session with and without recording, 60 s each, device-wide", flush=True)
    box = make_box(replace(settings, quiet_end_s=0))
    for tag, recording in (("without", False), ("with", True), ("without", False), ("with", True)):
        agent = start(box)
        if recording:
            say(box, agent, "record")
            if not session.recording():
                print("   recording didn't start; skipped", flush=True)
        else:
            say(box, agent, "ask")
        time.sleep(10)
        print(f"   NUMBERS {tag}: {measure(60)}", flush=True)
        stop(box)
        time.sleep(15)
    box.alive = False


def make_box(settings) -> h.Box:  # noqa: ANN001
    box = h.Box(settings)
    box.controller._screen = ScreenSource(PANEL)  # noqa: SLF001
    return box


def r_carried_refusal(settings) -> None:  # noqa: ANN001
    """The refusal says a new session can make another. So: a new session, right after."""
    print("R4 refused, then a new session as it said, and 'record this'", flush=True)
    box = make_box(settings)
    agent = start(box)
    say(box, agent, "record")
    say(box, agent, "stop")
    say(box, agent, "record")
    stop(box)
    time.sleep(2)
    agent = start(box)
    CALLS.clear()
    said = say(box, agent, "record")
    print(f"   carried in: {len(agent.earlier)} turns", flush=True)
    check("new session after a refusal: the tool is called and it records",
          bool(CALLS) and session.recording(), f"calls={CALLS} said={said}"[:400])
    stop(box)
    box.alive = False


def r_carried_claim(settings) -> None:  # noqa: ANN001
    """A session that recorded and ended; the next one is asked whether it's recording."""
    print("R5 recorded, ended; new session: 'are you recording?'", flush=True)
    box = make_box(settings)
    agent = start(box)
    say(box, agent, "record")
    stop(box)
    time.sleep(2)
    agent = start(box)
    CALLS.clear()
    said = say(box, agent, "status")
    words = [s for w, s in said if w == "cyclops"]
    print(f"   carried in: {len(agent.earlier)} turns · recording={session.recording()}", flush=True)
    print(f"   JUDGE new session, nothing recording, calls={CALLS}, he hears: {words}", flush=True)
    stop(box)
    box.alive = False


def r_status_in_session(settings) -> None:  # noqa: ANN001
    print("R6 'are you recording?' inside one session, before, during, after", flush=True)
    box = make_box(settings)
    agent = start(box)
    for moment in ("before", "during", "after"):
        if moment == "during":
            say(box, agent, "record")
        if moment == "after":
            say(box, agent, "stop")
        CALLS.clear()
        said = say(box, agent, "status")
        words = [s for w, s in said if w == "cyclops"]
        print(f"   JUDGE {moment}: recording={session.recording()} calls={CALLS} he hears: {words}",
              flush=True)
    stop(box)
    box.alive = False


def round_two(settings) -> None:  # noqa: ANN001
    """Each scenario on its own controller, so a previous session's turns aren't carried in."""
    print("Round 2: fresh controller each time", flush=True)

    def fresh(run) -> None:  # noqa: ANN001
        guard_live()
        box = make_box(settings)
        try:
            run(box)
        except Exception as exc:  # noqa: BLE001
            check("step ran", False, f"{type(exc).__name__}: {exc}")
            if box.controller.running:
                stop(box)
        box.alive = False
        time.sleep(3)

    fresh(s3_never_stopped)
    fresh(s4_limit)
    fresh(lambda b: s5_cant_record(b, "no ffmpeg", record, which=lambda _n: None))
    fresh(lambda b: s5_cant_record(
        b, "no space", session, disk_usage=lambda _p: type("U", (), {"free": 10 * 1024**2})()))
    for run in (r_carried_refusal, r_carried_refusal, r_carried_refusal,
                r_carried_claim, r_carried_claim, r_status_in_session):
        guard_live()
        try:
            run(settings)
        except Exception as exc:  # noqa: BLE001
            check("step ran", False, f"{type(exc).__name__}: {exc}")
        time.sleep(3)


def finish() -> None:
    (OUT / "results.json").write_text(json.dumps(RESULTS, indent=1))
    fails = [r for r in RESULTS if r[1] == "FAIL"]
    print(f"RESULT {len(RESULTS) - len(fails)} passed, {len(fails)} failed", flush=True)
    for name, _, why in fails:
        print(f"  FAILED {name}: {why}", flush=True)


def main() -> None:
    if not str(Path(cyclops.__file__).resolve()).startswith(str(OUT.parent)):
        raise SystemExit(f"refusing to run: importing {cyclops.__file__}, not the copy")
    install()
    settings = load_settings()
    for f in fields(settings):
        if f.name.endswith(("_dir", "_file")):
            value = getattr(settings, f.name)
            if value and not Path(value).resolve().is_relative_to(SCRATCH.resolve()):
                raise SystemExit(f"refusing to run: {f.name}={value} is not scratch")
    for where in (h.tasks.TASKS_FILE, h.config_module.PANEL_FILE, h.config_module.VOICE_FILE):
        if not Path(where).resolve().is_relative_to(SCRATCH.resolve()):
            raise SystemExit(f"refusing to run: {where} is not scratch")
    if settings.quiet_end_s != QUIET or not settings.record or settings.slug or settings.remember:
        raise SystemExit(f"refusing to run: settings not as asked {settings.quiet_end_s}")
    print(f"scratch: {SCRATCH} · code {cyclops.__file__} · model {settings.model}", flush=True)
    if "--round2" in sys.argv:
        round_two(settings)
        finish()
        return
    if "--one" in sys.argv:
        # One scenario per process. Round 2's stale ears shared the one fake room, so they took
        # blocks of speech from each other and the model heard nonsense ("Hegyesek").
        which = sys.argv[sys.argv.index("--one") + 1]
        guard_live()
        if which == "nospace":
            box = make_box(settings)
            s5_cant_record(box, "no space", session,
                           disk_usage=lambda _p: type("U", (), {"free": 10 * 1024**2})())
            box.alive = False
        else:
            {"refusal": r_carried_refusal, "claim": r_carried_claim,
             "status": r_status_in_session}[which](settings)
        finish()
        return
    box = make_box(settings)
    steps = (
        lambda: s1_default(box),
        lambda: s2_full(box),
        lambda: s3_never_stopped(box),
        lambda: s4_limit(box),
        lambda: s5_cant_record(box, "no ffmpeg", record, which=lambda _n: None),
        lambda: s5_cant_record(
            box, "no space", session,
            disk_usage=lambda _p: type("U", (), {"free": 10 * 1024**2})()),
        lambda: s6_not_on_its_own(box),
    )
    for step in steps:
        guard_live()
        try:
            step()
        except Exception as exc:  # noqa: BLE001 - one broken step mustn't hide the rest
            check("step ran", False, f"{type(exc).__name__}: {exc}")
            if box.controller.running:
                stop(box)
        time.sleep(3)
    box.alive = False
    s7_numbers(settings)
    finish()


if __name__ == "__main__":
    main()
