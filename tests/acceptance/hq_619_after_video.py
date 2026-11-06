#!/usr/bin/env python3
"""hq#619, on the Pi: after a video is taken back, does the browser keep talking to the video host?

Second headless browser, PR's flags, real offer and real panel.js, scratch admin. Two cases, a fresh
browser each: a long video taken back partway, and a short one played to its end and then taken back.
Run with the live install's python, PYTHONPATH=/tmp/qa-619/src. Writes only under /tmp/qa-619/vid.
Stops if a session opens. Prints hosts only: the stream URL is signed.
"""
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

S = Path("/tmp/qa-619/vid")
PHASE = Path("/tmp/qa-619/phase")
home, run_dir = S / "home", S / "run"
ADMIN, DEBUG = 18623, 18624
CASES = (("long", "aqz-KE-bpKQ", 15), ("short", "jNQXAC9IVRw", 28))
MARKS = (5, 60, 120, 180, 300)


def live_idle() -> bool:
    return '"name": null' in urllib.request.urlopen("http://localhost/api/live", timeout=3).read().decode()


if not live_idle():
    sys.exit("ABORT: session open")

for d in (home / ".cache" / "cyclops", run_dir, S / "sessions", S / "captures", S / "projects", S / "manuals", S / "worker"):
    d.mkdir(parents=True, exist_ok=True)
run_dir.chmod(0o700)
env = {
    **os.environ,
    "HOME": str(home), "XDG_CONFIG_HOME": str(home / ".config"), "XDG_CACHE_HOME": str(home / ".cache"),
    "XDG_RUNTIME_DIR": str(run_dir),  # away from the sound server, so nothing can reach a speaker
    "OPENAI_API_KEY": "scratch-not-a-key",
    "CYCLOPS_SESSIONS_DIR": str(S / "sessions"), "CYCLOPS_CAPTURES_DIR": str(S / "captures"),
    "CYCLOPS_PROJECTS_DIR": str(S / "projects"), "CYCLOPS_MANUALS_DIR": str(S / "manuals"),
    "CYCLOPS_WORKER_DIR": str(S / "worker"), "CYCLOPS_ABOUT_FILE": str(S / "about.md"),
}
env.pop("WAYLAND_DISPLAY", None)
env.pop("DISPLAY", None)
PY = sys.executable


def py(code: str, *args: str) -> str:
    return subprocess.run([PY, "-c", code, *args], env=env, cwd=S, check=True, capture_output=True, text=True).stdout.strip()


CHECK = (
    "from pathlib import Path\nfrom cyclops.config import PANEL_FILE, load_settings as l\n"
    "s = l(require_api_key=False)\n"
    "for p in (PANEL_FILE, s.sessions_dir, s.projects_dir, s.manuals_dir, s.worker_dir, s.about_file):\n"
    "    assert str(Path(p).resolve()).startswith('/tmp/qa-619/vid'), p\n"
    "print('scratch paths checked')"
)
print(py(CHECK), flush=True)
flags = json.loads(py("import json; from cyclops.kiosk import CHROME_FLAGS as f; print(json.dumps(f))"))
flags = [f for f in flags if f not in ("--kiosk", "--default-background-color=FF000000")]
assert any(f.startswith("--host-resolver-rules") for f in flags), "not the PR's flags"
STATE = """(() => { const v = document.querySelector('video.clip');
 return JSON.stringify(v ? {ready: v.readyState, t: Math.round(v.currentTime), dur: Math.round(v.duration||0), paused: v.paused, ended: v.ended, err: v.error && v.error.code} : null); })()"""


def tree(root: int) -> set[str]:
    out = subprocess.run(["ps", "-eo", "pid,ppid"], capture_output=True, text=True).stdout.split("\n")[1:]
    pairs = [line.split() for line in out if line.strip()]
    found, grew = {str(root)}, True
    while grew:
        grew = False
        for pid, ppid in pairs:
            if ppid in found and pid not in found:
                found.add(pid)
                grew = True
    return found


def sockets(root: int) -> list[str]:
    pids = tree(root)
    out = subprocess.run(["ss", "-tunap"], capture_output=True, text=True).stdout
    found = set()
    for line in out.splitlines():
        if not any(f"pid={p}," in line for p in pids):
            continue
        parts = line.split()
        peer = parts[5]
        if not peer.startswith(("127.", "[::1]", "[::ffff:127", "0.0.0.0", "*", "[::]")):
            found.add(f"{parts[0]} {parts[1]} {peer}")
    return sorted(found)


async def probe(ws, n: list[int]):
    n[0] += 1
    await ws.send(json.dumps({"id": n[0], "method": "Runtime.evaluate", "params": {"expression": STATE, "returnByValue": True}}))
    while True:
        m = json.loads(await ws.recv())
        if m.get("id") == n[0]:
            return m["result"]["result"].get("value")


async def case(name: str, ident: str, play_s: int, root: int) -> None:
    import websockets

    stream = py("import sys; from cyclops import youtube; v = youtube.details(sys.argv[1]); print(v.stream if v else '')", ident)
    if not stream:
        print(f"{name}: NOT RUN, no stream", flush=True)
        return
    print(f"{name}: stream host {urlparse(stream).hostname}", flush=True)
    tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{DEBUG}/json", timeout=3).read())
    page = next(t for t in tabs if t["type"] == "page")
    n = [0]
    async with websockets.connect(page["webSocketDebuggerUrl"], max_size=None) as ws:
        print(f"{name}: before: {sockets(root)}", flush=True)
        PHASE.write_text(f"{name}-play\n")
        py("import sys; from cyclops import panel; panel.offer_video(sys.argv[1], 'qa', 2, b'', 60.0)", stream)
        await asyncio.sleep(play_s)
        print(f"{name}: video {await probe(ws, n)}", flush=True)
        print(f"{name}: sockets at the end of play: {sockets(root)}", flush=True)
        py("from cyclops import panel; panel.withdraw()")
        start, last = time.time(), 0
        PHASE.write_text(f"{name}-0-60s\n")
        for mark in MARKS:
            await asyncio.sleep(max(0, start + mark - time.time()))
            print(f"{name}: +{mark}s {sockets(root)} session idle: {live_idle()}", flush=True)
            if mark >= 60:
                PHASE.write_text(f"{name}-after-{mark}s\n")
            last = mark
        PHASE.write_text("gap\n")


def one(name: str, ident: str, play_s: int) -> None:
    if not live_idle():
        sys.exit("ABORT: session open")
    shutil.rmtree(S / "profile", ignore_errors=True)
    browser = subprocess.Popen(
        ["chromium", "--headless=new", "--ozone-platform=headless", "--mute-audio", f"--remote-debugging-port={DEBUG}",
         f"--user-data-dir={S / 'profile'}", *flags, f"http://127.0.0.1:{ADMIN}/#/sessions"],
        env=env, cwd=S, stdout=(S / f"browser-{name}.log").open("wb"), stderr=subprocess.STDOUT)
    try:
        for _ in range(40):
            time.sleep(1)
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{DEBUG}/json", timeout=2).read()
                break
            except OSError:
                continue
        else:
            sys.exit("NOT RUN: browser never answered")
        time.sleep(5)
        asyncio.run(case(name, ident, play_s, browser.pid))
    finally:
        browser.terminate()
        try:
            browser.wait(timeout=10)
        except subprocess.TimeoutExpired:
            browser.kill()


admin = subprocess.Popen([PY, "-m", "cyclops.admin.server", "--host=127.0.0.1", f"--port={ADMIN}"],
                         env=env, cwd=S, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{ADMIN}/api/panel", timeout=1).read()
            break
        except OSError:
            time.sleep(0.5)
    else:
        sys.exit("NOT RUN: scratch admin never answered")
    for name, ident, play_s in CASES:
        one(name, ident, play_s)
finally:
    admin.terminate()
    admin.wait(timeout=10)
print("DONE", flush=True)
