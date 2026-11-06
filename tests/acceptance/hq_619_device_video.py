#!/usr/bin/env python3
"""hq#619, on the Pi: one video through the real offer and the real panel.js, on the Pi's browser
build with the PR's flags, in a second headless browser that can't make a sound.

Run with the live install's python, PYTHONPATH=/tmp/qa-619/src. Everything writable is in
/tmp/qa-619/vid. Refuses to run if a session is open. Prints hosts only: the stream URL is signed.
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
home, run_dir = S / "home", S / "run"
ADMIN, DEBUG = 18623, 18624
VIDEO = "jNQXAC9IVRw"


def live_idle() -> bool:
    return '"name": null' in urllib.request.urlopen("http://localhost/api/live", timeout=3).read().decode()


if not live_idle():
    sys.exit("ABORT: session open")

for d in (home / ".cache" / "cyclops", run_dir, S / "sessions", S / "captures", S / "projects", S / "manuals", S / "worker"):
    d.mkdir(parents=True, exist_ok=True)
run_dir.chmod(0o700)
env = {
    **os.environ,
    "HOME": str(home),
    "XDG_CONFIG_HOME": str(home / ".config"),
    "XDG_CACHE_HOME": str(home / ".cache"),
    # The sound server lives in the real runtime dir. Point away from it and nothing can reach a speaker.
    "XDG_RUNTIME_DIR": str(run_dir),
    "OPENAI_API_KEY": "scratch-not-a-key",
    "CYCLOPS_SESSIONS_DIR": str(S / "sessions"),
    "CYCLOPS_CAPTURES_DIR": str(S / "captures"),
    "CYCLOPS_PROJECTS_DIR": str(S / "projects"),
    "CYCLOPS_MANUALS_DIR": str(S / "manuals"),
    "CYCLOPS_WORKER_DIR": str(S / "worker"),
    "CYCLOPS_ABOUT_FILE": str(S / "about.md"),
}
env.pop("WAYLAND_DISPLAY", None)
env.pop("DISPLAY", None)
PY = sys.executable

code = (
    "import sys, json; from pathlib import Path\n"
    "from cyclops.config import PANEL_FILE, load_settings as l\n"
    "s = l(require_api_key=False)\n"
    "for p in (PANEL_FILE, s.sessions_dir, s.projects_dir, s.manuals_dir, s.worker_dir, s.about_file):\n"
    "    assert str(Path(p).resolve()).startswith('/tmp/qa-619/vid'), p\n"
    "from cyclops import kiosk, youtube\n"
    "v = youtube.details(sys.argv[1])\n"
    "print(json.dumps({'stream': v.stream if v else '', 'flags': kiosk.CHROME_FLAGS}))\n"
)
got = json.loads(subprocess.run([PY, "-c", code, VIDEO], env=env, cwd=S, check=True, capture_output=True, text=True).stdout)
stream, flags = got["stream"], got["flags"]
if not stream:
    sys.exit("NOT RUN: no stream")
print("scratch paths checked; stream host:", urlparse(stream).hostname, flush=True)
flags = [f for f in flags if f not in ("--kiosk", "--default-background-color=FF000000")]
assert any(f.startswith("--host-resolver-rules") for f in flags), "not the PR's flags"

admin = subprocess.Popen([PY, "-m", "cyclops.admin.server", "--host=127.0.0.1", f"--port={ADMIN}"],
                         env=env, cwd=S, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
browser = None


async def probe(ws, expr: str, n: list[int]):
    n[0] += 1
    await ws.send(json.dumps({"id": n[0], "method": "Runtime.evaluate",
                              "params": {"expression": expr, "returnByValue": True}}))
    while True:
        m = json.loads(await ws.recv())
        if m.get("id") == n[0]:
            return m["result"]["result"].get("value")


STATE = """(() => { const v = document.querySelector('video.clip');
 return JSON.stringify({kiosk: document.body.classList.contains('kiosk'), clipping: document.body.classList.contains('clipping'),
 video: v ? {ready: v.readyState, t: Math.round(v.currentTime), dur: Math.round(v.duration||0), paused: v.paused,
 w: v.videoWidth, err: v.error && v.error.code} : null}); })()"""


async def drive() -> None:
    import websockets

    tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{DEBUG}/json", timeout=3).read())
    page = next(t for t in tabs if t["type"] == "page")
    n = [0]
    async with websockets.connect(page["webSocketDebuggerUrl"], max_size=None) as ws:
        print("before the offer:", await probe(ws, STATE, n), flush=True)
        offer = "import sys; from cyclops import panel; print(panel.offer_video(sys.argv[1], 'qa 619', 2, b'', 60.0))"
        print("offer left:", subprocess.run([PY, "-c", offer, stream], env=env, cwd=S, check=True,
                                            capture_output=True, text=True).stdout.strip(), flush=True)
        for wait in (5, 5, 5, 5):
            await asyncio.sleep(wait)
            print("playing?", await probe(ws, STATE, n), "live idle:", live_idle(), flush=True)
        real = {**os.environ, "XDG_RUNTIME_DIR": f"/run/user/{os.getuid()}"}
        inputs = subprocess.run(["pactl", "list", "short", "sink-inputs"], env=real, capture_output=True, text=True).stdout
        print("streams on the real sound server while playing:", inputs.strip() or "none", flush=True)
        painted = (home / ".cache" / "cyclops").glob("*paint*")
        print("painted flag in scratch:", [p.name for p in painted], flush=True)


try:
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{ADMIN}/api/panel", timeout=1).read()
            break
        except OSError:
            time.sleep(0.5)
    else:
        sys.exit("NOT RUN: scratch admin never answered")
    browser = subprocess.Popen(
        ["chromium", "--headless=new", "--ozone-platform=headless", "--mute-audio",
         f"--remote-debugging-port={DEBUG}", f"--user-data-dir={S / 'profile'}", *flags,
         f"http://127.0.0.1:{ADMIN}/#/sessions"],
        env=env, cwd=S, stdout=(S / "browser.log").open("wb"), stderr=subprocess.STDOUT)
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
    asyncio.run(drive())
finally:
    if browser:
        browser.terminate()
        try:
            browser.wait(timeout=10)
        except subprocess.TimeoutExpired:
            browser.kill()
    admin.terminate()
    admin.wait(timeout=10)
print("DONE", flush=True)
