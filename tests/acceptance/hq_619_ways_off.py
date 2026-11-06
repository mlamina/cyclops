#!/usr/bin/env python3
"""hq#619, Mac only, never on the Pi: every way a video leaves the panel, and what it leaves behind.

Usage (from a Cyclops checkout, PYTHONPATH naming the code under test):
    STREAM=<signed url> uv run --with playwright python hq_619_ways_off.py <case> <label> <scratch dir> <port>

STREAM comes from youtube.details once, so a batch of side-by-side runs asks YouTube one time. Own
scratch data and Chrome profile per run, muted, with the real offers and the real panel.js. It prints
hosts only, because the stream URL is signed. The frame log is what "nothing changes on screen" is
read from: it looks at the stage on every painted frame, so an emptied video that got painted, or a
longer gap before the next one, shows up here.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

from cyclops.kiosk import CHROME_FLAGS

CASES = ("withdraw", "press", "picture", "scratchpad", "next", "badurl", "quick")
MARKS = (5, 30, 60, 90, 120)
case, label, scratch, port = sys.argv[1], sys.argv[2], Path(sys.argv[3]).resolve(), int(sys.argv[4])
assert case in CASES, case
stream = os.environ["STREAM"]
scratch.mkdir(parents=True, exist_ok=True)
base = f"http://127.0.0.1:{port}"
profile = scratch / "chrome-profile"
FLAGS = [f for f in CHROME_FLAGS if f not in ("--kiosk", "--default-background-color=FF000000")] + ["--mute-audio"]
home = scratch / "home"
(home / ".cache" / "cyclops").mkdir(parents=True, exist_ok=True)
env = {**os.environ, "HOME": str(home), "OPENAI_API_KEY": "scratch-not-a-key"}
for folder, var in (("sessions", "SESSIONS_DIR"), ("captures", "CAPTURES_DIR"), ("projects", "PROJECTS_DIR"),
                    ("manuals", "MANUALS_DIR"), ("worker", "WORKER_DIR")):
    (scratch / folder).mkdir(exist_ok=True)
    env[f"CYCLOPS_{var}"] = str(scratch / folder)
env["CYCLOPS_ABOUT_FILE"] = str(scratch / "about.md")
PY = sys.executable

FRAME_LOG = """(() => {
  const ids = new WeakMap(); let next = 1; window.__frames = [];
  const tick = () => {
    const s = document.getElementById('stage');
    const v = s && s.querySelector('video');
    if (v) {
      if (!ids.has(v)) ids.set(v, next++);
      window.__frames.push([performance.now(), ids.get(v), v.readyState, v.currentTime > 0, !!v.getAttribute('src')]);
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
})();"""
STATE = """() => { const v = document.querySelector('video.clip');
 return v ? {ready: v.readyState, t: Math.round(v.currentTime), paused: v.paused, err: v.error && v.error.code} : null; }"""
READ_FRAMES = """() => {
  const f = window.__frames, ids = [...new Set(f.map(x => x[1]))];
  const out = {frames: f.length, videos: ids.length, paintedWithoutSource: f.filter(x => !x[4]).length, gapsMs: []};
  for (let i = 1; i < ids.length; i++) {
    const last = f.filter(x => x[1] === ids[i - 1]).pop();
    const first = f.find(x => x[1] === ids[i] && x[3]);
    out.gapsMs.push(first ? Math.round(first[0] - last[0]) : null);
  }
  return out;
}"""


def say(text: str) -> None:
    print(f"{label} {case}: {text}", flush=True)


def run_py(code: str, *args: str) -> str:
    return subprocess.run([PY, "-c", code, *args], env=env, cwd=scratch, check=True,
                          capture_output=True, text=True).stdout.strip()


def video(url: str = stream) -> None:
    run_py("import sys; from cyclops import panel; assert panel.offer_video(sys.argv[1], 'swe', 2, b'', 60.0)", url)


def withdraw() -> None:
    run_py("from cyclops import panel; panel.withdraw()")


def pids() -> list[str]:
    mains = subprocess.run(["pgrep", "-f", f"user-data-dir={profile}"], capture_output=True, text=True).stdout.split()
    found = set(mains)
    for m in mains:
        found.update(subprocess.run(["pgrep", "-P", m], capture_output=True, text=True).stdout.split())
    return sorted(found)


def sockets() -> list[str]:
    out = subprocess.run(["lsof", "-nP", "-a", "-i", "-p", ",".join(pids())], capture_output=True, text=True).stdout
    found = []
    for line in out.splitlines()[1:]:
        m = re.search(r"(TCP|UDP) (\S+->\S+)(?: \((\w+)\))?", line)
        if m and not re.search(r"->(127\.|\[::1\])", m.group(2)):
            found.append(f"{m.group(1)} {m.group(2).split('->')[1]} {m.group(3) or ''}".strip())
    return sorted(set(found))


def watch_after(page) -> None:
    start = time.time()
    for mark in MARKS:
        page.wait_for_timeout(max(0, int((start + mark - time.time()) * 1000)))
        say(f"+{mark}s: {sockets() or 'nothing off-box'}")


def picture_shown(page) -> str:
    return page.evaluate("() => [...document.getElementById('stage').children].map(e => e.tagName).join(',')")


def shot(page, name: str) -> None:
    path = scratch / f"{name}.png"
    page.screenshot(path=str(path))
    say(f"screenshot {name}: {hashlib.sha256(path.read_bytes()).hexdigest()[:12]}")


TINY_JPEG = (
    "import cv2, numpy as np; from cyclops import panel;"
    "ok, b = cv2.imencode('.jpg', np.full((480, 800, 3), 90, np.uint8)); assert panel.offer_image(b.tobytes(), 'swe')"
)
admin = subprocess.Popen([PY, "-m", "cyclops.admin.server", "--host=127.0.0.1", f"--port={port}"],
                         env=env, cwd=scratch, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(60):
        try:
            urllib.request.urlopen(base + "/api/panel", timeout=1).read()
            break
        except OSError:
            time.sleep(0.5)
    else:
        sys.exit("NOT RUN: scratch admin never answered")
    served = urllib.request.urlopen(base + "/static/panel.js", timeout=3).read().decode()
    say(f"served panel.js lets go of the source: {'removeAttribute' in served}")
    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(str(profile), channel="chrome", headless=True, args=FLAGS,
                                                       viewport={"width": 800, "height": 480})
        context.add_init_script(FRAME_LOG)
        page = context.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(base + "/#/sessions", wait_until="load")
        page.wait_for_timeout(1500)
        assert page.evaluate("document.body.classList.contains('kiosk')"), "not the kiosk's page"
        shot(page, "before")

        if case == "badurl":
            # What he sees when it breaks: a stream that fails, taken off, and then one that works.
            video("https://rr1---bad.googlevideo.com/videoplayback?nope=1")
            page.wait_for_timeout(8000)
            say(f"bad stream: {page.evaluate(STATE)}")
            withdraw()
            page.wait_for_timeout(2000)
        if case == "quick":
            # Two things at once: taken off while it's still starting, and offered again at once.
            video()
            page.wait_for_timeout(700)
            withdraw()
            video()
            page.wait_for_timeout(1500)
            withdraw()
            page.wait_for_timeout(2000)

        video()
        page.wait_for_timeout(15000)
        say(f"playing: {page.evaluate(STATE)} {sockets()}")

        if case == "press":
            # The press pauses and asks to leave. In this scratch setup no kiosk answers, so the
            # withdraw it would send back is made by hand.
            page.mouse.click(400, 200)
            page.wait_for_timeout(500)
            say(f"after the press: {page.evaluate(STATE)}")
        if case in ("withdraw", "press", "badurl", "quick"):
            withdraw()
            page.wait_for_timeout(1500)
            say(f"stage: [{picture_shown(page)}]")
            shot(page, "after")
            watch_after(page)
        if case == "picture":
            run_py(TINY_JPEG)
            page.wait_for_timeout(1500)
            say(f"stage: [{picture_shown(page)}]")
            shot(page, "over")
            watch_after(page)
            withdraw()
        if case == "scratchpad":
            run_py("from cyclops import panel; assert panel.offer_scratchpad('<p>swe</p>')")
            page.wait_for_timeout(1500)
            say(f"stage: [{picture_shown(page)}]")
            shot(page, "over")
            watch_after(page)
            withdraw()
        if case == "next":
            video()
            page.wait_for_timeout(10000)
            say(f"next one: {page.evaluate(STATE)}")
            withdraw()
            page.wait_for_timeout(1500)
            say(f"stage: [{picture_shown(page)}]")
            shot(page, "after")
            watch_after(page)

        page.wait_for_timeout(1500)
        say(f"stage at the end: [{picture_shown(page)}]")
        say(f"frames: {page.evaluate(READ_FRAMES)}")
        say(f"page errors: {errors or 'none'}")
        context.close()
finally:
    admin.terminate()
    admin.wait(timeout=10)
say("DONE")
