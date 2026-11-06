#!/usr/bin/env python3
"""hq#619: does a video asked for out loud still play on the panel behind the wall? (qa's check)

Usage (from a Cyclops checkout, Mac only, never on the Pi):
    uv run --with playwright python hq_619_video_offer.py <scratch dir> <youtube id>

A scratch cyclops-admin with HOME and every data dir in <scratch dir>. The real
youtube.details() gets the stream, the real panel.offer_video() leaves the offer, the real
panel.js picks it up, as the session does at agent.py `_play_video`. Runs today's flags, then the
real ones. Uses installed Chrome because Playwright's Chromium has no H.264. Muted. No session, no
mic. Prints hosts only: the stream URL is signed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

from cyclops.kiosk import CHROME_FLAGS

scratch, video_id = Path(sys.argv[1]).resolve(), sys.argv[2]
PORT = 18619
base = f"http://127.0.0.1:{PORT}"
NEW = (
    "--disable-background-networking",
    "--proxy-server=127.0.0.1:9",
    "--proxy-bypass-list=*.googlevideo.com",
    "--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE *.googlevideo.com, EXCLUDE localhost, EXCLUDE 127.0.0.1",
)
assert all(flag in CHROME_FLAGS for flag in NEW), "not the real flags"
# Not about the network, and refused or meaningless headless.
usable = [f for f in CHROME_FLAGS if f not in ("--kiosk", "--default-background-color=FF000000")]
TODAY = [f for f in usable if f not in NEW] + ["--mute-audio"]
WALL = usable + ["--mute-audio"]

home = scratch / "home"
(home / ".cache" / "cyclops").mkdir(parents=True, exist_ok=True)
env = {
    **os.environ,
    "HOME": str(home),
    "OPENAI_API_KEY": "scratch-not-a-key",
    "CYCLOPS_SESSIONS_DIR": str(scratch / "sessions"),
    "CYCLOPS_CAPTURES_DIR": str(scratch / "captures"),
    "CYCLOPS_PROJECTS_DIR": str(scratch / "projects"),
    "CYCLOPS_MANUALS_DIR": str(scratch / "manuals"),
    "CYCLOPS_ABOUT_FILE": str(scratch / "about.md"),
    "CYCLOPS_WORKER_DIR": str(scratch / "worker"),
}
for name in ("sessions", "captures", "projects", "manuals", "worker"):
    (scratch / name).mkdir(exist_ok=True)
PY = sys.executable


def run_py(code: str, *args: str) -> str:
    return subprocess.run(
        [PY, "-c", code, *args], env=env, cwd=scratch, check=True, capture_output=True, text=True
    ).stdout.strip()


stream = run_py(
    "import sys; from cyclops import youtube; v = youtube.details(sys.argv[1]);"
    "print(v.stream if v else '')",
    video_id,
)
if not stream:
    sys.exit("NOT RUN: youtube.details gave no stream, so this proves nothing")
stream_host = urlparse(stream).hostname
print(f"stream host from youtube.details: {stream_host}")

OFFER = (
    "import sys; from cyclops import panel;"
    "print(panel.offer_video(sys.argv[1], 'qa 619', 5, b'', 60.0))"
)

admin = subprocess.Popen(
    [PY, "-m", "cyclops.admin.server", "--host=127.0.0.1", f"--port={PORT}"],
    env=env, cwd=scratch, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)


def visit(p, name: str, args: list[str]) -> None:  # noqa: ANN001
    browser = p.chromium.launch(channel="chrome", headless=True, args=args)
    page = browser.new_page(viewport={"width": 800, "height": 480})
    got: list[str] = []
    failed: list[str] = []

    def host(url: str) -> str:
        return urlparse(url).hostname or url[:30]

    page.on("response", lambda r: got.append(f"{r.status} {host(r.url)}")
            if host(r.url) != "127.0.0.1" else None)
    page.on("requestfailed", lambda r: failed.append(f"{host(r.url)} {r.failure}")
            if host(r.url) != "127.0.0.1" else None)
    page.goto(base + "/#/sessions", wait_until="load")
    page.wait_for_timeout(1000)
    assert page.evaluate("document.body.classList.contains('kiosk')"), "not the kiosk's page"
    print(f"{name}: offer left: {run_py(OFFER, stream)}")
    page.wait_for_timeout(15000)
    state = page.evaluate(
        """() => { const v = document.querySelector('video.clip');
        return v ? {found: true, readyState: v.readyState, networkState: v.networkState,
                    currentTime: Math.round(v.currentTime), duration: Math.round(v.duration || 0),
                    paused: v.paused, error: v.error && v.error.code,
                    message: v.error && v.error.message} : {found: false}; }"""
    )
    print(f"{name}: video element: {json.dumps(state)}")
    print(f"{name}: off-box responses: {sorted(set(got)) or 'none'}")
    print(f"{name}: off-box failures: {sorted(set(failed)) or 'none'}")
    page.screenshot(path=str(scratch / f"{name}.png"))
    browser.close()
    run_py("from cyclops import panel; panel.withdraw()")


try:
    for _ in range(60):
        try:
            urllib.request.urlopen(base + "/api/panel", timeout=1).read()
            break
        except OSError:
            time.sleep(0.5)
    else:
        sys.exit("NOT RUN: scratch admin never answered")
    with sync_playwright() as p:
        visit(p, "today", TODAY)
        visit(p, "wall", WALL)
finally:
    admin.terminate()
    admin.wait(timeout=10)
