#!/usr/bin/env python3
"""hq#619 round 3, Mac only, never on the Pi: does a press on a companion's screen while a video is
still starting leave that screen's poll alive?

Usage (PYTHONPATH naming the code under test):
    STREAM=<signed url> uv run --with playwright python hq_619_companion_press.py <label> <scratch dir> <port> <lan ip>

A companion is any browser not on loopback (views._is_local), so the scratch admin binds the Mac's LAN
address and the page is opened there. The press lands the moment the video is on the stage, while
clip() is still waiting. Then the offer is withdrawn by hand (no kiosk here) and a picture is offered:
a live poll paints it, a dead poll doesn't. Scratch data only, muted, no session. Hosts only printed.
"""
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

label, scratch, port, ip = sys.argv[1], Path(sys.argv[2]).resolve(), int(sys.argv[3]), sys.argv[4]
stream = os.environ["STREAM"]
base = f"http://{ip}:{port}"
home = scratch / "home"
(home / ".cache" / "cyclops").mkdir(parents=True, exist_ok=True)
env = {**os.environ, "HOME": str(home), "OPENAI_API_KEY": "scratch-not-a-key"}
for folder, var in (("sessions", "SESSIONS_DIR"), ("captures", "CAPTURES_DIR"), ("projects", "PROJECTS_DIR"),
                    ("manuals", "MANUALS_DIR"), ("worker", "WORKER_DIR")):
    (scratch / folder).mkdir(exist_ok=True)
    env[f"CYCLOPS_{var}"] = str(scratch / folder)
env["CYCLOPS_ABOUT_FILE"] = str(scratch / "about.md")
PY = sys.executable


def run_py(code: str, *args: str) -> str:
    return subprocess.run([PY, "-c", code, *args], env=env, cwd=scratch, check=True, capture_output=True, text=True).stdout.strip()


OFFER = "import sys; from cyclops import panel; print(panel.offer_video(sys.argv[1], 'qa', 5, b'', 60.0))"
PICTURE = (
    "import cv2, numpy as np; from cyclops import panel;"
    "ok, b = cv2.imencode('.jpg', np.full((480, 800, 3), 90, np.uint8)); assert panel.offer_image(b.tobytes(), 'qa')"
)
WITHDRAW = "from cyclops import panel; panel.withdraw()"
assert run_py("from cyclops.config import PANEL_FILE; print(PANEL_FILE)").startswith(str(scratch)), "not scratch"

admin = subprocess.Popen([PY, "-m", "cyclops.admin.server", f"--host={ip}", f"--port={port}"],
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
    print(f"{label}: served panel.js lets go of the source: {'removeAttribute' in served}")
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True, args=["--mute-audio"])
        page = browser.new_page(viewport={"width": 1000, "height": 700})
        polls: list[float] = []
        page.on("request", lambda r: polls.append(time.time()) if r.url.endswith("/api/panel") else None)
        page.goto(base + "/#/sessions", wait_until="load")
        page.wait_for_timeout(1500)
        kiosk = page.evaluate("document.body.classList.contains('kiosk')")
        print(f"{label}: kiosk page: {kiosk}")
        if kiosk != (os.environ.get("WANT_KIOSK") == "1"):
            sys.exit("NOT RUN: not the kind of page asked for")

        # The video's requests are held, so the press lands at a known point and readyState proves it.
        held: list = []
        holding = [True]

        def hold(route):
            if holding[0]:
                held.append(route)
            else:
                route.continue_()

        page.route("**/videoplayback**", hold)

        # Control first: a press after the video has really started. Then the two waits in clip():
        # the ready wait (up to CLIP_READY_MS = 4 s), and play() still pending after it.
        for name, wait_ms, hold_it in (("press once playing", 6000, False),
                                       ("press in the ready wait", 500, True),
                                       ("press with play() pending", 5000, True)):
            holding[0] = hold_it
            run_py(OFFER, stream)
            page.wait_for_selector("video.clip", timeout=8000)
            page.wait_for_timeout(wait_ms)
            ready = page.evaluate("document.querySelector('video.clip').readyState")
            page.mouse.click(500, 300)
            page.wait_for_timeout(300)
            on_stage = page.evaluate("!!document.querySelector('video.clip')")
            print(f"{label}: {name}: readyState at the press {ready}, held requests {len(held)}, video on stage after it {on_stage}")
            holding[0] = False
            for route in held:
                try:
                    route.continue_()
                except Exception as e:
                    print(f"{label}: {name}: held request already gone: {type(e).__name__}")
            held.clear()
            run_py(WITHDRAW)
            page.wait_for_timeout(6000)
            polls.clear()
            page.wait_for_timeout(5000)
            print(f"{label}: {name}: polls in the next 5 s: {len(polls)}")
            run_py(PICTURE)
            page.wait_for_timeout(3000)
            shown = page.evaluate("!!document.querySelector('#stage img.shot')")
            print(f"{label}: {name}: a new picture reaches this screen: {shown}")
            run_py(WITHDRAW)
            page.wait_for_timeout(2000)
        browser.close()
finally:
    admin.terminate()
    admin.wait(timeout=10)
print(f"{label}: DONE")
