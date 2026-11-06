#!/usr/bin/env python3
"""hq#619, Mac only: after one video is taken back, what does the browser keep open, and does it
still move bytes? Reads sockets (lsof) and per-process bytes (nettop) of this run's Chrome only.

Usage: uv run --with playwright python hq_619_linger.py <scratch dir> <youtube id>
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

from cyclops.kiosk import CHROME_FLAGS

scratch, ident = Path(sys.argv[1]).resolve(), sys.argv[2]
PORT = 18622
base = f"http://127.0.0.1:{PORT}"
WALL = [f for f in CHROME_FLAGS if f not in ("--kiosk", "--default-background-color=FF000000")] + ["--mute-audio"]
home = scratch / "home"
(home / ".cache" / "cyclops").mkdir(parents=True, exist_ok=True)
env = {**os.environ, "HOME": str(home), "OPENAI_API_KEY": "scratch-not-a-key"}
for name, var in (("sessions", "SESSIONS_DIR"), ("captures", "CAPTURES_DIR"), ("projects", "PROJECTS_DIR"),
                  ("manuals", "MANUALS_DIR"), ("worker", "WORKER_DIR")):
    (scratch / name).mkdir(exist_ok=True)
    env[f"CYCLOPS_{var}"] = str(scratch / name)
env["CYCLOPS_ABOUT_FILE"] = str(scratch / "about.md")
PY = sys.executable


def run_py(code: str, *args: str) -> str:
    return subprocess.run([PY, "-c", code, *args], env=env, cwd=scratch, check=True,
                          capture_output=True, text=True).stdout.strip()


def pids() -> list[str]:
    mains = subprocess.run(["pgrep", "-f", "playwright_chromiumdev_profile"], capture_output=True, text=True).stdout.split()
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


def off_box_bytes() -> tuple[int, int]:
    """Cumulative bytes in and out on this run's non-loopback flows, from nettop."""
    total_in = total_out = 0
    for pid in pids():
        out = subprocess.run(["nettop", "-n", "-L", "1", "-J", "bytes_in,bytes_out", "-p", pid],
                             capture_output=True, text=True).stdout
        for line in out.splitlines():
            parts = line.split(",")
            if "<->" in line and not re.search(r"<->(127\.|::1)", line):
                total_in += int(parts[-3] or 0) if parts[-3].isdigit() else 0
                total_out += int(parts[-2] or 0) if parts[-2].isdigit() else 0
    return total_in, total_out


admin = subprocess.Popen([PY, "-m", "cyclops.admin.server", "--host=127.0.0.1", f"--port={PORT}"],
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
    stream = run_py("import sys; from cyclops import youtube; v = youtube.details(sys.argv[1]);"
                    "print(v.stream if v else '')", ident)
    if not stream:
        sys.exit("NOT RUN: no stream")
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True, args=WALL)
        page = browser.new_page(viewport={"width": 800, "height": 480})
        page.goto(base + "/#/sessions", wait_until="load")
        page.wait_for_timeout(1000)
        print(f"before any video: {sockets()} bytes {off_box_bytes()}", flush=True)
        run_py("import sys; from cyclops import panel; panel.offer_video(sys.argv[1], 'qa', 2, b'', 60.0)", stream)
        page.wait_for_timeout(15000)
        print(f"playing: {sockets()} bytes {off_box_bytes()}", flush=True)
        run_py("from cyclops import panel; panel.withdraw()")
        start = time.time()
        for mark in (5, 30, 60, 120, 180, 300, 450, 600, 750, 900):
            page.wait_for_timeout(max(0, int((start + mark - time.time()) * 1000)))
            print(f"+{mark}s: {sockets()} bytes {off_box_bytes()}", flush=True)
        browser.close()
finally:
    admin.terminate()
    admin.wait(timeout=10)
