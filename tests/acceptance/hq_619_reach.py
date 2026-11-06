#!/usr/bin/env python3
"""hq#619 round 2, Mac only: the real flags against several videos, a list of off-box targets, and
what connections the browser keeps after a video is taken back.

Usage: uv run --with playwright python hq_619_round2.py <scratch dir> <youtube id>...
Scratch admin, HOME and data dirs in <scratch dir>. Muted. No session, no mic. Prints hosts only.
"""

from __future__ import annotations

import http.server
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

from cyclops.kiosk import CHROME_FLAGS

scratch, ids = Path(sys.argv[1]).resolve(), sys.argv[2:]
PORT, OTHER = 18620, 18621
base = f"http://127.0.0.1:{PORT}"
WALL = [f for f in CHROME_FLAGS if f not in ("--kiosk", "--default-background-color=FF000000")]
WALL += ["--mute-audio"]
print("flags under test:", [f for f in WALL if "proxy" in f or "resolver" in f or "background-net" in f])

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


class Quiet(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        self.send_response(200)
        self.end_headers()

    def log_message(self, *a):  # noqa: ANN002
        pass


threading.Thread(target=http.server.HTTPServer(("127.0.0.1", OTHER), Quiet).serve_forever, daemon=True).start()


def our_chrome_sockets() -> list[str]:
    """Connected non-loopback sockets of this run's Chrome and its children, never Marco's own."""
    mains = subprocess.run(["pgrep", "-f", "playwright_chromiumdev_profile"], capture_output=True, text=True).stdout.split()
    pids = set(mains)
    for m in mains:
        pids.update(subprocess.run(["pgrep", "-P", m], capture_output=True, text=True).stdout.split())
    if not pids:
        return ["(no chrome found: reading proves nothing)"]
    out = subprocess.run(["lsof", "-nP", "-a", "-i", "-p", ",".join(sorted(pids))], capture_output=True, text=True).stdout
    found = []
    for line in out.splitlines()[1:]:
        m = re.search(r"(TCP|UDP) (\S+->\S+)(?: \((\w+)\))?", line)
        if m and not re.search(r"->(127\.|\[::1\])", m.group(2)):
            found.append(f"{m.group(1)} {m.group(2).split('->')[1]} {m.group(3) or ''}".strip())
    return sorted(set(found))


TARGETS = [
    ("loopback, other port", f"http://127.0.0.1:{OTHER}/"),
    ("localhost, other port", f"http://localhost:{OTHER}/"),
    ("video host", "https://redirector.googlevideo.com/"),
    ("example.com", "https://example.com/"),
    ("www.google.com", "https://www.google.com/"),
    ("youtube.com", "https://www.youtube.com/"),
    ("i.ytimg.com", "https://i.ytimg.com/"),
    ("mtalk push", "https://mtalk.google.com:5228/"),
    ("8.8.8.8 by address", "https://8.8.8.8/"),
    ("dns.google", "https://dns.google/"),
    ("googlevideo lookalike", "https://googlevideo.com.example.com/"),
]

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
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True, args=WALL)
        page = browser.new_page(viewport={"width": 800, "height": 480})
        failed: list[str] = []
        page.on("requestfailed", lambda r: failed.append(f"{urlparse(r.url).hostname} {r.failure}"))
        page.goto(base + "/#/sessions", wait_until="load")
        page.wait_for_timeout(1000)
        assert page.evaluate("document.body.classList.contains('kiosk')"), "not the kiosk's page"

        print("--- reach, from the kiosk page")
        for label, url in TARGETS:
            got = page.evaluate("""u => { const c = new AbortController(); setTimeout(() => c.abort(), 8000);
                return fetch(u, {mode: 'no-cors', cache: 'no-store', signal: c.signal})
                .then(() => 'REACHED').catch(e => 'refused'); }""", url)
            print(f"  {label}: {got}")
        failed.clear()

        for ident in ids:
            stream = run_py("import sys; from cyclops import youtube; v = youtube.details(sys.argv[1]);"
                            "print(v.stream if v else '')", ident)
            if not stream:
                print(f"--- {ident}: no stream, not run")
                continue
            print(f"--- {ident}: stream host {urlparse(stream).hostname}")
            run_py("import sys; from cyclops import panel; panel.offer_video(sys.argv[1], 'qa', 2, b'', 60.0)", stream)
            page.wait_for_timeout(12000)
            state = page.evaluate("""() => { const v = document.querySelector('video.clip');
                return v ? {ready: v.readyState, t: Math.round(v.currentTime), paused: v.paused,
                            err: v.error && v.error.code} : null; }""")
            print(f"  video: {json.dumps(state)}")
            print(f"  failures: {sorted(set(failed)) or 'none'}")
            failed.clear()
            print(f"  sockets while playing: {our_chrome_sockets()}")
            run_py("from cyclops import panel; panel.withdraw()")
            page.wait_for_timeout(2000)
            print(f"  sockets 2 s after it was taken back: {our_chrome_sockets()}")

        for wait, total in ((30, 30), (90, 120), (180, 300)):
            page.wait_for_timeout(wait * 1000)
            print(f"sockets {total} s after the last video: {our_chrome_sockets()}")
        browser.close()
finally:
    admin.terminate()
    admin.wait(timeout=10)
