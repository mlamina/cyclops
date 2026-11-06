#!/usr/bin/env python3
"""hq#602: the admin page's Sessions and Highlights against real session folders, on a laptop.

Usage: hq_602_screens.py <folders dir> <shots dir> <port>. Expects `cyclops-admin` already serving
those folders (CYCLOPS_SESSIONS_DIR) on the port, and playwright available (`uv run --with playwright`).
Prints what a person would hit: rows, the player, errors, and where a tapped transcript line lands.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

folders, shots, port = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
base = f"http://127.0.0.1:{port}"
shots.mkdir(parents=True, exist_ok=True)


def api(path: str):  # noqa: ANN201
    with urllib.request.urlopen(base + path, timeout=30) as r:  # noqa: S310
        return json.load(r)


rows = api("/api/sessions")
rows = rows["sessions"] if isinstance(rows, dict) else rows
print("api sessions:", [(r["name"], r["video"], r["verdict"]) for r in rows])
print("api highlights:", json.dumps(api("/api/highlights"))[:400])

with sync_playwright() as p:
    browser = p.webkit.launch()  # Playwright Chromium has no H.264
    page = browser.new_page(viewport={"width": 1100, "height": 760})
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    page.on("console", lambda m: errors.append(f"console: {m.text}") if m.type == "error" else None)
    page.on("response", lambda r: errors.append(f"{r.status} {r.url}") if r.status >= 400 else None)

    page.goto(f"{base}/#/sessions", timeout=20000)
    page.wait_for_timeout(1500)
    page.screenshot(path=str(shots / "sessions.png"))
    print("list says:", page.locator("body").text_content()[:0] or "")

    for one in rows:
        page.goto(f"{base}/#/s/{one['name']}", timeout=20000)
        page.wait_for_timeout(2500)
        shot = shots / f"{one['name']}.png"
        page.screenshot(path=str(shot))
        state = page.evaluate(
            """() => {
              const v = document.getElementById('video');
              return v ? {hidden: v.hidden, src: v.currentSrc, dur: v.duration, err: v.error && v.error.code}
                       : null;
            }"""
        )
        lines = page.evaluate(
            "() => [...document.querySelectorAll('.line[data-t]')]"
            ".map(l => [parseFloat(l.dataset.t), l.textContent.trim().slice(0, 60)])"
        )
        print(f"{one['name']}: player {state}; {len(lines)} lines")
        if state and not state.get("hidden") and state.get("dur"):
            for t, text in lines:
                if not any(w in text.lower() for w in ("record this", "recording is on", "stop recording", "stopped and saved", "torque")):
                    continue
                page.evaluate("(t) => [...document.getElementById('talk').querySelectorAll('.line[data-t]')].find(l => parseFloat(l.dataset.t) === t).click()", t)
                page.wait_for_timeout(500)
                at = page.evaluate(f"() => document.getElementById('video').currentTime")
                print(f"   tapped {t:.1f} s {text!r}: the video goes to {at:.1f} s of "
                      f"{state['dur']:.1f} s")
    page.goto(f"{base}/#/highlights", timeout=20000)
    page.wait_for_timeout(2000)
    page.screenshot(path=str(shots / "highlights.png"))
    print("highlights text:", " ".join(page.locator("#v-highlights").text_content().split())[:300])
    print("browser errors:", errors or "none")
    browser.close()
