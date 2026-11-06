#!/usr/bin/env python3
"""hq#602: on a real recording started mid-session, does tapping a line land where it was said?

Usage: hq_602_seek.py <base url, by LAN address so the transcript shows> <session name> <log path>.
Expects cyclops-admin serving that session. WebKit, because Playwright's Chromium has no H.264.
"""

from __future__ import annotations

import json
import sys

from playwright.sync_api import sync_playwright

base, name, log = sys.argv[1], sys.argv[2], sys.argv[3]
records = [json.loads(x) for x in open(log, encoding="utf-8")]
began = next(float(r.get("video_at") or 0) for r in records if r["type"] == "recording_asked")
print(f"video starts at {began} s on the session's clock")
with sync_playwright() as p:
    browser = p.webkit.launch()
    page = browser.new_page(viewport={"width": 1100, "height": 760})
    page.goto(f"{base}/#/s/{name}", timeout=20000)
    page.wait_for_timeout(2500)
    print("duration:", page.evaluate("() => document.getElementById('video').duration"))
    lines = page.evaluate(
        "() => [...document.querySelectorAll('#talk .line[data-t]')]"
        ".map((l, i) => [i, parseFloat(l.dataset.t), l.textContent.slice(0, 50)])")
    for index, t, text in lines[:14]:
        page.evaluate(
            "(i) => { document.querySelectorAll('#talk .line[data-t]')[i].click();"
            " document.getElementById('video').pause(); }", index)
        page.wait_for_timeout(500)
        at = page.evaluate("() => document.getElementById('video').currentTime")
        print(f"  video_t {t:6.2f} -> landed {at:6.2f} · {text!r}")
    browser.close()
