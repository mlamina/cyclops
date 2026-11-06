#!/usr/bin/env python3
"""hq#602: is the player really gone on a session with no video? `hidden` alone doesn't prove it.

Usage: hq_602_no_video_player.py <base url> <no-video session name> <shots dir>. Expects
cyclops-admin serving a card with that session. Needs playwright (`uv run --with playwright`).
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

base, name, shots = sys.argv[1], sys.argv[2], Path(sys.argv[3])
shots.mkdir(parents=True, exist_ok=True)
MEASURE = """() => {
  const v = document.getElementById('video'), t = document.getElementById('talk');
  const box = (e) => { const r = e.getBoundingClientRect(); return [r.x, r.y, r.width, r.height].map(Math.round); };
  return {hidden: v.hidden, display: getComputedStyle(v).display, video: box(v), controls: v.controls,
          lines: t.querySelectorAll('.line').length, talk: box(t), viewport: [innerWidth, innerHeight]};
}"""
with sync_playwright() as p:
    for browser_name, launch in (("chromium", p.chromium), ("webkit", p.webkit)):
        browser = launch.launch()
        for width, height in ((800, 480), (1100, 760), (390, 844)):
            page = browser.new_page(viewport={"width": width, "height": height})
            page.goto(f"{base}/#/s/{name}", timeout=20000)
            page.wait_for_timeout(2000)
            print(browser_name, f"{width}x{height}", page.evaluate(MEASURE))
            page.screenshot(path=str(shots / f"no-video-{browser_name}-{width}.png"))
            page.close()
        page = browser.new_page(viewport={"width": 1100, "height": 760})
        page.goto(f"{base}/#/sessions", timeout=20000)
        page.wait_for_timeout(1500)
        row = page.locator(".thumb.none").first
        print(browser_name, "list row says:", row.text_content(), "· visible:", row.is_visible())
        row.scroll_into_view_if_needed()
        page.screenshot(path=str(shots / f"list-{browser_name}.png"))
        browser.close()
