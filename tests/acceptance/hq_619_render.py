#!/usr/bin/env python3
"""hq#619: the panel's pages with today's browser flags and with the wall, side by side, on a laptop.

Usage: hq_619_render.py <shots dir> <admin port> <other loopback port>. Expects `cyclops-admin` serving
scratch folders on the admin port, and something answering on the other port, standing in for the
ears and camera servers. Needs playwright (`uv run --with playwright`). No Pi, no session, no mic.
Prints every failed request, whether each page matches pixel for pixel, and what the wall lets through.
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

from cyclops.kiosk import CHROME_FLAGS

shots, port, other = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
base = f"http://127.0.0.1:{port}"
shots.mkdir(parents=True, exist_ok=True)
ROUTES = ("/#/sessions", "/#/p/demo/overview", "/#/s/2026-01-02_10-00-00_torque-check")
NEW = ("--disable-background-networking", "--proxy-server=127.0.0.1:9")
assert all(flag in CHROME_FLAGS for flag in NEW), "the wall isn't in CHROME_FLAGS"


def tag(route: str) -> str:
    return route.strip("/#").replace("/", "_")[:24]


def visit(p, name: str, args: tuple[str, ...]) -> dict[str, bytes]:  # noqa: ANN001
    browser = p.chromium.launch(args=list(args))
    page = browser.new_page(viewport={"width": 800, "height": 480}, reduced_motion="reduce")
    seen: list[str] = []
    bad: list[str] = []
    page.on("pageerror", lambda e: bad.append(f"pageerror: {e}"))
    page.on("requestfinished", lambda r: seen.append(r.url))
    page.on("requestfailed", lambda r: bad.append(f"failed {r.url} {r.failure}"))
    page.on("response", lambda r: bad.append(f"{r.status} {r.url}") if r.status >= 400 else None)
    out = {}
    for route in ROUTES:
        page.goto("about:blank")
        page.goto(base + route, wait_until="load")
        page.wait_for_timeout(1500)
        out[route] = page.screenshot()
        (shots / f"{name}-{tag(route)}.png").write_bytes(out[route])
    print(f"{name}: {len(seen)} requests finished, {len(bad)} failed")
    for line in sorted(set(bad))[:20]:
        print("   ", line[:170])
    outside = sorted({u for u in seen if not u.startswith(("http://127.0.0.1", "data:", "blob:"))})
    print(f"    finished outside loopback: {outside or 'none'}")
    if name == "wall":
        page.goto(base + "/#/sessions")
        for label, url in (("loopback, another port", f"http://127.0.0.1:{other}/"),
                           ("the internet", "http://example.com/")):
            got = page.evaluate(
                "u => fetch(u, {mode: 'no-cors', cache: 'no-store'})"
                ".then(() => 'reached').catch(e => 'refused: ' + e.message)", url)
            print(f"    {label}: {got}")
    browser.close()
    return out


with sync_playwright() as p:
    # Neither of these two is about the network, and the headless shell refuses the second.
    usable = tuple(f for f in CHROME_FLAGS if f not in ("--kiosk", "--default-background-color=FF000000"))
    today = visit(p, "today", tuple(f for f in usable if f not in NEW))
    wall = visit(p, "wall", usable)
for route in ROUTES:
    print(f"{route}: {'identical' if today[route] == wall[route] else 'DIFFERENT'}")
