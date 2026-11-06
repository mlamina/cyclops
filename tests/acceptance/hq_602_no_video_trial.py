"""hq#602: the no-video session page as is and with a trial CSS rule; Highlights with no recordings."""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

base, name, shots = sys.argv[1], sys.argv[2], Path(sys.argv[3])
shots.mkdir(parents=True, exist_ok=True)
errors = []
with sync_playwright() as p:
    browser = p.chromium.launch()
    for w, h in ((800, 480), (1100, 760), (1600, 900)):
        page = browser.new_page(viewport={"width": w, "height": h})
        page.on("pageerror", lambda e: errors.append(f"pageerror {e}"))
        page.on("response", lambda r: errors.append(f"{r.status} {r.url}") if r.status >= 400 else None)
        page.goto(f"{base}/#/s/{name}", timeout=20000)
        page.wait_for_timeout(1500)
        page.screenshot(path=str(shots / f"asis-{w}.png"))
        print(w, "as is, title:", repr(page.locator("#stitle").text_content() if page.locator("#stitle").count() else None),
              "title visible:", page.locator(".stitle").first.is_visible(),
              "video visible:", page.locator("#video").is_visible())
        page.add_style_tag(content=".video[hidden] { display: none; }")
        page.wait_for_timeout(300)
        page.screenshot(path=str(shots / f"trial-{w}.png"))
        print(w, "trial, title visible:", page.locator(".stitle").first.is_visible(),
              "video visible:", page.locator("#video").is_visible(),
              "talk lines:", page.locator("#talk .line").count())
        page.close()
        page = browser.new_page(viewport={"width": w, "height": h})
        page.on("response", lambda r: errors.append(f"{r.status} {r.url}") if r.status >= 400 else None)
        page.goto(f"{base}/#/highlights", timeout=20000)
        page.wait_for_timeout(1500)
        page.screenshot(path=str(shots / f"highlights-empty-{w}.png"))
        print(w, "highlights text:", repr(page.locator("body").inner_text()[:260]))
        page.close()
    browser.close()
print("errors:", errors)
