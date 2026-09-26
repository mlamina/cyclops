---
state: done
opened: 2026-09-26
---

# There is no way for me to change to a different Wi-Fi

Right now, there is no way for me to change to a different Wi-Fi in cyclops. If I boot up cyclops
and it has no Wi-Fi connection, I just see a black screen. Instead, I want to be able to select an
available Wi-Fi to join. The screen for joining the Wi-Fi should automatically pop up when cyclops
starts and there is no Wi-Fi available. In the menu that shows me the shutdown and reboot buttons,
there should be a Wi-Fi button which opens the Wi-Fi screen. i'm fine building a custom UI for it
if the native UI is not accessible or not very touch friendly. investigate first (pi is on) and
help me make a decision. take screenshots that I can observe.

## What was found first

Measured on the Pi on 2026-09-26, before any of this was planned:

- **There is no native UI to reuse, and no keyboard to type a password with.** No squeekboard,
  onboard, florence, matchbox or wvkbd is installed; `install-panel-look.sh` removed the taskbar
  that used to carry the stock wifi icon; Chromium's virtual keyboard is ChromeOS-only. Every
  route ends in a hand-built keyboard, so the question is only where to draw it.
- **The picker belongs on the native panel, not the admin page.** `overlay.py:2653` already
  argues this for the power menu - "the page is a browser that has to be uncovered, and the one
  moment you most want to shut a box down is the moment it is behaving badly enough that you
  would rather not ask Chromium for anything first". No network is that moment. And
  `cyclops-admin.service:40` sets `NoNewPrivileges=yes`, so the page could not run `nmcli`
  anyway - it would need the note-file dance the volume control uses.
- **Privilege is a non-issue in the kiosk.** `/etc/sudoers.d/010_pi-nopasswd` gives cyclops
  `NOPASSWD: ALL`, so `sudo -n nmcli` works exactly as `power.take_down` uses `sudo -n systemctl`.
  Unprivileged nmcli can read the cached list but is refused scan, join and radio (polkit checked).
- **The boot is not what is black.** With the radio off, `nm-online -s -q` returns in 0s, so
  nothing stalls, and a `grim` of the panel with Wi-Fi off is *identical* to one with Wi-Fi on -
  camera, eye, "Press button to start_", no hint at all. So the black screen comes after the
  button is pressed and the session cannot reach OpenAI. That path was not reproduced here (it
  needs a finger on the button with the network down); `ui.py:329-333` maps only
  `invalid_api_key` today and lets everything else through as the raw SDK exception.
- **A real scan** returned 18 rows / 14 unique SSIDs, with per-band duplicates and three hidden
  `--` entries. That sample is what the parsing test should be built from.

Screenshots in `~/Downloads/cyclops-wifi/`: `01` the panel now, `08` the panel with Wi-Fi **off**
(identical), `02`/`03` the menu before and with a WI-FI row, `04` the list from the real scan,
`06`/`07` the keyboard with SHIFT and with a wrong-password line.

## Plan

Everything is drawn in the kiosk's own renderer, in the vocabulary the power menu already uses:
scrim, rounded card, ruled rows, pressed-inverts. No browser, no new package on the Pi.

**`src/cyclops/wifi.py`** - new. `sudo -n nmcli` then bare `nmcli`, the two-route idiom from
`power.py:33`. Scan, join, forget, status. Parsing is pure and testable: `nmcli -t -f` output to a
list of networks, deduped by SSID, strongest kept, hidden dropped, **saved profiles first, then by
signal**. `deploy/install-tether.sh:35-67` is the prior art for the join, including
`autoconnect-priority` so a newly joined network is preferred next boot.

**`src/cyclops/overlay.py`** - two screens and their hit maps, beside `_menu_layout`/`menu_hit`:
- the list: 52px rows, six visible, a MORE row paging to the next six, CANCEL under its own rule.
- the keyboard: a field with SHOW, a status line, and QWERTY at 69x66 per key - larger than the
  menu's 62px thumb target - with SHIFT, 123, SPACE, DEL and JOIN.
- `MENU_ROWS` gains a `WIFI` row; `MENU_TITLE` becomes `SYSTEM`, since the card is no longer only
  power. `MENU_NOTE` stays as it is. A `_glyph_wifi` (three arcs and a dot) is added - **without
  it `overlay.py:9010` silently gives the new row the restart mark.**

**`src/cyclops/kiosk.py`** - a screen flag beside `_menu`, modal in `_on_mouse` the way the menu
is. Scanning and joining run on a worker thread; the render loop never waits on nmcli.
**`kiosk.py:907` turns any menu key that is not POWER_OFF into REBOOT** - the WIFI key must be
handled before that line.

**The two ways in, and the one way it announces itself:**
- the WI-FI row in the menu, always.
- at startup with no connection, the picker comes up on its own, before any button press.
- pressing the button with no network opens the picker instead of starting a doomed session
  (`kiosk.py:1454-1462`, before `controller.start()`), and `ui.py:329-333` learns to say
  "No Wi-Fi" for a connect failure rather than printing the SDK's exception.
- a small mark in the chrome **only when there is no connection**, absent otherwise.

Never on its own during a live session, and never on a drop mid-session - it must not become the
task. Out of scope: hidden SSIDs, WPA-Enterprise, captive portals.

**`tools/panel_shot.py`** gains `--wifi`, `--keyboard` and `--offline`, so every visual line below
can be rendered and judged **with no Pi**.

Nothing here talks to the model, so there is no live-session budget to agree.

## Done when

- [x] The menu has a WI-FI row, with a Wi-Fi mark and not the restart one — `uv run python tools/panel_shot.py --menu --out menu.png`, look at it
- [x] Choosing WI-FI opens the picker and does not reboot the box — pytest on `_choose`, guarding the `kiosk.py:907` fall-through
- [x] The four rows still clear a thumb and their labels still fit — `uv run pytest tests/test_power.py`, which already iterates `MENU_ROWS`
- [x] One row per SSID, saved first then strongest, hidden dropped — pytest over the 18-row scan sample captured above
- [x] Six networks fit, and MORE pages to the next six — `tools/panel_shot.py --wifi --out list.png`, look at it
- [x] The keyboard carries SHIFT, 123, SPACE, DEL, JOIN, and no key is under 62px tall — `tools/panel_shot.py --keyboard --out kb.png` plus a pytest over the hit map
- [ ] A saved network rejoins on one tap, with no password asked — yours, on the Pi
- [ ] A new network joins with its password typed on the panel, and is still joined after a reboot — yours, on the Pi
- [ ] A wrong password says so and leaves you on the keyboard with what you typed — yours, on the Pi
- [ ] Booting with no Wi-Fi puts the picker up on its own, with nothing pressed — yours, on the Pi, radio off then reboot
- [ ] Pressing the button with no network opens the picker instead of a failed session — yours, on the Pi
- [x] The chrome carries a no-connection mark only when there is no connection — `tools/panel_shot.py --offline` for both states, look at them
- [x] The picker never opens by itself while a session is live — pytest
- [x] A join never stalls the render loop — `tools/panel_shot.py --bench`, the numbers before and after in the outcome
- [ ] `uv run pytest` green, suite still under ten seconds — the run
- [x] `docs/panel.md:37-50` describes four rows, not three — read it

## Built — 2026-09-26
Hold his eye and the menu (now titled SYSTEM) has a WI-FI row with its own Wi-Fi mark, handled before the line that turns other rows into a reboot. It opens a list drawn on the panel from `sudo -n nmcli`: one row per SSID, the one in use first, then saved, then by signal, six a page. MORE pages on; with only one page that cell reads SCAN and looks again. A saved or open network joins on one tap. Anything else opens a keyboard (66 px keys, SHIFT one-shot, 123 and #+= for symbols, SHOW/HIDE, CANCEL back to the list). A wrong password says so and keeps the text. A failed new profile is deleted, and a joined one gets autoconnect-priority 100. Scans and joins run on worker threads. At boot the kiosk waits for NetworkManager (`nm-online -s`, up to 30 s), and if there's still no network the picker comes up once, never over a session. Pressing the button with no network opens the picker instead of starting a session. A session that fails on its socket now says "No Wi-Fi" instead of the SDK's error. While offline, an amber NO WI-FI plate sits on the top rail. Decided on the way: the picker switches the radio on before scanning (a radio switched off would otherwise list nothing, and the panel has no other switch for it), and the picker lets go of the panel after 120 s untouched. Unmet: the suite is green (1025 passed) but takes 20.8 s. It already took 20.4 s before this job, so the ten-second line was broken before this job started. The new tests add ~0.4 s.
Hands-on: `/try 018`, then: hold eye → WI-FI → tap a saved network (one tap, no password); join a new one with its password and reboot; type a wrong one; `sudo nmcli radio wifi off` then reboot and watch the picker come up by itself; with no network press the button and expect the picker, not a session.
factory/html/018.html
