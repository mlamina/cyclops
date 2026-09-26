---
state: done
opened: 2026-09-26
---

# In the menu that shows shutdown and reboot, I want another item that flips the screen

In the menu that shows shutdown and reboot, i want another item to select that flips the screen
orientation along the Y axis.

The only point of flipping the screen is because sometimes, I need to position the device upside
down.

## What was found first

Measured on the Pi on 2026-09-26, before any of this was planned.

- **The flip itself needs no code and no privilege.** `wlr-randr --output DSI-2 --transform 180`
  turns the panel over on the live box. It is a Wayland client call, so what it needs is the
  socket, not root — the kiosk has one, the admin service does not
  (`deploy/cyclops-admin.service:40`). `src/cyclops/backlight.py:8-11` is the precedent for a
  thing that needs no escalation at all.
- **Three readings of "along the Y axis" were captured from the glass**, in
  `~/Downloads/cyclops-flip/`: `t-flipped.png` is mirror writing, `t-flipped-180.png` is a
  reflection in water, and only `t-180.png` reads correctly once the box is turned over. **180°
  rotation is the transform**; the other two are ruled out.
- **`grim` compensates and `wf-recorder` does not.** A screenshot of a flipped panel comes out
  upright, so `panel_shot` and the admin page stay honest evidence. A session recording does not:
  `wf-recorder` hands over the physical framebuffer, so everything filmed while flipped is
  upside down. Verified both ways, same minute.
- **The camera is the other half.** `src/cyclops/webcam.py:41` sets `RPICAM_ROTATION = 180`
  because the module went into the case upside down. Turn the box over and the module is the
  right way up, so that correction has to come *off* — 180 → 0. A display transform cannot do
  this: the camera frame is composited into the picture the transform then rotates, so without
  it the bench stays upside down on the glass and in every photo sent to the model.
- **The connector name is not stable.** Fitting the Camera Module renumbered the panel from
  DSI-1 to DSI-2 once already, and `deploy/install-panel-look.sh:20-25` is the scar. Discover it,
  do not bake it in. The listing to parse:

  ```
  DSI-2 "(null) (null) (DSI-2)"
    Physical size: 154x86 mm
    Enabled: yes
    Modes:
      800x480 px, 60.028999 Hz (preferred, current)
    Position: 0,0
    Transform: normal
    Scale: 1.000000
  ```

- **Touch is mapped by the compositor, not by us** (`src/cyclops/kiosk.py:721-723`), and its
  mapping was stale: `~/.config/labwc/rc.xml` named output `DSI-1` and device `10-0038 generic
  ft5x06` while the box has `DSI-2` and `11-0038`. Fixed by hand today to `<touch
  mapToOutput="DSI-2" mouseEmulation="yes"/>`, with `deviceName` dropped so the i2c bus may
  renumber again without breaking it. **Whether labwc rotates taps with the output is still
  unverified — it needs a finger.**
- **There is room.** The card's height is a function of `len(MENU_ROWS)`, and the header came
  off it in `8329322`: four rows is now 268 px of 480 and five is 330. A fifth row needs no
  geometry change, and `tests/test_power.py` iterates `MENU_ROWS` so it covers it for free.
  Note the first row no longer draws its own rule (`overlay.py:9098`) - a new row in the middle
  of the list is unaffected, but one put *above* SHUT DOWN would move that exemption.
- **Job 018 merged while this was being explored** (`9f83671`), so the menu is already
  `SHUT DOWN / RESTART / WI-FI / CANCEL`, and its header came off afterwards in `8329322`. Every
  line number here is from after both.

## Plan

One row that turns the whole box over, and the three things that have to turn with it.

**`src/cyclops/flip.py`** — new. `enabled()` / `request()` over `~/.cache/cyclops/flip`, the pair
from `src/cyclops/steady.py:62-74`. `apply(on)` runs `wlr-randr --output <conn> --transform
180|normal` with the discipline of `wifi._nmcli` (`src/cyclops/wifi.py:131-146`): named timeout
constant, failure is a printed line and not an exception, **never called from the render loop**.
The connector comes from `wlr-randr`'s own listing, parsed by a pure function tested against the
sample above.

**`src/cyclops/config.py`** — `FLIP_FILE`, beside `STEADY_FILE`. `~/.cache`, not `/tmp`, for the
reason at `config.py:14-15`.

**`src/cyclops/overlay.py`** — a `FLIP` key and a `(FLIP, "FLIP SCREEN")` row between WI-FI and
CANCEL, plus a `_glyph_flip`. **Without its own glyph, `overlay.py:9119` silently gives the new
row the restart mark.** The card has no header any more, so the row is all there is to say it.

**`src/cyclops/kiosk.py`** — handle FLIP **above `kiosk.py:944`, which reads every other row as a
way to end the box**; that line is why `tests/test_wifi.py:87` exists and this job needs its
twin. Tapping it closes the menu, writes the note, and on a worker thread applies the transform
and restarts the camera. At startup the kiosk applies whatever the note says, so a box that
reboots while upside down comes back upside down.

**`src/cyclops/webcam.py`** — the rotation becomes `0 if flip.enabled() else RPICAM_ROTATION`.
Free in the ISP, so no per-frame cost; it does mean toggling restarts the camera and the picture
blinks for a second.

**`src/cyclops/screen.py`** — `cv2.flip(frame, -1)` in `_read()` (`screen.py:259-274`) while
flipped. One line, and it fixes the session video **and** the phone stream together. Deliberately
*not* a `-F` filter on `wf-recorder`: that path is raw `bgr0` on purpose, its ~15% of a core is a
measured number, and `screen.py:13-14` records that the bytes already arrive in an order a filter
could silently change — every red on the panel would come out blue.

**`deploy/install-panel-look.sh`** — write the touch mapping with the connector it already
discovers at lines 37-40, so a fresh box gets what was fixed by hand today.

The comments that say "two of its three rows" (`kiosk.py:773`, `tests/test_button.py:153,192`)
are already wrong and get wronger; fix the ones this job touches.

Out of scope: 90° and 270° (`kiosk.py:103-123` reads the DRM mode list, which a transform does
not change, so the window and the fullscreen watchdog would fight forever); the boot splash,
which Plymouth draws before a compositor exists and which stays the old way up for the ~20 s of a
boot taken while flipped; and the admin page, which has neither the socket nor the privilege.

Nothing here talks to the model, so there is no live-session budget to agree.

## Done when

- [x] The menu has a FLIP SCREEN row with its own mark, not the restart one — `uv run python tools/panel_shot.py --menu --out menu.png`, look at it
- [x] Five rows still clear a thumb and every label still fits beside its mark — `uv run pytest tests/test_power.py`, which iterates `MENU_ROWS`
- [x] Choosing FLIP SCREEN does not reboot the box — pytest on the `kiosk.py:944` fall-through, the twin of `tests/test_wifi.py:87`
- [x] The connector is discovered from `wlr-randr`, not hardcoded — pytest over the parser with the listing captured above
- [x] Applying a flip never runs on the render loop — pytest that the tap dispatches to a worker thread
- [ ] Tapping it turns the panel over, and the box reads right way up once you turn it — yours, on the Pi
- [ ] Taps land where you point while flipped — the menu rows, his eye, the Wi-Fi keyboard — yours, on the Pi
- [ ] The camera picture is the right way up while flipped — yours, on the Pi
- [ ] A box that reboots while flipped comes back flipped, and unflipping sticks the same way — yours, on the Pi
- [ ] A session recorded while flipped plays the right way up — yours, on the Pi, record a short one and look at the mp4
- [ ] The phone stream is the right way up while flipped — yours, on the Pi
- [x] `uv run pytest` green and no slower than before — the run (20.8 s before this job; the ten-second line was already broken)
- [x] `docs/panel.md:31-36` describes five rows, not four — read it

## Built — 2026-09-26
The menu is now SHUT DOWN / RESTART / WI-FI / FLIP SCREEN / CANCEL. FLIP SCREEN has its own mark,
an up arrow beside a down arrow. Tapping it closes the menu, writes `~/.cache/cyclops/flip`, and
then on a worker thread turns the panel over with `wlr-randr --transform 180` (the panel name
comes from wlr-randr's own listing) and restarts the camera module without its `--rotation 180`.
Tapping it again turns everything back. At startup the kiosk turns the panel over again if the
note says so. The screen recorder's reader checks the note twice a second and turns frames back
while flipped, which covers the session video and the phone stream. The restart is module-only:
a C920 is never taken down and put back up, because that wedges it. `install-panel-look.sh` now
writes the touch mapping for whichever output it finds, with no deviceName. Tests: 1026 passed,
~21 s before and after (20.8–21.0 s on master, 20.9–21.5 s here).
Hands-on: /try it, then tap FLIP SCREEN. Expect the panel to turn over and the camera to blink
for about a second, then show the bench right way up once the box is inverted. Check that taps
land under your finger: menu rows, the eye, the Wi-Fi keyboard. That depends on labwc turning
touch with the output, which is still unverified; if taps land mirrored, that is the one piece
missing. Then check a reboot while flipped, a short recording and the phone stream. The boot
splash stays the old way up (out of scope).
factory/html/019.html
