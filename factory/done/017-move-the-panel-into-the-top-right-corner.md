---
state: done
opened: 2026-09-20
---

# The top-centre panel moves into the top-right corner, and the USB module gets its glass

the UI panel in the top-center of the screen should move into the top-right corner. the USB panel
should not have a transparent background , but the same background as the center panel. the goal is
to create a more unified look

Agreed at capture: the pod goes **flush into the corner** — plate running off the top and right
edges, steel only on its two inner edges, mirroring the USB module — rather than keeping its
free-standing shape near the edge. And "the same background as the center panel" means **the pod's
own window**: the ~80 % opaque dark-green instrument glass with the phosphor tint and the rolled
steel flange, not the fully opaque screen of the caption box at the bottom.

## Plan

### One routine draws both windows

`_draw_pod_face` (`src/cyclops/overlay.py:6313`) and its helpers `_pod_field`, `_pod_glass` and
`_pod_rake` are keyed on `tags` and reach into `self.pods[tags]` / `self.pod_boxes[tags]`. They get
re-cut to take a `Bracket` and a `Rect`, so the same flange → reveal → glass → rake → glare pass can
be run over the USB module's spine as well as the pod's. That shared routine *is* the unified look;
everything else in this job is placement. Do not fork the pass and tune a second copy — two windows
that agree by coincidence will drift the first time either is touched.

### The USB module gets the pod's window

`_usb_chassis` (`overlay.py:7044`) lays the shared plate — `self._filter`, ~70 % opaque — through
the module's polygon and stops, which is why the camera picture moves behind the device names. It
gains the pod's pass inside its rail:

* the `POD_LAND` (4.0) steel flange in from the rail, with `POD_REVEAL` (1.0) before the glass
* the terminal's glass, `TERM_ALPHA` (205) over `SCREEN`, with `POD_TINT_A` phosphor and the
  `TERM_SCAN` raster
* the `POD_REBATE` step and `POD_AO` wall falloff, so the pane sits *below* the steel
* its own rake lamp off its outer shoulder, the same fiction `_pod_rake` uses, so every face on the
  module agrees about where the light is

It stays baked once per width and cached in `self._usb`, so a frame still costs one dictionary
lookup — the bake gets more expensive, the frame does not. The category glyphs, the device names and
`NO USB` all sit on the glass instead of on the picture.

### The pod moves into the top-right corner, mirrored

`left = width // 2 - flat // 2` (`overlay.py:3695`) becomes a right anchor: the flat ends at the
panel's right edge, and the spine drops its right-hand ramp to close against the corner `(width, 0)`
exactly the way the USB module's closes against `(0, 0)`.

* The plate runs off the **top and right** edges — pixel `(width - 1, 0)` is plate. This is the same
  corner fix `bd4ca47` made on the left: cutting the plate back to the case's rounded corner leaves
  the surround's bright corner shining through as a grey nub, which is the one reading a chassis
  part cannot have.
* Steel, bolts and the cast shadow stay on the bottom run and the single chamfered **left** end.
* `CYCLOPS` stays milled into that rail via the same `_legend_box` / `_pocket` / `_mark` trio, now
  anchored off the left knee — the mirror of where `USB DEVICES` sits.
* Readouts keep their order and spacing (meter, tags, clock), still measured off `pod_boxes[tags]`,
  so the clock lands hard against the right edge.
* Only the **left** knee moves as REC and HOT light. Nothing at the right edge shifts, which is the
  same promise the centred pod made by growing symmetrically.

### The head rail's right-hand bend

`_draw_head` (`overlay.py:5067`) runs the rail across at `HEAD_DROP` and turns it down into a leg at
x ≈ 778, which is exactly where the pod's new corner lands. That collision is already solved at the
other end, where the USB module sits over the rail's left bend — give the pod the same treatment, so
the rail dies into the module rather than crossing it.

### The USB module keeps the room it has today

`usb_room()` (`overlay.py:6993`) derives its width budget from the pod's left knee: 203 reference px
at 800×480. With the pod out of the middle that would silently become ~450 and a long device list
would sprawl towards the reticle. It becomes a constant at today's value instead. **Which devices
show, and where they drop off the rail, must not change in this job.**

### Tests that pin the old arrangement

Rewritten to the new claim, never deleted and never loosened to pass:

* `test_every_pod_is_centred_and_packed` (`tests/test_eye.py:1536`) — the "centred" half becomes
  "right-anchored"; the "packed" half is still true and still worth keeping
* `test_it_clears_the_pod_at_its_widest` (`tests/test_devices.py:304`)
* the HOT-lamp symmetry check (`tests/test_health.py:137`)

### Cost

Nothing here talks to the model. No live realtime sessions, no API spend.

## Done when

- [x] the pod's plate reaches the panel's top and right edges with no frame showing between it and either — `uv run pytest tests/test_devices.py`, read off the polygon, mirroring `test_the_plate_reaches_the_top_and_the_left`
- [x] the pod has one chamfer, on its left end, and steel only on that end and its bottom run — same file, off the spine
- [x] nothing at the right edge moves when REC and HOT light — pytest: `pod_boxes[0].right == pod_boxes[2].right`
- [x] the USB window lets no more of the room through than the pod's does — pytest: two different photos behind the same frame, comparing how much the pixels inside each window change
- [x] the two top corners are the same part — pytest: same depth, same flange width, same glass treatment, read off both modules
- [x] the USB list shows exactly the devices it shows today — `uv run pytest tests/test_devices.py`, the width table and the drop-off order unchanged
- [ ] the suite passes and stays under ten seconds — `uv run pytest`
  - Not met: 999 pass, but in 20.0 s. It was already 19.2 s before this job (994 tests, same machine); the new tests add under half a second. The slow tests are older recall and caption ones this job doesn't touch.
- [x] per-frame render cost has not grown — `uv run python tools/panel_shot.py --bench`, the numbers before and after written into the outcome
- [x] four renders on the outcome page — empty, one device, two devices, four devices — `uv run python tools/panel_shot.py --bg captures/<a real photo>.jpg --usb ...`
- [ ] the top row reads as one machine from a pace away — yours, on the Pi
  - Needs /try. It should read as two mirrored corner parts joined by the head rail, with nothing floating in the middle.
- [ ] the clock and the meter are still readable in the corner at arm's length — yours, on the Pi
  - Needs /try. Same size and brightness as before; only the position changed.

## Built — 2026-09-20
The pod now sits flush in the top-right corner as the USB module's mirror. Its plate runs off the top and right edges, with steel only on the bottom run and one chamfered left end. CYCLOPS is cut into the rail off the left knee, and only that knee moves as REC and HOT light. The window pass (flange, reveal, glass, rake, glare) now takes a bracket and runs over both modules, so the USB module has the same dark-green glass instead of a see-through plate: its clearest pixel lets 2.35% of the room through, like the pod's, down from 14%. Each module's lamp and reflection sit at its own inner shoulder, so the pair mirror each other. The head rail's right bend is hidden under the pod the same way its left bend is under the USB module; nothing needed changing there. The USB room is now a fixed 203 reference px (today's value), so the list and its drop-off are unchanged at 800×480 and 1280×720. The one knock-on is the unused 480×320 test size, which now fits two names instead of one. The eye's "dials" glance was re-aimed at the pod's new spot. Frame cost is unchanged (listening 9.1–9.3 ms both sides). The suite passes but runs in 20 s, which was already 19 s before this job.
Hands-on: /try 017 — stand a pace back and look at the top row, then check the clock and meter at arm's length. Start a recording and only the pod's left end should move.
factory/html/017.html
