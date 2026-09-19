---
state: done
opened: 2026-09-18
---

# Video stabilization — the mixed approach, because every video is shaky

> Would it be possible to implement video stabilization on cyclops?

> it already is a handheld device. every video is shaky

> so the actual video becomes smaller/cropped? is that a hard requirement?

> the mixed approach

"The mixed approach" means: use the picture the panel already throws away for sideways shake, let
the metal frame cover small up/down jolts, and crop slightly only for what is left.

Asked what camera-mode recordings (no metal frame to hide anything) should do: **steady them with a
crop**.

## Plan

### What it looks like when it works

You hold Cyclops and the picture on the panel stops jittering: it glides after your hand instead
of shaking with it. The session video is the panel (SCREEN is the Pi's record source), so it gets
the same steady picture for free. Photos for the model are untouched.

It will not fix blur *inside* a frame. At 15 fps indoors a hard jolt still smears that one frame;
the picture stops jumping, it does not get sharper.

### The shake, measured (2026-09-18)

Feature tracking on the camera area of the Pi's session videos, chrome and reticle masked out.
Jitter = the picture's path minus its own 1 s moving average, 90th percentile, in panel px:

| session | jitter p90 |
|---|---|
| `2026-09-18_12-32-46_stapler-staples-reload` | 31 |
| `2026-09-04_16-17-13_bmw-r80rt-build-status` | 41 |
| `2026-09-17_18-20-18_r80rt-front-brake-junction` | 20 |

The correction a ~0.5 s causal smoother would need, per axis: 12–25 panel px at p90, up to ~50 at
p99. Sessions where the panel held a photo measure ~0 and are not evidence of a steady camera.

Beware the obvious measurement: phase correlation over the panel centre reads ~0 px on every
session, because the static focus brackets dominate it. That wrong number nearly killed this job.

### 1. A stabilizer in the camera reader

New `src/cyclops/steady.py`, called from `CameraSource._read_loop` right after `cap.read()`
(camera.py:295–312 today — the repo moves under long plans, re-verify every line before touching
it). Per frame:

- **Measure** how far the picture moved since the last frame: corners
  (`goodFeaturesToTrack`) tracked with `calcOpticalFlowPyrLK` on a 320×180 grey downscale, with
  `estimateAffinePartial2D` and its inliers for a robust translation. `focus_score`
  (camera.py:48–57) already makes exactly that downscale and throws it away — compute it once and
  hand it to both.
- **Smooth**: keep an exponential smoothed path (time constant ~0.5 s at 15 fps) and shift the
  picture by `smoothed − actual`. That difference is the shake.
- **Clamp** the shift to the reach below. When the reach runs out, drag the smoothed path along
  with it, so the picture follows your hand instead of revealing an edge and instead of building
  up lag.
- **Translation only.** No roll correction in this job.
- One tunable for strength (the time constant), a module constant — no setting, no UI.

### 2. The reach — the mixed approach in numbers

1 panel px = 1.5 camera px at 1280×720. `fit_to_window` (overlay.py:8851) centre-crops 16:9 to 5:3,
throwing away 40 camera px each side. The metal frame is fully opaque (alpha 255) for exactly
11 panel px (≈16 camera px) on all four edges, in every panel state (`frame_band`,
overlay.py:745). Inside that, the top cove and side shadows are part-transparent — a gap there
shows.

- **Live picture:** a window of the raw frame at **96 % of its height** (a 4 % crop — the only
  visible cost), its centre moved by the correction. `fit_to_window` then does its usual crop and
  resize, so the side strip it already discards becomes sideways reach for free.
- **Sideways reach ≈ 80 camera px** (the discarded strip + the 4 % crop + the opaque band).
- **Up/down reach ≈ 30 camera px** (the 4 % crop + the opaque band).
- Past the raw frame's edge, fill with repeated edge pixels. The reach is sized so fill only ever
  lands where the panel does not show it: cropped away by `fit_to_window`, or under fully opaque
  metal. The gap test below is the arbiter of the exact numbers, not this paragraph.
- No extra full-frame resize: the shift and crop should cost no more than one frame copy.
  `fit_to_window` already resizes every frame.

### 3. Who gets what

- **Panel, SCREEN recordings, the admin-page camera stream** (companion.py:208): the steadied
  live picture, via `latest()` / `frame()`.
- **Photos for the model** (`snapshot()`, camera.py:160–182) and therefore pointing marks: raw,
  full frame, as today. `_recent` keeps the raw frames and the sharpest-frame score is computed
  on raw, so replicated edges never sway the pick. `snapshot()`'s fallback must hand out raw too.
- **CAMERA recordings** (kiosk.py:1600–1603 hands the recorder the `CameraSource` itself, and
  record.py samples `frame()`): steadied, then a **fixed 1120×630 window** of it — 16:9, about
  12 % tighter, and large enough in its margins (80 px each side, 45 top/bottom) that no fill is
  ever in frame. Fixed size, so the encoder's first-frame size holds all session.

### 4. Resets

- A reopen — framing change, reconnect, wake — starts from zero. Each entry into `_read_loop` is
  a fresh open; that is the natural place.
- A frame it cannot track (few corners, a whip pan, a hand over the lens, a blank wall, a step
  larger than any shake) counts as **no motion**, so the shift drains away over ~0.5 s instead of
  snapping back.

### 5. Scope

- 16:9 frames only (the Camera Module 3 Wide and the C920). The 640×480 endoscope passes through
  untouched, in both the live picture and CAMERA recordings.
- Frames too small to track pass through untouched — the existing camera tests feed `_read_loop`
  16×16 frames (tests/test_framing.py, tests/test_health.py).

### 6. The shake meter

`tools/shake.py <video.mp4>` prints the jitter p90 above for an 800×480 panel recording:

- Tracked features, not phase correlation (see the warning above).
- Mask: keep y 60–400, x 60–740; drop the focus brackets (x 335–465, y 175–305), the eye
  (x < 260, y > 260), the gauges (x > 610, y > 280) and the caption bar (y > 395).
- Per frame: `goodFeaturesToTrack` (200, 0.01, 8) → `calcOpticalFlowPyrLK` →
  `estimateAffinePartial2D` translation. Jitter = cumulative path minus its 15-frame moving
  average, 90th percentile of the distance, ends trimmed.
- `--bench`: the stabilizer's mean ms per frame on 1280×720 frames.

It is this job's check and the before/after number on the Pi.

### The synthetic clip, for a builder with no Pi

No raw camera footage exists off the Pi. Build one: take a photo from the main checkout's
`captures/` (e.g. `captures/latest.jpg`, 1024×576), enlarge it and cut a moving 1280×720 window
along a **seeded** handheld path — slow sway (0.5–2 Hz) plus tremor (4–8 Hz), sized so the
unstabilized render scores ≈30 px on the meter, like the stapler session. Render it through the
real panel path (`fit_to_window` + `composite` over `Overlay(800, 480)`, as `tools/panel_shot.py`
does) twice, without and with the stabilizer, and score both.

### Not known

- The CPU cost on the Pi. Expected 3–5 ms a frame (~5–8 % of one core at 15 fps). Not measured.
  The Pi runs warm; the render loop is 13.2 ms of a 33 ms slot (kiosk.py:130–136).
- Whether roll (twist) is visible enough to want a second job.
- Whether ~0.5 s feels right in the hand. That is Marco's to judge on the Pi.

## Done when

- [x] The synthetic handheld clip scores ≈30 px unstabilized and ≤ 12 px stabilized —
  `tools/shake.py` on both renders, both numbers in Built
- [x] A 100-px jolt never moves the picture past the reach, on either axis — pytest
- [x] At full reach in all 8 directions, no fill shows anywhere the metal frame is not fully
  opaque — pytest: fill painted a sentinel colour, `fit_to_window` + `composite` over one shared
  `Overlay(800, 480)`, zero sentinel pixels visible
- [x] At full reach in all 8 directions, a CAMERA-recording frame (the 1120×630 window) contains
  no fill — pytest, same sentinel method
- [x] A steady pan is followed: within 1.5 s of the pan stopping, the shift is back under 5 camera
  px — pytest
- [x] After a frame it cannot track, the shift never changes by more than 5 camera px from one
  frame to the next — pytest
- [x] `snapshot()` returns the raw, unshifted frame while the live frame is shifted — pytest
- [x] The stabilizer costs ≤ 1.5 ms a frame on the Mac (≈5 ms on the Pi at the ÷3.5 rule) —
  `tools/shake.py --bench`, the number in Built
- [x] A switch on the panel's settings screen turns stabilization on and off, on by default. Off,
  the panel, both kinds of recording and the admin stream get the raw picture, as before this job.
  It takes effect without a restart and survives one — pytest
- [ ] The switch fits: nothing on the settings screen is drawn under the close bar (its column
  already holds four controls and has no room for a fifth) — rendered on the Pi and looked at,
  screenshot in Built
- [x] The suite grows by less than 0.5 s — `uv run pytest --durations=10`, before and after
- [ ] With the panel live, the kiosk process uses at most 8 points of one core more than before —
  **yours, on the Pi**
- [ ] A handheld session recorded like the stapler one scores ≤ 15 px on `tools/shake.py` (stapler:
  31) — **yours, on the Pi**
- [ ] Panning across the bench by hand, the view follows your aim without feeling rubbery, and no
  black or smeared edge ever shows — **yours, on the Pi**
- [ ] A CAMERA-mode session video is steady and shows no edge fill — **yours, on the Pi**

## Stopped — 2026-09-18
Built as planned, but two here-checkable criteria conflict, so the state stays `ready`. The only
setting the plan allows is the time constant. At the plan's 0.5 s, the synthetic clip goes
30.0 → **11.1** px (target ≤ 10). A pan that uses the whole reach is still at **4.2** px 1.5 s
after it stops (target < 2). Even a 120 px/s pan is at 2.9. Pans need ≤ 0.39 s. The synthetic
clip needs ≥ 0.8 s (9.8, but a 10-seed median of 10.5 still misses). The limit is the vertical
reach: 29 camera px, set by the gap test. With unlimited vertical reach the synthetic clip would
score 7.9. Replaying the stapler session's own measured shake through the same smoother
(perfect tracking, no Pi) predicts **~25 px, not ≤ 15**. Its shake is mostly under 1 Hz, and
more crop barely helps: a 16% crop gets ~20. 18 big jumps (most likely the panel switching to a
photo) make up about a fifth of its 31. Everything else holds and is tested: the jolt clamp, no
fill at full reach (79 × 29) on the panel or in the 1120×630 recording window, the lost-frame
drain (this needed a 5 px/frame cap, DRAIN_PX, that the plan didn't have), raw photos, 0.9–1.05 ms
a frame on the Mac, and +0.22 s on the suite. The work is committed on the branch at 0.5 s, and
`tools/shake.py` is the meter (it reproduces 30.8 / 40.7 on the stapler / BMW videos).
**Needs from you:** either `/try 012` and judge it by eye (my recommendation — the fast jitter
is gone, and the p90 mostly measures slow sway), then relax the numbers to what it does; or
keep ≤ 15 as the goal, which needs a different approach, not a bigger crop.
factory/html/012.html

## Feedback — 2026-09-18
add a flag in the settings screen that lets me enable/disable this feature

Asked what to do about the two numbers round 1 stopped on (11.1 vs ≤ 10 px, 4.2 vs < 2 px):
"Relax to what it does".

## Built — 2026-09-18
Round 2: the stabilizer from round 1, now with a STABILIZE switch on the settings screen (on by
default). Off, the panel, SCREEN and CAMERA recordings and the admin stream all get the raw frame,
as before this job. The page writes a note and the camera's reader checks it twice a second, so a
flip lands within half a second and survives a restart. Photos for the model are raw either way.
The column had no room for a fifth row, so each switch's hint now sits under its name, next to the
switch, instead of on its own line. Gaps went 10 u → 8 u. The five rows end 0.05 px above the
close bar (headless 800×480 render on the Mac, on the page — not the Pi's glass, so that box stays
unticked). Master moved the camera to 30 fps after this was planned. The smoother is time-based,
so it behaves the same, but `tools/shake.py` now keeps its 15-frame window, because recordings are
still 15 fps. The synthetic clip feeds the stabilizer every camera frame and records one in two.
Numbers: synthetic clip **30.0 → 11.5 px** (six other seeds 10.5–13.7, median 12.1 — right at
the line). A full-reach pan is at **3.9 / 1.4 px** 1.5 s after it stops. **1.1 ms** a frame on
the Mac (LK tracking 0.6, corner-finding 0.4). Suite: median of 7 runs each, 17.06 → 17.14 s; the
new tests take 0.36 s.
**Watch the CPU:** at 30 fps the ÷3.5 rule predicts ~3.8 ms × 30 ≈ 11 % of a core, over the
8-point budget. If the Pi confirms it, measure every other frame or track fewer corners. And round
1's replay of the stapler shake predicted ~25 px, so the ≤ 15 real-session box will probably miss.
Hands-on: `/try 012`. Pan across the bench by hand. Flip STABILIZE on the SYSTEM screen and watch
the panel change. Check nothing sits under CLOSE. Record a handheld session, then run
`uv run python tools/shake.py <video>` on it. Check the kiosk's CPU with the panel live.
factory/html/012.html
