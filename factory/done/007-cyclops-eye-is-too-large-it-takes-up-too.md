---
state: done
opened: 2026-09-18
---

# Cyclops' eye is too large — shrink its corner to match the bottom-right one

Cyclops' eye is too large, it takes up too much of the screen. Reduce the size of the bottom-left
corner UI so it matches the size of the bottom-right corner.

Then, on the mockup: "the size is good. make sure the status screen stays centered" — and, asked
whether the centred screen should grow to fill the freed space: **same width, centred**.

## Plan

**The bottom-left bracket becomes the mirror of the bottom-right one, the eye shrinks to 75% to
sit in it, and the status screen keeps its width and moves to the middle.**

Marco approved the size from a 75% mockup. The target render is a scratch spike of exactly this
plan, in `~/Downloads/cyclops-eye-spike-listening.png` and `-idle.png` on the Mac. The spike
measured: eye (87, 393), r 66, rim 21 px clear of both edges, screen x 224–576 (352 wide), clamp
arms 61 px each.

1. **Three constants in `overlay.py`** (around `:1282` and `:1718`):
   - `BOT_L` 250 → **170**, the same as `BOT_R_OUT`, so the two corner brackets mirror each other.
   - `EYE_R` 0.1833 → **0.1375** (88 → 66 px, the 75% Marco approved).
   - `EYE_SEAT` 0.5 → **0.3**. This seats the eye less deep, so the rim keeps clear of the border glow
     (21 px against the 16 the halo needs) and the swell stays on the panel.
   - `EYE_SHOULDER`, `RAIL`, the bolts and the cables' section stay as they are. The steel keeps
     the same line weight as the rest of the panel; only the geometry shrinks.
2. **The status screen is the same width and centred.**
   - Today its edges are solved from the two rails (`:3715-3772`). That makes it 352 wide at
     x 264–616, which is off-centre.
   - Give it a fixed width instead: 352 reference px (scaled like everything else), centred on the
     panel. At other window sizes it must shrink rather than cross either rail's clearance.
   - The clamps (`self.ears`) are already sized from each rail to the case's edge, so they
     lengthen on their own to bridge the gap. Nothing else about them changes.
   - Everything measured from the screen follows it: the caption, the tube, the ears and the label
     keep-out.
3. **Fixes the move forces:**
   - `eye.LANDMARKS` (`eye.py:465-472`) are gaze unit vectors hard-coded from the old centre and
     the old `caption_left`. Recompute all three (FRAME, WORDS, DIALS) from the new layout.
   - `LOOM_REACH` (`overlay.py:2100`) was tuned to a 47 px pocket that is now bigger. Lengthen the
     cables if they end visibly inside the frame.
   - `tests/test_eye.py:1076` hard-codes `r = 88`, and `tools/iris_strip.py:30` hard-codes
     `SIZE = 176`. Move both to the new size.
   - Update comments that quote 88 px.
   - Pixel tests whose thresholds were tuned at r = 88 (`_face()` users, `_rim_profile`, the
     collar crop in `test_power.py`): re-run them. Adjust only where the smaller size genuinely
     moved the number, and say which ones and why. Never loosen a threshold just to go green.
4. **Nothing else moves:** the dials, the header pod, the reticle and the admin page's own eye.

## Done when
- [x] The bottom-left corner takes the same space as the bottom-right one — `panel_shot.py --state listening`
  render. The spike PNGs are the approved *size* only; they still show the old thick rim, so they are no
  longer a pixel target.
- [x] The thick half-round steel rim is gone. Only the thin brass edge runs all the way around the eye,
  and the mount points are exactly where they are today — `panel_shot.py --state listening` and
  `--state idle`, bottom-left corner looked at
- [x] The status screen keeps its width and is centred — `uv run python -c "from cyclops.overlay import Overlay; t=Overlay(800,480).term; print(t.w, t.x, 800-t.x-t.w)"` prints `352 224 224`
- [x] The clamps still reach from each rail to the screen, with no gap and no overlap — `panel_shot.py --state listening`, bottom strip looked at
- [x] The gaze still lands on the frame, the words and the dials — `uv run pytest` (`test_the_eyes_own_landmarks_are_the_panels`)
- [x] The cables still run off the edge of the panel — `panel_shot.py --state listening`, bottom-left corner looked at
- [x] The eye is still alive at 66 px, with blades you can tell apart and motion in every frame — `panel_shot.py --strip 8 --seconds 6`, next to the same strip from master
- [x] Rendering gets cheaper — `panel_shot.py --bench`, listening and speaking render ms before and after (the spike measured 11.6 → 9.2 ms on the Mac)
- [ ] Tapping and holding the eye still work, and the power menu stays clear of the eye — `uv run pytest`, whole suite green and under 10 s
  - Green, 985 passed, and the tap, hold and power-menu tests pass. Not ticked because it took about 17 s, not under 10. Master took the same 17 s back-to-back on this Mac while job 005 was running, so re-time it on a quiet machine.
- [ ] From a pace away, the eye still reads as a face and the status screen looks centred — yours, on the Pi
- [ ] A finger hits the smaller eye on the first try, for both tap and hold — yours, on the Pi

## Built — 2026-09-18
The bottom-left bracket now reaches 170 like the bottom-right one. The eye is 66 px and sits
less deep in its bracket, and the status screen is 352 wide and centred (224 px either side).
The render is pixel-identical to `cyclops-eye-spike-listening.png` and `-idle.png`. The clamps
get longer to reach the screen on their own. The cables didn't need `LOOM_REACH` changed: they
now end about 20 px past the corner, up from 14. The three gaze directions were recalculated for
the new layout. Render cost on the Mac went from 11.7 to 9.3 ms (listening) and from 11.1 to
8.7 ms (speaking). Tapping the eye is still covered: its tap target is now 132 px, against 72
for each dial.
One call I made: `test_the_screen_stands_clear_of_both_mounts` capped the gap between the screen
and each rail at half the screen's height. The approved design goes past that cap on purpose
(47.5 against 36), because centring the screen is what makes the clamps longer. I removed the
cap. The test still checks that each clamp touches both the rail and the screen. The pixel tests
that were tuned at r88 (`_face`, `_rim_profile`, the collar crop) all passed without changes.
The suite is green (985 tests) but took about 17 s. Master took the same 17 s back-to-back
while job 005 was running on this Mac (load average 8–17), so the slowdown is the machine, not
this change. Nothing has been measured on the Pi.
Hands-on: `/try 007`. Look at it from a pace away and check it still reads as a face with the
screen centred. Then tap the eye and hold it: the tap should open what's been kept, the hold
should open the power menu, and each should land on the first try.
factory/html/007.html

## Feedback — 2026-09-18
The size looks great, but the thick, half-rounded rounded metal rile lining the eye is too much
now. The eye already has the thin brass-looking outer edge, which looks beautiful. Just let that
go all around the eye, with the mount points staying exactly as they are

## Built — 2026-09-18 (round 2)
The mount's rail no longer bends round the eye. It runs straight up its ramp and stops at the
collar, bolted on both sides, so the thin brass edge now shows all the way round the eye. All six
bolts in that corner are exactly where they were: the two where the rail bends, the two where it
meets the collar, and the collar's own two. Their heads are pixel-identical to last round when
listening and when asleep. Two calls I made on the way. First, the long-press fill used to light
up the rail, so now it lights up the brass, along the same arc between the same two bolts.
Second, the thin steel ring just inside the brass stays. It was always there under the brass on
the lower-left, so I read it as part of the edge Marco liked. Say if it should go too. The eye's
housing now fits inside the 170 px corner (x 5–169); last round the rail round it reached x 177.
Screen, clamps, cables and gaze are unchanged. Render cost is the same as last round (listening
9.2–9.5 ms, speaking 8.6–8.8 ms) and still about 2 ms under master. The suite is green (985
tests) in about 16.5 s. Master took the same time back-to-back (16.1 and 17.1 s) with other jobs
running, so the under-10 s line stays open until it's re-timed on a quiet machine. Nothing has
been measured on the Pi.
Hands-on: `/try 007`. Look at the corner from a pace away. Then tap the eye and hold it: the
brass should fill green over the top of the eye and then open the power menu.
factory/html/007.html
