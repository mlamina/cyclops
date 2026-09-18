---
state: done
opened: 2026-09-18
---

# A manual page fills the whole width of the screen and scrolls with a finger

when a manual page is displayed, I want it to automatically fill the entire width of the screen,
allowing me to scroll up and down using my finger. right now, a portrait page is fit onto the wide
screen, creating large black bars on each side and making it difficult to read

Asked whether Cyclops should also move the page itself (by voice, or by opening it at the
answer): **finger only**. The page opens at the top.

## Plan

**Today:** a recalled manual page takes the same path as every other picture:
`_show_found` → `imagine.for_panel` (1024 on the long edge) → `panel.offer_image` →
`panel.js` `shot()`. There, `.stage .shot` is `object-fit: contain` inside the bezel and the
17 px gutter (766×446), so an A4 page shows at about 315 px wide with about 225 px of black on
each side. Any `pointerdown` on `#stage` puts the picture away straight away (`panel.js:34-59`),
so today a scroll gesture would dismiss the page.

1. **The page knows it is a page.** `_recall` already knows `best.item.kind == "page"`
   (`agent.py:~3023`). Pass that through `_show_found` to `panel.offer_image`, which adds a marker
   to the payload next to `image` (for example `"page": true`). `image` stays the key the picture
   travels under, so `still.py` and the session recording keep working unchanged. Photos, edits
   and diagrams send exactly what they send today. Update `panel.py`'s "one kind of picture"
   docstring so it stays true.
2. **Sharp at full width.** A page goes to the panel 800 px wide, height following (A4 comes
   out 800×1131), instead of `for_panel`'s 1024 on the long edge. That way filling the width never
   upscales. Every other picture keeps `for_panel` as it is.
3. **Edge to edge.** In `panel.js`, a page sets its own body class. That class drops the bezel
   and the body padding, the way `body.paper` does for a scratchpad (`panel.css:76-77`). The image
   is `width: 100%; height: auto` inside a stage with `overflow-y: auto`. It opens scrolled to the
   top. No scrollbar: text running off the bottom edge is the cue. Landscape (rotated) pages fill
   the width too, with a little scroll.
4. **Finger scroll.** On the Pi a finger arrives as a mouse (labwc `mouseEmulation`), so native
   touch panning is not enough. Reuse `dragScroll` from `app.js:1422` on the stage, panel only as
   it is for the lists: 6 px slop, drag to scroll, flick to glide. A companion's own touchscreen
   pans natively through `overflow-y: auto`.
5. **Tap to put it away.** On a page only, dismissal moves from `pointerdown` to a tap: a press
   and lift that stayed within the slop. A drag never dismisses; `dragScroll` already swallows the
   click after a drag. Every other picture keeps putting itself away on the press, exactly as
   today.

Unchanged: the recording's copy (the whole page, letterboxed), point_at, the 15-minute cap, and a
new offer replacing the old one (the new page opens at the top). No voice scrolling and no new
tool.

## Done when
- [x] A manual page fills the full 800 px width, edge to edge, with the top of the page at the top of the screen — `node tests/render_check.mjs`, a new A4-shaped page case: the image's measured width is 800 and its left edge is 0, plus a screenshot
- [x] Dragging scrolls the page, the page stays up, and the bottom can be reached — `render_check.mjs`: a 200 px mouse drag changes the stage's scrollTop, repeated drags reach scrollHeight − clientHeight, and nothing is POSTed to `/close`
- [x] A tap without dragging puts the page away — `render_check.mjs`: a click on the page POSTs `/close`
- [x] Photos and diagrams look and behave as before: contained inside the bezel and put away on the press — `render_check.mjs`, the existing photo and drawing screenshots next to master's, and a press on a photo still POSTs `/close`
- [ ] A 1240×1754 page leaves for the panel 800 px wide, and a photo still leaves at 1024 on the long edge — `uv run pytest`, whole suite green and under 10 s
- [ ] Asked about something in a real manual, the page comes up full width and is readable from working distance — yours, on the Pi
- [ ] A finger drag scrolls smoothly, a flick glides to the bottom, and a drag never puts the page away — yours, on the Pi
- [ ] One tap puts the page away, first try — yours, on the Pi

## Built — 2026-09-18
A recalled manual page now travels to the panel 800 px wide, height following (a 150 dpi A4 page is
800×1132, not 1131 — rounding), and the offer carries `page: true`. The panel drops the bezel and
gutter for it, lays it edge to edge from the top, and scrolls it: on the kiosk the drag is turned
into a scroll exactly as the lists do it (dragScroll, now with an optional "only while" gate so the
stage only scrolls while a page is up and the video seeker keeps its own drag); a companion's
touchscreen pans natively. A page goes away on a tap, never on a drag; every other picture still
goes away on the press. Render check: 800 px at x 0, top 0; a 200 px drag moved it 200 px (652
after the glide), drags reached 652 of 652, no close while dragging, one click → one close; photo,
drawing, scratchpad, sketch and companion screenshots are pixel-identical to master's, and a
press on a photo still closes before the lift. Real pages (Pi 5 brief p2, A4; mo.unit p12,
landscape) rendered before/after on the page.
Unticked on purpose: the pytest line. The new test passes and the suite is green (966), but it
runs 16–17 s on this Mac at load ~5 with other builds going — master runs 17.5 s at the same load,
and the new test costs 0.05 s, so the ten-second bar is missed by the machine, not by this job.
The render check's console errors (connection refused) are the same on master: the live stream
isn't running on a Mac.
Hands-on: `/try` it and ask about something in a real manual. Expect the page full width at the
top, a finger drag to scroll, a flick to glide, one tap to put it away. Edge case worth a look: a
tap while a flick is still gliding puts the page away rather than just stopping the glide.
factory/html/010.html
