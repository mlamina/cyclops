---
state: done
opened: 2026-09-12
---

# when audio is played on a companion app, the volume knob on the kiosk should change appearance

when audio is played on a companion app, the volume knob on the kiosk should change appearance,
indicating this state

## Plan

The knob's **speaker mark** carries it. `Overlay._paint_speaker` (`overlay.py:8310`) draws a cone
and its throat in the gap under the knob's hub, inside the knob's cached tile, and today the only
thing that varies about it is its colour (`GREEN_MID` idle, `WHITE` under a finger, `GREEN_DIM`
with no sink). Under handover it becomes a *different mark*.

Which mark is Marco's to pick, from real renders and not from a description. Cut two or three at
panel scale — a phone, a cone with a slash, a cone with the sound leaving it — render them with
`panel_shot.py`, put them on the Pi, and ship your own pick first so there is something to react
to.

**The pointer and the lit arc stay green and live.** This is the one design call the exploration
changed, so do not undo it: under handover the *sink* still drives the sound cues. The knob's own
rung click, the shutter and the wake chime are deliberately not on the voice stream
(`kiosk.py:1787-1791`, `audio.py:360-364`) and still come out of the panel's amp at the knob's
level. The knob is not a dead control while a companion has his voice — only his voice has left.
Dimming the whole hand, or drawing it the way a knob with no sink behind it is drawn, would be a
lie about what the control still does.

**When it shows: whenever the claim is held, session or not.** Agreed at capture. A companion tab
holding the claim with nothing playing still gets the mark, because that is the case worth
catching — you glance at the panel *before* you start talking and see the voice is routed away.
The cost, accepted: a forgotten tab leaves a mark on the glass.

Four small edits:

- `Kiosk._handed_over` already tracks the state and today only prints a line (`kiosk.py:1802`).
  Pass it into the render call at `kiosk.py:2150`.
- `Overlay.render` takes a new keyword and forwards it to `_draw_hands` (`overlay.py:6012`), which
  forwards it to `_knob`.
- `Overlay._knob`'s cache key grows a third element (`overlay.py:7942`, today `(level, turning)`)
  and the tile picks the mark. Keep the white-under-a-finger behaviour: a finger on the knob still
  wins on the pointer, and the companion mark stays.
- `tools/panel_shot.py` gets a flag on the `--recording` pattern (argparse → one key in `shown()`'s
  `kw` dict, `panel_shot.py:45-66`), so the state is renderable headless on the Mac. Nothing can
  render it today.

Out of scope, unless Marco says otherwise: the column a drag opens, the gauge, the eye, the pilot
lamp, the companion page itself, and any change to the drag gesture.

**Known, and not this job's bug.** The signal is `companion.listening()` — an 8 s claim the page
renews every 3 s (`companion.py:62-63`). A backgrounded browser tab gets its timers throttled, so
it drops and retakes the voice about once a minute; the kiosk log from 2026-09-12 13:45 shows
exactly that, with "a companion has his voice" landing mid-sentence. The new mark will blink at
that rate, which is the mark telling the truth about a fault that is already there. Fixing the flap
is a separate job.

**Watch out for:** `tests/test_eye.py` asserts both dial hitboxes are byte-identical in every
state, and the pilot lamp's glow is capped so it can never reach the knob (`overlay.py:2126-2130`).
Anything you add lives inside the knob's own tile.

## Done when

- [x] The whole knob goes blue while a companion holds the voice — not one mark on it, the hand,
      the arc, the hub, the face — `panel_shot.py` with the flag, the two PNGs side by side
- [ ] You cannot miss it from a pace away, and without being told what changed — **yours, on the
      Pi**. The bar is "I saw it without looking for it", not "I can tell them apart"
- [ ] The knob takes no interaction while the claim is held: a drag does nothing, no column opens,
      no rung clicks — **yours, on the Pi**, a finger on it and nothing happens
- [x] It agrees with the amp: it goes blue within ~0.5 s of his voice leaving the panel and back to
      green within ~0.5 s of it coming back — a `grim` burst while opening and closing the
      companion page, against the `· a companion has his voice` lines in `/tmp/kiosk_live.log`
- [x] It is blue with the claim held and no session running — open the companion page, take no
      session, screenshot the panel
- [x] Render stays inside budget — `panel_shot.py --bench`, the number before and after
- [ ] Both dial hitboxes stay byte-identical in every state, and the suite stays under ten
      seconds — `uv run pytest`

## Built — 2026-09-12

The knob's speaker mark becomes a **phone** while a companion holds his voice, and nothing else on
the panel changes. The pointer, the lit arc and the rung click stay green and live, as agreed — the
sound cues are still on the panel's own amp at this knob's level, so anything that dimmed the hand
would be a lie about what the control still does. A finger still takes the whole hand white, mark
and all. It shows whenever the claim is held, session or not.

Four cuts were drawn at panel scale and looked at at 16×, and three of them lose for a reason worth
writing down: **slash** reads as *muted*, which is the one thing it must not say; **leaving** —
the cone with its arcs detached — is, at ten pixels, just the loudspeaker glyph every volume control
on earth draws, so it reads as *louder*; and an **arrow** beside the cone's wedge reads as a tick,
so that cut became a chevron, which is legible but is still a cone and so still reads as a volume.
The phone is the only one whose silhouette is not a volume state at all. All four are still in the
file behind one constant (`DIAL_AWAY`), so picking a different one is a word rather than a rebuild.

**The Pi was off the network for the whole build** — `cyclops.local` does not resolve and a sweep of
the LAN every forty seconds for half an hour found nothing. So nothing here has been deployed or
photographed on the real panel, and the three criteria that need one are not met. Everything that
`panel_shot.py` can answer is answered. The timing criterion is proved in the code instead of on the
glass: the mark and the amp are set from the same answer in the same call, 0.0 ms apart, so the
whole lag is one poll plus one frame — 433 ms.

Hands-on: `/try 003`, then look at the knob with the companion page open and with it closed, and
drag it while a companion has the voice. That is the whole of what is left to judge.

factory/html/003.html

Two notes on the last criterion, which is half met. The hitboxes do stay byte-identical, and there
is a new test that pins it — 946 pass. The suite is **17 seconds**, though, which is over the bar.
It was 17 seconds before this job as well: I measured it both ways, with the change and with the
three files put back, and the difference is noise. Not this job's doing, and worth its own.

## Feedback — 2026-09-12

> the volume control should be enabled, the whole knob should have a blue color and simply be an
> icon, no interactoin. the current solution is way too subtle and too easy to miss

Asked which way "enabled ... no interaction" cut, and he picked **dead — icon only**: while a
companion holds the voice the knob stops being a control. No drag, no column, no rung click. A blue
icon that says the voice is elsewhere, and that is all it is.

This overrides the plan's "the pointer and the lit arc stay green and live", and the reasoning under
it about the knob not being a dead control. That call was argued from the panel's own sound cues;
Marco has looked at the result and decided the other way. The earlier rounds stay as they were
written — this paragraph is what the next build follows.

Six of the seven criteria changed. Three are rewrites of what was there (unmissable rather than
legible; the whole knob blue rather than one mark; blue with the claim held and no session). One is
**inverted**: "the knob still drags, still opens its column and still clicks once per rung" is now
"the knob takes no interaction while the claim is held". The appearance and bench lines lost their
ticks, because what they were ticked for is not what is being built now.

Two things for the next round. `tests/test_eye.py` asserts both dial hitboxes are byte-identical in
every state — an inert knob should keep its hitbox geometry and ignore the events, not lose the
hitbox, so that test should still pass untouched. And the ten-second suite line is still on the
list, but it was 17 s before this job and 17 s after, measured both ways: it is a real miss and it
is not this job's to fix.

## Built — 2026-09-12 (round two)

The whole instrument goes over now. Pointer, lit arc, hub, graduations, unlit track, the reveal
that leaks into the seat of the bezel, and the mark in the gap — all of it blue, and the press
path taken away with it: a finger on the knob sets nothing, lights nothing, opens no column and
clicks no rung. It is consumed rather than ignored, because the fall-through behind that hitbox
is "stop talking" and a finger that landed square on a control has not missed every control. The
hitbox itself is untouched, so `test_eye.py` passes as it stood. A claim that lands while you are
already mid-drag ends the drag with it.

The blue is the panel's own — the one the eye wears while he works alone — rather than a second
blue invented for this. A new hue would argue with the green; this one already lives here. It is
the only blue in that corner, and on the Pi it is the only blue on the glass.

One thing I found on the way that is worth writing down. Painting the knob over the top does not
work: the graduations and the unlit track are printed under the glass and inside the ring, and a
second pass at them from outside leaves a green halo where the old phosphor skirt still shows,
plus a green tip on every tick where the ring's aperture cuts a moving mark shorter than a baked
one. So the still half of the knob is now **built twice from the same numbers** — one green
instrument, one blue — and the blue one is stamped over the green when the claim is held. The
bezel's seat reveal takes the face's phosphor as an argument for the same reason: a blue dial in
a ring with a green glow round it is two instruments in one hole. The green panel comes out
byte-for-byte what it was, which I checked rather than assumed.

**The Pi was up for this one.** It is photographed on the real glass, and the timing criterion is
measured there rather than argued from the code: 1252 grabs of the knob 18 ms apart, against the
kiosk's own lines stamped as they were written. The glass went blue 27 ms after the amp switched
and green 11 ms after it switched back — both inside one frame, because both are set from one
answer in one call. With the 0.4 s poll in front of it that is 0.43 s worst case from the page
opening. Closing the page shows green about eight seconds later, which is the claim going stale
and not the panel being slow; that is the mechanism that takes his voice back when a phone walks
out of wifi range.

Two things about the state of the Pi, both honest rather than tidy. **Job 002 was building at the
same time** and deploys to the same box, so it pushed over this build partway through — my
measurements were all taken before that and on this code, which I checked frame by frame. And
when I went to push again at the end, **the Pi dropped off the network**: `cyclops.local` stopped
resolving mid-rsync, and it had not come back twenty minutes later — three sweeps of the subnet
in that time turned up no host answering as cyclops. So the last push that actually landed on the
box was 002's, and `/try 003` is how you put this one on the glass. Everything photographed above
was taken while this build was running there, which I checked frame by frame before trusting it.

The suite line is still half met and I re-measured it myself rather than repeating the last
round's word for it: three runs at the branch point were 16.9 / 18.0 / 17.5 s for 945 tests, and
three runs here were 17.2 / 16.8 / 16.9 s for 949. Same, within noise, and over the ten-second
bar either way. A real miss, not this job's.

Hands-on: `/try 003`, then open the companion page on your phone and look at the corner from
where you would actually be standing — that is the criterion I cannot judge. Then put a finger on
the knob and drag it while the page is open: nothing should happen at all, no column, no click.
Close the page and it comes back green about eight seconds later.

factory/html/003.html
