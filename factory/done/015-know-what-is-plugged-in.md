---
state: done
opened: 2026-09-20
---

# Cyclops should be aware of the USB devices connected to it

Cyclops should be aware of the USB devices connected to it -except for the lav mic that's always
connected and should be ignored. it should recognize when i connect or disconnect usb devices,
identify the device name and category (music/sound, camera, storage, other) and display a
category icon + device name on the kiosk UI. it should also have a list of connected USB devices
in its instructions when we start a new session. cuyclops should dynamically adjust its greeting
message based on the USB devices connected. e.g. if the snake cam / endoscope is connected, it
should infer that we're taking a close-up look at something. if a synth is connected, it should
infer that we're gonna do some music stuff

## Plan

### What is on the bus today

Read off the Pi on 2026-09-20, because the naming and the categories both turn on it:

```
1-1   2ce3:3828  dclass=ef  mfr=(empty)            prod=(empty)                ifaces=ff,ff
3-1   4c4a:4155  dclass=00  mfr=Jieli Technology   prod=USB Composite Device   ifaces=01,01,03
usb1-4 1d6b:*    dclass=09  ...                    xHCI Host Controller        ifaces=09
```

Three facts that set the design. The **endoscope reports no manufacturer and no product string at
all**, and its interfaces are vendor-specific (`ff`), so neither its name nor its category can be
read off the bus - it has to come from its ID. The **lav mic is `4c4a:4155`**, and its product
string is "USB Composite Device", which is not a name anybody would want on the glass anyway. The
**root hubs are device class `09`**, which is how they get skipped.

### Knowing

A new module, `cyclops/devices.py`. One pass over `/sys/bus/usb/devices` - no new dependency, no
`lsusb`, no pyusb, no udev hook. Per device: the IDs, `product`/`manufacturer`, and each
interface's `bInterfaceClass`.

- **Category** from the first interface class that matches: video (`0e`) → camera, audio (`01`) →
  music, mass storage (`08`) → storage, anything else → other. Video before audio, so a webcam
  with a mic in it is a camera.
- **An ID table overrides both category and name**, for devices the bus describes badly:
  `2ce3:3828` and `0329:2022` (both endoscopes, from `deploy/99-useeplus-camera.rules`) → camera,
  "Endoscope".
- **Never listed:** device class `09` (hubs, including the four root hubs) and `4c4a:4155` (the
  lav mic). The mic is ignored by ID, per his answer.
- **Name** is `product`, else the ID table, else `manufacturer`, else the bare `vid:pid` - tidied
  and capped to what the rail can hold.
- Devices come back in bus order, so the row on the panel never reshuffles.
- Off Linux there is no sysfs and the list is empty. That is what keeps the Mac, the suite and
  `panel_shot.py` honest, and it means nothing about this job renders or prompts differently
  there by accident.

### Showing

**A USB module in the top-left corner, built like the status pod.** Approved from mockups on
2026-09-20, kept beside this file in `factory/mockups/015-usb-*.png` - **look at them before
building this.** They are drawn images, so they are the target's *look* and never a measurement:
no pixel in them is a number. (In the empty one the image model also dropped the panel's
left-hand chassis rail; that is its carelessness, not the design.) A first pass as a tall boxed
panel was rejected for exactly the reason that matters here: *"all those rails between the display
and the edge of our screen is wasted space. the top-center panel uses its space efficiently, the
usb panel should do the same."*

So it is **a shallow strip flush into the top-left corner**, the pod's own depth - a chassis part,
not an overlay. Built out of the same `Bracket` / `_draw_rail` / `_draw_bolt` / `_pocket`
machinery the pod is, not out of a rounded rectangle, and following the pod in two details he
corrected by hand:

- **The green plate runs off the screen.** It bleeds to the very top edge and the very left edge
  with no steel and no border between it and them, exactly as the pod's plate runs off the top.
  The machined steel and its bolts are on the module's inner edges only - along its bottom and
  its chamfered right-hand end, where it meets the chassis.
- **`USB DEVICES` is engraved into the steel rail along its bottom**, cut into the metal in the
  same treatment and size as the `CYCLOPS` legend under the pod (`_pocket` / `_mark`). It is not
  a label on the green.

- **It grows sideways with the list and shrinks back**, exactly as the pod widens for its REC and
  HOT tags - as wide as its contents and not a pixel wider. Pre-baked per device count, the way
  `self.pods` is pre-baked per tag count, not laid out per frame.
- Devices run left to right along the strip: a standard glyph over the device name, in phosphor
  monospace at a size that reads from a pace.
- **Standard glyphs, his word:** a camera body with a round lens and a viewfinder bump for
  camera; an eighth note for music; a **floppy disk** for storage; the **USB trident** for other.
- **It is always there, and says so when it is empty** - shrunk to a stub reading `NO USB`.
- Drawn into the chassis layer, which is cached per window size, with the device entries as
  composited tiles. Nothing animates, so `test_only_two_things_move_while_he_is_asleep` passes.
- A device appearing **is** the connect feedback. No toast, no caption line, nothing to dismiss.
- At the pod's depth it costs a band of live picture along the top edge and nothing else. The
  reticle's empty middle (`_draw_framing` owns it) and the caption are untouched.
- `tools/panel_shot.py` gets `--usb camera:Endoscope,music:MiniLab 3` and `--usb ""` so every
  width can be rendered and looked at with no Pi.

The kiosk polls `devices.connected()` in the `_sync_*` row of `Kiosk.run()` (`kiosk.py:1847`) on a
timer, the way the temperature is polled - not per frame.

### Telling

A block from `build_instructions` (`agent.py:1273`), shaped like `_projects_block`: a header and
one line per device with its category. It goes after the manuals block and **before WHEN IT IS**,
so the clock stays last - the ordering comment at `agent.py:1288` says why that matters for the
greeting.

### Speaking

**Plugged in mid-session, Cyclops says so straight away** (his answer). The change arrives as a
`conversation.item.create` plus a `response.create`, and:

- It waits for a gap. Never while a response is in flight, never on top of a user turn - the
  change is held and spoken at the next quiet moment.
- It is debounced: a device has to be steadily there (or steadily gone) for a couple of seconds,
  and two announcements never land inside ten of each other. A flapping connector must not become
  a conversation.
- **Unplugging updates the panel and tells the model, but does not ask for a response.** Being
  told "snake cam's out" as you coil it up is a turn nobody wanted. If that is the wrong call,
  it is one line of `/rework`.

### The greeting

Two halves, per his answer - *"one more thing it can add, and simplify the existing instructions,
so every greeting is more of a surprise (not so prescriptive)"*.

1. **The device list becomes greeting material**, alongside the day, the hour, the gap and the
   count. `MOMENT_HEADER` (`session.py:798`) currently says "Four things make this moment and no
   other", and HOW YOU TALK says "The day, the hour, the gap and the count are what you have, and
   they are enough" (`agent.py:1008`). Both sentences have to change or the block will be ignored.
   The webcam being plugged in every single time is already covered by the rule right above it -
   *do not build the line out of a fact that is identical every time you say anything* - so
   nothing new is needed to stop it going stale.
2. **The greeting rules get cut down.** HOW YOU TALK's greeting section (`agent.py:988`-`1015`) is
   twenty-eight lines and mostly bans. Cut it to roughly half, keeping the bans that measurement
   shows are load-bearing and dropping the ones that are not. Job 014 already measured two of
   them: the banned-openers list and the do-not-hand-them-the-floor rule both hold up (16/16 and
   15/16). Anything that cannot be shown to be doing work goes. Measure before and after over the
   same number of wakes, and put both sets of transcripts in the outcome - the variety is
   checkable, but whether they are a *surprise* is his to read.

This is the delicate part of the job. Job 014's outcome is the warning: the greeting is about a
second from ready against a 1.67 s lid, and its wording is already tuned. Cut the block, do not
rewrite its voice, and let the numbers say which cuts survived.

### Not touched

Which camera `open_camera` picks; `preferred_source`'s choice of mic; the pod; the settings
screen; the udev rules. And **what a storage device is for** - a stick gets a name on the rail and
a line in the prompt, and nothing reads it.

## Done when

- [x] From a recorded sysfs tree: a webcam is `camera`, a MIDI keyboard is `music`, a mass-storage
      stick is `storage`, and a hub is not listed at all. — pytest over fixture trees
- [x] `2ce3:3828` comes back as `camera` named "Endoscope", not `other` with an empty name, and
      `4c4a:4155` is not listed. — pytest, the two IDs read off the Pi
- [x] Off Linux the list is empty, so no tag renders and no prompt block is added. — pytest on the Mac
- [ ] The module reads as a bolted-in chassis part, not an overlay: steel surround, bolts,
      chamfered end, the head rail running into it. — `panel_shot.py --usb` with four devices,
      beside the approved mockup, in the outcome
- [x] It is no deeper than the status pod. — the module's box and `pod_boxes[0].height`, both
      numbers said
- [x] The green plate touches the top and left edges: the pixels at (0, 0) belong to the plate,
      not to steel. — `panel_shot.py --usb`, the corner pixel read
- [x] `USB DEVICES` is cut into the steel along the module's bottom, not printed on the green. —
      the render beside the approved mockup, in the outcome
- [ ] It is as wide as its contents and no wider: the width at zero, one, two and four devices
      are four different numbers. — `panel_shot.py --usb` at each, the four widths said
- [x] With nothing plugged in the module is still there and says so. — `panel_shot.py --usb ""`
- [x] The four glyphs are the standard ones and are told apart at panel size: camera body,
      eighth note, floppy disk, USB trident. — one `panel_shot.py --usb` render with all four,
      in the outcome
- [x] Nothing outside the module moved: pod, eye, dials and caption are pixel-identical to
      master. — `panel_shot.py`, same args, difference confined to the module's box
- [x] A sleeping panel is still byte-identical frame to frame with four devices listed. —
      the existing `test_only_two_things_move_while_he_is_asleep`, extended
- [x] Render and composite cost do not move. — `panel_shot.py --bench`, both numbers before and after
- [x] The instructions carry a line per connected device with its category. — pytest on
      `build_instructions` with a faked device source
- [x] Reading the bus costs under 5 ms, so the greeting's margin behind the lid is unchanged. —
      pytest timing the call
- [x] A device that appears mid-session is announced once, never on top of a response or a user
      turn; a connector flapping five times in a second produces one announcement; an unplug
      produces none. — pytest, fake connection and clock
- [x] The greeting uses a plugged-in device when there is one to use. — **measured in round one,
      9 of 16. Do not re-run.** `factory/html/015/greetings_usb.txt`
- [x] The cut greeting block is materially shorter, and over the same number of wakes the lines
      are no less distinct, no opener is a banned one, and no more of them hand him the floor than
      master's did. — **measured in round one, and the "materially shorter" half of it came out at
      a fifth rather than a half. Do not re-run; it is his to accept or send back.**
      `factory/html/015/greetings_compared.txt`
- [x] **The live camera being unplugged does not kill the kiosk.** The endoscope is pulled while
      it is the camera on screen; the process survives, the picture goes, the panel stays up and
      the strip falls to `NO USB`. — pytest: the capture's close and the supervisor's reopen
      driven against a fake that raises and aborts the way libusb does, with no real device
- [x] **No steel in the very top-left corner.** The green plate runs clean into (0, 0) — the grey
      chassis nub still sitting over it is gone, and nothing else about the corner moved. —
      `panel_shot.py --usb`, the corner crop in the outcome
- [x] Nothing outside this job's own files is changed: `.claude/commands/` and `factory/` are back
      as master has them. — `git diff --stat master..HEAD`, in the outcome
- [ ] Plug and unplug the endoscope with no session running: the tag comes and goes within a
      couple of seconds, the lav mic never appears, and the panel is still there afterwards. —
      yours, on the Pi
- [ ] The row is readable from a pace away. — yours, on the Pi
- [ ] Wake it with the endoscope plugged in and it is about looking closely; wake it with a synth
      plugged in and it is about music. — yours, on the Pi
- [ ] Plug a synth in mid-session: it says so, in a gap, without talking over you. — yours, on the Pi
- [ ] Read ten greetings in a row. They are more of a surprise than master's ten. — yours, on the Pi
- [x] `uv run pytest` passes and is no slower than master. — `uv run pytest`, both numbers said
      (master is ~17 s on a Mac, per job 014 — that is not this job's to fix)

## Feedback — 2026-09-20

to fix this + remove that little bit of silver rail in the top-left corner of the screen and VERY
IMPORTANT: Don't let it do any live tests, do this quickly, so I can review again

**What "this" is.** `/try 15` put the branch on the Pi and it worked - the strip showed
`Endoscope` under the camera glyph, the lav mic stayed off it. He then pulled the endoscope out
and the whole kiosk died:

```
· camera useeplus stopped delivering - looking for it again
python3: ../../libusb/os/threads_posix.h:58: usbi_mutex_destroy: Assertion `pthread_mutex_destroy(mutex) == 0' failed.
```

libusb aborts the process when the endoscope is yanked mid-stream and the supervisor reopens it.
**This job did not cause it** - it never touched `camera.py` or `webcam.py`, and `devices.py`
reads sysfs and never opens libusb. It is a fault on master that this job made reachable, by
giving him a reason to pull the cable. It is in scope now because "the tag comes and goes" cannot
be judged while unplugging kills the panel. The panel was restarted by hand and is up.

**The silver rail.** At the very top-left of the screen a small grey chassis corner still sits
over the green plate - see the corner crop. The green is supposed to run clean into the corner,
as it does off the top edge. Take that nub off.

**No live tests this round, and be quick.** Last round spent most of forty minutes on real-model
wake probes and was killed mid-flight at `EXIT 137`. The greeting numbers are already measured and
are in `factory/html/015/greetings_compared.txt`; they stand. Do not run `talk_probe.py`, do not
open a realtime session, do not call a model at all. Nothing this round needs one.

## Built — 2026-09-20 (round two)

Both of the things you sent it back for are done. **The grey nub is off the corner:** the module's
plate was being cut back to the case's rounded corner, which left a bite out of it with the
surround's brightest corner — the one directly under the lamp — shining through the hole. Pixel
(0, 0) is plate now, (39, 42, 41), matching the top edge, and nothing else about the corner moved.

**Pulling the endoscope out no longer kills the kiosk.** Destroying a libusb handle whose device
has gone is an `assert()` in C: it raises SIGABRT and the process is gone, so there is no `except`
that catches it and no `finally` that runs after. The only cure is not to make the call, so an
endoscope that has left the bus is never handed back and its handle is left for the process exit —
the same trade `stop()` already makes for a reader stuck inside a read. Every other camera close
now goes through one guard as well, because a close that merely *raises* would take the
supervisor's reconnect loop down, which is the same outage more quietly. Three tests pin it, all
three fail without the fix.

Two things you should know. **The module is at its floor at one device as well as at none** — 126
px both — because the `USB DEVICES` pocket has to fit inside the module that carries it, and one
short name is narrower than the legend. That criterion is left unticked; shrinking under the
legend means a shorter legend or moving it, and that is yours to call. And **the corner holds about
150 px of columns**, which is two real product names or four short ones; past that, devices drop
off the rail in bus order. They are still in the instructions and Cyclops still knows about them.

No deploys, no sessions, no model calls this round, as asked. The greeting numbers from round one
stand untouched.

Hands-on: pull the endoscope while it is the camera on screen — that is the one thing a test
cannot settle, and it is what the round was for. Then the rest of the Pi list: plug and unplug with
no session running, the row from a pace, waking with the endoscope in versus a synth in, and a
synth going in mid-session.
factory/html/015.html
