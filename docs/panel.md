# The panel

`cyclops-kiosk` is the whole front-end and needs no browser: it opens the camera, draws the live
picture fullscreen, and lays a green terminal bezel over it — a status pod hanging off the middle
of the top edge (signal meter, `REC`, `HOT`, session clock), the picture and a four-arc reticle
through the middle, a terminal bolted across the bottom that says in plain English what he is
doing, and two corner mounts carrying three round controls sized for a thumb in a glove:

| Control | What it does |
| --- | --- |
| **the eye**, bottom left | Cyclops himself, half-sunk into the big mount. No label — he is a face, and you tap a face to ask what it remembers: the [admin page](admin-page.md) fullscreen on the panel, on the sessions list. **Hold him** and you get the power menu instead |
| **the knob**, bottom right | the output volume. Press it and drag, and a column comes up the right-hand side: the level is where your finger is, and the number stands above the track where your hand is not. Nothing reaches the speaker until you let go, so a tap on the knob is just a tap. It writes the same note the admin page's slider does, so the two always agree |
| **the gauge**, in the corner | the board's temperature, on the Pi 5's own scale — green, amber from where the clock starts being capped, red past it. Tap it and the admin page opens on **SYSTEM** |

…and one thing down there that is not a control at all:

| Readout | What it says |
| --- | --- |
| **the pilot lamp**, on the plate below the two dials | a brass fitting with a glass dome in it, turned and graduated like the collar round his eye. **Dark asleep** — so the lamp coming on is itself the news that something is happening. **Amber and blinking** while a session is coming up or going down, **bright phosphor green and steady** for as long as one is live, **blue** for a job running in the background with nobody in a conversation, and **red and dead still** on a fault. It is an indicator and nothing else: pressing it does what pressing the plate has always done, which is interrupt him |

A live session is one look rather than five. Listening, speaking, looking at a photograph,
searching and drawing all burn the same green, because what the lamp answers is *is he up* — the
eye says what kind of up in far more detail than a lamp could, and the terminal spells it out
underneath. The fault is the one lit state that does not move, which is the rule the whole panel
keeps: nothing on this screen blinks at you about something broken twice.

Taking a photo and starting a session are the button beside the panel, not the glass: a tap is
the photo, a hold is the conversation.

## Holding his eye shuts the box down

Press and hold him for 0.7 s and a power menu comes up over the picture: **SHUT DOWN**,
**RESTART**, **CANCEL**, with anywhere off the card being a cancel too. It is behind a hold
because it is the one control here you cannot take back. While you hold, the rule that arcs over
his head fills in from the left; when it reaches the far side the menu is open, and the panel
blips, because the menu appears under the very finger covering it. The card sits above his face
rather than over it and says *the whole box, not the session*, because **GO TO SLEEP** on the tab
beside it means the other thing.

Choosing does not cut the power there and then. The panel says `Shutting down…`, the kiosk tears
itself down exactly as `q` would — the session is finished, the video muxed, the folder named and
the camera released — and only then does `systemctl poweroff` (or `reboot`) run. So the box goes
down the way it would have over ssh, which is the whole point: an unplugged Pi loses the
conversation it was in the middle of. It goes through `sudo -n systemctl` first and bare
`systemctl` after, so it works whether the kiosk was started by the desktop at boot or by
`deploy/push.sh` over ssh.

## The border

It runs along the panel's own edge, carries the state in its colour and glows inwards — **green
asleep, amber waking, white awake, red on a fault** — so the state reads from across the room,
and while he is up it breathes. A colour rather than a brightness, because dim green and bright
green are the same thing a pace away. The white is the tube's own (`#E1FFF0`), a single-phosphor
screen blooming towards white with its colour still in it, so it reads as the same screen turned
up rather than a second hue arguing with the first.

The same accent goes on the handful of things that only mean something during a session: the
signal meter, the session clock, the caption's `›`, the WAKE UP / GO TO SLEEP button's mic and
word, and him. Everything else — the brand, the rules, the ticks, SNAP, the caption's own words —
stays phosphor green whatever he is doing, because a panel where everything is an accent has
none.

Asleep the whole screen is green and completely still, with one exception: the **WAKE UP** cell
breathes slowly, *towards the white* — so it is both the only thing moving and the only thing not
green, and it wears the colour the whole screen turns when you press it. The status line, bottom
right, spells it out for anyone close enough to read, including what an error actually said.

## The eye

**The middle of the row is Cyclops.** It is the one thing on the panel that is a face rather than
a readout, and it is what lets the rest stay this terse: a glance answers *is he there, and what
is he up to*. He is the boot mark brought to life — the splash's diaphragm inside concentric HUD
rings, keeping that mark's vocabulary of castellated ring, dotted ring, part-way knurl, stator
vanes and eight-leaf hatched diaphragm. The machine is deliberately dim and the middle of him is
not: the core is a disc of radial filaments converging on a hot centre, which is what makes him
look back at you rather than merely spin.

**He looks around.** The whole optic — blades, bezel and core — slides inside a shell that never
moves, so it reads as an eye in a socket rather than a picture being panned. The shell staying
put sells it, and so does the brow, because a highlight on the outer glass does not travel with
what is underneath it.

**And he looks at things, not around.** A mood names the *places* he attends to — out of the
glass at you, the middle of the picture, his own caption, the readouts, the bench, or nothing at
all — and the first is his anchor: where he rests, and what a glance comes back to. Attention
runs in windows, and a walk decides whether each is spent on the anchor or as a peek elsewhere.
Runs of anchor windows merge into one hold, so the dwell varies without a dwell setting, and
under every fixation there is a tremor of about a pixel, because an eye that stops moving is a
dead one.

Everything he does is a **mood** — a dozen-odd numbers, a list of places and a colour, one row
per state in `overlay.MOODS`, with `cyclops/eye.py` as the mechanism underneath:

* **Asleep** — a narrow ember in dim green, turning slowly and floating, never darting, because a
  sleeping face that flicks about is a dreaming one.
* **Waking** — amber, half open, rings running fast with a scanning arc, checking his own
  instruments as he comes round.
* **Listening** — calm and attending, blinking every few seconds and never quite on the beat, his
  iris opening to your voice; he holds your eye for five to twelve seconds at a stretch, takes a
  second's peek at the picture he is sitting on, and comes straight back.
* **Speaking** — breathing faster and wider, holding your eye harder, glancing at his own caption.
* **Looking at a photo** — staring at the middle of the picture and never leaving it.
* **Searching** — narrowed, rings tearing round, a bright trace sweeping a dimmed rim, place to
  place twice a second and hardly ever home: looking *for* something rather than *at* it.
* **Drawing** — head down, looking up at you now and then.
* **A fault** — red, and not moving at all, gaze included.

He crossfades between them rather than snapping, because a face that jumped colour between two
frames would read as a different creature. Most of the variety is free rather than tuned: the
core can only be as big as the hole the blades leave, and that hole grows as the iris opens — so
a stare is a wide hot core and a fault is a narrow one, off the one number every mood already
sets.

These are meant to be pushed around rather than argued about:

```bash
uv run python tools/eye_sheet.py --seconds 8 --level 1.0   # every mood, over a strip of time
uv run python tools/eye_sheet.py --state searching --frames 16 --scale 2
```

## The line along the bottom

It is a terminal rather than a speech bubble: a small monochrome monitor standing in the middle
of the bottom bay, with the caption printed on its glass. It is there whether or not there is
anything to say, which is the whole difference — a bubble with no words in it is a bug, a
terminal with a blank screen is a terminal. A phrase ending in `…` is work in flight and gets a
blinking `_` where the ellipsis was; anything else is a state and holds still.

The front is one surface, not a case with a window cut in it: opaque and near-black hard against
the outside, easing over the moulding's width into glass you can see the room through, blank at
the edge and coming up green towards the middle so it reads slightly bulged. A white lamp up and
to the left throws a wipe across the whole of it, moulding included — that continuity is most of
what says "glass" — and leaves a rim highlight that follows the case round rather than stopping
on a straight row. Every letter blooms into its own skirt, which is the one part of the terminal
that is not baked into the cached chrome.

It stands clear of both mounts and off the panel's bottom edge, held at either end by a clamp: a
strap standing on the case's edge, an arm back to the mount, bolted top and bottom. The clamps
are machined from the same rail the mounts are — the same extrusion drawn by the same code, in a
thinner section — which is why they read as part of the frame rather than as shapes beside it.

While he is asleep **nothing on the panel moves at all** — not the eye, not the border, not the
caption. Two frames of a resting panel are identical, and that stillness is what makes any of the
rest read as awake. The picture keeps its own colours and is not filtered at all; the strip and
the tab row are, and they are see-through rather than opaque, so the camera runs edge to edge
behind the chrome as well as between it. Only the chrome is green, because the point of the panel
is still to see the room.

## Developing against it

`q` or `ESC` quits, `f` toggles fullscreen, and `--windowed` / `--size=WxH` are there for working
on a laptop. Set an initial speaker level with `CYCLOPS_VOLUME`; after that the volume — and the
INTERRUPT switch — lives on the [admin page](admin-page.md), behind his eye.
