# What he does with the screen

Three tools put something on the panel instead of saying it out loud, and one ledger records the
work they start.

## The scratchpad

Cyclops has a screen and he writes on it himself, without being asked. **The scratchpad** is what
both of you call it — say "put that on your scratchpad" and he knows what you mean.

`write_on_scratchpad` takes a small piece of HTML that **he writes**, and it is on the panel about
a second later: a torque figure set large while he talks you through the job, the steps of it as a
numbered list, a part number worth reading rather than hearing, a simple shape as inline SVG.
Colour, emoji and layout are his to choose. It costs nothing but the tokens to write it, which is
the whole point — this is the tool he reaches for by default, and [diagrams](#diagrams) are the
expensive exception for when the answer really is a drawing.

It lands in a document of its own — an `<iframe srcdoc>` on the same stage a photo lands on — so
his `<style>` cannot reach the dashboard behind it. **It is always a blank white screen**, the
whole 800×480, no bezel and nothing framing it. Plain markup with nothing styled comes out dark
on white and sized for the panel; colour is worth spending when it means something — a value in
red because it is out of range, wires drawn in the colours they actually are.

Scripts do not run and nothing loads from the network. One of those is about speed rather than
safety: the frame's `load` event waits on subresources, so a hallucinated
`<img src="https://…">` would stall the paint for as long as DNS takes.

**A press anywhere puts it away**, the way a photograph does and unlike a diagram, which keeps a
corner square. It gets the whole panel until then, covering his eye and the way out of the
session, so it is meant for things worth covering them for.

Nothing is written to the card — a scratchpad is a sentence he said with the screen instead of his
mouth, and `session.md` records it as one line naming what it said. `CYCLOPS_SCRATCHPAD=0`
withholds the tool entirely.

## Diagrams

Some answers are a picture. Ask *"how do I wire this relay to GPIO 17"* and Cyclops draws it on
the touchscreen — a wiring diagram in the idiom of a printed service manual, a pinout chart, a
block diagram, an assembly sketch. It takes about half a minute, fills the panel when it
lands, and keeps a small **Close** square in the corner rather than the full-width bar: a diagram
is a thing you point at while you talk about it, so pressing it must not put it away.

**The voice model does not draw it.** It calls `draw_diagram` with two things — one sentence
saying what to draw, with every value that matters in it, and a second saying what kind of
drawing it should be. That second one is not a menu: the model is asked to name the conventions
of whatever field the subject belongs to, because a diagram in the wrong idiom is read wrong. Ask
for wires and terminals on a piece of software and the picture will invent terminals that do not
exist. `gpt-image-2.5-sunburst` draws it at `quality="high"`, which takes about 26 seconds.

**It is drawn, not checked.** There is no schema and no validation behind the picture — nothing
distinguishes a wire drawn to the right pin from one drawn to the pin beside it. (Until
2026-09-04 a text model emitted a JSON scene that we validated and rendered with JointJS; it went
because the pictures were worse — colliding labels, boxes clipped off the edge of the panel.)
`draw_diagram` is told to say a connection out loud when getting it wrong would cost you a part,
and nothing in a generated picture is to scale or measurable, whatever it looks like.

**A diagram is a photo.** It lands in the session's `photos/` as `14-35-01_drawn.jpg`, exactly
where the shutter's `14-33-12_you.jpg` goes, and everything downstream already knew what to do
with it: the index service captions it, `session.md` embeds it, the Media grid lists it, and the
project sweep files it with the rest. There is no `diagrams/` folder and no second search path —
ask *"put that relay wiring back up"* and `recall` finds it by meaning, like any other picture.

Because it is a photo, `edit_photo` works on it too — *"drop the status LED"* edits the drawing in
front of you. Bear in mind that an edit redraws every pixel including the lettering, so for
anything whose labels matter, drawing it again is the safer move.

`CYCLOPS_DIAGRAMS=0` withholds `draw_diagram`.

## Imagining a change

Some answers are not a picture of connections but a picture of the thing itself, changed. Press
SNAP at the cabinet and ask *"what would those doors look like painted matt black?"* and a minute
later the photo you just took is back on the panel with the doors black and everything else where
it was. **Press it anywhere to put it away** — a drawing keeps its corner button because it is a
thing you read and point at while you talk; a picture of your own bench is a thing you look at and
are then done with.

`edit_photo` sends one picture to `gpt-image-2.5-sunburst` on OpenAI's image edits endpoint, along
with one sentence saying what to change. By default that is the newest picture in play — including a
previous edit, so *"now make it darker"* compounds the first change rather than starting over.

**Any picture from the session can be named instead.** Every picture Cyclops is shown arrives
carrying its own filename stem — `14-32-40_you`, `14-35-01_drawn` — and it hands that back to say
which one it means, so *"make the ball red"* three photos later edits the ball and not the desk.
The names are Cyclops' own: you never hear one, and it is told never to ask you for one. A name it
was never given is refused before anything is drawn, and it is handed the real ones to try again
with — a stem it invents costs a round trip, where an invented number would have been somebody
else's photo, redrawn, on the panel, half a minute later.

**What comes back is an illustration, and never evidence.** An edit with no mask redraws the
whole frame, so every pixel in the result is the model's, including the ones that look untouched.
Nothing in it is measured and nothing in it is a fact about your hardware. That is why colour,
finish, a part moved, a thing that is not there yet and *shown finished* are what it is for, and
why connections, orientation and assembly order are not: those are `draw_diagram`, which draws
the answer from a description instead of painting over your hardware.

**Cyclops is shown the result the moment it lands**, so it can tell you when the edit did not do
what you asked rather than leaving you to notice. It is told not to describe it back — you are
looking at the same picture — and told three times over, in the tool, in the result and in the
item the picture arrives in, that this is a drawing. The one time it will describe the picture is
when there was no free panel to put it on, because then you have nothing to look at.

The size comes from your photo, snapped to what the model will accept, so the edit keeps the
framing you were looking at: a crop would change the subject. It renders at `low` quality on
purpose — the panel is 800×480 and the picture is halved on the way there, so the setting is
mostly invisible and buys twenty seconds instead of sixty.

**Edits are kept like photos, because that is what they are on the card**: one jpg in the
session's `photos/`, named `14-33-05_edit.jpg` where a shutter press would be `14-32-40_you.jpg`.
So an edit is in `session.md` under *Imagined a change*, in the MEDIA tab, in the lightbox, and
eligible to be filed into a project like any other picture — the curator is told plainly that it
redrew an earlier photo, so it never files one believing it is a record of the bench.

`CYCLOPS_IMAGINE=0` withholds the tool entirely.

## Background tasks

A **task** is a piece of work that outlives the call that started it. Drawing a diagram takes
about half a minute; naming, summarising and filing a session after you tap stop takes the better
part of a minute. Both used to happen with nothing anywhere saying so.

```
~/.cache/cyclops/tasks.yaml
```

```yaml
tasks:
  - id: 7f3a91
    what: drawing the relay wiring…
    state: running
    started: 2026-09-05 12:41:07+0200
  - id: 2b0c44
    what: editing the picture to paint the doors matt black…
    state: failed
    started: 2026-09-05 12:38:02+0200
    ended: 2026-09-05 12:38:24+0200
    result: 'gpt-image-2.5-sunburst refused: content policy'
```

Newest first, the last twenty rows, and `cat` is the intended reader — that plus
`curl cyclops.local/api/status | jq .task` is the whole interface. `cyclops.tasks` owns the file
and any process or thread may write to it, which is the point: the agent, the kiosk, the admin
service and the detached child that tidies up after a session share no memory at all. Writes are
serialised with `flock` and land atomically through `cyclops.card`, so a reader never sees half a
file and a box that loses power mid-write has nothing stale to unwedge.

**It is a ledger, not a queue.** Nothing pulls work off it, nothing retries, nothing can be
cancelled. By the time a row appears the work is already running. Its only job is so you can see
whether anything is going on.

**Cyclops never opens a task himself.** He calls a tool, and the tool spawns the work and opens
the row. Three things do so today: `draw_diagram`, `edit_photo`, and the three jobs
`cyclops.after` runs once a session has ended.

**Where you see it.** The panel's terminal, along the bottom — the same line that says
"searching the web…", with its cursor blinking. It is the last thing that line falls back to,
behind the kiosk's own notices and the session's, so it is what fills the minute after a session
ends. And, on the admin page, one amber line at the right-hand end of the header, on every
screen, updated by the poll that was already running.

**A tool that starts a task answers straight away** — it says it is drawing and the model keeps
talking, rather than the call staying open for half a minute with nothing able to be said. When
the picture lands or fails Cyclops is told in a synthetic user turn, of the kind a photo arrives
in, and says so out loud. Nothing polls, nothing waits, and a session that ends mid-drawing
simply has nobody left to tell.
