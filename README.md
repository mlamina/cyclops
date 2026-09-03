# cyclops

A low-latency speech-to-speech agent built on the **OpenAI Realtime API** (WebSocket, `openai` Python SDK 3.x).
It listens on your microphone, answers through your speakers, and **you** decide when it looks:
press the shutter, and the photo goes straight into its context so it answers about what you are
holding. It never takes a picture on its own.

## How it works

```
mic ──PCM16 24kHz──▶ input_audio_buffer.append ──▶ OpenAI Realtime (server VAD)
                                                        │
speakers ◀──PCM16 24kHz── response.output_audio.delta ◀─┤
                                                        │
SNAP button ──▶ webcam ──▶ JPEG (≤1024px) ──▶ input_image item ──▶ response.create
```

* One WebSocket session, server-side voice activity detection, no push-to-talk.
* Barge-in: when you start talking the local playback buffer is flushed and the assistant's
  audio item is truncated to what you actually heard, so the model doesn't think you heard the rest.
  Over an open speaker that judgement is a guess about a room and can be wrong the worst way —
  Cyclops hearing its own voice as you — so the panel's **SYSTEM** screen carries an INTERRUPT
  switch that turns it off: the mic then stays shut until Cyclops has finished.
* The shutter shrinks the frame to 1024 px and injects it as an `input_image`, then asks for a
  response, so the model's next reply is about what it just saw. On the panel it borrows a frame
  from the preview already on screen; from the CLI it opens the camera in a worker thread and
  discards warm-up frames. Every capture is also written into the session's own folder (see
  **Sessions** below).

## Setup

Requirements: macOS (Apple Silicon) or Linux / Raspberry Pi, Python 3.12+ (uv fetches it),
[uv](https://docs.astral.sh/uv/), PortAudio (`brew install portaudio` on macOS,
`sudo apt install libportaudio2` on Debian/Pi), a webcam, and an OpenAI API key. See the
**Raspberry Pi + touchscreen** section below for the Pi setup.

```bash
cp .env.example .env     # then paste your OPENAI_API_KEY
uv sync
```

## Run

```bash
uv run cyclops
```

Talk. Press `Enter` to take a photo and show it to Cyclops. Press `Ctrl+C` to quit.

Headless smoke test (no mic/speakers; sends text turns and hands the model a photo):

```bash
uv run cyclops-smoke
```

## Sessions

Every session — from `cyclops` or `cyclops-kiosk` — gets **one folder of its own**,
holding everything it produced. Pull the SD card into a laptop and it reads like a GoPro's:

```
sessions/
  2026-08-26_14-32-05_lego-falcon/
    session.md        the conversation, with the photos in it, for you
    session.jsonl     the same events, one JSON object per line, for a program
    summary.md        one sentence and one paragraph: what this session was
    project.md        which project this got filed under, or why it didn't
    video.mp4         the recording - the panel, or the camera (kiosk only)
    photos/
      14-33-12_you.jpg
      14-33-05_edit.jpg      one of those redrawn with a change - see Imagining a change
      14-32-40_cyclops.jpg   (only on older cards - Cyclops used to hold its own shutter)
    diagrams/
      14-35-01_relay-driven-from-gpio.svg    the picture, embedded in session.md
      14-35-01_relay-driven-from-gpio.json   the spec, so it can be drawn again
```

`session.md` opens in any markdown viewer with both images inline and every line stamped as an
offset from the start, so you can scrub straight to it in `video.mp4`. `session.jsonl` has the
same thing for machines: who said what and when, which photos were taken and by whom, what was
searched, and which of Cyclops's turns you interrupted. The session's UUID lives inside both
files — never in a name, because nobody can read a UUID.

**The name and the summary.** When a session ends, `gpt-5.4-nano` reads the whole transcript
once and answers three things at a stroke: two to four words to rename the folder with
(`2026-08-26_14-32-05` becomes `2026-08-26_14-32-05_lego-falcon`), one sentence saying what the
session was, and one paragraph of what actually happened — what was decided, what was measured,
and what was left open. The sentence and the paragraph become `summary.md`. One call for all
three, so the name and the summary can never disagree about the same conversation.

With no network, no key, or `CYCLOPS_SLUG=0`, the folder keeps its date-time name and gets no
summary. The call is capped at ten seconds, runs alongside the video mux, and never blocks a
shutdown; the three answers are read out of the reply independently, so a mangled one costs you
one of them rather than all three.

**Continuity.** Every new session starts knowing where the last one got to. Cyclops's
instructions gain a short *Where you left off* section built from the cards already written:
**the previous session's paragraph in full, plus the one-sentence titles of the last five
sessions**, oldest first. It is told not to recite it back at you and not to assume today is a
continuation — you say what today is. It is also told that this is *all* it remembers, so it
says so plainly rather than inventing something when you ask about a month ago.

Only folders that already have a `summary.md` count, which is what keeps a session out of its
own recap. The line printed when a session connects says what was handed over:

```
· continuity: last session yesterday (22m 14s) + 4 earlier headlines
```

**The video.** H.264 with a **stereo** audio track, **you on the left channel and Cyclops on the
right**. The two voices are never mixed, so you can listen to either side alone. Only the kiosk
records, and now for two reasons: `cyclops` opens the camera per photo instead of holding it open,
so there is no continuous video for it to record — and it has no panel, which is the other thing
there is to record.

**Which of the two is a switch on the settings screen** — tap the eye, and it sits under
INTERRUPT. It settles what the *next* session records; an encoder is opened once, at one frame
size, so a recording already running cannot be handed something else halfway through.

| | what lands in `video.mp4` |
| --- | --- |
| **SCREEN** *(the default)* | The panel, 1:1 at 800×480 — the sharpened preview with the halo, the timer, the caption, the REC tag and the tab row composited on it, cropped to the panel's 5:3. Watching it back is watching the session happen: you can see when Cyclops was thinking, when you cut in, and where the shutter went off. |
| **CAMERA** | The sensor alone, at its own resolution and its own shape, nothing drawn over it and nothing trimmed off the sides. No pixel is spent on chrome. The one to reach for when the recording is evidence rather than a memory — a part number, a wiring colour, a serial you will squint at later. |

`CYCLOPS_RECORD_SOURCE` decides it on a box where nobody has ever touched the switch. The photos
are unaffected either way: SNAP, the agent's look tool and everything in `photos/` come off the
raw camera. Nothing on the box is mirrored — not the panel, not either recording, not the
photos — so text held up to the lens reads the right way round wherever it turns up later.

Two smaller consequences of recording the screen. A session started with **no camera** now still
produces a video — the panel saying `No camera found`, with the chrome and the full stereo audio —
where before there was no frame to record and the recorder declined. And while a diagram or the
admin page is covering the panel the recording goes **black**, because the screen is genuinely no
longer the kiosk's to hand over; the audio carries on throughout.

Needs `ffmpeg` on `PATH` (`brew install ffmpeg`, or `apt install ffmpeg` on the Pi). Without it
the session runs exactly as before and says once that it isn't recording — nothing here ever
blocks a conversation, recording and logging included.

**What each costs.** Measured on a Pi 5 at 15 fps, against the ~93% of one core the panel already
spends drawing itself at 25 fps whether anything is recording or not:

| | frame | over drawing alone | on the card |
| --- | --- | --- | --- |
| **SCREEN** | 800×480 | **+18%** of one core | **2.5 MB/min** |
| **CAMERA** | the sensor's own — 1280×720 on a C920 | **+35%** of one core | **8.5 MB/min** |

Two thirds of each figure is the kiosk handing frames over rather than ffmpeg taking them, which
is why the bigger frame costs the more: it is a memcpy per frame, and 720p is 2.4× the bytes.
`CYCLOPS_RECORD_WIDTH` caps it if the card matters more than the detail does; 640 was the old
default, and it costs you a resample that softens the chrome. Nothing is ever pruned; delete
what you don't want.

**If a session is cut off** — power loss, a killed process — nothing is lost and nothing has to
be done by hand. The Pi finishes it on the next boot, and the two rules that make that work are
worth knowing.

*Every file lands whole or not at all.* Everything written into a session folder goes through
`cyclops.card`: to a scratch name, fsynced, then renamed into place. A power cut leaves the
previous state or nothing — never a half-written file. This was not always true, and the failure
was nasty: `Path.write_text` creates the inode immediately and lets ext4 write the data back at
its leisure, so a cut ten seconds after teardown left a **zero-byte `session.md`** wearing the
name that means "this session finished". Every check in the codebase asked `is_file()`, so the
damage made itself invisible to the repair path that existed to fix it.

*Presence is not existence.* "Finished" now means `session.md` **has bytes in it**. So a
zero-byte page is something to repair rather than something to skip.

An interrupted session keeps a `parts/` with a playable `video-raw.mp4` and the two WAVs, and a
`session.jsonl` that ends wherever the power did — `read_log` drops the half-written last line
and the page says so. `cyclops-sessions` lists it as `UNFINISHED`, or `EMPTY` if nothing at all
survived.

```bash
uv run cyclops-sessions                  # what's on the card
uv run cyclops-sessions --fix            # finish anything half-done (offline, no key)
uv run cyclops-sessions --fix --name     # ...and name/summarise what the live call missed
uv run cyclops-sessions --recover        # all of the above over the whole card, then file it
uv run cyclops-sessions --recover --dry-run   # ...say what that would do, touch nothing
```

`--recover` is what `cyclops-recover.service` runs at boot, and it is the two verbs above plus
the two things nobody should get by accident: a folder **nothing** survived in (no records, no
video, no photos, no `parts/`) is deleted, and what was repaired is handed to the projects sweep.
Anything with any salvage at all is repaired as far as it goes and kept. A folder holding a file
cyclops did not write is never deleted, whatever else is true of it. A folder a session is
writing into right now is skipped entirely — a live session holds a `flock` on its own log, which
is how a boot sweep can tell, and renaming one out from under a conversation is the one mistake
here that would actually cost you something.

Install it once, on the Pi:

```bash
sudo deploy/install-recover.sh
journalctl -u cyclops-recover -b     # what it found and what it did
```

Photos taken with no session running land in `captures/`, which keeps only the last 20 alongside
`latest.jpg`. Nothing sees those — with no session there is no model to show them to — and the
session page says so. A session's own photos are never pruned.

## What it knows about you

Continuity is what *happened*; projects are what is being *built*. Neither of them is about the
person on the other side of the bench, and without that a machine that has talked to you for a
month still opens on a stranger — no name, no idea what you do, no memory that you said last
week you wanted to get better at dovetails.

So there is a third kind of memory, and it is one file at the root of the card:

```markdown
# About you

- Goes by Priya.
- Restores furniture at weekends; ten years at it.
- Has a small garage shop: bandsaw, thicknesser, no lathe.
- Wants to get better at hand-cut dovetails.
- Deaf in the left ear, so say things once, clearly.
```

That is the whole format, and it is a file rather than a database for the same reason
`Project Data.xlsx` is a spreadsheet: **if a line is wrong you open it and fix it.** Delete a
line and it stays deleted unless Cyclops hears the same thing again; delete the file and it
starts over knowing nobody. The prose at the top is written by the code and is never read back,
so what you edit is only ever the list.

**A line earns its place by changing how Cyclops talks to you** — your name, what you do and how
practised you are at this kind of work, what tools and machines and software you have, what
kind of work you want to get better at, and how you want to be spoken to: which language, which units,
anything about hearing or sight or handedness that changes what help looks like. It is told just
as firmly what does not belong: anything about *today* rather than about *you* (the bore is
12.7 mm, and that goes in the sheet), anything you did not say about yourself, and anything
private you did not offer as a standing fact — health, money, who you live with, where you live.
Held up against a session that mentioned chemotherapy, an address and a tight month, it wrote
down two lines: *works as a nurse, not an experienced builder* and *has a hammer drill and a
spirit level*.

**The list is rewritten, not added to.** When a session ends, a model gets the list as it stands
plus the transcript and answers with the whole list back, so *uses Fusion 360* becomes *uses
OnShape* in place rather than sitting on the card next to its own contradiction — which is what
an append-only list gives you after a month. It is a second call beside the naming one and goes
out at the same moment, so the two share one twelve-second budget and a shutdown costs what it
always did.

It is `gpt-5.6-terra` rather than the `gpt-5.4-nano` that names the session, and that is the one
place these two calls genuinely differ. Naming is a transcription — read a conversation, say what
it was. This is a *reconciliation*: hold a list of sentences about a person against a new
conversation and work out which of them it has just made untrue. Handed *uses Fusion 360* and a
conversation about moving to OnShape, five times over, the small model got it clean four times
and once left a spare line about sketching being quicker — a sentence about a week, in a list
that is meant to be about a person. It is not even the cheaper trade it looks: over the 32 real
sessions on the card the small model ran a median 1.4 s against 1.0 s, and a slowest 12.3 s
against 5.3 s.

Rewriting has one sharp edge, and the whole of `cyclops.about` is built around it: **an empty
answer never empties the file.** No key, no network, a refusal, a timeout, a reply in the wrong
shape, or a model that decided to say nothing — all of them mean *leave the list alone*, never
*forget everything*. The only things that can shorten it are a well-formed shorter reply and
you, with an editor.

Most conversations teach it nothing about you, and that is the normal case: an identical answer
is not written at all.

**What it does when it knows nothing.** An empty list is not silence. Cyclops is told outright
that nothing has ever been written down about you, and to ask — *one* open question, once,
somewhere in the session, never as the opening line and never while you are mid-cut or waiting
on an answer. If you brush it off it drops the subject for the day. That is the same "offer
once" rule the project question already lives under, and for the same reason: the fastest way to
ruin this would be a box that interviews someone who came in to fix a tap.

What was handed over is printed when a session connects, beside the continuity line:

```
· about you: 6 thing(s) known
· about you: nothing yet - it will ask once
```

The list itself is not on the admin page. It was, above the session list, and it earned its
place there for about a day: the screen you open to find a recording is not the screen you open
to audit a memory, and a block standing over every visit to the first in order to serve the rare
second is the wrong trade on a 480 px panel. The file is the interface — `about-you.md`, in any
editor, which is also where you correct it.

`CYCLOPS_REMEMBER=0` turns the whole thing off: no call at the end of a session, and not a word
about you in the instructions at the start of one.

## Projects

A session is one conversation. A **project** is the thing you keep coming back to, and it gets a
folder of its own that a person can read without knowing anything about any of this:

```
projects/
  Pelican Display Mount/
    README.md            what it is, where it stands, what's still open, what was decided
    Log.md               one dated entry per session, oldest first, never rewritten
    Project Data.xlsx    the numbers: torques, sizes, part numbers, codes
    Photos/
      2026-08-26_16-48-48_cyclops.jpg
```

**A project starts because you said so.** Cyclops is told the names of the projects it is keeping
at the start of every session. When you're plainly working on something that isn't one of them,
and it looks like a thing you'll come back to, it asks — once, in one sentence — whether to keep
notes on it. Say yes and the folder appears there and then. Nothing else can create one, which is
what stops the card filling up with "Pelican Case", "Pelican Display" and "Pelican Mount" as a
model changes its mind about what today was.

It's told the names and nothing else. When you come back to something it calls `open_project` to
read what was decided last time — so twenty projects cost twenty lines of its attention, not
twenty pages.

**The filing happens afterwards, on its own.** When a session ends, a detached `cyclops-projects
--sweep` starts and the kiosk forgets about it: an orchestrator on
[Pydantic AI](https://ai.pydantic.dev) reads the session and delegates to four smaller agents —
one reads the transcript, one decides which project it advanced, one writes the log entry and the
new page, one picks a few photos worth keeping. It never invents a project: it either files under
one that exists or writes down that it filed nothing.

`Log.md` is the truth and `README.md` is a picture built from it, so you can delete a README and
the next sweep makes it again. Nothing on the card is a database; the machine-readable part is
the YAML block at the top of each README, and one invisible HTML comment per log entry holding
the session's UUID — which is what stops an entry being written twice.

Nothing is filed twice, and nothing is lost. `project.md` in the session folder is the receipt: if
it's there the session has been read, and if it isn't, the next sweep reads it. So a power cut
costs a re-read and nothing else, and it's safe to run over a whole card as often as you like.

```bash
uv run cyclops-projects              # what's on the card, and how many sessions are waiting
uv run cyclops-projects --check      # offline: pages that drifted from their log, torn entries
uv run cyclops-projects --sweep      # file everything unfiled, oldest first (needs a key)
uv run cyclops-projects --sweep --dry-run   # say what it would write, write nothing
uv run cyclops-projects --sweep --again  # re-read everything; the log ledger prevents duplicates
```

Oldest first is not cosmetic: a project's page is rewritten against what the session before it
left behind. `cyclops-sessions` flags anything still `unfiled`.

### The numbers

The prose says what was decided. **`Project Data.xlsx`** says the bore is 12.7 mm. Numbers want
different storage from sentences — a torque that goes through a summariser comes back as "around
25" — so they live in a sheet, one row each, filed under tabs Cyclops picks: *Torque specs*,
*Dimensions*, *Paint*. Three columns, **Key | Value | Note**, and the value is stored exactly as
it was spoken or read off the plate, unit included. Nothing here is a database either: it's a
spreadsheet, and if a value is wrong you open it and fix it.

Hold a spec plate up, press SNAP, and every number on it goes down in one write. Ask
*"what was the caliper torque?"* three weeks later and it looks the value up rather than
remembering it — the difference being that looking it up can come back empty, and it will tell
you so instead of producing a plausible number.

**Values are never loaded into the conversation.** All Cyclops is told when it opens a project is
the tab names and how many rows are in each — one line — and every value after that is a tool
call. So a project with two hundred numbers in it costs the same attention as one with two, which
is the entire reason this is a file and not part of the notes. Retrieval is a plain word-overlap
score with a spelling fallback, so "caliper torq" through a microphone still finds *Caliper bolt
torque*; there is no index to rebuild and no embedding to go stale.

It only ever writes into a project you already track, and it will ask which one rather than
guess. Deleting is the one thing it refuses to do on a guess: if more than one value could be the
one you meant, it deletes nothing and asks.

Every session end costs one model call even when nothing gets filed, because deciding that *is*
the call. At these model sizes it rounds to nothing, but `CYCLOPS_PROJECTS=0` turns the whole
feature off — no sweep, and the two project tools aren't offered to the voice agent at all.

## Diagrams

Some answers are a picture. Ask *"how do I wire this relay to GPIO 17"* and Cyclops draws it on
the touchscreen — a wiring diagram with named pins and routed wires, a pinout, a block diagram, a
flow or a state machine. It appears a few seconds later, fills the panel, and has the same
full-width **Close** bar as the system page.

**The voice model does not draw it.** It calls `draw_diagram` with one sentence, and that sentence
goes to a text model which returns a small JSON description — nodes, ports and wires. The panel
translates that into shapes with [JointJS](https://www.jointjs.com). The model never writes
JointJS and never writes coordinates for anything that does not need them: it says
`{"type": "resistor", "label": "1k"}` and the zigzag lives in the template. That split is the
whole reliability story, because it moves correctness onto a schema we validate ourselves in a
millisecond — and a diagram that fails validation is redrawn once, with the complaint attached.

It draws boxes and wires, so it is good at connections and useless at shapes. A cutting list, a
joinery detail or anything to scale is not something it can express, and it is told to say so
rather than draw a wrong picture.

**Diagrams are kept, like photos.** Each one lands in the session folder as a pair — the `.json`
Cyclops re-reads and the `.svg` you open — and when the session is filed, *every* diagram is
copied into the project:

```
projects/Pelican Display Mount/
  Diagrams/
    2026-08-29_19-40-12_relay-driven-from-gpio.svg    the picture, in the log entry
    2026-08-29_19-40-12_relay-driven-from-gpio.json   the spec, so it can be put back up
```

Unlike photos there is no curator and no limit: a photo is a frame caught in passing and three of
forty are worth keeping, but a diagram was asked for out loud and looked at. The svg carries its
own dark background, so it opens the way it looked on the panel rather than as invisible green on
white.

Ask for one again — *"put that relay wiring back up"* — and `find_diagram` matches on the title
and re-renders the stored spec. That is instant and costs no model call, which is also why it is
preferred to drawing it twice: a redraw would come back subtly different.

The three vendored bundles under `src/cyclops/admin/static/` are served from the Pi, never a CDN
(see `NOTICE.md` there). Nothing about the panel needs the internet.

## Imagining a change

Some answers are not a picture of connections but a picture of the thing itself, changed. Press
SNAP at the cabinet and ask *"what would those doors look like painted matt black?"* and a minute
later the photo you just took is back on the panel with the doors black and everything else where
it was. **Press it anywhere to put it away.** A drawing keeps its corner button because it is a
thing you read and point at while you talk about it; a picture of your own bench is a thing you
look at and are then done with, so the whole panel is the way out rather than a square in the
corner of it.

`edit_photo` sends the last photo *you* were shown holding to `gpt-image-2` on OpenAI's image
edits endpoint, along with one sentence saying what to change. It always works on the last real
photo and never on a previous edit, so a second change — *"now make it darker"* — starts from
what the camera actually saw rather than compounding the first.

**What comes back is an illustration, and never evidence.** An edit with no mask redraws the
whole frame, so every pixel in the result is the model's, including the ones that look untouched.
Nothing in it is measured and nothing in it is a fact about your hardware. That is why colour,
finish, a part moved, a thing that is not there yet and *shown finished* are what it is for, and
why connections, orientation and the order to assemble something are not: those are
`draw_diagram`, whose output is checked against a schema we own before it is drawn.

**Cyclops is shown the result the moment it lands**, so it can tell you when the edit did not do
what you asked rather than leaving you to notice. It is told not to describe it back — you are
looking at the same picture — so it stays quiet unless there is something to say. It is told
three times over, in the tool, in the result and in the item the picture arrives in, that what it
is looking at is a drawing: never a measurement, and never a fact about your hardware. The one
time it will describe the picture is when there was no free panel to put it on, because then you
have nothing to look at.

The size is taken from your photo rather than from a menu, snapped to what the model will accept,
so the edit keeps the framing you were looking at — a crop would change the subject, which is the
one thing an edit of your own photo must not do. It renders at `low` quality on purpose: the
panel is 800×480 and the picture is halved on the way there, so what that setting costs is mostly
invisible and what it buys is twenty seconds instead of sixty.

**Edits are kept like photos, because that is what they are on the card**: one jpg in the
session's own `photos/`, named `14-33-05_edit.jpg` where a shutter press would be
`14-32-40_you.jpg`. So an edit is in `session.md` under *Imagined a change*, in the MEDIA tab, in
the lightbox, and eligible to be filed into a project like any other picture — the curator is
told plainly that it redrew an earlier photo, so it never files one believing it is a record of
the bench.

While it is up the panel belongs to it, which means a `screen` recording goes black for as long
as you leave it there — the same as a diagram, and for the same reason. `CYCLOPS_IMAGINE=0`
withholds the tool entirely.

## Admin page

`cyclops-admin` serves a small Django page on **port 80**, so from anywhere on your network you
can open `http://cyclops.local/` and see both how the box is doing and what it has recorded. It
is read-only, has no database and no login — a private-LAN dashboard, not an exposed service.

Four screens, picked by the tabs in the header and by the hash in the address bar, so nothing
ever navigates (the kiosk's browser is kept warm on this page and a reload would be felt). The
kiosk opens it on **SESSIONS**, which is what its tab promises:

| | |
|---|---|
| **SYSTEM** `#/` | **CPU temperature**, **memory**, **disk**, and **how many sessions have been recorded**. A session counts as finished once it has written its `session.md`; anything else shows as in progress. |
| **SESSIONS** `#/sessions` | Every session, newest first, each with a still lifted straight out of its own recording. Open one to watch it. |
| **PROJECTS** `#/projects` | Every project in `projects/`, most recently worked on first. Open one for a plain file browser over its folder; open a file to read it. |
| **MEDIA** `#/media` | Every photo and every drawing on the card as one stream, newest first. Tap one to fill the screen and flip through with the arrows, the arrow keys, or the columns down either side. |

**The session view changes shape with the screen.** On a laptop it is the recording on the left
and the whole conversation on the right, with the photos and drawings sitting inline where they
were taken; every line is stamped with its offset, and clicking one seeks the video to that
moment — the transcript's `t` *is* the recording's timeline. On the 7" panel and on a phone it is
the video and what it is called, and nothing else: at a bench you are not there to read. The
narrow layout does not hide the transcript, it never asks the server for it.

**A project opens as its folder.** `#/p/<project>/<folder>` browses, `#/f/<project>/<file>`
reads, and a bar across the top says where you are and is the way back out. `README.md` and
`Log.md` are rendered as markdown, with the pictures they link to shown inline — the paths are
rewritten server-side, which is why a relative `![](Photos/x.jpg)` written by the sweep resolves
at all. `Project Data.xlsx` is shown as what it is: the remembered pairs, under the tab name they
were written on, rather than a spreadsheet grid nobody wants on a 7" panel. Pictures, drawings
and recordings open in place; anything else says what it weighs and leaves it at that.

The markdown is turned into HTML on the server, where the project folder is known. The
frontmatter is dropped (it is the code's half of the file, not the model's), the
`<!-- cyclops:session … -->` terminators go with it, and everything else that could be a tag is
escaped — so the only tags the page is handed are ones the renderer wrote. Project files are
served from `/project-media/<project>/<file>`, by the same suffix allow-list plus `.png`, and
every path is resolved before it is compared against the project folder, so `..`, an absolute
name and a symlink out of the tree all fail the same check.

Recordings, photos and drawings are served from `/media/<session>/<file>` with byte ranges, which
is what lets a video seek (and what lets Safari play one at all). Only `.mp4`, `.jpg` and `.svg`
inside a session folder are ever served, and the folder has to be a direct child of `sessions/` —
so there is nothing to escape out of and nothing else on the card to reach.

The temperature tile is colour-coded on the Pi 5's own limits: green below 70 °C, orange from
70, red from **80 °C**, where the firmware starts capping the clock. A permanently red tile is
not a bug in the page; it means the board wants better cooling.

On the kiosk, **tapping Cyclops' eye** opens the same page fullscreen in Chromium on the panel — same
green terminal chrome, so it reads as the next screen of the same device — and the page grows a
volume slider, an INTERRUPT switch and a full-width **Close** bar to get you back to the camera.
All three appear only for the Pi's own browser: from a laptop there is nothing to close,
everything else on the page is a read-only copy of what is on the card, and the panel is meant to
be the one place the box is set. The slider and the switch sit on **SYSTEM** only, so browsing
does not spend a fifth of the panel's 480 px carrying controls you did not come for.

INTERRUPT is barge-in: **ON** and you can talk over Cyclops to cut it off, **OFF** and the mic is
shut until it has finished. It is a note on the card (`~/.cache/cyclops/barge-in`), which the
kiosk reads twice a second and hands to the session holding the microphone — so flipping it lands
on the next 20 ms of audio, not at the end of the turn, which is the point: you reach for it
having just been cut off by your own voice. Closing the page returns it to
SESSIONS, so the next tap lands where the tab says it will. A drawing arriving from the agent outranks
whatever you were looking at, takes the whole panel, and hands the screen back when it clears. If the panel ever gets stuck showing the
browser, `touch ~/.cache/cyclops/browser-close` over ssh takes it down; so does quitting the
kiosk.

```bash
uv run cyclops-admin --port=8080      # try it anywhere; port 80 needs a capability (below)
```

Install it as an always-on service on the Pi:

```bash
sudo deploy/install-admin.sh          # unit file, enable, start
systemctl status cyclops-admin
```

The unit runs as your own user and binds port 80 with `CAP_NET_BIND_SERVICE` rather than root.
Its `WorkingDirectory` must match the one the kiosk runs in — `sessions/` is relative to the
working directory, so starting it elsewhere reports zero sessions with no error anywhere. Edits
to `.env` need `systemctl restart cyclops-admin` to be picked up.

`deploy/push.sh [user@host]` rsyncs the working tree to the Pi, runs `uv sync`, restarts
`cyclops-admin`, and then restarts the kiosk via `deploy/start-kiosk.sh`. It never copies your
local `.env`.

That last step is not a convenience. The kiosk is the only long-lived process here — it runs
whatever code it loaded at startup — so a deploy that does not restart it leaves the panel on the
old build while `cyclops-smoke`, the CLI and every other check happily report the new one. The
restart is verified: `start-kiosk.sh` waits for the process to reappear and exits non-zero if it
does not, which aborts the push rather than printing a success line over a dead panel. Pass
`SKIP_KIOSK=1` to opt out — a docs-only push, or when someone is mid-conversation with it.

```bash
deploy/push.sh                       # deploy, restart both, verify the kiosk came back
SKIP_KIOSK=1 deploy/push.sh          # leave the running kiosk alone
ssh cyclops@cyclops.local cyclops/deploy/start-kiosk.sh   # just restart it
```

## Configuration (`.env`)

| Variable               | Default        | Meaning                                                    |
|------------------------|----------------|------------------------------------------------------------|
| `OPENAI_API_KEY`       | —              | Required.                                                  |
| `CYCLOPS_MODEL`        | `gpt-realtime-2.1` | Realtime model id (`gpt-realtime-2.1-mini` is ~3× cheaper). |
| `CYCLOPS_REASONING_EFFORT` | `low`      | `minimal` for lowest latency, up to `xhigh` (2.x models).  |
| `CYCLOPS_VOICE`        | `marin`        | Output voice.                                              |
| `CYCLOPS_VOLUME`       | `100`          | Output volume percent (0–100); also a slider in the touchscreen UI. |
| `CYCLOPS_CAMERA_INDEX` | `auto`         | `auto` probes cameras, remembers the one that delivers frames, and falls back to a useeplus endoscope if no `/dev/video*` answers; a number pins a specific `/dev/video*`. |
| `CYCLOPS_HALF_DUPLEX`  | `auto`         | `auto`: speaker mode on when the output device is a speaker. `1`/`0` force it. |
| `CYCLOPS_BARGE_IN_DB`  | `8`            | In speaker mode, how many dB over the echo your voice must be to interrupt. Lower = easier to interrupt, but risks the assistant cutting itself off; `off` disables barge-in. The panel's INTERRUPT switch overrides this once it has been touched. |
| `CYCLOPS_LANG`         | `en`           | Language hint (ISO-639-1) for transcribing what you say; `auto` to let it detect. Set this to your spoken language for accurate transcripts. |
| `CYCLOPS_INPUT_DEVICE` | default        | Microphone: a device index or name substring (from `uv run cyclops-devices`). Needed when there's no default mic (e.g. a Raspberry Pi). |
| `CYCLOPS_OUTPUT_DEVICE`| default        | Speaker: a device index or name substring. |
| `CYCLOPS_SESSIONS_DIR` | `sessions`     | Where session folders are written (relative to the CWD; `~` ok). |
| `CYCLOPS_CAPTURES_DIR` | `captures`     | Where a photo goes when no session is running (relative to the CWD; `~` ok). |
| `CYCLOPS_PROJECTS_DIR` | `projects`     | Where project folders are written (relative to the CWD; `~` ok). |
| `CYCLOPS_ABOUT_FILE`   | `about-you.md` | Where the standing facts about you are kept (relative to the CWD; `~` ok). |
| `CYCLOPS_PROJECTS`     | `1`            | Keep `projects/` up to date, and offer Cyclops the `open_project` / `track_project` tools; `0` turns the whole feature off. |
| `CYCLOPS_PROJECT_PHOTOS` | `3`          | Hero shots copied into a project per session; `0` keeps `Photos/` empty. |
| `CYCLOPS_DIAGRAMS`     | `1`            | Let Cyclops draw diagrams on the panel and keep them; `0` withholds both drawing tools. |
| `CYCLOPS_IMAGINE`      | `1`            | Let Cyclops redraw the last photo with a change and show it on the panel; `0` withholds `edit_photo`. |
| `CYCLOPS_SOUNDS`       | `1`            | Cues: the box booting, waking and going to sleep, the shutter; `0` disables. |
| `CYCLOPS_SLEEP_AFTER_S`| `60`           | Idle seconds before the panel blanks and the camera is released; `0` keeps it lit. This is the panel's own light — Cyclops has his own sleep, on the WAKE UP tab, and the glass only ever goes dark once he is already asleep. |
| `CYCLOPS_SLUG`         | `1`            | Name **and** summarise each finished session from its transcript; `0` leaves it date-stamped with no `summary.md` (and so with nothing to carry into the next session). |
| `CYCLOPS_REMEMBER`     | `1`            | Keep `about-you.md` up to date from what you say, and hand it to Cyclops at the start of a session; `0` turns the whole feature off and writes nothing about you. |
| `CYCLOPS_RECORD`       | `1`            | Record the session into its folder; `0` disables. |
| `CYCLOPS_RECORD_SOURCE`| `screen`       | What a session's video is of: `screen` for the panel, chrome and all, or `camera` for the raw picture. The switch on the settings screen wins over this; it only decides on a box where nobody has ever touched it. |
| `CYCLOPS_RECORD_FPS`   | `15`           | Frame rate of the recorded video. |
| `CYCLOPS_RECORD_WIDTH` | `0`            | Cap the recorded width, never upscaling. `0` keeps whatever the source is — the panel 1:1 at 800 wide, a camera at its own resolution. |
| `CYCLOPS_ADMIN_HOST`   | `0.0.0.0`      | Interface the admin page binds; `127.0.0.1` keeps it off the LAN. |
| `CYCLOPS_ADMIN_PORT`   | `80`           | Port for the admin page. Tapping the eye on the panel opens the same port. |

Variables already exported in your shell take precedence over `.env`. List audio devices with `uv run cyclops-devices`.

## macOS notes

* **Permissions:** the first run triggers Microphone and Camera prompts for the app you launched
  from (Terminal, iTerm, PyCharm…). If a capture times out, check
  *System Settings → Privacy & Security → Camera / Microphone*.
* **Speakers vs. headphones.** With open speakers the mic hears the assistant, so a naive setup
  answers itself in a loop. When the output device is a speaker (e.g. "MacBook Pro Speakers")
  cyclops runs **speaker mode**: the mic is muted while the assistant talks, *except* that it
  listens for you clearly out-talking the echo of its own voice — so you can still interrupt by
  speaking up. It learns the echo level of your room automatically; talk over it and it stops.
  With headphones/earbuds it runs full duplex with instant barge-in. The startup line tells you
  which mode you're in; `CYCLOPS_HALF_DUPLEX=1|0` forces the mode (use `1` for a monitor's
  speakers, which don't have "speaker" in their name), and `CYCLOPS_BARGE_IN_DB` tunes how loud
  you must be to interrupt over speakers (`off` to disable it). Turning barge-in off shuts the
  mic while the assistant talks whichever mode you are in, headphones included: that is what
  "wait for it to finish" has to mean.
* `chmod 600 .env` keeps your key private on a shared machine.
* If the wrong camera is used (e.g. your iPhone via Continuity Camera), pin the right one with
  `CYCLOPS_CAMERA_INDEX=<n>`; in `auto` mode a camera that opens but never delivers a frame is skipped.

## Raspberry Pi + touchscreen

cyclops runs headless-free on a Raspberry Pi (tested on a Pi 5, 64-bit Bookworm) with a USB
webcam (a Logitech C920 gives camera + mic), a USB speaker, and the official 7" touch display.

**Install** (OpenCV is the `-headless` build, so no desktop GL libs are needed):

```bash
sudo apt install -y libportaudio2            # runtime for the audio library
curl -LsSf https://astral.sh/uv/install.sh | sh   # if you don't have uv
cd ~/cyclops && cp .env.example .env         # paste your OPENAI_API_KEY
uv sync                                       # fetches Python 3.12 + aarch64 wheels
uv run cyclops-smoke                          # camera + API check, no audio needed
```

**Cameras that aren't webcams.** Most USB cameras are UVC devices: the kernel binds them, a
`/dev/video*` node appears, and `CYCLOPS_CAMERA_INDEX=auto` finds them. Some aren't. The cheap
endoscopes sold as *supercamera* (Geek szitman, and the Oasis/Depstech rebadges) expose two
vendor-specific USB interfaces and speak a proprietary protocol to a phone app, so no kernel
driver binds them and **no `/dev/video*` node is ever created** — they sit on the bus repeating a
heartbeat at a host that never answers. `lsusb` shows them, `v4l2-ctl --list-devices` does not,
and every camera probe comes up empty.

cyclops drives them anyway, from userspace over libusb. Once the `/dev/video*` probe finds
nothing, `open_camera()` looks for a useeplus endoscope and wraps it in the same
`read()`/`release()` shape the rest of the code already expects, so nothing above it knows the
difference — preview, SNAP and a `camera` recording all behave as usual, at 640×480. A real webcam
still wins if one is plugged in. The startup line says which you got:

```
· cyclops kiosk on camera useeplus     # the endoscope, over libusb
· cyclops kiosk on camera 0            # an ordinary UVC webcam at /dev/video0
```

The raw USB device is root-only by default, so access has to be granted: `deploy/push.sh`
installs `deploy/99-useeplus-camera.rules`, handing the device to group `video` — the one the
kiosk user is already in for `/dev/video*`. Without that rule the kiosk finds no camera at all.
A few `Corrupt JPEG data` lines at startup are the connect handshake and are expected; a steady
stream of them is not.

There is also an out-of-tree V4L2 kernel module for these cameras, which would produce a real
`/dev/video0` and need no cyclops code whatsoever. It is deliberately not used here: it ships
without DKMS, so it stops loading the next time the kernel updates. On a box whose whole point is
being left alone, a camera that dies on `apt upgrade` is worse than sixty lines of Python.

**640×480 is the ceiling, whatever the box says.** These are sold as "1920P HD" — the listing for
this one claims 1920×1440. That number is exactly 640×480 × 3 in each axis: it describes what the
phone app upscales to, not what the sensor reads out. The device streams 640×480 JPEG at
quality ≈44 (~19 KB a frame, 0.45 bits per pixel) at 20 fps, and there is no way to ask it for
more:

* The protocol's whole vocabulary is four commands — open (`BB AA 05 00 00`), ack, stop
  (`BB AA 08 00 00`), and switch camera/resolution (`BB AA 0B 00 02 <cam> <res> 00`). There is no
  still-capture command, so there is no high-resolution photo path hiding behind the video one.
* This unit ignores the resolution command. Every `cam`×`res` combination was tried against it,
  both mid-stream and before the stream is opened; it never acknowledges `0B` and never emits
  anything but 640×480 in type-`0x07` packets. The multi-resolution code in the vendor's Android
  app serves other products in the same family.
* Nothing is negotiable in the descriptors either: two vendor-specific interfaces, four bulk
  endpoints, one alternate setting, and no format descriptors at all.

Nor is the softness a focus problem, which is the intuitive diagnosis and the wrong one. A hard
edge takes the same 5 px to transition at 0.5 m as it does at 2.5 m — a fixed-focus lens that is
equally soft everywhere, not a focal plane you are standing outside. Distant things look worse
than near ones purely because the same pixels are spread over the same angle: at five times the
distance a feature is five times smaller, and it falls below what 640×480 and a quality-44
encoder can carry.

So the only lever left is downstream, and cyclops pulls it. `overlay.sharpen()` is a threshold-
and-ceiling-gated unsharp mask applied to the preview before it is enlarged onto the panel, and
to every photo before it is encoded for the model. **It is switched off at the moment** —
`overlay.SHARPEN` is `False`, which makes the function a no-op on both paths; the rest of this
section describes what turning it back on does. On a real frame it takes the Laplacian
variance from 18.5 to 50.1 — 2.7× the detail — while the noise floor in flat areas moves 2.10 to
2.20. The gates are what make that trade good: the floor leaves detail weaker than the encoder's
own blocking alone, and the ceiling caps how far any pixel may travel, which is what stops a face
against a bright window growing a white halo. A `camera` recording is untouched by it — the
recorder samples raw frames straight from the device — but a `screen` one is not, and should not
be: it is the glass, and the glass is sharpened.

It was free when it was measured, at 5.98 ms a frame against 5.42 ms before, on a 25 fps loop
with 40 ms to spend — because the same work found a much older waste. The panel used to be
mirrored, and the flip was written `frame[:, ::-1]`: a reversed slice is a negative-stride view
that OpenCV copies into a contiguous buffer on *every* call downstream of it. Doing it properly
with `cv2.flip` paid for the sharpening and the move from `INTER_LINEAR` to `INTER_CUBIC`
besides. The mirror is gone now, and `fit_to_window` is what keeps the buffer contiguous.

**Pick audio devices.** A Pi has several and often no default mic, so set them explicitly:

```bash
uv run cyclops-devices          # lists inputs/outputs
```

Add to `.env`. On a Pi running PipeWire/PulseAudio the simplest robust choice is to route
through `pulse` (it resamples to whatever the USB speaker wants and follows the default
sink/source), and to force speaker mode since a USB speaker leaks into the mic:

```ini
CYCLOPS_INPUT_DEVICE=pulse
CYCLOPS_OUTPUT_DEVICE=pulse
CYCLOPS_HALF_DUPLEX=1
```

(Direct-hardware devices like `hw:4,0` are often locked to 48 kHz and reject cyclops's 24 kHz —
`pulse` avoids that.) Then `uv run cyclops` works from the terminal.

**The panel.** `cyclops-kiosk` is the whole front-end and needs no browser: it opens the camera,
draws the live picture fullscreen, and lays a green terminal bezel over it — a readout strip
along the top (mode, signal meter, `REC`, session clock), the picture through the middle, and a
row of three tabs along the bottom sized for a thumb in a glove, with Cyclops himself standing in
the middle of them:

| Tab | What it does |
| --- | --- |
| **SNAP** | takes a photo and shows it to Cyclops, which answers out loud — into the running session's `photos/`, or into `captures/` and seen by nobody if there is no session |
| **the eye** | Cyclops, in the middle of the row, centred on the bar's own top rule. No label — he is a face, and you tap a face to ask what it remembers: the [admin page](#admin-page) fullscreen on the panel, on the sessions list. **Hold him** and you get the power menu instead — see below |
| **WAKE UP** | wakes Cyclops and starts the conversation; it says **GO TO SLEEP** while he is up, and the tab stays lit |

**Holding his eye shuts the box down.** Press and hold him for 0.7 s and a power menu comes up
over the picture: **SHUT DOWN**, **RESTART**, **CANCEL**, with anywhere off the card being a
cancel too. It is the gesture every phone's side button already has, and it is behind a hold for
the same reason theirs is: it is the one control here you cannot take back. While you hold, the
rule that arcs over his head fills in from the left — when it reaches the far side the menu is
open, and the panel blips, because the menu appears under the very finger that is covering it.
The card sits above his face rather than over it, and says *the whole box, not the session*,
because **GO TO SLEEP** on the tab beside it means the other thing.

Choosing does not cut the power there and then. The panel says `Shutting down…`, the kiosk tears
itself down exactly as `q` would — the session is finished, the video muxed, the folder named and
the camera released — and only then does `systemctl poweroff` (or `reboot`) run. So the box goes
down the way it would have if you had shut it down over ssh, which is the whole point: an
unplugged Pi loses the conversation it was in the middle of. It goes through `sudo -n systemctl`
first and bare `systemctl` after it, so it works whether the kiosk was started by the desktop at
boot or by `deploy/push.sh` over ssh.

The border runs along the panel's own edge, carries the state in its colour and glows inwards
from it — **green asleep, amber waking, white awake, red on a fault** — so the state reads from
across the room, and while he is up it breathes. A colour rather than a brightness, because dim
green and bright green are the same thing to anyone more than a pace away. The white is the
tube's own: a single-phosphor screen driven hard blooms towards white with its colour still in
it, which is why it is `#E1FFF0` and not paper white, and why nothing clashes — there is no
second hue to argue with the first. It reads as the same screen turned up. The same accent goes on the handful of things that only mean something during a session:
the signal meter, the session clock, the caption's `›`, the WAKE UP / GO TO SLEEP button's mic and word, and him. Everything else — the brand, the rules, the ticks, SNAP, the caption's own words — stays
phosphor green whatever he is doing, because a panel where everything is an accent has none.

Asleep the whole screen is green and completely still, with one exception: the **WAKE UP** cell
breathes, slowly — and it breathes *towards the white*, so it is both the only thing moving and
the only thing not green. That is a promise as much as a signal: the button wears the colour the
whole screen turns when you press it. The status line, bottom right, spells it
out for anyone close enough to read it, including what an error actually said.

**The middle of the row is Cyclops.** It is the one thing on the panel that is a face rather than
a readout, and it is what lets the rest stay this terse: a glance answers *is he there, and what
is he up to*, so the strip is free to spell it out only for whoever is close enough to read it.
He is the boot mark brought to life — the splash's iris inside concentric HUD rings, with the
rings turning, the iris breathing and a highlight sweeping the rim when he is hunting for
something. The bar's top rule runs in from both sides, lifts over his head and comes down the
other side; that shoulder is what makes him part of the row rather than a badge sitting on it.

Everything he does is a **mood** — nine numbers and a colour, one row per state in
`overlay.MOODS`, with `cyclops/eye.py` as the mechanism underneath. Asleep he is a shut iris in
dim green and completely still. Waking, he is amber, half open, rings running fast with a
scanning arc. Listening, he is green and calm, blinking every few seconds and never quite on the
beat, and his iris opens to your voice. Speaking, he breathes faster and wider. Searching, he
narrows to a point, the rings tear round and a bright trace sweeps a dimmed rim. A fault is red and does not move at all. He
crossfades between them rather than snapping, because a face that jumped colour between two
frames would read as a different creature.

These are meant to be pushed around rather than argued about:

```bash
uv run python tools/eye_sheet.py --seconds 8 --level 1.0   # every mood, over a strip of time
uv run python tools/eye_sheet.py --state searching --frames 16 --scale 2
```

And while he is asleep **nothing on the panel moves at all** — not the eye, not the border, not
the caption, which used to breathe whatever was happening. Two frames of a resting panel are
identical, and that stillness is what makes any of the rest read as awake. The picture keeps its own colours and is not
filtered at all; the strip and the tab row are, and they are see-through rather than opaque, so
the camera runs edge to edge behind the chrome as well as between it. Only the chrome is green,
because the point of the panel is still to see the room.

It must run **in your desktop session** — it needs PipeWire for audio and a Wayland socket for
the window, neither of which a bare systemd unit has. On the Wayland (labwc) desktop, put this
in `~/.config/labwc/autostart`:

```sh
#!/bin/sh
cd /home/<you>/cyclops && .venv/bin/cyclops-kiosk >/tmp/kiosk_live.log 2>&1 &
```

Set an initial speaker level with `CYCLOPS_VOLUME` (percent); after that the volume — and the
INTERRUPT switch — lives on the admin page, behind his eye. Nothing leaves the Pi except the audio and vision the agent
sends to OpenAI.

`q` or `ESC` quits, `f` toggles fullscreen, and `--windowed` / `--size=WxH` are there for
developing against a laptop.
