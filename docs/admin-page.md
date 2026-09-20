# The admin page

`cyclops-admin` serves a small Django page on **port 80**, so from anywhere on your network you
can open `http://cyclops.local/` and see both how the box is doing and what it has recorded. It
has no database and no login — a private-LAN dashboard, not an exposed service.

Almost all of it is read-only. The two exceptions are in the project browser, where anything on
the network can make a folder and upload a file into `projects/` — putting a datasheet on the box
from the machine the datasheet is on is the whole point of them. What stands in for a login is
containment rather than a credential: a destination is resolved before it is compared against the
project it must be under, every name is sanitised rather than trusted, and a file is refused past
128 MB. That defends against a mistake and against a browser on another origin, not against
something on your LAN that is not a browser — if the Pi ever sits on a network you do not own,
this is the first thing to revisit.

Five screens, picked by the tabs in the header and by the hash in the address bar, so nothing
ever navigates (the kiosk's browser is kept warm on this page and a reload would be felt). The
kiosk opens it on **SESSIONS**, which is what its tab promises:

| | |
|---|---|
| **LIVE** `#/live` | Companion mode — see below. Not on the panel. |
| **SYSTEM** `#/` | **CPU temperature**, **memory**, **disk**, and **how many sessions have been recorded**. A session counts as finished once it has written its `session.md`; anything else shows as in progress. |
| **SESSIONS** `#/sessions` | Every session, newest first, each with a still lifted straight out of its own recording. Open one to watch it. |
| **HIGHLIGHTS** `#/highlights` | The reel — every clip on the card, newest first, playing one after another on its own, with sound. Tap the left or right third to step and the middle to pause; the arrow keys and space do the same on a laptop. Every switch is marked by a white flash across the picture, because clip after clip of the same bench in the same room otherwise runs together. Nothing is written over the picture at all: the progress bar sits under it, the title beside it on the panel — the clip goes hard right, at the row's height less the strip the bar takes — and under it on anything wider. See "Clips" below. |
| **PROJECTS** `#/projects` | Every project in `projects/`, most recently worked on first. Open one and it has a screen of its own, with a submenu down the left: **Overview**, **Log**, **Spreadsheets**, **Photos** and **Files**. |
| **MEDIA** `#/media` | Every photo and every drawing on the card as one stream, newest first. Tap one to fill the screen and flip through with the arrows, the arrow keys, or the columns down either side. |

**The session view changes shape with the screen.** On a laptop it is the recording on the left
and the whole conversation on the right, with the photos and drawings sitting inline where they
were taken; every line is stamped with its offset, and clicking one seeks the video to that
moment — the transcript's `t` *is* the recording's timeline. On the 7" panel and on a phone it is
the video and what it is called, and nothing else: at a bench you are not there to read. The
narrow layout does not hide the transcript, it never asks the server for it.

## Companion mode

**LIVE is the box on a second screen.** A phone or an iPad on the LAN, propped against the bench
or carried into the next room: **the panel itself**, as it is drawn — the eye, the dials, the
terminal line, and the photograph, sketch or manual page lying over them — the conversation as it
is spoken, and a switch that moves his voice onto the device in your hand. The panel is never
sent this screen; the screen you would be reading it on is the one Cyclops is talking out of.

**It is one stage and two pads.** The picture fills everything under the menu, at the largest 5:3
the screen will hold, and two pads float in its top-right corner: **AUDIO**, which takes his
voice, and **TEXT**, which swaps the picture for the transcript in the same frame. Exactly one of
the picture, the words and the "nothing running" mark is ever on the glass. Both pads remember
their setting per device. There is no session title strip: a session that is running is a picture
that is moving and a light travelling under the header, and the strip was the second of two marks
answering one question.

**The picture is always there, session or no session**, because the panel is always drawn. This
is a window onto the bench rather than a waiting room.

**It is the compositor's frames, not the camera's.** `cyclops.screen.ScreenSource` — the same
capture a session's video is made of — is **held** rather than started, so a session beginning
under a watching phone does not restart it and a session ending does not end it. There is never
more than one `wf-recorder` whoever is holding it, and a box that cannot capture its own screen
falls back to the camera exactly as the recording does.

**The two streams come from the kiosk's own port, not from this service.** `cyclops-admin` is two
gunicorn workers of two threads, and one viewer holding an endless response for the picture and
another for the voice is half of it. So `cyclops.companion` opens **port 8081** inside the kiosk
process, where the screen and the speaker already are — no second service, no file to publish
through, and no note another process could leave armed. Three routes: `/camera.mjpg`,
`/camera.jpg` and `/voice.pcm`, plus `/` for what state they are in, `/listening` for the speaker
claim and `/sketch.sse` for a drawing arriving.

**Nothing is produced until something connects, and the connection is the whole gate.** With the
screen shut there is no capture, no encoder thread, and the speaker's tap is a load and a test.
Pressing **TEXT** closes the stream too, so the pad that hides the picture stops paying for it;
the capture is held three seconds past the last viewer so a reload does not blink it. With a
viewer it is 800x480 at 12 fps, measured on the Pi on 2026-09-19 at **about 24% of one core** —
19% inside the kiosk (the reader thread's colour conversion, plus the JPEG encode) and 5% for
`wf-recorder` itself. That is a good deal more than the 2.75% the old 640x360 camera crop cost,
and it is the price of the whole panel at its own size; during a session the `wf-recorder` half
is already being paid for the recording. A panel that is not changing is **not** a fault and gets
no card — asleep, nothing on it moves at all, and every frame is both old and true. A camera that
is unplugged or has stopped answering still sends the panel's own words on a black card.

**The switch hands the voice over rather than copying it.** Turning it on plays what Cyclops says
through the device, and the panel's own amp stops playing him — two copies of the same sentence a
few hundred milliseconds apart is worse than one. The page POSTs nothing to do it: while the
switch is on it says every three seconds that it is still the speaker, and the claim lapses after
eight. So a phone that walks out of range, a locked screen, a closed tab and a browser that died
all end the same way — his voice is back on the panel within a few seconds, and nobody has to
remember to switch anything back. The four routes that really do set the box still answer to
loopback and nothing else.

**A held connection is deliberately not the claim**, and that was learned the hard way: a `curl`
left running on another machine held the voice stream open, the panel handed its voice to it, and
there was no sound anywhere — not on the panel, which had given the voice away, and not on the
phone, which was not the thing holding the socket. A peer that has gone keeps a socket for as long
as TCP takes to notice, which is a minute of send buffer or never. A claim that has to be renewed
cannot fail that way round: the panel is the speaker somebody is standing at, so it keeps his
voice unless a device is actively still asking to have it.

**It takes his voice and nothing else.** Turning the *volume* down was the first attempt at this
and it was wrong: the sound cues — the shutter, the wake chime, the rungs of the volume knob —
play on their own stream beside the speaker rather than through it, so a sink at zero took them
with it and they were audible on neither the panel nor the phone. What the switch reaches now is
one flag on the speaker, read on the audio thread: his voice does not leave the amplifier, and
everything the box says about itself still does. The cues stay on the panel deliberately, because
the button they are answering for is on the panel.

What it carries is his voice alone — 24 kHz mono PCM, straight off the speaker's own output. The
microphone is never on it, and neither are the sound cues, which
[sit beside the speaker rather than in it](audio.md) and so are not in the session's video
either. `AudioWorklet` is not available: it is secure-context only and this is plain HTTP on a
LAN name, so the page uses a `ScriptProcessorNode` — deprecated, and the only thing that works
here.

**A project opens as a project.** A folder holds a README, a log, a workbook and its pictures,
and until this screen existed they were four rows of a directory listing told apart only by their
filenames. The rail down the left names them instead — `#/p/<project>/<section>`, five of them:

| | |
|---|---|
| **Overview** | `README.md`. What it is, where it stands, what is still open and what was decided. Rewritten by the sweep from the log. |
| **Log** | `Log.md`. One dated entry per session, oldest first, with the photographs that came out of each. |
| **Spreadsheets** | Every `.xlsx` in the folder — in practice `Project Data.xlsx` — as the remembered pairs under the tab name they were written on, rather than a spreadsheet grid nobody wants on a 7" panel. |
| **Photos** | Every picture anywhere in the project, newest first, and not only the ones the sweep filed into `Photos/` — a folder of drawings you dropped in is as much this project as a photograph Cyclops took. Tap one for the whole frame, with whatever the index service saw in it along the bottom. |
| **Files** | The plain file browser it used to open as. `#/p/<project>/files/<folder>` browses, `#/f/<project>/<file>` reads, and a bar across the top says where you are inside it. |

Markdown is rendered with the pictures it links to shown inline — the paths are rewritten
server-side, which is why a relative `![](Photos/x.jpg)` written by the sweep resolves at all, and
why the README's own link to `Log.md` lands on the **Log** section rather than on a file. Pictures,
drawings and recordings open in place; anything else says what it weighs and leaves it at that.

The rail is a column beside the pane on the panel and on anything with room for one, and two rows
above it on a phone — the same question the header's tabs ask, answered by the same media query in
`lan.css`. Nothing is ever a tap out of sight.

**And you can put things in.** At the right-hand end of the **Files** bar, from anything that is
not the panel, **+ Folder** names a new folder inline and **↑ Upload** opens a file picker — or you
drag files onto the browser. Files go up one request at a time, so the listing repainting under
you *is* the progress bar and a failure names the file it belongs to. A name already there is
replaced whole: the bytes land on a scratch name beside the target and are renamed onto it, so
the file you had is intact until the instant the new one takes over.

The body of an upload is the file itself and not a form: no multipart to parse, and no spool
through `/tmp` — Django's own upload handling would put every file over 2.5 MB onto the SD card
once before we put it there again. Names are sanitised by the same function that turns a model's
project title into a folder name, which leaves no separator alive, so a name cannot become a
path. Nothing you upload can begin with a dot, because a file the browser hides is a worse answer
than a refusal.

Markdown is turned into HTML on the server, where the project folder is known: the frontmatter
and the `<!-- cyclops:session … -->` terminators are dropped, and everything else that could be a
tag is escaped, so the only tags the page is handed are ones the renderer wrote. Project files
are served from `/project-media/<project>/<file>` by the same suffix allow-list plus `.png`, and
every path is resolved before it is compared against the project folder — so `..`, an absolute
name and a symlink out of the tree all fail the same check.

Recordings, clips, photos and drawings are served from `/media/<session>/<file>` with byte ranges,
which is what lets a video seek (and what lets Safari play one at all). Only `.mp4`, `.jpg` and
`.jpeg` inside a session folder or one level down in `photos/` or `clips/` are ever served, and
the folder has to be a direct child of `sessions/` — so there is nothing to escape out of and
nothing else on the card to reach.

The page's own CSS and JS under `src/cyclops/admin/static/` are served from the Pi, never a CDN;
nothing about the panel needs the internet.

The temperature tile is colour-coded on the Pi 5's own limits: green below 70 °C, orange from
70, red from **80 °C**, where the firmware starts capping the clock. A permanently red tile is
not a bug in the page; it means the board wants better cooling.

## On the panel

**Tapping Cyclops' eye** opens the same page fullscreen in Chromium on the panel — same green
terminal chrome, so it reads as the next screen of the same device — and the page grows a volume
slider, an INTERRUPT switch and a full-width **Close** bar to get you back to the camera. Tapping
the **heat gauge** in the bottom-right corner opens it on **SYSTEM** instead. The browser is warm
from boot and is never navigated, so the panel leaves the screen it wants in
`~/.cache/cyclops/page-screen` and the page routes itself off the poll it already runs several
times a second — see `views.panel`.

All three of those controls appear only for the Pi's own browser: from a laptop there is nothing
to close, and the panel is meant to be the one place the box is set. The split runs the other way
too — the project browser's **+ Folder** and **↑ Upload** are sent to everything *except* the
panel, which has no filesystem to upload from and no file picker worth opening at 800×480. The
slider and the switch sit on **SYSTEM** only, so browsing does not spend a fifth of the panel's
480 px carrying controls you did not come for.

INTERRUPT is barge-in: **ON** and you can talk over Cyclops to cut it off, **OFF** and the mic is
shut until it has finished — or until you tap the panel. With the switch off, a press anywhere
that is not the knob, his eye or the heat gauge stops him mid-word and opens the microphone on
the spot; those three are under 6% of the screen between them, so the gesture is "anywhere". It
is a note on the card (`~/.cache/cyclops/barge-in`) which the kiosk reads twice a second and
hands to the session holding the microphone, so flipping it lands on the next 20 ms of audio
rather than at the end of the turn — which is the point, since you reach for it having just been
cut off by your own voice.

STABILIZE holds the camera picture still in the hand (`cyclops.steady`): **ON**, the default,
the panel (and so the recording), and the admin page's camera stream get a steadied window of
the frame; **OFF**, they get the raw frame, as before stabilizing existed. Photos for the model
are raw either way. It is a note (`~/.cache/cyclops/steady`) the camera's reader looks at twice a
second, so it lands while you are still on the page and outlives a restart.

Closing the page returns it to SESSIONS, so the next tap lands where the tab says it will. A
drawing arriving from the agent outranks whatever you were looking at, takes the whole panel, and
hands the screen back when it clears. If the panel ever gets stuck showing the browser,
`touch ~/.cache/cyclops/browser-close` over ssh takes it down; so does quitting the kiosk.

## Running it

```bash
uv run cyclops-admin --port=8080      # try it anywhere; port 80 needs a capability
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
