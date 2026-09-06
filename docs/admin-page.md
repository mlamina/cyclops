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

Four screens, picked by the tabs in the header and by the hash in the address bar, so nothing
ever navigates (the kiosk's browser is kept warm on this page and a reload would be felt). The
kiosk opens it on **SESSIONS**, which is what its tab promises:

| | |
|---|---|
| **SYSTEM** `#/` | **CPU temperature**, **memory**, **disk**, and **how many sessions have been recorded**. A session counts as finished once it has written its `session.md`; anything else shows as in progress. |
| **SESSIONS** `#/sessions` | Every session, newest first, each with a still lifted straight out of its own recording. Open one to watch it. |
| **PROJECTS** `#/projects` | Every project in `projects/`, most recently worked on first. Open one for a plain file browser over its folder; open a file to read it. From anything that is not the panel, also **+ Folder** and **↑ Upload**. |
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

**And you can put things in.** At the right-hand end of that same bar, from anything that is not
the panel, **+ Folder** names a new folder inline and **↑ Upload** opens a file picker — or you
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

Recordings, photos and drawings are served from `/media/<session>/<file>` with byte ranges, which
is what lets a video seek (and what lets Safari play one at all). Only `.mp4`, `.jpg` and `.svg`
inside a session folder are ever served, and the folder has to be a direct child of `sessions/` —
so there is nothing to escape out of and nothing else on the card to reach.

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
