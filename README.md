# cyclops

A low-latency speech-to-speech agent built on the **OpenAI Realtime API** (WebSocket, `openai` Python SDK 3.x).
It listens on your microphone, answers through your speakers, and has exactly one tool: it can
**snap a picture with your webcam and look at it**. Hold something up to the camera, say
"take a look at this", and talk about it.

## How it works

```
mic ──PCM16 24kHz──▶ input_audio_buffer.append ──▶ OpenAI Realtime (server VAD)
                                                        │
speakers ◀──PCM16 24kHz── response.output_audio.delta ◀─┤
                                                        │ function call: capture_webcam_image
webcam ──▶ JPEG (≤1024px) ──▶ function_call_output + input_image item ──▶ response.create
```

* One WebSocket session, server-side voice activity detection, no push-to-talk.
* Barge-in: when you start talking the local playback buffer is flushed and the assistant's
  audio item is truncated to what you actually heard, so the model doesn't think you heard the rest.
* The webcam tool runs in a worker thread, discards warm-up frames, shrinks the frame to 1024 px,
  and injects it as an `input_image` so the model's next reply is about what it sees.
  Every capture is also written into the session's own folder (see **Sessions** below).

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

Talk. Say something like *"Can you see what I'm holding?"* to trigger the camera. Press `Ctrl+C` to quit.

Headless smoke test (no mic/speakers; sends text turns and exercises the webcam tool):

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
    video.mp4         the recording (kiosk only)
    photos/
      14-32-40_cyclops.jpg
      14-33-12_you.jpg
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

**The video.** H.264 of what the camera saw (raw frames — no mirror, no UI chrome) with a
**stereo** audio track, **you on the left channel and Cyclops on the right**. The two voices are
never mixed, so you can listen to either side alone. Only the kiosk records: `cyclops` opens the
camera per photo instead of holding it open, so there is no continuous video for it to record.

Needs `ffmpeg` on `PATH` (`brew install ffmpeg`, or `apt install ffmpeg` on the Pi). Without it
the session runs exactly as before and says once that it isn't recording — nothing here ever
blocks a conversation, recording and logging included.

At the defaults (640 wide, 15 fps) a recording costs roughly 3 MB per minute and about 4% of one
Pi 5 core (measured on a Pi 5 with a C920). Nothing is ever pruned; delete what you don't want.

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

Photos taken with no session running (the smoke test, mostly) still land in `captures/`, which
keeps only the last 20 alongside `latest.jpg`. A session's own photos are never pruned.

## Projects

A session is one conversation. A **project** is the thing you keep coming back to, and it gets a
folder of its own that a person can read without knowing anything about any of this:

```
projects/
  Pelican Display Mount/
    README.md    what it is, where it stands, what's still open, what was decided
    Log.md       one dated entry per session, oldest first, never rewritten
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

Every session end costs one model call even when nothing gets filed, because deciding that *is*
the call. At these model sizes it rounds to nothing, but `CYCLOPS_PROJECTS=0` turns the whole
feature off — no sweep, and the two project tools aren't offered to the voice agent at all.

## Admin page

`cyclops-admin` serves a small Django status page on **port 80**, so from anywhere on your
network you can open `http://raspberrypi.local/` and see how the box is doing: **CPU
temperature**, **memory**, **disk**, and **how many sessions have been recorded**. It is
read-only, has no database and no login — a private-LAN dashboard, not an exposed service.
A session counts as finished once it has written its `session.md`; anything else shows as in
progress.

The temperature tile is colour-coded on the Pi 5's own limits: green below 70 °C, orange from
70, red from **80 °C**, where the firmware starts capping the clock. A permanently red tile is
not a bug in the page; it means the board wants better cooling.

On the kiosk, the **SYSTEM tab** opens the same page fullscreen in Chromium on the panel — same
green terminal chrome, so it reads as the next screen of the same device — and the page grows a
full-width **Close** bar to get you back to the camera. That bar only appears for the Pi's own
browser — from a laptop there is nothing to close. If the panel ever gets stuck showing the
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
ssh dobby@raspberrypi.local cyclops/deploy/start-kiosk.sh   # just restart it
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
| `CYCLOPS_BARGE_IN_DB`  | `8`            | In speaker mode, how many dB over the echo your voice must be to interrupt. Lower = easier to interrupt, but risks the assistant cutting itself off; `off` disables barge-in. |
| `CYCLOPS_LANG`         | `en`           | Language hint (ISO-639-1) for transcribing what you say; `auto` to let it detect. Set this to your spoken language for accurate transcripts. |
| `CYCLOPS_INPUT_DEVICE` | default        | Microphone: a device index or name substring (from `uv run cyclops-devices`). Needed when there's no default mic (e.g. a Raspberry Pi). |
| `CYCLOPS_OUTPUT_DEVICE`| default        | Speaker: a device index or name substring. |
| `CYCLOPS_SESSIONS_DIR` | `sessions`     | Where session folders are written (relative to the CWD; `~` ok). |
| `CYCLOPS_CAPTURES_DIR` | `captures`     | Where a photo goes when no session is running (relative to the CWD; `~` ok). |
| `CYCLOPS_PROJECTS_DIR` | `projects`     | Where project folders are written (relative to the CWD; `~` ok). |
| `CYCLOPS_PROJECTS`     | `1`            | Keep `projects/` up to date, and offer Cyclops the `open_project` / `track_project` tools; `0` turns the whole feature off. |
| `CYCLOPS_PROJECT_PHOTOS` | `3`          | Hero shots copied into a project per session; `0` keeps `Photos/` empty. |
| `CYCLOPS_SLUG`         | `1`            | Name **and** summarise each finished session from its transcript; `0` leaves it date-stamped with no `summary.md` (and so with nothing to carry into the next session). |
| `CYCLOPS_RECORD`       | `1`            | Record the camera into the session folder; `0` disables. |
| `CYCLOPS_RECORD_FPS`   | `15`           | Frame rate of the recorded video. |
| `CYCLOPS_RECORD_WIDTH` | `640`          | Recorded video is fit to this width, never upscaled. |
| `CYCLOPS_ADMIN_HOST`   | `0.0.0.0`      | Interface the admin page binds; `127.0.0.1` keeps it off the LAN. |
| `CYCLOPS_ADMIN_PORT`   | `80`           | Port for the admin page. The kiosk's SYSTEM tab opens the same port. |

Variables already exported in your shell take precedence over `.env`. List audio devices with `uv run cyclops-devices`.

## macOS notes

* **Permissions:** the first run triggers Microphone and Camera prompts for the app you launched
  from (Terminal, iTerm, PyCharm…). If the camera tool times out, check
  *System Settings → Privacy & Security → Camera / Microphone*.
* **Speakers vs. headphones.** With open speakers the mic hears the assistant, so a naive setup
  answers itself in a loop. When the output device is a speaker (e.g. "MacBook Pro Speakers")
  cyclops runs **speaker mode**: the mic is muted while the assistant talks, *except* that it
  listens for you clearly out-talking the echo of its own voice — so you can still interrupt by
  speaking up. It learns the echo level of your room automatically; talk over it and it stops.
  With headphones/earbuds it runs full duplex with instant barge-in. The startup line tells you
  which mode you're in; `CYCLOPS_HALF_DUPLEX=1|0` forces the mode (use `1` for a monitor's
  speakers, which don't have "speaker" in their name), and `CYCLOPS_BARGE_IN_DB` tunes how loud
  you must be to interrupt over speakers (`off` to disable it).
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
difference — preview, SNAP and session recording all behave as usual, at 640×480. A real webcam
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
row of three tabs along the bottom sized for a thumb in a glove:

| Tab | What it does |
| --- | --- |
| **SNAP** | takes a photo now — into the running session's `photos/`, or into `captures/` if none |
| **SYSTEM** | opens the [admin page](#admin-page) fullscreen on the panel |
| **SESSION** | starts and stops the conversation; the tab stays lit while one is up |

The border runs along the panel's own edge, carries the state in its colour and glows inwards
from it — dim green idle, amber connecting, bright green live, red on a fault — so the state
reads from across the room. The line under the picture spells it out for anyone close enough to
read it, including what an error actually said. The picture keeps
its own colours; only the chrome is green, because the point of the panel is still to see the
room.

It must run **in your desktop session** — it needs PipeWire for audio and a Wayland socket for
the window, neither of which a bare systemd unit has. On the Wayland (labwc) desktop, put this
in `~/.config/labwc/autostart`:

```sh
#!/bin/sh
cd /home/<you>/cyclops && .venv/bin/cyclops-kiosk >/tmp/kiosk_live.log 2>&1 &
```

Set an initial speaker level with `CYCLOPS_VOLUME` (percent); after that the volume lives on the
admin page, behind the SYSTEM tab. Nothing leaves the Pi except the audio and vision the agent
sends to OpenAI.

`q` or `ESC` quits, `f` toggles fullscreen, and `--windowed` / `--size=WxH` are there for
developing against a laptop.
