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

Every session — from `cyclops`, `cyclops-ui` or `cyclops-kiosk` — gets **one folder of its own**,
holding everything it produced. Pull the SD card into a laptop and it reads like a GoPro's:

```
sessions/
  2026-08-26_14-32-05_lego-falcon/
    session.md        the conversation, with the photos in it, for you
    session.jsonl     the same events, one JSON object per line, for a program
    summary.md        one sentence and one paragraph: what this session was
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
never mixed, so you can listen to either side alone. Only the kiosk records: `cyclops` and
`cyclops-ui` open the camera per photo instead of holding it open, so there is no continuous
video for them to record.

Needs `ffmpeg` on `PATH` (`brew install ffmpeg`, or `apt install ffmpeg` on the Pi). Without it
the session runs exactly as before and says once that it isn't recording — nothing here ever
blocks a conversation, recording and logging included.

At the defaults (640 wide, 15 fps) a recording costs roughly 3 MB per minute and about 4% of one
Pi 5 core (measured on a Pi 5 with a C920). Nothing is ever pruned; delete what you don't want.

**If a session is cut off** — power loss, a killed process — its folder keeps a `parts/` with a
playable `video-raw.mp4` and the two WAVs, and a `session.jsonl` that ends wherever the power
did. `cyclops-sessions` lists it as `UNFINISHED`; `cyclops-sessions --fix` muxes the video,
removes `parts/`, and writes the missing `session.md`. It works entirely offline;
`cyclops-sessions --name` is the separate step that names anything still unnamed and writes any
`summary.md` that is missing — the one part that does need a key and a network. A folder that
already has both is never sent to the model, so it is safe to run over a whole card repeatedly.

```bash
uv run cyclops-sessions              # what's on the card
uv run cyclops-sessions --fix        # finish anything left half-done (no network needed)
uv run cyclops-sessions --fix --name # ...and name/summarise what the live call missed
```

Photos taken with no session running (the smoke test, mostly) still land in `captures/`, which
keeps only the last 20 alongside `latest.jpg`. A session's own photos are never pruned.

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

On the kiosk, the **gear in the top-left corner** opens the same page fullscreen in Chromium on
the panel, and the page grows a full-width **Close** bar to get you back to the camera. That bar
only appears for the Pi's own browser — from a laptop there is nothing to close. If the panel
ever gets stuck showing the browser, `touch ~/.cache/cyclops/browser-close` over ssh takes it
down; so does quitting the kiosk.

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

`deploy/push.sh [user@host]` rsyncs the working tree to the Pi, runs `uv sync` and restarts the
service. It never copies your local `.env`.

## Configuration (`.env`)

| Variable               | Default        | Meaning                                                    |
|------------------------|----------------|------------------------------------------------------------|
| `OPENAI_API_KEY`       | —              | Required.                                                  |
| `CYCLOPS_MODEL`        | `gpt-realtime-2.1` | Realtime model id (`gpt-realtime-2.1-mini` is ~3× cheaper). |
| `CYCLOPS_REASONING_EFFORT` | `low`      | `minimal` for lowest latency, up to `xhigh` (2.x models).  |
| `CYCLOPS_VOICE`        | `marin`        | Output voice.                                              |
| `CYCLOPS_VOLUME`       | `100`          | Output volume percent (0–100); also a slider in the touchscreen UI. |
| `CYCLOPS_CAMERA_INDEX` | `auto`         | `auto` probes cameras and remembers the one that delivers frames; a number pins it. |
| `CYCLOPS_HALF_DUPLEX`  | `auto`         | `auto`: speaker mode on when the output device is a speaker. `1`/`0` force it. |
| `CYCLOPS_BARGE_IN_DB`  | `8`            | In speaker mode, how many dB over the echo your voice must be to interrupt. Lower = easier to interrupt, but risks the assistant cutting itself off; `off` disables barge-in. |
| `CYCLOPS_LANG`         | `en`           | Language hint (ISO-639-1) for transcribing what you say; `auto` to let it detect. Set this to your spoken language for accurate transcripts. |
| `CYCLOPS_INPUT_DEVICE` | default        | Microphone: a device index or name substring (from `uv run cyclops-devices`). Needed when there's no default mic (e.g. a Raspberry Pi). |
| `CYCLOPS_OUTPUT_DEVICE`| default        | Speaker: a device index or name substring. |
| `CYCLOPS_SESSIONS_DIR` | `sessions`     | Where session folders are written (relative to the CWD; `~` ok). |
| `CYCLOPS_CAPTURES_DIR` | `captures`     | Where a photo goes when no session is running (relative to the CWD; `~` ok). |
| `CYCLOPS_SLUG`         | `1`            | Name **and** summarise each finished session from its transcript; `0` leaves it date-stamped with no `summary.md` (and so with nothing to carry into the next session). |
| `CYCLOPS_RECORD`       | `1`            | Record the camera into the session folder; `0` disables. |
| `CYCLOPS_RECORD_FPS`   | `15`           | Frame rate of the recorded video. |
| `CYCLOPS_RECORD_WIDTH` | `640`          | Recorded video is fit to this width, never upscaled. |
| `CYCLOPS_ADMIN_HOST`   | `0.0.0.0`      | Interface the admin page binds; `127.0.0.1` keeps it off the LAN. |
| `CYCLOPS_ADMIN_PORT`   | `80`           | Port for the admin page. The kiosk's gear button opens the same port. |

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

**Touchscreen UI.** `cyclops-ui` serves a full-screen page from a local web server, shown in
Chromium kiosk mode: a big eye you **tap to start/stop** a session (its iris recolours and pulses
with the audio — connecting / listening / speaking / looking), a row of **status icons**
(link / mic / speaker / camera), and a **volume slider**. Set an initial level with
`CYCLOPS_VOLUME` (percent). The page reloads itself whenever the server restarts, so UI updates
apply without touching Chromium.

Run **both the server and the kiosk from your desktop session** — the server needs the session's
PipeWire/PulseAudio (it opens `pulse`), so a system-wide systemd service won't work (it has no
`XDG_RUNTIME_DIR` and can't see your audio; you'd get *"No output device matching 'pulse'"*).
On the Wayland (labwc) desktop, put this in `~/.config/labwc/autostart`:

```sh
#!/bin/sh
# start the UI server in-session (has audio), then open the kiosk once it's up
cd /home/<you>/cyclops && /home/<you>/.local/bin/uv run cyclops-ui --port=8730 >/tmp/cyc_ui.log 2>&1 &
for i in $(seq 1 30); do curl -sf http://localhost:8730/status >/dev/null 2>&1 && break; sleep 1; done
chromium-browser --ozone-platform=wayland --kiosk --noerrdialogs --disable-infobars \
  --user-data-dir=/tmp/cyclops-chrome http://localhost:8730/ &
```

The page is one self-contained file (`src/cyclops/ui.html`, no external assets) and only talks to
the local server on `127.0.0.1`, so nothing leaves the Pi except the audio/vision the agent sends
to OpenAI.
