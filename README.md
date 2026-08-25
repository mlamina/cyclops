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
  Every capture is also written to `captures/` (relative to where you run it): a timestamped
  JPEG plus `latest.jpg`.

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
| `CYCLOPS_CAPTURES_DIR` | `captures`     | Where snapshots are written (relative to the CWD; `~` ok). |

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
