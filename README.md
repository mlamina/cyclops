# cyclops

A low-latency speech-to-speech agent built on the **OpenAI Realtime API** (WebSocket, `openai`
Python SDK 3.x). It listens on your microphone, answers through your speakers, and **you** decide
when it looks: press the shutter, and the photo goes straight into its context so it answers
about what you are holding. It never takes a picture on its own.

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
  audio item is truncated to what you actually heard, so the model doesn't think you heard the
  rest. Over an open speaker that judgement is a guess about a room and can be wrong the worst
  way — Cyclops hearing its own voice as you — so the panel's **SYSTEM** screen carries an
  INTERRUPT switch that turns it off: the mic then stays shut until Cyclops has finished.
* The shutter shrinks the frame to 1024 px and injects it as an `input_image`, then asks for a
  response, so the model's next reply is about what it just saw. On the panel it borrows a frame
  from the preview already on screen; from the CLI it opens the camera in a worker thread and
  discards warm-up frames. Every capture is also written into the session's own folder.

## Setup

Requirements: macOS (Apple Silicon) or Linux / Raspberry Pi, Python 3.12+ (uv fetches it),
[uv](https://docs.astral.sh/uv/), PortAudio (`brew install portaudio` on macOS,
`sudo apt install libportaudio2` on Debian/Pi), a webcam, and an OpenAI API key.
See [docs/raspberry-pi.md](docs/raspberry-pi.md) for the Pi, which is the platform that matters.

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

## Docs

**What it keeps**

* [Sessions](docs/sessions.md) — one folder per conversation: the transcript, a stereo recording
  with you on one channel and Cyclops on the other, the photos, and what survives a power cut.
* [What it remembers](docs/memory.md) — the one file of standing facts about you, and the
  `projects/` shelf a session gets filed into afterwards, numbers and all.

**What it does**

* [What he does with the screen](docs/tools.md) — the scratchpad he writes himself, the diagrams
  he draws, the photo he redraws with a change, and the ledger of work still in flight.
* [The admin page](docs/admin-page.md) — `http://cyclops.local/`: how the box is doing, every
  session, every project, every picture. No database, no login.

**The box**

* [The panel](docs/panel.md) — the 7" kiosk: three controls, the power menu, the border that
  carries the state, and the eye with its moods.
* [Raspberry Pi](docs/raspberry-pi.md) — install, the autostart, what a boot looks like, and
  `deploy/push.sh`.
* [The camera](docs/camera.md) — why MJPG and not YUYV, driving a non-UVC endoscope over libusb,
  and where the sharpening sits.
* [Sound on the Pi](docs/audio.md) — the I2S amp on the GPIO header, and why stereo has to be
  folded to mono before it gets there.
* [Configuration](docs/configuration.md) — every `CYCLOPS_*` variable, plus the macOS notes.

Nothing leaves the Pi except the audio and vision the agent sends to OpenAI.
