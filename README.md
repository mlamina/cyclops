# Cyclops — your workshop droid

A one-eyed robot that works with you. It remembers every project, has read every manual, and
answers any question — so your hands stay free and your phone stays in your pocket.

## Why it exists

An assistant like Claude is enormously helpful on a project — but it lives in a phone, a tablet,
an app, all built for sitting down, none of them for when your hands are full. Cyclops brings that
help to where the work is. It takes four interruptions away:

* **It remembers.** Everything about a project is written down and kept — what was decided, what
  was measured, what is still open — without anyone filing it.
* **It has read the manual.** Questions get answered when they are asked, not after a search.
* **No phone, laptop, or tablet.** Nothing to unlock, type into, or scroll through. It sits next
  to you and you talk to it.
* **It is good company.** It has an eye and a character. Working with it is more fun than working
  alone.

Because it is designed to be intuitive — no app, no account, no prompt to write, just talking —
people who aren't technical can use it as easily as anyone. You switch it on and say what you're
working on.

## Core values

The tiebreakers. When two designs are both reasonable, the one that keeps these wins.

1. **Hands stay free.** Voice in; voice and the screen out. If it needs a keyboard, a phone, or a
   menu, it is wrong.
2. **It never becomes the task.** No settings to tend, no attention of its own. The moment you are
   working on Cyclops instead of the project, it has failed.
3. **Nothing gets lost.** Every session is written down. What was decided, measured, and left open
   is there next week without anyone having filed it.
4. **Good company.** It has a face and a character. It should be more fun to work with than to
   work without.
5. **It does what you expect.** Talking to it works the way talking works; showing it something
   works the way showing works. Nothing to learn, nothing to be shown.

## Criteria for success

1. **The phone stays in the pocket.** Nothing gets looked up on another device during a session.
   Reaching for one is a miss worth writing down.
2. **One question, one answer.** A question that needs a follow-up — "what?", "say that again",
   "no, the other one" — is a miss. The transcripts have the count.
3. **Zero time on Cyclops itself.** Turns about the tool — settings, restarts, "why didn't you" —
   tend to none.
4. **No unexpected waits.** Answers arrive at the speed of talking. Anything slower is announced
   before it starts — what it is doing and roughly how long — so a wait is never a surprise and
   never a silence.
5. **Fun and surprise.** A turn you did not see coming, a thing it noticed before you did. A
   session with no surprise in it is one an app could have done.
6. **No learning curve.** It behaves the way you assumed it would before anyone told you anything.
   Having to work out what it wants — how to phrase it, when it's listening, what it can do — is a
   miss.

## How it works

Under the hood it is a low-latency speech-to-speech loop on the **OpenAI Realtime API**
(WebSocket, `openai` Python SDK 3.x). It listens on your microphone, answers through your
speakers, and **you** decide when it looks: press the shutter, and the photo goes straight into
its context so it answers about what you are holding. It never takes a picture on its own.

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
