# Configuration

| Variable               | Default        | Meaning                                                    |
|------------------------|----------------|------------------------------------------------------------|
| `OPENAI_API_KEY`       | —              | Required.                                                  |
| `CYCLOPS_MODEL`        | `gpt-realtime-2.1` | Realtime model id (`gpt-realtime-2.1-mini` is ~3× cheaper). |
| `CYCLOPS_REASONING_EFFORT` | `low`      | `minimal` for lowest latency, up to `xhigh` (2.x models).  |
| `CYCLOPS_VOICE`        | `marin`        | Output voice: `marin`, `cedar`, `alloy`, `ash`, `ballad`, `coral`, `echo`, `sage`, `shimmer` or `verse`. `marin` and `cedar` are the two OpenAI recommends. The panel's VOICE stepper overrides this once it has been touched, and plays each voice as you step through them. |
| `CYCLOPS_VOLUME`       | `100`          | Output volume percent (0–100); also a slider in the touchscreen UI. |
| `CYCLOPS_CAMERA_INDEX` | `auto`         | `auto` probes cameras, remembers the one that delivers frames, and falls back to a useeplus endoscope if no `/dev/video*` answers; a number pins a specific `/dev/video*`. |
| `CYCLOPS_HALF_DUPLEX`  | `auto`         | `auto`: speaker mode on when the output device is a speaker. `1`/`0` force it. |
| `CYCLOPS_BARGE_IN_DB`  | `8`            | In speaker mode, how many dB over the echo your voice must be to interrupt. Lower = easier to interrupt, but risks the assistant cutting itself off; `off` disables barge-in. The panel's INTERRUPT switch overrides this once it has been touched. |
| `CYCLOPS_LANG`         | `en`           | Language hint (ISO-639-1) for transcribing what you say; `auto` to let it detect. Set this to your spoken language for accurate transcripts. |
| `CYCLOPS_MIC_COMPRESS` | `1`            | Even out how loud you are on the way in: quiet speech up about 20 dB, loud speech pulled down, nothing clipping. Applies to what the model hears *and* what the recording keeps; `0` leaves the microphone exactly as it is. |
| `CYCLOPS_MIC_GAIN_DB`  | `0`            | A fixed gain in front of the compressor, for a microphone that is quiet before anything else has an opinion. Both plus and minus. |
| `CYCLOPS_INPUT_DEVICE` | default        | Microphone: a device index or name substring (from `uv run cyclops-devices`). Needed when there's no default mic (e.g. a Raspberry Pi). |
| `CYCLOPS_OUTPUT_DEVICE`| default        | Speaker: a device index or name substring. |
| `CYCLOPS_SESSIONS_DIR` | `sessions`     | Where session folders are written (relative to the CWD; `~` ok). |
| `CYCLOPS_CAPTURES_DIR` | `captures`     | Where a photo goes when no session is running (relative to the CWD; `~` ok). |
| `CYCLOPS_PROJECTS_DIR` | `projects`     | Where project folders are written (relative to the CWD; `~` ok). |
| `CYCLOPS_ABOUT_FILE`   | `about-you.md` | Where the standing facts about you are kept (relative to the CWD; `~` ok). |
| `CYCLOPS_PROJECTS`     | `1`            | Keep `projects/` up to date, and offer Cyclops the `open_project` / `track_project` tools; `0` turns the whole feature off. |
| `CYCLOPS_DIAGRAMS`     | `1`            | Let Cyclops draw diagrams on the panel and keep them with the photos; `0` withholds `draw`. |
| `CYCLOPS_IMAGINE`      | `1`            | Let Cyclops redraw any photo from this session with a change and show it on the panel; `0` withholds `edit_photo`. |
| `CYCLOPS_SCRATCHPAD`   | `1`            | Let Cyclops write on the panel himself, as a small piece of HTML, without being asked; `0` withholds `write_on_scratchpad`. |
| `CYCLOPS_SOUNDS`       | `1`            | Cues: the box booting, waking and going to sleep, the shutter; `0` disables. |
| `CYCLOPS_SLEEP_AFTER_S`| `60`           | Idle seconds before the panel blanks and the camera is released; `0` keeps it lit. This is the panel's own light — Cyclops has his own sleep, on the WAKE UP tab, and the glass only ever goes dark once he is already asleep. |
| `CYCLOPS_SLUG`         | `1`            | Name **and** summarise each finished session from its transcript; `0` leaves it date-stamped with no `summary.md` (and so with nothing to carry into the next session). |
| `CYCLOPS_REMEMBER`     | `1`            | Keep `about-you.md` up to date from what you say, and hand it to Cyclops at the start of a session; `0` turns the whole feature off and writes nothing about you. |
| `CYCLOPS_RECORD`       | `1`            | Record the session into its folder; `0` disables. |
| `CYCLOPS_RECORD_FPS`   | `30`           | Frame rate of the recorded video. The screen repaints at about 29 fps, so more buys nothing. |
| `CYCLOPS_RECORD_WIDTH` | `0`            | Cap the recorded width, never upscaling. `0` keeps whatever the source is — the screen 1:1 at 800 wide, or the camera at its own resolution when the screen cannot be captured. |
| `CYCLOPS_ADMIN_HOST`   | `0.0.0.0`      | Interface the admin page binds; `127.0.0.1` keeps it off the LAN. |
| `CYCLOPS_ADMIN_PORT`   | `80`           | Port for the admin page. Tapping the eye on the panel opens the same port. |

Variables already exported in your shell take precedence over `.env`. List audio devices with `uv run cyclops-devices`.

## macOS

None of this is needed on the Pi; it is here for developing against a laptop.

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
  `CYCLOPS_CAMERA_INDEX=<n>`; in `auto` mode a camera that opens but never delivers a frame is
  skipped.
