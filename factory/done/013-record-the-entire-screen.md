---
state: done
opened: 2026-09-18
---

# Recording should record the entire screen, with everything in it

> Would it be possible to have cyclops record the exact video stream that is displayed on the pi?
> instead of having to manually stitch together things like displayed images, etc?

> 15fps is not enough, we want a smooth video

> recording should record the entire screen, with everything in it. no more manual stitching things
> together

Asked whether the SCREEN/CAMERA switch on the settings screen should stay once the screen is
captured whole: **remove it**.

## Plan

### What it looks like when it works

`video.mp4` shows exactly what was on the glass, at 30 fps: the camera and chrome, then the
diagram, the photo, the scratchpad and the manual page scrolling under your finger. Nothing is
rebuilt and nothing goes black. The shortened clip is 30 fps too, so the session screen plays
smoothly.

Measured on the Pi before capture (2026-09-18): grabbing the screen with `wf-recorder` cost about
15% of a core; running `grim` once per frame cost about 30%. A 177 s `wf-recorder` recording of
the 800×480 screen at 30 fps worked by hand. The camera already runs at 30 fps (`b66b000`).

The repo moves under long plans — re-verify every file and symbol below before touching it.

### 1. A screen source for the recorder

A new `ScreenSource` runs `wf-recorder` for the whole session, sending raw frames through a pipe,
and keeps the newest frame. It plugs into the existing `SessionRecorder` as a frame source, so
these all stay as they are:

- the 30 fps sampling on the wall clock;
- the audio lining up with the picture;
- the two audio channels;
- crash repair of an interrupted recording.

If `wf-recorder` is missing, or no frame arrives within the first second, the session records the
camera instead and the session log says why. Today, no first frame means no audio is recorded
either, so this also closes that gap. The camera stays only as this fallback.

The capture process is stopped when the session ends, even if the session crashes.

### 2. 30 fps everywhere

`record_fps` (config.py) defaults to 30, and so does `cut.FPS`. The cut's boundaries then snap to
1/30 s and the clip comes out at 30 fps. Clip render time on the Pi roughly doubles; the estimate
for the longest session is about 2 minutes, against the 6-minute timeout.

### 3. Delete the stitching

- The recording-only parts of `record.py`, `kiosk.py`, `panel.py` and `config.py`: the kiosk
  handing over its own frames (`PanelSource`), the page's still coming back (`panel.keep_still`,
  `PANEL_STILL_FILE`), and the kiosk-side still logic (`_keep_sketch_still`, `_restill`,
  `_publish_still`, `_page_still`).
- `still.py`, all of it.
- The page's scratchpad-to-picture step in `panel.js`, whose picture rides in the body of
  `POST /panel/painted` (the POST itself stays, with an empty body).
- The tests for all of the above.

### 4. Delete the switch

- `filming.py`.
- The RECORD section on the settings screen and its click handler in `status.js`.
- The `/record-source` route and the `record_screen` value in the status data.
- `RECORD_SOURCE_FILE`, the `record_source` setting and `CYCLOPS_RECORD_SOURCE`.

Job 012 (stabilization, done) steadied camera-mode recordings with a crop. With no camera mode,
that part of 012 stops applying to recordings; the panel's steady picture is what gets recorded.

### 5. One-time install

A new `deploy/install-screen-capture.sh` that runs `apt install wf-recorder`, copying the pattern
of `install-tether.sh`, plus a line in `docs/raspberry-pi.md`.

### 6. Docs

Update the video section of `docs/sessions.md` (what's recorded and what it costs),
`docs/configuration.md` and `.env.example`.

## Done when

- [x] A fake capture process writing raw frames reaches the recorder, which muxes it into a 30 fps
      `video.mp4` of the right length (within one frame) — pytest
- [x] With `wf-recorder` missing, or dead before its first frame, the session records the camera
      and the session log gives the reason — pytest
- [x] The capture process is stopped when the session ends, even if the session crashes — pytest
- [x] The cut snaps boundaries to 1/30 s and its clip comes out at 30 fps — pytest
- [x] The stitching and the switch are gone:
      `rg "keep_still|PANEL_STILL|PanelSource|of_panel|_restill|_keep_sketch_still|record_source|RECORD_SOURCE|record-source|record_screen" src tests`
      finds nothing — `rg`
- [ ] The suite stays under 10 s — `uv run pytest --durations=10`, before and after
- [x] The settings screen shows no RECORD switch and nothing moved into its place — a Playwright
      shot at 800×480
- [ ] `wf-recorder` is installed on the Pi by the install script — **yours, on the Pi**
- [ ] One session with a diagram, a photo, a scratchpad and a manual page you scroll: `video.mp4`
      shows each one as it was on the glass, with no black stretch — **yours, on the Pi**
- [ ] Clap in front of the camera: in the video, the sound and the hands meeting are within 2
      frames of each other — **yours, on the Pi**
- [ ] While the camera pans, the recording has at least 25 distinct frames per second (counted by
      ffmpeg's `mpdecimate` filter) — **yours, on the Pi**
- [ ] Recording adds at most 40% of one core over not recording (today, at 15 fps, it adds 18%),
      and `get_throttled` stays `0x0` through a 10-minute session — **yours, on the Pi**
- [ ] That 10-minute session's clip renders in under 3 minutes (the index journal's "rendered in
      Ns" line) — **yours, on the Pi**

## Built — 2026-09-18

Every kiosk session now records the whole screen. A new `ScreenSource` (`cyclops/screen.py`) runs
`wf-recorder` for the length of the session, using the exact command tested on the Pi before
capture (`-c rawvideo -m rawvideo -x bgr0`, into a named pipe). The Pi's wf-recorder 0.3 asks
before overwriting a file and has no flag to skip that, so the kiosk answers "Y" on stdin. The
recorder samples the newest frame at 30 fps, as it sampled the camera before. The session log
starts the capture and waits up to 1 s for a first frame. If none comes, or wf-recorder is
missing, the session records the camera instead. Every kiosk session's `session.jsonl` now
gets a `recording` line: `source` is `screen` or `camera`, plus `why` when it's the camera. The
capture is stopped on every way out of the session, crashes included. The frame size is the
panel's resolution, which the kiosk already reads (raw frames carry no header). A second
monitor next to the panel is untested. All of the stitching is deleted: the kiosk handing over
its own frames, the swap/restill logic, the sketch photographs, `still.py`, the page's
scratchpad raster (`/panel/painted` now posts an empty body). The RECORD switch, its route,
note file, setting and `CYCLOPS_RECORD_SOURCE` are gone too. `record_fps`, the recorder's
default and `cut.FPS` are 30. The camera fallback still uses 012's steadied camera window.
Left alone as out of scope: a video offer still carries its thumbnail under `image`, though
nothing reads it any more.
The suite criterion is unticked: it was already over 10 s here before this job
(17.6 / 22.2 s before, 17.7 / 17.8 s after, alternating runs at load ~5), and this job doesn't
change it.

Hands-on: after `/try`, run `ssh cyclops@cyclops.local cyclops/deploy/install-screen-capture.sh`
once. It should end by saying the screen can be captured. Until it runs, sessions record the
camera, and their logs say wf-recorder isn't installed. Then one session with a diagram, a photo, a
scratchpad and a scrolled manual page, a clap in front of the camera, and a pan. Watch
`video.mp4` and check `session.jsonl` says `"source": "screen"`. The clap is the one I'd
watch: the screen shows the camera slightly late and nothing corrects for that. CPU and heat
over 10 minutes are unmeasured.
factory/html/013.html
