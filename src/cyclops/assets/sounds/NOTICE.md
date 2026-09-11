# The shipped sound cues

Designed cues, played by `cyclops.sfx` for the moments an oscillator has nothing to say: the box
booting, the panel coming up, his face being pressed, something appearing on the glass, and the
steel cover winding across his face. The synthesized cues beside them (`sfx.CUES`) are still numpy and still the default;
these are the exceptions, listed in `sfx.SAMPLES`.

They are **converted**, not masters. The originals are 96 kHz / 24-bit / stereo and live in
`sounds/` at the repo root, which is not deployed. These are cut to what the Pi's PipeWire sink
actually runs at — `s16le 1ch 48000Hz` after upmix — so nothing resamples on a board that
overheats:

```sh
mkdir -p src/cyclops/assets/sounds
for f in sounds/*.wav; do
  ffmpeg -y -i "$f" \
    -af "volume=-3dB" -ac 1 -ar 48000 -sample_fmt s16 -dither_method triangular \
    "src/cyclops/assets/sounds/$(basename "$f")"
done
```

Three of those flags are decisions rather than defaults:

- **`-ac 1`.** The sink reports `analog-stereo`, but that is the dongle's two output pins and
  not a promise about what is soldered to them. `cyclops_eye_pressed` and
  `cyclops_system_boot_finished` have *negatively* correlated channels, so if one speaker sits
  on one pin a stereo asset would play the half of the sound the designer did not intend, and
  fail silently. Mono is duplicated to both pins by PipeWire and comes out whole on any wiring.
  The rest of the box is mono too — `cyclops.audio.Speaker` opens one channel.
- **`volume=-3dB`.** Not make-up gain: it is measured *after* the downmix. The loudest 50 ms of
  the busiest synthesized cue is 0.176 of full scale, and −3 dB lands these four at 0.155–0.206,
  within 1.4 dB of it, while keeping the relative balance between them. A uniform gain, so
  retuning is one number. `tests/test_sfx.py` holds the bar.
- **`-dither_method triangular`.** This is a 24→16-bit reduction, and dither costs nothing.

`sfx.load` insists on exactly `1ch / 16-bit / 48000 Hz` — a file re-cut at the wrong rate is
turned into silence with a line on stderr rather than played at the wrong speed, and
`tests/test_sfx.py` is what catches it before it ships.

## The cues that arrived on their own

`cyclops_iris_open.wav`, `cyclops_iris_close.wav`, `cyclops_locked_in.wav`,
`cyclops_gears.wav` and `cyclops_button_pressed.wav` came one at a time rather than out of the
pack above, and are cut by
`tools/cut_cues.py`, whose `CUTS` table is the whole of what is per-cue: a master, whether the
cue is that master backwards, how much faster it is played, and how much of the front of it is
kept.

The two iris cues are one master played both ways. The cover opening and the cover closing are
the same mechanism running two directions, and there is only one recording of it, so the close is
the open reversed - which is what that mechanism would actually sound like and costs nothing to
be certain of.

`cyclops_iris_open.wav` is the one cue here with a length another module depends on:
`eye.COVER_OPEN_S` is set to it to the sample, so the lid on the panel takes exactly as long to
wind clear as the sound of it does. `tests/test_sfx.py` holds the two together - re-cut this and
that constant has to move with it.

`cyclops_locked_in.wav` is the bolt at the far end of the *open* - the blades reaching the rim
and stopping, which is the one movement on this panel that arrives somewhere. Sounded
`eye.COVER_OPEN_S - kiosk.LOCK_LEAD_S` after the wake begins rather than on any event, because
nothing on the box is told the cover has arrived; `cyclops.kiosk.Kiosk._render` is the only clock
that knows. The lead is the quiet tail on `cyclops_iris_open.wav`: the mechanism in that recording
runs out at 1.15 s and the file goes on to 1.4542 s, so a bolt at the true arrival lands after a
third of a second of near-silence and is heard as a second event. It cuts that tail instead.

It is cut at `speed=5.0`. The master is a 1.46 s clunk with a long ring under it, which is a
vault door; this lid is a set of blades the size of a coin. 5x is a quarter of a second of hit,
better than two octaves up. It was walked there on the panel - 1.25x read as a door swinging shut
elsewhere in the building, and 2.5x and 3.5x were each still too low. Resampling this hard would
normally alias; here there is nothing to fold, the master having 0.07% of its energy over 4 kHz.

`cyclops_gears.wav` is gears turning over: a long press landing, sounded once and from the
button's own thread (`cyclops.kiosk.Kiosk._toggle_session`). Both directions - starting a session
and ending one are the same mechanism engaging, and what tells them apart is the lid that
follows, which is the half you can watch. It was called `connecting` while it only marked the
start. It replaced a synthesized pair of blips that looped until the session arrived, and it does
not repeat: a mechanism you hear start and then stop has done its work, and one that keeps going
is stuck.

`cyclops_button_pressed.wav` is the rising note under a finger on the shutter button. It is the
one cue here whose length is not its master's: it has a window to fill, and the window is
`LONG_PRESS_S - PRESS_GRACE_S` - 0.6 s, between a finger settling and the hold landing. Move
either constant and this has to be cut again, or the rise stops arriving where the gears start.

Its master is not the eight-second swell it looks like. All of the rise is in the first 0.75 s -
20 dB, and a spectral centroid climbing from 1.5 kHz to 9 - and the seven seconds after that are
a flat bright bed, so a window taken from anywhere but the front is a drone rather than a rise.
The cut is `head_s=0.6` and nothing else: after the head trim the rise runs 0.62 s, so 0.6 s of
it at its own rate is still climbing when the gears take over. A shorter window has to have the
arc resampled into it - `speed` is that knob, and at the 0.5 s window this cue had first it took
1.25x. What ships rises 20 dB across its 0.6 s.

The masters are kept, in `sounds/`, and re-running the tool is the cut:

```sh
uv run python tools/cut_cues.py                  # all of them
uv run python tools/cut_cues.py --only iris_open # just one
```

It does not use the ffmpeg line above, for the same reason `voice_clips.py` does not: these
have to land inside the two bands `tests/test_sfx.py` measures rather than on a fixed dB figure,
and a uniform `-3dB` is for material that was mastered together, which this was not. It arrived
on its own at 44.1 kHz, so ffmpeg still does the decode, the downmix and the resample - a
resample is the one step here that numpy would alias - and everything after it is numpy.

The silence at both ends is trimmed **before** the reversal, and that ordering is the point: a
lead-in on the open is a tail on the close, and a cue that ends in a quarter-second of nothing is
one the panel has moved past by the time it finishes. The master carried 0.22 s of it.

Provenance is not recorded here - these masters were handed over as single files rather than
sourced from the pack below, so if this repo ever gains a public remote they want checking
alongside those.

## The ten voice samples

`voice_*.wav` are a different kind of thing from the five above and are cut by a different
route. They are one sentence — "I'm Cyclops. Show me what you're working on." — spoken by each
of the ten voices the Realtime API offers, played by the VOICE stepper on the settings screen so
that choosing between them is done by ear (`cyclops.voice`, `cyclops.kiosk._sync_voice`).

No masters are kept. The five designed cues came from a sample pack and could never be made
again; these are model output from one line of text and one model name, both of which are
written down in `tools/voice_clips.py`. Re-running that is the master:

```sh
uv run python tools/voice_clips.py            # all ten
uv run python tools/voice_clips.py --only cedar
```

It does the cut itself, in numpy, rather than handing off to the ffmpeg line above — what these
have to hit is not a dB figure but the two bands `tests/test_sfx.py` measures, and measuring
exactly what the test measures is the only way to be sure of landing inside them. Every clip is
matched to the same loudest-50 ms figure as the beeps, which matters more here than it does for
a cue: ten voices heard one after another are being *compared*, and one that is merely louder
than the one before it sounds better than it is.

## Licence

Sourced from a **Zenhiser** sample pack (`Transformer FX 2`); the artist and copyright tags are
left intact in the converted files rather than stripped. Zenhiser's terms permit use in
productions but prohibit redistributing the samples as samples, so if this repo ever gains a
public remote or a distributed wheel, these four files are the thing to reconsider.

The ten `voice_*.wav` are not Zenhiser material and are not covered by any of that: they
are OpenAI model output, generated here, and are governed by the API terms the rest of this
box already runs under.
