# The shipped sound cues

Four designed cues, played by `cyclops.sfx` for the moments an oscillator has nothing to say:
the box booting, the panel coming up, his face being pressed, and something appearing on the
glass. The synthesized cues beside them (`sfx.CUES`) are still numpy and still the default;
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
