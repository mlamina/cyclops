# Sound on the Pi

**The speaker is an I2S amp on the GPIO header.** A NULLLAB NS4168 module — a class-D amplifier
with the DAC built into it, so no analogue ever crosses the board and there is no USB device to
enumerate, lose and re-enumerate. Five wires, and the module's own silkscreen names them:

| Module | Pi GPIO | Header pin |
| --- | --- | --- |
| G | GND | 39 |
| V | 5V | 4 |
| BCLK | GPIO18 (PCM_CLK) | 12 |
| LRCLK | GPIO19 (PCM_FS) | 35 |
| DIN | GPIO21 (PCM_DOUT) | 40 |

`deploy/install-speaker.sh` does the rest, once per Pi: it adds `dtparam=i2s=on` and
`dtoverlay=hifiberry-dac` to `/boot/firmware/config.txt` (keeping a dated backup), and installs
the mono sink described below. The card only exists after a reboot, and turns up as
`sndrpihifiberry`. It has **no ALSA mixer element** — the overlay presents a codec with no
controls — so PipeWire does the volume in software and `amixer` has nothing to show; `mixer.py`
is unaffected, because it always asked `pactl` about `@DEFAULT_SINK@` rather than the card.

**The supply voltage picks the channel, which is not a typo.** The NS4168 is mono and chooses
which of the two I2S channels it plays from the level on its CTRL pin — 0.9–1.15 V is left,
≥1.5 V is right. The module ties CTRL to a fixed 2.2k/1k divider off VCC, so 3.3 V feeds it
1.03 V and 5 V feeds it 1.56 V: **the rail you pick is the channel you get.** Take 5 V, for the
full 2.5 W. That same fixed divider is why the vendor's "tune the gain and filters in software"
does not apply to this board — the filter corner is set by pulsing CTRL, and CTRL is soldered to
a divider. The gain is what it is.

## Stereo has to be folded before it reaches it

Session recordings are stereo *on purpose* — `mux_command` puts `user.wav` on the left channel
and `agent.wav` on the right, so the two voices stay separable. Played back over a mono amp that
reads one channel and ignores the other, exactly one voice survives, and with VCC on 5 V it is
Cyclops': you hear the answers and none of the questions.

The DAC cannot be the thing that reconciles them. It is strictly two-channel — `aplay
--dump-hw-params` reports `CHANNELS: 2` and nothing else — so the fold happens one node upstream,
in `deploy/51-mono-speaker.conf`: a `module-loopback` mono sink named `mono_speaker`, which is the
default sink. Everything plays into a real MONO node, which sums L+R, and the loopback duplicates
that sum into both channels of the DAC. Inside it, `playback.props.audio.position` is `[ MONO ]`
and not `[ FL FR ]`, because a loopback maps its channels one to one — a two-channel output side
would feed FL and leave FR silent, which on an amp reading the other channel is no sound at all.

Three cheaper routes were tried against the hardware first. All three lose audio rather than
folding it, so don't re-try them:

* `audio.position = "MONO,MONO"` on the ALSA sink is accepted — `pactl` will even report
  `Channel Map: mono,mono` — and then passes both channels straight through. Duplicate target
  positions give PipeWire no matrix to build, so it falls back to identity.
* `audio.channels = 1` produces a channel called `aux0` rather than a mono one, and the mix into
  an unpositioned channel **drops** the right channel instead of summing it.
* `audio.position = "MONO"` alone is ignored outright; the map comes back `front-left,front-right`.

**Measure it at the DAC, not at the channel map**, since the map was the thing that lied. Record
the hardware sink's monitor while playing a tone that exists on one channel only, and read the
per-channel RMS back out:

```bash
ffmpeg -f lavfi -i "sine=f=440:duration=2" -af "pan=stereo|c0=c0|c1=0*c0" -y /tmp/left.wav
parecord -d alsa_output.platform-soc_107c000000_sound.stereo-fallback.monitor \
  --channels=2 --rate=48000 --format=s16le /tmp/m.wav &
pw-play /tmp/left.wav; kill %1
ffmpeg -i /tmp/m.wav -af astats=metadata=1 -f null - 2>&1 | grep -E "Channel:|RMS level dB"
```

Both channels equal, for a left-only *and* a right-only tone, is the pass. Before the fold a
left-only tone measured `L −24.1 dB, R −inf`; after it, both channels carry −29.6 dB, which is
the −6 dB an honest (L+R)/2 costs. Cyclops' own voice is untouched by any of it: mono content
measures −23.571656 dB on both channels either way, bit for bit. The loopback costs about 0.3% of
a core.

## Picking devices

A Pi has several and often no default mic, so set them explicitly:

```bash
uv run cyclops-devices          # lists inputs/outputs
```

Add to `.env`. On a Pi running PipeWire/PulseAudio the simplest robust choice is to route through
`pulse` — it resamples to whatever the speaker wants and follows the default sink/source, which is
how the amp above is found without naming it anywhere — and to force speaker mode, since an open
speaker leaks into the mic. Forcing it is not optional here: PortAudio reports the device through
the pulse plugin as plain `default`, so the name-matching in `output_is_speaker()` has nothing to
recognise and would guess headphones.

```ini
CYCLOPS_INPUT_DEVICE=pulse
CYCLOPS_OUTPUT_DEVICE=pulse
CYCLOPS_HALF_DUPLEX=1
```

Direct-hardware devices like `hw:4,0` are often locked to 48 kHz and reject cyclops's 24 kHz;
`pulse` avoids that. Then `uv run cyclops` works from the terminal.
