#!/usr/bin/env python3
"""Cut the designed cues that arrive on their own, one master at a time.

    uv run python tools/cut_cues.py                  # all of CUTS, into src/cyclops/assets/sounds/
    uv run python tools/cut_cues.py --only iris_open
    uv run python tools/cut_cues.py --keep 0.004     # ...with a different silence floor

The four cues from the sample pack were mastered together and are converted by the one ffmpeg
line in ``assets/sounds/NOTICE.md``. These were not: each turned up by itself, at its own rate
and its own level, so each has to be measured and placed rather than nudged by a fixed number of
dB. :data:`CUTS` is the whole of what is per-cue - a master, whether the cue is that master
backwards, how much faster it is played, and how much of the front of it is kept.

The last two are for the one cue with a window to fill rather than a length of its own. The rise
under the button runs from the moment a finger settles to the moment the hold lands, and a cue
that has to arrive at the end of a window has to be *fitted* to it: taken from the front, where
the rise actually is, and sped up until the arc finishes inside it.

The iris is the reason the reverse is here. A diaphragm winding open and a diaphragm winding shut
are the same mechanism running two directions and there is only one recording of it, so the close
*is* the open reversed - which is what the machine itself would sound like and costs nothing to
be certain of. The reversal is also why the ends are trimmed first: a lead-in of silence on the
open becomes a tail of silence on the close, and a cue that ends in a quarter-second of nothing
is one the panel has already moved past by the time it finishes.

Two routes into one file, and the split is deliberate. ffmpeg decodes, downmixes and resamples -
the master is 44.1 kHz and the sink runs at 48, and a resample is the one step here where doing
it by hand in numpy would alias. Everything after that is numpy, for the reason
``tools/voice_clips.py`` gives: what these have to hit is not a dB figure but the two bands
``tests/test_sfx.py`` measures, and measuring exactly what the test measures is the only way to
be sure of landing inside them. The uniform ``volume=-3dB`` in ``assets/sounds/NOTICE.md`` is for
material already mastered together; this arrived on its own.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import wave
from pathlib import Path
from typing import NamedTuple

import numpy as np

from cyclops.sfx import SAMPLE_HZ, SOUNDS, _envelope

MASTERS = Path(__file__).resolve().parent.parent / "sounds"

class Cut(NamedTuple):
    master: str
    backwards: bool = False  # the cue is that master played end to end backwards
    speed: float = 1.0  # played this much faster, pitch rising with it, as a wind-up does
    head_s: float = 0.0  # keep only this much of the front of it
    hpf_hz: float = 0.0  # roll off everything under this, in the cut's own Hz rather than the
    # master's - see `decode`, which divides by `speed` to get there. What it is for is the one
    # speaker this box has: a 28 mm cone on a plastic case cannot make 200 Hz and tries anyway,
    # so a cue with its body down there arrives as the case buzzing rather than as the cue.
    loud: float = 0.0  # the loudest 50 ms this one is cut to; TARGET_RMS when left at zero. A
    # per-cue number because "as loud as the others" is right for a cue that is an event and
    # wrong for one that is punctuation on the end of another cue.


CUTS: dict[str, Cut] = {
    "iris_open": Cut("cyclops_iris_open.wav"),
    "iris_close": Cut("cyclops_iris_open.wav", backwards=True),
    # Gears turning over: a long press landing, either direction. This was two blips and a long
    # gap for a long time; what it replaced them with is the same thing the iris says, which is
    # what the box sounds like from the outside while something inside it is moving. Once, not
    # looped - a mechanism you hear start and stop has done its work, one that keeps going is
    # stuck.
    "gears": Cut("cyclops_gears.wav"),
    # The bolt going home: sounded as the cover arrives, not while it travels. The master is a
    # 1.46 s clunk with a long ring under it, which is a vault door - and this lid is a set of
    # blades the size of a coin. Eight times faster is a sixth of a second of hit and three
    # octaves up, which is what puts it back at the size of the thing it comes off. Marco walked
    # it up 1.25 -> 2.5 -> 3.5 -> 5 -> 8 on the panel; the aliasing this would normally cost is
    # not there to pay, the master having 0.07% of its energy over 4 kHz.
    #
    # The other two numbers are what it is *against*. It is the only cue here that lands on the
    # end of another one rather than on a moment of its own, so it is cut far under the bar the
    # rest sit on - punctuation, not an event, and punctuation at the level of the sentence is a
    # second sentence. And the corner is above the body rather than under it: the whole low half
    # of this was a door, and what is wanted off a lid the size of a coin is the click at the
    # top of it. Both are free of each other by construction - the loudness is matched after the
    # filter, so moving the corner changes the tone and not the level.
    "locked_in": Cut("cyclops_locked_in.wav", speed=8.0, hpf_hz=1600.0, loud=0.025),
    # The rising note under a finger on the button. Unlike every other cue here its length is
    # not the master's: it has a window to fill and the window is LONG_PRESS_S - PRESS_GRACE_S,
    # the stretch between a finger settling and the hold landing.
    #
    # The master is not the eight-second swell it looks like. All of the rise is in its first
    # 0.75 s - 20 dB and a centroid climbing 1.5 kHz to 9 - and the seven seconds after it are a
    # flat bright bed, so a window taken from anywhere but the front is a drone. The whole cue
    # is therefore the front of it.
    #
    # No `speed` at this window, and that is measured rather than assumed: after the head trim
    # the rise runs 0.62 s, so 0.6 s of it at its own rate is still climbing when the gears take
    # over. A shorter window needs the arc resampled into it or it is cut off part way up - at
    # 0.5 s it took 1.25x.
    "button_pressed": Cut("cyclops_button_pressed.wav", head_s=0.6),
}

# The middle of the band tests/test_sfx.py holds the shipped cues to (0.05-0.25), and the same
# figure tools/voice_clips.py aims at - which is what makes the iris sit beside the shutter and
# the beeps rather than over them.
TARGET_RMS = 0.18
PEAK_CEILING = 0.85  # inside PEAK_FS's 0.90, with room for the int16 rounding
SILENCE_FS = 0.005  # below this the mechanism has not started, or has stopped
PAD_S = 0.01  # a hair either side, so the first click of the mechanism is not shaved off it


def loudest_50ms(pcm: np.ndarray) -> float:
    """Exactly what tests/test_sfx.py measures, so this cannot pass here and fail there."""
    window = int(0.05 * SAMPLE_HZ)
    usable = len(pcm) // window * window
    blocks = (pcm[:usable].astype(np.float64) / 32768).reshape(-1, window)
    return float(np.sqrt((blocks**2).mean(axis=1)).max())


def decode(path: Path, hpf_hz: float = 0.0) -> np.ndarray:
    """The master as float mono at the sink's rate. ffmpeg owns the resample; see the docstring.

    ...and the high-pass, for the same reason it owns the resample: a biquad hand-rolled here
    would be one more thing to be right about, and ffmpeg is already the decoder. *hpf_hz* is
    the corner in the master's Hz. :func:`main` is what turns a cue's corner into it, by
    dividing by the speed: :func:`faster` is a pure resample, so filtering at ``f/speed`` before
    it is exactly filtering at ``f`` after it, and the number in :data:`CUTS` can then mean the
    frequency you actually hear rather than one you have to work out.
    """
    chain = ["-ac", "1", "-ar", str(SAMPLE_HZ)]
    if hpf_hz:
        chain = ["-af", f"highpass=f={hpf_hz:.4f}"] + chain
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), *chain, "-f", "f32le", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(out, dtype="<f4").astype(np.float64)


def faster(samples: np.ndarray, factor: float) -> np.ndarray:
    """The same recording played *factor* times faster, pitch going up with it.

    Resampling rather than a time stretch, because a mechanism winding up faster is a real
    thing and a pitch-preserved one is not: what the ear reads off a rising note is the rate it
    is rising at, and holding the pitch while squeezing the clock takes that away.
    """
    if factor == 1.0:
        return samples
    want = int(len(samples) / factor)
    return np.interp(np.linspace(0, len(samples) - 1, want), np.arange(len(samples)), samples)


def trim(samples: np.ndarray, floor: float) -> np.ndarray:
    """Drop the silence at both ends, which the reversal would otherwise put at the wrong one."""
    loud = np.flatnonzero(np.abs(samples) > floor)
    if not len(loud):
        return samples
    pad = int(PAD_S * SAMPLE_HZ)
    return samples[max(0, loud[0] - pad) : min(len(samples), loud[-1] + pad + 1)]


def level(samples: np.ndarray, target: float = TARGET_RMS) -> np.ndarray:
    """Fade the ends, match the loudness, then back off if that would slam a peak.

    Both bands, in that order, because the RMS one is the one an ear actually compares - and the
    fade goes on before the measurement rather than after, so what is measured is what ships.

    The peak is brought inside full scale *before* anything is measured, and that is not tidiness.
    The measurement casts to int16 to be exactly what the test does, and a master that reaches
    past full scale - the gears arrive at 1.46, being a hot recording read as floats - wraps in
    that cast rather than clipping. Wrapped samples measure as noise, the scale comes out wrong,
    and the cue lands off the bar with nothing to show for it. This is a pure gain: it changes
    what the number is taken from, never what the cut sounds like.
    """
    samples = samples * _envelope(len(samples), SAMPLE_HZ)
    if (hot := np.abs(samples).max()) > 1.0:
        samples = samples / hot
    scale = target / max(loudest_50ms((samples * 32767).astype(np.int16)), 1e-9)
    peak = np.abs(samples).max() * scale
    if peak > PEAK_CEILING:
        scale *= PEAK_CEILING / peak
    return np.clip(samples * scale * 32767, -32768, 32767).astype(np.int16)


def write(path: Path, pcm: np.ndarray) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_HZ)
        wav.writeframes(pcm.tobytes())


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--only", action="append", help="just these cues (repeatable)")
    ap.add_argument("--keep", type=float, default=SILENCE_FS,
                    help="the silence floor, as a fraction of full scale")
    args = ap.parse_args()

    wanted = args.only or list(CUTS)
    if unknown := [name for name in wanted if name not in CUTS]:
        sys.exit(f"not cues: {', '.join(unknown)} (have {', '.join(CUTS)})")

    SOUNDS.mkdir(parents=True, exist_ok=True)
    # Levelled per cue rather than per master, so each one's own loudest 50 ms lands on the bar.
    # The iris pair differ: a mechanism's noisiest moment is not in the same place going each
    # way, and matching one of them leaves the other off it.
    for name in wanted:
        cut = CUTS[name]
        if not (src := MASTERS / cut.master).is_file():
            sys.exit(f"no master at {src}")
        samples = trim(faster(decode(src, cut.hpf_hz / cut.speed), cut.speed), args.keep)
        if cut.head_s:
            samples = samples[: int(cut.head_s * SAMPLE_HZ)]
        pcm = level(np.ascontiguousarray(samples[::-1] if cut.backwards else samples),
                    cut.loud or TARGET_RMS)
        path = SOUNDS / f"cyclops_{name}.wav"
        write(path, pcm)
        print(f"· {path.name} — {len(pcm) / SAMPLE_HZ:.2f}s, "
              f"loudest 50 ms {loudest_50ms(pcm):.3f}, peak {np.abs(pcm).max() / 32767:.3f}",
              flush=True)


if __name__ == "__main__":
    main()
