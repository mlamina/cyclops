#!/usr/bin/env python3
"""Cut the designed cues that arrive on their own, one master at a time.

    uv run python tools/cut_cues.py                  # all of CUTS, into src/cyclops/assets/sounds/
    uv run python tools/cut_cues.py --only iris_open
    uv run python tools/cut_cues.py --keep 0.004     # ...with a different silence floor

The four cues from the sample pack were mastered together and are converted by the one ffmpeg
line in ``assets/sounds/NOTICE.md``. These were not: each turned up by itself, at its own rate
and its own level, so each has to be measured and placed rather than nudged by a fixed number of
dB. :data:`CUTS` is the whole of what is per-cue - a master, and whether the cue is that master
backwards.

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
    loops: bool = False  # ...and it is played on a loop, so its own tail is the gap between
    # repeats and is left where it is. Trimming it would butt the mechanism against itself and
    # turn a cue you are meant to ignore for several seconds into a drone in a small workshop.


CUTS: dict[str, Cut] = {
    "iris_open": Cut("cyclops_iris_open.wav"),
    "iris_close": Cut("cyclops_iris_open.wav", backwards=True),
    # Gears turning over while the socket comes up. This was two blips and a long gap for a long
    # time; what it replaced them with is the same thing the iris says, which is what the box
    # sounds like from the outside while something inside it is moving.
    "connecting": Cut("cyclops_connecting.wav", loops=True),
    # The rising note under a finger on the button, started on the way down and cut dead on the
    # way up (cyclops.button). Not trimmed at the tail for the usual reason - a cue nobody hears
    # the end of has no end to tidy - but at the head for a sharper one: what is in front of the
    # first sample is the delay between pressing a button and hearing that you did.
    "button_pressed": Cut("cyclops_button_pressed.wav"),
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


def decode(path: Path) -> np.ndarray:
    """The master as float mono at the sink's rate. ffmpeg owns the resample; see the docstring."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SAMPLE_HZ),
         "-f", "f32le", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(out, dtype="<f4").astype(np.float64)


def trim(samples: np.ndarray, floor: float, *, tail: bool = True) -> np.ndarray:
    """Drop the silence at both ends, which the reversal would otherwise put at the wrong one.

    ``tail=False`` keeps whatever is after the last loud sample: on a looping cue that is not
    silence to be tidied away, it is the rest between one turn of the mechanism and the next.
    """
    loud = np.flatnonzero(np.abs(samples) > floor)
    if not len(loud):
        return samples
    pad = int(PAD_S * SAMPLE_HZ)
    end = len(samples) if not tail else min(len(samples), loud[-1] + pad + 1)
    return samples[max(0, loud[0] - pad) : end]


def level(samples: np.ndarray) -> np.ndarray:
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
    scale = TARGET_RMS / max(loudest_50ms((samples * 32767).astype(np.int16)), 1e-9)
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
        samples = trim(decode(src), args.keep, tail=not cut.loops)
        pcm = level(np.ascontiguousarray(samples[::-1] if cut.backwards else samples))
        path = SOUNDS / f"cyclops_{name}.wav"
        write(path, pcm)
        print(f"· {path.name} — {len(pcm) / SAMPLE_HZ:.2f}s, "
              f"loudest 50 ms {loudest_50ms(pcm):.3f}, peak {np.abs(pcm).max() / 32767:.3f}",
              flush=True)


if __name__ == "__main__":
    main()
