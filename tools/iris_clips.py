#!/usr/bin/env python3
"""Cut the two iris cues - the cover opening, and the same sound backwards for it closing.

    uv run python tools/iris_clips.py            # both, into src/cyclops/assets/sounds/
    uv run python tools/iris_clips.py --keep 0.004   # ...with a different silence floor

One master, two cues. A diaphragm winding open and a diaphragm winding shut are the same
mechanism running two directions, and the sample pack only has the one - so the close *is* the
open reversed, which is what the machine itself would sound like and costs nothing to be sure of.
The reversal is why the ends have to be trimmed first: a lead-in of silence on the open becomes a
tail of silence on the close, and a cue that ends in a quarter-second of nothing is one the panel
has already moved past by the time it finishes.

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

import numpy as np

from cyclops.sfx import SAMPLE_HZ, SOUNDS, _envelope

MASTER = Path(__file__).resolve().parent.parent / "sounds" / "cyclops_iris_open.wav"

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


def trim(samples: np.ndarray, floor: float) -> np.ndarray:
    """Drop the silence at both ends, which the reversal would otherwise put at the wrong one."""
    loud = np.flatnonzero(np.abs(samples) > floor)
    if not len(loud):
        return samples
    pad = int(PAD_S * SAMPLE_HZ)
    return samples[max(0, loud[0] - pad) : min(len(samples), loud[-1] + pad + 1)]


def level(samples: np.ndarray) -> np.ndarray:
    """Fade the ends, match the loudness, then back off if that would slam a peak.

    Both bands, in that order, because the RMS one is the one an ear actually compares - and the
    fade goes on before the measurement rather than after, so what is measured is what ships.
    """
    samples = samples * _envelope(len(samples), SAMPLE_HZ)
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
    ap.add_argument("--master", type=Path, default=MASTER)
    ap.add_argument("--keep", type=float, default=SILENCE_FS,
                    help="the silence floor, as a fraction of full scale")
    args = ap.parse_args()

    if not args.master.is_file():
        sys.exit(f"no master at {args.master}")
    opened = trim(decode(args.master), args.keep)
    SOUNDS.mkdir(parents=True, exist_ok=True)
    # Levelled separately rather than once and reversed, so each cue's own loudest 50 ms lands on
    # the bar. The two differ: a mechanism's noisiest moment is not in the same place going each
    # way, and matching one of them leaves the other off it.
    for name, samples in (("open", opened), ("close", opened[::-1])):
        pcm = level(np.ascontiguousarray(samples))
        path = SOUNDS / f"cyclops_iris_{name}.wav"
        write(path, pcm)
        print(f"· {path.name} — {len(pcm) / SAMPLE_HZ:.2f}s, "
              f"loudest 50 ms {loudest_50ms(pcm):.3f}, peak {np.abs(pcm).max() / 32767:.3f}",
              flush=True)


if __name__ == "__main__":
    main()
