#!/usr/bin/env python3
"""Cut the ten voice samples the panel plays, one per voice the Realtime API offers.

Which voice you want to be answered by is not a thing that can be read off a list of names, so
the stepper on the settings screen plays one every time it is touched. These are what it plays.

    uv run python tools/voice_clips.py              # all ten, into src/cyclops/assets/sounds/
    uv run python tools/voice_clips.py --only cedar # re-cut one
    uv run python tools/voice_clips.py --line "..." # say something else

The masters are not kept. Unlike the designed cues beside them - which came from a sample pack
and could never be made again - these are model output from one line of text and one model name,
and both are written down right here. Re-running this is the master.

The cut is done in numpy rather than by the ffmpeg line in ``assets/sounds/NOTICE.md``, because
what these have to hit is not a dB figure but the two bands ``tests/test_sfx.py`` measures, and
measuring exactly what the test measures is the only way to be sure of landing inside them. The
loudness match matters more here than for the cues: ten voices played one after another are being
compared, and a voice that is merely louder than the one before it sounds better than it is.
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np
from openai import OpenAI

from cyclops import voice
from cyclops.config import load_settings
from cyclops.sfx import SAMPLE_HZ, SOUNDS, _envelope

MODEL = "gpt-4o-mini-tts"
# Long enough to hear a pace and a warmth, short enough that stepping through ten of them is a
# thing you do standing up. In his own voice, because that is what you are choosing.
LINE = "I'm Cyclops. Show me what you're working on."
PCM_HZ = 24_000  # what response_format="pcm" is: mono, 16-bit, little-endian, always this rate

# The middle of the band tests/test_sfx.py holds the shipped cues to (0.05-0.25), and within a
# hair of the 0.176 the busiest synthesized cue measures. "As loud as the beeps" for broadband
# material, which is what that test's 50 ms window exists to say.
TARGET_RMS = 0.18
PEAK_CEILING = 0.85  # inside PEAK_FS's 0.90, with room for the int16 rounding
SILENCE_FS = 0.005  # below this the model is not talking yet, or has stopped


def loudest_50ms(pcm: np.ndarray) -> float:
    """Exactly what tests/test_sfx.py measures, so this cannot pass here and fail there."""
    window = int(0.05 * SAMPLE_HZ)
    usable = len(pcm) // window * window
    blocks = (pcm[:usable].astype(np.float64) / 32768).reshape(-1, window)
    return float(np.sqrt((blocks**2).mean(axis=1)).max())


def trim(samples: np.ndarray) -> np.ndarray:
    """Drop the lead-in and the tail. The model leaves a beat of room at both ends, and ten
    clips each with a quarter-second of nothing in front is a stepper that feels broken."""
    loud = np.flatnonzero(np.abs(samples) > SILENCE_FS)
    if not len(loud):
        return samples
    pad = int(0.02 * PCM_HZ)  # a hair either side, so no consonant is cut off its own word
    return samples[max(0, loud[0] - pad) : min(len(samples), loud[-1] + pad + 1)]


def cut(raw: bytes) -> np.ndarray:
    """Model output to one shipped cue: trimmed, resampled, faded, and matched for loudness."""
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768
    samples = trim(samples)
    # 24k to 48k is exactly 2x, so this is an interpolation with nothing to alias: every second
    # output sample is an input sample and the ones between are the midpoints.
    grid = np.arange(len(samples) * SAMPLE_HZ // PCM_HZ) * PCM_HZ / SAMPLE_HZ
    samples = np.interp(grid, np.arange(len(samples)), samples)
    samples *= _envelope(len(samples), SAMPLE_HZ)
    # Loudness first, then back off if that would slam a peak. Both bands, in that order,
    # because the RMS one is the one an ear actually compares.
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
    ap.add_argument("--only", action="append", help="just these voices (repeatable)")
    ap.add_argument("--line", default=LINE, help="what each of them says")
    args = ap.parse_args()

    wanted = args.only or list(voice.VOICES)
    if unknown := [name for name in wanted if name not in voice.NAMES]:
        sys.exit(f"not voices: {', '.join(unknown)}")

    client = OpenAI(api_key=load_settings().api_key)
    SOUNDS.mkdir(parents=True, exist_ok=True)
    for name in wanted:
        raw = client.audio.speech.create(
            model=MODEL, voice=name, input=args.line, response_format="pcm"
        ).content
        pcm = cut(raw)
        path = SOUNDS / f"voice_{name}.wav"
        write(path, pcm)
        print(
            f"· {path.name} — {len(pcm) / SAMPLE_HZ:.2f}s, "
            f"loudest 50 ms {loudest_50ms(pcm):.3f}, peak {np.abs(pcm).max() / 32767:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
