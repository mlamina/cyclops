"""qa check for arc-eng/hq#561: would Cyclops's own sounds open a session if the ear were not deaf?

Feeds every bundled cue (src/cyclops/assets/sounds/*.wav) through the speech gate at the
mic rate, with no deaf guard, and prints one line per file: TRIGGERS or quiet. A trigger
here means the kiosk's deaf-while-a-cue-sounds guard is what keeps that sound from starting
a session. Played straight in, not through a speaker and lav, so it is the worst case.

Run: uv run python tests/acceptance/gate_on_own_sounds.py [sounds-dir]
The licensed cyclops_*.wav cues are not in git; pass a folder that has them.
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

from cyclops.audio import BLOCK_FRAMES, SAMPLE_RATE
from cyclops.ear import SpeechGate

SOUNDS = Path(__file__).resolve().parents[2] / "src" / "cyclops" / "assets" / "sounds"


def load(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"{width * 8}-bit")
    samples = np.frombuffer(raw, dtype=np.int16).reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE:
        n = int(len(samples) * SAMPLE_RATE / rate)
        samples = np.interp(np.linspace(0, len(samples) - 1, n), np.arange(len(samples)), samples)
    return samples.astype(np.int16)


def main() -> int:
    triggered = 0
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else SOUNDS
    for path in sorted(folder.glob("*.wav")):
        try:
            pcm = load(path)
        except Exception as exc:  # noqa: BLE001
            print(f"skip      {path.name}: {exc}")
            continue
        gate = SpeechGate()
        hit = None
        for i in range(0, len(pcm) - BLOCK_FRAMES + 1, BLOCK_FRAMES):
            if gate.feed(pcm[i:i + BLOCK_FRAMES].tobytes()):
                hit = i / SAMPLE_RATE
                break
        if hit is None:
            print(f"quiet     {path.name}")
        else:
            triggered += 1
            print(f"TRIGGERS  {path.name} at {hit:.2f} s")
    print(f"{triggered} file(s) would start a session without the deaf guard")
    return 0


if __name__ == "__main__":
    sys.exit(main())
