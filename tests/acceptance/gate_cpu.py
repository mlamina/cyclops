"""qa check for arc-eng/hq#561: what the idle ear costs the CPU, without touching the running kiosk.

Runs cyclops.ear's speech gate (level, flatness, WebRTC VAD) over 30 s of mic-shaped audio -
a mix of quiet room, hum and a voice-like signal - block by block, the way the audio thread does,
and prints the CPU time it took as a share of one core. The ear's real cost is this plus the
compressor and one extra capture stream in PipeWire.

Run: uv run python tests/acceptance/gate_cpu.py
"""

from __future__ import annotations

import time

import numpy as np

from cyclops.audio import BLOCK_FRAMES, SAMPLE_RATE, Compressor
from cyclops.ear import SpeechGate

SECONDS = 30
rng = np.random.default_rng(561)
t = np.arange(SECONDS * SAMPLE_RATE) / SAMPLE_RATE
signal = (
    200 * rng.standard_normal(t.size)  # room
    + 400 * np.sin(2 * np.pi * 60 * t)  # hum
    + 3000 * np.sin(2 * np.pi * 180 * t) * (np.sin(2 * np.pi * 3 * t) > 0)  # voice-ish bursts
)
pcm = np.clip(signal, -32768, 32767).astype(np.int16).tobytes()
blocks = [pcm[i:i + BLOCK_FRAMES * 2] for i in range(0, len(pcm), BLOCK_FRAMES * 2)]

gate = SpeechGate()
comp = Compressor(gain_db=20.0, compress=True)
start = time.process_time()
for b in blocks:
    gate.feed(comp.process(b))
used = time.process_time() - start
print(f"{len(blocks)} blocks ({SECONDS} s of audio) in {used:.3f} s CPU: "
      f"{100 * used / SECONDS:.2f}% of one core, {1e3 * used / len(blocks):.3f} ms a block")
