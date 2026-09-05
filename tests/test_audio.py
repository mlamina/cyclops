"""Levelling the microphone: what the compressor does to a quiet voice, and to a loud one.

No device anywhere - the thing under test takes a block of bytes and hands one back, which is
exactly how the audio thread uses it. Levels are read back as RMS in dBFS, because that is the
unit the compressor is written in and the one a recording is judged in.

Every measurement is taken from the *last* of several blocks fed in. That is not a workaround:
the gain is ramped across each block so it never steps, so a single block in isolation is
half-way through a change and says nothing about where the compressor settles.
"""

from __future__ import annotations

import math

import numpy as np

from cyclops.audio import BLOCK_FRAMES, BYTES_PER_FRAME, Compressor

FULL_SCALE = 32768.0
SILENCE = b"\x00" * (BLOCK_FRAMES * BYTES_PER_FRAME)


def tone(dbfs: float) -> bytes:
    """One block of a 200 Hz sine at the given RMS level, as PCM16."""
    amplitude = 10 ** (dbfs / 20) * FULL_SCALE * math.sqrt(2)
    phase = np.arange(BLOCK_FRAMES) * (2 * math.pi * 200 / 24_000)
    return (np.sin(phase) * amplitude).astype(np.int16).tobytes()


def level(block: bytes) -> float:
    """RMS of a block in dBFS, the way the compressor and every meter measure it."""
    samples = np.frombuffer(block, dtype=np.int16).astype(np.float64)
    rms = math.sqrt(float(np.mean(samples * samples)))
    return 20 * math.log10(max(rms, 1.0) / FULL_SCALE)


def settled(shaping: Compressor, dbfs: float, blocks: int = 40) -> float:
    """Feed the same tone until the gain stops moving; return where the output lands."""
    out = b""
    for _ in range(blocks):
        out = shaping.process(tone(dbfs))
    return level(out)


# ---- the point of the exercise ----


def test_a_quiet_voice_comes_up_by_the_makeup() -> None:
    """Under the threshold nothing is compressed, so the makeup is all of it."""
    assert settled(Compressor(), -55.0) == pytest_approx(-55.0 + Compressor.MAKEUP_DB)


def test_a_loud_voice_is_pulled_down_rather_than_up() -> None:
    """Well over the threshold the compression outweighs the makeup, which is what pays for it."""
    assert settled(Compressor(), -12.0) < -12.0


def test_the_range_between_them_shrinks() -> None:
    """A real session's fifth and ninety-ninth percentile, 32 dB apart, come out half that.

    The two numbers are measured, not invented - see :class:`~cyclops.audio.Compressor`.
    """
    quiet, loud = settled(Compressor(), -54.0), settled(Compressor(), -22.0)
    assert loud - quiet < 16.0


def test_nothing_clips() -> None:
    """Whatever the compressor asks for, the block's own peak has the last word."""
    shaping = Compressor(gain_db=20.0)
    peak = 0
    for _ in range(40):
        out = np.frombuffer(shaping.process(tone(-6.0)), dtype=np.int16)
        peak = max(peak, int(np.max(np.abs(out))))
    assert peak < FULL_SCALE


# ---- the knobs ----


def test_gain_alone_is_a_plain_multiplier() -> None:
    """With compression off, +6 dB is +6 dB and nothing else has an opinion."""
    shaping = Compressor(gain_db=6.0, compress=False)
    assert settled(shaping, -40.0) == pytest_approx(-34.0)


def test_turning_both_off_leaves_the_audio_alone() -> None:
    """Byte for byte: a session that asked for nothing gets exactly what the mic heard."""
    shaping = Compressor(gain_db=0.0, compress=False)
    block = tone(-20.0)
    assert shaping.process(block) == block


def test_silence_stays_silence() -> None:
    """An empty room must not be lifted into a hiss, and must not divide by its own zero."""
    shaping = Compressor()
    for _ in range(10):
        assert shaping.process(SILENCE) == SILENCE


def pytest_approx(value: float) -> object:
    """A dB comparison is never exact: quantising to int16 moves the last fraction of one."""
    import pytest

    return pytest.approx(value, abs=0.3)
