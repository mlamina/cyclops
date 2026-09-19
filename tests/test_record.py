"""At what size a session's video is encoded, and what reaches its microphone track.

None of it needs ffmpeg, a camera or a panel. The size the encoder is opened at is arithmetic on a
frame's shape, and the mic hook is handed a stand-in track that only remembers what it was given.
What is being guarded here is a pair of silent failures: a recording that is quietly resampled to
mush, and a microphone track with holes cut in it. What the video is *of* - the screen, or the
camera when the screen cannot be had - is ``tests/test_screen.py``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cyclops import record
from cyclops.record import SessionRecorder

PANEL = (800, 480)  # the official 7" panel, and what a screen recording is
CAMERA = (640, 480)  # the endoscope, and what a camera recording is
NOTHING = SimpleNamespace(frame=lambda: None)  # a source that has not produced a frame yet


def frame(width: int, height: int):
    return np.zeros((height, width, 3), dtype=np.uint8)


def size_for(frame_size: tuple[int, int], width: int) -> tuple[int, int]:
    """What the encoder would be opened at for this source and this ``CYCLOPS_RECORD_WIDTH``."""
    recorder = SessionRecorder(NOTHING, Path("unused"), width=width)
    return recorder._output_size(frame(*frame_size))


# ---- the size the encoder is opened at ----


def test_the_panel_is_recorded_pixel_for_pixel() -> None:
    """The default records the source's own width, so nothing resamples the chrome to mush."""
    assert size_for(PANEL, width=0) == PANEL
    assert size_for(CAMERA, width=0) == CAMERA


def test_a_width_is_a_ceiling_and_never_an_upscale() -> None:
    assert size_for((1280, 720), width=800) == (800, 450)
    assert size_for(CAMERA, width=800) == CAMERA, "800 must not blow a 640 frame up"
    assert size_for(PANEL, width=640) == (640, 384), "what setting a width still costs you"


def test_both_dimensions_come_back_even_for_yuv420p() -> None:
    """An odd dimension is rejected by the pixel format, not by us, and only at encode time."""
    for width in (0, 1281):
        out_w, out_h = size_for((1281, 721), width=width)
        assert out_w % 2 == 0 and out_h % 2 == 0


# ---- the microphone track ----


class Track:
    """A stand-in for the recorder's ``_Track``: it remembers, and it does no I/O."""

    def __init__(self) -> None:
        self.written: list[bytes] = []

    def append(self, pcm: bytes, at: float) -> None:
        self.written.append(pcm)


def recorder_with_a_track() -> tuple[SessionRecorder, Track]:
    recorder = SessionRecorder(NOTHING, Path("unused"))
    track = Track()
    recorder._user = track  # what start() would have opened, minus the wave file
    return recorder, track


def loud() -> bytes:
    """One block of speaker output well over ECHO_FLOOR - Cyclops mid-word."""
    return b"\x00\x20" * 480  # 0x2000 = 8192, an ordinary speaking level


def test_every_block_the_microphone_heard_reaches_the_track() -> None:
    """With the speaker quiet, whatever the echo guard did with a block the recording keeps it.

    The guard is the model's business: it stops Cyclops answering his own voice, and it holds
    the mic shut across a whole utterance to do it. The recording used to write that verdict
    down, and three quarters of a session's mic track came out digital silence.
    """
    recorder, track = recorder_with_a_track()
    blocks = [bytes([n]) * 960 for n in range(1, 6)]
    for block in blocks:
        recorder.on_mic_block(block)
    assert track.written == blocks, "no block dropped, none replaced with silence, none reordered"


def test_a_block_is_written_on_the_callback_that_brought_it() -> None:
    """There is no delay line any more: nothing is held back waiting to be un-muted."""
    recorder, track = recorder_with_a_track()
    recorder.on_mic_block(b"\x01\x02" * 480)
    assert len(track.written) == 1


def test_the_microphone_is_cut_while_the_speaker_is_sounding() -> None:
    """Otherwise the left channel is a second, room-coloured copy of the right one."""
    recorder, track = recorder_with_a_track()
    recorder._agent = Track()  # the other half of the tap, which is what notices the speaker
    recorder.on_speaker_block(loud())
    recorder.on_mic_block(b"\x11" * 960)
    assert track.written == [bytes(960)], "silenced, and still exactly one block long"


def test_silence_out_of_the_speaker_is_not_an_echo() -> None:
    """The speaker callback zero-fills, so it runs constantly. Only real audio closes the mic."""
    recorder, track = recorder_with_a_track()
    recorder._agent = Track()
    recorder.on_speaker_block(bytes(960))  # between utterances: handed over, but silent
    recorder.on_mic_block(b"\x11" * 960)
    assert track.written == [b"\x11" * 960]


def test_the_room_comes_back_once_the_sound_has_died_away() -> None:
    """The hold covers the room's tail; after it, the gaps inside a session are room again."""
    recorder, track = recorder_with_a_track()
    recorder._agent = Track()
    recorder.on_speaker_block(loud())
    recorder._agent_at -= record.ECHO_HOLD_S + 0.01  # as if that block were a moment ago
    recorder.on_mic_block(b"\x11" * 960)
    assert track.written == [b"\x11" * 960]


def test_a_broken_track_disables_the_recording_rather_than_the_session() -> None:
    """An audio thread may not raise: it sets the flag, and the writer thread reports it."""
    recorder, _ = recorder_with_a_track()
    recorder._user = object()  # no append(); the next block is going to hurt
    recorder.on_mic_block(b"\x00" * 960)
    assert recorder.failed
    recorder.on_mic_block(b"\x00" * 960)  # and it stays quiet about it afterwards
