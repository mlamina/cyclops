"""A stopped speaker is silent, and says so (arc-eng/hq#572, qa round 2).

Stopping a session while Cyclops talks leaves the rest of the answer unplayed. Nothing will ever
drain it, and the session's video can take a while to save, so a speaker that went on counting that
audio as audible kept the ear deaf and dropping what he said for the whole close.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from cyclops import audio, ear, ui
from cyclops.audio import AUDIBLE_TAIL_S, BLOCK_FRAMES, BYTES_PER_FRAME
from cyclops.config import Settings

BLOCK = b"\x10\x10" * BLOCK_FRAMES


class Stream:
    def __init__(self, **_kw: object) -> None:
        pass

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
def no_sound_card(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio.sd, "RawOutputStream", Stream)
    monkeypatch.setattr(ear.sd, "RawInputStream", Stream)
    monkeypatch.setattr(ear, "pin_input_source", lambda device: "room")


def played(speaker: audio.Speaker) -> None:
    """One block goes out through the real callback, which starts the tail clock as on a device."""
    out = bytearray(BLOCK_FRAMES * BYTES_PER_FRAME)
    speaker._callback(out, BLOCK_FRAMES, None, None)  # noqa: SLF001


def test_what_never_played_stops_counting_once_the_speaker_stops() -> None:
    speaker = audio.Speaker()
    speaker.feed(BLOCK * 50)
    played(speaker)
    assert speaker.is_audible
    speaker.stop()
    assert not speaker.has_unplayed_audio


def test_the_rooms_tail_is_still_covered_right_after_a_stop() -> None:
    speaker = audio.Speaker()
    speaker.feed(BLOCK * 50)
    played(speaker)
    speaker.stop()
    assert speaker.is_audible, "what already left the speaker is still sounding in the room"
    time.sleep(AUDIBLE_TAIL_S + 0.05)
    assert not speaker.is_audible


def test_a_late_feed_cant_make_a_stopped_speaker_audible_again() -> None:
    speaker = audio.Speaker()
    speaker.stop()
    speaker.feed(BLOCK * 50)  # the last of an answer arriving as the session comes down
    assert not speaker.has_unplayed_audio
    assert not speaker.is_audible


def test_a_speaker_stopped_with_nothing_left_changes_nothing() -> None:
    speaker = audio.Speaker()
    speaker.feed(BLOCK)
    played(speaker)
    assert not speaker.has_unplayed_audio
    speaker.stop()
    assert speaker.is_audible  # the tail


def test_what_he_says_after_a_stop_mid_answer_is_kept() -> None:
    controller = ui.SessionController(Settings(api_key=""), frames=None, entrypoint="test")
    speaker = audio.Speaker()
    speaker.feed(BLOCK * 50)
    speaker.stop()
    controller._speaker = speaker  # noqa: SLF001 - as it is until the thread's last moments
    time.sleep(AUDIBLE_TAIL_S + 0.05)
    assert not controller.speaking

    listening = ear.Ear(device=None, gain_db=0.0, compress=False)
    listening.loud = lambda: controller.speaking
    words = np.full((BLOCK_FRAMES, 1), 3000, dtype=np.int16)
    for _ in range(50):  # one second of him talking, during the close
        listening._callback(words, BLOCK_FRAMES, None, None)  # noqa: SLF001
    kept = listening.take()
    listening.close()
    assert len(kept) > 0
