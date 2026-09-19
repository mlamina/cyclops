"""The interrupt switch: what it settles, and what it actually does to the microphone.

Two halves, matching the two the feature has. The note is what the settings screen writes and
every session reads; the guard is the thing in the audio path that either lets you cut in or
holds the mic shut until Cyclops has finished. Neither half needs a device: the guard is handed
a stand-in speaker, because what is being checked is which blocks come back out of it.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from cyclops import barge
from cyclops.audio import BLOCK_FRAMES, BYTES_PER_FRAME, EchoGuard
from cyclops.config import Settings

SILENCE = b"\x00" * (BLOCK_FRAMES * BYTES_PER_FRAME)
ENV = Settings(api_key="", barge_in_db=8.0)  # a box whose .env says nothing about barge-in
ENV_OFF = replace(ENV, barge_in_db=None)  # ...and one whose .env turned it off


# ---- the note ----


def test_no_note_leaves_the_environment_to_decide(tmp_path: Path) -> None:
    note = tmp_path / "barge-in"
    assert barge.requested(note) is None
    assert barge.margin_db(ENV, note) == 8.0
    assert barge.margin_db(ENV_OFF, note) is None


def test_the_switch_survives_a_round_trip(tmp_path: Path) -> None:
    note = tmp_path / "cache" / "barge-in"  # the directory does not exist yet, as at first boot
    barge.request(False, note)
    assert barge.requested(note) is False
    barge.request(True, note)
    assert barge.requested(note) is True


def test_off_beats_the_environment(tmp_path: Path) -> None:
    """The whole point: the switch is the one of the two you can reach mid-sentence."""
    note = tmp_path / "barge-in"
    barge.request(False, note)
    assert barge.margin_db(ENV, note) is None
    assert barge.enabled(ENV, note) is False


def test_on_comes_back_to_a_number_even_when_the_environment_had_none(tmp_path: Path) -> None:
    """CYCLOPS_BARGE_IN_DB=off leaves no dB to return to, so the built-in one is the answer."""
    note = tmp_path / "barge-in"
    barge.request(True, note)
    assert barge.margin_db(ENV, note) == 8.0
    assert barge.margin_db(ENV_OFF, note) == Settings.barge_in_db


def test_an_unreadable_note_is_not_an_answer(tmp_path: Path) -> None:
    """A preference must never take a session down; anything unrecognised means "nobody said"."""
    note = tmp_path / "barge-in"
    note.write_text("perhaps\n")
    assert barge.requested(note) is None
    assert barge.margin_db(ENV, note) == 8.0


# ---- what the guard does with it ----


class FakeSpeaker:
    """Just enough Speaker for the guard: is it audible, and how loud was it lately."""

    def __init__(self, audible: bool, on_air: bool = True) -> None:
        self.is_audible = audible
        self.item_serial = 1
        self.on_air = on_air  # False is a companion holding his voice: see COMPANION_LAG_S

    def recent_output_level(self, window_s: float) -> float:
        return 4000.0 if self.is_audible else 0.0

    def flush(self) -> int:
        return 0


def guard(
    *,
    half_duplex: bool,
    margin_db: float | None,
    audible: bool = True,
    on_air: bool = True,
) -> EchoGuard:
    loop = asyncio.new_event_loop()
    loop.close()  # never run: nothing here triggers, which is what the assertions say
    speaker = FakeSpeaker(audible, on_air)
    return EchoGuard(loop, speaker, half_duplex=half_duplex, margin_db=margin_db)


def test_barge_in_off_shuts_the_mic_while_cyclops_talks() -> None:
    for half_duplex in (True, False):
        shut = guard(half_duplex=half_duplex, margin_db=None)
        assert shut.admit(SILENCE) == [], "the switch is off; nothing may reach the model"


def test_barge_in_off_still_hears_you_once_cyclops_has_finished() -> None:
    quiet = guard(half_duplex=True, margin_db=None, audible=False)
    assert quiet.admit(SILENCE) == [SILENCE]


def test_headphones_with_barge_in_on_never_close_the_mic() -> None:
    """Full duplex: there is no echo to guard against, so the guard is not in the way."""
    open_mic = guard(half_duplex=False, margin_db=8.0)
    assert open_mic.admit(SILENCE) == [SILENCE]


@pytest.mark.parametrize("half_duplex", [True, False])
def test_the_switch_lands_in_a_session_already_running(half_duplex: bool) -> None:
    live = guard(half_duplex=half_duplex, margin_db=8.0)
    live.set_barge_in(None)
    assert live.admit(SILENCE) == [], "flipped off mid-sentence, the mic shuts on the next block"
    live.set_barge_in(8.0)
    assert live.barge_in_enabled


def _tapped() -> tuple[EchoGuard, FakeSpeaker]:
    """A guard with the switch off, mid-sentence, and the stand-in speaker it is watching."""
    speaker = FakeSpeaker(True)
    loop = asyncio.new_event_loop()
    loop.close()
    return EchoGuard(loop, speaker, half_duplex=True, margin_db=None), speaker


def test_a_tap_does_not_open_the_mic_onto_a_speaker_still_sounding() -> None:
    """The regression this pair exists for, and it is not a hypothetical one.

    `cut` empties our buffer, but audio already handed to the device keeps coming out of the
    speaker for AUDIBLE_TAIL_S. The first version of the tap forced the mic open across that
    gap, so Cyclops's own last words went up as if they were yours - he answered his own echo,
    and every sentence after a tap arrived chopped into fragments.
    """
    shut, _ = _tapped()
    shut.cut()
    assert shut.admit(SILENCE) == [], "he is still sounding; nothing of it may go up as you"


def test_the_mic_opens_once_the_tap_has_made_the_room_quiet() -> None:
    """And the other half: the wait is the tail and nothing longer."""
    shut, speaker = _tapped()
    shut.cut()
    speaker.is_audible = False  # AUDIBLE_TAIL_S later: the room is genuinely quiet
    assert shut.admit(SILENCE) == [SILENCE], "he was stopped; the next thing said must be heard"


LOUD = b"\x00\x40" * BLOCK_FRAMES  # int16 16384: far over anything this room could predict


def _armed(*, on_air: bool = True) -> EchoGuard:
    """A guard past warmup and calibrated on a quiet room, which is where barge-in can fire."""
    ready = guard(half_duplex=True, margin_db=8.0, on_air=on_air)
    for _ in range(EchoGuard.WARMUP_BLOCKS + EchoGuard.LEARN_BLOCKS):
        ready.admit(SILENCE)
    return ready


def test_a_companion_holding_his_voice_is_never_you_talking_over_him() -> None:
    """The bug of 2026-09-19, and the control that proves the same blocks would otherwise fire.

    Barge-in measures the room against what this box is playing *now*. A companion plays the
    same audio up to COMPANION_LAG_S later, so once our buffer drains the predicted echo is
    nothing at all and his own voice, arriving late off a phone, reads as several times over
    it. He answered himself for twenty-five minutes. There is no reference signal for audio we
    did not play, so the only honest answer while a phone has his voice is to stop guessing.
    """
    phone = _armed(on_air=False)
    heard = [block for _ in range(EchoGuard.CONSEC_BLOCKS * 10) for block in phone.admit(LOUD)]
    assert heard == [], "his own voice, late, off a phone: none of it may go up as you"
    assert phone.triggers == 0

    panel = _armed()
    over = [block for _ in range(EchoGuard.CONSEC_BLOCKS) for block in panel.admit(LOUD)]
    assert over, "the same room on this box's own amp: that is a barge-in, and it must still cut"
    assert panel.triggers == 1
