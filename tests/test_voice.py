"""Which of the ten voices answers you, and how the panel comes to decide that.

The note is a file and the ten names are a tuple, so none of this needs a websocket, a speaker or
a key. What is guarded here is the set of quiet wrongs: a name that only OpenAI would reject, the
three lists of voices in this repo drifting apart from each other, and a session that opens in
the voice ``.env`` holds rather than the one somebody chose on the glass.

The sample clips themselves are not tested here - they are shipped cues like any other, and
``tests/test_sfx.py`` already measures every one of them for format, loudness, headroom and
clicks the moment they appear in ``sfx.SAMPLES``.
"""

from __future__ import annotations

import typing
from pathlib import Path

import pytest
from openai.types.realtime import realtime_audio_config_output_param as sdk

from cyclops import sfx, voice
from cyclops.config import ConfigError, Settings, _voice

ENV_MARIN = Settings(api_key="", voice="marin")
ENV_VERSE = Settings(api_key="", voice="verse")


# ---- the note ----


def test_no_note_leaves_the_environment_to_decide(tmp_path: Path) -> None:
    note = tmp_path / "voice"
    assert voice.requested(note) is None
    assert voice.chosen(ENV_MARIN, note) == "marin"
    assert voice.chosen(ENV_VERSE, note) == "verse"


def test_the_choice_survives_a_round_trip(tmp_path: Path) -> None:
    note = tmp_path / "cache" / "voice"  # the directory does not exist yet, as at boot
    voice.request("cedar", note)
    assert voice.requested(note) == "cedar"
    voice.request("ash", note)
    assert voice.requested(note) == "ash"


def test_the_panel_beats_the_environment(tmp_path: Path) -> None:
    """The whole point: it is the one of the two you can reach without an ssh session."""
    note = tmp_path / "voice"
    voice.request("cedar", note)
    assert voice.chosen(ENV_MARIN, note) == "cedar"
    assert voice.chosen(ENV_VERSE, note) == "cedar"


def test_an_unreadable_note_is_not_an_answer(tmp_path: Path) -> None:
    """A preference must never cost a session; anything unrecognised means "nobody said"."""
    note = tmp_path / "voice"
    note.write_text("morgan freeman\n")
    assert voice.requested(note) is None
    assert voice.chosen(ENV_MARIN, note) == "marin"


def test_a_name_off_the_list_is_never_written(tmp_path: Path) -> None:
    """The page validates too, but this is the layer that cannot be gone around."""
    note = tmp_path / "voice"
    with pytest.raises(ValueError):
        voice.request("morgan freeman", note)
    assert not note.exists()


def test_case_and_whitespace_are_not_typos(tmp_path: Path) -> None:
    note = tmp_path / "voice"
    assert voice.request("  CEDAR \n", note) == "cedar"
    assert voice.requested(note) == "cedar"


# ---- the environment ----


def test_a_typo_in_the_environment_is_caught_at_startup(monkeypatch) -> None:
    """Not at the tap, and not by OpenAI. A box that will not sound like you asked should say so
    while you are watching, rather than opening a session that is refused three layers down."""
    monkeypatch.setenv("CYCLOPS_VOICE", "morgan freeman")
    with pytest.raises(ConfigError):
        _voice("CYCLOPS_VOICE")
    monkeypatch.setenv("CYCLOPS_VOICE", "CEDAR")  # case is not a typo
    assert _voice("CYCLOPS_VOICE") == "cedar"
    monkeypatch.delenv("CYCLOPS_VOICE")
    assert _voice("CYCLOPS_VOICE") == Settings.voice


def test_the_config_module_knows_the_same_ten(monkeypatch) -> None:
    """``config`` writes the names out again rather than importing them, because ``voice``
    imports ``config``. This is the seam."""
    for name in voice.VOICES:
        monkeypatch.setenv("CYCLOPS_VOICE", name)
        assert _voice("CYCLOPS_VOICE") == name


# ---- the ten themselves ----


def test_the_ten_are_the_ten_the_api_will_accept() -> None:
    """The one thing here that rots on its own: OpenAI adds a voice and this list does not.

    Read off the installed SDK's own literal rather than the docs, because that is the thing the
    request will actually be held to.
    """
    literals = [
        arg for arg in typing.get_args(sdk.Voice) if typing.get_origin(arg) is typing.Literal
    ]
    assert len(literals) == 1, "the SDK's Voice alias changed shape; go and look at it"
    assert set(typing.get_args(literals[0])) == voice.NAMES


def test_the_default_is_one_of_them() -> None:
    assert Settings.voice in voice.NAMES


def test_every_voice_has_something_to_say() -> None:
    """A voice you can step to but not hear is the list of names this replaced."""
    for name in voice.VOICES:
        assert voice.cue(name) in sfx.SAMPLES, f"no sample clip for {name}"
