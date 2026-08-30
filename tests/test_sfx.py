"""What a sound cue amounts to, before anything tries to play it.

Only the rendering is checked here - `cyclops.sfx.play` needs a real output device and belongs
in the on-the-Pi checks. The three failures worth catching are all silent ones: a cue that
renders to nothing, a cue loud enough to wrap the int16 cast into a square burst, and a cue
with a hard edge that clicks. None of them raise; all of them are audible on the panel.
"""

from __future__ import annotations

import numpy as np
import pytest

from cyclops import sfx

RATE = 24_000  # what cyclops.audio.SAMPLE_RATE is; passed explicitly so the two never drift


@pytest.mark.parametrize("name", sorted(sfx.CUES))
def test_every_cue_renders_to_audible_int16(name: str) -> None:
    pcm = sfx.render(name, RATE)
    assert pcm.dtype == np.int16
    assert len(pcm) > 0, "a cue that renders to nothing is a cue that silently never plays"
    peak = int(np.abs(pcm.astype(np.int32)).max())
    assert peak > 0, "silence would pass every other check here"
    # Headroom, not loudness: a sine at full scale sits far above the voice beside it, and an
    # overshoot wraps rather than clips.
    assert peak <= int(0.35 * 32767), f"{name} peaks at {peak}, over its headroom"


@pytest.mark.parametrize("name", sorted(sfx.CUES))
def test_every_cue_fades_in_and_out(name: str) -> None:
    """The click test. A cue that starts or ends on a non-zero sample steps the speaker cone,
    which is heard as a tick in front of the sound - and for a looping cue, once per bar."""
    pcm = sfx.render(name, RATE).astype(np.int32)
    quiet = 0.05 * 32767
    assert np.abs(pcm[:3]).max() < quiet, f"{name} starts on an edge"
    assert np.abs(pcm[-3:]).max() < quiet, f"{name} ends on an edge"


def test_render_follows_the_rate_it_is_given() -> None:
    """The rate is a parameter precisely so it cannot drift from the speaker's."""
    at_24 = sfx.render("ready", RATE)
    at_48 = sfx.render("ready", 2 * RATE)
    assert len(at_48) == pytest.approx(2 * len(at_24), rel=0.01)


def test_render_is_cached_and_immutable() -> None:
    """It is built once per process and handed out by reference, so it must not be editable."""
    assert sfx.render("ready", RATE) is sfx.render("ready", RATE)
    with pytest.raises(ValueError):
        sfx.render("ready", RATE)[0] = 1


def test_unknown_cue_names_itself() -> None:
    with pytest.raises(KeyError, match="no such cue"):
        sfx.render("nope", RATE)


def test_connecting_loops_seamlessly() -> None:
    """It is played with loop=True, so its own end meets its own start every 1.1 s."""
    pcm = sfx.render("connecting", RATE)
    quiet = 0.05 * 32767
    assert abs(int(pcm[0])) < quiet and abs(int(pcm[-1])) < quiet
    assert len(pcm) / RATE == pytest.approx(1.1, abs=0.01), "the gap is what keeps it ignorable"
