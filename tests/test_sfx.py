"""What a sound cue amounts to, before anything tries to play it.

Only the rendering is checked here - `cyclops.sfx.play` needs a real output device and belongs
in the on-the-Pi checks. The three failures worth catching are all silent ones: a cue that
renders to nothing, a cue loud enough to wrap the int16 cast into a square burst, and a cue
with a hard edge that clicks. None of them raise; all of them are audible on the panel.

There are two kinds of cue now, and the split here follows the one real difference between
them. A sine's peak sits ~3 dB over its loudness, so for a synthesized cue a peak ceiling is a
usable loudness rule; a designed sting's peak sits ~12 dB over, so the same ceiling would say
nothing about how loud it lands. The shipped ones are measured on a short-window RMS instead,
against the number the synthesized ones actually come out at.
"""

from __future__ import annotations

import wave

import numpy as np
import pytest

from cyclops import sfx

RATE = 24_000  # what cyclops.audio.SAMPLE_RATE is; passed explicitly so the two never drift
SYNTH = sorted(sfx.CUES)
SHIPPED = sorted(sfx.SAMPLES)
EVERY = SYNTH + SHIPPED

# The loudest 50 ms of the busiest synthesized cue measures 0.176 of full scale, and that is the
# bar the shipped ones are cut to sit on. The window is the point: it is what "as loud as the
# beep beside it" means for broadband material, where a peak is not.
#
# Two bands rather than one, because the five files are two shapes. The four that sound for a
# second or more land at 0.155-0.206 here; the shutter is a 0.53 s transient and reads 0.083,
# not because it is quiet - it peaks higher than any of them - but because averaging a click
# over 50 ms is what a short window does to it. So the RMS floor only has to catch a cue nobody
# would hear at all, and the peak band catches both a cue with no presence and a slammed one.
LOUD_FS = (0.05, 0.25)
PEAK_FS = (0.25, 0.90)


def pcm_of(name: str) -> np.ndarray:
    """A cue's samples, whichever of the two kinds it turns out to be."""
    return sfx.cue(name, RATE)[0]


def loudest_50ms(pcm: np.ndarray) -> float:
    window = int(0.05 * sfx.SAMPLE_HZ)
    usable = len(pcm) // window * window
    blocks = (pcm[:usable].astype(np.float64) / 32768).reshape(-1, window)
    return float(np.sqrt((blocks**2).mean(axis=1)).max())


def test_a_cue_is_one_kind_or_the_other() -> None:
    """Two tables, one namespace of names - so `cue` can decide by lookup and nothing else."""
    assert not set(sfx.CUES) & set(sfx.SAMPLES)


@pytest.mark.parametrize("name", SYNTH)
def test_every_synthesized_cue_renders_to_audible_int16(name: str) -> None:
    pcm = sfx.render(name, RATE)
    assert pcm.dtype == np.int16
    assert len(pcm) > 0, "a cue that renders to nothing is a cue that silently never plays"
    peak = int(np.abs(pcm.astype(np.int32)).max())
    assert peak > 0, "silence would pass every other check here"
    # Headroom, not loudness: a sine at full scale sits far above the voice beside it, and an
    # overshoot wraps rather than clips. A rule about oscillators - see the module docstring.
    assert peak <= int(0.35 * 32767), f"{name} peaks at {peak}, over its headroom"


@pytest.mark.parametrize("name", SHIPPED)
def test_every_shipped_cue_is_the_file_the_speaker_expects(name: str) -> None:
    """The conversion happened, and happened right.

    `load` turns a mis-cut file into silence rather than an exception, which is what keeps a bad
    deploy from taking the panel down - and is exactly why it has to be caught here instead.
    """
    path = sfx.SOUNDS / sfx.SAMPLES[name]
    assert path.is_file(), f"{path} did not ship"
    with wave.open(str(path), "rb") as wav:
        cut = (wav.getnchannels(), wav.getsampwidth(), wav.getframerate())
    assert cut == (1, 2, sfx.SAMPLE_HZ), f"{path.name} is {cut}; re-cut it with the ffmpeg line"
    pcm = sfx.load(name)
    assert pcm.dtype == np.int16 and len(pcm) > 0
    assert not pcm.flags.writeable, "cached and handed out: nobody gets to edit it in place"


@pytest.mark.parametrize("name", SHIPPED)
def test_every_shipped_cue_sits_where_the_beeps_do(name: str) -> None:
    """A sting mastered hot would shout over the voice it is meant to sit beside."""
    pcm = sfx.load(name)
    loud = loudest_50ms(pcm)
    assert LOUD_FS[0] <= loud <= LOUD_FS[1], f"{name} is {loud:.3f} of full scale, off the bar"
    peak = float(np.abs(pcm.astype(np.int32)).max()) / 32767
    assert PEAK_FS[0] <= peak <= PEAK_FS[1], f"{name} peaks at {peak:.3f}, outside its headroom"


@pytest.mark.parametrize("name", EVERY)
def test_every_cue_fades_in_and_out(name: str) -> None:
    """The click test. A cue that starts or ends on a non-zero sample steps the speaker cone,
    which is heard as a tick in front of the sound - and for a looping cue, once per bar."""
    pcm = pcm_of(name).astype(np.int32)
    quiet = 0.05 * 32767
    assert np.abs(pcm[:3]).max() < quiet, f"{name} starts on an edge"
    assert np.abs(pcm[-3:]).max() < quiet, f"{name} ends on an edge"


@pytest.mark.parametrize("name", EVERY)
def test_a_cue_says_which_rate_it_wants(name: str) -> None:
    """The load-bearing one. A shipped cue keeps its own rate whatever it is asked for; a
    synthesized cue is built at the rate it was asked for. Get this backwards and a sting plays
    at half speed, which is the one failure here that is not silent."""
    wanted = sfx.SAMPLE_HZ if name in sfx.SAMPLES else RATE
    assert sfx.cue(name, RATE)[1] == wanted


def test_a_rung_is_short_enough_to_be_a_detent() -> None:
    """Twenty of these cross the whole volume column, one per rung the finger passes. At this
    length they read as a ladder being crossed; much longer and a sweep is a tune with twenty
    notes in it, and the ticks start arriving before the one before them has finished."""
    assert len(sfx.render("rung", RATE)) / RATE <= 0.04


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


def test_a_shipped_cue_is_read_once() -> None:
    """1.4 MB of fanfare, read on the one call that plays it and not again."""
    assert sfx.load("booted") is sfx.load("booted")


def test_a_shipped_cue_that_will_not_load_is_a_missing_beep() -> None:
    """A file that did not deploy is a quiet panel, not a dead one - the same bargain `play`
    makes with a device that refuses."""
    sfx.load.cache_clear()
    try:
        sfx.SAMPLES["pressed"] = "not-a-file.wav"
        assert len(sfx.load("pressed")) == 0
    finally:
        sfx.SAMPLES["pressed"] = "cyclops_eye_pressed.wav"
        sfx.load.cache_clear()


def test_unknown_cue_names_itself() -> None:
    with pytest.raises(KeyError, match="no such cue"):
        sfx.render("nope", RATE)
    with pytest.raises(KeyError):
        sfx.cue("nope", RATE)


def test_connecting_loops_seamlessly() -> None:
    """It is played with loop=True, so its own end meets its own start every time round.

    Two properties, and the second is the one that stops a mechanism from becoming a drone: it
    has to end quieter than it runs, so there is a rest between one turn and the next. The
    length is not asserted - it is a recording now, and how long a gear takes to turn is the
    recording's business.
    """
    pcm = sfx.load("connecting")
    quiet = 0.05 * 32767
    assert abs(int(pcm[0])) < quiet and abs(int(pcm[-1])) < quiet
    body = np.abs(pcm[: len(pcm) // 2].astype(np.int32)).mean()
    rest = np.abs(pcm[-len(pcm) // 10 :].astype(np.int32)).mean()
    assert rest < body / 4, "it butts up against itself; the rest between turns is gone"
