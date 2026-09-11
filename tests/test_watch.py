"""Finding a video and the moment in it - the pure halves, with no network and no model."""

from __future__ import annotations

from cyclops import watch, youtube


def _video(**over) -> youtube.Video:
    base = dict(
        id="abc123",
        title="How to Sharpen a Chisel",
        channel="Paul Sellers",
        duration=600,
        stream="https://example.invalid/v.mp4",
        thumb="https://example.invalid/t.jpg",
        chapters=[],
    )
    return youtube.Video(**{**base, **over})


def test_cues_fold_into_half_minute_windows() -> None:
    events = [
        {"tStartMs": 0, "segs": [{"utf8": "first"}]},
        {"tStartMs": 12_000, "segs": [{"utf8": "still first"}]},
        {"tStartMs": 31_000, "segs": [{"utf8": "second"}]},
        {"tStartMs": 95_000, "segs": [{"utf8": "fourth"}]},
    ]
    assert youtube.buckets(events) == [
        (0, "first still first"),
        (30, "second"),
        (90, "fourth"),
    ]


def test_a_cue_wrapped_over_lines_becomes_one_line() -> None:
    """A window is one line of a prompt, and captions arrive hard-wrapped."""
    events = [{"tStartMs": 0, "segs": [{"utf8": "over\ntwo"}, {"utf8": " lines"}]}]
    assert youtube.buckets(events) == [(0, "over two lines")]


def test_a_transcript_never_reaches_the_sections_of_a_video_with_chapters() -> None:
    """The token invariant, as the cheapest possible test.

    A video with chapters is described by its chapters alone. If the transcript ever joined
    them the prompt would grow by a couple of thousand tokens without anybody noticing, and
    the whole reason this module exists is that those tokens must not be paid for twice.
    """
    spoken = [(0, "a sentence nobody should see here"), (30, "nor this one")]
    sections = watch.sections_of(_video(chapters=[(0, "Intro"), (99, "Honing")]), spoken)
    assert sections == "[0] Intro\n[99] Honing"
    assert "nobody should see" not in sections


def test_a_video_without_chapters_is_described_by_what_is_said_in_it() -> None:
    sections = watch.sections_of(_video(), [(0, "first"), (30, "second")])
    assert sections == "[0] first\n[30] second"


def test_only_ids_that_were_offered_survive_a_ranking_reply() -> None:
    known = {"aaa", "bbb", "ccc"}
    assert watch.ids_named("- bbb\nzzz\n  ccc \nbbb", known) == ["bbb", "ccc"]


def test_a_second_past_the_end_is_pulled_back_inside_the_video() -> None:
    assert watch.seconds_named("99999", 600) == 599
    assert watch.seconds_named("300", 600) == 300


def test_a_reply_with_no_number_in_it_starts_at_the_beginning() -> None:
    assert watch.seconds_named("I could not tell", 600) == 0
    assert watch.seconds_named("", 600) == 0


def test_the_start_is_said_as_a_clock() -> None:
    assert watch.Watch("i", "t", "c", 312, "s", "th").clock == "5:12"
    assert watch.Watch("i", "t", "c", 7, "s", "th").clock == "0:07"
