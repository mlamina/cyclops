"""One session into one video, with the silence played fast - and the ways that can go wrong.

Almost all of it is arithmetic about time, tested with no subprocess. One test at the end renders
a real, short, synthetic recording with the system ffmpeg, because "the file is as long as the
numbers say" is the one claim nothing but ffmpeg can answer. It is skipped where there is none.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from cyclops import card, cut
from cyclops.config import Settings

# ------------------------------------------------------------------ building a card to clip


def make(root, name, *, log=None, summary="# S\n\nA session.\n", page="done", video=b"mp4",
         plan=None, made=()):
    """One session folder, in whatever state of repair the test needs.

    ``plan`` is a dict written to ``clips/plan.json``; ``made`` is the clip numbers that already
    have a file. A summary by default, because that is what :func:`cyclops.cut.settled` waits for.
    """
    folder = root / name
    folder.mkdir(parents=True)
    if log is not None:
        (folder / card.LOG_NAME).write_text(log, encoding="utf-8")
    if summary is not None:
        (folder / card.SUMMARY_NAME).write_text(summary, encoding="utf-8")
    if page is not None:
        (folder / card.PAGE_NAME).write_text(page, encoding="utf-8")
    if video is not None:
        (folder / card.VIDEO).write_bytes(video)
    if plan is not None:
        (folder / card.CLIPS).mkdir(exist_ok=True)
        cut.plan_path(folder).write_text(json.dumps(plan), encoding="utf-8")
    for n in made:
        (folder / card.CLIPS).mkdir(exist_ok=True)
        cut.clip_path(folder, n).write_bytes(b"mp4")
    return folder


WHOLE = [[11.85, 20.25, 1.0], [20.25, 58.85, 8.0], [58.85, 66.25, 1.0]]


def a_plan(**over):
    """``clips/plan.json`` as a dict, with the one video unless told otherwise."""
    body = {
        "decided": "2026-09-01T18:00:00+0200",
        "seconds": 90.0,
        "why": "",
        "speech": [],
        "clips": [{"title": "A bolt", "ranges": WHOLE, "why": ""}],
        "source": {},
    }
    return body | over


def clips_of(*specs):
    """``[{title, ranges, why}]`` from ``(title, ranges, why)`` triples."""
    return [{"title": t, "ranges": [list(r) for r in rs], "why": w} for t, rs, w in specs]


# Long enough in words and seconds to clear worth_asking, because almost every test below is
# about what happens *after* that gate and a fixture that trips it tests nothing.
LOG = (
    '{"t": 0.0, "type": "session", "uuid": "u", "entrypoint": "kiosk", "model": "m"}\n'
    '{"t": 12.0, "type": "you", "text": "how tight should the rear caliper bolts go? the manual '
    'is in the shed and I have the torque wrench out already", "dur": 3.0}\n'
    '{"t": 17.0, "type": "cyclops", "text": "about 25 newton metres, dry, and the bracket bolts '
    'are 40. Check them again after a heat cycle because the alloy relaxes."}\n'
    '{"t": 30.0, "type": "photo", "by": "you", "file": "photos/14-32-30_you.jpg"}\n'
    '{"t": 60.0, "type": "you", "text": "this one has a lip on the disc, is that past the wear '
    'limit or can I get another season out of it", "dur": 2.0}\n'
    '{"t": 64.0, "type": "cyclops", "text": "that lip means it is at the limit. Measure across '
    'the swept area with a vernier and compare it to the number stamped on the hub."}\n'
    '{"t": 90.0, "type": "end", "reason": "stopped", "seconds": 90.0, "photos": 1}\n'
)

SUMMARY = "# How tight the bolt goes\n\nTwenty-five newton metres, and why.\n"


def settings_for(root, **over):
    return Settings(api_key="", sessions_dir=root, **over)


def records_of(log=LOG):
    return [json.loads(line) for line in log.strip().splitlines()]


def calm(monkeypatch):
    """No live session and a cool board, so a sweep gets as far as the work."""
    monkeypatch.setattr(cut.card, "locked", lambda f: False)
    monkeypatch.setattr(cut.stats, "cpu_temp_c", lambda: 50.0)


# ------------------------------------------------------------------ is it worth a video


def test_a_session_nobody_said_anything_in_is_never_clipped() -> None:
    assert cut.worth_asking([{"t": 4.0, "type": "end", "seconds": 4.0}])


def test_a_mic_storm_is_not_a_conversation() -> None:
    """Thirty "you" turns inside four seconds is a fan. Only the clock catches this one."""
    storm = [{"t": i * 0.1, "type": "you", "text": "the", "dur": 0.1} for i in range(30)]
    storm += [{"t": 0.2, "type": "cyclops", "text": "sorry?"}, {"t": 4.1, "type": "end",
                                                               "seconds": 4.1}]
    assert cut.worth_asking(storm)


def test_a_long_session_of_almost_no_words_is_not_worth_a_video() -> None:
    """Two turns of "yeah" over three minutes passes every count but the character one."""
    thin = [
        {"t": 10.0, "type": "you", "text": "yeah", "dur": 1.0},
        {"t": 20.0, "type": "cyclops", "text": "ok"},
        {"t": 30.0, "type": "you", "text": "sure", "dur": 1.0},
        {"t": 40.0, "type": "cyclops", "text": "right"},
        {"t": 180.0, "type": "end", "seconds": 180.0},
    ]
    assert cut.worth_asking(thin)


def test_a_conversation_gets_a_video_whether_or_not_the_picture_changed() -> None:
    """The story needed a new picture to hold its turn. There is no story any more."""
    talk = records_of()
    assert cut.worth_asking(talk) == ""
    assert cut.worth_asking([r for r in talk if r.get("type") != "photo"]) == ""


# ------------------------------------------------------------------ what plays how fast


def speeds(found):
    return [speed for _, _, speed in found]


def test_nothing_between_the_first_word_and_the_last_is_dropped() -> None:
    """Back to back from the first audible moment to the last: no holes, no overlaps."""
    speech = [(3.0, 5.2), (5.9, 8.0), (12.0, 15.5), (15.7, 16.0), (40.0, 41.3), (41.8, 60.0)]
    found = cut.segments(speech, 90.0)
    assert found[0][0] <= speech[0][0] and found[-1][1] >= speech[-1][1]
    assert found[0][0] > 0.0 and found[-1][1] < 90.0, "the empty head and tail are trimmed"
    assert all(a < b for a, b, _ in found)
    assert all(found[n][1] == found[n + 1][0] for n in range(len(found) - 1))


def test_a_short_hole_between_two_spans_stays_one_unbroken_1x_segment() -> None:
    assert speeds(cut.segments([(10.0, 12.0), (12.5, 14.0)], 90.0)) == [1.0]


def test_a_long_hole_between_two_spans_plays_fast() -> None:
    assert speeds(cut.segments([(10.0, 12.0), (15.0, 17.0)], 90.0)) == [1.0, cut.SPEED, 1.0]


def test_every_audible_span_is_padded_before_the_gaps_are_measured() -> None:
    """No word onset is clipped: the gap starts after LEAD_OUT_S and ends LEAD_IN_S early."""
    assert cut.tighten([(10.0, 12.0), (20.0, 22.0)], 90.0) == [
        [10.0 - cut.LEAD_IN_S, 12.0 + cut.LEAD_OUT_S],
        [20.0 - cut.LEAD_IN_S, 22.0 + cut.LEAD_OUT_S],
    ]


def test_padding_that_makes_two_spans_touch_merges_them() -> None:
    hole = cut.LEAD_IN_S + cut.LEAD_OUT_S - 0.05
    assert len(cut.tighten([(10.0, 12.0), (12.0 + hole, 14.0)], 90.0)) == 1


def test_padding_never_reaches_outside_the_recording() -> None:
    assert cut.tighten([(0.05, 2.0), (88.0, 89.95)], 90.0) == [[0.0, 2.25], [87.85, 90.0]]


def test_every_boundary_lands_on_a_thirtieth_of_a_second() -> None:
    """The recording's own frames, so no cut falls between two of them - and a thirtieth, not
    the fifteenth the screen used to be recorded at."""
    speech = [(10.31, 12.77), (20.02, 30.4), (40.0 + 1 / 30 + cut.LEAD_IN_S, 45.0)]
    bounds = [t for start, end, _ in cut.segments(speech, 90.0) for t in (start, end)]
    assert all(abs(t * 30 - round(t * 30)) < 1e-9 for t in bounds)
    assert any(abs(t * 15 - round(t * 15)) > 1e-9 for t in bounds), "and not only fifteenths"


def test_a_recording_nobody_can_be_heard_in_has_no_segments() -> None:
    assert cut.segments((), 90.0) == ()


# ------------------------------------------------------------------ listening


SILENCE = """\
[silencedetect @ 0x1] silence_start: 0
[silencedetect @ 0x1] silence_end: 12.0 | silence_duration: 12.0
[silencedetect @ 0x1] silence_start: 18.0
[silencedetect @ 0x1] silence_end: 24.0 | silence_duration: 6.0
"""


def test_the_silence_report_inverts_into_the_audible_spans() -> None:
    assert cut.read_silence(SILENCE, 40.0) == ((12.0, 18.0), (24.0, 40.0))


def test_a_recording_with_no_silence_in_it_is_audible_throughout() -> None:
    assert cut.read_silence("", 40.0) == ((0.0, 40.0),)


# ------------------------------------------------------------------ naming it


def test_the_video_is_titled_with_the_summarys_first_line(tmp_path) -> None:
    folder = make(tmp_path, "2026-09-01_18-13-08_bolt", log=LOG, summary=SUMMARY)
    assert cut.naming(folder) == "How tight the bolt goes"


def test_a_session_with_no_summary_still_gets_a_title(tmp_path) -> None:
    folder = make(tmp_path, "2026-09-01_18-13-08_bolt-torque", log=LOG, summary=None)
    assert cut.naming(folder) == "Bolt torque"


def test_a_folder_nobody_has_named_yet_still_gets_a_title(tmp_path) -> None:
    folder = make(tmp_path, "2026-09-01_18-13-08", log=LOG, summary=None)
    assert cut.naming(folder).startswith("Session ")


# ------------------------------------------------------------------ the command


def a_clip(ranges=tuple(tuple(r) for r in WHOLE), title="A bolt"):
    return cut.Clip(title=title, ranges=ranges)


def test_the_command_is_one_encode_of_one_input() -> None:
    argv = cut.clip_command(a_clip(), "1.mp4.tmp")
    assert argv[0] == "ffmpeg" and argv.count("-i") == 1
    assert argv[-1] == "1.mp4.tmp"


def test_nothing_but_numbers_reaches_the_filtergraph() -> None:
    argv = cut.clip_command(a_clip(title="a/b:c'd,[e]"), "1.mp4.tmp")
    graph = argv[argv.index("-filter_complex") + 1]
    assert "a/b" not in graph and "c'd" not in graph


# ------------------------------------------------------------------ the state on the card


def test_a_session_nobody_has_looked_at_is_waiting(tmp_path) -> None:
    folder = make(tmp_path, "2026-09-01_18-13-08", log=LOG)
    assert cut.state(folder) == "asked"


def test_three_clips_with_one_made_is_still_owed(tmp_path) -> None:
    """Every plan now holds one; a hand edit may hold more, and the counts still add up."""
    folder = make(
        tmp_path, "2026-09-01_18-13-08", log=LOG,
        plan=a_plan(clips=clips_of(("A", WHOLE, ""), ("B", WHOLE, ""), ("C", WHOLE, ""))),
        made=(1,),
    )
    assert cut.progress(folder) == cut.Progress("asked", 1, 3, 0)


def test_a_scratch_file_beside_the_target_means_a_render_is_running(tmp_path) -> None:
    folder = make(tmp_path, "2026-09-01_18-13-08", log=LOG, plan=a_plan())
    card.tmp_for(cut.clip_path(folder, 1)).write_bytes(b"half")
    assert cut.state(folder) == "clipping"


def test_a_plan_with_no_clips_in_it_is_a_finished_answer(tmp_path) -> None:
    folder = make(tmp_path, "2026-09-01_18-13-08", log=LOG, plan=a_plan(clips=[]))
    assert cut.state(folder) == "none"
    assert cut.work(tmp_path) is None, "and it is never offered again"


def test_a_clip_carrying_a_reason_is_skipped_and_the_next_one_offered(tmp_path) -> None:
    folder = make(
        tmp_path, "2026-09-01_18-13-08", log=LOG,
        plan=a_plan(clips=clips_of(("A", WHOLE, "ffmpeg said no"), ("B", WHOLE, ""))),
    )
    assert cut.work(tmp_path) == cut.Work(folder, 2)


def test_a_zero_byte_clip_is_not_a_clip(tmp_path) -> None:
    folder = make(tmp_path, "2026-09-01_18-13-08", log=LOG, plan=a_plan())
    cut.clip_path(folder, 1).write_bytes(b"")
    assert cut.state(folder) == "asked"


def test_a_plan_from_before_the_speeds_plays_at_1x(tmp_path) -> None:
    """Ninety-six plans on the card hold pairs. A pair is a segment at normal speed."""
    folder = make(tmp_path, "2026-09-01_18-13-08", log=LOG,
                  plan=a_plan(clips=clips_of(("A", [[12.0, 20.0]], ""))))
    assert cut.read_plan(folder).clips[0].ranges == ((12.0, 20.0, 1.0),)


# ------------------------------------------------------------------ the queue


def test_a_session_with_a_recording_and_no_plan_is_decided_first(tmp_path) -> None:
    folder = make(tmp_path, "2026-09-01_18-13-08", log=LOG)
    assert cut.work(tmp_path) == cut.Work(folder, -1)


def test_the_newest_unconsidered_session_jumps_the_backfill(tmp_path) -> None:
    make(tmp_path, "2026-09-01_10-00-00_old", log=LOG)
    new = make(tmp_path, "2026-09-05_10-00-00_new", log=LOG)
    assert cut.work(tmp_path).folder == new


def test_the_backfill_drains_newest_first_one_at_a_time(tmp_path) -> None:
    """One unit per sweep is the whole rate limit, so the order it drains in is the guarantee."""
    for n in range(5):
        make(tmp_path, f"2026-09-0{n + 1}_10-00-00", log=LOG)
    seen = []
    while (todo := cut.work(tmp_path)) is not None:
        seen.append(todo.folder.name)
        cut.write_plan(todo.folder, cut.Plan(clips=()))
    assert seen == sorted(seen, reverse=True), "newest first"
    assert len(seen) == 5, "and every one of them is eventually looked at"


def test_a_session_still_recording_is_never_looked_at(tmp_path, monkeypatch) -> None:
    make(tmp_path, "2026-09-01_18-13-08", log=LOG)
    monkeypatch.setattr(cut.card, "locked", lambda f: True)
    assert cut.work(tmp_path) is None


def test_a_plan_nobody_can_read_is_replaced_rather_than_decided_again(tmp_path) -> None:
    folder = make(tmp_path, "2026-09-01_18-13-08", log=LOG)
    (folder / card.CLIPS).mkdir(exist_ok=True)
    cut.plan_path(folder).write_text("{ not json", encoding="utf-8")
    assert cut.work(tmp_path) is None
    assert cut.read_plan(folder).why


def test_asking_again_throws_away_what_was_decided(tmp_path) -> None:
    folder = make(tmp_path, "2026-09-01_18-13-08", log=LOG, plan=a_plan(), made=(1,))
    cut.forget(folder)
    assert cut.state(folder) == "asked"
    assert cut.work(tmp_path) == cut.Work(folder, -1)


# ------------------------------------------------------------------ the sweep


def only_one(monkeypatch, done=None):
    """Render nothing, and record what was asked for."""
    done = done or cut.Cut(True)
    asked = []

    def one_render(folder, clip, n, root):
        asked.append((folder, n, clip))
        cut.clip_path(folder, n).write_bytes(b"mp4")
        return done

    monkeypatch.setattr(cut, "render", one_render)
    return asked


def _never(what):
    def refuse(*a, **kw):
        raise AssertionError(f"{what} should not have been reached")
    return refuse


def test_a_session_that_is_still_recording_stops_the_whole_sweep(tmp_path, monkeypatch) -> None:
    make(tmp_path, "2026-09-01_18-13-08", log=LOG, plan=a_plan())
    monkeypatch.setattr(cut.card, "locked", lambda f: f.name.endswith("live"))
    make(tmp_path, "2026-09-02_18-13-08_live", log=LOG)
    monkeypatch.setattr(cut.stats, "cpu_temp_c", lambda: 50.0)
    asked = only_one(monkeypatch)
    cut.one(settings_for(tmp_path))
    assert asked == []


def test_a_hot_board_is_left_to_cool(tmp_path, monkeypatch) -> None:
    make(tmp_path, "2026-09-01_18-13-08", log=LOG, plan=a_plan())
    monkeypatch.setattr(cut.card, "locked", lambda f: False)
    monkeypatch.setattr(cut.stats, "cpu_temp_c", lambda: 86.0)
    asked = only_one(monkeypatch)
    cut.one(settings_for(tmp_path))
    assert asked == []


def test_only_one_clip_is_made_per_sweep(tmp_path, monkeypatch) -> None:
    """The finished file rings the index service's own bell, which schedules the next one."""
    make(tmp_path, "2026-09-01_18-13-08", log=LOG,
         plan=a_plan(clips=clips_of(("A", WHOLE, ""), ("B", WHOLE, ""))))
    calm(monkeypatch)
    asked = only_one(monkeypatch)
    cut.one(settings_for(tmp_path))
    assert len(asked) == 1


def test_switching_it_off_does_nothing_at_all(tmp_path) -> None:
    make(tmp_path, "2026-09-01_18-13-08", log=LOG)
    assert cut.one(settings_for(tmp_path, cut=False)) == ""


def test_a_session_too_thin_to_bother_with_costs_nothing_but_a_read(
    tmp_path, monkeypatch
) -> None:
    """Marco: "make sure that empty sessions don't waste resources". No ffprobe, no ffmpeg."""
    thin = '{"t": 0.0, "type": "session"}\n{"t": 4.0, "type": "end", "seconds": 4.0}\n'
    folder = make(tmp_path, "2026-09-01_18-13-08", log=thin)
    calm(monkeypatch)
    monkeypatch.setattr(cut, "probe", _never("probe"))
    monkeypatch.setattr(cut, "listen", _never("listen"))
    cut.one(settings_for(tmp_path))
    assert cut.state(folder) == "none", "and it is never considered again"


def test_deciding_plans_exactly_one_video_of_the_measured_speech(tmp_path, monkeypatch) -> None:
    folder = make(tmp_path, "2026-09-01_18-13-08_bolt", log=LOG, summary=SUMMARY)
    calm(monkeypatch)
    heard = ((12.0, 20.0), (59.0, 66.0))
    monkeypatch.setattr(cut, "probe", lambda folder, records: (90.0, True))
    monkeypatch.setattr(cut, "listen", lambda folder, seconds: heard)
    cut.one(settings_for(tmp_path))
    plan = cut.read_plan(folder)
    assert [c.title for c in plan.clips] == ["How tight the bolt goes"]
    assert plan.clips[0].ranges == cut.segments(heard, 90.0)


def test_a_recording_with_no_picture_in_it_never_reaches_ffmpeg(tmp_path, monkeypatch) -> None:
    folder = make(tmp_path, "2026-09-01_18-13-08", log=LOG, summary=SUMMARY)
    calm(monkeypatch)
    monkeypatch.setattr(cut, "probe", lambda folder, records: (90.0, False))
    asked = only_one(monkeypatch)
    cut.one(settings_for(tmp_path))
    assert asked == [] and cut.read_plan(folder).why
    assert cut.work(tmp_path) is None, "and it is never retried"


def test_a_recording_nobody_can_be_heard_in_is_not_retried(tmp_path, monkeypatch) -> None:
    folder = make(tmp_path, "2026-09-01_18-13-08", log=LOG, summary=SUMMARY)
    calm(monkeypatch)
    monkeypatch.setattr(cut, "probe", lambda folder, records: (90.0, True))
    monkeypatch.setattr(cut, "listen", lambda folder, seconds: ())
    cut.one(settings_for(tmp_path))
    assert cut.state(folder) == "failed"
    assert cut.work(tmp_path) is None


def test_a_plan_that_exists_is_never_measured_twice(tmp_path, monkeypatch) -> None:
    """What makes a deploy that kills a render cost the encode and nothing else."""
    make(tmp_path, "2026-09-01_18-13-08", log=LOG, plan=a_plan())
    calm(monkeypatch)
    monkeypatch.setattr(cut, "probe", _never("probe"))
    monkeypatch.setattr(cut, "listen", _never("listen"))
    asked = only_one(monkeypatch)
    cut.one(settings_for(tmp_path))
    assert len(asked) == 1, "it went straight to the render"


def test_a_hand_edited_plan_is_held_to_what_can_render(tmp_path, monkeypatch) -> None:
    """Out of order, overlapping and past the end: rendered in order, back to back, inside."""
    make(tmp_path, "2026-09-01_18-13-08", log=LOG, plan=a_plan(clips=clips_of(
        ("A", [[40.0, 80.0, 8.0], [10.0, 45.0, 1.0], [80.0, 500.0, 0.5]], ""))))
    calm(monkeypatch)
    asked = only_one(monkeypatch)
    cut.one(settings_for(tmp_path))
    ranges = asked[0][2].ranges
    assert ranges == ((10.0, 45.0, 1.0), (45.0, 80.0, 8.0), (80.0, 90.0, 1.0))


def test_a_render_that_actually_failed_stops_being_retried(tmp_path, monkeypatch) -> None:
    folder = make(tmp_path, "2026-09-01_18-13-08", log=LOG, plan=a_plan())
    calm(monkeypatch)
    monkeypatch.setattr(cut, "render", lambda f, c, n, root: cut.Cut(False, "no such filter"))
    cut.one(settings_for(tmp_path))
    assert cut.read_plan(folder).clips[0].why == "no such filter"
    assert cut.work(tmp_path) is None


def test_a_pause_leaves_the_clip_owed(tmp_path, monkeypatch) -> None:
    folder = make(tmp_path, "2026-09-01_18-13-08", log=LOG, plan=a_plan())
    calm(monkeypatch)
    monkeypatch.setattr(
        cut, "render", lambda f, c, n, root: cut.Cut(False, "paused for a conversation")
    )
    cut.one(settings_for(tmp_path))
    assert cut.read_plan(folder).clips[0].why == "", "a pause is not a failure"
    assert cut.work(tmp_path) == cut.Work(folder, 1)


def test_a_plan_with_nothing_renderable_left_is_never_offered_again(tmp_path, monkeypatch) -> None:
    """Otherwise it sits at the head of a newest-first queue and starves everything behind it."""
    folder = make(tmp_path, "2026-09-01_18-13-08", log=LOG,
                  plan=a_plan(clips=clips_of(("A", [[120.0, 130.0, 1.0]], ""))))
    calm(monkeypatch)
    monkeypatch.setattr(cut, "render", _never("render"))
    cut.one(settings_for(tmp_path))
    assert cut.read_plan(folder).clips[0].why
    assert cut.work(tmp_path) is None


# ------------------------------------------------------------------ the ledger


def test_a_render_killed_by_a_deploy_stops_claiming_to_be_running(tmp_path, monkeypatch) -> None:
    """push.sh kills renders by design, and the ledger row goes with the process."""
    from cyclops import tasks

    orphan = tasks.start("Cutting the video of something that was killed…")
    assert [one.id for one in tasks.running()] == [orphan]

    make(tmp_path, "2026-09-01_18-13-08_bolt", log=LOG, summary=SUMMARY, plan=a_plan())
    calm(monkeypatch)
    only_one(monkeypatch)
    cut.one(settings_for(tmp_path))

    closed = {one.id: one for one in tasks.read()}
    assert closed[orphan].state == tasks.FAILED


def test_a_stranded_row_is_closed_even_with_nothing_left_to_render(tmp_path) -> None:
    from cyclops import tasks

    orphan = tasks.start("Cutting the video of a render nobody is running…")
    assert cut.one(settings_for(tmp_path)) == "", "there is nothing to render"
    assert tasks.running() == [], "and nothing left claiming to be running either"
    assert {one.id: one.state for one in tasks.read()}[orphan] == tasks.FAILED


def test_somebody_elses_row_is_left_alone(tmp_path, monkeypatch) -> None:
    from cyclops import tasks

    theirs = tasks.start("Drawing a diagram of the wiring…")
    make(tmp_path, "2026-09-01_18-13-08_bolt", log=LOG, summary=SUMMARY, plan=a_plan())
    calm(monkeypatch)
    only_one(monkeypatch)
    cut.one(settings_for(tmp_path))

    assert {one.id for one in tasks.read() if one.is_running} >= {theirs}


# ------------------------------------------------------------------ the render itself


def test_a_missing_ffmpeg_is_a_reason_and_not_a_crash(tmp_path, monkeypatch) -> None:
    """systemd's PATH is not a login shell's - record.mux's own note."""
    folder = make(tmp_path, "2026-09-01_18-13-08", log=LOG)
    monkeypatch.setattr(cut.subprocess, "Popen", _raise(FileNotFoundError("no ffmpeg")))
    done = cut.render(folder, a_clip(), 1, tmp_path)
    assert not done.ok and "FileNotFoundError" in done.why


def test_a_failed_run_leaves_no_scratch_and_no_clip(tmp_path, monkeypatch) -> None:
    folder = make(tmp_path, "2026-09-01_18-13-08", log=LOG)
    monkeypatch.setattr(cut.subprocess, "Popen", _fake_popen(1, b"no such filter"))
    done = cut.render(folder, a_clip(), 1, tmp_path)
    assert not done.ok and done.why == "no such filter"
    assert not card.tmp_for(cut.clip_path(folder, 1)).exists()
    assert not cut.clip_path(folder, 1).exists()


def test_a_clip_lands_only_when_ffmpeg_returned_zero(tmp_path, monkeypatch) -> None:
    folder = make(tmp_path, "2026-09-01_18-13-08", log=LOG)

    def popen(argv, **kw):
        card.tmp_for(cut.clip_path(folder, 1)).write_bytes(b"a finished clip")
        return _Proc(0, b"")

    monkeypatch.setattr(cut.subprocess, "Popen", popen)
    assert cut.render(folder, a_clip(), 1, tmp_path).ok
    assert card.written(cut.clip_path(folder, 1))
    assert not card.tmp_for(cut.clip_path(folder, 1)).exists()


class _Proc:
    def __init__(self, code, err):
        self.returncode = code
        self.stderr = _Reader(err)

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self):
        pass

    def kill(self):
        pass


class _Reader:
    def __init__(self, data):
        self._data = data

    def read(self):
        return self._data


def _fake_popen(code, err):
    def popen(argv, **kw):
        return _Proc(code, err)
    return popen


def _raise(exc):
    def popen(argv, **kw):
        raise exc
    return popen


# ------------------------------------------------------------------ the listing


def test_the_reel_holds_the_finished_clips_newest_session_first(tmp_path) -> None:
    make(tmp_path, "2026-09-01_10-00-00_early", log=LOG, summary=SUMMARY,
         plan=a_plan(), made=(1,))
    make(tmp_path, "2026-09-03_10-00-00_late", log=LOG, summary=SUMMARY,
         plan=a_plan(), made=(1,))
    made, _ = cut.clips(tmp_path)
    assert [one.id.split("_")[-1] for one in made] == ["late/1", "early/1"]
    assert all(one.src.startswith("/media/") and "/clips/" in one.src for one in made)


def test_the_reel_says_how_long_the_finished_video_runs(tmp_path) -> None:
    make(tmp_path, "2026-09-01_10-00-00", log=LOG, summary=SUMMARY, plan=a_plan(), made=(1,))
    made, _ = cut.clips(tmp_path)
    assert made[0].seconds == round(8.4 + 38.6 / cut.SPEED + 7.4, 2)


def test_a_clip_that_is_not_rendered_yet_is_not_on_the_reel(tmp_path) -> None:
    make(tmp_path, "2026-09-01_10-00-00", log=LOG, summary=SUMMARY, plan=a_plan())
    made, _ = cut.clips(tmp_path)
    assert made == []


def test_the_counts_tell_an_empty_reel_from_an_unlooked_at_card(tmp_path) -> None:
    make(tmp_path, "2026-09-01_10-00-00_looked", log=LOG, summary=SUMMARY, plan=a_plan(clips=[]))
    make(tmp_path, "2026-09-02_10-00-00_waiting", log=LOG, summary=SUMMARY)
    _, seen = cut.clips(tmp_path)
    assert seen == {"looked": 1, "found": 0, "waiting": 1}


# ------------------------------------------------------------------ the card knows about them


def test_a_folder_holding_clips_is_not_a_surprise(tmp_path) -> None:
    """Without this, surprises() calls them "not ours" and the folder can never be removed."""
    folder = make(tmp_path, "2026-09-01_18-13-08", log=LOG, plan=a_plan(), made=(1,))
    (folder / card.CLIPS / "1.ass").write_text("[Script Info]\n", encoding="utf-8")
    assert card.surprises(folder) == []


def test_a_folder_holding_the_old_design_is_still_a_folder_that_can_be_removed(tmp_path) -> None:
    folder = make(tmp_path, "2026-09-01_18-13-08", log=LOG)
    for name in (card.CUT_REQUEST, card.CUT_PLAN, card.CUT_SUBS, card.CUT):
        (folder / name).write_bytes(b"x")
    assert card.surprises(folder) == []


def test_a_session_holding_clips_is_actually_removed(tmp_path) -> None:
    """rmdir refuses on a directory with anything in it, so clips/ has to empty itself first."""
    from cyclops import session

    folder = make(tmp_path, "2026-09-01_18-13-08", log=LOG, video=None, page=None,
                  plan=a_plan(), made=(1,))
    said = session._remove(folder)
    assert "removed" in said and not folder.exists(), said


# ------------------------------------------------------------------ one real render


# Left channel: you, 0.6-1.6 s and 5-5.8 s. Right channel: Cyclops, 1.8-2.6 s. Silence around and
# between, so the plan is 1x, one long quiet stretch at SPEED, 1x - and a head and tail to trim.
SYNTH_S = 8
SYNTH = [
    "ffmpeg", "-v", "error", "-y",
    "-f", "lavfi", "-i", f"testsrc2=size=160x96:rate=30:duration={SYNTH_S}",
    "-f", "lavfi", "-i",
    "aevalsrc=exprs='if(between(t,0.6,1.6)+between(t,5,5.8),sin(2*PI*440*t)/2,0)"
    f"|if(between(t,1.8,2.6),sin(2*PI*660*t)/2,0)':s=48000:d={SYNTH_S}",
    "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", card.VIDEO,
]


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    """One synthetic session, decided and rendered by two real sweeps with no model reachable."""
    if not shutil.which("ffmpeg"):
        pytest.skip("no ffmpeg on this box")
    import openai

    from cyclops import tasks

    root = tmp_path_factory.mktemp("sessions")
    folder = make(root, "2026-09-01_18-13-08_bolt", log=LOG, summary=SUMMARY, video=None)
    subprocess.run(SYNTH, cwd=folder, check=True, capture_output=True, timeout=60)  # noqa: S603

    def no_model(*a, **kw):
        raise AssertionError("the cut built an OpenAI client")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tasks, "TASKS_FILE", root / "tasks.yaml")
        mp.setattr(tasks, "TASKS_LOCK", root / "tasks.lock")
        mp.setattr(tasks, "_memo", None)
        mp.setattr(openai.OpenAI, "__init__", no_model)
        mp.setattr(openai.AsyncOpenAI, "__init__", no_model)
        calm(mp)
        cut.one(settings_for(root))
        cut.one(settings_for(root))
    return folder


def test_a_session_renders_a_finished_video_with_no_model_to_ask(rendered) -> None:
    assert cut.state(rendered) == "done"
    assert speeds(cut.read_plan(rendered).clips[0].ranges) == [1.0, cut.SPEED, 1.0]


def test_the_rendered_video_runs_as_long_as_the_arithmetic_says(rendered) -> None:
    done = subprocess.run(  # noqa: S603
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
         str(cut.clip_path(rendered, 1))],
        capture_output=True, check=True, timeout=30,
    )
    predicted = cut.read_plan(rendered).clips[0].seconds
    assert abs(float(done.stdout) - predicted) <= 1.0 / cut.FPS


def test_the_rendered_video_plays_at_thirty_frames_a_second(rendered) -> None:
    done = subprocess.run(  # noqa: S603
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=r_frame_rate", "-of", "csv=p=0", str(cut.clip_path(rendered, 1))],
        capture_output=True, check=True, timeout=30,
    )
    assert done.stdout.strip() == b"30/1"
