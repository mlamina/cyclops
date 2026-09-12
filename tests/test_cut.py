"""Finding the moments of a session worth watching, and the several ways that can go wrong.

Imports ``cyclops.cut``, which is stdlib plus ``card``, ``library`` and ``stats`` - so this suite
runs anywhere in a second and never opens a socket. The two things it deliberately does not test
are the model call and ffmpeg itself: one needs a key and the other needs a Pi, and both are
covered by running ``cyclops-index --once`` against a real session on the box.

What is here instead is everything between them - which is where the bugs would be, because it is
all arithmetic about time.
"""

from __future__ import annotations

import json

from cyclops import card, cut
from cyclops.config import Settings

# ------------------------------------------------------------------ building a card to clip


def make(root, name, *, log=None, summary="# S\n\nA session.\n", page="done", video=b"mp4",
         plan=None, made=()):
    """One session folder, in whatever state of repair the test needs.

    ``plan`` is a dict written to ``clips/plan.json``; ``made`` is the clip numbers that already
    have a file, which is the only thing separating "owed" from "done".

    A summary by default, because that is what :func:`cyclops.cut.settled` waits for now - a
    folder without one is one ``cyclops.after`` may be about to rename. Pass ``summary=None`` to
    get that folder.
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


STORY = [[12.0, 20.0], [30.0, 38.0], [40.0, 48.0]]  # three beats, 24 s, over STORY_LOW_S
BEATS_OF = [
    {"kind": "setup", "what": "asks", "saw": ""},
    {"kind": "turn", "what": "it lands", "saw": "drawing: a wiring diagram"},
    {"kind": "payoff", "what": "likes it", "saw": "drawing: a wiring diagram"},
]


def a_plan(**over):
    """``clips/plan.json`` as a dict, with one three-beat story unless told otherwise."""
    body = {
        "decided": "2026-09-01T18:00:00+0200",
        "by": "model",
        "tier": 3,
        "seconds": 90.0,
        "why": "",
        "speech": [],
        "clips": [
            {
                "title": "A bolt",
                "story": "He asks for the torque and it goes up on the panel.",
                "ranges": [list(r) for r in STORY],
                "beats": BEATS_OF,
                "retelling": "He wanted the torque. It appeared. He said that was the one.",
                "why": "",
            }
        ],
        "source": {},
    }
    return body | over


def clips_of(*specs):
    """``[{title, ranges, beats, why}]`` from ``(title, ranges, why)`` triples."""
    return [
        {"title": t, "ranges": [list(r) for r in rs], "beats": BEATS_OF[: len(rs)], "why": w}
        for t, rs, w in specs
    ]


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


# ------------------------------------------------------------------ what is worth asking about


def test_a_session_nobody_said_anything_in_is_never_asked_about() -> None:
    """Twenty-six of the ninety-eight sessions on the card are this, and each is a free no."""
    assert cut.worth_asking([{"t": 4.0, "type": "end", "seconds": 4.0}])


def test_a_mic_storm_is_not_a_conversation() -> None:
    """Thirty "you" turns inside four seconds is a fan. Only the clock catches this one."""
    storm = [{"t": i * 0.1, "type": "you", "text": "the", "dur": 0.1} for i in range(30)]
    storm += [{"t": 0.2, "type": "cyclops", "text": "sorry?"}, {"t": 4.1, "type": "end",
                                                               "seconds": 4.1}]
    assert cut.worth_asking(storm)


def test_a_real_conversation_is_worth_asking_about() -> None:
    assert cut.worth_asking(records_of()) == ""


def test_a_long_session_of_almost_no_words_is_not_worth_asking_about() -> None:
    """Two turns of "yeah" over three minutes passes every count but the character one."""
    thin = [
        {"t": 10.0, "type": "you", "text": "yeah", "dur": 1.0},
        {"t": 20.0, "type": "cyclops", "text": "ok"},
        {"t": 30.0, "type": "you", "text": "sure", "dur": 1.0},
        {"t": 40.0, "type": "cyclops", "text": "right"},
        {"t": 180.0, "type": "end", "seconds": 180.0},
    ]
    assert cut.worth_asking(thin)


def test_a_session_where_the_picture_never_changed_holds_no_story() -> None:
    """The newest and biggest clause: no TURN is possible, so no story is, so nobody is asked.

    Twenty of the eighty-seven sessions on the card are this - a question answered out of memory -
    and each one used to buy a model call to be told there was nothing in it.
    """
    talk = records_of()
    assert cut.worth_asking([r for r in talk if r.get("type") != "photo"])


def test_a_drawing_on_the_panel_is_a_new_picture() -> None:
    """``sketch`` replaced ``screen`` on 2026-09-11 and nothing here had noticed."""
    assert cut.made_a_picture({"type": "sketch", "code": "Heading('x')"})
    assert cut.made_a_picture({"type": "screen", "html": "<h1>25 Nm</h1>"})
    assert cut.made_a_picture({"type": "photo", "file": "photos/a.jpg"})


def test_an_old_picture_out_of_storage_is_not_a_new_one() -> None:
    """A story built on a recall is a story about the filing cabinet, and it repeats on the reel."""
    assert not cut.made_a_picture({"type": "recall", "title": "the caliper"})
    assert not cut.made_a_picture({"type": "search", "query": "torque"})
    assert not cut.made_a_picture({"type": "photo"})  # a photo record with no file: nothing landed


def test_a_drawing_reaches_the_model_with_what_is_in_it() -> None:
    """Every session after 2026-09-11 used to reach the model with its drawings missing."""
    shown = cut.timeline([{"t": 13.5, "type": "sketch", "code": "Metric(value='25 Nm')"}], {})
    assert "25 Nm" in shown


# ------------------------------------------------------------------ the shape of a story

RECS = [
    {"t": 10.0, "type": "you", "text": "draw me the wiring for that", "dur": 3.0},
    {"t": 16.0, "type": "cyclops", "text": "here it is"},
    {"t": 40.0, "type": "sketch", "code": "x"},
    {"t": 50.0, "type": "you", "text": "that is the one", "dur": 2.0},
]
BEATS = ((11.0, 14.0), (40.0, 44.0), (50.0, 53.0))


def test_a_story_is_three_beats_in_session_order() -> None:
    found = cut.shape(BEATS, (), 120.0, RECS)
    assert len(found) == 3
    assert list(found) == sorted(found)
    assert all(found[n][1] <= found[n + 1][0] for n in range(2))


def test_a_clip_opens_on_the_persons_own_line() -> None:
    """The first thing anybody notices, and what the crew is routinely a second or two out on."""
    assert cut.shape(BEATS, (), 120.0, RECS)[0][0] == 10.0


def test_the_opening_is_never_pulled_onto_cyclops_talking() -> None:
    """Only a ``you`` turn, and only backwards. 16.0 is Cyclops answering, 2.5 s away."""
    assert cut.shape(((17.0, 20.0), (40.0, 44.0), (50.0, 53.0)), (), 120.0, RECS)[0][0] == 17.0


def test_no_shot_in_a_clip_is_shorter_than_a_shot() -> None:
    """MIN_KEEP_S was 0.6 s, which is what produced twelve seconds of twenty splices."""
    found = cut.shape(((11.0, 12.0), (40.0, 41.0), (50.0, 50.8)), (), 120.0, RECS)
    assert found and all(end - start >= cut.MIN_SHOT_S - 1e-9 for start, end in found)


def test_the_turn_holds_its_picture_long_enough_to_be_seen() -> None:
    """A viewer who does not see what arrived has watched nothing at all."""
    found = cut.shape(((11.0, 14.0), (40.0, 41.0), (50.0, 53.0)), (), 120.0, RECS)
    assert found[1][1] - found[1][0] >= cut.TURN_HOLD_S - 1e-9


def test_a_clip_runs_between_fifteen_and_forty_five_seconds() -> None:
    short = cut.shape(((11.0, 13.0), (40.0, 44.0), (50.0, 52.0)), (), 120.0, RECS)
    assert cut.STORY_LOW_S - 0.5 <= sum(b - a for a, b in short) <= cut.STORY_CAP_S
    long = cut.shape(((10.0, 30.0), (31.0, 55.0), (56.0, 68.0)), (), 200.0, RECS)
    assert sum(b - a for a, b in long) <= cut.STORY_CAP_S


def test_a_clip_never_lands_a_frame_short_of_the_floor() -> None:
    """Six boundaries rounded to the nearest frame put two clips on the card at 14.93 s."""
    for beats in (
        ((11.0, 13.4), (40.0, 44.3), (50.0, 52.7)),
        ((10.3, 12.77), (30.31, 34.4), (50.09, 52.9)),
    ):
        found = cut.shape(beats, (), 120.0, RECS)
        assert sum(b - a for a, b in found) >= cut.STORY_LOW_S


def test_a_beat_with_nowhere_to_grow_still_grows_as_far_as_it_can() -> None:
    """Best-effort, not all-or-nothing: a payoff against the end still stretches."""
    found = cut.shape(((11.0, 14.0), (40.0, 44.0), (50.0, 53.0)), (), 56.0, RECS)
    assert found[-1][1] > 53.0
    assert sum(b - a for a, b in found) >= cut.STORY_LOW_S


def test_beats_on_top_of_each_other_are_not_a_story() -> None:
    assert cut.shape(((40.0, 44.0), (11.0, 14.0), (50.0, 53.0)), (), 120.0, RECS) == ()
    assert cut.shape(((11.0, 45.0), (40.0, 44.0), (50.0, 53.0)), (), 120.0, RECS) == ()


def test_a_story_reaching_across_more_of_the_session_than_a_story_can_is_dropped() -> None:
    assert cut.shape(((11.0, 14.0), (40.0, 44.0), (150.0, 155.0)), (), 300.0, RECS) == ()


# ------------------------------------------------------------------ the ranges


def test_a_range_that_runs_past_the_end_of_the_recording_is_clamped_to_it() -> None:
    assert cut.sanitize([[80.0, 500.0]], 90.0)[-1][1] == 90.0


def test_an_overlap_is_resolved_without_two_beats_becoming_one() -> None:
    """The version before this merged anything closer than a quarter second. A range is a beat now.

    Merging would silently turn three shots into two, and the beat labels beside them - which say
    which one holds the picture - would then describe the wrong seconds.
    """
    found = cut.sanitize([[10.0, 30.0], [25.0, 40.0]], 120.0)
    assert len(found) == 2
    assert found[0][1] <= found[1][0]


def test_two_beats_that_touch_stay_two_beats() -> None:
    found = cut.sanitize([[10.0, 20.0], [20.0, 30.0]], 120.0)
    assert len(found) == 2


def test_ranges_come_back_sorted_however_they_were_ordered() -> None:
    found = cut.sanitize([[40.0, 50.0], [10.0, 30.0]], 120.0)
    assert list(found) == sorted(found)
    assert found[0][0] < found[1][0]


def test_a_range_written_backwards_still_meant_a_range() -> None:
    found = cut.sanitize([[30.0, 10.0]], 120.0)
    assert found and found[0][0] < found[0][1]


def test_a_range_too_short_to_be_a_shot_is_dropped() -> None:
    assert cut.sanitize([[10.0, 10.1]], 120.0) == ()


def test_a_clip_never_spans_more_of_the_session_than_one_story() -> None:
    """WINDOW_S is the only enforceable definition of "one story" there is."""
    found = cut.sanitize([[10.0, 20.0], [200.0, 210.0]], 300.0)
    assert found and found[-1][1] - found[0][0] <= cut.WINDOW_S


def test_a_boolean_is_not_a_number_however_much_python_thinks_it_is() -> None:
    assert cut.sanitize([[True, 30.0]], 120.0) == ()


def test_every_boundary_lands_on_a_frame() -> None:
    for start, end in cut.sanitize([[10.31, 30.77]], 120.0):
        assert abs(start * cut.FPS - round(start * cut.FPS)) < 1e-9
        assert abs(end * cut.FPS - round(end * cut.FPS)) < 1e-9


def test_nothing_usable_is_an_empty_answer_and_not_a_bad_cut() -> None:
    assert cut.sanitize([], 90.0) == ()
    assert cut.sanitize([[10.0, 30.0]], 0.0) == ()


# ------------------------------------------------------------------ trimming the edges only

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


def test_a_beat_is_trimmed_at_its_edges_and_never_split_at_a_hole() -> None:
    """The other half of the repair. This used to come back as two shots with the pause removed.

    The pause it removed was somebody looking at a picture that had just arrived, which is the
    shot the clip exists for.
    """
    assert cut.tighten([(10.0, 30.0)], [(10.0, 15.0), (25.0, 30.0)]) == ((10.0, 30.0),)


def test_the_dead_air_at_the_ends_of_a_beat_still_goes() -> None:
    assert cut.tighten([(10.0, 30.0)], [(14.0, 22.0)]) == ((14.0, 22.0),)


def test_no_silence_measured_means_the_beats_pass_through_untouched() -> None:
    """The whole "a quality multiplier, not a dependency" claim, in one assertion."""
    assert cut.tighten([(10.0, 30.0)], ()) == ((10.0, 30.0),)


def test_tightening_keeps_a_silent_hold_on_a_picture() -> None:
    """A beat with nobody talking in it is the payoff shot, not dead air, and it survives whole."""
    assert cut.tighten([(40.0, 48.0)], [(10.0, 20.0)]) == ((40.0, 48.0),)
    # Nobody speaks over the drawing at all, and the beat still opens where the crew put it.
    turn = cut.shape(BEATS, [(9.0, 14.5), (50.0, 53.5)], 120.0, RECS)[1]
    assert turn[0] == 40.0 and turn[1] - turn[0] >= cut.TURN_HOLD_S


def test_a_boundary_near_one_of_the_persons_turns_is_pulled_onto_it() -> None:
    assert cut.opens_on_a_person(12.0, RECS) == 10.0


def test_a_boundary_far_from_any_turn_is_left_alone() -> None:
    assert cut.opens_on_a_person(80.0, RECS) == 80.0


# ------------------------------------------------------------------ naming it


def test_a_session_with_no_summary_still_gets_a_title(tmp_path) -> None:
    folder = make(tmp_path, "2026-09-01_18-13-08_bolt-torque", log=LOG, summary=None)
    assert cut.naming(folder, "") == "Bolt torque"


def test_an_empty_title_falls_back_to_the_one_the_list_would_show(tmp_path) -> None:
    folder = make(tmp_path, "2026-09-01_18-13-08_bolt", log=LOG, summary=SUMMARY)
    assert cut.naming(folder, "") == "How tight the bolt goes"


def test_a_folder_nobody_has_named_yet_still_gets_a_title(tmp_path) -> None:
    folder = make(tmp_path, "2026-09-01_18-13-08", log=LOG, summary=None)
    assert cut.naming(folder, "").startswith("Session ")


def test_the_models_own_title_outranks_the_summary(tmp_path) -> None:
    folder = make(tmp_path, "2026-09-01_18-13-08_bolt", log=LOG, summary=SUMMARY)
    assert cut.naming(folder, "The pads were glazed") == "The pads were glazed"


# ------------------------------------------------------------------ where a moment ends up


def test_time_moves_with_the_cut() -> None:
    ranges = ((10.0, 20.0), (50.0, 60.0))
    assert cut.shift(10.0, ranges) == 0.0
    assert cut.shift(15.0, ranges) == 5.0
    assert cut.shift(50.0, ranges) == 10.0
    assert cut.shift(30.0, ranges) is None, "a moment that was cut out is nowhere"


def a_clip(ranges=tuple(tuple(r) for r in STORY), title="A bolt"):
    return cut.Clip(title=title, ranges=ranges)


# ------------------------------------------------------------------ the command


def test_the_command_is_one_encode_of_one_input() -> None:
    argv = cut.clip_command(a_clip(), "1.mp4.tmp")
    assert argv[0] == "ffmpeg" and argv.count("-i") == 1
    assert argv[-1] == "1.mp4.tmp"


def test_the_command_seeks_to_the_clip_and_rebases_every_timestamp() -> None:
    """Input seeking is what keeps the decode proportional to the clip, not to the session."""
    argv = cut.clip_command(a_clip(((80.0, 92.0),)), "1.mp4.tmp")
    assert argv[argv.index("-ss") + 1] == "80.000"
    graph = argv[argv.index("-filter_complex") + 1]
    assert "trim=start=0.000:end=12.000" in graph


def test_nothing_a_model_wrote_reaches_the_filtergraph() -> None:
    """The burn is gone, so the escaping surface is gone with it rather than being guarded."""
    graph = cut.clip_command(a_clip(title="a/b:c'd,[e]"), "1.mp4.tmp")[
        cut.clip_command(a_clip(), "1.mp4.tmp").index("-filter_complex") + 1
    ]
    assert "subtitles=" not in graph
    # [v0]/[vb] are the graph's own pad labels; what must not be in there is the title.
    assert "a/b" not in graph and "c'd" not in graph


def test_the_command_folds_the_two_voices_together_and_pads_no_cards_on() -> None:
    graph = cut.clip_command(a_clip(), "1.mp4.tmp")[
        cut.clip_command(a_clip(), "1.mp4.tmp").index("-filter_complex") + 1
    ]
    assert "pan=stereo" in graph
    assert "tpad" not in graph and "loudnorm" not in graph


# ------------------------------------------------------------------ the state on the card


def test_a_session_nobody_has_looked_at_is_waiting(tmp_path) -> None:
    folder = make(tmp_path, "2026-09-01_18-13-08", log=LOG)
    assert cut.state(folder) == "asked"


def test_three_clips_with_one_made_is_still_owed(tmp_path) -> None:
    folder = make(
        tmp_path, "2026-09-01_18-13-08", log=LOG,
        plan=a_plan(clips=clips_of(("A", STORY, ""), ("B", STORY, ""),
                                   ("C", ((50.0, 62.0),), ""))),
        made=(1,),
    )
    assert cut.progress(folder) == cut.Progress("asked", 1, 3, 0)


def test_a_scratch_file_beside_the_target_means_a_render_is_running(tmp_path) -> None:
    folder = make(tmp_path, "2026-09-01_18-13-08", log=LOG, plan=a_plan())
    card.tmp_for(cut.clip_path(folder, 1)).write_bytes(b"half")
    assert cut.state(folder) == "clipping"


def test_a_plan_with_no_clips_in_it_is_a_finished_answer(tmp_path) -> None:
    """"Looked at, nothing there" is the ordinary outcome and must never look like an error."""
    folder = make(tmp_path, "2026-09-01_18-13-08", log=LOG, plan=a_plan(clips=[]))
    assert cut.state(folder) == "none"
    assert cut.work(tmp_path) is None, "and it is never offered again"


def test_a_clip_carrying_a_reason_is_skipped_and_the_next_one_offered(tmp_path) -> None:
    folder = make(
        tmp_path, "2026-09-01_18-13-08", log=LOG,
        plan=a_plan(clips=clips_of(("A", STORY, "ffmpeg said no"), ("B", STORY, ""))),
    )
    assert cut.work(tmp_path) == cut.Work(folder, 2)


def test_a_zero_byte_clip_is_not_a_clip(tmp_path) -> None:
    folder = make(tmp_path, "2026-09-01_18-13-08", log=LOG, plan=a_plan())
    cut.clip_path(folder, 1).write_bytes(b"")
    assert cut.state(folder) == "asked"


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
    """Otherwise a broken file costs a model call on every bell for as long as it is broken."""
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
        asked.append((folder, n))
        cut.clip_path(folder, n).write_bytes(b"mp4")
        return done

    monkeypatch.setattr(cut, "render", one_render)
    return asked


def test_a_session_that_is_still_recording_stops_the_whole_sweep(tmp_path, monkeypatch) -> None:
    make(tmp_path, "2026-09-01_18-13-08", log=LOG, plan=a_plan())
    monkeypatch.setattr(cut.card, "locked", lambda f: f.name.endswith("live"))
    make(tmp_path, "2026-09-02_18-13-08_live", log=LOG)
    monkeypatch.setattr(cut.stats, "cpu_temp_c", lambda: 50.0)
    asked = only_one(monkeypatch)
    assert "conversation" in cut.one(settings_for(tmp_path))
    assert asked == []


def test_a_hot_board_is_left_to_cool(tmp_path, monkeypatch) -> None:
    make(tmp_path, "2026-09-01_18-13-08", log=LOG, plan=a_plan())
    monkeypatch.setattr(cut.card, "locked", lambda f: False)
    monkeypatch.setattr(cut.stats, "cpu_temp_c", lambda: 86.0)
    asked = only_one(monkeypatch)
    assert "too hot" in cut.one(settings_for(tmp_path))
    assert asked == []


def test_only_one_clip_is_made_per_sweep(tmp_path, monkeypatch) -> None:
    """The finished file rings the index service's own bell, which schedules the next one."""
    make(
        tmp_path, "2026-09-01_18-13-08", log=LOG,
        plan=a_plan(clips=clips_of(("A", STORY, ""), ("B", [[50.0, 58.0], [60.0, 68.0],
                                                             [70.0, 78.0]], ""))),
    )
    monkeypatch.setattr(cut.card, "locked", lambda f: False)
    monkeypatch.setattr(cut.stats, "cpu_temp_c", lambda: 50.0)
    asked = only_one(monkeypatch)
    cut.one(settings_for(tmp_path))
    assert len(asked) == 1


def test_switching_it_off_does_nothing_at_all(tmp_path) -> None:
    make(tmp_path, "2026-09-01_18-13-08", log=LOG)
    assert cut.one(settings_for(tmp_path, cut=False)) == ""


def test_a_session_too_thin_to_bother_with_costs_nothing_but_a_read(
    tmp_path, monkeypatch
) -> None:
    """Marco: "make sure that empty sessions don't waste resources". No ffprobe, no model."""
    thin = '{"t": 0.0, "type": "session"}\n{"t": 4.0, "type": "end", "seconds": 4.0}\n'
    folder = make(tmp_path, "2026-09-01_18-13-08", log=thin)
    monkeypatch.setattr(cut.card, "locked", lambda f: False)
    monkeypatch.setattr(cut.stats, "cpu_temp_c", lambda: 50.0)
    monkeypatch.setattr(cut, "probe", _never("probe"))
    monkeypatch.setattr(cut, "listen", _never("listen"))
    assert "tier 1, free" in cut.one(settings_for(tmp_path))
    assert cut.state(folder) == "none", "and it is never considered again"
    assert cut.read_plan(folder).tier == 1, "and the plan says which tier stopped it"


def test_a_session_with_no_new_picture_in_it_costs_no_model_call(tmp_path, monkeypatch) -> None:
    """Twenty of the eighty-seven on this card, and no client is ever built for one of them."""
    talk = "\n".join(
        line for line in LOG.strip().splitlines() if '"type": "photo"' not in line
    ) + "\n"
    folder = make(tmp_path, "2026-09-01_18-13-08", log=talk)
    monkeypatch.setattr(cut.card, "locked", lambda f: False)
    monkeypatch.setattr(cut.stats, "cpu_temp_c", lambda: 50.0)
    monkeypatch.setattr(cut, "probe", _never("probe"))
    monkeypatch.setattr(cut, "listen", _never("listen"))
    assert "tier 1, free" in cut.one(settings_for(tmp_path))
    assert cut.read_plan(folder).tier == 1
    assert cut.state(folder) == "none"


def test_a_recording_with_no_picture_in_it_never_reaches_ffmpeg(tmp_path, monkeypatch) -> None:
    folder = make(tmp_path, "2026-09-01_18-13-08", log=LOG, summary=SUMMARY)
    monkeypatch.setattr(cut.card, "locked", lambda f: False)
    monkeypatch.setattr(cut.stats, "cpu_temp_c", lambda: 50.0)
    monkeypatch.setattr(cut, "probe", lambda folder, records: (90.0, False))
    asked = only_one(monkeypatch)
    assert "no picture" in cut.one(settings_for(tmp_path))
    assert asked == [] and cut.read_plan(folder).why
    assert cut.work(tmp_path) is None, "and it is never retried"


def test_no_key_writes_no_plan_so_the_backlog_survives_the_night(tmp_path, monkeypatch) -> None:
    """"Nobody looked" and "somebody looked and there was nothing" must be different files."""
    folder = make(tmp_path, "2026-09-01_18-13-08", log=LOG, summary=SUMMARY)
    monkeypatch.setattr(cut.card, "locked", lambda f: False)
    monkeypatch.setattr(cut.stats, "cpu_temp_c", lambda: 50.0)
    monkeypatch.setattr(cut, "probe", lambda folder, records: (90.0, True))
    monkeypatch.setattr(cut, "listen", lambda folder, seconds: ())
    assert "no answer" in cut.one(settings_for(tmp_path))
    assert cut.read_plan(folder) is None
    assert cut.work(tmp_path) == cut.Work(folder, -1), "still pending for a sweep with a key"


def test_a_plan_that_exists_is_never_decided_twice(tmp_path, monkeypatch) -> None:
    """What makes a deploy that kills a render cost the encode and never the model call."""
    make(tmp_path, "2026-09-01_18-13-08", log=LOG, plan=a_plan())
    monkeypatch.setattr(cut.card, "locked", lambda f: False)
    monkeypatch.setattr(cut.stats, "cpu_temp_c", lambda: 50.0)
    only_one(monkeypatch)
    cut.one(settings_for(tmp_path))


def test_a_render_that_actually_failed_stops_being_retried(tmp_path, monkeypatch) -> None:
    folder = make(tmp_path, "2026-09-01_18-13-08", log=LOG, plan=a_plan())
    monkeypatch.setattr(cut.card, "locked", lambda f: False)
    monkeypatch.setattr(cut.stats, "cpu_temp_c", lambda: 50.0)
    monkeypatch.setattr(cut, "render", lambda f, c, n, root: cut.Cut(False, "no such filter"))
    assert "could not clip" in cut.one(settings_for(tmp_path))
    assert cut.read_plan(folder).clips[0].why == "no such filter"
    assert cut.work(tmp_path) is None


def test_a_pause_leaves_the_clip_owed(tmp_path, monkeypatch) -> None:
    folder = make(tmp_path, "2026-09-01_18-13-08", log=LOG, plan=a_plan())
    monkeypatch.setattr(cut.card, "locked", lambda f: False)
    monkeypatch.setattr(cut.stats, "cpu_temp_c", lambda: 50.0)
    monkeypatch.setattr(
        cut, "render", lambda f, c, n, root: cut.Cut(False, "paused for a conversation")
    )
    cut.one(settings_for(tmp_path))
    assert cut.read_plan(folder).clips[0].why == "", "a pause is not a failure"
    assert cut.work(tmp_path) == cut.Work(folder, 1)


def test_a_clip_that_evaporates_after_tightening_is_never_offered_again(
    tmp_path, monkeypatch
) -> None:
    """Otherwise it sits at the head of a newest-first queue and starves everything behind it."""
    folder = make(
        tmp_path, "2026-09-01_18-13-08", log=LOG,
        plan=a_plan(clips=clips_of(("A", ((12.0, 13.0),), ""))),
    )
    monkeypatch.setattr(cut.card, "locked", lambda f: False)
    monkeypatch.setattr(cut.stats, "cpu_temp_c", lambda: 50.0)
    monkeypatch.setattr(cut, "render", _never("render"))
    cut.one(settings_for(tmp_path))
    assert cut.read_plan(folder).clips[0].why
    assert cut.work(tmp_path) is None


def _never(what):
    def refuse(*a, **kw):
        raise AssertionError(f"{what} should not have been reached")
    return refuse


# ------------------------------------------------------------------ the ledger


def test_a_render_killed_by_a_deploy_stops_claiming_to_be_running(tmp_path, monkeypatch) -> None:
    """Marco read the header eight minutes after a deploy and asked whether it was true.

    push.sh kills renders by design, and the ledger row goes with the process - nothing writes
    "this process died". Holding CUT_LOCK is what makes a running row of ours provably stale.
    """
    from cyclops import tasks

    orphan = tasks.start("Cutting the video of something that was killed…")
    assert [one.id for one in tasks.running()] == [orphan]

    make(tmp_path, "2026-09-01_18-13-08_bolt", log=LOG, summary=SUMMARY, plan=a_plan())
    monkeypatch.setattr(cut.card, "locked", lambda f: False)
    monkeypatch.setattr(cut.stats, "cpu_temp_c", lambda: 50.0)
    only_one(monkeypatch)
    cut.one(settings_for(tmp_path))

    closed = {one.id: one for one in tasks.read()}
    assert closed[orphan].state == tasks.FAILED
    assert closed[orphan].result == "interrupted"


def test_a_stranded_row_is_closed_even_with_nothing_left_to_render(tmp_path) -> None:
    """The half of that bug the first fix missed.

    A killed render leaves a card with nothing owed on it, so a sweep that tidied up only on its
    way to rendering something never ran, and the header went on saying "Cutting the video of…"
    until the row aged out fifteen minutes later.
    """
    from cyclops import tasks

    orphan = tasks.start("Cutting the video of a render nobody is running…")
    assert cut.one(settings_for(tmp_path)) == "", "there is nothing to render"
    assert tasks.running() == [], "and nothing left claiming to be running either"
    assert {one.id: one.state for one in tasks.read()}[orphan] == tasks.FAILED


def test_somebody_elses_row_is_left_alone(tmp_path, monkeypatch) -> None:
    """The lock says nothing about a diagram being drawn in another process."""
    from cyclops import tasks

    theirs = tasks.start("Drawing a diagram of the wiring…")
    make(tmp_path, "2026-09-01_18-13-08_bolt", log=LOG, summary=SUMMARY, plan=a_plan())
    monkeypatch.setattr(cut.card, "locked", lambda f: False)
    monkeypatch.setattr(cut.stats, "cpu_temp_c", lambda: 50.0)
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
    monkeypatch.setattr(cut.subprocess, "Popen", _fake_popen(1, b"could not open 1.ass"))
    done = cut.render(folder, a_clip(), 1, tmp_path)
    assert not done.ok and done.why == "could not open 1.ass"
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
         plan=a_plan(clips=clips_of(("A", STORY, ""), ("B", [[50.0, 58.0], [60.0, 68.0],
                                                             [70.0, 78.0]], ""))),
         made=(1, 2))
    made, _ = cut.clips(tmp_path)
    assert [one.id.split("_")[-1] for one in made] == ["late/1", "late/2", "early/1"]
    assert all(one.src.startswith("/media/") and "/clips/" in one.src for one in made)


def test_a_clip_that_is_not_rendered_yet_is_not_on_the_reel(tmp_path) -> None:
    """Reverses the old listing on purpose: nobody arrives here having just pressed a button."""
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
    """Ninety-eight folders on the card hold these names. Dropping them strands every one."""
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
