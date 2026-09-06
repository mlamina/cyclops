"""Turning a recording into a video, and the several ways that can go wrong.

Imports ``cyclops.cut``, which is stdlib plus ``card``, ``library`` and ``stats`` - so this suite
runs anywhere in a second and never opens a socket. The two things it deliberately does not test
are the model call and ffmpeg itself: one needs a key and the other needs a Pi, and both are
covered by running ``cyclops-index --once`` against a real session on the box.

What is here instead is everything between them - which is where the bugs would be, because it is
all arithmetic about time.
"""

from __future__ import annotations

import json

import pytest

from cyclops import card, cut
from cyclops.config import Settings

# ------------------------------------------------------------------ building a card to cut


def make(root, name, *, log=None, summary=None, page="done", video=b"mp4", request=None,
         plan=None, made=None):
    """One session folder, in whatever state of repair the test needs."""
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
    if request is not None:
        (folder / card.CUT_REQUEST).write_text(request, encoding="utf-8")
    if plan is not None:
        (folder / card.CUT_PLAN).write_text(json.dumps(plan), encoding="utf-8")
    if made is not None:
        (folder / card.CUT).write_bytes(made)
    return folder


LOG = (
    '{"t": 0.0, "type": "session", "uuid": "u", "entrypoint": "kiosk", "model": "m"}\n'
    '{"t": 12.0, "type": "you", "text": "how tight should this bolt be?", "dur": 3.0}\n'
    '{"t": 17.0, "type": "cyclops", "text": "about 25 newton metres, dry."}\n'
    '{"t": 30.0, "type": "photo", "by": "you", "file": "photos/14-32-30_you.jpg"}\n'
    '{"t": 60.0, "type": "you", "text": "and the caliper bolts?", "dur": 2.0}\n'
    '{"t": 64.0, "type": "cyclops", "text": "thirty, and check them again after a heat cycle."}\n'
    '{"t": 90.0, "type": "end", "reason": "stopped", "seconds": 90.0, "photos": 1}\n'
)

SUMMARY = "# How tight the bolt goes\n\nTwenty-five newton metres, and why.\n"


def settings_for(root, **over):
    return Settings(api_key="", sessions_dir=root, **over)


# ------------------------------------------------------------------ the ranges


def test_a_range_that_runs_past_the_end_of_the_recording_is_clamped_to_it() -> None:
    """The clamp is against the file, so a model guessing at the length cannot overrun it."""
    (start, end), = cut.sanitize([[100.0, 9999.0]], 120.0)
    assert end == 120.0
    assert start < 120.0


def test_two_overlapping_ranges_become_one() -> None:
    """A negative gap is an overlap, so the merge that closes stutters closes these too."""
    assert len(cut.sanitize([[10.0, 40.0], [30.0, 60.0]], 120.0)) == 1


def test_ranges_come_back_sorted_however_the_model_ordered_them() -> None:
    found = cut.sanitize([[80.0, 95.0], [10.0, 30.0]], 120.0)
    assert [round(start) for start, _ in found] == sorted(round(start) for start, _ in found)
    assert found[0][0] < found[1][0]


def test_a_range_written_backwards_still_meant_a_range() -> None:
    (start, end), = cut.sanitize([[40.0, 10.0]], 120.0)
    assert start < end


def test_a_range_shorter_than_a_breath_is_dropped() -> None:
    """Padding runs first, so this is a span that is still too short after being widened."""
    found = cut.sanitize([[10.0, 10.05], [40.0, 70.0]], 120.0)
    assert len(found) == 1, "the flicker is gone and the real one is not"
    assert found[0][0] == pytest.approx(39.75, abs=0.07)  # padded, then snapped to a frame
    assert found[0][1] == pytest.approx(70.45, abs=0.07)


def test_four_hundred_ranges_become_sixteen() -> None:
    many = [[float(n * 3), float(n * 3 + 2)] for n in range(400)]
    assert len(cut.sanitize(many, 1500.0)) <= cut.MAX_RANGES


def test_nothing_usable_is_an_empty_answer_and_not_a_bad_cut() -> None:
    """Junk is dropped rather than repaired, and too little left over is not an edit."""
    assert cut.sanitize([["a", "b"], [float("nan"), 2.0], None, [1, 2, 3]], 120.0) == ()
    assert cut.sanitize([[1.0, 3.0]], 120.0) == ()  # under MIN_TOTAL_S
    assert cut.sanitize([[10.0, 30.0]], 0.0) == ()  # a recording of no length


def test_a_boolean_is_not_a_number_however_much_python_thinks_it_is() -> None:
    assert cut.sanitize([[True, False]], 120.0) == ()


def test_every_boundary_lands_on_a_frame() -> None:
    for start, end in cut.sanitize([[10.13, 40.77]], 120.0):
        assert abs(start * cut.FPS - round(start * cut.FPS)) < 1e-6
        assert abs(end * cut.FPS - round(end * cut.FPS)) < 1e-6


# ------------------------------------------------------------------ the fallback


def test_the_rules_cut_a_session_nobody_asked_a_model_about() -> None:
    """What makes the model an optimisation rather than a dependency."""
    found = cut.rules(_records(), 90.0)
    assert found, "a session with speech in it always has something to keep"
    assert found[0][0] < 12.0, "it starts before the first thing said"
    assert found[-1][1] <= 90.0


def test_the_rules_drop_a_long_silence_in_the_middle() -> None:
    records = [
        {"t": 5.0, "type": "you", "text": "a", "dur": 1.0},
        {"t": 8.0, "type": "cyclops", "text": "b"},
        {"t": 200.0, "type": "you", "text": "c", "dur": 1.0},
        {"t": 205.0, "type": "cyclops", "text": "d"},
    ]
    found = cut.rules(records, 240.0)
    assert len(found) == 2, "the three minutes of nothing in the middle comes out"
    assert found[0][1] < 100.0 < found[1][0]


def test_a_session_with_nothing_said_in_it_has_nothing_to_keep() -> None:
    assert cut.rules([{"t": 1.0, "type": "photo"}], 90.0) == ()


def _records():
    return [json.loads(line) for line in LOG.splitlines() if line.strip()]


# ------------------------------------------------------------------ naming it


def test_a_session_with_no_summary_still_gets_a_title(tmp_path) -> None:
    """summary.md is missing on a session that was never described, and the card is still named."""
    folder = make(tmp_path, "2026-09-01_18-13-08_bolt-torque", log=LOG, summary=None)
    title, desc = cut.naming(folder, "", "", 42.0)
    assert title == "Bolt torque"
    assert desc, "a description is never empty either"


def test_an_empty_title_falls_back_to_the_one_the_list_would_show(tmp_path) -> None:
    folder = make(tmp_path, "2026-09-01_18-13-08_bolt", log=LOG, summary=SUMMARY)
    assert cut.naming(folder, "", "", 42.0)[0] == "How tight the bolt goes"


def test_a_folder_nobody_has_named_yet_still_gets_a_title(tmp_path) -> None:
    folder = make(tmp_path, "2026-09-01_18-13-08", log=LOG, summary=None)
    assert cut.naming(folder, "", "", 42.0)[0].startswith("Session 2026-09-01")


def test_the_models_own_title_outranks_the_summary(tmp_path) -> None:
    folder = make(tmp_path, "2026-09-01_18-13-08_bolt", log=LOG, summary=SUMMARY)
    assert cut.naming(folder, "Bled the rear brake", "", 42.0)[0] == "Bled the rear brake"


# ------------------------------------------------------------------ where a moment ends up


def test_time_moves_with_the_cut() -> None:
    """The whole caption feature is this function being right.

    A record's ``t`` is a position in video.mp4; after the trims, the concat and the title card
    it is somewhere else, and a caption at the wrong somewhere is worse than none at all.
    """
    ranges = ((10.0, 20.0), (40.0, 50.0))
    assert cut.shift(15.0, ranges) == pytest.approx(2.0 + 5.0)  # inside the first
    assert cut.shift(45.0, ranges) == pytest.approx(2.0 + 10.0 + 5.0)  # inside the second
    assert cut.shift(30.0, ranges) is None  # in the gap that was cut out
    assert cut.shift(99.0, ranges) is None  # after the end of the last one


def test_a_caption_that_straddles_a_join_is_clipped_rather_than_dropped() -> None:
    ranges = ((10.0, 20.0), (40.0, 50.0))
    pieces = cut._clip_into(18.0, 25.0, ranges)
    assert len(pieces) == 1
    assert pieces[0][1] == pytest.approx(2.0 + 10.0), "clipped to the end of the shot it is in"


def test_a_sliver_left_after_clipping_is_dropped_rather_than_flashed() -> None:
    assert cut._clip_into(19.9, 25.0, ((10.0, 20.0),)) == []


# ------------------------------------------------------------------ the subtitle file


def _plan(**over):
    base = dict(ranges=((10.0, 30.0), (55.0, 75.0)), title="Bolt torque", desc="d", seconds=90.0)
    return cut.Plan(**(base | over))


def test_the_script_carries_both_cards_and_the_captions(tmp_path) -> None:
    text = cut.script(_records(), _plan(), None)
    assert "[Script Info]" in text and "PlayResX: 1280" in text
    assert text.count("\nStyle: ") == 4, "one per voice, plus the two card styles"
    assert "0:00:00.00,0:00:02.00,Card" in text, "the title card sits at the very front"
    assert "CYCLOPS" in text, "and the end card at the back"
    assert "newton metres" in text, "with what was said in between"


def test_a_caption_never_outlives_the_video_it_is_on() -> None:
    plan = _plan()
    for line in cut.script(_records(), plan, None).splitlines():
        if not line.startswith("Dialogue:"):
            continue
        _, start, end, *_ = line.split(",", 3)
        assert _seconds(end) <= plan.total_s + 0.01, line


def _seconds(stamp: str) -> float:
    hours, minutes, seconds = stamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def test_a_line_that_was_cut_out_gets_no_caption() -> None:
    """The photo at 30 s and the turn at 60 s are both outside the one range kept here."""
    text = cut.script(_records(), _plan(ranges=((10.0, 25.0),)), None)
    assert "newton metres" in text
    assert "caliper bolts" not in text


def test_a_long_turn_becomes_several_captions() -> None:
    long = "word " * 80
    records = [{"t": 12.0, "type": "you", "text": long.strip(), "dur": 20.0}]
    plan = _plan(ranges=((10.0, 40.0),))
    text = cut.script(records, plan, None)
    spoken = [one for one in text.splitlines()
              if one.startswith("Dialogue:") and "You," in one]
    assert len(spoken) > 1, "one unreadable wall of text is the same as no captions"


def test_a_short_caption_does_not_get_as_long_as_a_long_one() -> None:
    """Found on the Pi: a full line flicked past and "calipers." sat there for four seconds.

    Reading time is about how much there is to read, so a turn's span is shared out by the
    length of each chunk rather than by how many of them there are.
    """
    long_first, short_last = cut._spread(0.0, 12.0, ["a" * 80, "tail"])
    assert (long_first[1] - long_first[0]) > (short_last[1] - short_last[0])
    assert short_last[1] == pytest.approx(12.0, abs=1.3), "and they still fill the turn"


def test_one_chunk_takes_the_whole_turn() -> None:
    assert cut._spread(2.0, 9.0, ["just the one"]) == [(2.0, 9.0)]


def test_nothing_a_model_wrote_can_reach_the_subtitle_grammar() -> None:
    """A title is a string somebody's model chose. Restrict, do not escape."""
    nasty = 'a{b}c\\d\ne:f'
    text = cut.script([], _plan(title=nasty), None)
    card_line = next(one for one in text.splitlines() if ",Card," in one)
    assert "{" not in card_line.split(",,", 1)[1].replace("{\\pos(640,320)}", "")
    assert "\n" not in card_line


# ------------------------------------------------------------------ the command


def test_the_command_is_one_encode_of_one_input() -> None:
    argv = cut.cut_command(_plan(), ".cut.mp4.tmp")
    assert argv.count("-i") == 1
    assert argv.count("libx264") == 1
    assert "-f" in argv and argv[argv.index("-f") + 1] == "mp4", "the output is a scratch name"
    assert argv[-1] == ".cut.mp4.tmp"


def test_the_command_uses_bare_filenames_so_nothing_in_it_needs_escaping(tmp_path) -> None:
    """subtitles= is parsed as filter grammar, and a folder name comes partly from a model."""
    argv = cut.cut_command(_plan(), ".cut.mp4.tmp")
    assert card.VIDEO in argv
    named = [w for w in argv if w.endswith((".mp4", ".ass", ".tmp"))]
    assert named, "it names files at all"
    for word in named:
        assert "/" not in word and ":" not in word and "\\" not in word, word
    graph = argv[argv.index("-filter_complex") + 1]
    assert f"subtitles={card.CUT_SUBS}," in graph, "a bare name, in ffmpeg's own cwd"


def test_the_command_pads_the_cards_on_and_folds_the_two_voices_together() -> None:
    graph = cut.cut_command(_plan(), "x.tmp")[
        cut.cut_command(_plan(), "x.tmp").index("-filter_complex") + 1
    ]
    assert "tpad=start_duration=2.0:stop_duration=2.0" in graph
    # Left is the microphone and right is Cyclops; headphones need both in both ears.
    assert "pan=stereo" in graph
    assert "concat=n=2:v=1:a=1" in graph


# ------------------------------------------------------------------ the state on the card


def test_a_session_nobody_asked_about_has_no_video(tmp_path) -> None:
    assert cut.state(make(tmp_path, "2026-09-01_18-13-08", log=LOG)) == ""


def test_the_request_is_the_queue(tmp_path) -> None:
    """Absence of the derived file is the work item - after.py's bargain, and captions.py's."""
    folder = make(tmp_path, "2026-09-01_18-13-08", log=LOG, request="{}")
    assert cut.state(folder) == "asked"
    assert cut.waiting(tmp_path) == [folder]


def test_a_scratch_file_beside_the_target_means_a_render_is_running(tmp_path) -> None:
    folder = make(tmp_path, "2026-09-01_18-13-08", log=LOG, request="{}")
    card.tmp_for(folder / card.CUT).write_bytes(b"half a video")
    assert cut.state(folder) == "cutting"


def test_a_finished_video_outranks_everything_else_in_the_folder(tmp_path) -> None:
    folder = make(tmp_path, "2026-09-01_18-13-08", log=LOG, request="{}", made=b"mp4")
    assert cut.state(folder) == "done"


def test_a_zero_byte_video_is_not_a_video(tmp_path) -> None:
    """card.written, not is_file - the incident card.py's docstring is about."""
    folder = make(tmp_path, "2026-09-01_18-13-08", log=LOG, made=b"")
    assert cut.state(folder) != "done"


def test_a_failure_is_written_down_where_the_page_can_read_it(tmp_path) -> None:
    folder = make(tmp_path, "2026-09-01_18-13-08", log=LOG, plan={"why": "No space left"})
    assert cut.state(folder) == "failed"


def test_asking_again_replaces_the_video_it_had(tmp_path) -> None:
    folder = make(tmp_path, "2026-09-01_18-13-08", log=LOG, made=b"old", plan={"title": "old"})
    cut.request(folder)
    assert not (folder / card.CUT).exists(), "a re-cut is a replacement, not a second file"
    # "Re-cut" means the choice was wrong, not the encode, so the plan goes with it.
    assert not (folder / card.CUT_PLAN).exists()
    assert cut.state(folder) == "asked"


# ------------------------------------------------------------------ the sweep


def test_a_session_that_is_still_recording_is_never_cut(tmp_path, monkeypatch) -> None:
    """x264 at 200% of a core, against a realtime audio thread, on four cores."""
    folder = make(tmp_path, "2026-09-01_18-13-08", log=LOG, request="{}")
    monkeypatch.setattr(cut.card, "locked", lambda f: f == folder)
    assert "waiting for the conversation" in cut.busy(tmp_path)
    assert "waiting for the conversation" in cut.one(settings_for(tmp_path))
    assert (folder / card.CUT_REQUEST).exists(), "the request survives; this is a pause"


def test_a_hot_board_is_left_to_cool(tmp_path, monkeypatch) -> None:
    make(tmp_path, "2026-09-01_18-13-08", log=LOG, request="{}")
    monkeypatch.setattr(cut.card, "locked", lambda f: False)
    monkeypatch.setattr(cut.stats, "cpu_temp_c", lambda: 82.0)
    assert cut.busy(tmp_path) == "too hot to render"


def test_a_session_with_no_recording_is_taken_off_the_queue(tmp_path, monkeypatch) -> None:
    folder = make(tmp_path, "2026-09-01_18-13-08", log=LOG, video=None, request="{}")
    monkeypatch.setattr(cut.card, "locked", lambda f: False)
    monkeypatch.setattr(cut.stats, "cpu_temp_c", lambda: 50.0)
    assert "no recording" in cut.one(settings_for(tmp_path))
    assert not (folder / card.CUT_REQUEST).exists(), "it will never succeed; stop asking"


def test_a_recording_with_no_picture_in_it_says_so_in_a_sentence(tmp_path, monkeypatch) -> None:
    """Three sessions on the card turned out to be sound only - a camera that never gave a frame
    still muxes, and still leaves several megabytes. ffmpeg's answer to that is half a page of
    filtergraph ending "matches no streams", which is no use to somebody who pressed a button."""
    folder = make(tmp_path, "2026-09-01_18-13-08_quiet", log=LOG, summary=SUMMARY, request="{}")
    monkeypatch.setattr(cut.card, "locked", lambda f: False)
    monkeypatch.setattr(cut.stats, "cpu_temp_c", lambda: 50.0)
    monkeypatch.setattr(cut, "probe", lambda folder, records: (90.0, False))
    ran = []
    monkeypatch.setattr(cut, "render", lambda *a: ran.append(1) or cut.Cut(True))

    assert "no picture" in cut.one(settings_for(tmp_path))
    assert ran == [], "and ffmpeg is never asked to try"
    assert not (folder / card.CUT_REQUEST).exists(), "it will not become true later"
    assert cut.read_plan(folder).why == "that recording has sound but no picture in it"


def test_switching_it_off_leaves_the_queue_alone(tmp_path) -> None:
    folder = make(tmp_path, "2026-09-01_18-13-08", log=LOG, request="{}")
    assert cut.one(settings_for(tmp_path, cut=False)) == ""
    assert (folder / card.CUT_REQUEST).exists()


def test_only_one_video_is_made_per_sweep(tmp_path, monkeypatch) -> None:
    """The finished file rings the index service's own bell, which schedules the next one."""
    for name in ("2026-09-01_10-00-00", "2026-09-01_11-00-00", "2026-09-01_12-00-00"):
        make(tmp_path, name, log=LOG, request="{}")
    monkeypatch.setattr(cut.card, "locked", lambda f: False)
    monkeypatch.setattr(cut.stats, "cpu_temp_c", lambda: 50.0)
    made = []
    def one_render(folder, plan, root):
        made.append(folder)
        return cut.Cut(True)

    monkeypatch.setattr(cut, "render", one_render)
    monkeypatch.setattr(cut, "duration", lambda folder, records: 90.0)
    cut.one(settings_for(tmp_path))
    assert len(made) == 1


def test_no_key_still_produces_a_video(tmp_path, monkeypatch) -> None:
    """No key, no network, a refusing model - the rules cut it and the button has not lied."""
    folder = make(tmp_path, "2026-09-01_18-13-08_bolt", log=LOG, summary=SUMMARY, request="{}")
    monkeypatch.setattr(cut.card, "locked", lambda f: False)
    monkeypatch.setattr(cut.stats, "cpu_temp_c", lambda: 50.0)
    monkeypatch.setattr(cut, "duration", lambda folder, records: 90.0)
    monkeypatch.setattr(cut, "render", lambda folder, plan, root: cut.Cut(True))
    cut.one(settings_for(tmp_path))  # api_key="" - decide() never builds a client
    plan = cut.read_plan(folder)
    assert plan is not None and plan.by == "rules"
    assert plan.ranges, "and it has real ranges in it"
    assert plan.title == "How tight the bolt goes", "named from summary.md"
    assert (folder / card.CUT_SUBS).exists(), "with a subtitle file rendered from them"


def test_a_killed_render_keeps_its_plan_and_never_pays_the_model_twice(
    tmp_path, monkeypatch
) -> None:
    """The deploy path. push.sh restarts this service, and the cgroup takes ffmpeg with it."""
    folder = make(tmp_path, "2026-09-01_18-13-08_bolt", log=LOG, summary=SUMMARY, request="{}")
    monkeypatch.setattr(cut.card, "locked", lambda f: False)
    monkeypatch.setattr(cut.stats, "cpu_temp_c", lambda: 50.0)
    monkeypatch.setattr(cut, "duration", lambda folder, records: 90.0)
    paused = cut.Cut(False, "paused for a conversation")
    monkeypatch.setattr(cut, "render", lambda folder, plan, root: paused)

    asked = []
    monkeypatch.setattr(cut, "decide", lambda *a: asked.append(1) or ("", "", (), False))
    cut.one(settings_for(tmp_path))
    assert (folder / card.CUT_REQUEST).exists(), "a pause leaves the request where it was"

    cut.one(settings_for(tmp_path))
    assert len(asked) == 1, "the second attempt renders the plan already on the card"


def test_a_render_that_actually_failed_stops_being_retried(tmp_path, monkeypatch) -> None:
    """An ffmpeg that refused this input will refuse it every fifteen minutes for ever."""
    folder = make(tmp_path, "2026-09-01_18-13-08_bolt", log=LOG, summary=SUMMARY, request="{}")
    monkeypatch.setattr(cut.card, "locked", lambda f: False)
    monkeypatch.setattr(cut.stats, "cpu_temp_c", lambda: 50.0)
    monkeypatch.setattr(cut, "duration", lambda folder, records: 90.0)
    monkeypatch.setattr(cut, "render", lambda folder, plan, root: cut.Cut(False, "No space left"))
    cut.one(settings_for(tmp_path))
    assert not (folder / card.CUT_REQUEST).exists()
    assert cut.state(folder) == "failed"
    assert cut.read_plan(folder).why == "No space left"


# ------------------------------------------------------------------ the render itself


def test_a_missing_ffmpeg_is_a_reason_and_not_a_crash(tmp_path, monkeypatch) -> None:
    """systemd's PATH is not a login shell's - record.mux's own note."""
    folder = make(tmp_path, "2026-09-01_18-13-08", log=LOG)
    monkeypatch.setattr(cut.subprocess, "Popen", _raise(FileNotFoundError("no ffmpeg")))
    done = cut.render(folder, _plan(), tmp_path)
    assert not done.ok and "FileNotFoundError" in done.why


def test_a_failed_run_leaves_no_scratch_and_no_video(tmp_path, monkeypatch) -> None:
    folder = make(tmp_path, "2026-09-01_18-13-08", log=LOG)
    monkeypatch.setattr(cut.subprocess, "Popen", _fake_popen(1, b"could not open cut.ass"))
    done = cut.render(folder, _plan(), tmp_path)
    assert not done.ok and done.why == "could not open cut.ass"
    assert not card.tmp_for(folder / card.CUT).exists()
    assert not (folder / card.CUT).exists()


def test_a_video_lands_only_when_ffmpeg_returned_zero(tmp_path, monkeypatch) -> None:
    folder = make(tmp_path, "2026-09-01_18-13-08", log=LOG)

    def popen(argv, **kw):
        card.tmp_for(folder / card.CUT).write_bytes(b"a finished video")
        return _Proc(0, b"")

    monkeypatch.setattr(cut.subprocess, "Popen", popen)
    assert cut.render(folder, _plan(), tmp_path).ok
    assert card.written(folder / card.CUT)
    assert not card.tmp_for(folder / card.CUT).exists()


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


def test_the_videos_list_holds_the_finished_ones_newest_first(tmp_path) -> None:
    make(tmp_path, "2026-09-01_10-00-00_early", log=LOG, summary=SUMMARY, made=b"mp4")
    make(tmp_path, "2026-09-03_10-00-00_late", log=LOG, summary=SUMMARY, made=b"mp4")
    make(tmp_path, "2026-09-02_10-00-00_never-asked", log=LOG, summary=SUMMARY)
    found = cut.videos(tmp_path)
    assert [one.name.split("_")[-1] for one in found] == ["late", "early"]
    assert all(one.state == "done" for one in found)


def test_the_ones_still_being_made_are_in_the_list_too(tmp_path) -> None:
    """Pressing the button and then finding nothing on this screen is the outcome to avoid."""
    make(tmp_path, "2026-09-01_10-00-00_waiting", log=LOG, request="{}")
    found = cut.videos(tmp_path)
    assert [one.state for one in found] == ["asked"]
    assert found[0].title, "and it is named even before there is a plan"


# ------------------------------------------------------------------ the card knows about them


def test_every_file_a_cut_leaves_is_something_a_session_may_hold() -> None:
    """Without this, surprises() calls a video "not ours" and the folder can never be removed."""
    from cyclops import session

    names = {card.CUT_REQUEST, card.CUT_PLAN, card.CUT_SUBS, card.CUT}
    assert names <= card.KNOWN
    source = (session.__file__ and open(session.__file__, encoding="utf-8").read()) or ""
    unlinks = source.split("for name in (", 1)[1].split("):", 1)[0]
    for name in ("CUT_REQUEST", "CUT_PLAN", "CUT_SUBS", "CUT"):
        assert f"card.{name}" in unlinks, f"session._remove would leave {name} behind"


def test_a_folder_holding_only_a_cut_is_not_a_surprise(tmp_path) -> None:
    folder = make(tmp_path, "2026-09-01_18-13-08", log=LOG, request="{}", plan={}, made=b"mp4")
    (folder / card.CUT_SUBS).write_text("[Script Info]\n", encoding="utf-8")
    assert card.surprises(folder) == []
