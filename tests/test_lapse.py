"""The time-lapse recorder alone: a run, its folder, its stops, what an unplug leaves (hq#669).

No ffmpeg, camera or Pi. ffmpeg is a stand-in that counts the bytes it was piped and, asked to
remux, writes a file where the real one would. Frames come every 0 s instead of every second.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cyclops import card, lapse

FRAME = np.zeros((1296, 2304, 3), dtype=np.uint8)
PLENTY = 50 * 1024**3


class Pipe:
    def __init__(self, broken_after: int | None = None) -> None:
        self.frames, self.closed, self._broken_after = 0, False, broken_after

    def write(self, data: bytes) -> None:
        if self._broken_after is not None and self.frames >= self._broken_after:
            raise BrokenPipeError("ffmpeg died")
        self.frames += 1

    def close(self) -> None:
        self.closed = True


class FakeFFmpeg:
    """Popen for the encoder (writes the raw file as it starts) and run for remux and ffprobe."""

    def __init__(self, broken_after: int | None = None, remux_fails: bool = False) -> None:
        self.pipe = Pipe(broken_after)
        self.remux_fails = remux_fails
        self.argv: list[list[str]] = []

    def popen(self, argv, **_):
        self.argv.append(argv)
        Path(argv[-1]).write_bytes(b"fragmented")
        return SimpleNamespace(stdin=self.pipe, wait=lambda timeout=None: 0)

    def run(self, argv, **_):
        self.argv.append(argv)
        if argv[0] == "ffprobe":
            return SimpleNamespace(
                returncode=0, stdout=f"{self.pipe.frames / 30:.3f}\n", stderr=b""
            )
        if self.remux_fails:
            return SimpleNamespace(returncode=1, stdout=b"", stderr=b"moov atom not found\n")
        Path(argv[-1]).write_bytes(b"playable")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")


@pytest.fixture(autouse=True)
def fast(monkeypatch) -> None:
    monkeypatch.setattr(lapse, "CAPTURE_EVERY_S", 0.0)
    monkeypatch.setattr(lapse, "SPACE_EVERY_S", 0.0)
    monkeypatch.setattr(lapse, "FIRST_FRAME_S", 0.05)


def recorder(tmp_path: Path, ff: FakeFFmpeg, free=lambda _p: PLENTY, frames=lambda: FRAME, **more):
    return lapse.TimeLapse(tmp_path, frames, popen=ff.popen, run=ff.run, free_bytes=free, **more)


def wait_for(condition, timeout: float = 2.0) -> None:
    end = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.001)


def test_a_run_is_live_while_it_records_and_a_finished_session_after(tmp_path) -> None:
    ff = FakeFFmpeg()
    lapser = recorder(tmp_path, ff)
    assert lapser.start() == ""
    wait_for(lambda: lapser.frames >= 5)
    folder = lapser.folder
    assert folder.name.endswith("_time-lapse") and lapse.is_lapse(folder)
    assert card.triage(folder).verdict == "live"  # recovery leaves it alone mid-run

    last = lapser.stop()
    state = card.triage(folder)
    assert (
        state.verdict == "finished"
        and state.video
        and not state.parts
        and state.summary
        and state.named
    )
    assert not lapser.running() and last["saved"] and last["stopped"] == lapse.ASKED
    assert last["seconds"] == pytest.approx(ff.pipe.frames / 30, abs=0.01)


def test_the_panel_length_is_frames_over_thirty(tmp_path) -> None:
    lapser = recorder(tmp_path, FakeFFmpeg())
    lapser.frames = 360
    assert lapser.seconds() == 12.0


def test_the_encoder_writes_fragments_and_no_sound(tmp_path) -> None:
    ff = FakeFFmpeg()
    lapser = recorder(tmp_path, ff)
    lapser.start()
    lapser.stop()
    argv = ff.argv[0]
    assert "-an" in argv and "2304x1296" in argv and argv[argv.index("-framerate") + 1] == "30"
    assert "frag_keyframe" in argv[argv.index("-movflags") + 1]
    assert argv[argv.index("-g") + 1] == str(lapse.KEYFRAME_EVERY)


def test_a_second_run_is_refused_while_one_records(tmp_path) -> None:
    lapser = recorder(tmp_path, FakeFFmpeg())
    lapser.start()
    try:
        assert lapser.start() != "" and len(list(tmp_path.iterdir())) == 1
    finally:
        lapser.stop()


@pytest.mark.parametrize(
    "free, frames",
    [(lambda _p: 1024**3, lambda: FRAME), (lambda _p: PLENTY, lambda: None)],
    ids=["no space", "no camera picture"],
)
def test_a_refused_start_leaves_nothing_on_the_card(tmp_path, free, frames) -> None:
    lapser = recorder(tmp_path, FakeFFmpeg(), free=free, frames=frames)
    assert lapser.start() != "" and not lapser.running() and list(tmp_path.iterdir()) == []


def test_it_stops_itself_at_the_space_line_saves_and_says_why(tmp_path) -> None:
    left = iter([PLENTY] + [PLENTY] * 3 + [1024**3] * 100)
    told: list[str] = []
    lapser = recorder(tmp_path, FakeFFmpeg(), free=lambda _p: next(left), on_stop=told.append)
    lapser.start()
    wait_for(lambda: not lapser.running())
    assert (
        told == [lapse.NO_SPACE]
        and lapser.last["saved"]
        and lapser.last["stopped"] == lapse.NO_SPACE
    )
    assert "nearly full" in (lapser.folder / card.SUMMARY_NAME).read_text()


def test_a_broken_encoder_stops_the_run_and_keeps_what_was_written(tmp_path) -> None:
    told: list[str] = []
    lapser = recorder(tmp_path, FakeFFmpeg(broken_after=3), on_stop=told.append)
    lapser.start()
    wait_for(lambda: not lapser.running())
    assert (
        told == [lapse.BROKE] and lapser.last["saved"] and card.written(lapser.folder / card.VIDEO)
    )


def test_a_stalled_camera_repeats_the_last_picture_instead_of_leaving_a_gap(tmp_path) -> None:
    pictures = iter([FRAME, FRAME] + [None] * 1000)
    ff = FakeFFmpeg()
    lapser = recorder(tmp_path, ff, frames=lambda: next(pictures, None))
    lapser.start()
    wait_for(lambda: lapser.frames >= 20)
    lapser.stop()
    assert ff.pipe.frames == lapser.frames >= 20


# ---- after an unplug ----


def unplugged(tmp_path: Path) -> Path:
    """What a power cut mid-run leaves: the mark, a log nobody holds, and a raw file in parts/."""
    folder = tmp_path / "2026-11-08_01-00-00_time-lapse"
    (folder / card.PARTS).mkdir(parents=True)
    (folder / card.PARTS / lapse.RAW).write_bytes(b"fragments up to the unplug")
    (folder / lapse.MARK).write_text('{"started": "2026-11-08_01-00-00"}')
    (folder / card.LOG_NAME).write_text(
        '{"t": 0, "type": "session"}\n{"t": 0, "type": "timelapse"}\n'
    )
    return folder


def test_recovery_turns_an_unplugged_run_into_a_playable_video(tmp_path) -> None:
    folder = unplugged(tmp_path)
    assert card.triage(folder).verdict == "unfinished"
    ok, _ = lapse.finish(folder, FakeFFmpeg().run)
    state = card.triage(folder)
    assert ok and state.verdict == "finished" and state.video and not state.parts


def test_a_remux_that_fails_keeps_the_footage_for_the_next_boot(tmp_path) -> None:
    folder = unplugged(tmp_path)
    ok, why = lapse.finish(folder, FakeFFmpeg(remux_fails=True).run)
    assert not ok and "moov" in why and card.written(folder / card.PARTS / lapse.RAW)
    assert not (folder / card.VIDEO).exists()


def test_real_ffmpeg_reads_a_fragmented_file_cut_off_mid_run(tmp_path) -> None:
    """The one claim the whole unplug story rests on, with the real ffmpeg when there is one."""
    if subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0:
        pytest.skip("no ffmpeg here")
    folder = unplugged(tmp_path)
    raw = folder / card.PARTS / lapse.RAW
    frame = np.zeros((96, 128, 3), dtype=np.uint8)
    enc = subprocess.Popen(
        lapse.encoder_argv(128, 96, raw), stdin=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    for n in range(150):
        frame[:] = n
        enc.stdin.write(frame.tobytes())
    enc.stdin.flush()
    time.sleep(0.2)
    enc.kill()  # no clean close: the last fragment is torn, as an unplug leaves it
    enc.wait()
    ok, why = lapse.finish(folder)
    assert ok, why
    assert lapse.probe_seconds(folder / card.VIDEO) >= 2.0  # at least the first two full fragments


def test_boot_recovery_finishes_a_time_lapse_without_the_sound_a_session_mux_needs(
    tmp_path, monkeypatch
) -> None:
    from cyclops import session

    folder = unplugged(tmp_path)
    monkeypatch.setattr(subprocess, "run", FakeFFmpeg().run)
    session._fix(folder)
    state = card.triage(folder)
    assert state.verdict == "finished" and state.video and state.summary and state.named


def finished(tmp_path: Path, name: str, lapsed: bool) -> Path:
    folder = tmp_path / name
    folder.mkdir()
    for file in (card.VIDEO, card.PAGE_NAME, card.SUMMARY_NAME):
        (folder / file).write_text("x")
    (folder / card.LOG_NAME).write_text('{"t": 0, "type": "recording_asked"}\n')
    if lapsed:
        (folder / lapse.MARK).write_text("{}")
    return folder


def test_filing_and_the_highlight_cut_leave_a_time_lapse_alone(tmp_path) -> None:
    from cyclops import cut, projects

    talk = finished(tmp_path, "2026-11-08_01-00-00_bench", lapsed=False)
    finished(tmp_path, "2026-11-08_02-00-00_time-lapse", lapsed=True)
    assert projects.unfiled(SimpleNamespace(sessions_dir=tmp_path)) == [talk]
    assert cut.work(tmp_path) == cut.Work(talk, -1)


# ---- by voice, from any conversation ----


class FakeConn:
    def __init__(self) -> None:
        self.items: list[dict] = []
        self.response = SimpleNamespace(create=self._response)
        self.conversation = SimpleNamespace(item=SimpleNamespace(create=self._item))

    async def _response(self, **_kwargs) -> None: ...

    async def _item(self, item: dict) -> None:
        self.items.append(item)


@pytest.fixture
def installed(tmp_path):
    lapser = recorder(tmp_path, FakeFFmpeg())
    lapse.install(lapser)
    yield lapser
    if lapser.running():
        lapser.stop()
    lapse.install(None)


def conversation(tmp_path, monkeypatch):
    from cyclops import agent
    from cyclops.config import Settings

    made = agent.VoiceAgent(Settings(api_key="", sessions_dir=tmp_path, slug=False, projects=False))
    made._conn = FakeConn()
    monkeypatch.setattr(agent.VoiceAgent, "connected", property(lambda self: True))
    monkeypatch.setattr(made, "_log", lambda *a, **k: None)
    return made


def ask(voice, action: str) -> dict:
    import asyncio
    import json

    call = SimpleNamespace(name="timelapse", call_id="c", arguments=json.dumps({"action": action}))
    asyncio.run(voice._run_timelapse(call))
    found = [i for i in voice._conn.items if i.get("type") == "function_call_output"]
    return json.loads(found[-1]["output"])


def test_a_later_conversation_finds_and_stops_the_run_an_earlier_one_started(
    tmp_path, monkeypatch, installed
) -> None:
    first = conversation(tmp_path, monkeypatch)
    assert ask(first, "start")["running"] is True
    wait_for(lambda: installed.frames >= 3)
    later = conversation(tmp_path, monkeypatch)  # the first one has ended; the run hasn't
    assert ask(later, "status")["running"] is True
    stopped = ask(later, "stop")
    assert stopped["ok"] and stopped["saved"] and "s of video" in stopped["video"]
    assert ask(later, "status") == {"ok": True, "running": False}


def test_stop_and_status_never_claim_a_run_that_isnt_there(
    tmp_path, monkeypatch, installed
) -> None:
    voice = conversation(tmp_path, monkeypatch)
    assert ask(voice, "status")["running"] is False
    assert ask(voice, "stop")["ok"] is False


def test_a_box_with_no_time_lapse_says_so(tmp_path, monkeypatch) -> None:
    lapse.install(None)
    assert ask(conversation(tmp_path, monkeypatch), "start")["ok"] is False


def test_a_video_and_a_time_lapse_never_run_together(tmp_path, monkeypatch, installed) -> None:
    from cyclops import session

    monkeypatch.setattr(session, "current", lambda: SimpleNamespace(recording=True))
    voice = conversation(tmp_path, monkeypatch)
    assert ask(voice, "start")["ok"] is False and not installed.running()

    monkeypatch.setattr(session, "current", lambda: None)
    assert ask(voice, "start")["ok"] is True
    assert lapse.running_now()  # and session.start_recording refuses on exactly this


def test_a_run_that_stopped_by_itself_is_told_once_at_a_pause(tmp_path, monkeypatch) -> None:
    import asyncio

    left = iter([PLENTY] + [1024**3] * 100)
    lapser = recorder(tmp_path, FakeFFmpeg(), free=lambda _p: next(left))
    lapse.install(lapser)
    try:
        lapser.start()
        wait_for(lambda: not lapser.running())
        voice = conversation(tmp_path, monkeypatch)
        assert asyncio.run(voice._tell_lapse_if_due()) is True
        assert asyncio.run(voice._tell_lapse_if_due()) is False
        told = [
            p["text"] for i in voice._conn.items if i.get("role") == "user" for p in i["content"]
        ]
        assert len(told) == 1 and "nearly full" in told[0]
    finally:
        lapse.install(None)


def test_a_normal_video_is_refused_while_a_time_lapse_runs(
    tmp_path, monkeypatch, installed
) -> None:
    from cyclops import session
    from cyclops.config import Settings

    made: list = []
    monkeypatch.setattr(session, "SessionRecorder", lambda *a, **k: made.append(a) or pytest.fail())
    settings = Settings(
        api_key="", sessions_dir=tmp_path / "s", slug=False, projects=False, record=True
    )
    log = session.SessionLog(
        settings, SimpleNamespace(on_event=None, ended_by=None), entrypoint="kiosk",
        mic=SimpleNamespace(on_block=None), speaker=SimpleNamespace(on_block=None),
        frames=SimpleNamespace(frame=lambda: FRAME),
    )  # fmt: skip
    installed.start()
    with log:
        assert "time-lapse" in log.start_recording() and made == []


def test_the_camera_is_asked_for_whole_frames_for_exactly_as_long_as_a_run(tmp_path) -> None:
    held: list[bool] = []
    lapser = recorder(tmp_path, FakeFFmpeg(), hold=held.append)
    lapser.start()
    assert held == [True]
    lapser.stop()
    assert held == [True, False]
    recorder(tmp_path, FakeFFmpeg(), frames=lambda: None, hold=held.append).start()
    assert held == [True, False, True, False]  # a refused start gives the framing back


def test_a_run_waits_out_the_camera_reopening_for_its_first_frame(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(lapse, "FIRST_FRAME_S", 2.0)
    pictures = iter([None, None, None, FRAME])
    ff = FakeFFmpeg()
    lapser = recorder(tmp_path, ff, frames=lambda: next(pictures, FRAME))
    assert lapser.start() == ""
    lapser.stop()
    assert "2304x1296" in ff.argv[0]


# ---- the camera's side ----


def module(frame=None):
    from cyclops import camera, webcam

    source = camera.CameraSource()
    source._index = webcam.RPICAM
    source._frame, source._stamp = frame, time.monotonic()
    return source


def test_the_camera_hands_a_run_nothing_until_whole_frames_arrive() -> None:
    from cyclops import webcam

    source = module(np.zeros((720, 1280, 3), dtype=np.uint8))
    source.hold_full(True)
    assert source.framing == webcam.LAPSE and source.full_frame() is None
    source._full = FRAME
    assert source.full_frame() is FRAME


def test_the_framing_is_kept_during_a_run_and_given_back_after() -> None:
    from cyclops import webcam

    source = module()
    source.cycle_framing()
    was = source.framing
    source.hold_full(True)
    assert source.cycle_framing() is None and source.framing == webcam.LAPSE
    source.hold_full(False)
    assert source.framing == was


def test_the_module_sends_its_whole_wide_frame_and_the_panel_still_gets_its_size() -> None:
    from cyclops import webcam

    width, height = webcam.LAPSE_SIZE
    yuv = np.zeros((height * 3 // 2, width), dtype=np.uint8).tobytes()
    proc = SimpleNamespace(stdout=SimpleNamespace(read=lambda n, _b=[yuv]: _take(_b, n)))
    cap = webcam._RpicamCapture(proc, width, height, (webcam.FRAME_WIDTH, webcam.FRAME_HEIGHT))
    ok, shown = cap.read()
    assert ok and shown.shape[:2] == (webcam.FRAME_HEIGHT, webcam.FRAME_WIDTH)
    assert webcam.bgr_of(cap.full).shape == (height, width, 3)


def test_the_panel_picture_has_the_same_colours_as_the_whole_frame() -> None:
    """Shrinking the planes before converting must not swap or smear U and V."""
    import cv2

    from cyclops import webcam

    width, height = webcam.LAPSE_SIZE
    bgr = np.zeros((height, width, 3), dtype=np.uint8)
    bgr[:, : width // 2] = (30, 160, 220)  # an orange half and a blue half
    bgr[:, width // 2 :] = (200, 90, 20)
    plane = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
    small = cv2.cvtColor(webcam.shrink_i420(plane, (1280, 720)), cv2.COLOR_YUV2BGR_I420)
    whole = cv2.resize(webcam.bgr_of(plane), (1280, 720), interpolation=cv2.INTER_AREA)
    assert small.shape == (720, 1280, 3)
    for x in (300, 1000):
        assert np.abs(small[360, x].astype(int) - whole[360, x].astype(int)).max() <= 6


def _take(buffer: list[bytes], n: int) -> bytes:
    chunk, buffer[0] = buffer[0][:n], buffer[0][n:]
    return chunk


@pytest.mark.parametrize("running, dark", [(True, False), (False, True)])
def test_the_panel_stays_lit_and_keeps_its_camera_for_the_whole_run(monkeypatch, running, dark):
    import threading

    from cyclops import companion, kiosk, overlay

    panel = object.__new__(kiosk.Kiosk)
    panel._asleep, panel._touched_at, panel._menu, panel._wifi = False, 0.0, False, None
    panel._page_busy = threading.Event()
    panel.controller = SimpleNamespace(settings=SimpleNamespace(sleep_after_s=1.0))
    slept: list[bool] = []
    monkeypatch.setattr(panel, "_sleep", lambda: slept.append(True))
    monkeypatch.setattr(companion, "watching", lambda: False)
    monkeypatch.setattr(lapse, "running_now", lambda: running)
    panel._sleeping(overlay.IDLE)
    assert bool(slept) is dark


# ---- a full card, and the readers a time-lapse isn't a conversation for (qa, round 1) ----


def spaceless_run(cmd, **_kwargs):
    """ffmpeg that runs out of card during the copy; ffprobe that reads two seconds."""
    if cmd[0] == "ffmpeg":
        return SimpleNamespace(
            returncode=1, stderr=b"Error closing file: No space left on device", stdout=b""
        )
    return SimpleNamespace(returncode=0, stdout="2.0\n", stderr="")


def test_a_full_card_still_ends_in_a_real_playable_video(tmp_path) -> None:
    """Real ffmpeg: no room for a second copy, so the recording itself becomes the video."""
    if subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0:
        pytest.skip("no ffmpeg here")
    folder = unplugged(tmp_path)
    raw = folder / card.PARTS / lapse.RAW
    frame = np.zeros((96, 128, 3), dtype=np.uint8)
    enc = subprocess.Popen(
        lapse.encoder_argv(128, 96, raw), stdin=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    for n in range(90):
        frame[:] = n
        enc.stdin.write(frame.tobytes())
    enc.stdin.close()
    enc.wait()
    size = raw.stat().st_size
    ok, why = lapse.finish(folder, free_bytes=lambda _p: size)
    assert ok, why
    assert (folder / card.VIDEO).stat().st_size == size  # renamed, not copied
    assert not (folder / card.PARTS).exists()
    assert lapse.probe_seconds(folder / card.VIDEO) == pytest.approx(3.0, abs=0.1)


def test_a_card_that_fills_during_the_copy_still_keeps_the_recording_as_the_video(tmp_path) -> None:
    folder = unplugged(tmp_path)
    ok, _ = lapse.finish(folder, spaceless_run)
    assert ok and (folder / card.VIDEO).read_bytes() == b"fragments up to the unplug"
    assert not card.tmp_for(folder / card.VIDEO).exists()


def test_a_remux_is_still_used_whenever_a_second_copy_fits(tmp_path) -> None:
    calls: list[str] = []

    def spy(cmd, **kw):
        calls.append(cmd[0])
        return FakeFFmpeg().run(cmd, **kw)

    size = len(b"fragments up to the unplug")
    roomy = lapse.finish(unplugged(tmp_path), spy, lambda _p: size + lapse.REMUX_ROOM_BYTES + 1)
    assert roomy[0] and "ffmpeg" in calls


def test_an_unplugged_run_says_how_long_it_recorded(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(lapse, "CAPTURE_EVERY_S", 1.0)
    folder = unplugged(tmp_path)
    assert lapse.finish(folder, spaceless_run)[0]
    assert "1 min of recording" in (folder / card.SUMMARY_NAME).read_text()  # 2 s x 30 x 1 s


@pytest.mark.parametrize(
    "why, saved, ends",
    [(lapse.NO_SPACE, True, "· saved"), (lapse.NO_SPACE, False, "· NOT saved"),
     (lapse.BROKE, False, "· NOT saved")],
)  # fmt: skip
def test_the_panel_says_saved_only_when_it_was(why, saved, ends) -> None:
    from cyclops import kiosk

    assert kiosk._lapse_stopped_line(why, saved).endswith(ends)


def talk_then_lapse(tmp_path: Path) -> SimpleNamespace:
    talk = finished(tmp_path, "2026-11-08_01-00-00_amp", lapsed=False)
    (talk / card.SUMMARY_NAME).write_text("# Fixing the amp\n\nReflowed C12, still hums at idle.\n")
    run = finished(tmp_path, "2026-11-08_02-00-00_time-lapse", lapsed=True)
    (run / card.SUMMARY_NAME).write_text(
        "# Time-lapse\n\n1 s of video, from 12 min of recording.\n"
    )
    return SimpleNamespace(sessions_dir=tmp_path)


def test_a_time_lapse_does_not_replace_the_last_session_recap(tmp_path) -> None:
    from cyclops import session

    text = session.recent_context(talk_then_lapse(tmp_path)).text
    assert "Reflowed C12" in text and "Time-lapse" not in text


def test_a_time_lapse_is_not_a_time_he_switched_cyclops_on(tmp_path) -> None:
    from datetime import datetime

    from cyclops import session

    now = datetime(2026, 11, 8, 12, 0)
    said = str(session.now_context(talk_then_lapse(tmp_path), now))
    assert "2 times today" in said


def held(tmp_path: Path, name: str, lapsed: bool):
    folder = finished(tmp_path, name, lapsed=lapsed)
    handle = (folder / card.LOG_NAME).open("a", encoding="utf-8")
    assert card.claim(handle)
    return handle


def test_a_running_time_lapse_does_not_hold_up_cuts_but_a_conversation_does(
    tmp_path, monkeypatch
) -> None:
    from cyclops import cut

    monkeypatch.setattr(cut.stats, "cpu_temp_c", lambda: None)
    run = held(tmp_path, "2026-11-08_02-00-00_time-lapse", lapsed=True)
    assert cut.busy(tmp_path) == ""
    talk = held(tmp_path, "2026-11-08_03-00-00", lapsed=False)
    assert cut.busy(tmp_path) == "waiting for the conversation to end"
    run.close()
    talk.close()


def test_the_admin_live_view_is_for_conversations_not_a_time_lapse(tmp_path) -> None:
    from cyclops import library

    run = held(tmp_path, "2026-11-08_02-00-00_time-lapse", lapsed=True)
    assert library.live(tmp_path) is None
    talk = held(tmp_path, "2026-11-08_03-00-00", lapsed=False)
    assert library.live(tmp_path) == "2026-11-08_03-00-00"
    run.close()
    talk.close()
