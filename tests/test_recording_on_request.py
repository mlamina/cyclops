"""Sessions record only when asked (arc-eng/hq#602): by default nothing, and asking is honest.

A stand-in recorder and a fake connection: no ffmpeg, no camera, no model. Real encoding is for
the device. What's guarded here is what a session does and what he's told about it.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from cyclops import agent, card, cut, session
from cyclops.config import Settings


class Recorder:
    made: list[Recorder] = []
    start_says: str = ""  # why a start fails, if it should

    def __init__(self, frames, folder: Path, **_kwargs) -> None:
        self.folder = folder
        self.failed = ""
        self.stopped = False
        self.finish_fails = ""
        Recorder.made.append(self)

    def start(self) -> bool:
        self.failed = Recorder.start_says
        return not self.failed

    def stop(self) -> Path | None:
        self.stopped = True
        if self.failed or self.finish_fails:
            self.failed = self.failed or self.finish_fails
            return None
        video = self.folder / card.VIDEO
        video.write_bytes(b"video")
        return video

    def on_mic_block(self, _block: bytes) -> None: ...

    def on_speaker_block(self, _block: bytes) -> None: ...


class FakeAgent:
    on_event = None
    ended_by = None


class FakeConn:
    def __init__(self) -> None:
        self.items: list[dict] = []
        self.responses: list[dict] = []
        self.response = SimpleNamespace(create=self._response)
        self.conversation = SimpleNamespace(item=SimpleNamespace(create=self._item))

    async def _response(self, **kwargs) -> None:
        self.responses.append(kwargs)

    async def _item(self, item: dict) -> None:
        self.items.append(item)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(api_key="", sessions_dir=tmp_path / "sessions", slug=False, projects=False)


@pytest.fixture
def make_log(settings: Settings, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(session, "SessionRecorder", Recorder)
    Recorder.made = []
    Recorder.start_says = ""

    def make(chosen: Settings = settings, **kwargs) -> session.SessionLog:
        kwargs.setdefault("mic", SimpleNamespace(on_block=None))
        kwargs.setdefault("speaker", SimpleNamespace(on_block=None))
        kwargs.setdefault("frames", SimpleNamespace(frame=lambda: None))
        return session.SessionLog(chosen, FakeAgent(), entrypoint="kiosk", **kwargs)

    return make


@pytest.fixture
def voice(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> agent.VoiceAgent:
    made = agent.VoiceAgent(settings)
    made._conn = FakeConn()
    monkeypatch.setattr(agent.VoiceAgent, "connected", property(lambda self: True))
    monkeypatch.setattr(made, "_log", lambda *a, **k: None)
    return made


def _call(action: str) -> SimpleNamespace:
    return SimpleNamespace(name="recording", call_id="c", arguments=json.dumps({"action": action}))


def _answer(voice: agent.VoiceAgent) -> dict:
    found = [i for i in voice._conn.items if i.get("type") == "function_call_output"]
    return json.loads(found[-1]["output"])


def _told(voice: agent.VoiceAgent) -> list[str]:
    return [
        part["text"]
        for item in voice._conn.items
        if item.get("role") == "user"
        for part in item["content"]
    ]


def _records(log: session.SessionLog) -> list[dict]:
    records, _ = card.read_log(log.dir / card.LOG_NAME)
    return records


def test_a_session_records_nothing_unless_asked(make_log) -> None:
    log = make_log()
    with log:
        log.event("you", text="hello")
        assert not log.recording
        assert not session.recording()

    assert Recorder.made == [], "no recorder, so no encoder and no screen capture"
    assert not (log.dir / card.VIDEO).exists()
    assert not (log.dir / "parts").exists()


def test_a_session_with_no_video_is_kept_and_counts_as_finished(make_log) -> None:
    """Filing and the boot sweep must not take a folder with no video for a failed one."""
    log = make_log()
    with log:
        log.event("you", text="the torque is 12 Nm")
        log.event("cyclops", text="Noted.")

    assert log.dir.is_dir()
    assert card.written(log.dir / card.PAGE_NAME)
    assert card.triage(log.dir).verdict != "unfinished"


def test_the_screen_isnt_taken_until_it_records(make_log) -> None:
    screen = SimpleNamespace(calls=[])
    screen.acquire = lambda: screen.calls.append("acquire") or ""
    screen.release = lambda: screen.calls.append("release")
    screen.frame = lambda: None
    log = make_log(screen=screen)
    with log:
        log.event("you", text="hello")
        assert screen.calls == []
        assert log.start_recording() == ""
        assert screen.calls == ["acquire"]
    assert screen.calls == ["acquire", "release"]


def test_starting_and_stopping_makes_a_finished_video(make_log) -> None:
    log = make_log()
    with log:
        log.event("you", text="record this")
        assert log.start_recording() == ""
        assert log.recording and session.recording()
        assert log._mic.on_block is not None
        assert log.stop_recording() == ""
        assert not log.recording
        assert log._mic.on_block is None and log._speaker.on_block is None
    assert (log.dir / card.VIDEO).exists()
    assert any(r["type"] == "video" for r in _records(log))


def test_a_recording_never_stopped_finishes_with_the_session(make_log) -> None:
    log = make_log()
    with log:
        log.event("you", text="record this")
        log.start_recording()

    assert Recorder.made[0].stopped
    assert (log.dir / card.VIDEO).exists()


def test_asked_twice_changes_nothing(make_log) -> None:
    log = make_log()
    with log:
        assert log.stop_recording(), "stop with nothing running says so"
        assert log.start_recording() == ""
        assert log.start_recording(), "start while running says so"
        assert len(Recorder.made) == 1
        assert log.recording


def test_a_second_recording_cant_overwrite_the_first(make_log) -> None:
    log = make_log()
    with log:
        log.start_recording()
        log.stop_recording()
        assert log.start_recording(), "refused, with a reason"
    assert len(Recorder.made) == 1


def test_a_start_that_fails_says_why_and_leaves_it_able_to_try_again(make_log) -> None:
    log = make_log()
    with log:
        Recorder.start_says = "ffmpeg is not installed"
        why = log.start_recording()
        assert "ffmpeg" in why
        assert not log.recording
        Recorder.start_says = ""
        assert log.start_recording() == ""


def test_too_little_space_is_refused_before_anything_starts(make_log, monkeypatch) -> None:
    monkeypatch.setattr(session.shutil, "disk_usage", lambda _p: SimpleNamespace(free=1000))
    log = make_log()
    with log:
        assert log.start_recording()
        assert Recorder.made == []


def test_nothing_to_record_with_is_said(make_log) -> None:
    log = make_log(mic=None)
    with log:
        assert log.start_recording()


def test_switched_off_is_said(tmp_path, make_log) -> None:
    off = Settings(api_key="", sessions_dir=tmp_path / "s", slug=False, record=False)
    log = make_log(off)
    with log:
        assert log.start_recording()
        assert Recorder.made == []


def test_a_mux_that_fails_is_reported_not_claimed(make_log) -> None:
    log = make_log()
    with log:
        log.start_recording()
        Recorder.made[0].finish_fails = "mux failed"
        assert "mux failed" in log.stop_recording()


def test_two_starts_at_once_make_one_recorder(make_log) -> None:
    log = make_log()
    answers: list[str] = []
    barrier = threading.Barrier(6)

    def go() -> None:
        barrier.wait()
        answers.append(log.start_recording())

    with log:
        threads = [threading.Thread(target=go) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(Recorder.made) == 1
        assert answers.count("") == 1


def test_a_start_after_the_session_closed_doesnt_leave_one_running(make_log) -> None:
    log = make_log()
    with log:
        pass

    assert log.start_recording()
    assert Recorder.made == []


def test_the_tool_is_withheld_when_switched_off() -> None:
    def names(on: bool) -> set[str]:
        return {t["name"] for t in agent._offered_tools(Settings(api_key="", record=on))}

    assert "recording" in names(True)
    assert "recording" not in names(False)


def test_the_model_is_told_only_what_really_happened(voice, make_log) -> None:
    log = make_log()
    Recorder.start_says = "ffmpeg is not installed"

    async def go() -> None:
        await voice._run_recording(_call("start"))

    with log:
        asyncio.run(go())

    answer = _answer(voice)
    assert answer["ok"] is False and answer["recording"] is False
    assert voice._conn.responses, "and it gets to say so"


def test_the_tool_starts_and_stops(voice, make_log) -> None:
    log = make_log()

    async def go() -> None:
        await voice._run_recording(_call("start"))
        assert _answer(voice)["recording"] is True
        assert log.recording
        await voice._run_recording(_call("stop"))
        voice.ended_by = "stopped"
        await asyncio.gather(*voice._background)

    with log:
        asyncio.run(go())

    assert _answer(voice)["recording"] is False
    assert (log.dir / card.VIDEO).exists()


def test_no_session_is_a_plain_answer(voice) -> None:
    asyncio.run(voice._run_recording(_call("start")))

    assert _answer(voice)["ok"] is False


def test_quiet_doesnt_end_a_session_while_recording_and_does_after(voice, make_log) -> None:
    log = make_log()
    voice._response_active = False
    with log:
        assert not voice._busy()
        log.start_recording()
        assert voice._busy()
        log.stop_recording()
        assert not voice._busy()


def test_the_limit_stops_it_saves_the_video_and_tells_him(voice, make_log, monkeypatch) -> None:
    monkeypatch.setattr(agent, "RECORD_LIMIT_S", 0)
    monkeypatch.setattr(agent, "RECORD_CHECK_S", 0)
    log = make_log()
    with log:
        log.start_recording()
        asyncio.run(voice._watch_recording(log))
        assert not log.recording

    assert (log.dir / card.VIDEO).exists()
    assert len(_told(voice)) == 1
    assert "time limit" in _told(voice)[0]


def test_a_recording_that_breaks_midway_is_stopped_and_said(voice, make_log, monkeypatch) -> None:
    monkeypatch.setattr(agent, "RECORD_CHECK_S", 0)
    log = make_log()
    with log:
        log.start_recording()
        Recorder.made[0].failed = "writer: disk full"
        asyncio.run(voice._watch_recording(log))
        assert not log.recording

    assert "disk full" in _told(voice)[0]
    assert "video is saved" not in _told(voice)[0]


def test_stopping_by_voice_ends_the_watch_with_nothing_said(voice, make_log, monkeypatch) -> None:
    monkeypatch.setattr(agent, "RECORD_CHECK_S", 0.01)
    log = make_log()

    async def go() -> None:
        watch = asyncio.create_task(voice._watch_recording(log))
        await asyncio.sleep(0.03)
        log.stop_recording()
        await asyncio.wait_for(watch, timeout=2)

    with log:
        log.start_recording()
        asyncio.run(go())

    assert _told(voice) == []


def test_a_recording_he_asked_for_gets_its_highlight_however_quiet() -> None:
    quiet = [{"t": 1.0, "type": "recording_asked"}, {"t": 2.0, "type": "you", "text": "stop"}]

    assert cut.worth_asking(quiet) == ""
    assert cut.worth_asking(quiet[1:]), "without the ask, the old rule still applies"


def test_the_panel_tag_follows_the_recording_not_the_setting(make_log) -> None:
    assert not session.recording()
    log = make_log()
    with log:
        assert not session.recording(), "the setting is on and nothing is being made"
        log.start_recording()
        assert session.recording()
    assert not session.recording()


def test_a_recording_he_asked_for_is_never_cleaned_away(make_log) -> None:
    """With no turn in the log (the transcription failed), it would go as a button press."""
    log = make_log()
    with log:
        log.start_recording()

    assert (log.dir / card.VIDEO).exists()
    records = _records(log)
    assert card.made_something(records), "and so the mic-check tidy and the boot sweep keep it too"


def test_a_button_press_with_no_recording_still_goes(make_log) -> None:
    log = make_log()
    with log:
        pass

    assert not log.dir.exists()
