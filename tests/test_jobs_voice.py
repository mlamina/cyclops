"""The voice side of a background job: handing off at once, and telling him once, at a pause.

A fake connection, no model and no key. Whether the real model says it well is for the device.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from cyclops import agent, indexer, jobs, worker
from cyclops.config import Settings


class FakeConn:
    def __init__(self) -> None:
        self.items: list[dict] = []
        self.responses: list[dict] = []
        self.response = SimpleNamespace(create=self._response)
        self.deleted: list[str] = []
        self.conversation = SimpleNamespace(
            item=SimpleNamespace(create=self._item, delete=self._delete)
        )

    async def _response(self, **kwargs) -> None:
        self.responses.append(kwargs)

    async def _item(self, item: dict) -> None:
        self.items.append(item)

    async def _delete(self, item_id: str) -> None:
        self.deleted.append(item_id)


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setattr(jobs, "LEDGER", tmp_path / "cache" / "jobs.json")
    monkeypatch.setattr(jobs, "LEDGER_LOCK", tmp_path / "cache" / "jobs.lock")
    root = tmp_path / "worker"
    worker.prepare(root)
    return Settings(
        api_key="", sounds=False, worker=True, worker_dir=root, manuals_dir=tmp_path / "manuals"
    )


@pytest.fixture
def voice(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> agent.VoiceAgent:
    made = agent.VoiceAgent(settings)
    made._conn = FakeConn()
    monkeypatch.setattr(agent.VoiceAgent, "connected", property(lambda self: True))
    monkeypatch.setattr(made, "_log", lambda *a, **k: None)
    monkeypatch.setattr(agent.session, "note", lambda *a, **k: None)
    made._heard_at = time.monotonic() - 60
    return made


def _call(request: str = "the relay manual") -> SimpleNamespace:
    return SimpleNamespace(
        name="background_job", call_id="c1", arguments=json.dumps({"request": request})
    )


def _failed_job(settings: Settings) -> str:
    """A job that has finished, ready to be told. A failure, because that needs no real PDF."""
    assert jobs.submit(settings, "the relay manual") == jobs.STARTED
    job_id = jobs._read()[-1]["id"]
    (settings.worker_dir / worker.DONE / f"{job_id}.json").write_text(
        json.dumps({"state": "failed", "reason": worker.NOTHING_FOUND})
    )
    jobs.collect(settings)
    return job_id


def _told_items(voice: agent.VoiceAgent) -> list[str]:
    return [
        part["text"]
        for item in voice._conn.items
        if item.get("role") == "user"
        for part in item["content"]
        if "background job" in part["text"]
    ]


async def _reply_ends(voice: agent.VoiceAgent, status: str = "completed") -> None:
    """The server says the response is over, through the path the real event takes."""
    voice._want_response = False  # response.created clears this
    await voice._on_response_done(SimpleNamespace(status=status, output=[]))


def _output(voice: agent.VoiceAgent) -> dict:
    (item,) = [i for i in voice._conn.items if i.get("type") == "function_call_output"]
    return json.loads(item["output"])


def test_the_tool_is_only_offered_when_the_worker_is_on(settings: Settings) -> None:
    def names(on: bool) -> set[str]:
        chosen = Settings(api_key="", worker=on)
        return {tool["name"] for tool in agent._offered_tools(chosen)}

    assert "background_job" in names(True)
    assert "background_job" not in names(False)
    assert not Settings(api_key="").worker, "off unless asked for"


def test_handing_off_answers_at_once_and_asks_for_a_reply(voice, settings) -> None:
    asyncio.run(voice._run_background_job(_call()))

    assert _output(voice)["started"] is True
    assert len(voice._conn.responses) == 1
    assert list((settings.worker_dir / worker.INBOX).iterdir())


def test_a_second_ask_is_told_its_still_on_the_other_one(voice) -> None:
    async def go() -> None:
        await voice._run_background_job(_call())
        voice._conn.items.clear()
        await voice._run_background_job(_call("another"))

    asyncio.run(go())

    assert _output(voice)["ok"] is False


def test_a_failing_ledger_still_answers_the_model(voice, monkeypatch) -> None:
    def broken(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(jobs, "submit", broken)

    asyncio.run(voice._run_background_job(_call()))

    assert _output(voice)["ok"] is False
    assert voice._conn.responses, "and it still gets to say so"


def test_a_running_job_doesnt_hold_the_session_open(voice) -> None:
    """Two quiet minutes still end it: the job is built to outlive the session."""
    async def go() -> None:
        await voice._run_background_job(_call())
        await _reply_ends(voice)

    asyncio.run(go())

    assert not voice._busy()


def test_nothing_is_said_in_a_new_session_until_he_has_spoken(voice, settings) -> None:
    _failed_job(settings)

    asyncio.run(voice._tell_if_due())

    assert _told_items(voice) == []
    assert not voice._conn.responses


def test_after_he_speaks_and_a_pause_it_is_told(voice, settings) -> None:
    _failed_job(settings)
    voice._he_spoke = True

    asyncio.run(voice._tell_if_due())

    assert len(_told_items(voice)) == 1
    assert len(voice._conn.responses) == 1


@pytest.mark.parametrize(
    "state",
    [
        {"_user_speaking": True},
        {"_response_active": True},
        {"_tools_running": 1},
        {"_want_response": True},
        {"_heard_at": None},  # he finished speaking just now
    ],
)
def test_it_never_talks_over_anything(voice, settings, state) -> None:
    _failed_job(settings)
    voice._he_spoke = True
    for name, value in state.items():
        setattr(voice, name, time.monotonic() if value is None else value)

    asyncio.run(voice._tell_if_due())

    assert _told_items(voice) == []


def test_told_once_the_reply_finishes_and_never_again(voice, settings) -> None:
    _failed_job(settings)
    voice._he_spoke = True

    async def go() -> None:
        await voice._tell_if_due()
        await _reply_ends(voice)
        voice._tell_not_before = 0.0
        assert not voice._busy(), "so only having been told can stop a second telling"
        await voice._tell_if_due()

    asyncio.run(go())

    assert len(_told_items(voice)) == 1
    assert jobs.to_tell() is None


def test_one_he_talked_over_is_told_again_later(voice, settings) -> None:
    _failed_job(settings)
    voice._he_spoke = True

    async def go() -> None:
        await voice._tell_if_due()
        await _reply_ends(voice, "cancelled")  # he spoke over it
        await voice._tell_if_due()  # too soon: a failing call must not repeat at once
        assert len(_told_items(voice)) == 1
        voice._tell_not_before = 0.0
        await voice._tell_if_due()

    asyncio.run(go())

    assert len(_told_items(voice)) == 2
    assert jobs.to_tell() is not None, "still not marked"


def test_a_session_ended_mid_telling_leaves_it_for_the_next(voice, settings) -> None:
    job_id = _failed_job(settings)
    voice._he_spoke = True

    asyncio.run(voice._tell_if_due())  # the session dies before the reply is done

    assert jobs.to_tell()["id"] == job_id
    fresh = agent.VoiceAgent(settings)
    assert fresh._telling is None and not fresh._he_spoke


def test_an_ordinary_reply_doesnt_mark_anything_told(voice, settings) -> None:
    _failed_job(settings)

    asyncio.run(_reply_ends(voice))

    assert jobs.to_tell() is not None


def test_the_telling_loop_survives_a_bad_read(voice, monkeypatch) -> None:
    calls = 0

    async def broken() -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            voice.ended_by = "stopped"
        raise ValueError("bad ledger")

    monkeypatch.setattr(voice, "_tell_if_due", broken)
    monkeypatch.setattr(agent, "TELL_CHECK_S", 0)

    asyncio.run(voice._tell_when_quiet())

    assert calls == 2


def test_a_failing_collect_doesnt_cost_the_sweep(settings, monkeypatch) -> None:
    """A new step in a shared loop: it gets its own catch, and the rest still runs."""
    swept: list[str] = []

    def broken(_settings) -> list[str]:
        raise ValueError("not what it expected")

    async def reconcile(*_a, **_k) -> bool:
        swept.append("reconcile")
        return True

    monkeypatch.setattr(jobs, "collect", broken)
    monkeypatch.setattr(indexer, "reconcile", reconcile)
    monkeypatch.setattr(indexer.cut, "one", lambda _s: swept.append("cut") or "")

    asyncio.run(indexer._guarded(settings))

    assert swept == ["reconcile", "cut"]


def test_with_the_worker_off_the_sweep_doesnt_touch_it(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(jobs, "collect", lambda _s: pytest.fail("worker is off"))

    async def reconcile(*_a, **_k) -> bool:
        return True

    monkeypatch.setattr(indexer, "reconcile", reconcile)
    monkeypatch.setattr(indexer.cut, "one", lambda _s: "")

    asyncio.run(indexer._guarded(Settings(api_key="")))


def test_the_sweep_watches_the_workers_answers(settings) -> None:
    assert settings.worker_dir / worker.DONE in indexer._folders_to_watch(settings)


def _reported_job(settings: Settings, report: str = "Read out their notes.") -> str:
    assert jobs.submit(settings, "compare these three") == jobs.STARTED
    job_id = jobs._read()[-1]["id"]
    (settings.worker_dir / worker.DONE / f"{job_id}.json").write_text(
        json.dumps({"state": "ok", "report": report})
    )
    jobs.collect(settings)
    return job_id


def test_the_reply_that_tells_a_report_can_only_be_spoken(voice, settings) -> None:
    _reported_job(settings)
    voice._he_spoke = True

    asyncio.run(voice._tell_if_due())

    assert voice._conn.responses[-1]["response"] == {"tool_choice": "none"}


def test_tools_come_back_for_the_reply_after_the_telling(voice, settings) -> None:
    _reported_job(settings)
    voice._he_spoke = True

    async def go() -> None:
        await voice._tell_if_due()
        await _reply_ends(voice)
        await voice._request_response()

    asyncio.run(go())

    assert "response" not in voice._conn.responses[-1]


@pytest.mark.parametrize("status", ["completed", "cancelled"])
def test_the_report_is_gone_from_the_conversation_however_the_reply_ended(
    voice, settings, status
) -> None:
    _reported_job(settings)
    voice._he_spoke = True

    async def go() -> None:
        await voice._tell_if_due()
        await _reply_ends(voice, status)

    asyncio.run(go())

    assert voice._conn.deleted == [voice._conn.items[-1]["id"]]


def test_a_failing_delete_still_marks_it_told(voice, settings, monkeypatch) -> None:
    _reported_job(settings)
    voice._he_spoke = True

    async def broken(item_id: str) -> None:
        raise OSError("socket gone")

    voice._conn.conversation.item.delete = broken

    async def go() -> None:
        await voice._tell_if_due()
        await _reply_ends(voice)

    asyncio.run(go())

    assert jobs.to_tell() is None


def _check_call() -> SimpleNamespace:
    return SimpleNamespace(name="check_background_job", call_id="c2", arguments="{}")


def test_both_tools_are_offered_together_or_not_at_all() -> None:
    def names(on: bool) -> set[str]:
        return {tool["name"] for tool in agent._offered_tools(Settings(api_key="", worker=on))}

    assert {"background_job", "check_background_job"} <= names(True)
    assert not {"background_job", "check_background_job"} & names(False)


def test_checking_with_nothing_running_answers_and_asks_for_a_reply(voice) -> None:
    asyncio.run(voice._run_check_background_job(_check_call()))

    assert _output(voice)["running"] is False
    assert len(voice._conn.responses) == 1


def test_checking_a_running_job_answers_from_its_state(voice, settings) -> None:
    asyncio.run(voice._run_background_job(_call()))
    voice._conn.items.clear()

    asyncio.run(voice._run_check_background_job(_check_call()))

    assert _output(voice)["running"] is True


def test_checking_never_carries_the_report(voice, settings) -> None:
    _reported_job(settings, "the planted words")

    asyncio.run(voice._run_check_background_job(_check_call()))

    assert "planted" not in json.dumps(_output(voice))
    assert _output(voice)["finished_not_told_yet"] is True


def test_a_failing_read_still_answers_the_model_when_checking(voice, monkeypatch) -> None:
    def broken(*a, **k):
        raise OSError("disk")

    monkeypatch.setattr(jobs, "status", broken)

    asyncio.run(voice._run_check_background_job(_check_call()))

    assert _output(voice)["ok"] is False


def test_the_instructions_only_name_the_tools_when_theyre_offered(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(agent.session, "recent_context", lambda s: agent.session.Recap("", ""))
    on = agent._standing(Settings(api_key="", worker=True), ())[0]
    off = agent._standing(Settings(api_key="", worker=False), ())[0]

    assert "check_background_job" in on
    assert "check_background_job" not in off
