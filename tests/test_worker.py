"""The background worker's own side, run against a fake ``claude`` in a scratch folder.

No key, no network, no account of its own. What is proved on the device instead is the boundary:
that lives in the service unit and the firewall, which a test here can't reach.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import socket
import stat
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from cyclops import worker

GOOD = json.dumps({
    "result": json.dumps({"name": "Relay manual", "description": "A manual."}),
    "total_cost_usd": 0.12,
})


def _claude(tmp_path: Path, body: str) -> str:
    """A stand-in for ``claude``: a shell script that does what the test needs."""
    script = tmp_path / "fake-claude"
    script.write_text(f"#!/bin/sh\n{body}\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The worker's folder, with this process posing as another account: what the test writes is
    what Cyclops writes. A uid with no account also keeps _clear_account well away from kill(-1)."""
    monkeypatch.setattr(os, "geteuid", lambda: os.getuid() + 100000)
    folder = tmp_path / "worker"
    worker.prepare(folder)
    return folder


@pytest.fixture
def own_account(monkeypatch: pytest.MonkeyPatch) -> None:
    """What the test writes from here on is what a job writes: the worker's own account."""
    monkeypatch.setattr(os, "geteuid", os.getuid)


@pytest.fixture
def key(tmp_path: Path) -> Path:
    path = tmp_path / "key"
    path.write_text("test-key\n")
    return path


def _ask(root: Path, job_id: str = "abcdef01", request: str = "the relay manual") -> None:
    (root / worker.INBOX / f"{job_id}.json").write_text(json.dumps({"request": request}))


def _run(root: Path, claude: str, key: Path) -> dict:
    _ask(root)
    taken = worker.take_next(root)
    assert taken is not None
    worker.run_job(root, *taken, claude=claude, token_file=key)
    return json.loads((root / worker.DONE / f"{taken[0]}.json").read_text())


def test_a_finished_job_reports_its_file_name_and_cost(root, key, tmp_path) -> None:
    claude = _claude(tmp_path, f"printf '%%PDF-' > {worker.RESULT}\ncat <<'EOF'\n{GOOD}\nEOF")

    done = _run(root, claude, key)

    assert done["state"] == "ok"
    assert done["name"] == "Relay manual"
    assert done["cost_usd"] == 0.12
    assert (root / worker.JOBS / "abcdef01" / worker.WORK / worker.RESULT).exists()
    assert not (root / worker.JOBS / "abcdef01" / worker.RUNNING).exists()


def test_no_file_is_nothing_found_whatever_the_model_claims(root, key, tmp_path) -> None:
    done = _run(root, _claude(tmp_path, f"cat <<'EOF'\n{GOOD}\nEOF"), key)

    assert (done["state"], done["reason"]) == ("failed", worker.NOTHING_FOUND)


def test_a_link_is_left_for_cyclops_to_refuse_not_followed_here(root, key, tmp_path) -> None:
    """On the device a link into /home raises a permission error when followed (arc-eng/hq#599)."""
    blocked = tmp_path / "blocked"
    blocked.mkdir(mode=0o000)
    claude = _claude(tmp_path, f"ln -s {blocked}/.env {worker.RESULT}\ncat <<'EOF'\n{GOOD}\nEOF")
    try:
        done = _run(root, claude, key)
    finally:
        blocked.chmod(0o700)

    assert done["state"] == "ok", "what it is gets decided by the hand-in, which refuses a link"


def test_over_budget_is_said_as_that(root, key, tmp_path) -> None:
    # is_error is false on the real thing: seen on the device, Claude Code 2.1.61, on a subscription.
    out = '{"subtype": "error_max_budget_usd", "is_error": false, "total_cost_usd": 1.01}'

    done = _run(root, _claude(tmp_path, f"echo '{out}'"), key)

    assert (done["state"], done["reason"]) == ("failed", worker.OVER_BUDGET)
    assert done["cost_usd"] == 1.01


@pytest.mark.parametrize(
    "body",
    ["echo 'this is not json'", "exit 3", "echo '[1, 2]'", "echo '{\"is_error\": true}'"],
)
def test_a_bad_read_of_its_output_is_a_failure_not_a_crash(root, key, tmp_path, body) -> None:
    done = _run(root, _claude(tmp_path, body), key)

    assert (done["state"], done["reason"]) == ("failed", worker.BROKEN)


def test_a_mangled_last_message_still_counts_if_the_file_is_there(root, key, tmp_path) -> None:
    out = '{"result": "Done! I found it."}'
    claude = _claude(tmp_path, f"touch {worker.RESULT}\necho '{out}'")

    done = _run(root, claude, key)

    assert done["state"] == "ok"
    assert "name" not in done


def test_the_time_limit_kills_what_it_started_too(root, key, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(worker, "JOB_LIMIT_S", 0.3)
    pid_file = tmp_path / "child.pid"
    claude = _claude(tmp_path, f"sleep 30 &\necho $! > {pid_file}\nwait")

    done = _run(root, claude, key)

    assert (done["state"], done["reason"]) == ("failed", worker.TIMED_OUT)
    pid = int(pid_file.read_text())
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.02)
    pytest.fail("the process it started outlived the limit")


def test_a_missing_token_answers_rather_than_dying(root, tmp_path) -> None:
    _ask(root)
    taken = worker.take_next(root)
    worker.run_job(root, *taken, claude="/nonexistent", token_file=tmp_path / "absent")

    done = json.loads((root / worker.DONE / "abcdef01.json").read_text())
    assert (done["state"], done["reason"]) == ("failed", worker.BROKEN)


def test_its_environment_is_only_what_it_was_given(root, key, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-it")
    seen = tmp_path / "env"
    _run(root, _claude(tmp_path, f"env > {seen}\necho '{{}}'"), key)

    text = seen.read_text()
    assert "OPENAI_API_KEY" not in text
    assert "CLAUDE_CODE_OAUTH_TOKEN=test-key" in text
    assert "ANTHROPIC_API_KEY" not in text, "it would outrank the token"


def test_the_command_never_stops_at_a_prompt_and_has_a_budget() -> None:
    argv = worker.command("claude")

    assert argv[argv.index("--permission-mode") + 1] == "bypassPermissions"
    assert float(argv[argv.index("--max-budget-usd") + 1]) == worker.BUDGET_USD
    assert "-p" in argv


def test_a_job_cut_off_is_answered_at_the_next_start_and_not_run_again(root) -> None:
    _ask(root)
    worker.take_next(root)  # started; then the power goes

    assert worker.recover(root) == ["abcdef01"]

    done = json.loads((root / worker.DONE / "abcdef01.json").read_text())
    assert done["reason"] == worker.INTERRUPTED
    assert worker.recover(root) == [], "answered once"
    assert worker.take_next(root) is None


def test_a_request_from_before_the_stop_is_not_run_after_it(root) -> None:
    _ask(root)
    request = root / worker.INBOX / "abcdef01.json"
    old = time.time() - worker.STALE_REQUEST_S - 1
    os.utime(request, (old, old))

    assert worker.recover(root) == ["abcdef01"]
    assert worker.take_next(root) is None


def test_a_good_request_under_a_bad_name_is_never_run(root) -> None:
    (root / worker.INBOX / "not-an-id.json").write_text(json.dumps({"request": "x"}))

    assert worker.take_next(root) is None
    assert not list((root / worker.JOBS).iterdir())


def test_a_request_made_while_the_box_is_booting_is_kept(root) -> None:
    _ask(root)

    assert worker.recover(root) == []
    assert worker.take_next(root) is not None


@pytest.mark.parametrize("name", ["abcdef01.json", "not-an-id.json"])
def test_a_broken_request_cant_jam_the_queue(root, name) -> None:
    (root / worker.INBOX / name).write_text("{ not json")
    _ask(root, "abcdef02")

    taken = worker.take_next(root)

    assert taken is not None and taken[0] == "abcdef02"


def test_tidy_keeps_a_running_job_however_old(root) -> None:
    _ask(root)
    worker.take_next(root)
    old = root / worker.JOBS / "abcdef01"

    worker.tidy(root, now=time.time() + worker.KEEP_S * 2)

    assert old.exists()
    worker.finish(root, "abcdef01", "failed", reason=worker.BROKEN)
    worker.tidy(root, now=time.time() + worker.KEEP_S * 2)
    assert not old.exists()


def test_it_imports_nothing_from_cyclops_or_outside_the_standard_library() -> None:
    """It runs where the project's virtualenv can't be seen; one import would stop it starting."""
    import ast
    import sys

    tree = ast.parse(Path(worker.__file__).read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.add((node.module or "relative").split(".")[0])
        elif isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
    assert names <= set(sys.stdlib_module_names) | {"__future__"}


def test_a_request_written_from_its_own_account_never_runs(root, own_account) -> None:
    """A hijacked job could otherwise queue itself a fresh budget, forever and with nobody told."""
    _ask(root)

    assert worker.take_next(root) is None
    assert not list((root / worker.INBOX).iterdir())
    assert not (root / worker.JOBS / "abcdef01").exists()
    assert not (root / worker.DONE / "abcdef01.json").exists()


def test_a_request_from_its_own_account_is_not_kept_for_after_a_restart(root, own_account) -> None:
    _ask(root)

    assert worker.recover(root, now=time.time() + 10 * worker.STALE_REQUEST_S) == []
    assert not list((root / worker.INBOX).iterdir())


@pytest.mark.parametrize("plant", ["fifo", "link", "dangling", "folder"])
def test_something_planted_in_the_inbox_cant_hang_or_stop_it(root, tmp_path, plant) -> None:
    path = root / worker.INBOX / "abcdef09.json"
    if plant == "fifo":
        os.mkfifo(path)
    elif plant == "link":
        secret = tmp_path / "his.json"
        secret.write_text(json.dumps({"request": "planted through a link"}))
        path.symlink_to(secret)
    elif plant == "dangling":
        path.symlink_to(tmp_path / "nothing-here")
    else:
        path.mkdir()
    _ask(root, "abcdef02")

    taken = worker.take_next(root)

    assert taken == ("abcdef02", "the relay manual")


def test_a_job_that_exits_leaves_nothing_of_its_group_running(root, key, tmp_path) -> None:
    pid_file = tmp_path / "child.pid"
    claude = _claude(tmp_path, f"sleep 30 &\necho $! > {pid_file}\ncat <<'EOF'\n{GOOD}\nEOF")

    _run(root, claude, key)

    pid = int(pid_file.read_text())
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.02)
    pytest.fail("what the job started outlived its answer")


def _account(monkeypatch: pytest.MonkeyPatch, name: str) -> list[tuple[int, int]]:
    """Pose as an account and catch kill. The real kill(-1) here would end this whole login."""
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(worker.pwd, "getpwuid", lambda _uid: SimpleNamespace(pw_name=name))
    monkeypatch.setattr(worker.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    return sent


def test_when_a_job_ends_everything_on_the_workers_account_is_killed(
    root, key, tmp_path, monkeypatch
) -> None:
    sent = _account(monkeypatch, worker.ACCOUNT)
    claude = _claude(tmp_path, f"cat <<'EOF'\n{GOOD}\nEOF")

    _run(root, claude, key)

    assert sent == [(-1, signal.SIGKILL)]


def test_on_any_other_account_nothing_is_ever_killed_that_way(monkeypatch) -> None:
    sent = _account(monkeypatch, "mlamina")
    worker._clear_account()
    assert sent == []


def test_an_account_with_no_name_kills_nothing(monkeypatch) -> None:
    sent = _account(monkeypatch, "unused")

    def no_such_account(_uid: int) -> None:
        raise KeyError(_uid)

    monkeypatch.setattr(worker.pwd, "getpwuid", no_such_account)
    worker._clear_account()
    assert sent == []


def _within(seconds: float, call) -> None:
    """Call it on a thread, so a regression that hangs fails here rather than hanging the suite."""
    done = threading.Event()

    def run() -> None:
        call()
        done.set()

    threading.Thread(target=run, daemon=True).start()
    assert done.wait(seconds), "hung"


def test_a_pipe_in_place_of_its_answer_settles_as_broken(root, key, tmp_path) -> None:
    claude = _claude(tmp_path, "rm ../output.json\nmkfifo ../output.json")
    seen: list[dict] = []

    _within(5, lambda: seen.append(_run(root, claude, key)))

    assert seen[0]["reason"] == worker.BROKEN


@pytest.mark.parametrize("plant", ["pipe at the scratch name", "folder at the final name"])
def test_what_a_job_plants_in_done_cant_stop_it_being_answered(root, key, tmp_path, plant) -> None:
    done = "../../../done"
    make = (
        f"mkfifo {done}/.abcdef01.json.tmp"
        if plant.startswith("pipe")
        else f"mkdir -p {done}/abcdef01.json/inside"
    )
    seen: list[dict] = []

    _within(5, lambda: seen.append(_run(root, _claude(tmp_path, make), key)))

    assert seen[0]["state"] == "failed"


def test_a_folder_for_the_running_marker_doesnt_answer_a_job_twice(root, key, tmp_path) -> None:
    claude = _claude(tmp_path, f"rm ../running\nmkdir ../running\ncat <<'EOF'\n{GOOD}\nEOF")
    _run(root, claude, key)

    assert worker.recover(root) == []


def test_a_job_folder_with_a_name_too_long_to_answer_doesnt_stop_a_start(root) -> None:
    junk = root / worker.JOBS / ("a" * 250)
    junk.mkdir()
    (junk / worker.RUNNING).touch()
    real = root / worker.JOBS / "abcdef03"
    real.mkdir()
    (real / worker.RUNNING).touch()

    assert worker.recover(root) == ["abcdef03"]
    assert not junk.exists()


def test_a_cut_off_job_it_cant_answer_doesnt_stop_the_others(root) -> None:
    for job_id in ("abcdef03", "abcdef04"):
        (root / worker.JOBS / job_id).mkdir()
        (root / worker.JOBS / job_id / worker.RUNNING).touch()
    # A folder that can't be replaced: only a path this process can't clear will do.
    stuck = root / worker.DONE / "abcdef03.json"
    stuck.mkdir()
    (stuck / "inside").touch()
    stuck.chmod(0o500)
    try:
        assert "abcdef04" in worker.recover(root)
    finally:
        stuck.chmod(0o700)


def test_each_job_gets_a_new_home_so_nothing_planted_reaches_the_next(root, key, tmp_path) -> None:
    body = f"echo \"$HOME\" >> {tmp_path}/homes\ntouch \"$HOME/planted\"\necho '{{}}'"
    claude = _claude(tmp_path, body)
    _run(root, claude, key)
    _ask(root, "abcdef02")
    taken = worker.take_next(root)
    worker.run_job(root, *taken, claude=claude, token_file=key)

    first, second = (tmp_path / "homes").read_text().split()
    assert first != second
    assert Path(first).parent.name == "abcdef01" and Path(second).parent.name == "abcdef02"


def test_it_reports_in_while_a_job_runs_and_the_job_cant_see_where(
    root, key, tmp_path, monkeypatch
) -> None:
    # A short path of its own: a socket's path is limited, and tmp_path can be longer than that.
    with tempfile.TemporaryDirectory(dir="/tmp") as short, socket.socket(
        socket.AF_UNIX, socket.SOCK_DGRAM
    ) as listener:
        where = f"{short}/n"
        listener.bind(where)
        listener.settimeout(2)
        monkeypatch.setenv("NOTIFY_SOCKET", where)
        monkeypatch.setattr(worker, "BEAT_S", 0.05)
        claude = _claude(tmp_path, f"sleep 0.3\nenv > {tmp_path}/env\necho '{{}}'")

        _run(root, claude, key)

        beats = [listener.recv(64)]
        listener.settimeout(0)
        with contextlib.suppress(BlockingIOError):
            while True:
                beats.append(listener.recv(64))
    assert set(beats) == {b"WATCHDOG=1"} and len(beats) > 2
    assert "NOTIFY_SOCKET" not in (tmp_path / "env").read_text()


class _Libc:
    def __init__(self, answer: int) -> None:
        self.answer = answer
        self.calls: list[tuple] = []

    def prctl(self, *args) -> int:
        self.calls.append(args)
        return self.answer


def test_it_turns_off_attaching_to_itself(monkeypatch) -> None:
    libc = _Libc(0)
    monkeypatch.setattr(worker.ctypes, "CDLL", lambda *a, **k: libc)
    worker._no_attaching()
    assert libc.calls == [(worker.PR_SET_DUMPABLE, 0, 0, 0, 0)]


def test_it_wont_run_if_a_job_could_attach_to_it(monkeypatch) -> None:
    monkeypatch.setattr(worker.ctypes, "CDLL", lambda *a, **k: _Libc(-1))
    with pytest.raises(OSError):
        worker._no_attaching()
