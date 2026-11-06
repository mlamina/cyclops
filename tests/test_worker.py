"""The background worker's own side, run against a fake ``claude`` in a scratch folder.

No key, no network, no account of its own. What is proved on the device instead is the boundary:
that lives in the service unit and the firewall, which a test here can't reach.
"""

from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path

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
def root(tmp_path: Path) -> Path:
    folder = tmp_path / "worker"
    worker.prepare(folder)
    return folder


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
    out = '{"subtype": "error_max_budget_usd", "is_error": true, "total_cost_usd": 1.01}'

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
