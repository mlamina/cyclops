"""What every test in here needs to be true before it runs.

Two safety rules rather than conveniences. The first: no test may write to the
real background-task ledger. :mod:`cyclops.tasks` keeps it at a fixed path under ``~/.cache``,
because every process on the box has to find the same file without being told where it is - which
also means that a test driving a tool handler writes into the ledger of whoever is running the
suite. It did, before this existed: two rows saying "editing the picture to paint the doors matt
black…" turned up on a laptop that had never had a session open, and the panel would have shown
them for fifteen minutes.

Autouse and unconditional. A test that wants to read the ledger back points at these same paths;
a test that only happens to touch it gets a tmpdir it never has to know about.

The second: no test makes a sound. The suite runs on a laptop with someone at it, and a cue is
the laptop's speakers going off. Cut at sounddevice rather than at :mod:`cyclops.sfx`, so the
cue logic above it - lengths, ranks, who may cut whom - still runs for real.
"""

from __future__ import annotations

import pytest

from cyclops import sfx, tasks


@pytest.fixture(autouse=True)
def ledger(tmp_path, monkeypatch):
    """Point cyclops.tasks at a throwaway file for the duration of one test."""
    monkeypatch.setattr(tasks, "TASKS_FILE", tmp_path / "tasks.yaml")
    monkeypatch.setattr(tasks, "TASKS_LOCK", tmp_path / "tasks.lock")
    monkeypatch.setattr(tasks, "_memo", None)  # the stat memo in line(), from another test's file
    return tmp_path / "tasks.yaml"


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    """Every cue a test sounds goes nowhere."""
    monkeypatch.setattr(sfx.sd, "play", lambda *args, **kwargs: None)
