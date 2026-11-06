"""The panel's browser talks to this box and nothing else (arc-eng/hq#619)."""

from __future__ import annotations

import subprocess

from cyclops import kiosk


def test_the_panel_s_browser_starts_behind_the_wall(tmp_path, monkeypatch) -> None:
    """Checked on the command line it really starts with, not only in the tuple."""
    started = []

    def popen(argv, **_kwargs):
        started.append(argv)
        return object()

    monkeypatch.setattr(subprocess, "Popen", popen)
    with (tmp_path / "log").open("wb") as log:
        assert kiosk._spawn_browser("http://127.0.0.1:80/", log) is not None

    argv = started[0]
    assert "--proxy-server=127.0.0.1:9" in argv
    assert "--disable-background-networking" in argv
    # A bypass list could let the internet back round the wall, or push loopback into it.
    assert not any(arg.startswith("--proxy-bypass-list") for arg in argv)
    assert argv[-1] == "http://127.0.0.1:80/"
