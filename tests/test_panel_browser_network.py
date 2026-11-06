"""The panel's browser reaches this box and YouTube's video host, and nothing else (arc-eng/hq#619)."""

from __future__ import annotations

import subprocess

from cyclops import kiosk


def test_the_panel_s_browser_starts_behind_the_wall_with_videos_let_through(tmp_path, monkeypatch) -> None:
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
    # Exactly this list. Anything wider lets the internet back round the wall, without it a video
    # asked for out loud goes black, and <-loopback> would push the panel's own pages into the wall.
    assert [a for a in argv if a.startswith("--proxy-bypass-list")] == [
        "--proxy-bypass-list=*.googlevideo.com"
    ]
    # One argument, spaces and all, and it has to let the same video host through as the bypass list.
    assert "--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE *.googlevideo.com, EXCLUDE localhost, EXCLUDE 127.0.0.1" in argv
    assert argv[-1] == "http://127.0.0.1:80/"
