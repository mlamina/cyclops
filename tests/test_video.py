"""A video on the panel: the offer's shape, the microphone gate, and what is kept afterwards."""

from __future__ import annotations

import json
from pathlib import Path

from cyclops import agent, card, panel, session
from cyclops.config import Settings

STATIC = Path(agent.__file__).parent / "admin" / "static"


def _offer(tmp_path, monkeypatch, **over):
    monkeypatch.setattr(panel, "PANEL_FILE", tmp_path / "panel.json")
    kw = dict(
        url="https://x.invalid/v.mp4",
        title="A title",
        start_s=312,
        thumb=b"\xff\xd8jpg",
        hold=99.0,
    )
    kw.update(over)
    assert panel.offer_video(**kw)
    return json.loads((tmp_path / "panel.json").read_text())


def test_a_video_offer_carries_a_url_and_a_picture_of_it(tmp_path, monkeypatch) -> None:
    """The page branches on which key is there, so the keys are the contract."""
    found = _offer(tmp_path, monkeypatch)
    assert found["video"] == "https://x.invalid/v.mp4"
    assert found["start"] == 312
    assert found["image"].startswith(panel.JPEG_URL)
    assert "scratchpad" not in found and "sketch" not in found


def test_the_panel_is_held_open_only_for_a_video(tmp_path, monkeypatch) -> None:
    """A photograph after a video must not inherit its clock."""
    _offer(tmp_path, monkeypatch, hold=1800.0)
    assert panel.hold_s() == 1800.0
    panel.offer_image(b"\xff\xd8jpg", "a photo")
    assert panel.hold_s() == 0.0


def test_the_page_looks_for_a_video_before_it_looks_for_a_picture() -> None:
    """A lint that earns its place: reorder the branches and the video silently never plays.

    A video offer carries a thumbnail under `image` too (see ``panel.offer_video``), so testing
    the picture first would paint a title card and start nothing. There is no way to catch
    that without Chromium except by reading the source, so this reads the source.
    """
    js = (STATIC / "panel.js").read_text()
    assert js.index("found.video") < js.index("found.image")


def test_the_mic_is_shut_while_a_video_is_up_and_opens_when_it_comes_down() -> None:
    one = agent.VoiceAgent(Settings(api_key=""))
    assert not one.mic_shut
    one.mic_shut = True
    one.panel_closed(1.0)
    assert not one.mic_shut


def test_a_video_glanced_at_and_dismissed_is_not_kept(monkeypatch) -> None:
    """The watch is the filter: what nobody watched is not a reference, it is clutter."""
    kept = []
    monkeypatch.setattr(session, "keep_video", lambda **kw: kept.append(kw))
    one = agent.VoiceAgent(Settings(api_key=""))
    one._video = agent.Watch("v1", "T", "C", 60, "s", "th")
    one.panel_closed(agent.WATCHED_S - 1)
    assert kept == []


def test_a_video_that_was_watched_is_kept(monkeypatch) -> None:
    kept = []
    monkeypatch.setattr(session, "keep_video", lambda **kw: kept.append(kw))
    one = agent.VoiceAgent(Settings(api_key=""))
    one._video = agent.Watch("v1", "Title", "Channel", 60, "s", "th")
    one.panel_closed(agent.WATCHED_S + 5)
    assert len(kept) == 1
    assert kept[0]["video_id"] == "v1" and kept[0]["start"] == 60


def test_a_kept_video_lands_beside_its_sidecar(tmp_path, monkeypatch) -> None:
    """A reference is a real file on the card, which is what lets recall and store treat it
    as a thing rather than as a special case."""

    class Live:
        videos_dir = tmp_path / "videos"
        events: list = []

        def event(self, kind, **fields):
            self.events.append((kind, fields))

    live = Live()
    monkeypatch.setattr(session, "current", lambda: live)
    name = session.keep_video(
        video_id="abc", title="Sharpening a Chisel", channel="Paul Sellers",
        start=312, seconds=95, thumb=b"\xff\xd8jpg",
    )
    assert name.endswith("_sharpening-a-chisel.jpg")
    assert (live.videos_dir / name).read_bytes() == b"\xff\xd8jpg"
    kept = json.loads((live.videos_dir / card.VIDEOS_NAME).read_text())
    assert kept[name]["id"] == "abc" and kept[name]["start"] == 312
    assert live.events[0][0] == "watched"
    assert live.events[0][1]["file"] == f"videos/{name}"


def test_the_tool_is_left_out_rather_than_refused_when_it_is_off() -> None:
    assert agent._video_tools(Settings(api_key="", video=False)) == []
    assert agent._video_tools(Settings(api_key="", video=True))[0]["name"] == "watch_video"


def test_the_panel_line_for_a_video_still_ends_in_an_ellipsis() -> None:
    """overlay.BUSY_MARK reads the ellipsis to tell work in flight from a state."""

    class Call:
        name = "watch_video"
        arguments = json.dumps({"request": "bleeding shimano brakes"})

    line = agent._activity_line(Call())
    assert line.endswith("…") and "bleeding shimano brakes" in line


def test_the_kiosk_can_reach_a_running_session_to_say_the_panel_is_free() -> None:
    """The hop the microphone depends on: kiosk thread -> controller -> agent, or nothing.

    Without a live loop it must answer False rather than raise - the kiosk calls this from a
    `finally` on every picture, and an exception there would strand the panel.
    """
    from cyclops.ui import SessionController

    controller = SessionController(Settings(api_key=""), frames=None, entrypoint="test")
    assert controller.panel_closed(5.0) is False


def test_no_method_on_the_agent_is_defined_twice() -> None:
    """A class body's second `def foo` silently wins, and nothing warns you.

    Adding `_play` for a video quietly replaced the `_play` that feeds an audio delta to the
    speaker: the model went on answering, the transcript went on filling up, and the box made
    no sound at all. Nothing caught it - not ruff, not the suite, not a smoke run - because a
    shadowed method is legal Python. This is the cheapest thing that would have.
    """
    import ast
    import collections

    tree = ast.parse(Path(agent.__file__).read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        names = [
            child.name
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        twice = [n for n, count in collections.Counter(names).items() if count > 1]
        assert not twice, f"{node.name} defines {twice} more than once; the second one wins"


def _hit(tmp_path, **about):
    """One video reference on the card, as recall would hand it back."""
    from cyclops import card, recall

    folder = tmp_path / "videos"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "a.jpg").write_bytes(b"\xff\xd8thumb")
    kept = {"a.jpg": {"id": "abc", "title": "Honing", "channel": "PS", "start": 312, **about}}
    (folder / card.VIDEOS_NAME).write_text(json.dumps(kept))
    item = recall.Item(
        kind="video",
        path=str(folder / "a.jpg"),
        scope="session:2026-09-11_x",
        title="Honing",
        text="Honing. PS",
        mtime_ns=1,
        size=6,
    )
    return recall.Hit(item=item, score=0.9)


def test_asking_for_a_video_again_plays_it_rather_than_showing_a_picture(tmp_path, monkeypatch):
    """A reference stores the id and the second, never the URL - a signed one dies in six hours.

    So recalling it costs a fresh resolve, and what lands on the panel is the video playing
    from where it was found, not a still of it.
    """
    import asyncio

    played = {}

    async def fake_restream(video_id):
        played["asked"] = video_id
        return "https://x.invalid/fresh.mp4"

    async def fake_play(self, found, thumb):
        played["found"] = found
        played["thumb"] = thumb
        return {"ok": True, "title": found.title}

    monkeypatch.setattr(agent, "restream", fake_restream)
    monkeypatch.setattr(agent.VoiceAgent, "_play_video", fake_play)
    monkeypatch.setattr(session, "note", lambda *a, **k: None)
    one = agent.VoiceAgent(Settings(api_key=""))
    out = asyncio.run(one._replay("that honing video", _hit(tmp_path), []))
    assert out["ok"] and played["asked"] == "abc"
    assert played["found"].start == 312 and played["found"].stream.endswith("fresh.mp4")
    assert played["thumb"] == b"\xff\xd8thumb"  # the card's own copy, not fetched again


def test_a_video_that_will_not_resolve_still_puts_its_picture_up(tmp_path, monkeypatch):
    """Pulled, region-locked, or the box is offline. A dead end here would be the worst answer:
    the thumbnail is a real file on the card, so there is always something to show."""
    import asyncio

    shown = {}

    async def no_stream(video_id):
        return ""

    monkeypatch.setattr(agent, "restream", no_stream)
    monkeypatch.setattr(session, "note", lambda *a, **k: None)
    monkeypatch.setattr(
        agent.VoiceAgent, "_show_found", lambda self, path: (shown.setdefault("p", path), b"")
    )
    one = agent.VoiceAgent(Settings(api_key=""))
    out = asyncio.run(one._replay("that honing video", _hit(tmp_path), []))
    assert out["ok"] and out["kind"] == "video"
    assert "will not play" in out["note"]
    assert shown["p"].name == "a.jpg"
