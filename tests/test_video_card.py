"""A watched video on the card: filed onto a project, and found again by asking for it."""

from __future__ import annotations

import json
from pathlib import Path

from cyclops import card, recall
from cyclops.projects import store


def _session_with_video(tmp_path: Path, *, name="14-30-00_honing-a-chisel.jpg", **about) -> Path:
    folder = tmp_path / "sessions" / "2026-09-11_14-29-00_chisels"
    videos = folder / card.VIDEOS
    videos.mkdir(parents=True)
    (videos / name).write_bytes(b"\xff\xd8thumb")
    kept = {
        name: {
            "id": "abc123",
            "title": "Honing a Chisel",
            "channel": "Paul Sellers",
            "start": 312,
            "seconds": 95,
            **about,
        }
    }
    (videos / card.VIDEOS_NAME).write_text(json.dumps(kept))
    return folder


def _project(tmp_path: Path) -> store.Project:
    path = tmp_path / "projects" / "Chisels"
    path.mkdir(parents=True)
    return store.Project(path=path, key="chisels", name="Chisels")


# ---------------------------------------------------------------- filing


def test_a_watched_video_is_copied_onto_the_project_with_what_it_points_at(tmp_path) -> None:
    folder = _session_with_video(tmp_path)
    project = _project(tmp_path)
    written = store.copy_videos(project, folder)
    assert written == ["Videos/2026-09-11_14-30-00_honing-a-chisel.jpg"]
    landed = project.videos_dir / "2026-09-11_14-30-00_honing-a-chisel.jpg"
    assert landed.read_bytes() == b"\xff\xd8thumb"
    kept = json.loads((project.videos_dir / card.VIDEOS_NAME).read_text())
    assert kept[landed.name]["id"] == "abc123"
    assert kept[landed.name]["start"] == 312


def test_filing_the_same_session_twice_changes_nothing(tmp_path) -> None:
    """The sweep re-reads a session it crashed part-way through, so this has to be safe."""
    folder = _session_with_video(tmp_path)
    project = _project(tmp_path)
    assert store.copy_videos(project, folder) == store.copy_videos(project, folder)
    assert len(list(project.videos_dir.glob("*.jpg"))) == 1


def test_a_second_session_adds_to_the_sidecar_rather_than_replacing_it(tmp_path) -> None:
    """A project collects these over months; a rewrite would lose every earlier one."""
    project = _project(tmp_path)
    store.copy_videos(project, _session_with_video(tmp_path))
    later = tmp_path / "sessions" / "2026-10-02_09-00-00_more"
    (later / card.VIDEOS).mkdir(parents=True)
    (later / card.VIDEOS / "09-01-00_grinding.jpg").write_bytes(b"\xff\xd8two")
    (later / card.VIDEOS / card.VIDEOS_NAME).write_text(
        json.dumps({"09-01-00_grinding.jpg": {"id": "zzz", "title": "Grinding"}})
    )
    store.copy_videos(project, later)
    kept = json.loads((project.videos_dir / card.VIDEOS_NAME).read_text())
    assert len(kept) == 2


def test_a_session_that_kept_no_video_files_nothing(tmp_path) -> None:
    project = _project(tmp_path)
    folder = tmp_path / "sessions" / "2026-09-11_14-29-00_quiet"
    folder.mkdir(parents=True)
    assert store.copy_videos(project, folder) == []
    assert not project.videos_dir.exists()


# ---------------------------------------------------------------- finding it again


def test_a_reference_is_indexed_under_what_was_said_about_it(tmp_path) -> None:
    folder = _session_with_video(tmp_path)
    items = recall.session_items(folder)
    videos = [i for i in items if i.kind == "video"]
    assert len(videos) == 1
    one = videos[0]
    assert one.title == "Honing a Chisel"
    assert "Honing a Chisel" in one.text and "Paul Sellers" in one.text
    assert one.playable and not one.showable


def test_a_reference_is_indexed_once_and_not_also_as_a_picture(tmp_path) -> None:
    """Videos/ holds jpgs, so the sweep over a project's other folders has to skip it.

    Without that skip every reference is in the index twice, the second time as an ordinary
    picture whose only words are its filename.
    """
    project = _project(tmp_path)
    store.copy_videos(project, _session_with_video(tmp_path))
    (project.path / "README.md").write_text("# Chisels\n")
    kinds = [i.kind for i in recall.project_items(project.path) if i.path.endswith(".jpg")]
    assert kinds == ["video"]


def test_the_folder_name_is_not_mistaken_for_words_about_the_video(tmp_path) -> None:
    """`_where` turns a path into words; "videos" is not one somebody would ever ask by."""
    folder = _session_with_video(tmp_path)
    one = [i for i in recall.session_items(folder) if i.kind == "video"][0]
    assert "videos" not in one.text.lower()


def test_a_sidecar_that_outlived_its_picture_yields_nothing(tmp_path) -> None:
    folder = _session_with_video(tmp_path)
    (folder / card.VIDEOS / "14-30-00_honing-a-chisel.jpg").unlink()
    assert [i for i in recall.session_items(folder) if i.kind == "video"] == []


def test_the_captioner_is_never_pointed_at_a_title_card(tmp_path, monkeypatch) -> None:
    """image_folders is derived from `showable`, and a video already carries its own words.

    Folding "video" into showable would buy a model call per reference to be told what a
    thumbnail looks like.
    """
    from cyclops.config import Settings

    folder = _session_with_video(tmp_path)
    settings = Settings(
        api_key="", sessions_dir=folder.parent, projects_dir=tmp_path / "projects"
    )
    (tmp_path / "projects").mkdir(exist_ok=True)
    assert folder / card.VIDEOS not in recall.image_folders(settings)


# ---------------------------------------------------------------- the YOUTUBE tab


def test_the_tab_lists_a_reference_with_somewhere_to_watch_it(tmp_path) -> None:
    from cyclops import shelf

    project = _project(tmp_path)
    store.copy_videos(project, _session_with_video(tmp_path))
    found = shelf.videos("Chisels", project.path)
    assert len(found) == 1
    one = found[0]
    assert one["title"] == "Honing a Chisel"
    assert one["channel"] == "Paul Sellers"
    assert one["clock"] == "5:12"
    assert one["watch"] == "https://www.youtube.com/watch?v=abc123&t=312s"
    # The local copy, never i.ytimg.com: this page is the panel as well as a laptop.
    assert one["url"].startswith("/project-media/")


def test_a_title_card_is_not_listed_among_the_project_photos(tmp_path) -> None:
    """shelf.pictures walks the whole tree, so Videos/ has to be stepped over by name.

    Otherwise every video somebody watches turns up in PHOTOS as a still of a stranger's
    workshop, and the sweep that files them makes more every week.
    """
    from cyclops import shelf

    project = _project(tmp_path)
    store.copy_videos(project, _session_with_video(tmp_path))
    assert shelf.pictures("Chisels", project.path) == []


def test_a_stray_picture_in_videos_is_not_a_reference(tmp_path) -> None:
    """The sidecar is the list. A jpg with no line in it is a stray, not something to show."""
    from cyclops import shelf

    project = _project(tmp_path)
    store.copy_videos(project, _session_with_video(tmp_path))
    (project.videos_dir / "dropped-in-by-hand.jpg").write_bytes(b"\xff\xd8x")
    assert [one["path"] for one in shelf.videos("Chisels", project.path)] == [
        "Videos/2026-09-11_14-30-00_honing-a-chisel.jpg"
    ]


def test_the_tab_is_wired_all_the_way_through() -> None:
    """Six things have to agree on the word "youtube" or the tab is dead in a different way
    each time: the rail button, the section list, the render table, and the route."""
    from pathlib import Path as P

    from cyclops import admin

    root = P(admin.__file__).parent
    app = (root / "static" / "app.js").read_text()
    page = (root / "templates" / "cyclops" / "dashboard.html").read_text()
    urls = (root / "urls.py").read_text()
    assert 'data-sec="youtube"' in page
    assert "'youtube'" in app.split("const SECTIONS")[1].split("]")[0]
    assert "youtube:" in app and "showYouTube" in app
    assert "youtube" in urls
    # Its own array and its own attribute: gallery() overwrites `shots` for the picture
    # galleries, and a press on a video opens YouTube rather than the lightbox.
    assert "data-tube" in app and "data-tube" not in app.split("function gallery")[1][:400]


def test_a_reference_says_it_is_a_video(tmp_path) -> None:
    """The one word that tells a reference from a photograph, and it was measured.

    Without it, "that video about wiring the m-unit" ranked the session's own summary above
    the video it was summarising - the summary was the only thing in the session that said
    "video" at all. ROLES does the same job for a photograph against a diagram.
    """
    one = [i for i in recall.session_items(_session_with_video(tmp_path)) if i.kind == "video"][0]
    assert one.text.startswith("a video")
    assert "picture" not in one.text
