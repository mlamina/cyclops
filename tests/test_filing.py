"""Filing a session's photos into its project: all of them, captioned where the curator spoke."""

from __future__ import annotations

from pathlib import Path

from cyclops import projects
from cyclops.projects.deps import PhotoLine, SessionFolder
from cyclops.projects.models import PhotoPick, PhotoPicks


def test_every_photo_is_filed_and_the_curator_only_supplies_captions() -> None:
    """The curator captions; it does not choose. A photo it skipped keeps its focus as caption."""
    photos = (
        PhotoLine("photos/12-52-27_you.jpg", "you", 73.0, "", ""),
        PhotoLine("photos/12-55-10_you.jpg", "you", 236.0, "the patch bay", ""),
        PhotoLine("photos/12-57-47_you.jpg", "you", 393.0, "", "the nice one"),
    )
    session = SessionFolder(Path("s"), "u", "", "2026-09-12", "", "", "", "", photos)
    picks = PhotoPicks(picks=[PhotoPick(file="photos/12-57-47_you.jpg", caption="The Strat")])
    assert projects._captioned(session, picks) == [
        ("photos/12-52-27_you.jpg", ""),
        ("photos/12-55-10_you.jpg", "the patch bay"),
        ("photos/12-57-47_you.jpg", "The Strat"),
    ]
