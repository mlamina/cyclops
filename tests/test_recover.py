"""Repairing what a power cut left behind, and refusing to delete anything that survived it.

Imports ``cyclops.session``, so this suite pulls in OpenCV - keep the fast, pure checks in
``test_card.py``. Nothing here touches a camera, a model or the network: ``_describe`` is the
only step that would, and it is never reached (no key in the environment under test) or is
stubbed out.
"""

from __future__ import annotations

import json

import pytest

from cyclops import card, session

# The two shapes actually found on the Pi's card on 2026-08-28, which is what this is all for.
INCIDENT = "2026-08-28_18-34-03"  # good log + video; session.md and summary.md zero bytes
HUSK = "2026-08-28_16-07-12"  # a zero-byte session.jsonl and nothing else


def log_lines(*, end=True, talk=True):
    records = [{"t": 0.0, "type": "session", "uuid": "u", "started": "2026-08-28T18:34:03-07:00",
                "entrypoint": "kiosk", "model": "gpt-realtime-2.1"}]
    if talk:
        records.append({"t": 1.0, "type": "you", "text": "how tight should this bolt be?"})
        records.append({"t": 2.0, "type": "cyclops", "text": "about 25 newton metres."})
    if end:
        records.append({"t": 9.0, "type": "end", "reason": "stopped", "seconds": 9.0,
                        "slug": "", "photos": 0})
    return "".join(json.dumps(r) + "\n" for r in records)


@pytest.fixture
def card_root(tmp_path, monkeypatch):
    """A sessions/ directory, with the projects sweep and the model both disconnected."""
    root = tmp_path / "sessions"
    root.mkdir()
    monkeypatch.setattr(session, "_file_the_card", lambda settings: None)
    return root


def settings_for(root, *, api_key="", **over):
    from cyclops.config import Settings

    return Settings(api_key=api_key, sessions_dir=root, **over)


def build_incident(root):
    """A session whose teardown was cut off after the log and the video, before the rest."""
    folder = root / INCIDENT
    folder.mkdir()
    (folder / card.LOG_NAME).write_text(log_lines())
    (folder / card.VIDEO).write_bytes(b"an mp4 that survived")
    (folder / card.PAGE_NAME).write_text("")  # the zero-byte files the power cut left
    (folder / card.SUMMARY_NAME).write_text("")
    return folder


# ------------------------------------------------------------------ the incident, repaired


def test_fix_rebuilds_the_zero_byte_page(card_root):
    """The regression. Old --fix asked `page.is_file()`, which a 0-byte file satisfies."""
    folder = build_incident(card_root)
    did = session._fix(folder)

    assert did == [f"wrote {card.PAGE_NAME}"]
    page = (folder / card.PAGE_NAME).read_text()
    assert page, "a zero-byte page must be rewritten, not mistaken for a finished one"
    assert "how tight should this bolt be?" in page
    assert card.triage(folder).verdict == "finished"


def test_fix_is_a_no_op_the_second_time(card_root):
    folder = build_incident(card_root)
    assert session._fix(folder)
    assert session._fix(folder) == [], "a repaired session must not be repaired again"


def test_fix_does_not_invent_a_page_for_an_empty_log(card_root):
    """A zero-byte log used to earn a session.md, which then read as finished forever."""
    folder = card_root / HUSK
    folder.mkdir()
    (folder / card.LOG_NAME).write_text("")

    assert session._fix(folder) == []
    assert not (folder / card.PAGE_NAME).exists()


def test_a_torn_log_still_produces_a_page_that_says_so(card_root):
    folder = card_root / "2026-08-28_19-00-00"
    folder.mkdir()
    (folder / card.LOG_NAME).write_text(log_lines(end=False) + '{"t": 3.0, "type": "yo')

    assert session._fix(folder) == [f"wrote {card.PAGE_NAME}"]
    page = (folder / card.PAGE_NAME).read_text()
    assert "1 record dropped" in page
    assert "about 25 newton metres." in page, "everything before the torn line is still a session"


# ------------------------------------------------------------------ deleting, and refusing to


def test_the_husk_is_removed(card_root):
    folder = card_root / HUSK
    folder.mkdir()
    (folder / card.LOG_NAME).write_text("")

    assert session._remove(folder) == "nothing survived - removed"
    assert not folder.exists()


def test_a_husk_with_a_strangers_file_is_kept(card_root):
    folder = card_root / HUSK
    folder.mkdir()
    (folder / card.LOG_NAME).write_text("")
    (folder / "notes.txt").write_text("remember the brick wall")

    said = session._remove(folder)
    assert "left alone" in said and "notes.txt" in said
    assert folder.exists() and (folder / "notes.txt").is_file()


def test_dry_run_removes_nothing(card_root):
    folder = card_root / HUSK
    folder.mkdir()
    (folder / card.LOG_NAME).write_text("")

    assert session._remove(folder, dry_run=True) == "nothing survived - would remove"
    assert folder.exists()


@pytest.mark.parametrize("keep", ["video", "photo", "page"])
def test_a_folder_with_anything_in_it_is_never_reached_by_removal(card_root, keep):
    folder = card_root / "2026-08-28_20-00-00"
    folder.mkdir()
    (folder / card.LOG_NAME).write_text("")
    if keep == "video":
        (folder / card.VIDEO).write_bytes(b"mp4")
    elif keep == "photo":
        (folder / card.PHOTOS).mkdir()
        (folder / card.PHOTOS / "20-00-01_you.jpg").write_bytes(b"\xff\xd8")
    else:
        (folder / card.PAGE_NAME).write_text("# a page")

    assert card.triage(folder).verdict != "empty", "salvage means it is never a deletion candidate"


# ------------------------------------------------------------------ the whole card


def test_recover_repairs_the_incident_and_deletes_the_husk(card_root, capsys):
    incident = build_incident(card_root)
    husk = card_root / HUSK
    husk.mkdir()
    (husk / card.LOG_NAME).write_text("")

    code = session._recover(settings_for(card_root))
    out = capsys.readouterr().out

    assert not husk.exists(), "the husk goes"
    assert (incident / card.PAGE_NAME).read_text(), "the incident is rebuilt"
    assert f"wrote {card.PAGE_NAME}" in out
    assert "nothing survived - removed" in out
    # No key in these settings, so naming never ran and the folder is honestly still unnamed.
    assert code == 0, "with no key, an unnamed session is not something a retry could fix"


def test_recover_dry_run_changes_nothing(card_root, capsys):
    incident = build_incident(card_root)
    husk = card_root / HUSK
    husk.mkdir()
    (husk / card.LOG_NAME).write_text("")
    before = sorted((p.relative_to(card_root), p.stat().st_size)
                    for p in card_root.rglob("*") if p.is_file())

    session._recover(settings_for(card_root), dry_run=True)
    out = capsys.readouterr().out

    after = sorted((p.relative_to(card_root), p.stat().st_size)
                   for p in card_root.rglob("*") if p.is_file())
    assert before == after, "--dry-run must not touch a single byte"
    assert husk.exists() and incident.exists()
    assert "would write session.md" in out
    assert "would remove" in out


def test_recover_leaves_a_live_session_alone(card_root, capsys):
    """The failure that would matter most: renaming a folder out from under a conversation."""
    folder = build_incident(card_root)
    handle = (folder / card.LOG_NAME).open("a")
    try:
        assert card.claim(handle)
        session._recover(settings_for(card_root))
        out = capsys.readouterr().out
    finally:
        handle.close()

    assert "a session is writing here; left alone" in out
    assert (folder / card.PAGE_NAME).read_text() == "", "not repaired, because it is not ours yet"


def test_recover_sweeps_a_scratch_file_a_kill_left_behind(card_root, capsys):
    folder = build_incident(card_root)
    stray = card.tmp_for(folder / card.PAGE_NAME)
    stray.write_text("half a page")

    session._recover(settings_for(card_root))

    assert not stray.exists()
    assert f"swept {stray.name}" in capsys.readouterr().out


def test_a_failed_mux_asks_systemd_to_come_back(card_root, monkeypatch):
    """A folder a later run could still finish must make the unit retry, via the exit code."""
    from cyclops.record import Mux

    folder = build_incident(card_root)
    (folder / card.PARTS).mkdir()
    monkeypatch.setattr(session, "mux", lambda work, out: Mux(False, "ffmpeg: not found"))
    monkeypatch.setattr(session, "_describe", lambda folder, settings, state=None: (folder, []))

    code = session._recover(settings_for(card_root, api_key="sk-test"))

    assert code == 1
    assert (folder / card.PARTS).is_dir(), "a failed mux keeps the parts for the next attempt"


def test_no_key_is_not_a_failure(card_root):
    """A keyless box is not failing at something it could retry; it is doing what it was asked."""
    build_incident(card_root)
    assert session._recover(settings_for(card_root)) == 0
    assert session._recover(settings_for(card_root, api_key="sk-test"), offline=True) == 0


def test_recover_on_an_empty_card_says_so(tmp_path, capsys):
    root = tmp_path / "sessions"
    root.mkdir()
    assert session._recover(settings_for(root)) == 0
    assert "no sessions in" in capsys.readouterr().out


# ------------------------------------------------------------------ not asking twice


def described(root, name, *, summary=True, slug=""):
    """A session the model has already looked at. An empty slug is its "leave this dated"."""
    folder = root / (f"{name}_{slug}" if slug else name)
    folder.mkdir()
    (folder / card.LOG_NAME).write_text(log_lines())
    (folder / card.PAGE_NAME).write_text("# a page")
    if summary:
        (folder / card.SUMMARY_NAME).write_text("# Checked the microphone.\n")
    return folder


def test_a_session_with_a_summary_and_no_name_is_never_asked_again(card_root, monkeypatch):
    """slugify maps the model's "chat" to "", meaning leave this one dated. That is an answer.

    Asking again buys the same answer at the same price - on every boot, forever, because the
    boot unit retries on a non-zero exit.
    """
    folder = described(card_root, "2026-08-26_17-13-09")
    asked = []
    monkeypatch.setattr("cyclops.slug.describe_session",
                        lambda text, settings: asked.append(text) or None)

    back, did = session._describe(folder, settings_for(card_root, api_key="sk-test"))

    assert asked == [], "a described session must not go back to the model"
    assert (back, did) == (folder, [])


def test_an_unnamed_but_described_session_does_not_fail_the_run(card_root):
    described(card_root, "2026-08-26_17-13-09")
    assert session._recover(settings_for(card_root, api_key="sk-test")) == 0


def test_a_session_that_was_never_described_still_gets_asked(card_root, monkeypatch):
    folder = described(card_root, "2026-08-26_18-00-00", summary=False)
    from cyclops.slug import Description

    monkeypatch.setattr("cyclops.slug.describe_session",
                        lambda text, settings: Description(slug="mic-test", title="A mic test."))

    back, did = session._describe(folder, settings_for(card_root, api_key="sk-test"))

    assert back.name == "2026-08-26_18-00-00_mic-test"
    assert f"wrote {card.SUMMARY_NAME}" in did
