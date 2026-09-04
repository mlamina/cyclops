"""What the web page is told is in the projects folder.

The counterpart to ``test_library.py``, on the other half of the card. Imports
``cyclops.shelf`` and ``cyclops.projects.data`` only: between them they need no key, no camera
and no network, so this runs anywhere in a second like the rest of the suite.

Two things get most of the attention here, because they are the two that would be quiet failures
rather than loud ones. A path that escapes the project folder would serve a file nobody meant to
publish, and it would do it without an error anywhere. And ``render_markdown`` puts a string into
the page's ``innerHTML``, which is the one place in this repo where the wrong answer is executed
rather than displayed.
"""

from __future__ import annotations

import pytest

from cyclops import shelf
from cyclops.projects import data, store

FRONT = """---
project: Pelican Display Mount
key: pelican-display-mount
status: active
updated: 2026-08-26
sessions: 3
photos: 2
---

# Pelican Display Mount

A VESA arm mount cut from 4 mm acrylic.
"""


# ------------------------------------------------------------------ building a shelf to read


def make(root, name="Pelican Display Mount", *, readme=FRONT, log=None, photos=(), extra=()):
    """One project folder, in whatever state the test needs it."""
    folder = root / name
    (folder / shelf.store.PHOTOS).mkdir(parents=True)
    if readme is not None:
        (folder / shelf.store.README_NAME).write_text(readme, encoding="utf-8")
    if log is not None:
        (folder / shelf.store.LOG_NAME).write_text(log, encoding="utf-8")
    for picture in photos:
        (folder / shelf.store.PHOTOS / picture).write_bytes(b"\xff\xd8jpeg")
    for filename, text in extra:
        (folder / filename).write_text(text, encoding="utf-8")
    return folder


def names(listing):
    return [entry["name"] for entry in listing["entries"]]


# ------------------------------------------------------------------ the shelf


def test_projects_lists_a_folder_with_its_first_photo(tmp_path):
    make(tmp_path, photos=["2026-08-26_16-48-48_cyclops.jpg", "2026-08-27_09-00-00_you.jpg"])
    found = shelf.projects(tmp_path)
    assert len(found) == 1
    one = found[0]
    assert one["name"] == "Pelican Display Mount"  # the folder, which is the id in every URL
    assert one["title"] == "Pelican Display Mount"
    assert one["tagline"] == "A VESA arm mount cut from 4 mm acrylic."
    assert one["sessions"] == 3
    # The earliest photo, and reachable: spaces in a folder name have to survive the trip.
    assert one["thumb"] == (
        "/project-media/Pelican%20Display%20Mount/Photos/2026-08-26_16-48-48_cyclops.jpg"
    )


def test_a_folder_without_a_readme_is_not_a_project(tmp_path):
    """``store.index``'s rule, and the same answer the voice agent gets - so no second opinion."""
    (tmp_path / "not a project").mkdir()
    assert shelf.projects(tmp_path) == []


def test_projects_with_no_photos_has_no_thumbnail(tmp_path):
    make(tmp_path)
    assert shelf.projects(tmp_path)[0]["thumb"] == ""


# ------------------------------------------------------------------ staying inside the folder


def test_resolve_finds_a_project_and_refuses_anything_else(tmp_path):
    make(tmp_path)
    assert shelf.resolve(tmp_path, "Pelican Display Mount") is not None
    assert shelf.resolve(tmp_path, "nobody") is None
    assert shelf.resolve(tmp_path, "..") is None


def test_inside_refuses_every_way_out_of_the_tree(tmp_path):
    folder = make(tmp_path, photos=["a.jpg"])
    (tmp_path / "secret.txt").write_text("not yours", encoding="utf-8")

    assert shelf.inside(folder, "Photos/a.jpg") is not None
    assert shelf.inside(folder, "") == folder.resolve()
    for escape in ("..", "../secret.txt", "Photos/../../secret.txt", "/etc/passwd"):
        assert shelf.inside(folder, escape) is None, escape


def test_inside_refuses_a_symlink_pointing_out_of_the_project(tmp_path):
    """Resolution happens before the comparison, so a link is judged by where it lands."""
    folder = make(tmp_path)
    (tmp_path / "secret.txt").write_text("not yours", encoding="utf-8")
    (folder / "shortcut.txt").symlink_to(tmp_path / "secret.txt")
    assert shelf.inside(folder, "shortcut.txt") is None


# ------------------------------------------------------------------ one directory


def test_listing_puts_folders_first_and_hides_our_own_bookkeeping(tmp_path):
    folder = make(tmp_path, log="# Log", photos=["a.jpg"])
    # What card.write_bytes leaves behind when a power cut catches it mid-save.
    (folder / ".README.md.tmp").write_text("half a file", encoding="utf-8")

    found = shelf.listing(folder)
    assert names(found) == ["Photos", "Log.md", "README.md"]
    assert found["path"] == "" and found["parent"] == ""
    assert [e["kind"] for e in found["entries"]] == ["dir", "file", "file"]


def test_listing_descends_and_knows_the_way_back(tmp_path):
    folder = make(tmp_path, photos=["a.jpg"])
    found = shelf.listing(folder, "Photos")
    assert names(found) == ["a.jpg"]
    assert found["parent"] == ""
    assert found["entries"][0]["path"] == "Photos/a.jpg"
    assert found["entries"][0]["suffix"] == ".jpg"


def test_listing_refuses_a_path_outside_the_project(tmp_path):
    folder = make(tmp_path)
    assert shelf.listing(folder, "../..") is None
    assert shelf.listing(folder, "README.md") is None  # a file is not a directory


# ------------------------------------------------------------------ markdown


def test_render_markdown_drops_the_frontmatter(tmp_path):
    """It is our machine state, not the model's prose - and rendered it is only field names."""
    html = shelf.render_markdown(FRONT, "Pelican Display Mount")
    assert "<h1>Pelican Display Mount</h1>" in html
    assert "pelican-display-mount" not in html
    assert "sessions:" not in html


def test_render_markdown_points_a_picture_at_a_url_the_page_can_fetch(tmp_path):
    text = "![The bracket](Photos/2026-08-26_16-48-48_cyclops.jpg)"
    html = shelf.render_markdown(text, "Pelican Display Mount")
    assert (
        'src="/project-media/Pelican%20Display%20Mount/Photos/2026-08-26_16-48-48_cyclops.jpg"'
        in html
    )


def test_render_markdown_keeps_a_link_to_another_file_inside_the_page(tmp_path):
    """A hash, never an href: the kiosk's browser must not navigate. See kiosk.py:112-116."""
    html = shelf.render_markdown("The history is in [Log.md](Log.md).", "Pelican Display Mount")
    assert 'href="#/f/Pelican%20Display%20Mount/Log.md"' in html


def test_render_markdown_leaves_an_absolute_link_alone(tmp_path):
    html = shelf.render_markdown("[the spec](https://example.com/x)", "P")
    assert 'href="https://example.com/x"' in html


def test_render_markdown_drops_the_log_terminator(tmp_path):
    """``<!-- cyclops:session ... -->`` is how the sweep knows an entry finished, not prose."""
    text = "Cut the acrylic.\n\n<!-- cyclops:session 1111 · 2026-08-26 · 18m -->"
    html = shelf.render_markdown(text, "P")
    assert "cyclops:session" not in html
    assert "Cut the acrylic." in html


def test_render_markdown_never_hands_the_page_a_tag_it_did_not_write(tmp_path):
    """The page sets this with innerHTML, so a tag someone typed must arrive as words."""
    html = shelf.render_markdown("Careful: <script>alert(1)</script> here.", "P")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_render_markdown_still_renders_a_table(tmp_path):
    """Someone hand-editing their own README is the reason for the extensions."""
    html = shelf.render_markdown("| Part | Size |\n|---|---|\n| Bolt | M4 |\n", "P")
    assert "<table>" in html and "<td>Bolt</td>" in html


# ------------------------------------------------------------------ one file


def test_view_reads_a_workbook_as_its_tabs_in_sheet_order(tmp_path):
    folder = make(tmp_path)
    book = data.Book()
    book.put("Torque specs", "Caliper bolt torque", "28 Nm", "cold, threadlocked")
    book.put("Torque specs", "Lid screws", "2 Nm")
    book.put("Dimensions", "Acrylic thickness", "4 mm")
    (folder / shelf.store.DATA_NAME).write_bytes(data.dump(book))

    found = shelf.view("P", folder, shelf.store.DATA_NAME)
    assert found["kind"] == "sheet"
    assert [tab["name"] for tab in found["tabs"]] == ["Torque specs", "Dimensions"]
    assert found["tabs"][0]["rows"] == [
        {"key": "Caliper bolt torque", "value": "28 Nm", "note": "cold, threadlocked"},
        {"key": "Lid screws", "value": "2 Nm", "note": ""},
    ]


def test_view_reads_a_workbook_nobody_can_open_as_an_empty_one(tmp_path):
    """``data.load``'s promise, kept all the way out to the page: never a traceback."""
    folder = make(tmp_path, extra=[(shelf.store.DATA_NAME, "not a zip")])
    assert shelf.view("P", folder, shelf.store.DATA_NAME)["tabs"] == []


def test_view_classifies_by_suffix(tmp_path):
    folder = make(tmp_path, log="# Log", photos=["a.jpg"], extra=[("spec.json", "{}")])
    (folder / "notes.odt").write_bytes(b"\x00" * 40)

    assert shelf.view("P", folder, "Log.md")["kind"] == "markdown"
    assert shelf.view("P", folder, "spec.json")["kind"] == "text"
    assert shelf.view("P", folder, "Photos/a.jpg") == {
        "name": "a.jpg",
        "path": "Photos/a.jpg",
        "size": 6,
        "kind": "image",
        "url": "/project-media/P/Photos/a.jpg",
    }
    # A format with no viewer here is still a file with a size, and saying so beats a blank page.
    nothing = shelf.view("P", folder, "notes.odt")
    assert nothing["kind"] == "none" and nothing["size"] == 40


def test_view_refuses_a_file_outside_the_project(tmp_path):
    folder = make(tmp_path)
    (tmp_path / "secret.txt").write_text("not yours", encoding="utf-8")
    assert shelf.view("P", folder, "../secret.txt") is None
    assert shelf.view("P", folder, "Photos") is None  # a directory is not a file


def test_view_caps_what_it_will_read(tmp_path):
    """A reader is not a way to pull a gigabyte off the card into a JSON body."""
    folder = make(tmp_path, extra=[("huge.txt", "x" * (shelf.MAX_TEXT_BYTES + 5000))])
    assert len(shelf.view("P", folder, "huge.txt")["text"]) == shelf.MAX_TEXT_BYTES


# ------------------------------------------------------------------ and putting something in


def blocks(data, size=7):
    """One upload, arriving the way the WSGI stream hands it over: in pieces, not all at once."""
    for start in range(0, len(data), size):
        yield data[start : start + size]


def test_a_new_folder_is_made_under_the_name_it_comes_back_as(tmp_path):
    folder = make(tmp_path)
    assert store.make_folder(folder, "Datasheets") == "Datasheets"
    assert (folder / "Datasheets").is_dir()
    # Twice is the same answer: a folder that is already there is the outcome that was asked for.
    assert store.make_folder(folder, "Datasheets") == "Datasheets"


def test_a_new_folder_cannot_be_named_its_way_out_of_the_project(tmp_path):
    """The half of the upload story that is not ``shelf.inside``.

    Containment resolves the *directory* a write is aimed at. This is the other half: the last
    segment, which is a name and never a path, and which arrives from whoever posted the form.
    Every one of these lands inside the project or does not land at all.
    """
    folder = make(tmp_path)
    for escape in ("..", "../secret", "a/b", "/etc", "..\\..\\etc", "."):
        made = store.make_folder(folder, escape)
        if made is not None:
            assert (folder / made).parent == folder
            assert "/" not in made and made not in ("..", ".")
    assert not (tmp_path / "secret").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["Pelican Display Mount"]


def test_a_name_with_nothing_in_it_is_declined_rather_than_invented(tmp_path):
    folder = make(tmp_path)
    assert store.make_folder(folder, "///") is None
    assert store.make_folder(folder, "   ") is None
    assert store.make_folder(folder, "") is None


def test_an_upload_lands_whole_and_keeps_its_extension(tmp_path):
    folder = make(tmp_path)
    landed = store.receive(folder, "brake.jpg", blocks(b"\xff\xd8jpeg" * 40), limit=9999)
    assert landed == "brake.jpg"
    assert (folder / "brake.jpg").read_bytes() == b"\xff\xd8jpeg" * 40
    # And nothing of the write itself is left beside it for the browser to list.
    assert [p.name for p in folder.iterdir() if p.name.startswith(".")] == []


def test_an_upload_keeps_its_extension_even_when_the_name_is_too_long(tmp_path):
    """The reason the name is sanitized in two pieces rather than one.

    ``safe_folder_name`` cuts at 64 characters on a word boundary, so a long enough filename put
    through it whole comes back without its suffix - and the suffix is what ``shelf.view`` and
    the page's mark column dispatch on, so losing it changes what the file is.
    """
    folder = make(tmp_path)
    long = ("a very long name that goes on and on and on and on and on and on and on" * 2) + ".jpg"
    landed = store.receive(folder, long, blocks(b"x"), limit=99)
    assert landed.endswith(".jpg")
    assert shelf.view("P", folder, landed)["kind"] == "image"


def test_an_upload_cannot_be_named_its_way_out_of_the_folder(tmp_path):
    folder = make(tmp_path)
    (tmp_path / "keep.txt").write_text("not yours", encoding="utf-8")
    for escape in ("../keep.txt", "../../keep.txt", "/etc/passwd", "Photos/../../keep.txt"):
        landed = store.receive(folder, escape, blocks(b"pwned"), limit=99)
        assert landed is None or (folder / landed).parent == folder
    assert (tmp_path / "keep.txt").read_text(encoding="utf-8") == "not yours"


def test_an_upload_named_only_a_dot_is_declined(tmp_path):
    """A dotfile would land and then be invisible, which is a worse answer than a refusal.

    ``shelf.listing`` hides dotfiles - that is how the scratch names ``card.py`` writes through
    stay out of the browser - so a file uploaded as ``.bashrc`` would be on the card and absent
    from the only view of the card there is.
    """
    folder = make(tmp_path)
    landed = store.receive(folder, ".bashrc", blocks(b"x"), limit=99)
    assert landed is None or not landed.startswith(".")
    assert store.receive(folder, "...", blocks(b"x"), limit=99) is None


def test_an_upload_replaces_a_file_of_the_same_name(tmp_path):
    folder = make(tmp_path)
    store.receive(folder, "spec.txt", blocks(b"first draft"), limit=99)
    store.receive(folder, "spec.txt", blocks(b"the corrected one"), limit=99)
    assert (folder / "spec.txt").read_text(encoding="utf-8") == "the corrected one"
    assert len([p for p in folder.iterdir() if p.name.startswith("spec")]) == 1


def test_an_upload_over_the_ceiling_leaves_nothing_behind(tmp_path):
    """Refused and not truncated. A file cut off at the cap would look whole and not be."""
    folder = make(tmp_path)
    with pytest.raises(ValueError):
        store.receive(folder, "huge.bin", blocks(b"x" * 500), limit=100)
    assert not (folder / "huge.bin").exists()
    assert [p.name for p in folder.iterdir() if p.name.startswith(".")] == []
