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

from cyclops import shelf
from cyclops.projects import data

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
