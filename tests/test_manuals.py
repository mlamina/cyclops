"""Manuals: what one is, what reading produces, and how a page reaches the voice agent.

Pure logic. No PDF is committed to the repo - Pillow, which is already a dependency, writes the
fixture - and no page is ever rendered or read by a model here. What a vision model makes of a
manual page is not a thing a fast suite can assert on; what the code does with the answer is.
"""

from __future__ import annotations

import asyncio
import io
import json

import pytest
from PIL import Image

from cyclops import manuals, reading, recall
from cyclops.config import Settings

PAGES = 3


@pytest.fixture(scope="module")
def pdf_bytes() -> bytes:
    """A real, multi-page PDF. Module-scoped: building one per test is setup nobody asked for."""
    sheets = [Image.new("RGB", (612, 792), "white") for _ in range(PAGES)]
    out = io.BytesIO()
    sheets[0].save(out, "PDF", save_all=True, append_images=sheets[1:])
    return out.getvalue()


@pytest.fixture
def card(tmp_path, pdf_bytes):
    """A card with one manual on it, read, with a figure on its second page."""
    folder = tmp_path / "manuals" / "mo-unit"
    folder.mkdir(parents=True)
    (folder / "mo-unit.pdf").write_bytes(pdf_bytes)
    (folder / manuals.PAGES_NAME).write_text(
        json.dumps(
            {
                "1": {"heading": "Cover", "text": "mo.unit blue instruction manual", "figures": []},
                "2": {
                    "heading": "Connecting indicator lights",
                    "text": "two diodes 1N4001 to Turn L and Turn R",
                    "figures": [{"what": "a wiring diagram for the indicators", "kind": "wiring"}],
                },
            }
        )
    )
    pages = folder / manuals.PAGES_DIR
    pages.mkdir()
    for n in (1, 2):
        (pages / f"{n:04d}.jpg").write_bytes(b"\xff\xd8\xff\xe0 not really a jpeg")
    return tmp_path


@pytest.fixture
def settings(card) -> Settings:
    return Settings(
        api_key="test-key",
        manuals_dir=card / "manuals",
        projects_dir=card / "projects",
        sessions_dir=card / "sessions",
    )


@pytest.fixture
def manual(settings) -> manuals.Manual:
    return manuals.catalog(settings)[0]


# ------------------------------------------------------------------ what a manual is


def test_a_folder_with_no_pdf_is_not_a_manual(card):
    """A half-finished upload must not be offered to the model as something it can read."""
    (card / "manuals" / "empty").mkdir()
    assert [m.path.name for m in manuals.index(card / "manuals")] == ["mo-unit"]


def test_identity_survives_a_round_trip_through_the_frontmatter(manual):
    manual.part = "motogadget mo.unit blue body control module"
    manual.aliases = ["m.unit", "the brain"]
    manual.pages, manual.read = PAGES, 2
    manuals.write_identity(manual)

    back = manuals._manual_from(manual.path)
    assert back.part == manual.part
    assert back.aliases == manual.aliases
    assert (back.pages, back.read) == (PAGES, 2)


def test_a_manual_is_found_by_the_names_it_gets_called_out_loud(manual):
    """The whole reason this is a domain object: one manual, many spoken names."""
    manual.part = "motogadget mo.unit blue body control module"
    manual.aliases = ["m.unit", "munit"]
    manuals.write_identity(manual)
    held = [manuals._manual_from(manual.path)]

    for spoken in ("m.unit", "M.UNIT", "munit", "motogadget mo.unit"):
        assert manuals.find(held, spoken) is not None, spoken
    assert manuals.find(held, "brake pads") is None


def test_only_a_pdf_is_taken_in(tmp_path, pdf_bytes):
    """This folder has one meaning, so a .docx in it would be a manual nobody can ever read."""
    root = tmp_path / "manuals"
    assert manuals.receive(root, "notes.docx", iter([pdf_bytes]), limit=1 << 20) is None
    assert manuals.receive(root, "GX160.pdf", iter([pdf_bytes]), limit=1 << 20) == "GX160"


# ------------------------------------------------------------------ reading one


def test_the_page_count_comes_off_the_file(manual):
    assert reading.page_count(manual.pdf) == PAGES


def test_something_that_is_not_a_pdf_has_no_pages(tmp_path):
    """Never raises: a truncated upload is a manual with nothing to read, not a failed sweep."""
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf at all")
    assert reading.page_count(broken) == 0


def test_rendering_writes_one_jpeg_per_page(manual, tmp_path):
    made = reading.render(manual.pdf, [1, 3], tmp_path / "out")
    assert sorted(made) == [1, 3]
    assert all(path.read_bytes().startswith(b"\xff\xd8") for path in made.values())


def test_only_pages_with_no_entry_are_queued(manual):
    """The queue is the sidecar, so a power cut costs the pages in flight and nothing else."""
    assert reading.unread(manual, manuals.read_pages(manual.path), PAGES) == [3]


def test_a_page_that_will_not_parse_keeps_its_words(manual):
    """A reply that arrives and is not JSON is not a failure - the words are still the page's.

    Measured on the m.unit manual: one page in 45 came back as truncated JSON, and its text was
    perfectly good. Returning None there would re-read that page every sweep, forever.
    """

    class Answer:
        output_text = '{"heading": "Notice", "text": "this got trunc'

    class Client:
        class responses:
            @staticmethod
            async def create(**_):
                return Answer()

    page = asyncio.run(reading.read_page(manual.pages_dir / "0001.jpg", Client()))
    assert page is not None and "trunc" in page.text


def test_a_failed_call_is_not_recorded_as_a_blank_page(manual):
    """The other half of the same contract: None means ask again, a Page means do not."""

    class Client:
        class responses:
            @staticmethod
            async def create(**_):
                raise OSError("no network")

    assert asyncio.run(reading.read_page(manual.pages_dir / "0001.jpg", Client())) is None


# ------------------------------------------------------------------ finding a page again


def test_a_read_page_becomes_a_showable_item(settings):
    pages = [i for i in recall.corpus(settings) if i.kind == "page"]
    assert len(pages) == 2
    assert all(item.showable for item in pages)
    assert {item.scope for item in pages} == {"manual:mo-unit"}


def test_a_page_item_points_at_the_render_and_not_the_pdf(settings):
    """What the panel shows and what the model reads are pixels, which the PDF is not."""
    page = next(i for i in recall.corpus(settings) if i.kind == "page")
    assert page.path.endswith(".jpg")


def test_what_an_illustration_shows_is_searchable(settings):
    """'Is there a diagram about this' is answered by the figure description, or not at all."""
    page = next(i for i in recall.corpus(settings) if "page 2" in i.title)
    assert "wiring diagram for the indicators" in page.text


def test_the_manual_itself_is_findable_without_naming_a_page(settings, manual):
    """So that "have we got anything on the m.unit" lands even when no one page is a good match."""
    manual.part = "motogadget mo.unit"
    manuals.write_identity(manual)
    entries = [i for i in recall.corpus(settings) if i.kind == "entry"]
    assert any("mo.unit" in item.text for item in entries)


def test_a_page_render_with_no_entry_is_not_indexed(settings, manual):
    """A page nobody has read yet has nothing to search by, so it is not askable for."""
    (manual.pages_dir / "0003.jpg").write_bytes(b"\xff\xd8\xff\xe0 third")
    assert len([i for i in recall.corpus(settings) if i.kind == "page"]) == 2


def test_the_captioner_is_never_pointed_at_a_manual_page(settings):
    """The guard on the expensive mistake, and it is the reason `image_folders` keys on kind.

    A page is showable, and `image_folders` used to mean "every folder holding something
    showable". Left that way the photo captioner describes all of a manual's pages - a vision
    call each, to produce a worse copy of what `pages.json` already holds, and two readings of
    one page that disagree about the numbers printed on it.
    """
    assert recall.image_folders(settings) == []


def test_a_root_that_does_not_exist_yet_is_watched_through_its_parent(tmp_path):
    """Otherwise the first manual ever uploaded waits for the 15-minute sweep.

    ``manuals/`` is created by the first upload, so on a card that has never had one there is no
    directory to hand inotify - and the one upload where "it just appears" matters most was the
    one where it did not. Observed on the Pi, 2026-09-11.
    """
    from cyclops.indexer import _folders_to_watch

    (tmp_path / "sessions").mkdir()
    watched = _folders_to_watch(
        Settings(
            api_key="k",
            projects_dir=tmp_path / "projects",
            sessions_dir=tmp_path / "sessions",
            manuals_dir=tmp_path / "manuals",
        )
    )
    assert tmp_path in watched


def test_a_manual_with_every_page_read_still_gets_named(manual, monkeypatch):
    """The one piece of work a complete manual can still be owed.

    `fill` returns early when there is nothing left to read, and a manual read before `about`
    existed has every page and no description - so every later sweep took that exit and it could
    never gain one. Observed on the Pi, 2026-09-12, on a manual that had been read the day before.
    """
    asked = []

    async def identify(pages, client):
        asked.append(len(pages))
        return {"part": "Raspberry Pi 5", "about": "power, ports, pinout", "aliases": ["pi 5"]}

    monkeypatch.setattr(reading, "identify", identify)
    # every page accounted for, so there is no reading to do
    monkeypatch.setattr(reading, "page_count", lambda pdf: 2)
    manual.pages, manual.read = 2, 2

    made = asyncio.run(reading.fill(manual, None, None))

    assert made == 0, "nothing to read"
    assert asked, "but it still has to be named"
    assert manuals._manual_from(manual.path).about == "power, ports, pinout"


def test_a_description_that_restates_the_label_is_trimmed(manual):
    """The prompt says not to repeat the part name; the model does it anyway about half the time.

    The label is printed beside the description, so a repeat is paid for in every session and
    says nothing. Only a leading repeat goes - the same words further in are a topic.
    """
    manual.part = "mo.unit blue / basic body control module"
    manual.about = "mo unit blue basic body control module specs, installation, wiring"
    manual.aliases = []
    assert manual.line() == "- mo.unit blue / basic body control module — specs, installation, wiring"


def test_a_line_that_runs_long_ends_on_a_whole_topic(manual):
    """Half a topic and a dangling comma is worse than one topic fewer."""
    manual.part = "thing"
    manual.about = ", ".join(f"topic number {n}" for n in range(40))
    manual.aliases = []
    assert not manual.line().rstrip().endswith((",", "—", "-"))
    assert len(manual.line()) <= manuals.MAX_LINE_CHARS
