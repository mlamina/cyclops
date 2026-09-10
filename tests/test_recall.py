"""What the card flattens into, what a reconcile decides to do about it, and what ranks first.

Pure logic, per the rule in pyproject.toml: no camera, no key, no network. The two model calls
this feature makes - one to caption a picture, one to embed a string - are the two things not
exercised here, and they are the two things `cyclops-index --search` on the Pi is for.

The reconcile is tested by stubbing `recall.embed` with a function that turns text into vectors
deterministically, which is enough to check the part that can actually be wrong: *which* items it
decides to embed. Whether an embedding is any good is not a question a unit test can ask.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cyclops import around, captions, card, imagine, indexer, recall
from cyclops.config import Settings

# ------------------------------------------------------------------ fixtures


def png(width: int = 32, height: int = 24, colour=(200, 30, 30)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(out, format="PNG")
    return out.getvalue()


def jpeg(width: int = 32, height: int = 24) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (width, height), (30, 30, 200)).save(out, format="JPEG")
    return out.getvalue()


@pytest.fixture
def card_dir(tmp_path: Path) -> Path:
    """A project and a session, shaped exactly as store.py and session.py write them."""
    project = tmp_path / "projects" / "BMW R80RT"
    (project / "Photos").mkdir(parents=True)
    (project / "Datasheets").mkdir(parents=True)
    (project / "Eye Designs").mkdir(parents=True)

    (project / "README.md").write_text(
        "---\nproject: BMW R80RT\nkey: bmw r80rt\nstatus: active\n---\n\n"
        "# BMW R80RT\n\nA 1987 boxer being rewired.\n\n"
        "## Where it stands\n\nFront end mostly back together, wiring still open.\n",
        encoding="utf-8",
    )
    (project / "Log.md").write_text(
        "# BMW R80RT - log\n\nOne entry per session.\n\n"
        "## Friday 4 September 2026 - Recorded rebuild status and front brake torque\n\n"
        "Saved the front brake caliper cap screw torque at 60 to 65 newton meters.\n\n"
        "![Discussing the missing torque spec](Photos/2026-09-04_16-22-00_you.jpg)\n\n"
        "<!-- cyclops:session 602c3420-7d72-49aa-a56b-0dbee6dd191c · x · 6m 09s -->\n",
        encoding="utf-8",
    )
    (project / "Photos" / "2026-09-04_16-22-00_you.jpg").write_bytes(jpeg())
    (project / "Datasheets" / "bolts.txt").write_text("VESA 100x100, M4x10\n", encoding="utf-8")
    (project / "Eye Designs" / "eye-round6.png").write_bytes(png())

    session = tmp_path / "sessions" / "2026-09-04_16-17-13_bmw-r80rt-build-status"
    (session / card.PHOTOS).mkdir(parents=True)
    (session / card.SUMMARY_NAME).write_text(
        "# Recorded rebuild status\n\nWent over the front end and wrote down a torque figure.\n",
        encoding="utf-8",
    )
    (session / card.PHOTOS / "16-22-00_you.jpg").write_bytes(jpeg())
    (session / card.LOG_NAME).write_text(
        "\n".join(
            json.dumps(r)
            for r in (
                {"t": 0.0, "type": "session", "started": "2026-09-04T16:17:13-07:00"},
                {"t": 4.0, "type": "you", "text": "Here is the front end, have a look."},
                {"t": 12.0, "type": "photo", "by": "you", "file": "photos/16-22-00_you.jpg"},
                {
                    "t": 14.0,
                    "type": "cyclops",
                    "text": "That is the caliper hanging off its mount.",
                },
                {"t": 40.0, "type": "project", "action": "tracked", "name": "BMW R80RT"},
                {"t": 900.0, "type": "you", "text": "Anyway, about the kitchen tap."},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def settings(card_dir: Path) -> Settings:
    return Settings(
        api_key="test-key",
        projects_dir=card_dir / "projects",
        sessions_dir=card_dir / "sessions",
    )


def fake_vectors(texts: list[str]) -> np.ndarray:
    """Deterministic unit vectors from text, so ranking is checkable without a model.

    A bag of characters, which is not an embedding and does not pretend to be. It is stable and
    it distinguishes strings, which is all the ranking assertions below rely on.
    """
    rows = np.zeros((len(texts), recall.EMBED_DIMS), dtype=np.float32)
    for n, text in enumerate(texts):
        for character in text.lower():
            rows[n][ord(character) % recall.EMBED_DIMS] += 1.0
    return recall.normalise(rows)


@pytest.fixture
def no_network(monkeypatch):
    """`recall.embed`, without one. Counts what it was asked to embed."""
    asked: list[list[str]] = []

    async def embed(texts, client=None):
        asked.append(list(texts))
        return fake_vectors(texts)

    monkeypatch.setattr(recall, "embed", embed)
    monkeypatch.setattr(indexer.recall, "embed", embed)
    return asked


@pytest.fixture
def index_at(tmp_path, monkeypatch) -> Path:
    """Point the index file somewhere disposable."""
    target = tmp_path / "cache" / "recall.npz"
    monkeypatch.setattr(indexer, "RECALL_FILE", target)
    monkeypatch.setattr(recall, "RECALL_FILE", target)
    return target


# ------------------------------------------------------------------ the corpus


def test_every_kind_on_the_card_becomes_an_item(settings) -> None:
    kinds = {item.kind for item in recall.corpus(settings)}
    assert kinds == {"photo", "image", "entry", "file"}


def test_a_project_photo_carries_the_caption_the_curator_wrote(settings) -> None:
    """The `![caption](path)` in Log.md is the only description a filed photo starts with."""
    photo = next(
        i for i in recall.corpus(settings) if i.path.endswith("2026-09-04_16-22-00_you.jpg")
    )
    assert "Discussing the missing torque spec" in photo.text
    assert photo.scope == "project:BMW R80RT"


def test_a_photo_caption_joins_the_curators_words_rather_than_replacing_them(settings) -> None:
    """Both answer different questions about one picture, so both are embedded."""
    photos = settings.projects_dir / "BMW R80RT" / "Photos"
    captions.write(photos, {"2026-09-04_16-22-00_you.jpg": "A torque table, 60-65 Nm"})
    photo = next(
        i for i in recall.corpus(settings) if i.path.endswith("2026-09-04_16-22-00_you.jpg")
    )
    assert "A torque table, 60-65 Nm" in photo.text
    assert "Discussing the missing torque spec" in photo.text


def test_an_uncaptioned_dropped_image_still_says_something(settings) -> None:
    """Its folder and filename are real words about it, until the service captions it properly."""
    found = next(i for i in recall.corpus(settings) if i.path.endswith("eye-round6.png"))
    assert found.kind == "image"
    assert "eye round6" in found.text
    assert "Eye Designs" in found.text


def test_a_captioned_picture_still_says_where_it_lives(settings) -> None:
    """The caption is about the frame, so nothing in it says which project this belongs to."""
    photos = settings.projects_dir / "BMW R80RT" / "Photos"
    captions.write(photos, {"2026-09-04_16-22-00_you.jpg": "A torque table, 60-65 Nm"})
    found = next(
        i for i in recall.corpus(settings) if i.path.endswith("2026-09-04_16-22-00_you.jpg")
    )
    assert "BMW R80RT" in found.text
    assert "60-65 Nm" in found.text


def test_what_gets_read_out_is_the_caption_and_nothing_else(settings) -> None:
    """`title` is spoken and printed. Where a file lives has no business being either."""
    photos = settings.projects_dir / "BMW R80RT" / "Photos"
    captions.write(photos, {"2026-09-04_16-22-00_you.jpg": "A torque table, 60-65 Nm"})
    found = next(
        i for i in recall.corpus(settings) if i.path.endswith("2026-09-04_16-22-00_you.jpg")
    )
    assert found.title == "A torque table, 60-65 Nm"


def test_a_photograph_and_a_drawing_say_which_they_are(settings) -> None:
    """Asking for a photo of the bike returned a drawing of it three sessions running."""
    photos = settings.sessions_dir / "2026-09-04_16-17-13_bmw-r80rt-build-status" / card.PHOTOS
    (photos / "16-29-00_you.jpg").write_bytes(jpeg(33, 24))
    (photos / "16-30-00_drawn.jpg").write_bytes(jpeg(34, 24))
    found = {Path(i.path).name: i.text for i in recall.corpus(settings)}
    assert recall.ROLES["you"] in found["16-29-00_you.jpg"]
    assert recall.ROLES["drawn"] in found["16-30-00_drawn.jpg"]


def test_log_entries_carry_their_prose_not_just_the_heading(settings) -> None:
    entry = next(
        i for i in recall.corpus(settings) if i.kind == "entry" and "torque" in i.title.lower()
    )
    assert "60 to 65 newton meters" in entry.text


def test_the_bookkeeping_comment_never_reaches_the_index(settings) -> None:
    """`<!-- cyclops:session ... -->` is a terminator store.py needs and a reader never wants."""
    assert not any("cyclops:session" in item.text for item in recall.corpus(settings))


def test_readme_and_log_are_not_also_swept_up_as_plain_text(settings) -> None:
    """They are read as structured sources; indexing them twice would chunk them arbitrarily."""
    files = [i for i in recall.corpus(settings) if i.kind == "file"]
    assert [Path(i.path).name for i in files] == ["bolts.txt"]


def test_the_captions_sidecar_is_never_itself_an_item(settings) -> None:
    photos = settings.projects_dir / "BMW R80RT" / "Photos"
    captions.write(photos, {"2026-09-04_16-22-00_you.jpg": "A torque table"})
    assert not any(i.path.endswith(card.CAPTIONS_NAME) for i in recall.corpus(settings))


def test_a_half_edited_log_costs_its_items_and_nothing_else(settings) -> None:
    (settings.projects_dir / "BMW R80RT" / "Log.md").write_bytes(b"\xff\xfe not text at all")
    assert recall.corpus(settings)  # the photos and the datasheet are still there


def test_image_folders_are_the_ones_holding_pictures(settings) -> None:
    names = {f.name for f in recall.image_folders(settings)}
    assert names == {"Photos", "Eye Designs", card.PHOTOS}


# ------------------------------------------------------------------ chunking


def test_a_long_file_is_split_and_capped() -> None:
    chunks = recall._chunks("\n\n".join(f"paragraph {n} " + "word " * 200 for n in range(60)))
    assert len(chunks) == recall.MAX_CHUNKS
    assert all(len(c) <= recall.CHUNK_CHARS for c in chunks)


def test_short_paragraphs_are_packed_rather_than_one_item_each() -> None:
    """Forty one-line bullets as forty items would be forty things too short to carry signal."""
    assert len(recall._chunks("\n\n".join(f"- bullet {n}" for n in range(40)))) == 1


# ------------------------------------------------------------------ the index file


def test_an_index_survives_a_round_trip(tmp_path) -> None:
    items = [
        recall.Item("photo", "/a.jpg", "project:P", "A", "a torque table", 1, 2),
        recall.Item("entry", "/b.md", "project:P", "B", "the fork seals", 3, 4),
    ]
    index = recall.Index(items=items, vectors=fake_vectors([i.text for i in items]))
    target = tmp_path / "recall.npz"
    card.write_bytes(target, recall.dump(index))
    back = recall.load(target)
    assert [i.key for i in back.items] == [i.key for i in items]
    assert np.allclose(back.vectors, index.vectors)


def test_a_missing_index_is_empty_rather_than_an_error(tmp_path) -> None:
    assert len(recall.load(tmp_path / "nothing.npz")) == 0


def test_a_corrupt_index_is_empty_rather_than_an_error(tmp_path) -> None:
    target = tmp_path / "recall.npz"
    target.write_bytes(b"not an npz at all")
    assert len(recall.load(target)) == 0


def test_halves_that_disagree_are_thrown_away(tmp_path) -> None:
    """Vectors and names out of step would rank one thing and name another. Rebuild instead."""
    items = [recall.Item("photo", "/a.jpg", "project:P", "A", "a", 1, 2)]
    index = recall.Index(items=items, vectors=np.zeros((3, recall.EMBED_DIMS), dtype=np.float32))
    target = tmp_path / "recall.npz"
    card.write_bytes(target, recall.dump(index))
    assert len(recall.load(target)) == 0


# ------------------------------------------------------------------ ranking


def test_the_closest_item_ranks_first() -> None:
    items = [
        recall.Item("entry", "/b.md", "project:P", "B", "the fork seals were weeping", 1, 1),
        recall.Item("photo", "/a.jpg", "project:P", "A", "a torque table 60-65 Nm", 1, 1),
    ]
    index = recall.Index(items=items, vectors=fake_vectors([i.text for i in items]))
    hits = recall.rank(index, fake_vectors(["a torque table 60-65 Nm"])[0], scopes=None, limit=5)
    assert hits[0].item.title == "A"
    assert hits[0].score > hits[1].score


def test_scopes_keep_a_search_inside_what_was_asked_for() -> None:
    items = [
        recall.Item("photo", "/a.jpg", "project:Keep", "A", "torque", 1, 1),
        recall.Item("photo", "/b.jpg", "project:Other", "B", "torque", 1, 1),
    ]
    index = recall.Index(items=items, vectors=fake_vectors([i.text for i in items]))
    hits = recall.rank(index, fake_vectors(["torque"])[0], scopes={"project:Keep"}, limit=5)
    assert [h.item.title for h in hits] == ["A"]


def test_ranking_an_empty_index_is_empty_rather_than_an_error() -> None:
    assert recall.rank(recall.empty(), fake_vectors(["x"])[0], scopes=None, limit=5) == []


# ------------------------------------------------------------------ the reconcile


def reconcile(settings) -> bool:
    return asyncio.run(indexer.reconcile(settings))


def test_a_first_reconcile_embeds_everything(settings, no_network, index_at) -> None:
    assert reconcile(settings) is True
    assert len(no_network) == 1
    assert len(no_network[0]) == len(recall.corpus(settings))
    assert len(recall.load(index_at)) == len(recall.corpus(settings))


def test_an_unchanged_card_embeds_nothing_and_rewrites_nothing(settings, no_network, index_at):
    """The reason a 15-minute safety-net sweep is affordable on a box that overheats."""
    reconcile(settings)
    no_network.clear()
    before = index_at.stat().st_mtime_ns

    assert reconcile(settings) is False
    assert no_network == []
    assert index_at.stat().st_mtime_ns == before


def test_only_what_changed_is_embedded_again(settings, no_network, index_at) -> None:
    reconcile(settings)
    no_network.clear()
    (settings.projects_dir / "BMW R80RT" / "Datasheets" / "bolts.txt").write_text(
        "VESA 100x100, M4x10\nM6x20 cap screws, 10 Nm\n", encoding="utf-8"
    )

    assert reconcile(settings) is True
    assert len(no_network) == 1
    assert [t for t in no_network[0] if "cap screws" in t]
    assert all("fork seals" not in t for t in no_network[0])


def test_a_new_file_is_added_without_re_embedding_its_neighbours(settings, no_network, index_at):
    reconcile(settings)
    before = len(recall.load(index_at))
    no_network.clear()
    (settings.projects_dir / "BMW R80RT" / "Datasheets" / "vesa.txt").write_text(
        "VESA 100x100, M4x10\n", encoding="utf-8"
    )

    assert reconcile(settings) is True
    assert len(no_network[0]) == 1
    assert len(recall.load(index_at)) == before + 1


def test_a_deleted_file_leaves_the_index_without_costing_a_call(settings, no_network, index_at):
    reconcile(settings)
    before = len(recall.load(index_at))
    no_network.clear()
    (settings.projects_dir / "BMW R80RT" / "Datasheets" / "bolts.txt").unlink()

    assert reconcile(settings) is True
    assert no_network == []  # dropping a row needs no model
    assert len(recall.load(index_at)) == before - 1
    assert not any("bolts.txt" in i.path for i in recall.load(index_at).items)


def test_a_captioned_photo_is_embedded_again_with_its_new_words(settings, no_network, index_at):
    """A caption written on this sweep must be indexed on this sweep, not the next one."""
    reconcile(settings)
    no_network.clear()
    photos = settings.projects_dir / "BMW R80RT" / "Photos"
    captions.write(photos, {"2026-09-04_16-22-00_you.jpg": "A torque table, 60-65 Nm"})

    assert reconcile(settings) is True
    assert any("60-65 Nm" in t for t in no_network[0])


def test_words_that_changed_are_embedded_again_though_the_picture_did_not(
    settings, no_network, index_at
):
    """Half of what a picture is embedded under is not in the picture. The stamp cannot see it."""
    photos = settings.projects_dir / "BMW R80RT" / "Photos"
    captions.write(photos, {"2026-09-04_16-22-00_you.jpg": "A torque table, 60-65 Nm"})
    reconcile(settings)
    no_network.clear()

    # The curator's words land in Log.md, which leaves `title` - and so `key` - alone.
    log = settings.projects_dir / "BMW R80RT" / "Log.md"
    log.write_text(
        log.read_text(encoding="utf-8").replace(
            "![Discussing the missing torque spec]", "![Shot of the caliper he could not seat]"
        ),
        encoding="utf-8",
    )

    # Both halves in one string, because the Log.md *entry* also changed and carries the new
    # words too. Only the photo's own text has the caption beside them.
    assert reconcile(settings) is True
    assert any("could not seat" in t and "60-65 Nm" in t for t in no_network[0])


# ------------------------------------------------------------------ captions on the card


def test_uncaptioned_lists_only_what_has_no_caption(card_dir) -> None:
    photos = card_dir / "projects" / "BMW R80RT" / "Photos"
    assert [p.name for p in captions.uncaptioned(photos)] == ["2026-09-04_16-22-00_you.jpg"]
    captions.write(photos, {"2026-09-04_16-22-00_you.jpg": "A torque table"})
    assert captions.uncaptioned(photos) == []


def test_a_zero_byte_picture_is_not_ready_to_caption(card_dir) -> None:
    """`card.written`, not `is_file`: a photo still being written has nothing to look at yet."""
    photos = card_dir / "projects" / "BMW R80RT" / "Photos"
    (photos / "half-written.jpg").touch()
    assert "half-written.jpg" not in [p.name for p in captions.uncaptioned(photos)]


def test_a_hand_mangled_captions_file_costs_the_captions_and_not_the_sweep(card_dir) -> None:
    photos = card_dir / "projects" / "BMW R80RT" / "Photos"
    (photos / card.CAPTIONS_NAME).write_text("{ not json", encoding="utf-8")
    assert captions.read(photos) == {}
    assert captions.uncaptioned(photos)  # so it is simply written again


def test_captions_land_whole_or_not_at_all(card_dir) -> None:
    photos = card_dir / "projects" / "BMW R80RT" / "Photos"
    captions.write(photos, {"a.jpg": "one"})
    assert json.loads((photos / card.CAPTIONS_NAME).read_text()) == {"a.jpg": "one"}
    assert card.written(photos / card.CAPTIONS_NAME)


def test_the_sidecar_is_not_counted_as_a_photo(card_dir) -> None:
    """It lives beside the pictures, and the README's `photos:` count must not include it."""
    from cyclops.projects import store

    project = store.Project(key="k", name="BMW R80RT", path=card_dir / "projects" / "BMW R80RT")
    before = store.photo_count(project)
    captions.write(project.photos_dir, {"2026-09-04_16-22-00_you.jpg": "A torque table"})
    assert store.photo_count(project) == before


# ------------------------------------------------------------------ getting it onto the panel


def test_a_png_is_re_encoded_before_it_is_labelled_a_jpeg() -> None:
    """offer_image hard-labels its payload image/jpeg, so a PNG must stop being one first."""
    out = imagine.as_jpeg(png())
    with Image.open(io.BytesIO(out)) as image:
        assert image.format == "JPEG"


def test_a_jpeg_is_handed_back_untouched() -> None:
    blob = jpeg()
    assert imagine.as_jpeg(blob) is blob


def test_transparency_is_flattened_onto_white_rather_than_failing() -> None:
    out = io.BytesIO()
    Image.new("RGBA", (16, 16), (255, 0, 0, 0)).save(out, format="PNG")
    with Image.open(io.BytesIO(imagine.as_jpeg(out.getvalue()))) as image:
        assert image.format == "JPEG"
        assert image.convert("RGB").getpixel((0, 0)) == (255, 255, 255)


def test_something_that_is_not_a_picture_comes_back_unchanged() -> None:
    assert imagine.as_jpeg(b"not an image") == b"not an image"


def test_a_caption_for_a_deleted_picture_is_pruned(card_dir) -> None:
    """The file claims to describe what is in this folder; an entry for nothing makes it a lie."""
    photos = card_dir / "projects" / "BMW R80RT" / "Photos"
    captions.write(photos, {"2026-09-04_16-22-00_you.jpg": "A torque table", "gone.jpg": "nothing"})
    assert captions.departed(photos, captions.read(photos)) == ["gone.jpg"]


def test_nothing_is_pruned_when_the_folder_will_not_list(tmp_path) -> None:
    """Pruning on a failed read would delete every caption in the folder at once."""
    assert captions.departed(tmp_path / "not-there", {"a.jpg": "one"}) == []


def test_the_prune_actually_rewrites_the_file(card_dir, settings) -> None:
    photos = card_dir / "projects" / "BMW R80RT" / "Photos"
    captions.write(photos, {"2026-09-04_16-22-00_you.jpg": "A torque table", "gone.jpg": "nothing"})
    asyncio.run(captions.fill(photos, settings, client=None))
    assert set(captions.read(photos)) == {"2026-09-04_16-22-00_you.jpg"}


def test_an_emptied_captions_file_is_removed_rather_than_left_blank(card_dir, settings) -> None:
    """A `{}` in a folder somebody browses is litter, not a record."""
    designs = card_dir / "projects" / "BMW R80RT" / "Eye Designs"
    captions.write(designs, {"vanished.png": "nothing"})
    asyncio.run(captions.fill(designs, settings, client=None))
    assert not (designs / card.CAPTIONS_NAME).exists()


def test_the_same_picture_in_two_folders_is_one_item(settings) -> None:
    """copy_photos puts an identical jpeg in the project, and it took two of five search slots."""
    shots = [i for i in recall.corpus(settings) if i.kind == "photo"]
    assert len({i.mark for i in shots}) == len(shots)


def test_the_project_copy_is_the_one_that_survives(settings) -> None:
    """Both are the same pixels, so it turns on which is the better thing to hand somebody."""
    shot = next(i for i in recall.corpus(settings) if i.kind == "photo")
    assert shot.scope == "project:BMW R80RT"


def test_a_folder_whose_only_picture_is_a_duplicate_is_still_captioned(settings) -> None:
    """image_folders drives the captioner, so a deduped folder must not vanish from it."""
    assert (
        settings.sessions_dir / "2026-09-04_16-17-13_bmw-r80rt-build-status" / card.PHOTOS
    ) in recall.image_folders(settings)


def test_different_pictures_are_never_collapsed(tmp_path) -> None:
    items = [
        recall.Item("photo", "/a.jpg", "project:P", "A", "a", 1, 1, mark="aaa"),
        recall.Item("photo", "/b.jpg", "project:P", "B", "b", 1, 1, mark="bbb"),
    ]
    assert len(recall.dedupe(items)) == 2


def test_an_unhashable_picture_is_kept_rather_than_merged(tmp_path) -> None:
    """An empty mark means "could not read it", which must never collapse two real photos."""
    items = [
        recall.Item("photo", "/a.jpg", "project:P", "A", "a", 1, 1, mark=""),
        recall.Item("photo", "/b.jpg", "project:P", "B", "b", 1, 1, mark=""),
    ]
    assert len(recall.dedupe(items)) == 2


def test_one_picture_gets_one_caption_however_many_copies_exist(card_dir, settings) -> None:
    """Two passes over the same page disagreed about the numbers on it. Describe it once."""
    photos = card_dir / "projects" / "BMW R80RT" / "Photos"
    session_photos = (
        card_dir / "sessions" / "2026-09-04_16-17-13_bmw-r80rt-build-status" / card.PHOTOS
    )
    captions.write(photos, {"2026-09-04_16-22-00_you.jpg": "A torque table, 60-65 Nm"})

    known = indexer._known_captions([photos, session_photos])
    made = asyncio.run(captions.fill(session_photos, settings, client=None, known=known))

    assert made == 1  # written without a model call, because the content was already described
    assert captions.read(session_photos)["16-22-00_you.jpg"] == "A torque table, 60-65 Nm"


# ------------------------------------------------------------------ the words around a picture


@pytest.fixture
def photos(settings) -> Path:
    return settings.sessions_dir / "2026-09-04_16-17-13_bmw-r80rt-build-status" / card.PHOTOS


def test_the_words_around_a_picture_come_from_both_sides_of_it(photos) -> None:
    """A photograph opens a topic more often than it closes one, so before is the emptier half."""
    block = around.shots(photos)["16-22-00_you.jpg"]
    assert "have a look" in block
    assert "hanging off its mount" in block  # said two seconds after the shutter
    assert "kitchen tap" not in block  # and fifteen minutes later is not "around" anything


def test_a_picture_says_which_project_was_named_out_loud(photos) -> None:
    """The cheapest line here and the one that turns "a blue motorcycle" into the trike."""
    assert "BMW R80RT" in around.shots(photos)["16-22-00_you.jpg"]


def test_a_picture_nobody_put_there_has_no_session_behind_it(settings) -> None:
    """A project's own folders were never waiting on a conversation."""
    assert around.shots(settings.projects_dir / "BMW R80RT" / "Photos") == {}
    assert around.ready(settings.projects_dir / "BMW R80RT" / "Eye Designs") is True


def test_a_live_session_is_not_ready_to_describe(photos) -> None:
    """Its words are still arriving. The gate, and the only thing that pins it."""
    handle = (photos.parent / card.LOG_NAME).open("a", encoding="utf-8")
    try:
        card.claim(handle)
        assert around.ready(photos) is False
    finally:
        handle.close()
    assert around.ready(photos) is True


def test_a_session_that_was_never_described_is_given_up_on_in_the_end(photos) -> None:
    """The naming child can die, or be offline. A picture must not stay silent because of it."""
    (photos.parent / card.SUMMARY_NAME).unlink()
    assert around.ready(photos) is False
    old = time.time() - around.PATIENCE_S - 1
    os.utime(photos.parent / card.LOG_NAME, (old, old))
    assert around.ready(photos) is True


def test_a_live_session_keeps_its_pictures_undescribed(settings, photos, monkeypatch) -> None:
    """The gate, from the outside: no model is asked about a picture mid-conversation."""
    asked: list[Path] = []

    async def describe(path, client, words=""):
        asked.append(path)
        return "a caption"

    monkeypatch.setattr(captions, "describe", describe)
    handle = (photos.parent / card.LOG_NAME).open("a", encoding="utf-8")
    try:
        card.claim(handle)
        asyncio.run(captions.fill(photos, settings, object()))
        assert asked == []
    finally:
        handle.close()
    asyncio.run(captions.fill(photos, settings, object()))
    assert [p.name for p in asked] == ["16-22-00_you.jpg"]


def test_a_hero_shot_is_described_where_its_words_are(settings, monkeypatch) -> None:
    """Its copy in the project is the same bytes, so whichever is described first wins for both."""
    seen: list[str] = []

    async def describe(path, client, words=""):
        seen.append(words)
        return "a caption"

    monkeypatch.setattr(captions, "describe", describe)
    monkeypatch.setattr(captions, "client_for", lambda settings: object())
    asyncio.run(indexer.caption_pass(settings))
    assert seen and "BMW R80RT" in seen[0], "the copy with no session behind it was described first"


def test_a_picture_nothing_was_said_about_gets_the_prompt_it_always_got(photos) -> None:
    """Adding context must not quietly reword every caption on a card that has no logs."""
    assert captions.prompt_for("") == captions.CAPTION_PROMPT
    assert captions.prompt_for(around.shots(photos)["16-22-00_you.jpg"]) != captions.CAPTION_PROMPT


def test_recaptioning_writes_over_words_that_are_already_there(settings, photos, monkeypatch):
    """A caption is only ever written for a picture that has none, so nothing else reaches these."""

    async def describe(path, client, words=""):
        return "said again"

    monkeypatch.setattr(captions, "describe", describe)
    captions.write(photos, {"16-22-00_you.jpg": "said once"})
    asyncio.run(captions.fill(photos, settings, object()))
    assert captions.read(photos)["16-22-00_you.jpg"] == "said once"
    asyncio.run(captions.fill(photos, settings, object(), again=True))
    assert captions.read(photos)["16-22-00_you.jpg"] == "said again"
