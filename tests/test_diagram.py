"""The diagram spec: what is fatal, what is coerced, and what survives a round trip to the card.

Covered here: :func:`cyclops.diagram.validate` on both sides of the split the module docstring
describes, the sidecar written and read back, and the search that reuses the workbook's scorer.

Deliberately not covered: the model call, which needs a key and is `cyclops-smoke`'s job; and the
drawing itself, which happens in a browser - the render check in ``tests/render_check.mjs`` is
what watches that, because the two failures that matter there (a port label that silently does not
render, a label clipped off the paper) are both invisible to Python.
"""

from __future__ import annotations

import json

import pytest

from cyclops import diagram


def spec(**over):
    """A valid two-node drawing, with whatever a test wants changed."""
    found = {
        "title": "Relay driven from GPIO 17",
        "caption": "A 1k base resistor between the pin and the module.",
        "kind": "wiring",
        "layout": "manual",
        "nodes": [
            {"id": "pi", "type": "box", "label": "Raspberry Pi 5", "at": {"x": 10, "y": 100},
             "size": {"w": 150, "h": 112},
             "ports": [{"id": "GPIO17", "side": "right", "label": "GPIO17", "kind": "signal"}]},
            {"id": "r1", "type": "resistor", "label": "1k", "at": {"x": 266, "y": 96},
             "ports": [{"id": "a", "side": "left"}, {"id": "b", "side": "right"}]},
        ],
        "wires": [{"from": "pi:GPIO17", "to": "r1:a", "kind": "signal", "label": "GPIO 17"}],
    }
    found.update(over)
    return found


# ------------------------------------------------------------------ what is fatal


@pytest.mark.parametrize(
    "broken, says",
    [
        ({"nodes": []}, "no nodes"),
        ({"wires": [{"from": "pi:GPIO17", "to": "ghost:a"}]}, "ghost"),
        ({"wires": [{"from": "pi:GPIO17", "to": "r1:nope"}]}, "nope"),
        ({"nodes": [{"id": "", "type": "box"}]}, "no id"),
    ],
)
def test_structure_is_fatal(broken, says):
    """A wire to a node that is not there draws a wrong picture, and that is worse than none."""
    with pytest.raises(diagram.DiagramError) as caught:
        diagram.validate(spec(**broken))
    # The message is read by the model on the retry, so it has to name the thing to fix.
    assert says in str(caught.value), f"the complaint must name what to fix, got {caught.value!r}"


def test_duplicate_ids_are_fatal():
    """Two nodes with one id means every wire to it is ambiguous, silently."""
    with pytest.raises(diagram.DiagramError, match="share the id"):
        diagram.validate(spec(nodes=[spec()["nodes"][0], spec()["nodes"][0]]))


@pytest.mark.parametrize("payload", ["not an object", ["nope"], 7, None])
def test_non_objects_are_fatal(payload):
    with pytest.raises(diagram.DiagramError):
        diagram.validate(payload)


# ------------------------------------------------------------------ what is coerced


def test_unknown_cosmetics_are_coerced_not_refused():
    """A drawing that is right except for a word it invented is still the drawing asked for."""
    out = diagram.validate(spec(
        kind="interpretive-dance",
        layout="freehand",
        nodes=[{"id": "a", "type": "flux-capacitor", "label": "A",
                "ports": [{"id": "p", "side": "sideways", "kind": "purple"}]}],
        wires=[],
    ))
    assert out["nodes"][0]["type"] == "box"
    assert out["nodes"][0]["ports"][0]["side"] == "left"
    assert out["nodes"][0]["ports"][0]["kind"] == ""
    assert out["kind"] == "block"
    assert out["layout"] == "dagre"


def test_dagre_drops_the_geometry_it_would_overwrite():
    """Positions under an automatic layout are not honoured, so they are not carried either."""
    out = diagram.validate(spec(layout="dagre"))
    assert "at" not in out["nodes"][0] and "size" not in out["nodes"][0]


def test_manual_keeps_the_geometry_it_needs():
    out = diagram.validate(spec(layout="manual"))
    assert out["nodes"][0]["at"] == {"x": 10.0, "y": 100.0}


@pytest.mark.parametrize("bad", [float("nan"), 1e9, "over there", None])
def test_unusable_coordinates_are_dropped_not_drawn(bad):
    """A NaN or a coordinate a mile off the panel puts a node where nobody can see it."""
    out = diagram.validate(spec(nodes=[
        {"id": "a", "type": "box", "label": "A", "at": {"x": bad, "y": 10}, "ports": []},
    ], wires=[]))
    assert "at" not in out["nodes"][0]


def test_long_text_is_cut_to_something_that_fits():
    out = diagram.validate(spec(title="T" * 500, nodes=[
        {"id": "a", "type": "box", "label": "L" * 500, "ports": []}], wires=[]))
    assert len(out["title"]) <= diagram.MAX_TITLE_CHARS
    assert len(out["nodes"][0]["label"]) <= diagram.MAX_LABEL_CHARS


def test_a_diagram_with_no_title_still_has_one():
    """The title is the filename, the panel's header and what find_diagram matches on."""
    assert diagram.validate(spec(title=""))["title"] == "Diagram"


def test_too_many_nodes_is_refused_rather_than_truncated():
    """Half a wiring diagram is not a smaller wiring diagram."""
    many = [{"id": f"n{i}", "type": "box", "label": str(i), "ports": []}
            for i in range(diagram.MAX_NODES + 1)]
    with pytest.raises(diagram.DiagramError, match="keep it under"):
        diagram.validate(spec(nodes=many, wires=[]))


def test_duplicate_ports_are_dropped_not_fatal():
    """A repeated port is a typo; the wires still resolve, so the picture is still drawable."""
    out = diagram.validate(spec(nodes=[
        {"id": "a", "type": "box", "label": "A",
         "ports": [{"id": "p"}, {"id": "p"}, {"id": "q"}]}], wires=[]))
    assert [p["id"] for p in out["nodes"][0]["ports"]] == ["p", "q"]


# ------------------------------------------------------------------ the contract with the panel


@pytest.mark.parametrize("symbol", diagram.SYMBOLS)
def test_every_symbol_is_drawn_by_the_template(symbol):
    """SYMBOLS and the template's TYPES map are two halves of one contract.

    A shape named here and missing there renders as a plain box with no error anywhere, which is
    the failure this parametrize exists to make loud. Adding a symbol means adding it twice.
    """
    from pathlib import Path

    template = (Path(__file__).resolve().parents[1]
                / "src/cyclops/admin/templates/cyclops/dashboard.html").read_text()
    assert f"{symbol}:" in template.split("const TYPES = {")[1].split("}")[0], (
        f"{symbol!r} is in SYMBOLS but not in the template's TYPES map"
    )


@pytest.mark.parametrize("side", diagram.SIDES)
def test_every_side_is_a_port_group_in_the_template(side):
    from pathlib import Path

    template = (Path(__file__).resolve().parents[1]
                / "src/cyclops/admin/templates/cyclops/dashboard.html").read_text()
    assert f"'{side}': portGroup(" in template, f"{side!r} has no port group in the template"


# ------------------------------------------------------------------ the card


def test_write_then_read_is_the_same_diagram(tmp_path):
    kept = diagram.write(diagram.validate(spec()), tmp_path)
    assert kept.path is not None and kept.path.parent == tmp_path
    again = diagram.read(kept.path)
    assert again is not None
    assert (again.title, again.kind, again.spec) == (kept.title, kept.kind, kept.spec)


def test_the_filename_says_what_it_is(tmp_path):
    """Somebody opening Diagrams/ in a file browser should not need to open anything."""
    kept = diagram.write(diagram.validate(spec()), tmp_path)
    # slugify caps at four words (see slug.MAX_WORDS), so the trailing "17" is not in the name.
    # The filename is a convenience for someone browsing the folder; the title in the sidecar is
    # what find_diagram actually matches on.
    assert kept.ident.endswith("_relay-driven-from-gpio")


def test_a_diagram_with_an_unusable_title_still_gets_a_filename(tmp_path):
    kept = diagram.write(diagram.validate(spec(title="!!! ???")), tmp_path)
    assert kept.path is not None and kept.path.suffix == ".json"


@pytest.mark.parametrize("junk", ["", "{", '{"title": "x"}', "[]"])
def test_a_file_that_is_not_a_diagram_is_skipped_not_raised(tmp_path, junk):
    """A truncated sidecar from an older build must not take a whole search down."""
    bad = tmp_path / "bad.json"
    bad.write_text(junk)
    assert diagram.read(bad) is None
    assert diagram.index(tmp_path) == []


def test_index_is_newest_first(tmp_path):
    for stem in ("09-00-00_early", "17-00-00_late"):
        (tmp_path / f"{stem}.json").write_text(json.dumps(
            {"title": stem, "caption": "", "kind": "block", "created": "", "spec": {"nodes": []}}))
    assert [d.ident for d in diagram.index(tmp_path)][0] == "17-00-00_late"


# ------------------------------------------------------------------ finding one again


def keep(folder, title, kind="wiring", caption=""):
    return diagram.write({"title": title, "caption": caption, "kind": kind,
                          "layout": "dagre", "nodes": [], "wires": []}, folder)


def test_search_finds_it_through_a_microphone(tmp_path):
    """The whole reason this reuses the workbook's scorer: the query arrives misheard."""
    keep(tmp_path, "Relay driven from GPIO 17")
    keep(tmp_path, "Raspberry Pi 5 40-pin header", kind="pinout")
    found = diagram.search([tmp_path], "relay wiring diagrm")
    assert found and "Relay" in found[0].title


def test_search_says_nothing_rather_than_guessing(tmp_path):
    """An empty answer is an answer; the tool tells the model to offer to draw it instead."""
    keep(tmp_path, "Relay driven from GPIO 17")
    assert diagram.search([tmp_path], "sourdough starter") == []


def test_the_same_drawing_in_two_places_is_one_hit(tmp_path):
    """A diagram lives in its session and in the project it was filed into. It is still one.

    The filed copy is renamed with the session's date on the front, so matching on the filename
    would not catch it - which is why the title is what decides.
    """
    session_dir, project_dir = tmp_path / "s", tmp_path / "p"
    session_dir.mkdir(), project_dir.mkdir()
    kept = keep(session_dir, "Relay driven from GPIO 17")
    filed = project_dir / f"2026-08-29_{kept.path.name}"
    filed.write_bytes(kept.path.read_bytes())
    assert len(diagram.search([session_dir, project_dir], "relay")) == 1


def test_two_different_drawings_are_two_hits(tmp_path):
    """Deduplicating on the title must not collapse a pinout into a wiring diagram."""
    keep(tmp_path, "Relay driven from GPIO 17")
    keep(tmp_path, "Relay coil snubber diode", kind="wiring")
    assert len(diagram.search([tmp_path], "relay")) == 2


def test_search_survives_a_folder_that_is_not_there(tmp_path):
    """A project deleted between the catalog and the search is not an error."""
    assert diagram.search([tmp_path / "gone"], "anything") == []


# ------------------------------------------------------------------ offering it to the panel


def test_offer_writes_the_drawing_not_a_path_to_it(tmp_path, monkeypatch):
    """A diagram drawn with no session was never written down, so a path would point at nothing."""
    pending = tmp_path / "diagram.json"
    monkeypatch.setattr(diagram, "DIAGRAM_FILE", pending)
    assert diagram.offer(diagram.validate(spec()))
    payload = json.loads(pending.read_text())
    assert payload["spec"]["nodes"], "the panel is handed the drawing itself"
    assert payload["svg"] is None, "nowhere to keep a picture for a diagram nobody kept"


def test_each_offer_is_a_new_id(tmp_path, monkeypatch):
    """The page redraws when the id changes, so showing the same diagram twice must change it."""
    pending = tmp_path / "diagram.json"
    monkeypatch.setattr(diagram, "DIAGRAM_FILE", pending)
    diagram.offer(diagram.validate(spec()))
    first = json.loads(pending.read_text())["id"]
    diagram.offer(diagram.validate(spec()))
    assert json.loads(pending.read_text())["id"] != first


def test_offer_asks_for_a_picture_only_when_there_is_none(tmp_path, monkeypatch):
    monkeypatch.setattr(diagram, "DIAGRAM_FILE", tmp_path / "diagram.json")
    kept = keep(tmp_path, "Relay driven from GPIO 17")
    diagram.offer(kept.spec, kept)
    assert json.loads((tmp_path / "diagram.json").read_text())["svg"].endswith(".svg")
    # Once the panel has sent one back, re-showing it must not spend card writes redrawing it.
    kept.path.with_suffix(".svg").write_text("<svg/>")
    diagram.offer(kept.spec, kept)
    assert json.loads((tmp_path / "diagram.json").read_text())["svg"] is None


def test_show_without_a_panel_is_false_not_an_error(monkeypatch):
    """``uv run cyclops`` has a conversation and no screen. That is ordinary, not a failure."""
    monkeypatch.setattr(diagram, "_panel", None)
    assert diagram.show() is False
