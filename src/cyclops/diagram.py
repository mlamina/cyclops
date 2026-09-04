"""Technical diagrams: a second model draws, the panel shows, the project keeps.

Speech is a bad way to describe a wiring loom. "The resistor goes between GPIO 17 and the base,
and the relay's ground goes back to pin 6" is four seconds to say and a minute to picture, and
Cyclops has a screen sitting right there. So it draws.

**The realtime model does not draw.** It calls a tool with a sentence, and this module hands that
sentence to a text model that has nothing else to do. That is not a workaround for a weak model,
it is the same bridge :mod:`cyclops.search` documents: a Realtime session is a voice loop with a
4096-token response ceiling, and spending it emitting several hundred tokens of layout is time
the user spends listening to silence. Over here the drawing is a normal request that can be
retried, validated and thrown away.

**And the drawing model does not write JointJS.** It emits the small JSON in :func:`validate`
below, and the panel translates that into shapes. The reason is training data: every model has
seen mountains of Mermaid and almost no JointJS scene graphs, so asking for the library's own
format spends the reliability we picked the library for. Asking for ``{"type": "resistor"}``
instead moves the correctness question to a schema we own and can check in a millisecond, and the
symbol library - the actual zigzag - lives once in the template where no model can misdraw it.

What we give up is anything that is not a graph of boxes and wires. JointJS routes wires around
obstacles and puts named ports on things, which is exactly a wiring diagram, a pinout or a block
diagram; it has no idea what a dimensioned sketch or an exploded view is, and neither does this
schema. A request for one comes back as a diagram of the wrong kind rather than an error, which
is the cost of having one schema instead of two.

Validation is deliberately split. Structure is fatal - a wire to a node that does not exist draws
a wrong picture, and a wrong picture is worse than none - and gets one retry with the complaint
attached. Cosmetics are coerced: an unknown shape becomes a box, an unknown side becomes the
left, an unknown wire kind loses its colour. A diagram that is right except that the model said
``"colour": "purple"`` is still the diagram somebody asked for.
"""

from __future__ import annotations

import base64
import json
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from openai import APIError, AsyncOpenAI

from . import card
from .config import DIAGRAM_FILE, Settings
from .slug import fold, slugify

if TYPE_CHECKING:
    from .projects import data

# Measured over the same four requests (a relay wiring, a 40-pin pinout, a block diagram and a
# state machine), from a laptop, at effort "low": terra 4/4 valid first time, median 10.5 s, range
# 4.8-15.6 s; gpt-5.4-mini 4/4, median 9.3 s; gpt-5.6-luna 4/4, median 11.3 s. Every tier gets the
# schema right, and they are all the same speed, because what takes the time is emitting one to
# two thousand tokens of JSON rather than deciding what to draw - which is also why buying a
# cheaper model here buys nothing.
#
# So this is chosen on the drawing rather than on the clock: terra placed the cleanest layout of
# the three, and `projects/agents.py` already gives terra the two jobs needing judgement while
# nano does the reading. Placing a wiring diagram is a judgement job. Unlike `search` and `slug`,
# which run nano at effort "none" because a slow answer there is a worse answer, the worse answer
# here is a wrong picture.
#
# Ten seconds is a long time to say nothing, which is why the tool returns before this does: the
# panel says DRAWING and the conversation carries on. If that ever stops being true, fix the
# lifecycle rather than this line.
DIAGRAM_MODEL = "gpt-5.6-terra"
REASONING_EFFORT = "low"

# Generous next to search's 12 s, because nothing is waiting on a person's attention the way a
# spoken answer is: the tool has already returned, the panel says DRAWING, and the alternative to
# waiting is no diagram at all.
DIAGRAM_TIMEOUT_S = 45.0

MAX_REQUEST_CHARS = 600
MAX_TITLE_CHARS = 70
MAX_CAPTION_CHARS = 200
MAX_LABEL_CHARS = 28  # a label longer than this stops fitting in the box it names
MAX_NODES = 40
MAX_PORTS = 48  # a 40-pin header plus room; the pinout is the thing that sets this
MAX_WIRES = 80

# The symbol library, by name. Every one of these is drawn by the panel - see the `TYPES` map in
# admin/static/diagram.js - and this tuple is the contract between the two. Adding a shape means
# adding it in both places, and the test over this tuple is what says so out loud.
SYMBOLS = ("box", "resistor", "led", "header")
SIDES = ("left", "right", "left-out", "right-out", "top", "bottom")
# What a wire is carrying, which is only ever a colour on the panel. "" is an unremarkable wire.
WIRE_KINDS = ("", "signal", "power", "ground", "hot")
KINDS = ("wiring", "pinout", "block", "flow", "state")
LAYOUTS = ("manual", "dagre")

SPEC_NAME = "spec"  # the sidecar's own key for the drawing, so metadata can sit beside it

DIAGRAM_PROMPT = """\
Draw a technical diagram as JSON. It is rendered by JointJS on an 800x480 workshop panel and
read at arm's length, so it must be legible before it is complete.

Return ONLY a JSON object:

{{
  "title": "short name for this diagram, under 70 characters",
  "caption": "one sentence saying what it shows",
  "kind": "wiring|pinout|block|flow|state",
  "layout": "manual|dagre",
  "nodes": [
    {{"id": "pi", "type": "box", "label": "Raspberry Pi 5", "sublabel": "GPIO header",
      "at": {{"x": 10, "y": 100}}, "size": {{"w": 150, "h": 112}},
      "ports": [{{"id": "GPIO17", "side": "right", "label": "GPIO17", "kind": "signal"}}]}}
  ],
  "wires": [
    {{"from": "pi:GPIO17", "to": "r1:a", "kind": "signal", "label": "GPIO 17"}}
  ]
}}

Shapes: {symbols}. "box" is anything with a name - a board, a module, a load, a step.
"resistor" and "led" draw the real symbol; give them ports "a" and "b" ("k" for an LED cathode)
and no labels. "header" is a pin strip: one node, all the pins as ports.

Port sides: {sides}. Use "left"/"right" on a box - the label is drawn INSIDE it. Use
"left-out"/"right-out" on a header, where labels belong outside. Only label a port that a person
would read off the hardware: GPIO17, VCC, COM, NO. Never label "a"/"b".

Wire kinds colour the line: "signal" green, "power" amber, "ground" dim, "hot" red for mains or
anything above 24 V. Omit for an unremarkable wire. "from"/"to" are "nodeid:portid" and both must
exist in nodes.

Layout:
- "dagre" lays it out for you and you give NO "at" or "size". Use it for block, flow and state -
  anything where position carries no meaning. Prefer this; it cannot overlap.
- "manual" means you place everything, in a 800x420 space, and you must use it for wiring and
  pinout where position IS the meaning. Leave 150px between columns so wires have room to route,
  and put a node's title above it by leaving space - titles are drawn outside the box.

Keep it to what was asked: under {max_nodes} nodes. A diagram of nine things somebody can read
beats one of thirty they cannot. Use exact values - 220R, 12 V, GPIO 17 - never "a resistor".

The request: {request}"""

RETRY_NOTE = """

Your previous answer was rejected: {problem}
Return the corrected JSON object, nothing else."""


class DiagramError(RuntimeError):
    """The diagram could not be drawn. The message is meant for the voice model to relay."""


@dataclass(frozen=True)
class Diagram:
    """One drawing on the card: what it is called, what it shows, and how to draw it again."""

    ident: str  # the filename stem, which is also how a tool asks for it back
    title: str
    caption: str
    kind: str
    created: str
    spec: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None  # where the sidecar was read from; None for one not yet written

    def row(self) -> data.Row:
        """This diagram as something :func:`cyclops.projects.data.score` can rank.

        Reusing the workbook's matcher rather than writing a second one is the whole point: it is
        already tuned for words arriving through a microphone, so "relay wiring diagrm" finds
        this, and there is no index here to rebuild and nothing to keep in step.

        Imported here rather than at the top because reaching ``cyclops.projects.data`` runs the
        package's ``__init__``, which imports ``session`` and so drags OpenCV in behind it. This
        module is imported by the admin service, which has no camera and must not pay for one -
        the same reason ``stats.py`` keeps its distance from ``session``.
        """
        from .projects import data

        return data.Row(tab=self.kind, key=self.title, value=self.caption, note=self.ident)

    def as_dict(self) -> dict[str, Any]:
        """The shape handed to the voice model. The spec is never in it - it is not readable."""
        return {"id": self.ident, "title": self.title, "caption": self.caption, "kind": self.kind}


def _text(value: object, limit: int) -> str:
    """One field of model output, made safe to draw: collapsed, trimmed, and never a surprise."""
    out = " ".join(str(value or "").split())
    return out[:limit].rstrip()


def _pick(value: object, allowed: tuple[str, ...], fallback: str) -> str:
    """A closed vocabulary, coerced rather than enforced - see the module docstring's split."""
    found = str(value or "").strip().lower()
    return found if found in allowed else fallback


def _number(value: object, limit: float) -> float | None:
    try:
        found = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if found != found or abs(found) > limit:  # NaN, or a coordinate off the panel entirely
        return None
    return found


def validate(payload: object) -> dict[str, Any]:
    """The model's JSON, checked and cleaned, or :class:`DiagramError` saying what to fix.

    The message is written for the model that will read it on the retry, so it names the offending
    id rather than describing the rule.
    """
    if not isinstance(payload, dict):
        raise DiagramError("the answer was not a JSON object")

    nodes_in = payload.get("nodes")
    if not isinstance(nodes_in, list) or not nodes_in:
        raise DiagramError("there were no nodes")
    if len(nodes_in) > MAX_NODES:
        raise DiagramError(f"there were {len(nodes_in)} nodes; keep it under {MAX_NODES}")

    layout = _pick(payload.get("layout"), LAYOUTS, "dagre")
    nodes: list[dict[str, Any]] = []
    ports: dict[str, set[str]] = {}
    for raw in nodes_in:
        if not isinstance(raw, dict):
            raise DiagramError("a node was not an object")
        ident = _text(raw.get("id"), 40)
        if not ident:
            raise DiagramError("a node had no id")
        if ident in ports:
            raise DiagramError(f"two nodes share the id {ident!r}")

        node: dict[str, Any] = {
            "id": ident,
            "type": _pick(raw.get("type"), SYMBOLS, "box"),
            "label": _text(raw.get("label"), MAX_LABEL_CHARS),
            "sublabel": _text(raw.get("sublabel"), MAX_LABEL_CHARS),
        }
        # Geometry only means anything under "manual"; under dagre it would fight the layout, so
        # it is dropped rather than passed through to be silently overwritten.
        if layout == "manual":
            at, size = raw.get("at"), raw.get("size")
            if isinstance(at, dict):
                x, y = _number(at.get("x"), 4000), _number(at.get("y"), 4000)
                if x is not None and y is not None:
                    node["at"] = {"x": x, "y": y}
            if isinstance(size, dict):
                w, h = _number(size.get("w"), 2000), _number(size.get("h"), 2000)
                if w and h and w > 0 and h > 0:
                    node["size"] = {"w": w, "h": h}

        seen: set[str] = set()
        cleaned: list[dict[str, str]] = []
        for port in raw.get("ports") or []:
            if not isinstance(port, dict):
                continue
            pid = _text(port.get("id"), MAX_LABEL_CHARS)
            if not pid or pid in seen:
                continue  # a duplicate port is a typo, not a picture worth refusing to draw
            seen.add(pid)
            cleaned.append({
                "id": pid,
                "side": _pick(port.get("side"), SIDES, "left"),
                "label": _text(port.get("label"), MAX_LABEL_CHARS),
                "kind": _pick(port.get("kind"), WIRE_KINDS, ""),
            })
            if len(cleaned) >= MAX_PORTS:
                break
        node["ports"] = cleaned
        ports[ident] = seen
        nodes.append(node)

    wires: list[dict[str, str]] = []
    for raw in payload.get("wires") or []:
        if not isinstance(raw, dict):
            continue
        ends = []
        for side in ("from", "to"):
            spec = _text(raw.get(side), 90)
            node_id, _, port_id = spec.partition(":")
            if node_id not in ports:
                raise DiagramError(f"a wire runs to {node_id!r}, which is not one of the nodes")
            if port_id not in ports[node_id]:
                raise DiagramError(
                    f"a wire uses port {port_id!r} on {node_id!r}, which has no such port"
                )
            ends.append(f"{node_id}:{port_id}")
        wires.append({
            "from": ends[0],
            "to": ends[1],
            "kind": _pick(raw.get("kind"), WIRE_KINDS, ""),
            "label": _text(raw.get("label"), MAX_LABEL_CHARS),
        })
        if len(wires) >= MAX_WIRES:
            break

    title = _text(payload.get("title"), MAX_TITLE_CHARS) or "Diagram"
    return {
        "title": title,
        "caption": _text(payload.get("caption"), MAX_CAPTION_CHARS),
        "kind": _pick(payload.get("kind"), KINDS, "block"),
        "layout": layout,
        "nodes": nodes,
        "wires": wires,
    }


async def draw(request: str, settings: Settings) -> dict[str, Any]:
    """Turn a spoken request into a validated spec. Raises :class:`DiagramError`, never a traceback.

    One retry, and only one: the second attempt is handed the complaint from the first, which
    fixes a wire to a misspelt node almost every time. A third would just be spending the user's
    patience on a model that has already shown it cannot draw this.
    """
    request = _text(request, MAX_REQUEST_CHARS)
    if not request:
        raise DiagramError("no diagram was described")

    client = AsyncOpenAI(api_key=settings.api_key, timeout=DIAGRAM_TIMEOUT_S, max_retries=0)
    prompt = DIAGRAM_PROMPT.format(
        request=request,
        symbols=", ".join(SYMBOLS),
        sides=", ".join(SIDES),
        max_nodes=MAX_NODES,
    )
    try:
        problem = ""
        for attempt in range(2):
            try:
                response = await client.responses.create(
                    model=DIAGRAM_MODEL,
                    reasoning={"effort": REASONING_EFFORT},
                    text={"format": {"type": "json_object"}},
                    input=prompt + (RETRY_NOTE.format(problem=problem) if problem else ""),
                )
            except APIError as exc:
                raise DiagramError(f"the drawing failed: {exc.message or exc}") from exc
            except TimeoutError as exc:
                raise DiagramError(
                    f"the drawing took longer than {DIAGRAM_TIMEOUT_S:.0f}s"
                ) from exc

            answer = (getattr(response, "output_text", "") or "").strip()
            try:
                return validate(json.loads(answer))
            except json.JSONDecodeError:
                problem = "it was not valid JSON"
            except DiagramError as exc:
                problem = str(exc)
            if attempt == 0:
                print(f"· diagram: retrying ({problem})", flush=True)
        raise DiagramError(f"the diagram came back wrong twice ({problem})")
    finally:
        await client.close()


# ------------------------------------------------------------------ on the card


def write(spec: dict[str, Any], folder: Path, *, when: datetime | None = None) -> Diagram:
    """Put one diagram in a folder and hand back what it became.

    The sidecar is the whole diagram - metadata and spec in one file - so a diagram is never half
    on the card, and :func:`read` needs no second file to agree with. The picture lands beside it
    later, under the same stem, if the panel sends one back.
    """
    when = when or datetime.now()
    stem = f"{when.strftime('%H-%M-%S')}_{slugify(spec.get('title', '')) or 'diagram'}"
    payload = {
        "title": spec.get("title", ""),
        "caption": spec.get("caption", ""),
        "kind": spec.get("kind", ""),
        "created": when.strftime("%Y-%m-%d %H:%M:%S"),
        SPEC_NAME: spec,
    }
    path = folder / f"{stem}.json"
    card.write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return Diagram(
        ident=stem,
        title=payload["title"],
        caption=payload["caption"],
        kind=payload["kind"],
        created=payload["created"],
        spec=spec,
        path=path,
    )


_panel: object | None = None  # a cyclops.kiosk.Kiosk while one is running; see set_panel


def set_panel(panel: object | None) -> None:
    """Register the running kiosk, so a finished diagram can find a screen to appear on.

    Same shape and same reason as :func:`cyclops.webcam.set_live_source` and
    :data:`cyclops.session._live`: what wants the panel is the agent's tool coroutine, which sits
    behind a :class:`~cyclops.ui.SessionController` built with settings and a camera and no way
    back to the kiosk. Threading a reference through would mean a new argument on
    ``SessionController`` and another on ``VoiceAgent``, for something there is only ever one of.

    It lives here rather than in :mod:`cyclops.kiosk` so that the caller does not have to import
    the kiosk to reach it - that module owns OpenCV and a Qt window, and ``uv run cyclops`` has
    neither and wants neither.
    """
    global _panel
    _panel = panel


def show() -> bool:
    """Ask the panel to show whatever :func:`offer` last left. False if there is no panel.

    False is the ordinary answer under ``uv run cyclops``, where there is a conversation and no
    screen. The caller says so out loud rather than treating it as a failure.
    """
    panel = _panel
    return bool(panel is not None and panel.show_diagram())  # type: ignore[attr-defined]


def offer(spec: dict[str, Any], kept: Diagram | None = None) -> bool:
    """Leave a drawing where the panel's page will find it. False if it could not be left.

    The drawing travels in the file rather than a path to one, because the page is served by the
    admin process and a diagram drawn with no session running was never written to the card at
    all - there would be nothing for a path to point at.
    """
    # Where the page should send the picture back to, if anywhere. Only for a diagram that has
    # just been written and has no picture yet: re-showing an old one would spend a megabyte of
    # card writes redrawing a file that is already correct. Resolved to a string separately
    # rather than inline, because str(None) is "None" and would have the page writing a picture
    # to a file of that name.
    target = _svg_for(kept) if kept is not None else None
    return _leave({
        "title": spec.get("title", ""),
        SPEC_NAME: spec,
        "svg": str(target) if target is not None else None,
    })


def offer_image(jpeg: bytes, title: str) -> bool:
    """Leave a picture where the panel's page will find it, in place of a drawing.

    The same file and the same handshake as :func:`offer`, because a diagram was only ever the
    first thing the panel could be asked to show - see :mod:`cyclops.imagine` for the second.
    It rides in the payload as base64 for the reason :func:`offer` already gives about the spec:
    the page is served by the admin process, which shares no memory with whoever made this, and
    a picture made with no session running was never written to the card at all.

    Downscale it first. The caller does that, because the caller knows what the full-size copy is
    for - see :func:`cyclops.imagine.for_panel`.
    """
    url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
    return _leave({"title": title, SPEC_NAME: None, "image": url, "svg": None})


def withdraw() -> None:
    """Take back whatever was last offered, so the page falls back to the dashboard.

    The panel's browser is warm and polling, so an offer left lying about is not inert: the page
    paints it and keeps it up, behind our window, until something says otherwise. Anything that
    is about to hand the panel to that browser for some *other* reason therefore has to clear
    this first, or it uncovers onto a picture from an hour ago - which is exactly what tapping
    the eye did on 2026-09-02, after a throwaway script offered a picture to a panel that was
    never going to show it and exited without tidying up.

    Never raises. A payload that cannot be removed is not a reason to refuse a tap.
    """
    try:
        DIAGRAM_FILE.unlink(missing_ok=True)
    except OSError as exc:
        print(
            f"· could not take the panel's last picture back ({exc})",
            file=sys.stderr,
            flush=True,
        )


def _leave(payload: dict[str, Any]) -> bool:
    """One thing for the panel to show, written where the page is looking. False if it could not.

    The id is minted here rather than by the caller: all it has to do is differ from whatever the
    page last drew, and a stable one would make showing the same thing twice in a row silently do
    nothing the second time.
    """
    try:
        card.write_text(DIAGRAM_FILE, json.dumps({"id": uuid.uuid4().hex[:12], **payload}))
    except OSError as exc:
        print(f"· could not offer the panel something to show ({exc})", file=sys.stderr, flush=True)
        return False
    return True


def _svg_for(kept: Diagram) -> Path | None:
    """Where this diagram's picture belongs, or None if it is already there."""
    if kept.path is None:
        return None
    beside = kept.path.with_suffix(".svg")
    return None if card.written(beside) else beside


def read(path: Path) -> Diagram | None:
    """One sidecar off the card, or None if it is not one. Never raises - a bad file is skipped."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get(SPEC_NAME), dict):
        return None
    return Diagram(
        ident=path.stem,
        title=str(payload.get("title", "")),
        caption=str(payload.get("caption", "")),
        kind=str(payload.get("kind", "")),
        created=str(payload.get("created", "")),
        spec=payload[SPEC_NAME],
        path=path,
    )


def index(folder: Path) -> list[Diagram]:
    """Every diagram in one folder, newest first. An unreadable one is missing, not fatal."""
    if not folder.is_dir():
        return []
    found = [read(p) for p in sorted(folder.glob("*.json"), reverse=True)]
    return [d for d in found if d is not None]


def search(folders: list[Path], query: str, limit: int = 5) -> list[Diagram]:
    """The diagrams that answer ``query``, best first. Empty when none do, which is an answer.

    The scoring is :func:`cyclops.projects.data.score` unchanged - see :meth:`Diagram.row`.
    """
    from .projects import data

    # One drawing, once. A filed copy carries the session's date on the front of its name, so the
    # ident alone does not catch it; the title does. Two genuinely different diagrams sharing a
    # title are the same diagram redrawn, and the newest is the one wanted - which is what comes
    # first, because index() is newest-first and the live session's folder is passed first.
    seen: set[str] = set()
    scored: list[tuple[float, Diagram]] = []
    for folder in folders:
        for found in index(folder):
            mark = fold(found.title) or found.ident
            if mark in seen:
                continue
            seen.add(mark)
            scored.append((data.score(query, found.row()), found))
    hits = [(value, d) for value, d in scored if value >= data.MIN_SCORE]
    hits.sort(key=lambda pair: pair[0], reverse=True)
    return [d for _, d in hits[:limit]]
