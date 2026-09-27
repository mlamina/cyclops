"""What Cyclops is drawing on the glass right now, while it is still being written.

The scratchpad this replaces was a one-shot: the model wrote a screenful of HTML, it was
fsynced to a file, the page noticed on its next poll and painted it whole. One shot at the
markup, about a second of nothing, and a revision meant starting again.

This is the same idea with the waiting taken out, and the mechanism is borrowed from MCP Apps
(SEP-1865). What makes that protocol quick is not the components: it is that the renderer is
already mounted and running before the model has finished typing its arguments, so the host can
push *partial* arguments into it and the interface assembles itself mid-sentence. Cyclops
already owns the expensive half of that - a Chromium warm since boot - and was throwing away
the cheap half, because nothing here had ever read ``response.function_call_arguments.delta``.

So: the model writes Prefab Python, we compile whatever has arrived every :data:`THROTTLE_S`,
and every compile that succeeds goes out as a frame. Half-written code raises and is dropped,
and the last good frame stays on the glass - which is the whole error strategy. A ``with``
block needs no closing anything, so the prefix of a Prefab program is usually a valid Prefab
program describing less of the same thing.

**The compile happens here rather than in the browser.** Prefab's own way to do this ships
Pyodide into the renderer and executes the partial code there. That is right for a laptop and
wrong for this: it is a ten megabyte WebAssembly runtime and a second Python interpreter inside
Chromium, on a board that already throttles at 85 C. Prefab's architecture notes call the
alternative a first-class transport - "the server does the execution and the client is a pure
JSON renderer" - and the Pi is already running CPython, where a small component tree compiles
in well under a millisecond.

What leaves here is Prefab's wire format, ``{"$prefab": ..., "view": ..., "state": ...}``, and
nothing downstream understands it: :mod:`cyclops.companion` posts it through an SSE stream and
``sketch.js`` hands it to the renderer untouched. The one thing this module does add is
``mode``, and it is light. The panel wears phosphor green and everything else on it is dark, but
this is a sheet of paper: what gets drawn on it brings its own colour, and the chart library's
defaults are built for a white page. On black, a chart's axes and gridlines are dark grey on
near-black and the reading loses its scale.

The listener set is the same shape as the rest of the panel plumbing: whoever compiles is a
tool coroutine on the agent's thread, whoever writes the bytes is a request thread in the
companion server, and one lock around a set is all the two of them need.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import copy_context
from typing import Any

MAX_SKETCH_CHARS = 2000  # a panel at arm's length, not a report. Prefab says a lot per character.
THROTTLE_S = 0.1  # how often a growing program is recompiled while the model is still typing
QUEUE_DEPTH = 4  # frames a slow listener may fall behind before it starts losing the middle ones

# The only styling we impose, and it is about the room rather than the taste. Prefab is built for
# a browser window somebody is sitting in front of; this is 800x480 of glass read at arm's length
# across a bench, and its defaults come out about two thirds of the size that is legible there.
#
# One lever: the root font size. Every size in the component library is in rem, so moving this
# moves headings, labels, metrics, table rows and chart axes together and in proportion - which is
# the whole reason not to restyle the components one at a time.
#
# The rest is filling the panel. Prefab's root is a block that sizes to its content and sits at the
# top; a single number on this screen belongs in the middle of it.
PANEL_CSS = """
html { font-size: 22px; }
.pf-app-root {
  min-height: 100vh;
  display: flex; flex-direction: column; justify-content: center;
  padding: 3vh 4vw;
  overflow: hidden;
}
::-webkit-scrollbar { width: 0; height: 0; }
"""

_wire: dict[str, Any] | None = None
_listeners: set[queue.Queue[dict[str, Any] | None]] = set()
_lock = threading.Lock()
_namespace: dict[str, Any] | None = None


def _globals() -> dict[str, Any]:
    """Every name the model's code is allowed to see, built once.

    Imported here rather than at module import because ``uv run cyclops`` has no panel and
    should not pay for a UI framework to find that out.

    This is not a sandbox and is not trying to be one. ``exec`` with a restricted ``__builtins__``
    is a speed bump to a determined escape, not a wall, and the honest description of the trade
    is in the plan: sub-millisecond compiles with no daemon and no cold start, against code that
    runs as ``cyclops`` if something ever talks the model into writing it. Prefab ships a real
    sandbox (Deno + Pyodide) for the day that stops being worth it.
    """
    global _namespace
    if _namespace is None:
        from prefab_ui import actions, components, rx
        from prefab_ui.app import PrefabApp
        from prefab_ui.components import charts

        names: dict[str, Any] = {}
        for module in (components, charts, actions):
            for name in getattr(module, "__all__", None) or dir(module):
                if not name.startswith("_"):
                    names[name] = getattr(module, name)
        names["PrefabApp"] = PrefabApp
        names["Rx"] = rx.Rx
        # Drawing is the image tool's job, not the scratchpad's: a Mermaid diagram or a hand-typed
        # SVG never once came out more useful than a picture. Without the names they fail to
        # compile, and a sketch that only draws puts nothing on the glass.
        for name in ("Mermaid", "Svg"):
            names.pop(name, None)
        _namespace = names
    return dict(_namespace)


def warm() -> None:
    """Pay for the import now, so the first frame does not.

    A hundred and thirty milliseconds on this laptop and more on the Pi, and it would otherwise
    land on the first delta of the first sketch of a session - which is the single moment in the
    whole path where somebody is watching an empty screen. Called when a session opens, off the
    loop's thread, where there are seconds of slack before anyone can ask for anything.
    """
    _globals()


def compile(code: str) -> dict[str, Any] | None:  # noqa: A001 - it compiles; that is the name
    """Run the model's code and return a frame, or None if it did not get that far.

    Never raises. Half-written code is the ordinary case here, not the exceptional one: this is
    called on a prefix that grows by a few tokens at a time, and most of those prefixes are a
    syntax error. The caller's move on None is to leave the last frame where it is.

    Root-finding mirrors Prefab's own sandbox harness (``prefab_ui/sandbox/runner.js``): prefer
    the last ``PrefabApp`` in the namespace, else the last component that is nobody's child. It
    is copied rather than shared because that one lives inside a JavaScript string in a Deno
    subprocess we are deliberately not starting.
    """
    if not code.strip():
        return None

    def run() -> dict[str, Any] | None:
        from prefab_ui.app import PrefabApp
        from prefab_ui.components.base import Component, ContainerComponent, _component_stack

        _component_stack.set(None)
        namespace = _globals()
        try:
            exec(code[:MAX_SKETCH_CHARS], namespace)  # noqa: S102 - see _globals
        except BaseException:  # noqa: BLE001 - a prefix fails every way there is
            return None

        made = [v for k, v in namespace.items() if not k.startswith("_")]
        apps = [v for v in made if isinstance(v, PrefabApp)]
        comps = [v for v in made if isinstance(v, Component)]
        children = {
            id(child)
            for comp in comps
            if isinstance(comp, ContainerComponent)
            for child in comp.children
        }
        roots = [c for c in comps if id(c) not in children]
        target = apps[-1] if apps else (roots[-1] if roots else None)
        if target is None:
            return None
        try:
            wire = target.to_json() if isinstance(target, PrefabApp) else {
                "$prefab": {"version": "0.3"},
                "view": target.to_json(),
            }
        except BaseException:  # noqa: BLE001 - a tree half-built serialises about as well
            return None
        wire.setdefault("mode", "light")
        wire["css"] = [*(wire.get("css") or []), PANEL_CSS]
        return wire

    # A fresh context per compile, because Prefab nests components through a ContextVar and code
    # that dies inside a `with` never runs the __exit__ that would pop it. Without this, one
    # broken prefix leaves the stack dirty and every frame after it is drawn inside a ghost.
    return copy_context().run(run)


def offer(code: str) -> bool:
    """Compile a prefix and put it on the glass if it changed anything. True if a frame went out."""
    wire = compile(code)
    if wire is None or wire == current():
        return False
    push(wire)
    return True


def push(wire: dict[str, Any] | None) -> None:
    """Send a frame to every listener, and remember it for whoever connects next."""
    global _wire
    with _lock:
        _wire = wire
        listeners = list(_listeners)
    for box in listeners:
        try:
            box.put_nowait(wire)
        except queue.Full:
            # A listener this far behind is a browser that stopped painting. Drop its oldest
            # frame rather than the newest: what is on the glass should be the latest thing
            # written, and an intermediate state of a program nobody saw is worth nothing.
            try:
                box.get_nowait()
                box.put_nowait(wire)
            except (queue.Empty, queue.Full):
                pass


def current() -> dict[str, Any] | None:
    """The frame on the glass now, or None when nothing is."""
    return _wire


def clear() -> None:
    """Take the sketch down. Listeners are told, so a page still watching goes blank with it."""
    push(None)


@contextmanager
def listening() -> Iterator[queue.Queue[dict[str, Any] | None]]:
    """A queue of frames for one watcher, unsubscribed on the way out.

    It arrives holding whatever is already up, so a page that connects mid-sketch is not blank
    until the model's next keystroke.
    """
    box: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=QUEUE_DEPTH)
    with _lock:
        if _wire is not None:
            box.put_nowait(_wire)
        _listeners.add(box)
    try:
        yield box
    finally:
        with _lock:
            _listeners.discard(box)
