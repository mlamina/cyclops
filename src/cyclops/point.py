"""What Cyclops is pointing at: a few marks on a picture, and how long they last.

One gesture at a time, offered here and read once a frame by the kiosk - the same shape as
:data:`cyclops.panel._kiosk` and :func:`cyclops.tasks.line`, and for the same reason: what
makes a mark is a tool coroutine on the agent's thread or a watcher on its own, and what draws
one is the render loop. A single rebinding of one module slot needs no lock between them.

Two things put marks up and they want different pictures under them. ``point_at`` marks the
*photo the model was shown*, because a camera held in a hand has moved since the shutter and
the coordinates belong to what it saw - so the kiosk holds that still for as long as the
gesture lasts. :mod:`cyclops.notice` marks the live picture, because the thing it noticed is
being held up right now. That is the whole of what ``Gesture.picture`` says: a path, or None
for "whatever the camera is showing".

The mark language is one mark per line, and it is deliberately not JSON. A model writes it
while it is talking, and every closing brace is silence between the question and the answer -
which is the lesson Prefab learned the expensive way round, streaming JSON for a demo and
finding the Python it had been generated from was 70% smaller. Lines have no closing anything,
so a half-written one is simply the last line and :func:`partial_marks` drops it.

Nothing here draws, and nothing here knows what a panel looks like. Marks arrive in fractions
of whatever picture they were made against and leave as pixels through :func:`to_panel`, which
is the one piece of arithmetic in this module that has to be right: it mirrors
:func:`cyclops.overlay.fit_to_window`, and if the two ever disagree the ring lands next to the
thing instead of on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MAX_MARKS = 8  # more than this on a 800x480 panel is a diagram, and a diagram is drawn
MAX_LABEL = 40  # a label is a word or two at arm's length, not a sentence
HOLD_S = 3.0  # how long a gesture stays at full before it starts to go
FADE_S = 0.6  # ...and how long it takes going. A finger pulled back, not a slide dismissed.
TRAVEL_S = 0.35  # how long the reticle takes to reach the first mark
KINDS = ("ring", "n", "tag", "arrow")

# What a JSON string escape means, for reading a half-written tool argument off the wire.
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/"}


@dataclass(frozen=True)
class Mark:
    """One mark, in fractions of the picture it was made against.

    ``x``/``y`` are 0..1 from the left and top edges; ``x2``/``y2`` are where an arrow ends and
    are ignored by every other kind. ``n`` is the digit in a numbered marker. Fractions rather
    than pixels because the model is shown a 1024-wide photo, the panel is 800 wide, and the
    endoscope hands over 640 - the only coordinates that survive all three are proportions.
    """

    kind: str
    x: float
    y: float
    label: str = ""
    n: int = 0
    x2: float = 0.0
    y2: float = 0.0


@dataclass(frozen=True)
class PanelMark:
    """The same mark in panel pixels. What :class:`cyclops.overlay.Overlay` is handed."""

    kind: str
    x: int
    y: int
    label: str = ""
    n: int = 0
    x2: int = 0
    y2: int = 0


@dataclass(frozen=True)
class Gesture:
    """Marks, what they were made against, and when. ``picture`` None means the live frame."""

    marks: tuple[Mark, ...]
    picture: Path | None
    born: float
    hold_s: float = HOLD_S


_current: Gesture | None = None  # the one gesture up, or none


def show(gesture: Gesture) -> None:
    """Put a gesture up. One rebinding, so the render loop never sees half of one."""
    global _current
    _current = gesture


def current() -> Gesture | None:
    """Whatever is up. Read once a frame by the kiosk; may be None."""
    return _current


def clear() -> None:
    """Take it down - because it expired, or because they touched the screen."""
    global _current
    _current = None


def _clamp(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def _num(word: str) -> float | None:
    try:
        return float(word)
    except ValueError:
        return None


def _mark(words: list[str]) -> Mark | None:
    """One line's words as a mark, or None if it is not one. Never raises."""
    kind = words[0].casefold()
    rest = words[1:]
    if kind == "n":
        if len(rest) < 3:
            return None
        digit, x, y = _num(rest[0]), _num(rest[1]), _num(rest[2])
        if digit is None or x is None or y is None:
            return None
        return Mark("n", _clamp(x), _clamp(y), " ".join(rest[3:])[:MAX_LABEL], int(digit))
    if kind == "arrow":
        if len(rest) < 4:
            return None
        got = [_num(word) for word in rest[:4]]
        if any(value is None for value in got):
            return None
        x, y, x2, y2 = (_clamp(value) for value in got)  # type: ignore[arg-type]
        return Mark("arrow", x, y, " ".join(rest[4:])[:MAX_LABEL], 0, x2, y2)
    if kind in ("ring", "tag"):
        if len(rest) < 2:
            return None
        x, y = _num(rest[0]), _num(rest[1])
        if x is None or y is None:
            return None
        label = " ".join(rest[2:])[:MAX_LABEL]
        if kind == "tag" and not label:
            return None  # a tag with nothing written on it is a dot nobody asked for
        return Mark(kind, _clamp(x), _clamp(y), label)
    return None


def parse(text: str) -> list[Mark]:
    """Read the mark language. Lenient by design: a bad line is skipped, never an error.

    The model is writing this mid-sentence and cannot be sent back to correct it, so anything
    legible is drawn and anything else is dropped quietly. Out-of-range coordinates are clamped
    rather than rejected for the same reason - "just off the right edge" means the right edge.
    """
    marks = []
    for line in (text or "").splitlines():
        words = line.split()
        if not words:
            continue
        mark = _mark(words)
        if mark is not None:
            marks.append(mark)
        if len(marks) >= MAX_MARKS:
            break
    return marks


def fade(gesture: Gesture, now: float) -> float:
    """How much of the gesture is left to draw: 1.0 while it is held, then down over FADE_S."""
    left = (gesture.born + gesture.hold_s + FADE_S) - now
    if left <= 0.0:
        return 0.0
    return 1.0 if left >= FADE_S else left / FADE_S


def expired(gesture: Gesture, now: float) -> bool:
    """Nothing of it is left. The kiosk clears it and the live picture comes back."""
    return now >= gesture.born + gesture.hold_s + FADE_S


def to_panel(
    x: float, y: float, src_w: int, src_h: int, width: int, height: int
) -> tuple[int, int]:
    """A fraction of the source picture as a pixel on the panel.

    This is :func:`cyclops.overlay.fit_to_window` read backwards, and it is copied literally -
    the same ``int()``, the same ``// 2`` - rather than rewritten more cleanly. The camera is
    16:9 and the panel is 5:3, so a frame loses 64 columns on the way to the glass; a mark
    mapped through arithmetic that rounds differently lands about thirty pixels off the thing
    it is pointing at, which on this panel is a different component.
    """
    want = width / height
    have = src_w / src_h
    new_w, new_h, x0, y0 = src_w, src_h, 0, 0
    if have > want:  # too wide - the sides were trimmed
        new_w = int(src_h * want)
        x0 = (src_w - new_w) // 2
    elif have < want:  # too tall - the top and bottom were
        new_h = int(src_w / want)
        y0 = (src_h - new_h) // 2
    return (
        round((x * src_w - x0) * width / new_w),
        round((y * src_h - y0) * height / new_h),
    )


def _smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def reticle_at(
    gesture: Gesture, now: float, home: tuple[int, int], target: tuple[int, int]
) -> tuple[int, int]:
    """Where the reticle is this frame: easing off the lens axis to the mark, and back.

    Out on a smoothstep over TRAVEL_S, home on the fade - so it leaves deliberately and is
    pulled back with the rest of the gesture rather than snapping to centre when it ends.
    """
    travel = _smoothstep(_clamp((now - gesture.born) / TRAVEL_S)) * fade(gesture, now)
    return (
        round(home[0] + (target[0] - home[0]) * travel),
        round(home[1] + (target[1] - home[1]) * travel),
    )


def frame_args(
    gesture: Gesture, now: float, src_w: int, src_h: int, width: int, height: int
) -> dict:
    """Everything :meth:`cyclops.overlay.Overlay.render` needs for this gesture, this frame.

    One bridge rather than four, so the kiosk's loop grows one splat and ``tools/panel_shot.py``
    renders through the identical path - which is what makes a picture on this Mac evidence
    about the Pi rather than a second implementation that happens to look similar.
    """
    marks = [
        PanelMark(
            mark.kind,
            *to_panel(mark.x, mark.y, src_w, src_h, width, height),
            mark.label,
            mark.n,
            *to_panel(mark.x2, mark.y2, src_w, src_h, width, height),
        )
        for mark in gesture.marks
    ]
    home = (width // 2, height // 2)
    target = (marks[0].x, marks[0].y) if marks else home
    return {
        "marks": marks,
        "marks_fade": fade(gesture, now),
        "reticle": reticle_at(gesture, now, home, target),
        "look_at": target if marks else None,
    }


def partial_marks(prefix: str) -> list[Mark]:
    """The complete marks in a half-written tool argument, for drawing one while it is typed.

    ``prefix`` is however much of a ``point_at`` call's JSON arguments has arrived. This reads
    the value of ``"marks"`` out of it by hand: a JSON parser cannot help with a string that
    stops mid-word, and the only structure that matters here is where that string starts and
    whether it has ended. If it has not, the last line is still being written and is dropped -
    which is the property the line language was chosen for.
    """
    at = prefix.find('"marks"')
    if at < 0:
        return []
    start = prefix.find('"', at + len('"marks"'))
    if start < 0:
        return []
    body: list[str] = []
    closed = False
    i = start + 1
    while i < len(prefix):
        char = prefix[i]
        if char == "\\" and i + 1 < len(prefix):
            body.append(_ESCAPES.get(prefix[i + 1], prefix[i + 1]))
            i += 2
            continue
        if char == '"':
            closed = True
            break
        body.append(char)
        i += 1
    text = "".join(body)
    if not closed:
        text = text.rpartition("\n")[0]
    return parse(text)
