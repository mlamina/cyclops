"""What is plugged into the box, read straight off sysfs.

One pass over ``/sys/bus/usb/devices`` per poll - no ``lsusb`` to shell out to, no pyusb, no udev
hook to install and keep working. The kernel has already enumerated everything and written it
down; this reads the files it wrote.

Three things about the bus on this box set the shape of it. The endoscope reports no manufacturer
and no product string at all and its interfaces are vendor-specific, so neither its name nor its
category can be read off the bus - both come out of its extension, via :func:`known`. The lav mic
is always in, so it is never news, and its product string is "USB Composite Device", which is
not a name anybody would want on the glass anyway. And the four root hubs are on the bus like
anything else, so device class 09 is skipped or the panel lists the Pi's own silicon back at it.

Off Linux there is no sysfs and the list is empty. That is what keeps the Mac, the test suite and
``tools/panel_shot.py`` honest: nothing here renders or prompts differently on a laptop by
accident, and what the panel shows has to be asked for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

SYSFS = Path("/sys/bus/usb/devices")

# The four the panel and the prompt both talk in. Deliberately coarse: what a thing is for is the
# question ("are we looking at something, or playing something?"), not what its descriptors say.
CAMERA = "camera"
MUSIC = "music"
STORAGE = "storage"
OTHER = "other"

# Interface class -> category, in the order they are tried. Video before audio, so a webcam with
# a microphone in it is a camera and not a music device - which is the C920, every time.
BY_INTERFACE: tuple[tuple[str, str], ...] = (
    ("0e", CAMERA),   # USB video class
    ("01", MUSIC),    # USB audio class, which is also where MIDI interfaces live
    ("08", STORAGE),  # mass storage
)

HUB_CLASS = "09"  # a device class, not an interface one: root hubs and any hub hanging off them


@cache
def known() -> dict[str, tuple[str, str]]:
    """Devices the bus describes badly, by vid:pid: ``(category, name)``.

    Overrides both, because for these the bus gives neither - vendor-specific interfaces and
    empty strings, which is the endoscope. Built out of the extensions, since everything about a
    device lives in its own file (:mod:`cyclops.extensions`), and cached, since this is asked once
    per device per poll. Imported in here rather than at the top, because extensions must never
    import this module and the two would otherwise meet at load time.
    """
    from . import extensions

    return extensions.known(extensions.load())


# Never listed, whatever else is true of them. The lav mic is in every single session, so it is
# not news in the greeting and it is not worth a rail tag either - his answer, and the one device
# this job was told to ignore by name.
IGNORED = frozenset({"4c4a:4155"})

NAME_CHARS = 22  # what the rail can hold beside an icon; the prompt gets the same string

_PORT = re.compile(r"\d+")


@dataclass(frozen=True)
class Device:
    """One thing on the bus: what to call it, and which of the four it is."""

    vid: str
    pid: str
    name: str
    category: str

    @property
    def ident(self) -> str:
        return f"{self.vid}:{self.pid}"

    def line(self) -> str:
        """The prompt's one line for it."""
        return f"- {self.name} ({self.category})"


def _read(folder: Path, name: str) -> str:
    try:
        return folder.joinpath(name).read_text(errors="replace").strip()
    except OSError:
        return ""


def _tidy(text: str) -> str:
    """One line, no runs of space, no longer than the rail can set."""
    clean = " ".join(text.split())
    return clean[:NAME_CHARS].rstrip()


def _order(name: str) -> tuple[int, ...]:
    """A sysfs bus path as numbers, so 10-1 sorts after 2-1 rather than before it.

    Sorting on the *port path* and not on the device number is what stops the row reshuffling:
    a devnum is handed out afresh on every replug, where a port is where the plug physically is.
    """
    return tuple(int(part) for part in _PORT.findall(name))


def _interfaces(folder: Path) -> list[str]:
    """Every ``bInterfaceClass`` this device has, lowercased.

    An interface is a *sibling* of its device in this directory and not a child of it -
    ``1-2:1.0`` sits beside ``1-2`` - which is the one thing about the layout worth knowing.
    """
    classes = []
    for child in sorted(folder.parent.glob(f"{folder.name}:*")):
        if value := _read(child, "bInterfaceClass"):
            classes.append(value.lower())
    return classes


def _category(ident: str, folder: Path) -> str:
    table = known().get(ident)
    if table is not None:
        return table[0]
    classes = set(_interfaces(folder))
    for code, category in BY_INTERFACE:
        if code in classes:
            return category
    return OTHER


def _name(ident: str, folder: Path) -> str:
    """What to call it: the bus first, then the table, then the bare IDs.

    ``product`` is what the manufacturer wrote on it and is nearly always the right answer.
    The table (:func:`known`) comes second rather than first so a device that grows a sensible
    product string after a firmware update starts using it. ``manufacturer`` alone is a poor
    name ("Jieli Technology") but it beats four hex digits, which is what is left.
    """
    for candidate in (_read(folder, "product"), known().get(ident, ("", ""))[1],
                      _read(folder, "manufacturer")):
        if tidy := _tidy(candidate):
            return tidy
    return ident


def connected(root: Path | None = None) -> tuple[Device, ...]:
    """Everything on the bus worth naming, in bus order. Empty where there is no sysfs.

    Cheap enough to poll: a couple of dozen small reads out of a filesystem that is generated in
    memory. It is on the path of a session opening, so it is measured - see the suite.
    """
    folder = SYSFS if root is None else root
    found = []
    try:
        entries = sorted(folder.iterdir(), key=lambda p: _order(p.name))
    except OSError:
        return ()
    for entry in entries:
        vid, pid = _read(entry, "idVendor").lower(), _read(entry, "idProduct").lower()
        if not vid or not pid:
            continue  # an interface, or something without descriptors; not a device
        ident = f"{vid}:{pid}"
        if ident in IGNORED or _read(entry, "bDeviceClass").lower() == HUB_CLASS:
            continue
        found.append(Device(vid, pid, _name(ident, entry), _category(ident, entry)))
    return tuple(found)


def changes(before: tuple[Device, ...],
            after: tuple[Device, ...]) -> tuple[list[Device], list[Device]]:
    """What arrived and what left, between two readings. Identity is the vid:pid."""
    was = {one.ident: one for one in before}
    now = {one.ident: one for one in after}
    return ([one for ident, one in now.items() if ident not in was],
            [one for ident, one in was.items() if ident not in now])


SETTLE_S = 1.2  # how long a change has to hold before it is believed. A connector on its way in
# enumerates, disappears and comes back; a plug that is half in does that all afternoon
QUIET_S = 10.0  # ...and the least that may pass between two things being said out loud. The bus
# is not a conversation, and a box that announces its own cables twice in ten seconds is one


class Watch:
    """Readings in, changes out: what the panel shows, and what is worth saying out loud.

    Pure - it never reads the bus and never asks the time, both of which are handed in. That is
    what lets a flapping connector and a ten-second rule be tested with a fake clock instead of
    with a plug and a stopwatch.

    Two rules and they are different rules. The panel follows anything that *settles*
    (:data:`SETTLE_S`), because a label appearing is the whole of the feedback that something was
    plugged in and it has to arrive at about the speed of the plug going in. Speech follows a
    second, slower clock (:data:`QUIET_S`), because being told twice about a cable is worse than
    not being told at all - a held-back change is not dropped, it is said late and said once.

    The first reading is what is already plugged in, not news, so it primes and says nothing.
    """

    def __init__(self, settle_s: float = SETTLE_S, quiet_s: float = QUIET_S) -> None:
        self.listed: tuple[Device, ...] = ()  # what the panel draws
        self.settle_s, self.quiet_s = settle_s, quiet_s
        self._told: tuple[Device, ...] = ()  # ...and what the model has been told about
        self._seen: tuple[Device, ...] = ()
        self._since = 0.0
        self._said_at = -quiet_s
        self._primed = False

    def catch_up(self) -> None:
        """A session has just opened, so its instructions carry the list. Nothing to announce."""
        self._told = self.listed

    def seen(self, found: tuple[Device, ...], now: float) -> tuple[list[Device], list[Device]]:
        """One reading of the bus at monotonic *now*. What arrived and what left, or two empties.

        Returns at most one change per call and only once it has held still, so five flaps of a
        connector inside a second come out as the one thing that was actually true at the end
        of them.
        """
        if not self._primed:
            self._primed = True
            self._seen = self.listed = self._told = found
            self._since = now
            return ([], [])
        if found != self._seen:
            self._seen, self._since = found, now
        if found != self.listed and now - self._since >= self.settle_s:
            self.listed = found
        if self.listed == self._told:
            return ([], [])
        arrived, left = changes(self._told, self.listed)
        # Only speech is rated. A thing being unplugged is told quietly and asks for no answer,
        # so it must not spend the silence a later plug is going to want.
        if arrived and now - self._said_at < self.quiet_s:
            return ([], [])
        self._told = self.listed
        if arrived:
            self._said_at = now
        return (arrived, left)
