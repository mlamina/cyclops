"""Device extensions: one module per USB device, carrying what the model gets while it is in.

A module in this package declares one thing, ``EXTENSION = Extension(...)``: which devices it is
for, the lines that go into the prompt while one of them is on the bus, and zero or more tools.
That is the whole of adding a device - one new file here. The prompt block, the tool list, the
dispatch and the panel's caption are all read out of that one declaration by
:mod:`cyclops.agent`, and :mod:`cyclops.devices` builds its table of badly-described devices out
of it too.

Two ways to match, because a device is either a particular box or a kind of box: ``usb_ids`` for
the endoscope or a TD-3, ``categories`` for "any audio interface", which nobody can enumerate. An
extension matched by either is applied once, however many devices on the bus matched it.

Never imports :mod:`cyclops.devices`, and never must: devices builds its override table out of
this registry, so the dependency runs one way and a category is a plain string here for that
reason. And a module that raises on import is logged and skipped - a broken extension costs its
own device its paragraph, never the session.
"""

from __future__ import annotations

import importlib
import importlib.util
import pkgutil
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, Protocol

# Below the list of what is plugged in, and only when something on it has an extension. Says what
# those things can do that their names do not - the list above says what is there.
HEADER = """\
WHAT THOSE DEVICES ARE FOR
What you can do with the things above, beyond what their names say.
"""

BUSY_MARK = "…"  # every panel caption ends in one; see cyclops.overlay.BUSY_MARK


class Plugged(Protocol):
    """A device as this module sees one: :class:`cyclops.devices.Device`, without importing it."""

    name: str
    category: str

    @property
    def ident(self) -> str: ...


@dataclass(frozen=True, eq=False)
class Tool:
    """One tool an extension offers the model while its device is plugged in.

    The three things a built-in tool spreads across four places in :mod:`cyclops.agent`: the
    schema, the handler, and the line the panel shows while it runs. ``run`` is blocking and is
    run in a thread; it gets the model's arguments and the device that matched, and returns the
    dict the model is handed back. If it raises, the model is told so rather than left waiting.
    """

    schema: dict[str, Any]  # the realtime function-tool dict, as agent.py writes its own
    run: Callable[[dict[str, Any], Any], dict[str, Any]]
    caption: str  # what the panel says while it runs, e.g. "reading the patch…"

    @property
    def name(self) -> str:
        return self.schema["name"]

    @property
    def line(self) -> str:
        """The caption as the panel wants it: ending in the mark that says work is in flight."""
        return self.caption if self.caption.endswith(BUSY_MARK) else self.caption + BUSY_MARK


@dataclass(frozen=True, eq=False)
class Extension:
    """Everything about one device, in one place. Identity, not value, is what equality means."""

    name: str
    instructions: str  # what goes in the prompt while it is plugged in
    usb_ids: tuple[str, ...] = ()  # "vid:pid", lowercase - a particular box
    categories: tuple[str, ...] = ()  # one of cyclops.devices' four - any box of a kind
    category: str = ""  # what it is, for a device whose bus descriptors will not say
    tools: tuple[Tool, ...] = ()
    # A board that reboots into new firmware leaves the bus under one of its usb_ids and comes
    # back under another. Gone for less than this, it never left - see cyclops.devices.Watch.
    rejoin_s: float = 0.0

    def matches(self, device: Plugged) -> bool:
        return device.ident in self.usb_ids or device.category in self.categories


class Match(NamedTuple):
    """An extension that applies this session, and the first device on the bus it applies to."""

    extension: Extension
    device: Any


def _modules(where: Iterable[str], prefix: str) -> list[pkgutil.ModuleInfo]:
    return sorted(pkgutil.iter_modules(list(where), prefix), key=lambda info: info.name)


def _import(info: pkgutil.ModuleInfo, shipped: bool) -> Any:
    """The module behind *info*. Shipped ones are real submodules, imported once; an extra
    folder's are loose, and run afresh each time - they are fixtures, and a test may swap one."""
    if shipped:
        return importlib.import_module(info.name)
    spec = info.module_finder.find_spec(info.name, None)  # type: ignore[call-arg]
    module = importlib.util.module_from_spec(spec)
    sys.modules[info.name] = module  # a dataclass in it looks its own module up by name
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[info.name]
        raise
    return module


def load(extra: Path | None = None) -> tuple[Extension, ...]:
    """Every extension there is, in module-name order: this package's, then *extra*'s.

    Read once per session rather than once per process, so a module dropped in is picked up at
    the next wake - and it is cheap enough to, because an import is only paid once and what is
    left is a directory listing. *extra* is a folder of modules the same shape as this package,
    which is how a fixture gets in without being shipped or put behind a setting.
    """
    found = []
    for shipped, where, prefix in ((True, __path__, f"{__name__}."),
                                   (False, [str(extra)] if extra else [], "_cyclops_extra.")):
        for info in _modules(where, prefix):
            try:
                declared = getattr(_import(info, shipped), "EXTENSION", None)
            except Exception as exc:  # a broken extension costs its device, never the session
                print(f"· extension {info.name} failed to load, skipped: {exc!r}",
                      file=sys.stderr, flush=True)
                continue
            if isinstance(declared, Extension):
                found.append(declared)
            else:
                print(f"· extension {info.name} declares no EXTENSION, skipped",
                      file=sys.stderr, flush=True)
    return tuple(found)


def matching(plugged: Iterable[Plugged], loaded: Iterable[Extension]) -> tuple[Match, ...]:
    """Each extension something on the bus is for, once, in load order."""
    on_bus = tuple(plugged)
    matched = []
    for extension in loaded:
        device = next((one for one in on_bus if extension.matches(one)), None)
        if device is not None:
            matched.append(Match(extension, device))
    return tuple(matched)


def block(matched: Iterable[Match]) -> str:
    """The prompt's paragraph for what is plugged in, or nothing at all."""
    lines = [f"{m.extension.name}: {m.extension.instructions.strip()}" for m in matched]
    if not lines:
        return ""
    return f"{HEADER}\n" + "\n".join(lines) + "\n"


def schemas(matched: Iterable[Match]) -> list[dict[str, Any]]:
    """The tools to offer: every matched extension's, and nothing for what is not plugged in."""
    return [tool.schema for m in matched for tool in m.extension.tools]


def find(matched: Iterable[Match], name: str) -> tuple[Tool, Any] | None:
    """The tool the model called by *name*, and the device to run it against."""
    for m in matched:
        for tool in m.extension.tools:
            if tool.name == name:
                return tool, m.device
    return None


def known(loaded: Iterable[Extension]) -> dict[str, tuple[str, str]]:
    """``vid:pid -> (category, name)`` for every device an extension says its bus cannot."""
    return {ident: (ext.category, ext.name)
            for ext in loaded if ext.category for ident in ext.usb_ids}


def rejoins(loaded: Iterable[Extension]) -> dict[str, tuple[str, float]]:
    """``vid:pid -> (extension name, seconds)`` for every device that may drop off and come back
    as another of its own IDs. The name is what makes two IDs the same device."""
    return {ident: (ext.name, ext.rejoin_s)
            for ext in loaded if ext.rejoin_s > 0 for ident in ext.usb_ids}
