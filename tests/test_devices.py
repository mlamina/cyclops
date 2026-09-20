"""What is on the bus, off a recorded sysfs tree rather than off whatever is plugged into the Mac.

sysfs is a directory of small text files, which is the one kind of kernel interface a test can
build a copy of. Everything here is one of those copies: the two IDs read off the Pi on
2026-09-20, plus a webcam, a MIDI keyboard, a stick and a hub written the way the kernel writes
them.

The Mac case is a test in its own right and not an accident of where the suite runs: no sysfs
means no devices, which is what keeps the panel, the prompt and ``panel_shot.py`` from quietly
behaving differently on a laptop.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cyclops import agent, devices, overlay
from cyclops.config import Settings


class _Conn:
    """Records what the agent asks the server for - tests/test_greeting.py's FakeConn."""

    def __init__(self) -> None:
        self.responses: list[dict] = []
        self.items: list[dict] = []
        self.response = SimpleNamespace(create=self._response)
        self.conversation = SimpleNamespace(item=SimpleNamespace(create=self._item))

    async def _response(self, **kwargs) -> None:
        self.responses.append(kwargs)

    async def _item(self, **kwargs) -> None:
        self.items.append(kwargs)

# (folder, idVendor, idProduct, bDeviceClass, manufacturer, product, [interface classes])
C920 = ("1-2", "046d", "08e5", "ef", "", "HD Pro Webcam C920", ["0e", "01", "01"])
MINILAB = ("1-3", "1c75", "0288", "00", "Arturia", "MiniLab 3", ["01", "01"])
STICK = ("1-4", "0781", "5581", "00", "SanDisk", "Ultra", ["08"])
HUB = ("usb1", "1d6b", "0003", "09", "Linux 6.6.51", "xHCI Host Controller", ["09"])
# Off the Pi: no manufacturer, no product, and two vendor-specific interfaces.
ENDOSCOPE = ("1-1", "2ce3", "3828", "ef", "", "", ["ff", "ff"])
LAV = ("3-1", "4c4a", "4155", "00", "Jieli Technology", "USB Composite Device", ["01", "01", "03"])


def lay(root: Path, *entries) -> Path:
    """A sysfs tree holding these devices, written the way the kernel writes one."""
    bus = root / "devices"
    for folder, vid, pid, dclass, mfr, product, interfaces in entries:
        at = bus / folder
        at.mkdir(parents=True, exist_ok=True)
        for name, value in (("idVendor", vid), ("idProduct", pid), ("bDeviceClass", dclass),
                            ("manufacturer", mfr), ("product", product)):
            if value:
                at.joinpath(name).write_text(value + "\n")
        for n, code in enumerate(interfaces):
            iface = bus / f"{folder}:1.{n}"
            iface.mkdir(parents=True, exist_ok=True)
            iface.joinpath("bInterfaceClass").write_text(code + "\n")
    return bus


def test_the_bus_says_what_each_thing_is(tmp_path: Path) -> None:
    """A webcam, a keyboard and a stick sort themselves out of their interface classes."""
    found = devices.connected(lay(tmp_path, C920, MINILAB, STICK, HUB))
    assert [(one.name, one.category) for one in found] == [
        ("HD Pro Webcam C920", devices.CAMERA),
        ("MiniLab 3", devices.MUSIC),
        ("Ultra", devices.STORAGE),
    ], "a hub is on the bus like anything else and must not reach the panel"


def test_the_two_ids_off_the_pi(tmp_path: Path) -> None:
    """The endoscope describes itself as nothing at all, and the lav mic is never news."""
    found = devices.connected(lay(tmp_path, ENDOSCOPE, LAV))
    assert [(one.name, one.category) for one in found] == [("Endoscope", devices.CAMERA)]


def test_no_sysfs_means_nothing_plugged_in(tmp_path: Path) -> None:
    """Off Linux the list is empty, so no tag renders and no prompt block is added."""
    assert devices.connected(tmp_path / "there-is-no-such-bus") == ()
    assert devices.connected() == () or Path("/sys/bus/usb/devices").exists()


def test_reading_the_bus_is_quick_enough_for_a_greeting(tmp_path: Path) -> None:
    """It sits on the path of a session opening, which has about a second of margin.

    A Pi's bus is a handful of devices and four root hubs; this is a dozen, so the number is
    pessimistic by the only direction that matters.
    """
    crowd = [(f"1-{n}", "046d", f"08{n:02x}", "ef", "Logitech", f"Thing {n}", ["0e", "01"])
             for n in range(1, 13)]
    bus = lay(tmp_path, *crowd, HUB)
    devices.connected(bus)  # the first read warms the page cache, and the greeting's is not first
    start = time.perf_counter()
    for _ in range(10):
        devices.connected(bus)
    assert (time.perf_counter() - start) / 10 * 1000 < 5.0


def test_the_row_does_not_reshuffle_when_the_bus_grows(tmp_path: Path) -> None:
    """Ports sort as numbers, so 1-10 goes after 1-2 rather than between 1-1 and 1-2."""
    late = ("1-10", "0483", "5740", "00", "STMicro", "Probe", ["ff"])
    found = devices.connected(lay(tmp_path, C920, MINILAB, late))
    assert [one.name for one in found] == ["HD Pro Webcam C920", "MiniLab 3", "Probe"]


def test_a_thing_with_no_words_on_it_still_gets_a_name(tmp_path: Path) -> None:
    """Manufacturer, then the bare IDs - never an empty tag on the rail."""
    mute = ("1-5", "beef", "0001", "00", "Some Co", "", ["03"])
    anonymous = ("1-6", "dead", "0002", "00", "", "", ["03"])
    found = devices.connected(lay(tmp_path, mute, anonymous))
    assert [(one.name, one.category) for one in found] == [
        ("Some Co", devices.OTHER), ("dead:0002", devices.OTHER),
    ]


def test_a_long_product_string_is_cut_to_what_the_rail_holds(tmp_path: Path) -> None:
    long = ("1-7", "1234", "5678", "00", "", "Generic USB 2.0 Full Speed Audio Interface", ["01"])
    (one,) = devices.connected(lay(tmp_path, long))
    assert len(one.name) <= devices.NAME_CHARS and one.name == one.name.strip()


def test_changes_are_by_id_not_by_position(tmp_path: Path) -> None:
    """Unplug one of three and the other two are not reported as having moved."""
    before = devices.connected(lay(tmp_path / "a", C920, MINILAB, STICK))
    after = devices.connected(lay(tmp_path / "b", C920, STICK))
    arrived, left = devices.changes(before, after)
    assert not arrived and [one.name for one in left] == ["MiniLab 3"]


# ---------------------------------------------------------------- when a change is worth saying

BUS = (C920, MINILAB)


def watched(tmp_path: Path, *entries) -> tuple[devices.Device, ...]:
    return devices.connected(lay(tmp_path / f"t{len(entries)}-{id(entries)}", *entries))


def test_the_first_reading_is_not_news(tmp_path: Path) -> None:
    """What is already plugged in when the box comes up has never been plugged in."""
    watch = devices.Watch()
    arrived, left = watch.seen(watched(tmp_path, *BUS), 0.0)
    assert not arrived and not left
    assert [one.name for one in watch.listed] == ["HD Pro Webcam C920", "MiniLab 3"]


def test_a_flapping_connector_is_one_announcement(tmp_path: Path) -> None:
    """Five bounces inside a second come out as the one thing that was true at the end."""
    watch = devices.Watch()
    alone, both = watched(tmp_path, C920), watched(tmp_path, C920, MINILAB)
    watch.seen(alone, 0.0)
    said = []
    for n in range(5):  # in, out, in, out, in - none of them held long enough to be believed
        said.append(watch.seen(both if n % 2 == 0 else alone, 0.1 + n * 0.2))
    assert not any(a for a, _ in said), "a bouncing plug became a conversation"
    for tick in range(1, 12):  # it settles, and is announced exactly once
        said.append(watch.seen(both, 1.1 + tick * 0.5))
    spoke = [a for a, _ in said if a]
    assert len(spoke) == 1 and [one.name for one in spoke[0]] == ["MiniLab 3"]


def test_an_unplug_is_told_but_does_not_spend_the_silence(tmp_path: Path) -> None:
    """Nothing arrived, so nothing is said - and a plug a second later is not made to wait."""
    watch = devices.Watch()
    both, alone = watched(tmp_path, C920, MINILAB), watched(tmp_path, C920)
    watch.seen(both, 0.0)
    for tick in range(6):
        arrived, left = watch.seen(alone, 1.0 + tick)
        if left:
            assert not arrived and [one.name for one in left] == ["MiniLab 3"]
            unplugged_at = 1.0 + tick
            break
    else:
        raise AssertionError("the unplug never reached the model at all")
    for tick in range(6):
        arrived, _ = watch.seen(both, unplugged_at + 1.0 + tick)
        if arrived:
            assert unplugged_at + 1.0 + tick - unplugged_at < devices.QUIET_S
            return
    raise AssertionError("a silent unplug held a real announcement back")


def test_two_announcements_are_never_within_ten_seconds(tmp_path: Path) -> None:
    watch = devices.Watch()
    one, two, three = (watched(tmp_path, C920), watched(tmp_path, C920, MINILAB),
                       watched(tmp_path, C920, MINILAB, STICK))
    watch.seen(one, 0.0)
    spoke = []
    for tick in range(60):  # a minute of polling, with a second plug going in at t=5
        now = 1.0 + tick * 0.5
        arrived, _ = watch.seen(two if now < 5.0 else three, now)
        if arrived:
            spoke.append(now)
    assert len(spoke) == 2, spoke
    assert spoke[1] - spoke[0] >= devices.QUIET_S


def test_a_session_opening_is_not_told_twice(tmp_path: Path) -> None:
    """Its instructions already carry the list, so catching up must announce nothing after."""
    watch = devices.Watch()
    both = watched(tmp_path, C920, MINILAB)
    watch.seen(watched(tmp_path, C920), 0.0)
    for tick in range(6):
        watch.seen(both, 1.0 + tick)
    watch.catch_up()
    for tick in range(60):
        assert watch.seen(both, 10.0 + tick) == ([], [])


# ---------------------------------------------------------------- what the model is told

PLUGGED = (
    devices.Device("2ce3", "3828", "Endoscope", devices.CAMERA),
    devices.Device("1c75", "0288", "MiniLab 3", devices.MUSIC),
)


def test_the_instructions_carry_a_line_per_device(monkeypatch) -> None:
    """A session that opens with something on the bus is told what it is and what sort."""
    monkeypatch.setattr(agent.devices, "connected", lambda root=None: PLUGGED)
    written = agent.build_instructions(Settings(api_key=""))
    assert agent.PLUGGED_HEADER.splitlines()[0] in written
    for one in PLUGGED:
        assert one.line() in written
    # ...and the clock stays last, because the greeting is the next thing said.
    assert written.index(agent.PLUGGED_HEADER.splitlines()[0]) < written.rindex("WHEN IT IS")


def test_an_empty_bus_adds_no_block_at_all(monkeypatch) -> None:
    monkeypatch.setattr(agent.devices, "connected", lambda root=None: ())
    written = agent.build_instructions(Settings(api_key=""))
    assert agent.PLUGGED_HEADER.splitlines()[0] not in written


def test_a_plug_asks_for_a_word_and_an_unplug_does_not() -> None:
    """His call: arriving is news worth a sentence, leaving is a turn nobody wanted.

    Both reach the model - it must not go on believing a thing is there - so what separates
    them is the response, not the item.
    """
    made = agent.VoiceAgent(Settings(api_key="", sounds=False))
    made._conn = _Conn()  # ...which is also what makes `connected` true
    made.ready.set()

    async def go() -> None:
        await made.add_bus_change(PLUGGED[1:], ())
        plugged = len(made.conn.responses)
        await made.add_bus_change((), PLUGGED[1:])
        assert plugged == 1, "a thing plugged in mid-session was never spoken about"
        assert len(made.conn.responses) == 1, "being told the scope is out is a turn nobody wanted"
        assert len(made.conn.items) == 2, "the model was left believing it is still there"

    asyncio.run(go())


# ---------------------------------------------------------------- where a label lands

RAIL = (
    devices.Device("2ce3", "3828", "Endoscope", devices.CAMERA),
    devices.Device("1c75", "0288", "MiniLab 3", devices.MUSIC),
    devices.Device("0781", "5581", "SanDisk Ultra", devices.STORAGE),
)
PANELS = ((800, 480), (480, 320), (1280, 720))


def test_the_module_is_the_width_it_was() -> None:
    """A device label is not one of the module's tags, and must not widen it.

    The pod is laid out for at most two tags and its width is its own business. Anything on the
    rail that changed it would move the clock every time a cable went in.
    """
    was = {(w, h): dict(overlay.Overlay(w, h).pod_boxes) for w, h in PANELS}
    for (w, h), boxes in was.items():
        ov = overlay.Overlay(w, h)
        ov.render(state=overlay.LISTENING, level=0.2, elapsed=12.0, phase=1.0, plugged_in=RAIL)
        assert dict(ov.pod_boxes) == boxes


def test_every_label_lands_clear_of_the_module() -> None:
    """Both runs of rail, measured against the module at its widest so nothing shuffles."""
    for w, h in PANELS:
        ov = overlay.Overlay(w, h)
        box = ov.pod_boxes[2]
        for device, x in ov.device_places(RAIL):
            right = x + ov._device_w(device)
            assert x >= 0 and right <= w
            assert right <= box.x or x >= box.right, f"{device.name} is under the module"


def test_the_rail_shows_what_it_can_hold_and_no_more() -> None:
    """Three ordinary names fit; a crowd does not silently stack up on top of itself."""
    ov = overlay.Overlay(800, 480)
    assert len(ov.device_places(RAIL)) == 3
    crowd = RAIL * 4
    places = ov.device_places(crowd)
    assert len(places) < len(crowd)
    spans = sorted((x, x + ov._device_w(d)) for d, x in places)
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:], strict=False)), "two labels overlap"


def test_the_four_marks_are_four_different_shapes() -> None:
    """Told apart at panel size is Marco's to judge; being *different* is not.

    Cheap and blunt on purpose: four identical-sized tiles, and no two of their marks may be
    the same pixels. It is the regression that a fifth category copy-pasted from a fourth would
    otherwise pass.
    """
    ov = overlay.Overlay(800, 480)
    cut = {}
    for category in (devices.CAMERA, devices.MUSIC, devices.STORAGE, devices.OTHER):
        tile = ov._device_tile(devices.Device("0000", "0000", "Name", category))
        cut[category] = np.asarray(tile.crop((0, 0, round(overlay.DEV_PAD + overlay.DEV_ICON),
                                              tile.height))).tobytes()
    assert len(set(cut.values())) == 4, "two categories draw the same mark"
