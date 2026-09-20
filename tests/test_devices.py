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
from PIL import Image, ImageDraw

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


# ---------------------------------------------------------------- the module in the corner

RAIL = (
    devices.Device("2ce3", "3828", "Endoscope", devices.CAMERA),
    devices.Device("1c75", "0288", "MiniLab 3", devices.MUSIC),
    devices.Device("0781", "5581", "SanDisk Ultra", devices.STORAGE),
)
SHORT = (
    devices.Device("2ce3", "3828", "Cam", devices.CAMERA),
    devices.Device("1c75", "0288", "Synth", devices.MUSIC),
    devices.Device("0781", "5581", "Card", devices.STORAGE),
    devices.Device("0403", "6001", "Uno", devices.OTHER),
)
PANELS = ((800, 480), (480, 320), (1280, 720))


def shot(ov: overlay.Overlay, plugged) -> np.ndarray:
    return ov.render(state=overlay.LISTENING, level=0.2, elapsed=12.0, phase=1.0,
                     plugged_in=plugged)


def test_the_pod_is_the_width_it_was() -> None:
    """The USB module is not one of the pod's tags and must not widen it.

    The pod is laid out for at most two tags and its width is its own business. Anything in the
    other corner that changed it would move the clock every time a cable went in.
    """
    for w, h in PANELS:
        was = dict(overlay.Overlay(w, h).pod_boxes)
        ov = overlay.Overlay(w, h)
        shot(ov, RAIL)
        assert dict(ov.pod_boxes) == was


def test_it_is_no_deeper_than_the_pod() -> None:
    """A shallow strip flush into the corner, not a boxed panel down the left-hand edge."""
    for w, h in PANELS:
        ov = overlay.Overlay(w, h)
        assert max(y for _, y in ov.usb_spine(ov.usb_fit(RAIL)[1])) == ov.pod_boxes[0].h


def test_it_stays_in_its_own_corner() -> None:
    """Its room is its own now, not what the pod leaves: with the pod out of the middle, room
    measured off the pod would let a long list sprawl towards the reticle."""
    for w, h in PANELS:
        ov = overlay.Overlay(w, h)
        pod_left = min(x for x, _ in ov.pods[2].spine)
        for plugged in ((), RAIL[:1], RAIL, SHORT, RAIL * 3):
            corner = max(x for x, _ in ov.usb_spine(ov.usb_fit(plugged)[1]))
            assert corner <= ov.usb_room(), f"it outgrew its corner with {len(plugged)} plugged in"
            assert corner < pod_left, f"it runs into the pod with {len(plugged)} plugged in"


# What usb_fit gave for each prefix of each list before the pod moved into the other corner, off
# the code at 844f601. The 480x320 window is left out on purpose: its room used to be a smaller
# share of the panel than at the Pi's size, and the fixed room gives it the same share instead.
MIX = SHORT[:1] + RAIL + SHORT[1:]
SHOWED = {
    (800, 480): {
        "RAIL": [((), 126), (("Endoscope",), 126), (("Endoscope", "MiniLab 3"), 158),
                 (("Endoscope", "MiniLab 3"), 158)],
        "SHORT": [((), 126), (("Cam",), 126), (("Cam", "Synth"), 126),
                  (("Cam", "Synth", "Card"), 128), (("Cam", "Synth", "Card", "Uno"), 161)],
        "MIX": [((), 126), (("Cam",), 126), (("Cam", "Endoscope"), 126),
                (("Cam", "Endoscope"), 126), (("Cam", "Endoscope"), 126),
                (("Cam", "Endoscope", "Synth"), 163), (("Cam", "Endoscope", "Synth"), 163),
                (("Cam", "Endoscope", "Synth"), 163)],
    },
    (1280, 720): {
        "RAIL": [((), 189), (("Endoscope",), 189), (("Endoscope", "MiniLab 3"), 246),
                 (("Endoscope", "MiniLab 3"), 246)],
        "SHORT": [((), 189), (("Cam",), 189), (("Cam", "Synth"), 189),
                  (("Cam", "Synth", "Card"), 198), (("Cam", "Synth", "Card", "Uno"), 249)],
        "MIX": [((), 189), (("Cam",), 189), (("Cam", "Endoscope"), 189),
                (("Cam", "Endoscope"), 189), (("Cam", "Endoscope"), 189),
                (("Cam", "Endoscope", "Synth"), 253), (("Cam", "Endoscope", "Synth"), 253),
                (("Cam", "Endoscope", "Synth"), 253)],
    },
}


def test_it_shows_what_it_showed_before_the_pod_moved() -> None:
    """Which devices show, how wide the module comes to, and where the list drops off."""
    lists = {"RAIL": RAIL, "SHORT": SHORT, "MIX": MIX}
    for (w, h), table in SHOWED.items():
        ov = overlay.Overlay(w, h)
        for name, want in table.items():
            got = []
            for n in range(len(lists[name]) + 1):
                shown, right = ov.usb_fit(lists[name][:n])
                got.append((tuple(device.name for device, _, _ in shown), round(right)))
            assert got == want, f"{w}x{h} {name}"


def test_it_is_as_wide_as_its_contents() -> None:
    """It grows with the list and shrinks back, exactly as the pod widens for its tags."""
    ov = overlay.Overlay(800, 480)
    widths = [ov.usb_fit(SHORT[:n])[1] for n in range(5)]
    assert widths == sorted(widths), widths
    assert widths[0] < widths[-1], "it never grew at all"


def test_the_plate_reaches_the_top_and_the_left(tmp_path: Path) -> None:
    """It is a chassis part flush into the corner, not a box drawn on the panel.

    Read off the polygon rather than off pixels, because the case's own surround runs round the
    outside of everything on this panel - the pod's plate runs off the top edge under it in
    exactly the same way.
    """
    ov = overlay.Overlay(800, 480)
    spine = ov.usb_spine(ov.usb_fit(RAIL)[1])
    assert min(x for x, _ in spine) == 0 and min(y for _, y in spine) == 0


def test_the_pods_plate_reaches_the_top_and_the_right() -> None:
    """The pod is the USB module's mirror: flush into the other corner, with no frame showing
    between its plate and either edge - the corner pixel itself is plate."""
    for w, h in PANELS:
        ov = overlay.Overlay(w, h)
        for tags in range(3):
            pod = ov.pods[tags]
            assert pod.corner == (w, 0)
            assert max(x for x, _ in pod.spine) == w and min(y for _, y in pod.spine) == 0
            mask = Image.new("L", (w, h), 0)
            pod.plate(ImageDraw.Draw(mask))
            assert mask.getpixel((w - 1, 0)) == 255, f"pod {tags} stops short of its corner"


def test_the_pod_has_one_chamfer_on_its_left_end() -> None:
    """Steel on its bottom run and its one chamfered end; none up the edge it runs off."""
    for w, h in PANELS:
        ov = overlay.Overlay(w, h)
        for tags in range(3):
            spine = ov.pods[tags].spine
            flat_left = spine[1][0]
            assert spine[0] == (w, ov.pod_depth), "the bottom run does not leave by the right edge"
            runs = list(zip(spine, spine[1:], strict=False))
            slopes = [(a, b) for a, b in runs if a[0] != b[0] and a[1] != b[1]]
            assert len(slopes) == 1, f"pod {tags} has {len(slopes)} chamfers"
            (ax, ay), (bx, by) = slopes[0]
            assert max(ax, bx) <= flat_left and abs(ax - bx) == abs(ay - by), "not a left 45"
            # Every other run is the bottom one or the leg up from the chamfer, and every knee a
            # bolt goes through is on that end.
            for (ax, ay), (bx, by) in runs:
                assert (ay == by == ov.pod_depth) or max(ax, bx) <= flat_left
            assert all(x <= flat_left for x, _ in spine[1:-1])


def _behind(ov: overlay.Overlay, rgba: np.ndarray, level: int) -> np.ndarray:
    """The frame as the Pi shows it, over a room that is one flat *level*."""
    room = np.full((ov.height, ov.width, 3), level, np.uint8)
    return overlay.composite(room, rgba).astype(np.float32)


def _windows(ov: overlay.Overlay, plugged) -> dict[str, np.ndarray]:
    """Where each corner module's glass proper is, as a mask over the top of the panel."""
    modules = {"usb": overlay.Bracket((0, 0), ov.usb_spine(ov.usb_fit(plugged)[1])),
               "pod": ov.pods[0]}
    reveal = max(1.0, overlay.POD_REVEAL * ov.scale)
    return {name: ov._pod_field(m, 0, 0, ov.width, ov.pod_depth)[0] < -reveal
            for name, m in modules.items()}


def test_the_usb_window_lets_no_more_of_the_room_through_than_the_pods() -> None:
    """Two rooms as far apart as rooms go, behind the same frame: whatever a window lets through
    is the difference between the two. It was a plate you could see the camera through; it is
    the pod's glass now, and no pixel of it may be clearer than the pod's clearest."""
    ov = overlay.Overlay(800, 480)
    shot(ov, RAIL)
    rgba = shot(ov, RAIL)
    through = np.abs(_behind(ov, rgba, 255) - _behind(ov, rgba, 0)).mean(axis=2) / 255.0
    top = through[: ov.pod_depth]
    leak = {name: float(top[window].max()) for name, window in _windows(ov, RAIL).items()}
    assert leak["usb"] <= leak["pod"], leak


def test_the_two_top_corners_are_the_same_part() -> None:
    """Same depth, same flange, same glass - read down a column through each bottom run, the
    same distance from each module's own lamp, so the rake that grades both is the same too."""
    ov = overlay.Overlay(800, 480)
    shot(ov, ())
    rgba = shot(ov, ())
    dark = _behind(ov, rgba, 0)
    usb = overlay.Bracket((0, 0), ov.usb_spine(ov.usb_fit(())[1]))
    assert max(y for _, y in usb.spine) == max(y for _, y in ov.pods[0].spine)

    def column(module: overlay.Bracket, along: float) -> tuple[int, int, np.ndarray]:
        (kx, _), out = ov._shoulder(module)
        x = round(kx - out * along)
        rgb = dark[: ov.pod_depth + ov.rail_w, x]
        # The flange starts on the first neutral row under the glass, which is green.
        neutral = (np.ptp(rgb, axis=1) < 12) & (rgb.mean(axis=1) > 40)
        steel = next(y for y in range(round(ov.pod_depth * 0.6), len(rgb)) if neutral[y])
        solid = int(np.nonzero(rgba[: ov.pod_depth + ov.rail_w, x, 3] == 255)[0].max())
        return steel, solid, rgb[steel - 7 : steel - 1].mean(axis=0)

    # Clear of NO USB, the meter's cells, the clock's digits and the case's own surround.
    for along in (40.0, 135.0, 145.0):
        u, p = column(usb, along), column(ov.pods[0], along)
        assert u[0] == p[0], f"{along}: the flange starts at row {u[0]} against {p[0]}"
        assert u[1] == p[1], f"{along}: the rail ends at row {u[1]} against {p[1]}"
        assert np.abs(u[2] - p[2]).max() <= 4, f"{along}: glass {u[2]} against {p[2]}"


def test_it_is_still_there_with_nothing_plugged_in() -> None:
    """A module that vanished would be a part falling off the machine every time a cable came
    out."""
    ov = overlay.Overlay(800, 480)
    shown, right = ov.usb_fit(())
    assert not shown and right >= overlay.USB_MIN_W * ov.scale
    empty, full = shot(ov, ()), shot(ov, RAIL)
    box = ov.usb_box(right)
    assert not np.array_equal(empty[: box.bottom, : box.right],
                              full[: box.bottom, : box.right]), "the two states draw the same"



def test_nothing_outside_the_module_moved() -> None:
    """Plugging something in changes the corner it is in and nothing else on the panel."""
    ov = overlay.Overlay(800, 480)
    empty, full = shot(ov, ()).copy(), shot(ov, RAIL).copy()
    box = ov.usb_box(max(ov.usb_fit(())[1], ov.usb_fit(RAIL)[1]))
    empty[: box.bottom, : box.right] = 0
    full[: box.bottom, : box.right] = 0
    assert np.array_equal(empty, full)


def test_the_four_glyphs_are_four_different_marks() -> None:
    """Told apart at panel size is Marco's to judge; being *different* is not.

    Cheap and blunt on purpose: four tiles of one size, and no two of them the same pixels. It
    is the regression a fifth category copy-pasted from a fourth would otherwise pass.
    """
    ov = overlay.Overlay(800, 480)
    cut = {}
    for category in (devices.CAMERA, devices.MUSIC, devices.STORAGE, devices.OTHER):
        tile = ov._usb_entry(devices.Device("0000", "0000", "Name", category), "Name", 40.0)
        cut[category] = np.asarray(tile.crop((0, 0, 40, round(overlay.USB_GLYPH)))).tobytes()
    assert len(set(cut.values())) == 4, "two categories draw the same mark"


def test_a_long_name_is_cut_rather_than_eating_the_corner() -> None:
    ov = overlay.Overlay(800, 480)
    hog = devices.Device("0000", "0000", "Generic USB Audio In", devices.MUSIC)
    _, w = ov._usb_col(hog)
    assert w <= overlay.USB_NAME_W * ov.scale
