"""The Pico extension: its tools against a fake MicroPython, and the flash that is not a replug.

The fake speaks the raw REPL the way the board does - control characters in, ``OK``/``\\x04``
framing out - and runs what it is sent as real Python, so a traceback is a real traceback. Its
clock only moves when a read finds nothing, which is what lets a ten-second timeout run in no
time at all.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest

from cyclops import agent, card, devices, session
from cyclops.config import Settings
from cyclops.extensions import pico

BOOT = devices.Device("2e8a", "0003", "RP2 Boot", devices.OTHER)
MPY = devices.Device("2e8a", "0005", "Board in FS mode", devices.OTHER)
C920 = devices.Device("046d", "08e5", "HD Pro Webcam C920", devices.CAMERA)
BANNER = b"MicroPython v1.29.0 on 2026-08-24; Raspberry Pi Pico with RP2040\r\n" \
         b'Type "help()" for more information.\r\n>>> '
FOREVER = "while True:"  # the fake runs what comes before this, then never finishes


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class FakePico:
    """A serial port with MicroPython on the far end of it."""

    timeout = pico.READ_WAIT_S

    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.files: dict[str, str] = {}
        self.out = bytearray()
        self.raw = False
        self.code = b""
        self.hanging = False
        self.resets = 0

    @property
    def in_waiting(self) -> int:
        return len(self.out)

    def read(self, n: int = 1) -> bytes:
        if not self.out:
            self.clock.t += self.timeout  # a blocked read, costing the fake clock and not ours
            return b""
        data = bytes(self.out[:n])
        del self.out[:n]
        return data

    def write(self, data: bytes) -> None:
        for byte in data:
            self._byte(bytes([byte]))

    def _python(self, code: str) -> tuple[str, str, bool]:
        """(stdout, traceback, never finishes)"""
        forever = "\n" + FOREVER in code  # a line of its own, not inside a string being saved
        code = code.split("\n" + FOREVER)[0]
        printed, err = io.StringIO(), ""
        files = self.files

        class Handle(io.StringIO):
            def __init__(self, name: str) -> None:
                super().__init__()
                self.name_ = name

            def close(self) -> None:
                files[self.name_] = self.getvalue()

        with contextlib.redirect_stdout(printed):
            try:
                exec(code, {"open": lambda name, mode="r": Handle(name)})
            except Exception as exc:
                err = f"Traceback (most recent call last):\n  File \"<stdin>\", line 1\n" \
                      f"{''.join(traceback.format_exception_only(exc)).strip()}\n"
        return printed.getvalue(), err, forever and not err

    def _emit(self, text: str | bytes) -> None:
        self.out += text.encode() if isinstance(text, str) else text

    def _byte(self, b: bytes) -> None:
        if b == b"\x03":
            if self.hanging:
                self.hanging = False
                self._emit(b"\x04Traceback (most recent call last):\r\nKeyboardInterrupt: \r\n\x04>")
            return
        if not self.raw:
            if b == b"\x01":
                self.raw = True
                self._emit(pico.RAW_PROMPT)
            elif b == b"\x04":
                self.resets += 1
                self._emit(b"MPY: soft reboot\r\n")
                out, err, forever = self._python(self.files.get("main.py", ""))
                self._emit(out.replace("\n", "\r\n") + err.replace("\n", "\r\n"))
                if not forever:
                    self._emit(BANNER)
            return
        if b == b"\x02":
            self.raw = False
            self._emit(BANNER)
        elif b == b"\x04":
            out, err, forever = self._python(self.code.decode())
            self.code = b""
            self._emit(b"OK" + out.replace("\n", "\r\n").encode())
            if forever:
                self.hanging = True
            else:
                self._emit(b"\x04" + err.encode() + b"\x04>")
        elif b not in b"\r\x01":
            self.code += b


@pytest.fixture
def board() -> pico.Board:
    clock = Clock()
    return pico.Board(FakePico(clock), clock)


# ---------------------------------------------------------------- offered while plugged in


def tools_with(monkeypatch, *plugged: devices.Device) -> list[str]:
    monkeypatch.setattr(agent.devices, "connected", lambda root=None: tuple(plugged))
    config = agent.VoiceAgent(Settings(api_key="", sounds=False)).session_config()
    return [tool["name"] for tool in config["tools"]]


def test_its_tools_come_with_either_state_of_the_board_and_go_without_it(monkeypatch) -> None:
    ours = {"pico_program", "pico_run", "pico_output", "pico_show"}
    assert ours <= set(tools_with(monkeypatch, C920, BOOT))
    assert ours <= set(tools_with(monkeypatch, MPY))
    assert not ours & set(tools_with(monkeypatch, C920))


# ---------------------------------------------------------------- the raw REPL


def test_a_program_is_saved_restarted_and_its_first_words_come_back(board) -> None:
    got = board.program("print('hello from the bench')\n" + FOREVER + "\n    pass\n")
    assert board.port.files["main.py"].startswith("print('hello from the bench')")
    assert board.port.resets == 1
    assert got["state"] == pico.RUNNING and got["output"] == "hello from the bench"


def test_a_program_that_crashes_hands_back_its_traceback(board) -> None:
    got = board.program("print('on')\nfrom machine import Pin\n")
    assert got["state"] == pico.CRASHED and not got["ok"]
    assert "ModuleNotFoundError" in got["traceback"] and got["output"].startswith("on")


def test_a_check_answers_and_the_saved_program_starts_again(board) -> None:
    board.program("print('blinking')\n" + FOREVER + "\n    pass\n")
    got = board.run("print(1 + 1)")
    assert got == {"ok": True, "output": "2\n"}
    assert board.port.resets == 2 and not board.port.raw


def test_a_check_that_never_ends_gives_up_with_what_it_printed(board) -> None:
    got = board.run("print('tick')\n" + FOREVER + "\n    pass\n")
    assert got["output"] == "tick\n" and got["timed_out"] and not got["ok"]
    assert board.port.clock.t < pico.RUN_TIMEOUT_S + 5 * pico.REPL_WAIT_S
    assert board.port.resets == 1, "the saved program was not started again"


def test_output_is_what_was_printed_since_the_last_look(board) -> None:
    board.program("print('a')\n" + FOREVER + "\n    pass\n")
    board._feed(b"b\r\nc\r\n")
    assert board.output()["output"] == "b\nc"
    assert board.output()["output"] == ""


# ---------------------------------------------------------------- nothing gets lost


def test_each_program_lands_in_the_session_and_its_page(monkeypatch, tmp_path: Path) -> None:
    records: list[dict] = []
    live = SimpleNamespace(dir=tmp_path, event=lambda kind, **f: records.append(
        {"type": kind, "t": 0, **f}))
    monkeypatch.setattr(session, "_live", live)
    pico.keep("from machine import Pin\nPin('LED').on()\n", "led on")
    (kept,) = (tmp_path / card.CODE).iterdir()
    assert kept.name.endswith("_pico_led-on.py") and "Pin('LED')" in kept.read_text()
    assert "```python\nfrom machine import Pin\nPin('LED').on()\n```" in \
        session.render_markdown(records)
    assert card.surprises(tmp_path) == []


# ---------------------------------------------------------------- the flash is not a replug


def test_the_board_rebooting_into_micropython_is_not_a_plug(monkeypatch) -> None:
    """0003 leaves, the bus is empty for seconds, 0005 arrives: nothing reaches the model."""
    watch = devices.Watch()
    made = agent.VoiceAgent(Settings(api_key="", sounds=False))
    made._conn = conn = SimpleNamespace(items=[])

    async def item(**kwargs) -> None:
        conn.items.append(kwargs)

    conn.conversation = SimpleNamespace(item=SimpleNamespace(create=item))
    made.ready.set()
    watch.seen((C920, BOOT), 0.0)
    for tick in range(40):
        now = 1.0 + tick * 0.5
        on_bus = (C920, BOOT) if now < 3 else (C920,) if now < 9 else (C920, MPY)
        arrived, left = watch.seen(on_bus, now)
        asyncio.run(made.add_bus_change(tuple(arrived), tuple(left)))  # as the kiosk hands it on
    assert conn.items == [], "the flash reached the model as a plug or an unplug"
    assert MPY in watch.listed, "the rail still shows the bootloader"
    for tick in range(80):  # ...and pulling it out for good is still an unplug, told quietly
        arrived, left = watch.seen((C920,), 30.0 + tick)
        if left:
            assert left == [MPY]
            break
    else:
        raise AssertionError("a real unplug never reached the model")


def test_an_endoscope_leaving_is_not_held_back(monkeypatch) -> None:
    scope = devices.Device("2ce3", "3828", "Endoscope", devices.CAMERA)
    watch = devices.Watch()
    watch.seen((C920, scope), 0.0)
    told = [watch.seen((C920,), 1.0 + tick * 0.5)[1] for tick in range(6)]
    assert [scope] in told



# Where pins 1, 20, 21 and 40 are as you look at the board, worked out by hand from the
# datasheet's pinout (USB at the top, chip up: 1 top left, 20 bottom left, 21 bottom right, 40
# top right), turned for each way the USB can point and mirrored for chip down.
CORNERS = {
    ("up", "up"): ("TL", "BL", "BR", "TR"),
    ("left", "up"): ("BL", "BR", "TR", "TL"),
    ("down", "up"): ("BR", "TR", "TL", "BL"),
    ("right", "up"): ("TR", "TL", "BL", "BR"),
    ("up", "down"): ("TR", "BR", "BL", "TL"),
    ("left", "down"): ("TL", "TR", "BR", "BL"),
    ("down", "down"): ("BL", "TL", "TR", "BR"),
    ("right", "down"): ("BR", "BL", "TL", "TR"),
}


@pytest.mark.parametrize(("usb", "chip"), list(CORNERS))
def test_every_orientation_puts_the_pins_where_the_board_has_them(usb: str, chip: str) -> None:
    places = {n: pico.pin_place(n, usb, chip) for n in range(1, 41)}
    xs, ys = {x for x, _ in places.values()}, {y for _, y in places.values()}

    def corner(n: int) -> str:
        x, y = places[n]
        return ("T" if y == min(ys) else "B" if y == max(ys) else "?") + \
            ("L" if x == min(xs) else "R" if x == max(xs) else "?")

    assert tuple(corner(n) for n in (1, 20, 21, 40)) == CORNERS[usb, chip]

    for first in (1, 30, 60):  # pin 1's column is all they are asked for
        layout = pico.make_layout(usb, chip, first)
        holes = [pico.pin_hole(n, layout) for n in range(1, 41)]
        assert holes[0][0] == first and len(set(holes)) == 40
        assert all(col >= 1 and row in pico.ROWS for col, row in holes)
        assert {pico._across(row) for _, row in holes} == {
            pico._across(layout.pin1_row), pico._across(pico.partner(layout.pin1_row))}
        assert abs(pico._across(holes[0][1]) - pico._across(holes[20][1])) == pico.PICO_SPAN


def test_the_same_picture_is_the_same_bytes() -> None:
    layout = pico.make_layout("left", "up", 60, "h")
    parts = pico.resolve([
        {"kind": "resistor", "from": "GP15", "to": "37j", "label": "330 ohm"},
        {"kind": "led", "from": "37i", "to": "36i"},
        {"kind": "wire", "from": "GND", "to": "36j", "colour": "black"},
    ], layout)
    assert pico.render(layout, parts) == pico.render(layout, parts)


def test_a_lead_goes_beside_its_pin_and_never_where_it_cannot() -> None:
    layout = pico.make_layout("left", "up", 60, "h")
    [part] = pico.resolve([{"kind": "wire", "from": "GP15", "to": "36j"}], layout)
    assert part.ends[0] == (41, "j")
    for wrong in ([{"kind": "wire", "from": "50f", "to": "36j"}],  # under the Pico
                  [{"kind": "wire", "from": "45j", "to": "36j"}],  # GP12's strip, unnamed
                  [{"kind": "resistor", "from": "GP15", "to": "37j"},  # two leads, one hole
                   {"kind": "led", "from": "37j", "to": "36j"}]):
        with pytest.raises(ValueError):
            pico.resolve(wrong, layout)
