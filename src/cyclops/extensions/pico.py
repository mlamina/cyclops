"""A Raspberry Pi Pico on a USB cable: Cyclops programs it, runs checks on it, and reads it.

Three tools over MicroPython's raw REPL - the small, documented exchange ``mpremote`` itself
speaks, kept here rather than borrowed, because mpremote's Python API is not a surface anybody
promises to keep. ``pico_program`` is the hot-swap: interrupt, write ``main.py``, soft-reset, and
hand back what it printed first. ``pico_run`` is a one-off question asked of the board.
``pico_output`` is what the running program has printed since the last look.

A blank Pico is the bootloader (``2e8a:0003``), a drive and no serial port. The first call against
one copies a pinned MicroPython onto that drive and carries on once the port shows up; the board
drops off the bus and comes back as ``2e8a:0005``, which ``rejoin_s`` tells the bus watcher is one
board rebooting and not an unplug followed by a plug.

Every program written is also kept in the live session's ``code/`` folder, so what was on the
board last Tuesday is on the card and in ``session.md``. No session, nothing kept - the board is
programmed either way.

Never imports :mod:`cyclops.devices` (see the package), and reaches the session lazily, inside the
one function that writes to it, so loading the extension never pulls the session machinery in.
"""

from __future__ import annotations

import contextlib
import functools
import glob
import io
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.request
from collections import deque
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, NamedTuple

from . import Extension, Tool

BOOT_ID, MICROPYTHON_ID = "2e8a:0003", "2e8a:0005"
FIRMWARE_URL = "https://micropython.org/resources/firmware/RPI_PICO-20260824-v1.29.0.uf2"
FIRMWARE_CACHE = Path.home() / ".cache" / "cyclops"
BOOT_LABEL = "RPI-RP2"
BAUD = 115200  # ignored by USB CDC, but pyserial wants one

READ_WAIT_S = 0.5  # the most a blocked read holds the lock; the reader wakes twice a second
PROGRAM_WATCH_S = 2.0  # how long a new program is listened to before the tool answers
RUN_TIMEOUT_S = 10.0  # a one-off check that has not finished by now is stopped
RESTART_WATCH_S = 0.5  # enough to see the saved program start again after a check
REPL_WAIT_S = 3.0  # for the board to answer a control character
FLASH_WAIT_S = 30.0  # bootloader drive written -> MicroPython's serial port up
LINES_KEPT = 200
OUTPUT_CHARS = 4000  # what goes back to the model, from the end

RAW_PROMPT = b"raw REPL; CTRL-B to exit\r\n>"
BANNER = 'Type "help()"'  # the friendly REPL is back: whatever was running has ended
SOFT_REBOOT = "soft reboot"
TRACEBACK = "Traceback (most recent call last)"

RUNNING, FINISHED, CRASHED = "running", "finished", "crashed"

# Physical pin -> what is on it. USB end at the top, chip side up: 1-20 run down the left,
# 21-40 run back up the right, so 21 is at the far end and 40 sits beside the USB socket.
PINS = (
    "GP0", "GP1", "GND", "GP2", "GP3", "GP4", "GP5", "GND", "GP6", "GP7",
    "GP8", "GP9", "GND", "GP10", "GP11", "GP12", "GP13", "GND", "GP14", "GP15",
    "GP16", "GP17", "GND", "GP18", "GP19", "GP20", "GP21", "GND", "GP22", "RUN",
    "GP26/ADC0", "GP27/ADC1", "GND", "GP28/ADC2", "ADC_VREF", "3V3(OUT)", "3V3_EN", "GND",
    "VSYS", "VBUS",
)


def _pinout() -> str:
    left = ", ".join(f"{n} {PINS[n - 1]}" for n in range(1, 21))
    right = ", ".join(f"{n} {PINS[n - 1]}" for n in range(21, 41))
    return f"Left, from the USB end down: {left}. Right, from the far end back up: {right}."


INSTRUCTIONS = (
    "a Raspberry Pi Pico (listed above as RP2 Boot or Board in FS mode) that you run for them "
    "with the pico_ tools - they wire, you write the MicroPython. Listed as RP2 Boot it is "
    "blank: your first pico_ call installs MicroPython (about ten seconds), so say you are "
    "setting it up before you call. Onboard LED: Pin(\"LED\"), GP25, not on a pin.\n"
    "The board is yours to drive: anything that happens on the Pico - running code, a scan, a "
    "test, a restart, a new program - you do yourself, straight away, and then say what you "
    "found. Never tell them to run, scan, reset or restart something, and never ask whether "
    "they want you to: just do it. Never ask them something the board can tell you - which "
    "pins, what address, is it alive - find out with pico_run. When they finish wiring or say "
    "it doesn't work, check it on the board before you say anything else. Only their hands "
    "(wiring, replugging, power) are theirs.\n"
    "Pinout, USB end at the top, chip side up. " + _pinout() + "\n"
    "Wiring: always give both names and where the pin is - \"GP15, that's pin 20, last on the "
    "left counting from the USB end\". Once you know how it sits on their breadboard, show it "
    "with pico_show (never draw) and name holes as it answers - \"GP15, column 41, row j\". "
    "One wire at a time, and wait for them. 3.3 V logic: "
    "never 5 V, VBUS or VSYS into a GP pin; an LED needs a resistor (220-330 ohm) in series."
)


# ---------------------------------------------------------------- the raw REPL


class Board:
    """One open serial port to a MicroPython Pico, and the log of what it printed.

    The port is shared by the tools and a background reader, and a lock serialises them. The
    reader blocks on the port rather than polling it, so a quiet board costs nothing; a tool that
    wants the port raises ``_claim`` and the reader steps aside at its next wake.
    """

    def __init__(self, port: Any, clock: Callable[[], float] = time.monotonic) -> None:
        self.port = port
        self._clock = clock
        self._lock = threading.Lock()
        self._claim = threading.Event()
        self._pending = b""
        self._lines: deque[str] = deque(maxlen=LINES_KEPT)
        self._tail = ""
        self._total = 0
        self._looked = 0
        self._state = RUNNING
        self._traceback: list[str] = []
        self._in_traceback = False
        self.alive = True

    # ---- what it printed

    def _feed(self, data: bytes) -> None:
        text = self._tail + data.decode("utf-8", errors="replace").replace("\r", "")
        *lines, self._tail = text.split("\n")
        for line in lines:
            self._line(line)

    def _line(self, line: str) -> None:
        self._lines.append(line)
        self._total += 1
        if SOFT_REBOOT in line:
            self._state, self._traceback, self._in_traceback = RUNNING, [], False
        elif line.startswith(TRACEBACK):
            self._state, self._traceback, self._in_traceback = CRASHED, [line], True
        elif self._in_traceback:
            self._traceback.append(line)
            self._in_traceback = line.startswith(" ")  # the error line ends it
        elif line.startswith(BANNER) and self._state == RUNNING:
            self._state = FINISHED

    def output(self) -> dict[str, Any]:
        """What was printed since the last look, and whether the program is still going."""
        fresh = list(self._lines)[-min(self._total - self._looked, len(self._lines)):] \
            if self._total > self._looked else []
        self._looked = self._total
        printed = [one for one in fresh if SOFT_REBOOT not in one and not one.startswith(
            ("MicroPython v", BANNER))]
        result: dict[str, Any] = {"ok": self._state != CRASHED, "state": self._state,
                                  "output": "\n".join(printed)[-OUTPUT_CHARS:]}
        if self._state == CRASHED:
            result["traceback"] = "\n".join(self._traceback)
        return result

    # ---- the port

    def listen(self) -> None:
        """Keep reading what the running program prints, in a daemon thread, until it is gone."""
        threading.Thread(target=self._listen, name="pico-reader", daemon=True).start()

    def _listen(self) -> None:
        while self.alive:
            if self._claim.is_set():
                time.sleep(0.05)
                continue
            try:
                with self._lock:
                    chunk = self.port.read(self.port.in_waiting or 1)
            except (OSError, Exception):  # pyserial's SerialException is not an OSError
                self.alive = False
                return
            if chunk:
                self._feed(chunk)

    @contextlib.contextmanager
    def _hold(self) -> Iterator[None]:
        self._claim.set()
        try:
            with self._lock:
                yield
        finally:
            self._claim.clear()

    def _read(self) -> bytes:
        return self.port.read(self.port.in_waiting or 1)

    def _until(self, marker: bytes, deadline: float) -> tuple[bytes, bool]:
        """Everything up to *marker*, and whether it came before *deadline*."""
        buf = self._pending
        while marker not in buf:
            if self._clock() >= deadline:
                self._pending = b""
                return buf, False
            buf += self._read()
        head, _, self._pending = buf.partition(marker)
        return head, True

    def _raw(self) -> None:
        """Stop whatever runs and sit at the raw REPL. What it printed first goes in the log."""
        if waiting := self.port.in_waiting:
            self._feed(self.port.read(waiting))
        self._pending = b""
        for _ in range(2):
            self.port.write(b"\r\x03\x03\r\x01")
            if self._until(RAW_PROMPT, self._clock() + REPL_WAIT_S)[1]:
                return
        raise RuntimeError("the Pico did not answer - try unplugging it and plugging it back in")

    def _exec(self, code: str, limit_s: float) -> tuple[str, str, bool]:
        """Run *code* in the raw REPL: (stdout, stderr, timed out)."""
        self.port.write(code.encode() + b"\x04")
        if not self._until(b"OK", self._clock() + REPL_WAIT_S)[1]:
            raise RuntimeError("the Pico did not take the code")
        out, done = self._until(b"\x04", self._clock() + limit_s)
        if not done:
            self.port.write(b"\x03")
            rest, _ = self._until(b"\x04", self._clock() + REPL_WAIT_S)
            out += rest
        err, _ = self._until(b"\x04", self._clock() + REPL_WAIT_S)
        self._until(b">", self._clock() + REPL_WAIT_S)
        text = out.decode(errors="replace").replace("\r", "")
        err_text = "" if not done else err.decode(errors="replace").replace("\r", "")
        return text, err_text.strip(), not done

    def _restart(self, watch_s: float) -> None:
        """Back to the friendly REPL and soft-reset it, which runs ``main.py`` - and listen."""
        self.port.write(b"\x02")
        self._until(b">>> ", self._clock() + REPL_WAIT_S)
        self._pending = b""
        self._state = "starting"
        self.port.write(b"\x04")
        end = self._clock() + watch_s
        while self._clock() < end:
            if chunk := self._read():
                self._feed(chunk)
                if self._state in (FINISHED, CRASHED):
                    break
        if self._state == "starting":
            self._state = RUNNING

    # ---- what the tools do

    def program(self, code: str) -> dict[str, Any]:
        with self._hold():
            self._raw()
            _, err, _ = self._exec(f"f=open('main.py','w')\nf.write({code!r})\nf.close()",
                                   REPL_WAIT_S)
            if err:
                raise RuntimeError(f"could not write main.py: {err}")
            self.output()  # a look: what came before this program is not its output
            self._restart(PROGRAM_WATCH_S)
            return self.output()

    def run(self, code: str) -> dict[str, Any]:
        with self._hold():
            self._raw()
            out, err, timed_out = self._exec(code, RUN_TIMEOUT_S)
            self._restart(RESTART_WATCH_S)
        result: dict[str, Any] = {"ok": not err and not timed_out, "output": out[-OUTPUT_CHARS:]}
        if err:
            result["error"] = err
        if timed_out:
            result["timed_out"] = f"stopped after {RUN_TIMEOUT_S:.0f} s; this is what it printed"
        return result


# ---------------------------------------------------------------- finding it, and flashing it


def tty() -> str:
    """The serial port of a Pico running MicroPython, or "" when there is none."""
    for path in sorted(glob.glob("/dev/serial/by-id/*MicroPython*")):
        return path
    vid, pid = MICROPYTHON_ID.split(":")
    for folder in sorted(Path("/sys/bus/usb/devices").glob("*")):
        try:
            if (folder / "idVendor").read_text().strip() != vid or \
                    (folder / "idProduct").read_text().strip() != pid:
                continue
        except OSError:
            continue
        for node in sorted(folder.parent.glob(f"{folder.name}:*/tty/tty*")):
            return f"/dev/{node.name}"
    return ""


def firmware() -> Path:
    """The pinned MicroPython UF2, downloaded once."""
    path = FIRMWARE_CACHE / FIRMWARE_URL.rsplit("/", 1)[1]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(".part")
        urllib.request.urlretrieve(FIRMWARE_URL, part)
        part.replace(path)
    return path


def boot_drive() -> Path | None:
    """Where the bootloader's drive is mounted, mounting it if nobody has."""
    for found in glob.glob(f"/media/*/{BOOT_LABEL}") + glob.glob(f"/run/media/*/{BOOT_LABEL}"):
        if Path(found, "INFO_UF2.TXT").exists():
            return Path(found)
    done = subprocess.run(["udisksctl", "mount", "--no-user-interaction", "-b",
                           f"/dev/disk/by-label/{BOOT_LABEL}"],
                          capture_output=True, text=True, timeout=15)
    if (at := re.search(r" at (\S+)", done.stdout)) is not None:
        return Path(at.group(1).rstrip("."))
    return None


def install(find: Callable[[], str] = tty, sleep: Callable[[float], None] = time.sleep) -> str:
    """Put MicroPython on a blank Pico and return its serial port once it is up."""
    image, drive = firmware(), boot_drive()
    if drive is None:
        raise RuntimeError("the Pico is in its bootloader but its drive would not mount")
    with contextlib.suppress(OSError):  # the board reboots the moment the image lands
        shutil.copyfile(image, drive / image.name)
        os.sync()
    for _ in range(int(FLASH_WAIT_S / 0.5)):
        if port := find():
            sleep(1.0)  # the port appears a beat before MicroPython listens on it
            return port
        sleep(0.5)
    raise RuntimeError("MicroPython was copied on but the Pico never came back as a serial port")


_board: Board | None = None
_opened = ""


def board() -> Board:
    """The one open connection, reopened when the board was replugged or reflashed."""
    global _board, _opened
    path = tty() or install()
    if _board is not None and _board.alive and path == _opened:
        return _board
    import serial  # here, so a box without pyserial loses this tool's call and not the extension

    if _board is not None:
        _board.alive = False
        with contextlib.suppress(Exception):
            _board.port.close()
    _board, _opened = Board(serial.Serial(path, BAUD, timeout=READ_WAIT_S)), path
    _board.listen()
    return _board


# ---------------------------------------------------------------- nothing gets lost


def keep(code: str, name: str) -> None:
    """Write the program into the live session's ``code/`` folder and log it. No session: no-op."""
    from .. import card, session, slug

    live = session.current()
    if live is None:
        return
    file = f"{time.strftime('%H-%M-%S')}_pico_{slug.slugify(name) or 'program'}.py"
    card.write_text(live.dir / card.CODE / file, code)
    session.note("code", device="Pico", name=name, file=f"{card.CODE}/{file}", code=code)


# ---------------------------------------------------------------- where every pin is


# Which way the USB socket points, as a step on the screen (x right, y down). "Up" is away from
# them across the bench, which is up in the picture.
USB_WAYS = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}
USB_WORDS = {"left": "USB to your left", "right": "USB to your right",
             "up": "USB pointing away from you", "down": "USB pointing at you"}
ROWS = "abcdefghij"
BOARD_COLUMNS = 63  # a full-size breadboard; a half-size one has 30
PICO_SPAN = 7  # the Pico's two rows of pins are 0.7 in apart: c and h, b and g, d and i ...


class Layout(NamedTuple):
    """How the Pico sits on their breadboard: enough to put every pin in a hole."""

    usb: str  # a key of USB_WAYS
    chip: str  # "up" - the RP2040 facing you - or "down"
    pin1_column: int
    pin1_row: str  # the row letter pin 1 is in; pins 21-40 are in the row 0.7 in across
    count: int  # +1 if column numbers grow going away from the USB end, -1 if they shrink


def _across(row: str) -> int:
    """A row's distance from row a in pitches: the centre channel is three pitches wide."""
    i = ROWS.index(row)
    return i + (2 if i >= 5 else 0)


def partner(row: str) -> str:
    """The row across the channel that is 0.7 in from *row* - where the other side's pins go."""
    i = ROWS.index(row)
    return ROWS[i + 5] if i < 5 else ROWS[i - 5]


def axes(usb: str, chip: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """Two screen steps: away from the USB end, and toward the side pins 1-20 are on.

    Chip up with the USB at the top, pin 1 is top left: the pin-1 side is a quarter turn
    clockwise from "away from the USB". Turned chip down, the board is mirrored along its length.
    """
    ux, uy = USB_WAYS[usb]
    away = (-ux, -uy)
    side = (-away[1], away[0])
    if chip == "down":
        side = (-side[0], -side[1])
    return away, side


def pin_place(n: int, usb: str, chip: str) -> tuple[int, int]:
    """Where pin *n* is on the screen, in pitches from pin 1 - the board as they see it."""
    along, side = (n - 1, 0) if n <= 20 else (40 - n, 1)
    (ax, ay), (sx, sy) = axes(usb, chip)
    across = -PICO_SPAN * side  # pins 21-40 are the far side from pin 1's
    return along * ax + across * sx, along * ay + across * sy


def printed_count(usb: str, chip: str, pin1_row: str) -> int:
    """Which way the column numbers run, from how a breadboard is printed.

    A breadboard is printed one way round: with the numbers reading upright, 1 is on the left
    and row j is at the top. Turning it does not change that, so knowing which side of the
    channel pin 1's row is on fixes which way the numbers run.
    """
    away, side = axes(usb, chip)
    letters = side if ROWS.index(pin1_row) >= 5 else (-side[0], -side[1])  # a -> j, on screen
    numbers = (-letters[1], letters[0])  # 1 -> 60: a quarter turn clockwise from a -> j
    return 1 if numbers[0] * away[0] + numbers[1] * away[1] > 0 else -1


def pin_hole(n: int, layout: Layout) -> tuple[int, str]:
    """The breadboard hole pin *n* is in: (column, row)."""
    along = n - 1 if n <= 20 else 40 - n
    row = layout.pin1_row if n <= 20 else partner(layout.pin1_row)
    return layout.pin1_column + layout.count * along, row


def make_layout(usb: str, chip: str, pin1_column: int, pin1_row: str = "",
                pin20_column: int | None = None) -> Layout:
    """A layout from what they said, or ValueError saying what does not fit."""
    usb, chip = str(usb).strip().lower(), str(chip).strip().lower()
    if usb not in USB_WAYS:
        raise ValueError(f"usb must be one of {', '.join(USB_WAYS)}")
    if chip not in ("up", "down"):
        raise ValueError("chip must be up or down")
    first = int(pin1_column)
    row = str(pin1_row or "").strip().lower()[:1]
    if row and row not in ROWS:
        raise ValueError("pin1_row is a breadboard row letter, a to j")
    if pin20_column is not None:
        if abs(int(pin20_column) - first) != 19:
            raise ValueError("pin 1 and pin 20 are 19 columns apart on any breadboard - "
                             "check the two numbers with them")
        count = 1 if int(pin20_column) > first else -1
    elif row:
        count = printed_count(usb, chip, row)
    else:  # the usual straddle, c and h: the column pin 1 is in says which way the rest run
        count = -1 if first + 19 > BOARD_COLUMNS else 1 if first - 19 < 1 else \
            printed_count(usb, chip, "h")
    if not row:  # whichever of c and h a printed board puts on pin 1's side for that count
        row = "h" if printed_count(usb, chip, "h") == count else "c"
    layout = Layout(usb, chip, int(pin1_column), row, count)
    if min(pin_hole(20, layout)[0], layout.pin1_column) < 1:
        raise ValueError(f"with pin 1 in column {pin1_column}, pin 20 would be off the board - "
                         "check the column, or pass pin20_column")
    return layout


def describe(layout: Layout) -> str:
    return (f"{USB_WORDS[layout.usb]}, chip {layout.chip} · pin 1 in column "
            f"{layout.pin1_column}, row {layout.pin1_row}")


# ---- naming a hole, or a pin by what is on it

SHORT = {"GP26/ADC0": "GP26", "GP27/ADC1": "GP27", "GP28/ADC2": "GP28", "ADC_VREF": "VREF",
         "3V3(OUT)": "3V3"}


def _names(n: int) -> set[str]:
    full = PINS[n - 1].upper()
    return {full, *full.split("/"), SHORT.get(PINS[n - 1], full).upper()}


def _hole(text: str) -> tuple[int, str] | None:
    t = str(text).strip().lower().replace("column", "").replace("row", "")
    if m := re.fullmatch(r"\s*(\d+)\s*[-, ]*\s*([a-j])\s*", t):
        return int(m.group(1)), m.group(2)
    if m := re.fullmatch(r"\s*([a-j])\s*[-, ]*\s*(\d+)\s*", t):
        return int(m.group(2)), m.group(1)
    return None


def _pin(text: str, near: tuple[int, str] | None, layout: Layout) -> int | None:
    """The pin *text* names - "GP15", "pin 20", "GND" (the one nearest *near*) - or None."""
    t = re.sub(r"\s+", "", str(text).upper())
    if m := re.fullmatch(r"(?:PIN|PHYSICAL)(\d+)", t):
        return int(m.group(1)) if 1 <= int(m.group(1)) <= 40 else None
    if m := re.fullmatch(r"(.+?)(?:\(?PIN(\d+)\)?)", t):  # "GND pin 18"
        n = int(m.group(2))
        return n if 1 <= n <= 40 and m.group(1).strip("·-,") in _names(n) else None
    found = [n for n in range(1, 41) if t in _names(n)]
    if len(found) <= 1 or near is None:
        return found[0] if found else None

    def distance(n: int) -> float:
        col, row = pin_hole(n, layout)
        return abs(col - near[0]) + abs(_across(row) - _across(near[1]))

    return min(found, key=distance)


def blocked(hole: tuple[int, str], layout: Layout) -> str:
    """Why a lead cannot go in *hole*, or "" if it can."""
    col, row = hole
    for n in range(1, 41):
        if pin_hole(n, layout) == hole:
            return f"that is where pin {n} ({PINS[n - 1]}) sits"
    cols = sorted((layout.pin1_column, pin_hole(20, layout)[0]))
    rows = sorted((_across(layout.pin1_row), _across(partner(layout.pin1_row))))
    if cols[0] <= col <= cols[1] and rows[0] < _across(row) < rows[1]:
        return "that hole is under the Pico"
    return ""


def free_hole(n: int, layout: Layout, taken: set[tuple[int, str]]) -> tuple[int, str] | None:
    """The outermost free hole in the strip pin *n* is plugged into."""
    col, row = pin_hole(n, layout)
    i = ROWS.index(row)
    outward = range(i + 1, 10) if i >= 5 else range(i - 1, -1, -1)
    for j in reversed(list(outward)):
        if (col, ROWS[j]) not in taken:
            return col, ROWS[j]
    return None


def strip_pin(hole: tuple[int, str], layout: Layout) -> int | None:
    """The pin sharing *hole*'s five-hole strip, if one does."""
    half = ROWS.index(hole[1]) >= 5
    for n in range(1, 41):
        col, row = pin_hole(n, layout)
        if col == hole[0] and (ROWS.index(row) >= 5) == half:
            return n
    return None


def hole_words(hole: tuple[int, str], layout: Layout) -> str:
    words = f"column {hole[0]}, row {hole[1]}"
    if (n := strip_pin(hole, layout)) is not None:
        words += f" ({SHORT.get(PINS[n - 1], PINS[n - 1])}, pin {n})"
    return words


class Part(NamedTuple):
    kind: str  # wire | resistor | led | part
    ends: tuple[tuple[int, str], tuple[int, str]]
    label: str
    colour: str


KINDS = ("wire", "resistor", "led", "part")
PALETTE = ("#ffcc00", "#ff4040", "#4db8ff", "#5ad17a", "#e070ff", "#ff9a3c")
WIRES = {"black": "#1a1a1a", "red": "#d02020", "blue": "#2060d0", "green": "#20a040",
         "yellow": "#e0c000", "orange": "#f08020", "white": "#f4f4f4", "purple": "#8040c0",
         "brown": "#7a4a20", "grey": "#808080", "gray": "#808080"}
MAX_PARTS = 6


def resolve(raw: Any, layout: Layout) -> list[Part]:
    """The model's parts, every end turned into a real, free hole, or ValueError."""
    parts: list[Part] = []
    taken: set[tuple[int, str]] = set()
    for item in list(raw or [])[:MAX_PARTS]:
        if not isinstance(item, dict):
            raise ValueError("each part is an object with kind, from and to")
        kind = str(item.get("kind") or "part").lower()
        kind = kind if kind in KINDS else "part"
        texts = [str(item.get("from") or ""), str(item.get("to") or "")]
        holes: list[tuple[int, str] | None] = [_hole(t) for t in texts]
        for i, t in enumerate(texts):
            if holes[i] is None:
                n = _pin(t, holes[1 - i], layout)
                if n is None:
                    raise ValueError(f"{t!r} is neither a hole like \"41j\" nor a Pico pin")
                holes[i] = free_hole(n, layout, taken)
                if holes[i] is None:
                    raise ValueError(f"no free hole left beside pin {n}")
        cols = sorted((layout.pin1_column, pin_hole(20, layout)[0]))
        for i, hole in enumerate(holes):
            assert hole is not None
            where = f"column {hole[0]}, row {hole[1]}"
            if hole[0] < 1:
                raise ValueError(f"column {hole[0]} is off the board")
            if why := blocked(hole, layout):
                raise ValueError(f"nothing can go in {where}: {why}")
            if hole in taken:
                raise ValueError(f"{where} already has a lead in it - one lead per hole; the "
                                 f"other holes in column {hole[0]} on that side are joined to it")
            if _hole(texts[i]) is not None and (n := strip_pin(hole, layout)) is not None:
                raise ValueError(f"{where} is joined to {PINS[n - 1]} (pin {n}). To connect to "
                                 "a pin, name the pin; otherwise use a column outside "
                                 f"{cols[0]}-{cols[1]}")
            taken.add(hole)
        colour = WIRES.get(str(item.get("colour") or item.get("color") or "").lower(), "")
        parts.append(Part(kind, (holes[0], holes[1]), str(item.get("label") or "")[:40], colour))
    return parts


def legend(part: Part, layout: Layout) -> tuple[str, str]:
    """What a part is, and where its leads go - for the picture and for the model alike."""
    a, b = (hole_words(h, layout) for h in part.ends)
    if part.kind == "led":
        return part.label or "LED", f"long leg (+) {a}  ·  short leg (−) {b}"
    if part.kind == "wire":
        colour = next((k for k, v in WIRES.items() if v == part.colour), "")
        name = part.label or (f"{colour} wire" if colour else "wire")
        return name, f"{a}  →  {b}"
    return part.label or part.kind, f"{a}  →  {b}"


# ---- the picture

W, H = 1600, 960
BG, BOARD, CHANNEL, HOLE = "#101418", "#eceae4", "#d6d3cb", "#3a3a3a"
PCB, PCB_EDGE, GOLD = "#1f7a3a", "#155c2a", "#d9b44a"
FONTS = (
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/System/Library/Fonts/Supplemental/Arial.ttf",
     "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    ("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
     "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
)


@functools.lru_cache(maxsize=32)
def _font(size: int, bold: bool = False) -> Any:
    from PIL import ImageFont

    for pair in FONTS:
        if Path(pair[bold]).exists():
            return ImageFont.truetype(pair[bold], size)
    return ImageFont.load_default(size)


def render(layout: Layout, parts: list[Part], pins: tuple[int, ...] = ()) -> bytes:
    """The Pico on their breadboard as they see it, every lead in its hole. JPEG bytes.

    Nothing in it is drawn by a model: the pins come from ``PINS``, the holes from
    :func:`pin_hole`, so the same arguments give the same picture, byte for byte.
    """
    from PIL import Image, ImageDraw

    away, side = axes(layout.usb, layout.chip)
    vertical = away[0] == 0
    mid = (_across(layout.pin1_row) + _across(partner(layout.pin1_row))) / 2
    toward = 1 if ROWS.index(layout.pin1_row) >= 5 else -1  # letters grow toward pin 1's side

    def local(hole: tuple[int, str]) -> tuple[float, float]:
        return (hole[0] - layout.pin1_column) * layout.count, (_across(hole[1]) - mid) * toward

    ends = [local(h) for p in parts for h in p.ends]
    lo = min([-1.0, *(a for a, _ in ends)])
    lo = max(lo, -(layout.pin1_column - 1) if layout.count > 0 else lo)
    hi = max([19.0 + (6 if vertical else 14), *(a + 1 for a, _ in ends)])
    edges = [(_across(r) - mid) * toward for r in "aj"]
    c0, c1 = min(edges) - 3.0, max(edges) + 3.0
    a0, a1 = lo - 1.2, hi + 1.2

    title_h, rows_of_legend = 90, len(parts) + len([n for n in pins if n])
    if vertical:
        pitch = min(40.0, (H - title_h - 40) / (a1 - a0), (W * 0.5) / (c1 - c0))
    else:
        pitch = min(40.0, (W - 120) / (a1 - a0),
                    (H - title_h - 40 - 50 * max(rows_of_legend, 1)) / (c1 - c0))

    def raw(a: float, c: float) -> tuple[float, float]:
        return (pitch * (a * away[0] + c * side[0]), pitch * (a * away[1] + c * side[1]))

    corners = [raw(a, c) for a in (a0, a1) for c in (c0, c1)]
    bx0, by0 = min(x for x, _ in corners), min(y for _, y in corners)
    bx1, by1 = max(x for x, _ in corners), max(y for _, y in corners)
    if vertical:
        ox, oy = 60 - bx0, title_h - by0
    else:
        ox, oy = (W - (bx1 - bx0)) / 2 - bx0, title_h - by0

    def pt(a: float, c: float) -> tuple[float, float]:
        x, y = raw(a, c)
        return round(x + ox, 1), round(y + oy, 1)

    def box(a_0: float, c_0: float, a_1: float, c_1: float) -> tuple[float, float, float, float]:
        (x0, y0), (x1, y1) = pt(a_0, c_0), pt(a_1, c_1)
        return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    s = pitch / 40  # everything below was sized at the spike's 40 px pitch
    fs = lambda size: max(9, round(size * s))  # noqa: E731

    # the breadboard
    d.rounded_rectangle(box(a0, c0, a1, c1), round(18 * s), fill=BOARD)
    ce, cf = ((_across(r) - mid) * toward for r in "ef")
    d.rectangle(box(a0, min(ce, cf) + 0.3, a1, max(ce, cf) - 0.3), fill=CHANNEL)
    columns = [layout.pin1_column + layout.count * k for k in range(int(round(lo)), int(hi) + 1)]
    for col in columns:
        for row in ROWS:
            x, y = pt(*local((col, row)))
            r = 6 * s
            d.rectangle((x - r, y - r, x + r, y + r), fill=HOLE)
        if col % 5 == 0 or col == 1:
            a = (col - layout.pin1_column) * layout.count
            for c in (c0 + 0.7, c1 - 0.7):
                d.text(pt(a, c), str(col), font=_font(fs(20)), fill="#555", anchor="mm")
    for row in ROWS:
        c = (_across(row) - mid) * toward
        for a in (a0 + 0.55, a1 - 0.55):
            d.text(pt(a, c), row, font=_font(fs(20)), fill="#555", anchor="mm")

    # the Pico
    reach = PICO_SPAN / 2 + 0.75
    d.rounded_rectangle(box(-0.7, -reach, 19.7, reach), round(10 * s), fill=PCB,
                        outline=PCB_EDGE, width=max(1, round(3 * s)))
    d.rounded_rectangle(box(-1.6, -0.95, -0.35, 0.95), round(6 * s), fill="#b8bcc2")
    d.text(pt(-2.7, 0), "USB", font=_font(fs(22), True), fill="#b8bcc2", anchor="mm")
    if layout.chip == "up":
        d.rectangle(box(8.4, -1.1, 10.6, 1.1), fill="#222")
        d.text(pt(9.5, 0), "RP2040", font=_font(fs(16)), fill="#888", anchor="mm")
        d.text(pt(15.5, 0), "Pico" if vertical else "Raspberry Pi Pico", font=_font(fs(22)),
               fill="#cfe8d4", anchor="mm")
    else:
        d.text(pt(9.5, 0), "chip side down" if vertical else "Raspberry Pi Pico, chip side down",
               font=_font(fs(22)),
               fill="#cfe8d4", anchor="mm")

    lit: dict[int, str] = {}
    for i, part in enumerate(parts):
        for hole in part.ends:
            if (n := strip_pin(hole, layout)) is not None:
                lit.setdefault(n, PALETTE[i % len(PALETTE)])
    for k, n in enumerate(p for p in pins if p not in lit):
        lit[n] = PALETTE[(len(parts) + k) % len(PALETTE)]
    for n in range(1, 41):
        a, c = local(pin_hole(n, layout))
        x, y = pt(a, c)
        r = (12 if n in lit else 8) * s
        d.ellipse((x - r, y - r, x + r, y + r), fill=lit.get(n, GOLD), outline="#222",
                  width=max(1, round(2 * s)))
        inward = -0.55 if c > 0 else 0.55
        lx, ly = pt(a, c + inward)
        if vertical:
            anchor = "rm" if lx < x else "lm"
            lx, ly = pt(a, c + inward * 0.6)
        else:
            anchor = "mm"
        d.text((lx, ly), SHORT.get(PINS[n - 1], PINS[n - 1]), font=_font(fs(13), n in lit),
               fill="#fff" if n in lit else "#cfe8d4", anchor=anchor)

    # the parts, each lead in its hole, bent out past the edge when both ends share a row
    stacked: dict[float, int] = {}
    for i, part in enumerate(parts):
        colour = PALETTE[i % len(PALETTE)]
        (pa, pc), (qa, qc) = (local(h) for h in part.ends)
        if part.kind == "led":  # stands up off the board: legs straight up to the bulb
            lift = (-1 if (pc + qc) > 0 else 1) * (1.1 if pc == qc else 0)
            top = pt((pa + qa) / 2, (pc + qc) / 2 + lift)
            path = [pt(pa, pc), top, pt(qa, qc)]
            d.line(path[:2], fill="#aaa", width=max(2, round(4 * s)))
            d.line(path[1:], fill="#aaa", width=max(2, round(4 * s)))
            r = 22 * s
            d.ellipse((top[0] - r, top[1] - r, top[0] + r, top[1] + r), fill="#ff3030",
                      outline="#a00", width=max(1, round(3 * s)))
            d.text(top, "+", font=_font(fs(18), True), fill="#fff", anchor="mm")
            path = [path[0], path[2]]
        elif pc == qc:
            out = 1 if pc > 0 else -1
            level = stacked.get(pc, 0)
            stacked[pc] = level + 1
            bend = pc + out * (0.9 + 0.75 * level)
            path = [pt(pa, pc), pt(pa, bend), pt(qa, bend), pt(qa, qc)]
        else:
            path = [pt(pa, pc), pt(qa, qc)]
        m0, m1 = path[len(path) // 2 - 1], path[len(path) // 2]
        mx, my = (m0[0] + m1[0]) / 2, (m0[1] + m1[1]) / 2
        if part.kind == "led":
            pass
        elif part.kind == "wire":
            d.line(path, fill=part.colour or "#1a1a1a", width=max(2, round(7 * s)), joint="curve")
        else:
            d.line(path, fill="#999", width=max(2, round(5 * s)), joint="curve")
            if part.kind == "resistor":
                dx, dy = m1[0] - m0[0], m1[1] - m0[1]
                horizontal = abs(dx) >= abs(dy)
                hw, hh = (38 * s, 12 * s) if horizontal else (12 * s, 38 * s)
                d.rounded_rectangle((mx - hw, my - hh, mx + hw, my + hh), round(8 * s),
                                    fill="#d8b98a", outline="#8a6a3a", width=max(1, round(2 * s)))
                for k, band in enumerate(("#ff8c00", "#ff8c00", "#6b3a1a", "#c8a24a")):
                    o = (-22 + k * 15) * s
                    bw = 3.5 * s
                    d.rectangle((mx + o - bw, my - hh + 1, mx + o + bw, my + hh - 1) if horizontal
                                else (mx - hw + 1, my + o - bw, mx + hw - 1, my + o + bw),
                                fill=band)
            else:
                w = max(40 * s, d.textlength(part.label or "part", font=_font(fs(16))) / 2 + 8)
                d.rounded_rectangle((mx - w, my - 14 * s, mx + w, my + 14 * s), round(6 * s),
                                    fill="#333", outline="#888")
                d.text((mx, my), part.label or "part", font=_font(fs(16)), fill="#fff",
                       anchor="mm")
        for x, y in (path[0], path[-1]):
            r = 9 * s
            d.ellipse((x - r, y - r, x + r, y + r), fill=colour, outline="#222")

    # the legend: every lead as a hole
    lines = [(PALETTE[i % len(PALETTE)], *legend(p, layout)) for i, p in enumerate(parts)]
    for n in pins:
        if n in lit and all(strip_pin(h, layout) != n for p in parts for h in p.ends):
            col, row = pin_hole(n, layout)
            lines.append((lit[n], f"{SHORT.get(PINS[n - 1], PINS[n - 1])} · pin {n}",
                          f"column {col}, row {row}"))
    if vertical:
        x, y, step = bx1 + ox + 60, title_h + 20, 96
    else:
        x, y, step = 110, by1 + oy + 28, 48
    for colour, name, where in lines:
        d.ellipse((x - 40, y + 6, x - 16, y + 30), fill=colour)
        d.text((x, y), name, font=_font(26, True), fill="#fff")
        if vertical:
            d.text((x, y + 40), where, font=_font(22), fill="#aab")
        else:
            d.text((x + 330, y + 3), where, font=_font(24), fill="#aab")
        y += step
    d.text((W / 2, 40), describe(layout), font=_font(30, True), fill="#fff", anchor="mm")

    out = io.BytesIO()
    img.save(out, format="JPEG", quality=90)
    return out.getvalue()


# One layout per session: asked once, then every later picture uses it. Keyed by the session's
# folder, so next week's session asks again rather than trusting where the Pico used to sit.
_layouts: dict[str, Layout] = {}


def _session_key() -> str:
    from .. import session

    live = session.current()
    return str(live.dir) if live is not None else ""


def _show(args: dict[str, Any], device: Any) -> dict[str, Any]:
    key = _session_key()
    try:
        if args.get("usb") or args.get("chip") or args.get("pin1_column"):
            if not (args.get("usb") and args.get("chip") and args.get("pin1_column")):
                raise ValueError("usb, chip and pin1_column go together")
            _layouts[key] = make_layout(args["usb"], args["chip"], int(args["pin1_column"]),
                                        str(args.get("pin1_row") or ""),
                                        args.get("pin20_column"))
        layout = _layouts.get(key)
        if layout is None:
            raise ValueError("you do not know yet how the Pico sits: ask which way the USB "
                             "points, chip up or down, and which column pin 1 is in")
        parts = resolve(args.get("parts"), layout)
        pins = tuple(n for t in args.get("pins") or [] if (n := _pin(t, None, layout)))
    except (ValueError, TypeError) as exc:
        return {"ok": False, "error": str(exc)}
    picture = render(layout, parts, pins)
    title = "Pico wiring: " + (", ".join(legend(p, layout)[0] for p in parts) or "pinout")
    shown = _put_up(picture, title)
    leads = [f"{name}: {where}" for name, where in (legend(p, layout) for p in parts)]
    leads += [f"{SHORT.get(PINS[n - 1], PINS[n - 1])} (pin {n}): column {c}, row {r}"
              for n in pins for c, r in [pin_hole(n, layout)]]
    return {"ok": True, "shown": shown, "layout": describe(layout), "leads": leads,
            "note": ("On their screen now. Walk them through it one lead at a time by hole, "
                     "as listed, and wait for each." if shown else
                     "There is no screen to show it on; give them the holes out loud.")}


def _put_up(jpeg: bytes, title: str) -> bool:
    """Keep the picture with the session's photos, and put it on the panel. Whether it showed."""
    from .. import imagine, panel, session

    live = session.current()
    kept = None
    if live is not None:
        with contextlib.suppress(OSError):
            kept = imagine.write(jpeg, title, live.photos_dir, role="drawn")
    shown = panel.offer_image(imagine.for_panel(jpeg), title, announce=True) and panel.show()
    if kept is not None and kept.path is not None:
        session.note("photo", by="drawn", request=kept.request,
                     file=f"{session.PHOTOS}/{kept.path.name}", bytes=kept.bytes, panel=shown)
    return shown


# ---------------------------------------------------------------- the tools


def _program(args: dict[str, Any], device: Any) -> dict[str, Any]:
    code, name = str(args.get("code") or ""), str(args.get("name") or "program")
    result = board().program(code)
    keep(code, name)
    return result


def _run(args: dict[str, Any], device: Any) -> dict[str, Any]:
    return board().run(str(args.get("code") or ""))


def _output(args: dict[str, Any], device: Any) -> dict[str, Any]:
    return board().output()


PROGRAM = {
    "type": "function",
    "name": "pico_program",
    "description": (
        "Put a MicroPython program on the Pico and start it: stops what is running, saves this "
        "as main.py (so it runs again every time the Pico powers up), resets, and returns what "
        "it printed in its first two seconds, with the traceback if it crashed. Use it for "
        "anything that should keep running - blink, fade, read a sensor in a loop - and to "
        "change it (\"faster\" is a new program, not a replug). Send the whole program every "
        "time. Empty code stops it. Not for a quick question about the board: that is pico_run."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "the whole MicroPython program"},
            "name": {"type": "string", "description": "two or three words, e.g. \"led fade\""},
        },
        "required": ["code", "name"],
    },
}
RUN = {
    "type": "function",
    "name": "pico_run",
    "description": (
        "Run a short MicroPython snippet on the Pico once and get what it printed - is GP15 "
        "high, scan the I2C bus, read ADC0. Stopped after ten seconds with what it printed so "
        "far. The saved program starts again afterwards. Call it unasked whenever the board can "
        "answer instead of them: they just finished wiring, something doesn't work, or you "
        "would otherwise ask which pins or address - scan every pin pair in one snippet rather "
        "than ask. Not for something that should keep running: that is pico_program."
    ),
    "parameters": {
        "type": "object",
        "properties": {"code": {"type": "string", "description": "MicroPython to run once"}},
        "required": ["code"],
    },
}
OUTPUT = {
    "type": "function",
    "name": "pico_output",
    "description": (
        "What the program running on the Pico has printed since you last looked (up to the "
        "last 200 lines), and whether it is still running, finished, or crashed with a "
        "traceback. Use it when they ask what it is doing or why it stopped."
    ),
    "parameters": {"type": "object", "properties": {}},
}
SHOW = {
    "type": "function",
    "name": "pico_show",
    "description": (
        "Put a picture on their screen of the Pico on their breadboard exactly as it sits in "
        "front of them - every pin in its real hole - with the parts you want wired lit up and "
        "each lead in a hole. This is the only way to show Pico wiring: never use draw for it, "
        "because a drawn Pico puts the pins in the wrong places. Use it whenever they are about "
        "to wire something to the Pico or ask where something goes. Asked where a wire goes "
        "before you know how it sits, answer first with the pin and where it is on the Pico - "
        "\"GP15, that's pin 20, last on the left counting from the USB end\" - then ask how it sits"
        " so you can show the holes. To show it you need which way the USB points from where they"
        " stand, chip up or down, and the column number pin 1 (the pin nearest the USB, GP0) is "
        "in - ask for all three in one question, or take_a_look. After that leave them out: the "
        "layout is kept for the session. It answers with every lead as a hole; guide them by "
        "those holes, one lead at a time (\"the resistor goes from column 41, row j to column 37, "
        "row j\")."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "usb": {"type": "string", "enum": list(USB_WAYS),
                    "description": "where the USB socket points, as they look at the bench: "
                                   "up is away from them"},
            "chip": {"type": "string", "enum": ["up", "down"],
                     "description": "up if the side with the black RP2040 chip faces them"},
            "pin1_column": {"type": "integer",
                            "description": "the breadboard column number pin 1 is in"},
            "pin1_row": {"type": "string",
                         "description": "optional: the row letter pin 1 is in, if they said"},
            "pin20_column": {"type": "integer",
                             "description": "optional: the column of pin 20 at the far end, "
                                            "if they said - settles which way the numbers run"},
            "parts": {
                "type": "array",
                "description": "what to wire, up to six. Each end is a hole like \"37j\" or a "
                               "Pico pin like \"GP15\", \"3V3\" or \"GND\" (a free hole beside "
                               "that pin is picked; GND picks the nearest ground). A hole must "
                               "be in a column clear of the Pico's own, and holds one lead: two "
                               "leads that meet go in the same column, different rows.",
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": list(KINDS)},
                        "from": {"type": "string",
                                 "description": "one end; for an LED, the long leg (+)"},
                        "to": {"type": "string", "description": "the other end"},
                        "label": {"type": "string",
                                  "description": "e.g. \"330 ohm resistor\", \"button\""},
                        "colour": {"type": "string",
                                   "description": "a wire's real colour, e.g. black"},
                    },
                    "required": ["kind", "from", "to"],
                },
            },
            "pins": {"type": "array", "items": {"type": "string"},
                     "description": "Pico pins to light up with nothing wired to them yet"},
        },
    },
}

EXTENSION = Extension(
    name="Pico",
    usb_ids=(BOOT_ID, MICROPYTHON_ID),
    category="other",
    instructions=INSTRUCTIONS,
    rejoin_s=FLASH_WAIT_S,
    tools=(
        Tool(schema=PROGRAM, run=_program, caption="programming the Pico…"),
        Tool(schema=RUN, run=_run, caption="asking the Pico…"),
        Tool(schema=OUTPUT, run=_output, caption="reading the Pico…"),
        Tool(schema=SHOW, run=_show, caption="drawing the Pico…"),
    ),
)
