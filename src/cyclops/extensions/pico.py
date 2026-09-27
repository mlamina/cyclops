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
import glob
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
from typing import Any

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
    "Pinout, USB end at the top, chip side up. " + _pinout() + "\n"
    "Wiring: always give both names and where the pin is - \"GP15, that's pin 20, last on the "
    "left counting from the USB end\". One wire at a time, and wait for them. 3.3 V logic: "
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
        "far. The saved program starts again afterwards. Not for something that should keep "
        "running: that is pico_program."
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
    ),
)
