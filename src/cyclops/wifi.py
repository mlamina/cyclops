"""Which Wi-Fi the box is on, and changing it from the panel: nmcli, and the picker's state.

The box has no keyboard and no taskbar, so a network it cannot reach used to be a black screen
after the button. This owns the two halves of fixing that which are not drawing: talking to
NetworkManager, and remembering where somebody is in the picker - which page of networks, which
one they tapped, what they have typed so far.

It must never be called from the render loop. Every nmcli here can take seconds (a rescan is
three or four, a join up to :data:`JOIN_WAIT_S`), so the kiosk runs them on a worker thread and
only ever reads the :class:`Picker` back.

``sudo -n nmcli`` first, bare ``nmcli`` after it - the two-route idiom from
:func:`cyclops.power.take_down`. Unprivileged nmcli can read the cached list but polkit refuses
it a scan, a join and the radio; the kiosk's user has passwordless sudo on the Pi.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field

JOIN_WAIT_S = 30  # nmcli's own --wait: how long a join may take before it is a failure
TIMEOUT_S = JOIN_WAIT_S + 10.0  # ...and ours round the whole process, for an nmcli that hangs
READ_TIMEOUT_S = 8.0  # a status or a cached list, which never asks the radio for anything
SETTLE_S = 30  # at boot, how long NetworkManager gets to finish autoconnecting before we ask
PRIORITY = 100  # a network joined from the panel is the one it asked for, so it wins next boot
MIN_PSK = 8  # WPA's own floor; anything shorter is refused by nmcli before it tries

PAGE = 6  # networks on one page of the picker

LIST, KEYBOARD = "list", "keyboard"
JOINED, WRONG_PASSWORD, FAILED = "joined", "wrong-password", "failed"

# What nmcli says, in the words it says it, when the key was the problem rather than the air.
_WRONG_KEY = ("secrets were required", "802-1x", "psk", "password", "authentication")


@dataclass(frozen=True)
class Network:
    ssid: str
    signal: int  # 0..100, as nmcli reports it
    secure: bool
    saved: bool = False
    active: bool = False


# ---------------------------------------------------------------- parsing, pure


def split_terse(line: str) -> list[str]:
    r"""One line of ``nmcli -t`` output as its fields. ``\:`` is a colon inside a field."""
    fields, current, escaped = [], [], False
    for char in line:
        if escaped:
            current.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == ":":
            fields.append("".join(current))
            current = []
        else:
            current.append(char)
    fields.append("".join(current))
    return fields


def parse_scan(text: str, saved: set[str]) -> list[Network]:
    """``nmcli -t -f IN-USE,SSID,SIGNAL,SECURITY device wifi list`` as the picker's rows.

    One row per SSID - a dual-band router is one network to the person choosing, not two - and
    the strongest reading kept. Hidden networks are dropped: there is nothing to tap. The one in
    use first, then the saved ones, then everything else; by signal inside each.
    """
    best: dict[str, Network] = {}
    for line in text.splitlines():
        parts = split_terse(line)
        if len(parts) < 4:
            continue
        in_use, ssid, signal, security = parts[:4]
        ssid = ssid.strip()
        if not ssid or ssid == "--":
            continue
        try:
            strength = int(signal)
        except ValueError:
            strength = 0
        active = in_use.strip() == "*"
        seen = best.get(ssid)
        net = Network(
            ssid, max(strength, seen.signal if seen else 0),
            security.strip() not in ("", "--"), ssid in saved, active or bool(seen and seen.active),
        )
        best[ssid] = net
    return sorted(best.values(), key=lambda n: (not n.active, not n.saved, -n.signal, n.ssid))


def parse_saved(text: str) -> set[str]:
    """``nmcli -t -f NAME,TYPE connection show`` as the names of the Wi-Fi profiles.

    A profile's name is its SSID for every network this box joined itself - :func:`join` names
    it that way, and so does ``deploy/install-tether.sh``.
    """
    names = set()
    for line in text.splitlines():
        parts = split_terse(line)
        if len(parts) >= 2 and parts[1] == "802-11-wireless":
            names.add(parts[0])
    return names


def is_online(general: str) -> bool | None:
    """``nmcli -t -f STATE general`` as whether the box is on a network. None when unknown."""
    state = general.strip().splitlines()[0].strip() if general.strip() else ""
    if not state:
        return None
    return state == "connected"


def classify(returncode: int, output: str) -> str:
    """What a join came to: :data:`JOINED`, :data:`WRONG_PASSWORD` or :data:`FAILED`."""
    if returncode == 0:
        return JOINED
    said = output.lower()
    return WRONG_PASSWORD if any(word in said for word in _WRONG_KEY) else FAILED


# ---------------------------------------------------------------- nmcli, the edge


def _nmcli(args: list[str], timeout: float, privileged: bool = True) -> tuple[int, str]:
    """Run nmcli, with sudo first when it may need it. (127, "") if there is no nmcli at all."""
    routes = (["sudo", "-n", "nmcli"], ["nmcli"]) if privileged else (["nmcli"],)
    last = (127, "")
    for route in routes:
        try:
            done = subprocess.run(route + args, capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.SubprocessError):
            continue
        last = (done.returncode, done.stdout + done.stderr)
        if done.returncode == 0:
            return last
        if route[0] == "sudo" and done.stderr.startswith("sudo:"):
            continue  # sudo would not have us; the plain route may still read
        return last
    return last


def online() -> bool | None:
    """Whether NetworkManager says the box is connected. None off a Pi, where nobody can say."""
    code, out = _nmcli(["-t", "-f", "STATE", "general"], READ_TIMEOUT_S, privileged=False)
    return is_online(out) if code == 0 else None


def settle() -> None:
    """Wait for NetworkManager to finish its boot-time autoconnect, or give up. Never raises."""
    try:
        subprocess.run(["nm-online", "-s", "-q", "-t", str(SETTLE_S)], timeout=SETTLE_S + 5)
    except (OSError, subprocess.SubprocessError):
        pass


def scan() -> list[Network]:
    """What is on the air now, a rescan's worth of seconds from now."""
    # A radio somebody switched off lists nothing, and there is no other switch on this box.
    _nmcli(["radio", "wifi", "on"], READ_TIMEOUT_S)
    _, saved = _nmcli(["-t", "-f", "NAME,TYPE", "connection", "show"], READ_TIMEOUT_S)
    code, out = _nmcli(
        ["-t", "-f", "IN-USE,SSID,SIGNAL,SECURITY", "device", "wifi", "list", "--rescan", "yes"],
        TIMEOUT_S,
    )
    return parse_scan(out, parse_saved(saved)) if code == 0 else []


def join(net: Network, password: str = "") -> str:
    """Join *net*: its saved profile if it has one and no password was typed, else a new one.

    A failed new profile is deleted rather than left behind: NetworkManager would otherwise keep
    retrying a wrong password on every boot, and the picker would list it as saved.
    """
    if net.saved and not password:
        code, out = _nmcli(["--wait", str(JOIN_WAIT_S), "connection", "up", "id", net.ssid],
                           TIMEOUT_S)
        return classify(code, out)
    _nmcli(["connection", "delete", "id", net.ssid], READ_TIMEOUT_S)
    command = ["--wait", str(JOIN_WAIT_S), "device", "wifi", "connect", net.ssid]
    if password:
        command += ["password", password]
    code, out = _nmcli(command + ["name", net.ssid], TIMEOUT_S)
    outcome = classify(code, out)
    if outcome != JOINED:
        _nmcli(["connection", "delete", "id", net.ssid], READ_TIMEOUT_S)
        return outcome
    _nmcli(["connection", "modify", "id", net.ssid,
            "connection.autoconnect", "yes", "connection.autoconnect-priority", str(PRIORITY)],
           READ_TIMEOUT_S)
    return JOINED


# ---------------------------------------------------------------- the picker


@dataclass
class Picker:
    """Where somebody is in the picker. The render loop reads it; taps and the worker write it.

    ``busy`` is the worker's: a scan or a join is running, and the panel says so rather than
    taking a second one. ``status`` is the one line of news - scanning, joining, a wrong password.
    """

    screen: str = LIST
    networks: list[Network] = field(default_factory=list)
    page: int = 0
    chosen: Network | None = None
    typed: str = ""
    shift: bool = False
    symbols: bool = False
    show: bool = False
    status: str = ""
    busy: bool = False

    def shown(self) -> list[Network]:
        return self.networks[self.page * PAGE : (self.page + 1) * PAGE]

    def more(self) -> bool:
        return len(self.networks) > PAGE

    def next_page(self) -> None:
        pages = max(1, -(-len(self.networks) // PAGE))
        self.page = (self.page + 1) % pages

    def choose(self, net: Network) -> None:
        """A network tapped on the list: to the keyboard, with nothing typed yet."""
        self.chosen = net
        self.screen = KEYBOARD
        self.typed = ""
        self.shift = self.symbols = self.show = False
        self.status = ""

    def back(self) -> None:
        self.screen = LIST
        self.chosen = None
        self.typed = ""
        self.status = ""

    def type(self, char: str) -> None:
        self.typed += char
        if not self.symbols:
            self.shift = False  # one capital, as on every phone; the symbols' second page stays
        self.status = ""

    def delete(self) -> None:
        self.typed = self.typed[:-1]
        self.status = ""
