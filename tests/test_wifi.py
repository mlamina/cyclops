"""The Wi-Fi picker: what nmcli said, turned into rows, and the panel that joins one.

The scan is the real one - 18 rows off the Pi on 2026-09-26, with its per-band duplicates and
three hidden networks (tests/wifi_scan.txt). Nothing here runs nmcli: joins and scans are
replaced, and threads are run in line so a test can see what they left behind.
"""

from __future__ import annotations

import functools
import socket
from pathlib import Path

import pytest

from cyclops import kiosk as kiosk_module
from cyclops import overlay, ui, wifi

SCAN = (Path(__file__).parent / "wifi_scan.txt").read_text()
DOWN = kiosk_module.cv2.EVENT_LBUTTONDOWN


@functools.cache
def _overlay() -> overlay.Overlay:
    return overlay.Overlay(800, 480)


class _Inline:
    """threading.Thread, run where it is started."""

    def __init__(self, target, name: str = "", daemon: bool = False, args: tuple = ()) -> None:
        self._run = lambda: target(*args)

    def start(self) -> None:
        self._run()


@pytest.fixture
def kiosk(monkeypatch: pytest.MonkeyPatch) -> kiosk_module.Kiosk:
    k = object.__new__(kiosk_module.Kiosk)
    k.overlay = _overlay()
    k._menu = False
    k._menu_until = 0.0
    k._page_busy = kiosk_module.threading.Event()
    k.power = None
    k._pressed = None
    k._press_until = 0.0
    k._touched_at = 0.0
    k._asleep = False
    k._eye_down_at = None
    k._turning = False
    k._notice = ""
    k._notice_until = 0.0
    monkeypatch.setattr(kiosk_module.threading, "Thread", _Inline)
    monkeypatch.setattr(wifi, "scan", lambda: wifi.parse_scan(SCAN, {"Noise Upstairs 5g"}))
    return k


# ---------------------------------------------------------------- the scan


def test_one_row_per_network_saved_first_then_strongest() -> None:
    nets = wifi.parse_scan(SCAN, {"MommyKwong"})
    names = [n.ssid for n in nets]
    assert len(names) == len(set(names)) == 13, "a duplicate band or a hidden network got in"
    assert "" not in names and "--" not in names
    assert names[:2] == ["Noise Upstairs 5g", "MommyKwong"], "in use, then saved, first"
    assert next(n for n in nets if n.ssid == "MidnightFlour").signal == 79, "strongest band kept"
    rest = [n.signal for n in nets[2:]]
    assert rest == sorted(rest, reverse=True)


def test_a_colon_in_an_ssid_survives_the_terse_format() -> None:
    assert wifi.parse_scan(r":Shed\:2:70:WPA2", set())[0].ssid == "Shed:2"


def test_a_join_that_nmcli_refused_for_its_key_is_a_wrong_password() -> None:
    said = "Error: Connection activation failed: Secrets were required, but not provided."
    assert wifi.classify(4, said) == wifi.WRONG_PASSWORD
    assert wifi.classify(10, "Error: No network with SSID 'Shed' found.") == wifi.FAILED
    assert wifi.classify(0, "") == wifi.JOINED


# ---------------------------------------------------------------- the menu and the ways in


def test_choosing_wifi_opens_the_picker_and_does_not_reboot(kiosk) -> None:
    kiosk._menu = True
    kiosk._on_mouse(DOWN, *kiosk.overlay.menu_cells[overlay.WIFI].center, 0, None)
    assert kiosk.power is None, "the WI-FI row fell through to RESTART"
    assert not kiosk._menu and kiosk._wifi is not None
    assert len(kiosk._wifi.networks) == 13


def test_the_picker_never_opens_by_itself_over_a_session(kiosk) -> None:
    kiosk._offer_wifi(live=True)
    assert kiosk._wifi is None
    kiosk._offer_wifi(live=False)
    assert kiosk._wifi is not None


def test_the_button_with_no_network_opens_the_picker_instead(kiosk) -> None:
    started = []

    class Controller:
        def status(self) -> dict:
            return {"state": overlay.IDLE}

        def start(self) -> None:
            started.append(True)

    kiosk.controller = Controller()
    kiosk._online = False
    kiosk._toggle_session()
    assert kiosk._wifi is not None and not started


def test_a_connect_failure_is_not_the_sdks_exception() -> None:
    raw = socket.gaierror(8, "nodename nor servname provided")
    assert ui.failure(raw) != str(raw)
    assert ui.failure(ValueError("something else")) == "something else"


# ---------------------------------------------------------------- the list


def test_more_pages_through_six_at_a_time(kiosk) -> None:
    kiosk._open_wifi()
    first = [n.ssid for n in kiosk._wifi.shown()]
    kiosk._on_mouse(DOWN, *kiosk.overlay.wifi_cells[overlay.MORE].center, 0, None)
    second = [n.ssid for n in kiosk._wifi.shown()]
    assert len(first) == len(second) == wifi.PAGE
    assert not set(first) & set(second)


def test_a_saved_network_joins_on_one_tap(kiosk, monkeypatch: pytest.MonkeyPatch) -> None:
    asked = []
    monkeypatch.setattr(wifi, "join", lambda net, pw="": asked.append((net.ssid, pw)) or "joined")
    kiosk._open_wifi()
    kiosk._wifi.networks = wifi.parse_scan(SCAN, {"MidnightFlour"})  # nothing in use
    kiosk._wifi.networks = [n for n in kiosk._wifi.networks if not n.active]
    kiosk._on_mouse(DOWN, *kiosk.overlay.wifi_cells[f"{overlay.NET}0"].center, 0, None)
    assert asked == [("MidnightFlour", "")]
    assert kiosk._wifi is None and kiosk._online


# ---------------------------------------------------------------- the keyboard


@pytest.mark.parametrize(("symbols", "alt"), [(False, False), (True, False), (True, True)])
def test_every_key_is_a_thumb_target_and_answers_to_its_middle(symbols: bool, alt: bool) -> None:
    ov = _overlay()
    cells = ov.key_cells(symbols, alt)
    for key in (overlay.SHIFT, overlay.SYMBOLS, overlay.SPACE, overlay.DELETE, overlay.JOIN):
        assert key in cells
    boxes = list(cells.items())
    for key, cell in boxes:
        assert cell.h >= 62, f"{key} is {cell.h} px tall"
        assert ov.key_hit(*cell.center, symbols, alt) == key
        assert ov.key_card.contains(cell.x, cell.y) and cell.bottom <= ov.key_card.bottom
    for i, (_, a) in enumerate(boxes):
        for _, b in boxes[i + 1 :]:
            assert a.right <= b.x or b.right <= a.x or a.bottom <= b.y or b.bottom <= a.y


def test_a_wrong_password_stays_on_the_keyboard_with_what_was_typed(
    kiosk, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(wifi, "join", lambda net, pw="": wifi.WRONG_PASSWORD)
    kiosk._open_wifi()
    picker = kiosk._wifi
    picker.choose(next(n for n in picker.networks if n.ssid == "MidnightFlour"))
    cells = kiosk.overlay.key_cells()
    for key in (overlay.SHIFT, "h", "u", "n", "t", "e", "r", overlay.SYMBOLS, "2", "2"):
        kiosk._on_mouse(DOWN, *cells[key].center if len(key) > 1 or key.isalpha()
                        else kiosk.overlay.key_cells(True)[key].center, 0, None)
    assert picker.typed == "Hunter22"
    kiosk._on_mouse(DOWN, *cells[overlay.JOIN].center, 0, None)
    assert kiosk._wifi is picker and picker.screen == wifi.KEYBOARD
    assert picker.typed == "Hunter22" and picker.status
