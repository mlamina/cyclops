"""FLIP SCREEN: the connector out of wlr-randr, and the menu row that must not end the box."""

from __future__ import annotations

from cyclops import flip

# Off the Pi on 2026-09-26, after the Camera Module had renumbered the panel from DSI-1.
LISTING = """DSI-2 "(null) (null) (DSI-2)"
  Physical size: 154x86 mm
  Enabled: yes
  Modes:
    800x480 px, 60.028999 Hz (preferred, current)
  Position: 0,0
  Transform: normal
  Scale: 1.000000
"""


def test_the_connector_is_read_from_the_listing() -> None:
    assert flip.connector(LISTING) == "DSI-2"
    off = 'HDMI-A-1 "x"\n  Enabled: no\n' + LISTING
    assert flip.connector(off) == "DSI-2", "a disabled output came first and was taken"
    assert flip.connector("") is None

