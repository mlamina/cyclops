"""Which /dev/video node is a camera you plugged in, and which is the Pi's own silicon.

Worth a file because the wrong answer is silent and the right one cannot be tested with the
hardware attached. Fitting a Camera Module 3 brings up eight nodes before anything is plugged
in - ``rp1-cfe`` takes video0 to video7, ``pispbe`` the twenties - so a C920 that used to come
up as video0 now comes up as video8. The probe was a fixed ``range(3)``, which means fitting
the module would have quietly cost us every USB camera, and nothing would have said so: the
kiosk would just have shown the module and looked entirely healthy doing it.

The bus is the discriminator rather than the number, so what is pinned here is that a node is
chosen for sitting on USB and for nothing else - not for being low-numbered, and not for being
absent from a list of driver names that would need editing the next time Raspberry Pi ships a
new one. No camera and no Pi needed: sysfs is a directory of symlinks, and a directory of
symlinks is something a tmp_path can be.
"""

from __future__ import annotations

from cyclops import webcam


def fake_sysfs(root, nodes: dict[str, str]):
    """Build the corner of /sys/class/video4linux the probe reads: name -> subsystem it is on."""
    for name, subsystem in nodes.items():
        bus = root / "bus" / subsystem
        bus.mkdir(parents=True, exist_ok=True)
        device = root / "devices" / name
        device.mkdir(parents=True, exist_ok=True)
        (device / "subsystem").symlink_to(bus)
        node = root / "class" / name
        node.parent.mkdir(parents=True, exist_ok=True)
        node.mkdir()
        (node / "device").symlink_to(device)
    return root / "class"


def test_only_usb_nodes_are_candidates(tmp_path, monkeypatch):
    """A plugged-in camera is found by its bus, however high the module pushed its number."""
    nodes = fake_sysfs(
        tmp_path,
        {
            "video0": "platform",  # rp1-cfe: raw Bayer off the CSI bus, not a capture device
            "video3": "platform",
            "video8": "usb",  # the C920, landed above the module's nodes
            "video9": "usb",  # its metadata node; the probe read is what drops this one
            "video20": "platform",  # pispbe
        },
    )
    monkeypatch.setattr(webcam, "V4L_NODES", nodes)
    monkeypatch.setattr(webcam, "_last_good_index", None)

    assert webcam._usb_video_nodes() == [8, 9]
    assert webcam._candidate_indices(None) == [8, 9]
