"""The USEEPlus endoscope: a snake camera on a cable, and the smallest extension there can be.

Instructions and nothing else - no tools. What it changes is what the model knows it is holding:
a lens that goes where a hand and the case camera cannot, and sees small things up close.

Its two IDs are also in ``cyclops.webcam.USEEPLUS_IDS``, which is how the camera gets *opened*,
and in ``deploy/99-useeplus-camera.rules``, which is what lets it be. Those stay where they are -
a camera behind a plug-in loader is a dead camera the day an extension breaks - and a test holds
all three lists to agreement.
"""

from __future__ import annotations

from . import Extension

EXTENSION = Extension(
    name="Endoscope",
    # The bus gives neither a name nor a category for these: no strings, vendor-specific
    # interfaces. So this is also where cyclops.devices learns what they are.
    usb_ids=("2ce3:3828", "0329:2022"),
    category="camera",
    instructions=(
        "a snake camera on a thin cable. Its tip gets into tight spaces - inside a wall, behind "
        "a panel, down a pipe - and takes close-ups of small things. While it is plugged in, "
        "photos usually come from its tip: close, narrow, and with nothing in frame for scale."
    ),
)
