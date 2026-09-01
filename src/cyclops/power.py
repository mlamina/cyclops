"""Taking the box down: shut down and restart, for a panel with no keyboard.

A Pi that is only ever unplugged loses the session it was in the middle of. The kiosk already
knows how to finish one - the video is muxed, the folder is named - and all of that needs the
process to be allowed to end rather than have the power cut from under it.

``sudo -n systemctl`` first, because that is what the deploy scripts already rely on and it works
from wherever the kiosk was started. Bare ``systemctl`` after it, for a box where logind will
take it anyway: polkit waves a power-off through for a session that is active and local, which
the kiosk is when the desktop starts it at boot.
"""

from __future__ import annotations

import subprocess

# What systemctl calls them. The panel's own words for these live in cyclops.kiosk.
POWEROFF = "poweroff"
REBOOT = "reboot"

TIMEOUT_S = 15.0  # systemctl returns as soon as systemd has the job; the shutdown outlives us


def take_down(verb: str) -> bool:
    """Ask for a ``poweroff`` or a ``reboot``. False if neither route would have it.

    Called after the kiosk has torn itself down, never before: the session's video is still
    being muxed while the panel says goodbye, and systemd starts killing units the moment this
    returns.
    """
    for command in (["sudo", "-n", "systemctl", verb], ["systemctl", verb]):
        try:
            if subprocess.run(command, capture_output=True, timeout=TIMEOUT_S).returncode == 0:
                return True
        except (OSError, subprocess.SubprocessError):
            continue
    return False
