#!/usr/bin/env python3
"""hq#599: would the display server talk to this account? Connection setup only, then disconnect.

Sends the X11 opening with no credentials and reads the answer's first byte. Nothing is drawn or
read from the screen. Run by `sudo -u cyclops-worker` and, as the control, by `cyclops`.
"""

from __future__ import annotations

import os
import socket
import struct
import sys

NAME = "\0/tmp/.X11-unix/X0"
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.settimeout(4)
try:
    sock.connect(NAME)
    sock.sendall(struct.pack("<cxHHHHxx", b"l", 11, 0, 0, 0))
    head = sock.recv(8)
except OSError as exc:
    print(f"uid {os.getuid()}: no connection ({exc.strerror})")
    sys.exit(0)
finally:
    pass
if not head:
    print(f"uid {os.getuid()}: closed with no answer")
elif head[0] == 1:
    print(f"uid {os.getuid()}: ACCEPTED, with no credentials")
else:
    reason = sock.recv(head[1]).decode("ascii", "replace") if head[1] else ""
    print(f"uid {os.getuid()}: refused ({reason.strip()})")
sock.close()
