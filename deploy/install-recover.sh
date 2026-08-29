#!/bin/sh
# Install (or reinstall) boot-time session recovery. Idempotent; run on the Pi.
#   sudo deploy/install-recover.sh
set -eu

UNIT=cyclops-recover.service
HERE=$(cd "$(dirname "$0")" && pwd)

[ "$(id -u)" -eq 0 ] || { echo "run me with sudo" >&2; exit 1; }

install -m 644 "$HERE/$UNIT" "/etc/systemd/system/$UNIT"
systemctl daemon-reload
systemctl enable "$UNIT"        # oneshot: enabled for the next boot, not left running
systemctl start "$UNIT" || true # ...and run it once now, so a failure is seen here and not in a month

# `systemctl status` on a finished oneshot says "inactive (dead)", which reads like a failure to
# anyone who has not thought about it. The journal is what actually says what happened.
journalctl -u "$UNIT" -n 40 --no-pager
