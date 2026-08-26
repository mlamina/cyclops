#!/bin/sh
# Install (or reinstall) the cyclops status page as a system service. Idempotent; run on the Pi.
#   sudo deploy/install-admin.sh
set -eu

UNIT=cyclops-admin.service
HERE=$(cd "$(dirname "$0")" && pwd)

[ "$(id -u)" -eq 0 ] || { echo "run me with sudo" >&2; exit 1; }

# Fail early and legibly rather than leaving a service that restarts every 3 s forever.
holder=$(ss -ltnp 'sport = :80' 2>/dev/null | tail -n +2 || true)
case "$holder" in
  *cyclops-admin*|"") ;;
  *) echo "something else is already on :80:" >&2; echo "$holder" >&2
     echo "free it, or set CYCLOPS_ADMIN_PORT in .env and edit nothing else." >&2; exit 1 ;;
esac

install -m 644 "$HERE/$UNIT" "/etc/systemd/system/$UNIT"
systemctl daemon-reload
systemctl enable --now "$UNIT"
systemctl restart "$UNIT"      # enable --now is a no-op when it was already running
sleep 1
systemctl --no-pager --lines=10 status "$UNIT"
