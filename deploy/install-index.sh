#!/bin/sh
# Install (or reinstall) the cyclops recall indexer as a system service. Idempotent; run on the Pi.
#   sudo deploy/install-index.sh
set -eu

UNIT=cyclops-index.service
HERE=$(cd "$(dirname "$0")" && pwd)

[ "$(id -u)" -eq 0 ] || { echo "run me with sudo" >&2; exit 1; }

# It cannot embed or caption anything without a key, and it would say so once every 15 minutes
# forever. Warn rather than refuse: watching still works, and the first sweep after a key appears
# picks up everything at once.
if ! grep -q '^ *OPENAI_API_KEY=..' /home/cyclops/cyclops/.env 2>/dev/null; then
  echo "warning: no OPENAI_API_KEY in .env - the indexer will watch but index nothing" >&2
fi

install -m 644 "$HERE/$UNIT" "/etc/systemd/system/$UNIT"
systemctl daemon-reload
systemctl enable --now "$UNIT"
systemctl restart "$UNIT"      # enable --now is a no-op when it was already running
sleep 2
systemctl --no-pager --lines=15 status "$UNIT"
