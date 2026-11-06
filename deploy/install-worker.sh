#!/bin/sh
# Install the background worker on the Pi. Run on the Pi from the repo, as root:
#   sudo deploy/install-worker.sh           # the account, the folder, the service and its firewall
#   sudo deploy/install-worker.sh --remove  # all of that gone again, except the token
#
# Turning it on for the voice session is a separate step: CYCLOPS_WORKER=1 in .env.
# The token at /etc/cyclops-worker/claude-token is placed by hand, once. See docs/worker.md.
set -eu

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
USER_NAME=cyclops-worker
ROOT=/var/lib/cyclops-worker
OPT=/opt/cyclops-worker
UNIT=cyclops-worker.service
SHARED_GROUP=cyclops  # lets the voice side write requests and read answers, and nothing more

if [ "${1:-}" = "--remove" ]; then
  systemctl disable --now "$UNIT" 2>/dev/null || true
  /usr/sbin/nft delete table inet cyclops_worker 2>/dev/null || true
  rm -f "/etc/systemd/system/$UNIT"
  systemctl daemon-reload
  rm -rf "$OPT" "$ROOT"
  userdel "$USER_NAME" 2>/dev/null || true
  echo "removed; /etc/cyclops-worker is left in place"
  exit 0
fi

id "$USER_NAME" >/dev/null 2>&1 || \
  useradd --system --no-create-home --home-dir "$ROOT/home" --shell /usr/sbin/nologin "$USER_NAME"

# Setgid, so everything the worker makes stays readable by the voice side through the group. The
# worker is not in that group: this gives Cyclops a way in and gives the worker no way out.
install -d -o "$USER_NAME" -g "$SHARED_GROUP" -m 2770 "$ROOT" "$ROOT/inbox"
install -d -o "$USER_NAME" -g "$SHARED_GROUP" -m 2750 "$ROOT/jobs" "$ROOT/done" "$ROOT/home"

install -d -m 755 "$OPT" "$OPT/bin"
install -m 644 "$HERE/../src/cyclops/worker.py" "$OPT/worker.py"
install -m 644 "$HERE/cyclops-worker.nft" "$OPT/firewall.nft"
# A copy of its own: the one in /home/cyclops is out of its sight, and it mustn't update itself.
# The program, not ~/.local/bin/claude, which on this box is a wrapper that runs a path under
# /home. A wrapper or an empty file (failed updates leave those) would fail on every job.
CLAUDE=/home/cyclops/.local/bin/claude-real
[ -x "$CLAUDE" ] || CLAUDE=/home/cyclops/.local/bin/claude
CLAUDE=$(readlink -f "$CLAUDE")
if [ ! -s "$CLAUDE" ] || head -c 2 "$CLAUDE" | grep -q '#!'; then
  echo "error: $CLAUDE isn't the Claude Code program" >&2
  exit 1
fi
install -m 755 "$CLAUDE" "$OPT/bin/claude"

install -d -m 755 /etc/cyclops-worker
if [ ! -s /etc/cyclops-worker/claude-token ]; then
  echo "note: no token yet at /etc/cyclops-worker/claude-token; every job will fail until there is" >&2
fi

install -m 644 "$HERE/$UNIT" "/etc/systemd/system/$UNIT"
systemctl daemon-reload
systemctl enable --now "$UNIT"
systemctl restart "$UNIT"
sleep 2
systemctl --no-pager --lines=10 status "$UNIT"
