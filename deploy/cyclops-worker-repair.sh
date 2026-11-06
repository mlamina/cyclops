#!/bin/sh
# Runs as root before every start of the worker (and at install), with none of its processes up.
# A job can wreck anything in its folder, so this puts the folder back. rm doesn't follow links and
# the top folder's parent isn't the worker's, so what a job planted can't aim this anywhere else.
set -eu
ROOT=/var/lib/cyclops-worker
USER_NAME=cyclops-worker
SHARED_GROUP=$(cat /opt/cyclops-worker/shared-group)

# Setgid, so everything the worker makes stays readable by the voice side through the group. The
# worker is not in that group: this gives Cyclops a way in and gives the worker no way out. And not
# being in it, the worker couldn't set the bit back itself.
for name in "" inbox jobs done; do
  path="$ROOT/$name"
  if [ -L "${path%/}" ] || { [ -e "$path" ] && [ ! -d "$path" ]; }; then
    rm -rf "${path%/}"
  fi
done
install -d -o "$USER_NAME" -g "$SHARED_GROUP" -m 2770 "$ROOT" "$ROOT/inbox"
install -d -o "$USER_NAME" -g "$SHARED_GROUP" -m 2750 "$ROOT/jobs" "$ROOT/done"

# Nothing runs from the account's home: each job has its own. A unit left here would be the way
# back in if lingering were ever on again, so neither survives a restart.
rm -rf "$ROOT/home"
install -d -o "$USER_NAME" -g "$SHARED_GROUP" -m 2750 "$ROOT/home"
loginctl disable-linger "$USER_NAME" 2>/dev/null || true
