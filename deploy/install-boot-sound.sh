#!/bin/sh
# Install (or reinstall) the boot fanfare. Idempotent; run on the Pi.
#
# The unit file itself is refreshed on every deploy/push.sh, the same way the system units are.
# Enabling is what lives here, because it is a once-per-box step and because "enabled" is a
# claim worth proving rather than assuming.
set -eu

UNIT=cyclops-boot-sound.service
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}

systemctl --user daemon-reload
systemctl --user enable "$UNIT"   # runs at the next login, not now: it says the box just booted

echo "· $UNIT enabled — it sounds on the next boot"
systemctl --user is-enabled "$UNIT"
echo "· to hear it now:  systemctl --user start $UNIT"
echo "· to silence it:   CYCLOPS_SOUNDS=0 in ~/cyclops/.env, or systemctl --user disable $UNIT"
