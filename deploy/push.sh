#!/bin/sh
# Push the working tree to the Pi and restart what needs restarting. Run from the repo root.
#   deploy/push.sh [user@host]
# The Pi's own .env is never touched - the key lives there, not here.
set -eu

TARGET=${1:-cyclops@cyclops.local}
DEST=cyclops

# The leading slash on /projects and /manuals is load-bearing: an rsync pattern without one
# matches at every level, and src/cyclops/projects/ and src/cyclops/manuals.py are both real
# code. Anchored, each means the output directory beside sessions/ and nothing else. The three
# above have the same shape and are only safe because nothing under src/ is named that yet.
#
# Getting this wrong on a data directory is not a rebuild, it is a loss: --delete would take
# every manual on the Pi the first time somebody deployed after uploading one.
rsync -a --delete \
  --exclude '.env' --exclude '.venv' --exclude '.git' --exclude '__pycache__' \
  --exclude 'captures' --exclude 'recordings' --exclude 'sessions' \
  --exclude '/projects' --exclude '/manuals' \
  --exclude '.ruff_cache' --exclude 'Plans' \
  src pyproject.toml uv.lock README.md docs deploy "$TARGET:$DEST/"

ssh "$TARGET" "cd $DEST && ~/.local/bin/uv sync --quiet"

# Unit files are data that do nothing until they are in /etc. Refreshing them here means an
# edited .service actually reaches the Pi on a push, instead of sitting in deploy/ looking
# deployed. Installation proper - enable, and prove it runs - stays with install-*.sh.
ssh "$TARGET" "sudo install -m 644 $DEST/deploy/cyclops-*.service /etc/systemd/system/ \
  && sudo systemctl daemon-reload" || true

# Same argument for the udev rule, which is likewise inert until it is in /etc. Without it the
# endoscope's raw USB node is root-only and the kiosk finds no camera at all - so it is refreshed
# on every push rather than being a step someone has to remember once per Pi. The trigger applies
# it to a device already plugged in; a reload alone only affects the next hotplug.
ssh "$TARGET" "sudo install -m 644 $DEST/deploy/99-useeplus-camera.rules /etc/udev/rules.d/ \
  && sudo udevadm control --reload-rules && sudo udevadm trigger --subsystem-match=usb" || true

# The mono sink is the same shape of inert file - it does nothing until PipeWire reads it at
# start - so it is refreshed here too rather than being remembered once per Pi. PipeWire is
# deliberately *not* restarted: it holds the speaker, and bouncing it mid-conversation would cut
# the assistant off in the middle of a word. An edited conf lands now and takes effect at the
# next boot, which is soon enough for a file that changes about once a year.
ssh "$TARGET" "install -d ~/.config/pipewire/pipewire.conf.d \
  && install -m 644 $DEST/deploy/51-mono-speaker.conf ~/.config/pipewire/pipewire.conf.d/" || true

# The boot fanfare is a *user* unit: it plays through PipeWire, which lives in the user session
# and which a system unit cannot reach. Same argument as above for refreshing it on every push -
# and a different directory, which is also why deploy/user/ exists rather than one more name in
# deploy/. Enabling it stays with install-boot-sound.sh.
ssh "$TARGET" "sudo install -m 644 $DEST/deploy/user/cyclops-*.service /etc/systemd/user/ \
  && XDG_RUNTIME_DIR=/run/user/\$(id -u) systemctl --user daemon-reload" || true

# The two long-lived system units. `|| true` on both for the same reason: a box where one of them
# was never installed is a box being set up, not a failed deploy - install-admin.sh and
# install-index.sh are what enable them, once, per Pi. The indexer is restarted here rather than
# left alone because it holds cyclops.recall and cyclops.captions in memory, so like the kiosk it
# would otherwise quietly keep running the old build.
ssh "$TARGET" "sudo systemctl restart cyclops-admin || true"
# This can land on a video being rendered (cyclops.cut) and kill it halfway. That costs the
# encode and nothing else: the request and the plan are both on the card, and the catch-up
# sweep this very restart triggers picks it straight back up.
ssh "$TARGET" "sudo systemctl restart cyclops-index || true"

# Restart the kiosk too, and do not merely suggest it. Everything else here either re-reads the
# code on every invocation (the CLI, the smoke test) or is a service systemd restarts for us; the
# kiosk is the one long-lived process, so it is the one that silently keeps running the old build
# while every check you can think to run reports the new one. Deploying without this is how you
# ship a feature, verify it, and then watch the panel behave as though you had done neither.
# SKIP_KIOSK=1 opts out - for a docs-only push, or when someone is mid-conversation with it.
if [ "${SKIP_KIOSK:-0}" = "1" ]; then
  echo "· pushed to $TARGET:$DEST — kiosk NOT restarted (SKIP_KIOSK=1); the panel has the old code"
else
  ssh "$TARGET" "$DEST/deploy/start-kiosk.sh"
  echo "· pushed to $TARGET:$DEST — admin and kiosk both restarted"
fi

# The kiosk we just pkill'd may have been mid-session: SIGTERM is handled, but the session thread
# is a daemon joined for 20 s against a mux capped at 120, so a deploy during a long recording can
# kill a teardown halfway through and manufacture exactly the damage this repairs. --fix is free,
# offline and destroys nothing, so it runs on every push. Recovery proper - which deletes husks
# and spends on naming - stays on the boot unit, where nobody gets it by surprise.
ssh "$TARGET" "cd $DEST && .venv/bin/cyclops-sessions --fix" || true
