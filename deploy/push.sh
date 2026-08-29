#!/bin/sh
# Push the working tree to the Pi and restart what needs restarting. Run from the repo root.
#   deploy/push.sh [user@host]
# The Pi's own .env is never touched - the key lives there, not here.
set -eu

TARGET=${1:-dobby@raspberrypi.local}
DEST=cyclops

# The leading slash on /projects is load-bearing: an rsync pattern without one matches at
# every level, and src/cyclops/projects/ is a real package. Anchored, it means the output
# directory beside sessions/ and nothing else. The three above have the same shape and are
# only safe because nothing under src/ is named that yet.
rsync -a --delete \
  --exclude '.env' --exclude '.venv' --exclude '.git' --exclude '__pycache__' \
  --exclude 'captures' --exclude 'recordings' --exclude 'sessions' \
  --exclude '/projects' \
  --exclude '.ruff_cache' --exclude 'Plans' \
  src pyproject.toml uv.lock README.md deploy "$TARGET:$DEST/"

ssh "$TARGET" "cd $DEST && ~/.local/bin/uv sync --quiet"

# Unit files are data that do nothing until they are in /etc. Refreshing them here means an
# edited .service actually reaches the Pi on a push, instead of sitting in deploy/ looking
# deployed. Installation proper - enable, and prove it runs - stays with install-*.sh.
ssh "$TARGET" "sudo install -m 644 $DEST/deploy/cyclops-*.service /etc/systemd/system/ \
  && sudo systemctl daemon-reload" || true

ssh "$TARGET" "sudo systemctl restart cyclops-admin || true"

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
