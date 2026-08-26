#!/bin/sh
# Push the working tree to the Pi and restart what needs restarting. Run from the repo root.
#   deploy/push.sh [user@host]
# The Pi's own .env is never touched - the key lives there, not here.
set -eu

TARGET=${1:-dobby@raspberrypi.local}
DEST=cyclops

rsync -a --delete \
  --exclude '.env' --exclude '.venv' --exclude '.git' --exclude '__pycache__' \
  --exclude 'captures' --exclude 'recordings' --exclude '.ruff_cache' --exclude 'Plans' \
  src pyproject.toml uv.lock README.md deploy "$TARGET:$DEST/"

ssh "$TARGET" "cd $DEST && ~/.local/bin/uv sync --quiet"
ssh "$TARGET" "sudo systemctl restart cyclops-admin || true"
echo "· pushed to $TARGET:$DEST — run ~/start_kiosk.sh there to pick up kiosk changes"
