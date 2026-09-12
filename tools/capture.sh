#!/bin/sh
# Capture an idea, a bug or a feature request. One way in, for all three.
#   tools/capture.sh "the eye stutters when it blinks"
#   echo "..." | tools/capture.sh
# No model, no network, no questions asked - it writes a file and gets out of the way, so
# capturing costs nothing and never interrupts what you were doing. The thinking happens later,
# in /factory. Numbers are never reused: done/ counts too.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
BOARD=$ROOT/factory

if [ $# -gt 0 ]; then TEXT=$*; else TEXT=$(cat); fi
[ -n "$TEXT" ] || { echo "nothing to capture" >&2; exit 1; }

mkdir -p "$BOARD/done"
LAST=$(ls "$BOARD" "$BOARD/done" 2>/dev/null | sed -n 's/^\([0-9][0-9][0-9]\)-.*\.md$/\1/p' | sort -n | tail -1)
NEXT=$(printf '%03d' $(( ${LAST:-0} + 1 )))

SLUG=$(printf '%s' "$TEXT" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]\{1,\}/-/g; s/^-//; s/-$//' | cut -c1-40 | sed 's/-$//')
FILE=$BOARD/$NEXT-${SLUG:-note}.md

cat > "$FILE" <<INNER
---
state: new
opened: $(date +%Y-%m-%d)
---

# $TEXT

$TEXT
INNER

echo "${FILE#$ROOT/}"
