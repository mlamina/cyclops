#!/bin/bash
# hq#572: fingerprint of his records on the Pi, to compare before and after a check.
# Usage: bash hq_572_records_sum.sh <output file>
out=$1
{
  cd ~/cyclops || exit 1
  find projects manuals about-you.md -type f -print0 2>/dev/null | sort -z | xargs -0 sha256sum
  echo "--- sessions"
  ls sessions
  echo "--- cache"
  sha256sum ~/.cache/cyclops/tasks.yaml ~/.cache/cyclops/panel.json ~/.cache/cyclops/voice 2>/dev/null
} > "$out"
wc -l "$out"
sha256sum "$out"
