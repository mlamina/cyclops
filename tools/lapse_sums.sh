#!/bin/sh
# Checksums of everything of Marco's a time-lapse probe could touch (arc-eng/hq#669, #573).
# Compared before and after each device run. Prints one line: file count and a hash of the hashes.
cd /home/cyclops/cyclops || exit 1
find sessions projects captures recordings about-you.md /home/cyclops/.cache/cyclops/jobs.json \
  -type f 2>/dev/null | LC_ALL=C sort | xargs -d '\n' sha256sum > "${1:-/tmp/swe669-sums.txt}"
printf '%s files, %s\n' "$(wc -l < "${1:-/tmp/swe669-sums.txt}")" "$(sha256sum < "${1:-/tmp/swe669-sums.txt}" | cut -c1-16)"
