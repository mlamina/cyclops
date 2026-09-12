#!/bin/sh
# The factory. Leave it running in a terminal and never touch it.
#   factory/loop.sh
# Every pass: anything `state: ready` gets a worktree and a detached `claude -p /build NNN`.
# That is the whole job. It never writes the board, never merges, and never touches the Pi -
# every one of those is Marco's word, given through a command in his own session.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
LOGS=$ROOT/factory/logs
JOBS=$ROOT/../cyclops-jobs
MAX=3        # builds at once
WEDGED=90    # minutes before a build is presumed hung
EVERY=30     # seconds between passes

# A second loop would see the same ready jobs, cut the same worktrees and set two agents on one
# branch. A process fact rather than a lockfile, for tasks.py's reason: a lockfile outlives the
# thing it describes and this cannot. (BSD pgrep skips its own ancestors, so this finds only a
# *other* instance - on Linux it would also need `| grep -v $$`.)
pgrep -f 'factory/loop.sh' > /dev/null && { echo "· A LOOP IS ALREADY RUNNING" >&2; exit 1; }

mkdir -p "$LOGS"

# What "007 is building" looks like in ps - the whole invocation, not the number. Claude Code's
# Bash tool puts the command text in its shell's own argv, so a peer session running
# `cat factory/007-something.md` has `007` on its command line, and a loop matching the number
# would call that a build and never start the real one. Same trap as deploy/start-kiosk.sh:10-11,
# same answer: the pattern lives in this file, so it can never match the shell that asked.
pat() { printf 'claude -p /build %s --dangerously-skip-permissions' "$1"; }

start() {  # $1 number, $2 slug, $3 absolute log path
  # The log is opened here, first, by an absolute path, and both of those are load-bearing.
  # Written relative it would resolve after the `cd` below and land inside the worktree, where
  # the marker this loop reads never appears - so every job would restart every thirty seconds
  # for ever. And a worktree that cannot be cut has to leave the marker too, or that is the same
  # runaway with a different cause. The git error lands in the log, the only place anyone looks.
  date '+%Y-%m-%d %H:%M:%S' > "$3"
  wt=$JOBS/$1
  # An existing branch is this job's own earlier work - a crashed run, or a /rework - so it is
  # checked out rather than recut, and an existing directory is reused rather than re-added.
  # Every form of `git worktree add` is fatal, and a fatal command under `set -eu` inside a loop
  # nobody is watching means the factory stops and never says so.
  [ -d "$wt" ] || git worktree add "$wt" "job/$1-$2" >> "$3" 2>&1 \
               || git worktree add -b "job/$1-$2" "$wt" master >> "$3" 2>&1 \
               || { echo "· $1 NO WORKTREE — see ${3#$ROOT/}" >&2; return 0; }
  # `< /dev/null` for the reason start-kiosk.sh:49 has it: a backgrounded process that reads the
  # terminal gets SIGTTIN and stops. A stopped build still answers pgrep, so it would hold a slot
  # for ever while looking alive. `trap '' INT` survives exec, so Ctrl-C on the loop leaves the
  # builds standing and the next loop finds them and leaves them alone.
  ( trap '' INT; cd "$wt" && nohup claude -p "/build $1" --dangerously-skip-permissions \
      >> "$3" 2>&1 < /dev/null & )
  echo "· $1 building — $2"
}

pass() {
  git worktree prune || true  # a job directory deleted by hand leaves a registration, and a
                              # stale registration makes the next `add` fatal
  for f in "$ROOT"/factory/[0-9]*.md; do  # [0-9] so README.md is never taken for a job
    [ -e "$f" ] || continue               # an empty board is an unexpanded glob, not a file
    b=${f##*/}; b=${b%.md}; n=${b%%-*}; slug=${b#*-}; log=$LOGS/$n.log

    # `claude -p` prints nothing until it finishes, so the log's mtime is its start time and
    # nothing else - which makes `find -mmin` the whole timeout, with no ps parsing (macOS has
    # no `-o etimes`). Kill by pid and never `pkill -f`: a pattern that finds one build finds
    # three, and finds a peer's shell.
    pid=$(pgrep -f "$(pat "$n")" | head -1 || true)
    if [ -n "$pid" ] && [ -n "$(find "$log" -mmin +$WEDGED 2>/dev/null)" ]; then
      kill "$pid" 2>/dev/null || true
      echo "· $n WEDGED — killed after ${WEDGED}m. /rework $n to try again" >&2
      continue
    fi

    # Read the state here and not from a list gathered up front: a pass takes seconds, and a
    # build that reaches `review` inside one would otherwise be started a second time. head -8
    # keeps it to the frontmatter, so prose in the body saying `state: ready` is not an order.
    head -8 "$f" | grep -q '^state: ready' || continue
    [ -f "$log" ] && continue                    # already had its swing; /rework clears the log
    pgrep -f "$(pat "$n")" > /dev/null && continue
    [ "$(pgrep -f 'claude -p /build ' | wc -l)" -lt $MAX ] || continue
    start "$n" "$slug" "$log"
  done
}

echo "· factory up — watching ${ROOT##*/}/factory, $MAX at a time"
while :; do
  pass
  sleep "$EVERY"
done
