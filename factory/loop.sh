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
WEDGED=90    # minutes of silence before a build is presumed hung
EVERY=30     # seconds between passes

# A second loop would see the same ready jobs, cut the same worktrees and set two agents on one
# branch. A process fact rather than a lockfile, for tasks.py's reason: a lockfile outlives the
# thing it describes and this cannot.
#
# The pattern is anchored to the *end* of the command line, and that is the whole trick. Written
# loose as `factory/loop.sh` it matches the shell that just invoked this script - a `zsh -c`
# carries the command it was given in its own argv - so the first loop ever started reports that
# a loop is already running and exits. Measured, not guessed. Anchored, only a shell whose argv
# ends in the script itself matches, which is what an actually-running loop looks like.
#
# `if` and not `&&`: under `set -e` a bare `pgrep ... && { ... }` that finds nothing exits 1 and
# takes the script with it. A condition is exempt; a statement is not.
if pgrep -f 'sh .*factory/loop\.sh$' > /dev/null 2>&1; then
  echo "· A LOOP IS ALREADY RUNNING — one is enough" >&2
  exit 1
fi

mkdir -p "$LOGS"

# One timestamped line in a job's log. Everything this loop knows goes through here, and the
# markers - START, PID, EXIT, DONE, STALLED, WEDGED - are the whole of what /board reads.
#
# They exist because a build that finished and a build that gave up look identical from out here.
# `claude -p` exits the moment the model ends its turn, and a model ends its turn to report
# progress as readily as to report a result; either way what is left behind is a log with prose
# in it and no process. Job 002 backgrounded a sixty-turn measurement, ended its turn at forty-nine
# to say how it was going, and took the measurement down with it - and because nothing had written
# down that it ended, /board called it "building" for the next twenty minutes.
say() { printf '· %s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$2" >> "$1"; }

# What a job's frontmatter says right now, or "gone". Read once, after a build exits: `review` is
# the only thing a build that finished its work leaves behind, so it is how this loop tells the
# two endings apart without understanding anything about the job itself.
state_of() {  # $1 number
  for jf in "$ROOT"/factory/"$1"-*.md; do
    [ -e "$jf" ] || continue
    head -8 "$jf" | sed -n 's/^state: *//p' | head -1
    return 0
  done
  echo gone
}

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
  : > "$3"
  say "$3" "START $1 - $2"
  wt=$JOBS/$1
  # An existing branch is this job's own earlier work - a crashed run, or a /rework - so it is
  # checked out rather than recut, and an existing directory is reused rather than re-added.
  # Every form of `git worktree add` is fatal, and a fatal command under `set -eu` inside a loop
  # nobody is watching means the factory stops and never says so.
  if [ ! -d "$wt" ]; then
    # Asked rather than attempted, so the ordinary path doesn't write `fatal: invalid reference`
    # into a log somebody only ever opens when they already think something is wrong.
    if git show-ref --verify --quiet "refs/heads/job/$1-$2"; then
      add="git worktree add $wt job/$1-$2"
    else
      add="git worktree add -b job/$1-$2 $wt master"
    fi
    $add >> "$3" 2>&1 || {
      # An ending, written down like any other. A log that stops dead with a git error in it is
      # the shape a crashed build has, and out here they must not be the same thing.
      say "$3" "EXIT 1"
      say "$3" "STALLED - git could not cut the worktree"
      echo "· $1 NO WORKTREE — see ${3#$ROOT/}" >&2
      return 0
    }
  fi
  # Waited on rather than fired and forgotten, and the waiting is the point: `$?` from a build
  # is the one fact that says how it ended, and it exists for a few milliseconds unless something
  # is standing there to catch it.
  #
  # The supervisor is backgrounded so the pass carries on. `claude` inside it is backgrounded so
  # its pid can be written down, and waited on so its status is real. `< /dev/null` for the reason
  # start-kiosk.sh:49 has it: a backgrounded process that reads the terminal gets SIGTTIN and
  # stops, and a stopped build holds a slot for ever while looking alive. `trap '' INT HUP`
  # survives exec and covers both ways this loop can be walked away from - Ctrl-C leaves the
  # builds standing, and closing the terminal no longer kills the one process that would have
  # recorded the ending.
  ( trap '' INT HUP
    cd "$wt" 2>/dev/null || { say "$3" "EXIT 1"; say "$3" "STALLED - $wt is not there"; exit 0; }
    nohup claude -p "/build $1" --dangerously-skip-permissions >> "$3" 2>&1 < /dev/null &
    cpid=$!
    say "$3" "PID $cpid"
    # `|| code=$?` and not a bare `wait`: under `set -e` a build that exits non-zero would take
    # this subshell with it, and the ending would go unwritten in exactly the case that needs it.
    code=0; wait "$cpid" || code=$?
    say "$3" "EXIT $code"
    if [ "$(state_of "$1")" = review ]; then
      say "$3" "DONE - waiting on Marco"
      echo "· $1 done — /review $1"
    else
      # The build's own contract: the page, then the ticks, then `state: review`, in that order
      # and last. Exiting without it means it stopped somewhere in the middle, whatever its exit
      # status says - a model that ends its turn to describe what it is about to do next exits 0.
      say "$3" "STALLED - exited at 'state: $(state_of "$1")' without setting review"
      echo "· $1 STALLED — /rework $1 to send it round again" >&2
    fi
  ) &
  echo "· $1 building — $2"
}

pass() {
  git worktree prune || true  # a job directory deleted by hand leaves a registration, and a
                              # stale registration makes the next `add` fatal
  for f in "$ROOT"/factory/[0-9]*.md; do  # [0-9] so README.md is never taken for a job
    [ -e "$f" ] || continue               # an empty board is an unexpanded glob, not a file
    b=${f##*/}; b=${b%.md}; n=${b%%-*}; slug=${b#*-}; log=$LOGS/$n.log

    # Alive is the pid written into the log when the build started, never a pattern. `pgrep -f`
    # matches any shell whose argv carries the string, which includes the one asking - that is
    # how /board reported job 002 as building for twenty minutes after it had died. A pid cannot
    # match the process asking about it, and `kill -0` only asks.
    pid=$(sed -n 's/^· .* PID \([0-9][0-9]*\)$/\1/p' "$log" 2>/dev/null | tail -1)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      # Wedged is "has written nothing", not "has been going a while". The old test read the log's
      # mtime, which `claude -p` only touches when it finishes, so it was the start time and
      # nothing else: a build still working at minute 91 was killed for the crime of being slow,
      # and one stuck at minute 3 was left alone. What a working build always does is write files.
      if [ -z "$( { find "$JOBS/$n" -name .venv -prune -o -type f -mmin -$WEDGED -print
                    find "$ROOT/factory/html/$n" -type f -mmin -$WEDGED -print
                  } 2>/dev/null | head -1)" ]; then
        kill "$pid" 2>/dev/null || true
        say "$log" "WEDGED - killed, nothing written for ${WEDGED}m"
        echo "· $n WEDGED — killed after ${WEDGED}m idle. /rework $n to try again" >&2
      fi
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
