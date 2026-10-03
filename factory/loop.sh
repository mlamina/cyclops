#!/bin/sh
# The factory. Leave it running in a terminal and never touch it.
#   factory/loop.sh
# Every pass: anything `state: ready` gets a worktree and a detached `claude -p /build NNN`.
# That is the whole job. It never writes the board, never merges, and never touches the Pi -
# every one of those is Marco's word, given through a command in his own session.
#
# It also watches. A build reports nothing until it exits, so everything anyone knows about one
# while it runs is inferred out here and written down as it happens: START, PID, WORKING, WAITING,
# STATE, EXIT, DONE, STALLED, WEDGED. Those markers are the whole of what /board reads, and the
# terminal gets a line every few minutes so the window is worth glancing at.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
LOGS=$ROOT/factory/logs
JOBS=$(CDPATH= cd -- "$ROOT/.." && pwd)/cyclops-jobs
MAX=3        # builds at once
WEDGED=90    # minutes of silence before a build is presumed hung
QUIET=8      # minutes of silence before a build is called out as waiting on something
EVERY=30     # seconds between passes
BEAT=5       # minutes between status lines

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
# markers are the whole of what /board reads.
#
# They exist because a build that finished and a build that gave up look identical from out here.
# `claude -p` exits the moment the model ends its turn, and a model ends its turn to report
# progress as readily as to report a result; either way what is left behind is a log with prose
# in it and no process. Job 002 backgrounded a sixty-turn measurement, ended its turn at forty-nine
# to say how it was going, and took the measurement down with it - and because nothing had written
# down that it ended, /board called it "building" for the next twenty minutes.
say() { printf '· %s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$2" >> "$1"; }

# BSD and GNU disagree about everything except that both of these exist. Picked once, here, rather
# than at every call, and only these two functions know which platform this is.
if stat -f %m . > /dev/null 2>&1; then
  stats() { xargs -0 stat -f '%m %N' 2>/dev/null; }
  epoch_of() { date -j -f '%Y-%m-%d %H:%M:%S' "$1" +%s 2>/dev/null || echo 0; }
else
  stats() { xargs -0 stat -c '%Y %n' 2>/dev/null; }
  epoch_of() { date -d "$1" +%s 2>/dev/null || echo 0; }
fi

# The newest file a build has written, as "<epoch> <path>". One measurement answers all three
# questions out here - is it working, is it waiting, is it hung - because writing files is the only
# progress signal a `claude -p` gives from outside. Its log's mtime is the start time and nothing
# else until it exits, which is the trap the old wedged test fell into: a build still working at
# minute 91 was killed for being slow and one stuck at minute 3 was left alone.
#
# A fresh worktree is a whole checkout written at `add` time, so this has an answer from minute
# zero and the first number is never a mystery.
last_write() {  # $1 number
  { find "$JOBS/$1" \( -name .venv -o -name .git -o -name __pycache__ \) -prune -o -type f -print0
    find "$ROOT/factory/html/$1" -type f -print0
  } 2>/dev/null | stats | sort -rn | head -1
}

# Paths as a person reads them: the board's relative to the repo, a build's to its own worktree.
rel() {
  case $1 in
    "$ROOT"/*) printf '%s' "${1#$ROOT/}" ;;
    "$JOBS"/*) printf '%s' "${1#$JOBS/}" ;;
    *) printf '%s' "$1" ;;
  esac
}

# What a build is blocked on, read off the process tree - the one thing a quiet log cannot say.
# Job 003 spent eight minutes in a `ping cyclops.local` retry loop with its work already done, and
# from out here a wait and a hang look identical.
#
# Claude Code puts each Bash call's text in its own shell's argv, wrapped in a snapshot preamble
# and an `eval '<the real command>'`, so the payload is pulled out of the quotes and anything still
# carrying the preamble is dropped as unreadable. A `sleep` is the shape of a wait rather than the
# thing being waited on, so it is stepped over. Cosmetic and best-effort throughout: every line of
# it can come back empty without costing anything.
doing() {  # $1 pid -> one short line, or nothing
  d=$1 out=
  for _ in 1 2 3 4; do
    k=$(pgrep -P "$d" 2>/dev/null | tail -1)
    [ -n "$k" ] || break
    c=$(ps -o command= -p "$k" 2>/dev/null || true)
    case $c in *"eval '"*) c=${c#*eval \'}; c=${c%%\' <*} ;; esac
    case $c in *shell-snapshots*) c= ;; esac
    case $c in ''|sleep*|*"sleep "*) ;; *) out=$c ;; esac
    d=$k
  done
  if [ -n "$out" ]; then printf '%.110s' "$out"; fi
}

# What a job's frontmatter says right now, or "gone". head -8 keeps it to the frontmatter, so prose
# in the body saying `state: ready` is not an order.
state_of() {  # $1 number
  for jf in "$ROOT"/factory/"$1"-*.md; do
    [ -e "$jf" ] || continue
    head -8 "$jf" | sed -n 's/^state: *//p' | head -1
    return 0
  done
  echo gone
}

# Whether a log already has an ending. A build that wrote EXIT, DONE, STALLED, STOPPED or WEDGED is
# over, and from that moment its PID marker is a number the OS is free to hand to anything else -
# `kill -0` on it stops being a question about this build and becomes a question about a stranger
# who happens to have inherited the number. Job 023 finished on 2026-09-27 as pid 7113; five days
# later 7113 was Spotlight's mdworker_shared, so this loop counted a job sitting at `state: review`
# as a live build, measured its silence from the day it finished, and sent SIGTERM to the indexer
# once every pass - three of them landed before anyone saw it. The ending is the one fact here that
# cannot be read wrong: this loop writes it, and it is final.
#
# It also ends the repeat WEDGED line, which used to print every thirty seconds for ever: writing
# WEDGED gives the log an ending, so the next pass skips the block that wrote it. After a wedge
# kill there is nothing further this loop can do anyway - it sends one TERM and has no second move.
ended() {  # $1 log
  grep -q '^· [0-9][0-9-]* [0-9][0-9:]* \(EXIT\|DONE\|STALLED\|STOPPED\|WEDGED\)' "$1" 2>/dev/null
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
    # The Pi is not a build's to touch, and this is the half of that rule a machine can enforce:
    # deploy/push.sh refuses outright when it sees this. Prose in a skill is the other half, and
    # on its own it was not enough - job 003 deployed mid-build, collided with job 002 on the one
    # Pi, and then sat in an eight-minute ping loop waiting for a box that was off the network.
    # A build proves what it can on the Mac; the glass is /try's, and Marco's.
    export CYCLOPS_NO_PI=1
    nohup claude -p "/build $1" --dangerously-skip-permissions >> "$3" 2>&1 < /dev/null &
    cpid=$!
    say "$3" "PID $cpid"
    # `|| code=$?` and not a bare `wait`: under `set -e` a build that exits non-zero would take
    # this subshell with it, and the ending would go unwritten in exactly the case that needs it.
    code=0; wait "$cpid" || code=$?
    say "$3" "EXIT $code"
    # Three endings, because there are three, and the first version of this had two. `review` is
    # the build's contract met: the page, then the ticks, then the state, in that order and last.
    # `stopped` is build.md's other ending - a plan that cannot meet a criterion, which a build is
    # told to stop on rather than quietly improve. Anything else is a build that died somewhere in
    # the middle, whatever its exit status says, because a model that ends its turn to describe
    # what it is about to do next exits 0.
    #
    # Jobs 011 and 012 both stopped correctly on the same afternoon and both were written down as
    # STALLED, which is the word for a crash. Telling them apart cost a full reading of each log.
    # Two things that look identical from outside must not read the same - the rule this file was
    # built on, broken here until they made it obvious.
    st=$(state_of "$1")
    case $st in
      review)
        say "$3" "DONE - waiting on Marco"
        echo "· $1 done — /review $1" ;;
      stopped)
        say "$3" "STOPPED - the plan cannot meet a criterion"
        echo "· $1 stopped — the plan needs your word. /review $1" ;;
      *)
        say "$3" "STALLED - exited at 'state: $st' without setting review"
        echo "· $1 STALLED — /rework $1 to send it round again" >&2 ;;
    esac
  ) &
  echo "· $1 building — $2"
}

# One line about a build that is still going, into its log and onto the terminal. Written every
# BEAT minutes whether anything has changed or not, because "it is still going and here is what it
# last did" is the answer to the only question anyone has while waiting.
beat() {  # $1 number, $2 log, $3 pid, $4 minutes since START, $5 minutes silent, $6 last file
  if [ "$5" -ge "$QUIET" ]; then
    # Alive, and has written nothing for a while. Either it is waiting on something or it is
    # stuck, and the process tree usually says which.
    on=$(doing "$3" || true)
    if [ -n "$on" ]; then
      say "$2" "WAITING $4m - nothing written for ${5}m, on: $on"
      echo "· $1 waiting — ${5}m quiet at minute $4, on: $on" >&2
    else
      say "$2" "WAITING $4m - nothing written for ${5}m, no idea what on"
      echo "· $1 waiting — ${5}m quiet at minute $4 (killed at ${WEDGED}m)" >&2
    fi
  elif [ "$6" = - ]; then
    say "$2" "WORKING $4m - nothing written yet"
    echo "· $1 working — ${4}m in, nothing written yet"
  else
    say "$2" "WORKING $4m - last wrote $6 ${5}m ago"
    echo "· $1 working — ${4}m in, last wrote $6 ${5}m ago"
  fi
}

pass() {
  git worktree prune || true  # a job directory deleted by hand leaves a registration, and a
                              # stale registration makes the next `add` fatal
  # One stamp for the whole pass rather than one per job, so a status update is a moment in the
  # terminal - every live build on consecutive lines - instead of a dribble.
  if [ -z "$(find "$LOGS/.beat" -mmin -"$BEAT" 2>/dev/null | head -1)" ]; then
    beating=1
  else
    beating=
  fi
  live=0 waiting=
  now=$(date +%s)

  for f in "$ROOT"/factory/[0-9]*.md; do  # [0-9] so README.md is never taken for a job
    [ -e "$f" ] || continue               # an empty board is an unexpanded glob, not a file
    b=${f##*/}; b=${b%.md}; n=${b%%-*}; slug=${b#*-}; log=$LOGS/$n.log

    # State changes, written down as they happen rather than reconstructed afterwards from a
    # timestamp. `review` is the ending a build owns, `ready` is Marco's /rework, and either way
    # the board moved - the log is where anyone looks to find out when it moved and from what.
    st=$(state_of "$n")
    if [ -f "$log" ]; then
      was=$(cat "$LOGS/$n.state" 2>/dev/null || true)
      if [ "$st" != "${was:-}" ]; then
        if [ -n "${was:-}" ]; then
          say "$log" "STATE $was -> $st"
          echo "· $n $was → $st"
        fi
        printf '%s' "$st" > "$LOGS/$n.state"
      fi
    fi
    # Both endings wait on Marco, so both belong in the idle line. `stopped` was missing from it
    # for as long as `stopped` existed, which was about ten minutes.
    case $st in review|stopped) waiting="$waiting $n" ;; esac

    # Alive is the pid written into the log when the build started, never a pattern. `pgrep -f`
    # matches any shell whose argv carries the string, which includes the one asking - that is
    # how /board reported job 002 as building for twenty minutes after it had died. A pid cannot
    # match the process asking about it, and `kill -0` only asks.
    pid=$(sed -n 's/^· .* PID \([0-9][0-9]*\)$/\1/p' "$log" 2>/dev/null | tail -1)
    # An ended log is checked before the pid and not after: the point is never to ask about a
    # number this build has finished with. See ended().
    if [ -n "$pid" ] && ! ended "$log" && kill -0 "$pid" 2>/dev/null; then
      live=$((live + 1))
      lw=$(last_write "$n" || true)
      lwe=${lw%% *}; lwf=${lw#* }
      case $lwe in ''|*[!0-9]*) lwe=$now; lwf=- ;; esac
      began=$(sed -n 's/^· \(.*\) START .*/\1/p' "$log" 2>/dev/null | head -1)
      from=$(epoch_of "$began")
      # Silence is measured from the later of the last write and START, and that `later of` is
      # the whole of this line's history. A /rework reuses the worktree, so on the pass that
      # starts one every file in it is as old as the round that last used it - a day, a week -
      # and a build thirty seconds old looks like one that has written nothing for ninety
      # minutes. Job 002 was killed at thirty seconds on 2026-09-13 for exactly that. A build
      # that has just begun has not gone quiet; it has not started writing yet.
      if [ "$from" -gt "$lwe" ]; then lwe=$from; lwf=- ; fi
      silent=$(( (now - lwe) / 60 ))
      age=$(( from > 0 ? (now - from) / 60 : 0 ))
      if [ "$silent" -ge "$WEDGED" ]; then
        kill "$pid" 2>/dev/null || true
        say "$log" "WEDGED - killed, nothing written for ${silent}m"
        echo "· $n WEDGED — killed after ${silent}m idle. /rework $n to try again" >&2
      elif [ -n "$beating" ]; then
        beat "$n" "$log" "$pid" "$age" "$silent" "$(rel "$lwf")"
      fi
      continue
    fi

    # Read the state here and not from a list gathered up front: a pass takes seconds, and a
    # build that reaches `review` inside one would otherwise be started a second time.
    [ "$st" = ready ] || continue
    [ -f "$log" ] && continue                    # already had its swing; /rework clears the log
    pgrep -f "$(pat "$n")" > /dev/null && continue
    [ "$(pgrep -f 'claude -p /build ' | wc -l)" -lt $MAX ] || continue
    start "$n" "$slug" "$log"
  done

  if [ -n "$beating" ]; then
    if [ "$live" -eq 0 ]; then
      if [ -n "$waiting" ]; then
        echo "· idle — nothing building;$waiting waiting on you (/review)"
      else
        echo "· idle — nothing building, nothing ready"
      fi
    fi
    : > "$LOGS/.beat"
  fi
}

echo "· factory up — watching ${ROOT##*/}/factory, $MAX at a time, a status line every ${BEAT}m"
while :; do
  pass
  sleep "$EVERY"
done
