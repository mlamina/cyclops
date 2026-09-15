#!/bin/sh
# What the factory cannot say about itself, as it happens.
#   factory/watch.sh
# One line per thing worth waking someone for, and silence for everything else - a watcher that
# narrates is a watcher nobody leaves running. Every line it prints is an event for whoever is
# reading it; /run-factory turns each one into a diagnosis and, where the pipeline is at fault,
# a fix and another round.
#
# It never starts, stops or judges anything. It reads the markers loop.sh writes and the pids
# those markers name, and it says the four things the markers cannot: that a job is stranded with
# no ending, that a ready job is not being started, that a build ended badly, and that the loop
# itself has gone.
#
# `set -u` and deliberately not `set -e`. A watcher that exits on a transient `ps` failure goes
# quiet, and quiet is exactly what it looks like when everything is fine - the one failure mode
# this must not have.
set -u

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
LOGS=${WATCH_LOGS:-$ROOT/factory/logs}
BOARD=${WATCH_BOARD:-$ROOT/factory}
EVERY=${WATCH_EVERY:-20}    # seconds between passes
SETTLE=${WATCH_SETTLE:-4}   # passes a ready job may sit unstarted before it is called out
MAX=${WATCH_MAX:-3}         # loop.sh's own ceiling, so "are there free slots" has an answer
SEEN=$LOGS/.watch           # byte offsets and already-said flags

mkdir -p "$SEEN"

# The loop, and not one of its own supervisors. `start()` backgrounds a subshell and a forked sh
# keeps its parent's argv, so loop.sh's own guard pattern matches both - measured on 2026-09-13,
# pid 75807 the loop and 86009 a supervisor for job 002, command lines identical. Getting this
# wrong is not cosmetic: a surviving supervisor with no loop above it means nothing is starting
# new jobs, which must read as down rather than up.
#
# A supervisor is forked by the loop, so its parent pid is the loop's pid and is therefore in this
# same list. That is the whole test, and it deliberately reads no command line: the first version
# asked whether the parent's argv mentioned `loop.sh` and called the real loop a supervisor on its
# first run, because Claude Code's Bash tool carries the command it was given in its own shell's
# argv - the same trap as loop.sh:120-124, one level up.
loop_pid() {
  if [ -n "${WATCH_PID:-}" ]; then
    kill -0 "$WATCH_PID" 2>/dev/null && echo "$WATCH_PID"
    return
  fi
  all=$(pgrep -f 'sh .*factory/loop\.sh$' 2>/dev/null)
  for p in $all; do
    pp=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
    case " $(echo $all) " in
      *" ${pp:-x} "*) ;;             # parent is another match: this is a supervisor
      *) echo "$p"; return ;;
    esac
  done
}

# Said once, until the condition clears. Flags live beside the offsets so a restarted watcher
# does not repeat an hour of history into the terminal.
once() { [ -e "$SEEN/$1" ] && return 1; : > "$SEEN/$1"; }

state_of() {  # $1 number - head -8 keeps it to the frontmatter, as loop.sh does
  for jf in "$BOARD/$1"-*.md; do
    [ -e "$jf" ] || continue
    head -8 "$jf" | sed -n 's/^state: *//p' | head -1
    return 0
  done
  echo gone
}

pid_in() { sed -n 's/^· .* PID \([0-9][0-9]*\)$/\1/p' "$1" 2>/dev/null | tail -1; }

# Builds alive right now, counted off the pids in the logs and never off `pgrep`. board.md's rule
# and the bug that paid for it: pgrep -f matches the shell asking, and reported job 002 as
# building for twenty minutes after it had died.
live_builds() {
  c=0
  for jf in "$BOARD"/[0-9]*.md; do
    [ -e "$jf" ] || continue
    b=${jf##*/}; n=${b%%-*}
    p=$(pid_in "$LOGS/$n.log")
    [ -n "$p" ] && kill -0 "$p" 2>/dev/null && c=$((c + 1))
  done
  echo "$c"
}

# New markers since the last pass. A log found on the **first** pass is primed rather than
# replayed: a watcher armed onto a factory that has been running all afternoon would otherwise open
# with an hour of endings that are already on the board. What is in flight at arming time is
# /run-factory's to report, once, from the board itself.
#
# Only the first pass, and that is the whole of this function's history. Priming every log it had
# not seen before meant a log written *while* the watch was running - which is to say every build
# that starts from now on, START marker and all - was primed away unread, and the one event Marco
# actually waits for after a /capture never fired.
scan() {  # $1 number, $2 log
  f=$SEEN/$1.off
  size=$(wc -c < "$2" 2>/dev/null | tr -d ' ')
  case ${size:-} in ''|*[!0-9]*) return ;; esac
  if [ ! -e "$f" ] && [ -n "$first" ]; then printf '%s' "$size" > "$f"; return; fi
  off=$(cat "$f" 2>/dev/null || echo 0)
  case $off in ''|*[!0-9]*) off=0 ;; esac
  [ "$size" -lt "$off" ] && off=0        # /rework truncates the log; start again from the top
  printf '%s' "$size" > "$f"
  [ "$size" -le "$off" ] && return
  tail -c "+$((off + 1))" "$2" 2>/dev/null | while IFS= read -r line; do
    case $line in '· '*) ;; *) continue ;; esac   # the model's own prose shares this file
    rest=${line#· }; rest=${rest#* }; rest=${rest#* }
    case $rest in
      START*)   echo "BUILDING $1 ${rest#START * - }" ;;
      DONE*)    echo "REVIEW $1" ;;
      STALLED*) echo "STALLED $1 ${rest#STALLED - }" ;;
      WEDGED*)  echo "WEDGED $1 ${rest#WEDGED - }" ;;
      'EXIT '*) c=${rest#EXIT }; [ "$c" = 0 ] || echo "EXIT $1 $c" ;;
    esac
  done
}

# A log whose last marker is the pid, with that pid gone. Nothing wrote an ending, so from the
# markers alone this is indistinguishable from a build still going. It is what the README's
# `pkill -f 'factory/loop\.sh$'` leaves behind: the pattern matches the supervisors too, they
# take SIGTERM untrapped, and the nohup'd build carries on writing prose into a log that will
# never get EXIT, DONE or STALLED.
stranded() {  # $1 log
  last=$(sed -n 's/^· [0-9-]* [0-9:]* \([A-Z][A-Z]*\).*/\1/p' "$1" 2>/dev/null | tail -1)
  [ "$last" = PID ] || return 1
  p=$(pid_in "$1")
  [ -n "$p" ] && ! kill -0 "$p" 2>/dev/null
}

had= miss=0 first=1
while :; do
  lp=$(loop_pid)
  # Two passes, not one. This is the only event that ends the watch, so a transient `pgrep` or
  # `ps` failure must not be able to stop it - a watcher that has quietly exited looks exactly
  # like a factory with nothing to report.
  if [ -z "$lp" ]; then
    miss=$((miss + 1))
    if [ "$miss" -ge 2 ]; then
      echo "LOOP-DOWN ${had:-none} - nothing is starting jobs"
      exit 0
    fi
    sleep "$EVERY"
    continue
  fi
  miss=0
  had=$lp
  live=$(live_builds)

  for jf in "$BOARD"/[0-9]*.md; do   # [0-9] so README.md is never taken for a job
    [ -e "$jf" ] || continue         # an empty board is an unexpanded glob, not a file
    b=${jf##*/}; b=${b%.md}; n=${b%%-*}
    log=$LOGS/$n.log
    st=$(state_of "$n")

    if [ -f "$log" ]; then
      scan "$n" "$log"
      if stranded "$log"; then
        once "$n.stranded" && echo "STRANDED $n $(pid_in "$log") - log ends at PID, process gone, no ending written"
      else
        rm -f "$SEEN/$n.stranded"
      fi
    fi

    # Ready, never started, and room to start it. The loop gates new builds on
    # `pgrep -f 'claude -p /build '` (loop.sh:290), so any peer session whose argv merely carries
    # that text counts as a build and fills the last slot - and nothing is written down when it
    # does. Given SETTLE passes to be sure it is not just a pass that has not come round yet.
    if [ "$st" = ready ] && [ ! -f "$log" ] && [ "$live" -lt "$MAX" ]; then
      c=$(cat "$SEEN/$n.wait" 2>/dev/null || echo 0)
      case $c in ''|*[!0-9]*) c=0 ;; esac
      c=$((c + 1)); printf '%s' "$c" > "$SEEN/$n.wait"
      [ "$c" -eq "$SETTLE" ] && echo "NOT-STARTING $n - ready, no log, $live of $MAX slots in use"
    else
      rm -f "$SEEN/$n.wait"
    fi
  done

  first=
  sleep "$EVERY"
done
