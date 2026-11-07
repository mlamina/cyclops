# The background worker

The voice model can hand any bigger job to Claude Code: research across pages, comparing options,
finding and fetching something, working with files, small programs. Cyclops says it's on it and keeps
talking, Claude Code does the work in the background, and Cyclops tells you how it went. Ask how it's
going and you get the job's real state.

The model chooses by what the job is, from `background_job`'s description and the BIGGER JOBS block in
its instructions: what takes several steps or minutes goes to the worker; a quick fact, and anything
about your notes, the camera, the screen or the conversation, stays with it. "Look online" for one
fact is still `web_search`.

`CYCLOPS_WORKER=1` in `.env` offers the tool. It's off by default, because without the service
below the tool could only say it isn't available.

## How a job moves

```
voice session    background_job tool   -> writes inbox/<id>.json, answers at once
cyclops-worker   one `claude -p`       -> jobs/<id>/progress.json as it goes, any file in work/result,
                                          then its report in done/<id>.json
cyclops-index    jobs.collect          -> checks any file, lands a PDF in manuals/, records the result
voice session    check_background_job  -> how it's going, when you ask
voice session    at the next pause     -> tells you, once
```

The folder is `/var/lib/cyclops-worker`. `~/.cache/cyclops/jobs.json` is Cyclops's own ledger of
what's running and what you haven't heard about yet. The index service takes results in because it
is always running, so a job finishes properly with no session open.

- **Session still open:** you're told at the next pause: nobody speaking, no reply and no tool
  running, two seconds of quiet.
- **Session ended:** you're told in the next one, after you've spoken, so a session still opens with
  nothing said.
- **Once:** a telling counts when its audio has finished playing. If you talk over it, or the reply
  fails, or the power goes before then, you're told again at the next pause. Stopping it with a finger
  on the glass counts as heard: you chose to. You may hear one twice if the power goes in the last
  second.
- **One at a time:** a second request gets "still on the other one".
- **Power cut:** the worker restarts by itself, answers for the job that was cut off, and doesn't run
  it again. A request older than two minutes at start-up is treated the same way, not run late.
- **Worker dead:** a job with no answer 20 minutes on is settled as failed, so it can't block the next.
- **No screen and no settings.** It uses the panel's existing task line and the Manuals screen.

## What comes back

The job's last message is a short report, and whether it got the job done. It may also leave one file,
at a name that says nothing about its kind.

| The job left | What happens |
|---|---|
| A report | It's told, in the voice model's own words |
| A PDF, and maybe a report | All the checks below, then it lands in Manuals. Cyclops says so only once this code has seen it land |
| Any other file, with a report | The report is told, and that the file wasn't kept. Only PDFs land for now, and the job doesn't know that: code decides, from the bytes |
| Neither | Told as a failure |
| "I couldn't", or a limit hit | Told as a failure, with its report if there is one, and never as a success |

## "How's that going?"

`check_background_job` answers from what's really there: time from Cyclops's own ledger, then steps so
far and what it's doing now, counted by the worker from Claude Code's stream.

**What it's doing is only ever our own words** ("reading a web page", "running a command"), never the
page or the command. The job can write its own progress file, so anything richer would be a way for a
web page to speak to the voice model. A forged file can only lie about a count and pick from that list.

A dead worker shows as a job whose last step keeps getting older, until it's settled as failed. A job
that has just finished is told at the next pause. The check never carries its report.

## A report is free text off the web

It's going to a model that can read his notes and send things out, through `web_search` or through a
new `background_job`, which goes straight to a worker on the open web. So:

- **Cleaned and capped** on Cyclops's side, because the job can forge its record: one line, no invisible
  characters, no brackets, at most 1,500 characters.
- **Marked** as a report and never an instruction.
- **Told with no tools.** That one reply is created with `tool_choice: none`, so whatever it says, the
  reply can only be spoken.
- **Forgotten.** The item holding it is deleted from the conversation when that reply ends. After that
  the model has only what it said aloud, which he heard.
- **Gone if he talks.** The server answers him by itself when he stops, with every tool on offer. So the
  report is checked for before it's sent and again after, and deleted the moment he starts speaking. It's
  told again at the next pause.
- **Only the telling counts.** That reply carries a marker the server echoes back. No other reply marks a
  job as told, and no reply that had a report in front of it can run a tool.
- **One way in.** Only the telling carries a report.

The cost of forgetting: asked for more detail afterwards, the model has only what it said, so the
telling asks it to include the specifics.


There's no way to remove a manual on the Manuals screen. Delete its folder on the card.

## The wall

Claude Code runs here with no permission prompts, reading the web. So its settings aren't what keeps
it in: the operating system is.

| Wall | Where | Holds against |
|---|---|---|
| Its own account, not in any group of Cyclops's | `install-worker.sh` | reading `/home/cyclops` (mode 700): notes, sessions, `.env` |
| `ProtectSystem=strict`, `ReadWritePaths` | the unit | writing outside its own folder |
| `ProtectHome=yes` | the unit | the same again for every home folder |
| `NoNewPrivileges=yes`, not in `sudo` or `docker` | the unit | getting admin rights or using Docker |
| Firewall by account | `cyclops-worker.nft` | the admin page and companion, which have no login and serve sessions over the network, and the rest of the LAN |
| `PrivateDevices=yes` | the unit | the camera, mic and anything plugged in |
| `CPUQuota`, `CPUWeight`, `Nice`, `TasksMax` | the unit | starving a live conversation of CPU |
| `LimitAS=4G` per process, `OOMScoreAdjust=1000` | the unit | running the box out of memory: no process past 4 GB of address space, and the worker dies first if it's short |
| 15-minute limit | `worker.py` | a job that never ends |
| Everything on its account killed when a job ends | `worker.py` | a process that detached itself going on using the internet and the token |
| Requests only count from another account | `worker.py` | a job queuing more jobs for itself, each with a fresh budget |
| Nothing of `/run` but the watchdog's socket (`TemporaryFileSystem`) | the unit | local services the firewall can't see: printing on his printer through CUPS, avahi, the hotkey daemon, and the system bus, where polkit lets a job turn on lingering, and what that leaves runs in a user manager, outside everything above |
| A new home for each job | `worker.py` | one hijacked job planting a hook that runs in every later one |
| `WatchdogSec`, heard only from the main process | the unit | a job freezing the program that enforces its limit |
| Can't be attached to (`PR_SET_DUMPABLE`) | `worker.py` | a job, on the same account, writing the worker's memory or ptracing it to undo its limit |
| Folder put back before every start, as root | `cyclops-worker-repair.sh` | a job wrecking the folder so the service crash-loops until a person comes |

The firewall loads in `ExecStartPre`: if it won't load, the worker doesn't start.

The job runs as the same account as the program watching it, so it can write anything in the
folder and can kill that program. If it does, systemd stops everything in the service and restarts
it, and the job is answered as cut off. If it freezes that program instead (`kill -STOP`), the
limit stops counting, so the watchdog restarts the service within a minute. If it wrecks the folder,
a root step puts it back before every start, and what can't be answered is passed over.

**There's no memory limit on the job as a whole.** This kernel boots with `cgroup_disable=memory`, so
`MemoryMax` would be accepted, shown back by `systemctl show`, and enforced by nothing. Getting a real
one means editing the boot line and rebooting.

**Nothing from the worker is believed.** `cyclops.jobs` runs as `cyclops`, which can read everything,
and the worker chooses what's at the end of the path. So:

- No part of the path is followed through a symlink, and only a regular file is accepted. A link to
  `.env` would otherwise copy it into his manuals.
- It must start as a PDF and open as one, within 128 MB and 500 pages. What's checked and what's kept
  are the same bytes.
- A landing never replaces a manual. `manuals.receive` overwrites, so a name in use gets a number.
- The answer file gets the same care: no links, a regular file only, at most 64 KB. A pipe or a huge
  file there settles the job as failed instead of holding the ledger's lock.
- The name is cleaned and capped, the description is capped, and both reach the voice model marked as
  a label. Why a job failed is always one of our own words, never the worker's text.

**What this doesn't stop:**

- A web page can steer the worker into fetching the wrong file, or into sending out anything it can
  read: that's only its own folder and its token. It can also steer what the report says, so a report
  can be wrong: that's a lie told to him, in the voice model's words.
- For the one reply that tells it, a report is in front of a model that can read his notes. That reply
  has no tools, so the model can act on it only by speaking. Whatever it says aloud stays in the
  conversation, so a report that gets itself repeated aloud can still reach later turns, where tools are
  back.
- Text from a fetched PDF reaches a model that can read his notes, through the librarian, which can
  only read manuals. It's marked and capped, which reduces the risk and doesn't remove it.
- Reading a PDF that landed costs OpenAI money outside the job's budget, about a penny per 45 pages.
- Parsing a landed PDF happens under the ledger's lock, in the index service, outside the worker's
  limits. The worst of five hostile 128 MB files took 15.9 s and 1.4 GB on the device. A request made
  in that window waits, and the audio carries on.
- Nothing caps what a job writes to disk.

## The token

The worker runs on Marco's Claude subscription, with a one-year token from `claude setup-token`
(`CLAUDE_CODE_OAUTH_TOKEN`) at `/etc/cyclops-worker/claude-token`, owned by the worker, mode 0400.
The docs say it can only make model requests. It isn't a copy of the login on any machine.

**So no separate bill and no monthly cap.** Jobs share his Max limits with whatever else runs on that
account. Each job has `--max-budget-usd 1`, but that's checked against Claude Code's own estimate, and
it's only checked between turns: on the device, a $0.0001 budget stopped a job after one turn,
having spent about $0.065. So it's a soft limit, and the one to rely on is 15 minutes, one job at a
time. No other unprivileged account can read the token; `cyclops` can, because it has admin rights.

## Install

```bash
sudo deploy/install-worker.sh           # account, folder, firewall, service
sudo deploy/install-worker.sh --remove  # and back out
journalctl -u cyclops-worker -b
```

`deploy/push.sh` refreshes the installed `worker.py` and firewall on a box that has them.
