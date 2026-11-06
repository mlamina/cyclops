# The background worker

Say "find the manual for X and download it" and Cyclops says it's on it and keeps talking. Claude
Code does the job in the background, the PDF lands in [Manuals](manuals.md), and Cyclops tells you
how it went. This is a prototype of one job: finding and downloading a PDF. It answers one question:
is handing work off by voice useful, and does it run safely on the box?

`CYCLOPS_WORKER=1` in `.env` offers the tool. It's off by default, because without the service
below the tool could only say it isn't available.

## How a job moves

```
voice session    background_job tool   -> writes inbox/<id>.json, answers at once
cyclops-worker   one `claude -p`       -> jobs/<id>/work/result.pdf, then done/<id>.json
cyclops-index    jobs.collect          -> checks the file, lands it in manuals/, records the result
voice session    at the next pause     -> tells you, once
```

The folder is `/var/lib/cyclops-worker`. `~/.cache/cyclops/jobs.json` is Cyclops's own ledger of
what's running and what you haven't heard about yet. The index service takes results in because it
is always running, so a job finishes properly with no session open.

- **Session still open:** you're told at the next pause: nobody speaking, no reply and no tool
  running, two seconds of quiet.
- **Session ended:** you're told in the next one, after you've spoken, so a session still opens with
  nothing said.
- **Once:** a telling counts when the reply finishes. One you talk over is told again at the next
  pause. If the power goes in that gap you could hear it twice, which beats never.
- **One at a time:** a second request gets "still on the other one".
- **Power cut:** the worker restarts by itself, answers for the job that was cut off, and doesn't run
  it again. A request older than two minutes at start-up is treated the same way, not run late.
- **Worker dead:** a job with no answer 20 minutes on is settled as failed, so it can't block the next.
- **No screen and no settings.** It uses the panel's existing task line and the Manuals screen.

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
| 15-minute limit, killed as a process group | `worker.py` | a job that never ends |

The firewall loads in `ExecStartPre`: if it won't load, the worker doesn't start.

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
- The name is cleaned before it's used, and the description is capped and handed to the voice model
  marked as a description. Why a job failed is always one of our own words, never the worker's text.

**What this doesn't stop:**

- A web page can steer the worker into fetching the wrong file, or into sending out anything it can
  read: that's only its own folder and its token.
- Text from a fetched PDF or a description reaches a model that can read his notes. It's marked and
  capped, which reduces the risk and doesn't remove it. The librarian, which reads PDFs, can only
  read manuals.
- Reading a PDF that landed costs OpenAI money outside the job's budget, about a penny per 45 pages.

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
