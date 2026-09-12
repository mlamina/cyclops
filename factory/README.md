# The factory

A queue of small jobs, and a worker that takes them from a sentence to something running on the Pi.

One file per job, one folder, numbered. The number is the name: *"factory, build 7"*.

## Your side of it

You only ever write English. **You never edit `state:` and you never move a file** — the factory
does that, so there is no bookkeeping to forget.

| You want to | You do |
|---|---|
| Log an idea, a bug, a feature request | `tools/capture.sh "..."` or `/capture ...` |
| See what's going on | `/factory` |
| Answer the factory's questions | Type under them in the file. Nothing else. |
| Send a build back round | Write what's wrong under `## Feedback` |
| Accept a build | `/factory done 7` |

## Its side of it

`/factory` walks the board and moves everything it can, then stops and tells you what's waiting
on you.

| `state:` | Means | Who moves it |
|---|---|---|
| `new` | Captured, nothing read yet | factory — plans it |
| `asking` | Plan written, questions open | **you** — answer in the file |
| `ready` | Nothing left to guess at | factory — builds, commits, deploys |
| `review` | Running on the Pi, waiting for your eyes | **you** — accept or send back |
| `done` | Accepted; file moves to `done/` | — |

## The one rule that matters

**At most one job in `review` at a time.** Generating code stopped being the bottleneck a while
ago; reading it is. A factory that builds five things while you're out has not saved you an
afternoon, it has cost you one. So it plans as far ahead as it likes and builds one thing.

Everything else here is a convention and can be broken by saying so.
