# The factory

A queue of small jobs, and a worker that takes them from a sentence to something you can look at.

One file per job, one folder, numbered. The number is the name: *"factory, put 7 on the Pi"*.

## Your side of it

You only ever write English, and **you review outcomes, never code.** No diffs, no PRs. What
comes back is a picture of the panel, a number, or the Pi itself doing the thing.

| You want to | You do |
|---|---|
| Log an idea, a bug, a feature request | `tools/capture.sh "..."` or `/capture ...` |
| See the board, and start work | `/factory` |
| Answer the factory's questions | Type under them in the file |
| Try one with your hands | `/factory pi 7` |
| Send one back round | Write what's wrong under `## Feedback` |
| Ship it | `/factory done 7` |

You never edit `state:` and you never move a file. Bookkeeping you can forget is bookkeeping
that rots.

## Its side of it

| `state:` | Means | Who moves it |
|---|---|---|
| `new` | Captured, nothing read yet | factory — plans it |
| `asking` | Plan written, questions open | **you** — answer in the file |
| `ready` | Nothing left to guess at | factory — builds it in its own worktree |
| `building` | An agent has it right now | factory |
| `review` | Built, with something to look at | **you** — accept, or send back |
| `done` | Merged to master and on the Pi | — |

## Branches and worktrees

Every job gets a branch, `job/007-slug`, and its own worktree at `../cyclops-jobs/007/`. That's
what lets several run at once without treading on each other, and what makes "throw this one
away" free.

**The board itself only ever lives on master.** Jobs change code in their worktree; the factory
writes the board from the main checkout. Nothing to merge, nothing to conflict.

Accepting a job merges it to master and deploys master. Master is still what the Pi runs.

## The one rule that matters

**One job on the Pi at a time** — not a policy, a fact about there being one Pi.
`factory/ON-THE-PI` says which one is on it.

So: building fans out as wide as there's work for, because a screenshot costs you five seconds to
judge. Getting your *hands* on something is the scarce part, and that queue is one deep.

Everything else here is a convention and can be broken by saying so.
