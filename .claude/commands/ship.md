---
description: Merge a job to master, deploy it, and tear the worktree down
argument-hint: <job number>
---

Marco accepted job `$ARGUMENTS`.

```sh
git merge --no-ff job/NNN-slug        # on master, in the main checkout
deploy/push.sh && deploy/start-kiosk.sh
git worktree remove ../cyclops-jobs/NNN && git branch -d job/NNN-slug
```

Then set the job file to `state: done`, `git mv` it into `factory/done/`, remove
`factory/logs/NNN.log`, and clear `factory/ON-THE-PI` if it named this job. Commit the board.

If the merge conflicts — master moves under these worktrees, peers commit here constantly — stop
and say so. Don't resolve it silently.

One line at the end: shipped, and what `/board` still has waiting.
