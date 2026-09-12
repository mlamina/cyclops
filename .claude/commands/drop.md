---
description: Bin a job — branch, worktree and all
argument-hint: <job number>
---

Marco doesn't want job `$ARGUMENTS`.

```sh
git worktree remove --force ../cyclops-jobs/NNN
git branch -D job/NNN-slug
```

Remove `factory/logs/NNN.log`, set the job file to `state: dropped`, `git mv` it into
`factory/done/`, and commit.

**Keep the file.** "We tried that and didn't like it" is worth more in six months than a tidy
folder, and it stops the same idea being captured twice.

Nothing ever reached master, so there is nothing to revert. One line, then `/board`.
