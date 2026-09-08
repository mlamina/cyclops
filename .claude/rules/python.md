---
paths:
  - "src/**/*.py"
  - "tests/**/*.py"
---

# Writing Python here

- **One module, one job.** If you can't say in a sentence what a file owns, it is two files.
- **Open a module with a docstring** that says what it owns and what it must never do. The
  invariant that isn't written down is the one the next change breaks.
- **Short, single-purpose functions.** One that both decides and acts is two functions.
- **Annotate the boundaries** - parameters and return types on anything another module calls.
  Inside a function, let the reader infer.
- **Push I/O, hardware and global state to the edges** so the logic in the middle stays pure
  and testable without a Pi attached.
- **Return early.** Guard clauses beat nested branches.
- **Comments say why, never what.** The code already says what.
- **Don't abstract before the third use**, and don't copy-paste after it.

## Code Smells / Anti-patterns

What to avoid, and why:
- **Magic numbers** - use named constants or enums. A number in code is a bug waiting to happen.
- **Large modules** - if a file is more than 200 lines, it is probably doing two jobs. Split it.
- **Lots of loose functions** - Pure functions are hard to maintain if they are scattered across the repo. Use OOP or a single module to group them.
