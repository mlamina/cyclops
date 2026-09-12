---
description: Log an idea, a bug or a feature request to the factory board and get straight back to what you were doing
---

Capture `$ARGUMENTS` as a factory job. **This is an interruption — make it a short one.**

```
tools/capture.sh "$ARGUMENTS"
```

Then, only if this session already knows something the sentence alone doesn't — the file and line
we were just looking at, the error text on screen, the screenshot path — append it to the new file
under a `## Context` heading. One or two lines. Don't go looking: if you'd have to read code to
write it, leave it for the planning step.

Reply with the filename and nothing else. **Do not plan it, ask about it, or offer to build it.**
Capturing has to stay free or it stops happening.
