---
description: Put a job's branch on the Pi and restart the kiosk, so you can use it with your hands
argument-hint: <job number>
---

Job `$ARGUMENTS` goes on the Pi. **This is the only thing in the factory that touches it**, and it
happens because Marco asked, never on its own.

```sh
cd ../cyclops-jobs/NNN && deploy/push.sh && deploy/start-kiosk.sh
```

Then write `NNN` to `factory/ON-THE-PI` — one Pi, one job at a time, and `/ship` reads this to know
whether master needs putting back afterwards.

Tell him in one line **what to go and do**: the tap, the phrase, the thing to watch. That's what he
walked over for. Then: `/ship N` if it's right · `/rework N <what's wrong>` if not.

If the push fails, say so and leave the Pi alone — don't retry it.
