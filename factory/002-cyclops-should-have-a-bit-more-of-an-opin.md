---
state: ready
opened: 2026-09-12
---

# Cyclops should have a bit more of an opinionated personality — nerdier, sci-fi, never forced

Cyclops should have a bit more of an opinionated personality right now. It feels very generic, but
I feel like we can make it a bit more nerdy for example, I feel like it would be funny if every now
and then it threw in a terminator or matrix reference something that feels fun and relatable to
people like me who are tinkerers and sci-fi nerds. of course nothing that slows down work or feels
like unnecessary or forced chatter

## Plan

The problem is not missing jokes. `BASE_INSTRUCTIONS` (`agent.py:835-991`) has a `HOW YOU TALK`
section that is twenty lines of *restraint* — "Answer first", "End when the answer ends", "Saying
nothing is a real option", "Useful beats warm... Praise is not." Every line was earned, and
together they describe a tool. Nothing anywhere says who it is. **The generic feeling is designed
in**, and it was designed in on purpose: `74d01a1` replaced an earlier "friendly, quick-witted
voice assistant with one eye" because it was "a good operating manual attached to nobody in
particular." The fix is not to put that back — it is to give the droid a self that the existing
restraints can sit on top of.

**Read `f25f12e` before writing a word of prompt.** It is this exact job, done once already, with
numbers: across 40 sessions, 65 of 168 spoken turns closed with an offer, because a rule meant to
*cap* a behaviour "read as permission to spend one offer per turn, and it got spent every turn."
That is precisely how a personality instruction becomes one dutiful forced joke per session — the
thing Marco explicitly does not want.

So:

**Give it a self.** A short passage in `BASE_INSTRUCTIONS` — a machine that likes machines, that
grew up on the same films the person at the bench did, with taste of its own. The Terminator and
Matrix references are then a *symptom* of who it is, not a feature bolted on.

**A reference replaces words, it never adds them.** It lands inside the answer it was already
giving — the same sentence, differently coloured — or it does not happen. That is what makes it
compatible with "one or two sentences" and with hands that are busy and dirty.

**No frequency language anywhere in the prompt.** Not "occasionally", not "every now and then",
not "sparingly". Per `f25f12e`, a cap reads as a licence. The trigger is the situation genuinely
rhyming with something — the bracket really does look like an endoskeleton, the loom really is
going into the back of someone's head. Earned, not scheduled. Nothing counts anything.

**Decided with Marco, not to be relitigated:**

- **Cut `Useful beats warm. A number, a caution, the next step - that is the help. Praise is not.`
  entirely** (`agent.py:882`). His call, against a recommendation to keep the anti-praise half. The
  cost is that sycophancy is what models do by default, so there is an acceptance criterion below
  that exists purely to catch it coming back.
- **Opinions about the work only when asked.** It keeps the existing "speak up when you can see a
  mistake coming", but it does not volunteer a preferred method unasked. Taste is there when you
  reach for it.
- **The greeting carries character too.** "Hey Marco." is no longer the whole shape of a first
  turn. The risk he accepted is that the most-repeated line in the product goes stale fastest,
  which is what the greeting criterion below is for.

**Where it goes.** `BASE_INSTRUCTIONS` only — one constant, assembled by `build_instructions`
(`agent.py:1135`). Its prose is **entirely untested** and free to rewrite. Do not break the two
things that are pinned: `test_about.py:162` asserts the `WHO YOU ARE TALKING TO` header, and
`test_caption.py` pins the panel caption phrases. Neither is in scope. Keep character out of the
tool descriptions — those carry rules, not voice.

**Measuring it.** The corpus is on the Pi, not here: 87 folders, 83 usable, **412 Cyclops turns**,
~59k characters at `cyclops@cyclops.local:~/cyclops/sessions/`. `push.sh` excludes `sessions/`
deliberately, so the job starts with an rsync down. Walk it with `library._folders` →
`card.read_log` → `session.transcript_text`; `cyclops-sessions --tidy` is the working template for
"walk every session, judge each one, tally". Two traps in the denominator: of 98 sessions on this
card 55 never needed a model and 26 had no turns at all, so gate mic-checks out with
`cut.worth_asking`; and a turn marked `interrupted` was never fully heard, so it cannot count.

For the after-measurement, **`cyclops-smoke` will not do** — its turn 2 hard-requires a camera and
takes turns 3 and 4 down with it. Build the small text-only driver instead: `VoiceAgent` +
`send_text` + a `TurnObserver`-shaped sink collecting `response.output_audio_transcript.done` is
about thirty lines, and turns 1, 3 and 4 of smoke are already pure text.

## Done when

- [ ] Median words per Cyclops turn does not rise — baseline off the 412 turns on the Pi with
      `interrupted` turns dropped and mic-checks gated out by `cut.worth_asking`; after measured on
      the same scripted turns
- [ ] Across ten runs of one scripted conversation, a sci-fi reference lands in a **minority** of
      runs and never twice in one answer — the check that it is earned rather than scheduled
- [ ] Across those same ten runs the greeting is **not the same line twice** — it now carries
      character, and staleness is the risk that came with that
- [ ] **No praise or flattery in any of the ten runs** — "Praise is not" is being deleted, so this
      is the line that catches what that costs
- [ ] No extra turns and no extra model calls: the reference is inside an answer it was already
      giving
- [ ] It reads as funny rather than cringe, and is not stale by the third session — **yours, on the
      Pi**. This one cannot be checked mechanically; the five above exist so the builder cannot buy
      it with length or repetition instead.
