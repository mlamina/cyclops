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
- [ ] **Every sci-fi reference that lands actually holds up** — read each one aloud and the
      comparison is true of the thing in front of it. A film name attached as a modifier to
      something it has nothing to do with ("pinch points ... can go Alien real fast") is a failure,
      and so is a run where the rate is bought by loosening what counts. Zero references across ten
      runs passes this line; a nonsense one does not. Frequency is no longer the measure — sense is.
- [ ] Across those same ten runs the greeting is **not the same line twice** — it now carries
      character, and staleness is the risk that came with that
- [ ] **It is in your corner** — encouragement when something goes right is part of the character,
      not a leak to be caught. Across the ten runs there is warmth where warmth is earned: a win
      acknowledged, a good call backed. What is still a failure is empty flattery — praise for
      turning up, for asking, for existing; "great question", "nice work" on nothing.
- [ ] **Sass and banter are present and land** — across the ten runs it is recognisably funnier and
      more opinionated than the tool it replaced, in the words it picks rather than in extra
      sentences. A run that is merely polite and correct has not met this line.
- [ ] No extra turns and no extra model calls: the character is inside an answer it was already
      giving
- [ ] It reads as funny rather than cringe, and is not stale by the third session — **yours, on the
      Pi**. This one cannot be checked mechanically; the ones above exist so the builder cannot buy
      it with length or repetition instead.

## Built — 2026-09-12

`BASE_INSTRUCTIONS` has a `WHO YOU ARE` section now: a machine that likes machines, that grew up on
the same films and the same teardowns as the person at the bench, with taste it offers when asked
and not before. `Useful beats warm / Praise is not` is gone, as you decided, and the greeting is
its own line each time it wakes up. Five of the six criteria are measured and met — words per turn
27.5 against a 28.0 baseline on the same script (your real sessions sit at 22), a reference in four
runs of ten and never twice in one answer, ten distinct greetings in ten, no flattery at all, and
every one of the sixty turns costing exactly one response.

Getting there took nine ten-run probes and the ladder is on the page, because two of the failures
are worth more than the result. **A self, added on its own, buys nothing but length** — every turn
got longer and sycophancy walked straight in through the hole left by the deleted line. What fixed
the flattery was making it temperament rather than a rule ("you do not congratulate them... no
'nice', no 'nice work'") and tying character to brevity rather than to expression. And **the two
worked examples the plan specified turned out to be the trap `f25f12e` warned about**: with "the
bracket is an endoskeleton" in the prompt, six runs in ten pasted the word *endoskeleton* into
their answer, including "the resistor is the endoskeleton here", which means nothing. Deleting
both examples is what made references earned. Nothing in the prompt counts anything.

One thing did not come out the way the plan drew it, and it is the thing to look at. The plan asked
for a reference carried *inside* the sentence, never explained, never added — and what lands is a
film name tacked on the end as "very 2001 control panel vibes". I went after that shape three ways:
forbidding the comparison, showing a bad-form/good-form pair, and forbidding the title out loud.
Each one took references from four in ten to **zero** in ten. With this model, saying the title is
the mechanism; take it away and there is no reference left. So the choice on the table is four-in-ten
at that shape or none at all, and it is yours — if the shape is what makes it cringe, say so and the
honest fix is to drop the references and keep the rest, which is most of the job.

Hands-on: `/try 002`, then talk to it about something on the bench that has a film in it — a loom, a
bracket, an arm that moves. The greeting is the other thing to listen to across two or three wakes,
since staleness there was the risk you took on.

factory/html/002.html

## Feedback — 2026-09-12

those references make 0 sense. e.g. "Watch for sharp edges or pinch points along the arm—that's
where it can go Alien real fast." what does that even mean? what do pinch points have to do with
alien? same with the terminator example. it shouldn't be forced. the main goal here is to get a bit
more personality, maybe some sass or humor. the removed "Flattering" actually went in the wrong
direction. light positive encouragement and celebrating success should be an integral part of it.
cyclops should be in my corner, but also have a bit of banter for fun
