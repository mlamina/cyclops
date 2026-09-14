---
state: done
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

- [x] Median words per Cyclops turn does not rise — baseline off the 412 turns on the Pi with
      `interrupted` turns dropped and mic-checks gated out by `cut.worth_asking`; after measured on
      the same scripted turns
- [x] **Every sci-fi reference that lands actually holds up** — read each one aloud and the
      comparison is true of the thing in front of it. A film name attached as a modifier to
      something it has nothing to do with ("pinch points ... can go Alien real fast") is a failure,
      and so is a run where the rate is bought by loosening what counts. Zero references across ten
      runs passes this line; a nonsense one does not. Frequency is no longer the measure — sense is.
- [x] Across those same ten runs the greeting is **not the same line twice** — it now carries
      character, and staleness is the risk that came with that
- [x] **The greeting knows when it is** — it is built from the time of day, the day of the week,
      how long since the last session ended, and which session of the day this is. A Monday
      morning first wake and a Friday afternoon fifth wake do not get interchangeable lines, and
      two minutes after the previous session does not sound like three days after it. Checked by
      driving the wake with made-up clocks and session histories, not by ten runs at one moment:
      the same situation twice reads consistently, different situations read differently, and no
      line claims something the clock does not support.
- [x] **It is in your corner** — encouragement when something goes right is part of the character,
      not a leak to be caught. Across the ten runs there is warmth where warmth is earned: a win
      acknowledged, a good call backed. What is still a failure is empty flattery — praise for
      turning up, for asking, for existing; "great question", "nice work" on nothing.
- [x] **Sass and banter are present and land** — across the ten runs it is recognisably funnier and
      more opinionated than the tool it replaced, in the words it picks rather than in extra
      sentences. A run that is merely polite and correct has not met this line.
- [x] No extra turns and no extra model calls: the character is inside an answer it was already
      giving
- [x] **No prop word and no stock opener.** Ten distinct greetings turned out to be ten variations
      on the same two words: every line opened "Morning" or "Back again", and nearly every line
      reached for "the bench". A noun that recurs across most lines is a crutch, not a character,
      and it is the tell that the prompt is trying too hard. Counted across the ten runs and the
      sixteen wakes: no single noun or opening phrase carries more than a small minority of the
      lines. It also must not assume a workshop — not everyone has a bench, and a line that only
      works if you do is a failure. Cool, not eager.
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

## Built — 2026-09-12 (rework)

The forced references are gone. Not tuned, deleted: the whole paragraph that told it to say a
film name when it saw one is out of `BASE_INSTRUCTIONS`, and the films stay only as things it
grew up on. Zero references in 120 probe turns, which is what the reworked criterion asks for.
Worth knowing why it is zero and not "better ones": last round established that with this model
saying the title *is* the mechanism, and three separate attempts to keep the rhyme while losing
the name-drop each took the rate to zero anyway. The shape you objected to and the rate came
together, so the rate went.

What replaced it is two paragraphs that both pay for themselves. **A view, led with** - "either
can work, it depends" is named in the prompt as a shrug, and the verdict goes in the first
sentence. That turned out to be the shortening lever as well: the which-voltage turn fell from
81.5 words to 48, because committing is cheaper than surveying. **In your corner**, with
flattery called the opposite of encouragement rather than a milder version of it - a win gets
acknowledged in sixteen of twenty runs, always attached to what actually happened, and there is
no praise for asking, trying or turning up anywhere in 120 turns.

The question you actually asked - is it funnier - got read blind, because a comparison you make
knowing which side you wrote is not one. 100 lines, this prompt's and the one it replaced,
shuffled with the labels stripped, marked for "carries a committed view" and "carries wit", then
unmasked: view 30% -> 70%, wit 2% -> 22%. Not one of the old prompt's fifty lines carried both;
eleven of the new fifty do. `tools/talk_probe.py` grew `--blind` and `--score` for this.

One criterion is missed and it is the median: 29.0 against a 28.0 baseline. The whole of the
rise is the greeting (2.0 -> 8.5) and the acknowledgement (25.0 -> 29.5), which are the two turns
this job was asked to lengthen; every turn you actually wait through got shorter, mean 30.8 ->
27.1 and longest turn 100 -> 70. The statistic is also blunt at this sample size - bootstrap CI
about +-8 words, and two ten-run batches of one identical prompt landed at 28.5 and 31.5 - so I
have reported it as a miss rather than arguing it away, and the answering-turn number is beside
it. Buying it back would mean shortening the greeting or the acknowledgement, which is undoing
the job.

Hands-on: `/try 002`, then two things. Hold up something you have just finished and listen to
what comes back - that is the turn the whole rework hangs on. And ask it to pick between two
ways of doing something, because the bluntness is where most of the character ended up.

factory/html/002.html

## Feedback — 2026-09-12

for every session, it should get more context to create a greeting that lands even more
realistically. time of day, time since last session, day of week. that way, first session on monday
morning gives a differnt greeting than the 5th session friday afternoon. and after days of no
sessions, a different greeting should come than 2 minutes after the previous session

## Built — 2026-09-12 (greeting)

It was never told what time it was. That is the whole finding: with no clock in the prompt at
all, ten runs out of ten opened with "Morning", at any hour, and the only thing varying was the
wording. There is a `WHEN IT IS` block now, built at connect time from the card and the clock -
the weekday and the hour, how long since the last session switched off, and which wake of the day
this is. Sixteen wakes across eight made-up situations, each driven twice: "three days off and
we're back at the bench" on Monday morning, "fifth wake today" on Friday afternoon, "two minutes
was barely a blink" straight after the last one, "back on the bench after a few weeks away" after
three weeks. Sixteen distinct lines, every claim one the clock supports, and the pairs land on the
same thing twice.

The gap is measured from when the last session *ended*, off its log, and not off `summary.md` -
that file is written by the detached child a model round trip later, so the session that ended two
minutes ago has none, and two minutes ago is exactly the case you asked for.

Two things cost the four probes on the page. **A list of facts gets read as a ranking.** As three
bullets - hour, count, gap - it took the top one every time: twenty-four wakes and not one named a
day or a count. The same facts as three plain sentences with the count at the *end* produced
"Saturday switch-on", "fifth wake today" and "back for a third round tonight" in the next eight.
And **more to say let an old leak back in**: five wakes in eight started ending with "what are we
getting into?", which HOW YOU TALK has forbidden since long before this job. Moving that one rule
to where it is read last took it to zero - though one wake in sixteen still finished with "go
ahead", so it is not airtight.

The median that was a miss last round came back under the line on its own: 26.5 words a turn
against the 28.0 baseline, mean 25.9, and the longest thing said in sixty turns fell from 100
words to 59. Ten of ten distinct greetings, sixty turns each costing exactly one response, no film
references, and warmth still attached to what actually happened.

What is left for you is the thing that was always yours: every line still opens "Morning" or "Back
again". The situation lands; the first two words are a habit.

Hands-on: `/try 002`, then switch it on and off and on again - the short gap is the one that reads
best. Better still, come back to it tomorrow morning and listen to the first line then.

factory/html/002.html

## Feedback — 2026-09-13

there is an immense overuse of the term "bench". this "the bench" reference is not how real humans
talk. also, not everyhone has a bench. and it feels like it's trying to hard. keep it cool

## Built — 2026-09-13 (the prop word)

The word was in the prompt. Nine times: "the person at the bench", "the thing on the bench today",
and six more scattered through the tool descriptions - it was reading its own vocabulary back at
you. Those are gone, and in their place a rule that it does not know what room you are in and
cannot see it: a kitchen table, a driveway, a car park, a spare room with the carpet rolled back.
One greeting in twenty-six still says "the bench", against seven in twenty-six before, and shop,
workshop, garage and shed appear nowhere at all. The first word is no longer a hello either -
"Morning", "Evening", "Hey" and "Back" are banned by name, one line in twenty-six still opens on
one against twenty before - and the line is framed as an observation about the moment rather than
a greeting with the moment stapled on: one remark, eight words, nothing after it.

Three things the probes settled, and the first is the one worth keeping. **A ban that quotes the
word teaches the word**, unless the model was already leaning on it hard. "Never say the bench"
took it from eight wakes in sixteen to one. The same shape of rule naming "first run today" took
that phrasing from five in sixteen to nine, and "nothing is quiet" - written to stop it describing
a room it has not been shown - put that adjective at the front of six greetings in ten, in a batch
where it had never appeared before the rule against it existed. **The wording of a fact is the
wording of the greeting**: "This is the 5th time they have switched you on today" came back as
"fifth run today" nine times in sixteen, and saying it as a thing about your day instead took the
noun with it. And **the emptiest fact gets made the whole line** - "you have not been on yet
today" is not an observation about anybody, so the count is only told from the second wake on, and
the gap, which is different every single time, is read last.

One number went the wrong way and it is the median: 29.0 words a turn against the 28.0 baseline,
where last round was 26.5. Nothing in this round touched how long an answer is; the mean fell
(25.1 against 30.8) and the longest thing said in sixty turns is unchanged at 59 against 100. Two
batches of one identical prompt landed at 28.5 and 31.5 earlier in this job, so a word is inside
the noise - but it is a rise, and it is on the page rather than argued away.

Hands-on: `/try 002`, then switch it on and off and on again a few times across the day - the
greeting is the whole of this round and sixteen made-up clocks are not your Saturday. The tail is
the thing to listen for: three lines in twenty-six still finish by handing you the floor, which
nine separate bans have not taken to zero.

factory/html/002.html
