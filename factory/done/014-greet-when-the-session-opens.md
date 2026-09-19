---
state: done
opened: 2026-09-18
---

# Cyclops says its welcome message as soon as the session is active

cyclops should say its welcome message as soon as the session is active, without me having to say anything

## Plan

**What is wrong today.** Nothing asks the model to speak when the session opens. It only replies
after he has talked (server VAD `create_response`), so the greeting gets glued onto its first
answer. The two most recent sessions on the Pi show it:

- "Hey, can you draw a smiley face on this scratchpad?" → *"Seventh run today; keep it tidy."* → smiley
- "Hey, can you pull up a picture of my BMW?" → *"Three minutes between runs is barely a reboot."* → photo

**The change**, mostly in `VoiceAgent._on_session_ready` (`agent.py` ~1995):

1. **Ask for the greeting as soon as the session is up.** The `session.updated` that opens the lid
   also requests one response, with no user turn in front of it. The request sets
   `tool_choice: "none"` (a per-response param in the pinned SDK), so the greeting can only be
   words, never a tool call. A reconnect does not greet again - `_on_session_ready` already
   returns early once `ready` is set.
2. **Never talk over the lid.** `iris_open` lasts 1.67 s (`self.cues.play` returns it), and the
   model's thinking time overlaps it. Greeting audio that arrives before the lid sound ends is
   held until it ends, then plays. Measure first: if the probe shows the model is never faster
   than the lid (+ `sfx.SETTLE_S`), do not build the hold, and put the numbers in the outcome.
3. **A wake where only Cyclops spoke still counts as nothing.** Today `card.said_or_made`
   treats a lone `cyclops` record as content. With a greeting on every wake, every accidental tap
   would become a kept folder, get named and summarised by the after-child, and push up the
   "nth time today" count in `session.now_context`. So a session where only Cyclops spoke and
   nothing was made is `empty` in `card.triage`, and `SessionLog._discard` removes it at close
   like a tap-and-stop today. This cannot delete an existing folder: until now Cyclops only ever
   spoke after a `you` turn, and photos (`MADE`) and failed transcriptions (`transcript_failed`)
   already count on their own.
4. **The probes stop saying "Hey."** `tools/talk_probe.py` (every script, and `--wakes` via
   `_one_wake`) and `cyclops-smoke` turn 1 currently start the greeting by typing "Hey." / "Say
   hello". They wait for the greeting that arrives on its own instead, recorded as turn 0.
   Otherwise every measurement after this job would be measuring the old behaviour.

**Not touched:** what the greeting says (job 002's rules in HOW YOU TALK and the WHEN IT IS
block), the mic staying deaf until the lid has finished (`_listen(after_s=...)`), and the
existing barge-in handling. If he speaks over the greeting, it stops and his question gets
answered, like any other line. The `cut.worth_asking` thresholds already counted a greeting;
leave them.

## Done when

- [x] When the session comes up, exactly one response is requested with no user turn. A second session-ready event requests nothing. — pytest, fake connection
- [x] The greeting request carries `tool_choice: "none"`. — pytest, same fake
- [x] A log with only `session` + `cyclops` + `end` triages as `empty`; add one `you` line and it is `finished`. — pytest on `card.triage`
- [x] If the hold is built: audio arriving before the lid sound ends is fed to the speaker only after it ends, and audio after that plays at once. If it is not built: the 10-run probe shows no greeting audio earlier than 1.77 s after ready. — pytest with a fake speaker and clock, or the probe numbers in the outcome
- [ ] The greeting arrives no later than an ordinary first answer: time from ready to first greeting sound vs. time from question to first answer sound, median and worst of each over 10 runs. — `talk_probe.py --runs 10`, both numbers in the outcome
- [x] The greeting is its own turn. In 10 runs of wake → greeting → "Hey, can you draw a smiley face on this scratchpad?", the reply never mentions the hour, day, gap or run count. — new `talk_probe.py` script, all 10 transcripts in the outcome
- [ ] The unprompted greeting still meets job 002's bar: no line repeated, none opens on a hello word, and none asks what he is working on or hands him the floor. — `talk_probe.py --wakes` + `--wake-tally` + `--props`
- [x] The probes no longer type "Hey." to get a greeting, and `cyclops-smoke` turn 1 waits for the greeting instead of asking for one. — `cyclops-smoke` on the Mac, turn 1 passes
- [ ] Tap to wake, say nothing: it speaks as the lid finishes opening, not over the lid sound and not after a noticeable pause. — yours, on the Pi
- [ ] Tap to wake and start talking straight away: the greeting stops, and the question gets answered without a second greeting. — yours, on the Pi
- [ ] Tap on, hear the greeting, tap off: no new session folder, and the next wake's count does not include it. — yours, on the Pi
- [ ] `uv run pytest` passes in under ten seconds. — `uv run pytest`

## Built — 2026-09-18

Cyclops now speaks first. The `session.updated` that opens the lid also asks for one response with
`tool_choice: "none"`. A reconnect doesn't ask again. Measured first: the model had its greeting ready
before the lid sound ended in 35 of 36 sessions (0.81–1.42 s after ready, plus one at 3.41 s). So the
hold is built. Voice that arrives while the lid sound is playing is held, and plays at lid + `SETTLE_S`
(1.77 s), the same moment the mic opens. A log where only Cyclops spoke now triages as `empty`, so a
tap-on-and-off is removed at close and never counted as a wake. The probes and `cyclops-smoke` wait for
the greeting as turn 0 instead of typing one. `--script smiley` is new.

Stopped because two measured lines fail, and the plan rules out the only fixes I can see:

- **Greeting vs first answer (line 5).** Over 10 runs, the greeting's first sound came a median 0.96 s
  and worst 1.20 s after ready. The first answer came a median 0.61 s and worst 1.15 s after the
  question. Across all 36 greetings: median 1.05 s, worst 3.41 s. Master's first answer, which used to
  carry the greeting: median 0.80 s, worst 0.90 s. A side run with `reasoning: minimal` on the greeting
  request didn't help (median 1.03 s). The greeting just takes about a second. On the Pi the lid covers
  that, except in the rare slow session. To meet the line as written, the greeting would have to be
  requested before `session.updated` (saving about one round trip), and even that may not be enough.
  The alternative is to accept "ready before the lid ends" as the bar.
- **Job 002's bar (line 7).** 16 of 16 distinct, no hello openers, and the word-count check passes. But
  one wake in 16 hands him the floor: "Third run today—long pause, so what's next?" Three more lean
  that way. Elsewhere: "Show me what's on the docket." (bench) and "What are we fixing or building
  right now?" (smoke). On master, with "Hey." typed first, the same 16 wakes had none. Getting to zero
  means changing the greeting rules in HOW YOU TALK / WHEN IT IS, and this plan said not to.
- **Suite time (line 12)** is 17.2 s. Master takes 17.5 s on the same Mac, so it was over before this
  job. Job 013 found the same.

Worth a dry run once on the Pi: `--recover` now also treats a Cyclops-only folder as empty. The plan
says none exist on the card, and `cyclops-sessions --recover --dry-run` would list any before they go.

Hands-on: `/try 014`. Tap to wake and say nothing: the first word should start right as the lid
sound stops. Tap and talk just after the lid, over the greeting: it should stop and answer you. Tap
on, hear the greeting, tap off: no new folder, and the next greeting's count shouldn't include it.
factory/html/014.html
