Cyclops — your workshop droid. A one-eyed robot that works with you. It remembers every project,
has read every manual, and answers any question — so your hands stay free and your phone stays
in your pocket.

Why it exists: an assistant like Claude is enormously helpful on a project, but it lives in a
phone, a tablet, an app - all built for sitting down, none of them for when your hands are full.
Cyclops brings that help to where the work is, and because it is just talking, people who aren't
technical can use it as easily as anyone. README.md has the full version.

Core values - the tiebreakers. When two designs are both reasonable, the one that keeps these wins:
- Hands stay free - voice in, voice and the screen out. If it needs a keyboard, a phone or a menu, it's wrong.
- It never becomes the task - a setting to tend, a menu, a nag: that is attention Cyclops asked for, and it has failed.
- Nothing gets lost - every session is written down; what was decided, measured and left open is there next week.
- Good company - it has a face and a character. More fun to work with than to work without.
- It does what you expect - talking to it works the way talking works. Nothing to learn, nothing to be shown.

Criteria for success - each is counted in misses:
- The phone stays in the pocket - reaching for another device mid-session is a miss.
- One question, one answer - "what?", "say that again", "no, the other one" is a miss.
- Zero time on Cyclops itself - turns about settings, restarts, "why didn't you" tend to none.
- No unexpected waits - answers at the speed of talking; anything slower is announced before it starts.
- Fun and surprise - a session with no surprise in it is one an app could have done.
- No learning curve - having to work out what it wants, how to phrase it, when it's listening, is a miss.

This is a physical AI project.
It has a camera, a screen and a microphone and the main mode of interaction is via voice.
This project is supposed to run on a raspberry pi, which you can access via ssh at 'cyclops@cyclops.local'.
Once the pi is booted, the user "wakes" cyclops by pressing a button on the screen,
which opens a new session with OpenAI's realtime API.


Whenever you build something, it must be deployed there without asking.
Deploy early and often – I want to see the results of your work as soon as possible, so I can steer you if things are not to my linking.
None of this is required to work on Mac, the only platform that matters is the Pi.
Working by hand: no branching, all commits go straight to master. Factory jobs are the exception —
each gets a branch and a worktree so several can run at once (`factory/README.md`), and accepting
one merges it back to master, which is still what the Pi runs.
Don't over-engineer this. It's a proof-of-concept. What matters is speed of iteration.
I don't care about details like pixels, alpha values, etc. Keep your responses simple,
short and outcome-focused.
Anytime you want to show me an image, don't leave it in your scratchpad. Copy it into 
~/Downloads.

Design principles that matter for this project:
- Reduce cognitive load - Cyclops helps users focus on the task at hand, not the tool.
- Embrace familarity - Where possible, adopt well-established conventions and patterns to reduce the learning curve.
- CPU matters - The Pi heats quickly in its case. 
- Gender-agnostic - Cyclops is "it", not he or she.

Tests:
- Run them with `uv run pytest`. Not bare `pytest` - that resolves to another project's venv.
- The whole suite stays under ten seconds. If it doesn't, that's a bug in the code, not a
  reason to delete tests: find the setup being paid for over and over, and cache it.
- Speed matters more than coverage. A fast suite that gets run beats a thorough one that doesn't.
- Never build an expensive object per test. If the setup costs more than the assertion,
  share it or memoise it.
- Test behaviour, not wording. Don't assert on prose in prompts, CLI output or labels, and
  don't grep our own source for an exact line - that's a lint rule, and it fails on a reformat.
  Machine-facing text is fair game: file formats, argv, SDK enums, asset names.
- One claim, one test. Two tests that fail for the same reason are one test.
- Anything needing a camera, a mic, a key, Chromium or the Pi is not a pytest test. It goes in
  cyclops-smoke, or in tests/*.mjs which are run by hand.
