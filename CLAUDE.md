This is cyclops, a physical AI project that serves as an assistant for side projects.
It has a camera, a screen and a microphone and the main mode of interaction is via voice.
This project is supposed to run on a raspberry pi, which you can access via ssh at 'cyclops@cyclops.local'.
Once the pi is booted, the user "wakes" cyclops by pressing a button on the screen,
which opens a new session with OpenAI's realtime API.


Whenever you build something, it must be deployed there without asking.
Deploy early and often – I want to see the results of your work as soon as possible, so I can steer you if things are not to my linking.
None of this is required to work on Mac, the only platform that matters is the Pi.
No branching, all commits go straight to master.
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
