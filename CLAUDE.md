This is cyclops, a physical AI project that serves as an assistant for side projects.
It has a camera, a screen and a microphone and the main mode of interaction is via voice.
This project is supposed to run on a raspberry pi, which you can access via ssh at 'cyclops@cyclops'.
Once the pi is booted, the user "wakes" cyclops by pressing a button on the screen,
which opens a new session with OpenAI's realtime API.



Whenever you build something, it must be deployed there without asking.
None of this is required to work on Mac, the only platform that matters is the Pi.
No branching, all commits go straight to master.
Don't over-engineer this. It's a proof-of-concept. What matters is speed of iteration.

Design principles that matter for this project:
- Reduce cognitive load - Cyclops helps users focus on the task at hand, not the tool.
- Embrace familarity - Where possible, adopt well-established conventions and patterns to reduce the learning curve.
- CPU matters - The Pi heats quickly in its case. 