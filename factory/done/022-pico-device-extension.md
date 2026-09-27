---
state: done
opened: 2026-09-27
---

# A device extension for the Raspberry Pi Pico: Cyclops operates it and hot-swaps code on it

I have a raspberry pi pico plugged into cyclops. Let's build a device extension so that cyclops can
operate it for me and hot-swap code on it. E.g. I wanna put it on a breadboard and collaboratively
wire together a setup, with cyclops running the pico for me.

## Plan

### What is on the bus

Read off the Pi on 2026-09-27: `2e8a:0003` "Raspberry Pi RP2 Boot" - the RP2040 bootloader
(`INFO_UF2.TXT`: UF2 Bootloader v3.0, Board-ID RPI-RP2), auto-mounted at
`/media/cyclops/RPI-RP2`. It is **blank**: no MicroPython, no `/dev/ttyACM*`. It is a **plain Pico
(no W)** - his answer - so the firmware is MicroPython's `RPI_PICO` build and the onboard LED is
GP25 (`Pin("LED")` works too). With MicroPython on it, it enumerates as `2e8a:0005` with
`/dev/ttyACM0`; the `cyclops` user is already in `dialout`, so no udev rule is needed.

### One new file: `src/cyclops/extensions/pico.py`

Sits on job 016's extension system (`factory/done/016-device-extensions.md`), which already does
the prompt block, offering tools only while the device is plugged in, dispatch in a thread, the
panel caption and live re-arming on plug/unplug. **Read `extensions/__init__.py` and
`agent.py:_run_extension_tool` before implementing** - this repo moves under long plans.

- `usb_ids=("2e8a:0003", "2e8a:0005")`, name "Pico", category "other". Both states are the same
  device to Cyclops.
- Talks MicroPython's **raw REPL** over the tty with `pyserial` (the one new dependency, in
  `pyproject.toml`). Not `mpremote` - its Python API is not a stable surface. The raw-REPL
  exchange is small; keep it in this module.
- The tty is found from the device (e.g. `/dev/serial/by-id/*MicroPython*`, or sysfs under the
  matched device), not hardcoded.

### Three tools

- **`pico_program(code, name)`** - the hot-swap. Interrupts whatever runs, writes `code` as
  `main.py`, soft-resets, and returns what it printed in its first ~2 s, tracebacks included. It
  lives on the Pico, so it runs again on every power-up. Empty `code` stops it (writes an empty
  `main.py`).
- **`pico_run(code)`** - a one-off check ("is GP15 high?", "scan I2C"). Returns its output; gives up
  at 10 s with the partial output rather than hanging; then soft-resets so the saved program keeps
  running.
- **`pico_output()`** - what the running program printed since the last look (a background reader
  keeps the last ~200 lines), and whether it is running or died with a traceback. The reader blocks
  on the port, so it costs nothing at idle. Serial access from the tools and the reader is
  serialized by a lock.

When-to/when-not detail goes in each tool's description, not the system prompt.

### A blank Pico installs itself (his answer: install on first use)

The first tool call against a `2e8a:0003` device copies a pinned MicroPython `RPI_PICO` UF2 onto
the `RPI-RP2` drive (mount it with `udisksctl` if it is not mounted), waits for the tty to appear,
then carries on with the call. The UF2 is downloaded once from micropython.org at a pinned version
and cached under `~/.cache/cyclops/`. It takes ~10 s, so the tool descriptions and the extension's
bootloader-state instructions tell the model to say out loud that it is setting the Pico up first.

**The flash must not sound like a plug.** The board leaves the bus as `0003` and comes back as
`0005`; 015's bus watcher must not turn that into a "Pico unplugged / plugged in" turn. Both IDs
belong to the same extension and name, so the builder decides the mechanism (e.g. treat a
same-extension swap within the settle window as no change).

### Wiring together

While the Pico is plugged in, the extension's instructions carry a compact pinout - GP number ↔
physical pin (1-40, counting from the USB end, left side 1-20, right side 21-40), GND, 3V3, VBUS,
the onboard LED - and the rules: always name both ("GP15, that's pin 20, last on the left counting
from the USB end"), one wire at a time, 3.3 V logic, never 5 V into a GP pin.

### Nothing gets lost

Every `pico_program` writes its code to the live session's `code/HH-MM-SS_pico_<name>.py` (via
`session.current()`, `card.write_text`), logs a `session.note("code", ...)` record, and
`session.md` renders it as a code block (`session._render_record`). `code/` joins `card.KNOWN` and
the record kind joins `card.MADE`. No session live: nothing is written, the Pico is still
programmed.

### Not touched

The panel - no code view, no pinout picture (a pinout with the used pins lit is a good follow-up,
decided from a mockup). No proactive "your program crashed" announcement; the model learns it
from `pico_program`'s return or `pico_output`. Agent.py's extension seams, beyond what the flash
swap needs.

## Done when
- [x] The Pico's tools are offered while `2e8a:0003` or `2e8a:0005` is on a fake bus, and not
      otherwise. — pytest
- [x] Against a fake serial port, `pico_program` writes `main.py`, resets and returns the printed
      output, including a traceback when there is one. — pytest
- [x] `pico_run` returns its output and restarts the saved program; a snippet that never ends
      returns partial output at the timeout instead of hanging. — pytest, injected timeouts, no
      real sleeps
- [x] Each program lands in the session's `code/` and in `session.md`, and `code/` is not treated
      as a surprise by the card. — pytest
- [x] The `0003`→`0005` swap during a flash produces no plug/unplug turn. — pytest, fake bus
      sequence through the watcher and `add_bus_change`
- [x] The Pico's prompt cost is a stated number, and an empty bus leaves the prompt byte-identical
      to master's. — `tools/prompt_audit.py --usb`, both numbers said
- [ ] With a fake Pico on the bus, "blink the LED on the Pico" calls `pico_program` with code
      using the onboard LED, and "I've got an LED on GP15, where does the wire go?" answers pin 20.
      — new `talk_probe.py --script pico --runs 3` (3 live sessions, ~2 min; nothing more), tally
      in the outcome
- [x] `uv run pytest` passes and is no slower than master. — `uv run pytest`, both numbers said
- [ ] Blank Pico plugged in, "blink the Pico's LED": one line about the wait, then it blinks, with
      nothing else touched. — yours, on the Pi
- [ ] Breadboard: it tells you where an LED and resistor go, makes it fade, and "faster" swaps the
      program without a replug. — yours, on the Pi
- [ ] Unplug and replug the Pico: the last program runs again. — yours, on the Pi
- [ ] Kiosk CPU with the Pico printing in a loop is no higher than without it. — yours, on the Pi
      (`pidstat` before and after)
- [x] For all 8 orientations (USB left/right/up/down × chip up/down), pins 1, 20, 21 and 40 map
      to where the real board puts them, and with pin 1's column given, every pin maps to a
      breadboard hole. — pytest on the pure mapping function
- [ ] `pico_show` renders deterministically: the same arguments give byte-identical pictures, and
      it takes under 1 s on the Pi. — pytest for the bytes; the time, yours on the Pi
- [ ] The picture for "USB left, chip up, pin 1 in column 60, LED on GP15" puts GP15 at the bottom
      row's right end in column 41, like the approved spike. — the render, shown to Marco before
      any tests are written (`factory/mockups/022-pico-show-example.png` is the target)
- [x] With a fake Pico on the bus, "show me how to wire an LED to the Pico" calls `pico_show` and
      never `draw`. — `talk_probe.py --script pico --runs 3` (the same 3 sessions, one more line in
      the script; nothing more)
- [ ] On the breadboard, it tells you each lead as a hole ("GP15, column 41, row j"), the picture
      matches the board in front of you, and you get every lead in first time. — yours, on the Pi

## Built — 2026-09-27
The Pico extension is one new module (`extensions/pico.py`) with `pico_program`, `pico_run` and
`pico_output` over MicroPython's raw REPL (pyserial), a pinout and wiring rules in its
instructions, and first-use install of MicroPython v1.29.0 (pinned, cached in `~/.cache/cyclops/`).
Flash-swap mechanism, as the plan left to me: extensions can declare `rejoin_s`, and the bus
watcher keeps such a device listed for that long after it drops off, and treats its IDs as one
device (keyed by extension name) - so `0003`→gap→`0005` is no turn at all, while an endoscope
unplug is untouched. The cost: a real Pico unplug reaches the model up to 30 s late (silently,
as every unplug does). Programs land in `code/HH-MM-SS_pico_<name>.py` and render in `session.md`
as a python block; `code` is in `card.KNOWN` and `card.MADE`.
Numbers: prompt with a Pico +900 tok (9,956 → 10,856: instructions +490, tools +410); empty bus
instructions+tools byte-identical to master (same sha). Probe (3 live sessions): blink →
`pico_program` with `Pin("LED")` 3/3; GP15 → "pin 20, last on the left" 3/3. Tests: 1038 passed
in 19.4 s vs master 1029 in 19.4 s (the suite was already over its 10 s target on master).
Noticed, not fixed (out of scope): the bus names leak - greeting and rail say "Board in FS mode" /
"RP2 Boot", not "Pico"; letting an extension's name win for its IDs would be a small follow-up.
The four Pi criteria are unticked: never run against a real board. Expect ~10-15 s silence after
the "setting it up" line on a blank Pico (first download of the 680 KB UF2 included).
Hands-on: `/try`, then with the blank Pico: "blink the Pico's LED"; breadboard LED on GP15 + fade +
"faster"; unplug/replug; `pidstat` on the kiosk with a printing loop.
factory/html/022.html

## Feedback — 2026-09-27

observe the last session. Every time it draws a wiring diagram, the pins looked different / were in a different place. once the pico is plugged in, cyclops should know exactly where which pin is in whatever orientation and accurately guide me

(Shown `factory/mockups/022-pico-show-example.png`, rendered by `factory/mockups/022-pico-show-spike.py` - plain PIL from the pin table, no image model:) wow yes, that's what I want.

What that means for the build: every wiring picture in session `2026-09-27_11-04-12_pico-external-led-blink` came from `draw` (gpt-image) and put GP15/GND somewhere different each time, twice on the wrong side. Replace it for the Pico with a `pico_show` tool that renders the Pico on a breadboard in PIL, like the spike: the real 40-pin table, the orientation (USB direction, chip up or down - asked once, or read from a photo they took) and the breadboard column pin 1 sits in (asked once), every pin mapped to a real hole, used pins lit, parts (resistor, LED, wire) on real holes, a legend in holes. Voice guidance names holes ("GP15, column 41, row j"). While a Pico is plugged in, Pico wiring never goes to `draw` - say so in the Pico tools' descriptions. The spike's row lettering and pin-1 column were read loosely off his photo; the real tool takes them as arguments.

## Built — 2026-09-27 (round 2: pico_show)
`pico_show` renders the Pico on their breadboard in PIL from the 40-pin table, as in the spike.
Its arguments are USB direction (left/right/up = away from you/down), chip up/down and pin 1's
column. It takes pin 1's row letter or pin 20's column as optional overrides. The layout is kept
for the session, so it is asked once. Parts (resistor/LED/wire/part) end in holes like "37j" or
in named pins ("GP15", "GND" = nearest ground), which resolve to the outermost free hole in that
pin's strip. The tool refuses a hole under the Pico, an unnamed hole in a pin's strip, and two
leads in one hole, and says why. The picture goes into the session's photos as a `_drawn` jpg
and onto the panel. The tool answers with every lead as a hole, which is what the voice uses.
Pico wiring "never draw" is in `pico_show`'s description and in one clause of the extension's
instructions.
Decided on the way: pin 1's column alone does not say which way the column numbers run. I used
the printed-board rule (numbers upright means 1 on the left and j at the top; turning the board
keeps that), plus the 63-column edge when only one direction fits. If his board is printed
differently, `pin20_column` settles it. The centre channel is now the real 3 pitches, not the
spike's 2, so c↔h is 0.7 in like the real Pico. The agent's panel slot is not told about a
`pico_show` picture (agent.py seams untouched), so "edit that picture" will not find it.
Probe line: "Show me how to wire an LED to the Pico" plus the layout in the same sentence. A bare
"show me" makes the model ask for the layout, which is the right behaviour but gives no call to
count.
Numbers: probe (3 live sessions) wiring → `pico_show`, never `draw`: 3/3. Blink: 3/3.
**GP15 → "pin 20": 0/3, down from 3/3.** The model asked how the Pico sits instead of naming the
pin, so I have unticked that line. I fixed the description (name the pin first, then ask), but
the fix is not re-probed because the budget was 3 sessions. The probe also showed the model
putting two leads in one hole and using a hole in GP12's strip. The tool now refuses both, with
tests, but that is not re-probed either. Prompt with a Pico: 9,956 → 11,639 tok (+1,683;
pico_show ~745). Empty bus: instructions and tools byte-identical to master (same sha). Tests:
1048 pass in 19.9 s (master: 1029 in 21.7 s). Render: 0.06 s on the Mac.
Not ticked: the render vs the spike is Marco's call (shown on the page). I wrote no picture
tests, only the mapping and byte-determinism ones the criteria name. The <1 s on the Pi (I
expect ~0.3 s) and the breadboard walk-through need `/try`.
Hands-on: `/try`, Pico on the breadboard: "show me how to wire an LED to the Pico". It should
ask how it sits, then the picture should match the board and every lead go in first time. Also
ask "where does GP15 go?" before giving the layout. It should say pin 20 first.
factory/html/022.html
