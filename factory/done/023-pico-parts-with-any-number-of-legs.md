---
state: dropped
opened: 2026-09-27
---

# pico_show handles components that are not LEDs or resistors

how does this handle components which are not LEDs or resistors?

(Asked after job 022 shipped. Today anything else is a generic two-hole "part" box: a pot, a
sensor, an I2C module, a servo or an IC cannot be drawn whole; at most 6 parts, the rest dropped
silently; every resistor is drawn with 330 Ω bands whatever its value. The proposal below was
agreed: "go ahead".)

## Plan

All inside `src/cyclops/extensions/pico.py`'s `pico_show` (job 022). **Read it before
implementing** - `resolve`, `legend`, the parts loop in the renderer and the `SHOW` schema. Nothing
else in Cyclops changes.

### Parts with any number of legs

A part may carry `leads` - a list of `{name, to}` - instead of `from`/`to`:

- a pot: `A→35j, W→GP26, B→GND`
- an OLED: `VCC→3V3, GND→GND, SDA→GP4, SCL→GP5`
- a servo: `signal→GP16, +→VBUS, −→GND`

Every lead goes through the same resolution and refusals a two-leg part's ends do today: a real,
free hole; one lead per hole; nothing under the Pico; a hole in a pin's strip must be named as
that pin. `to` takes a hole ("37j") or a Pico pin ("GP4", "GND") exactly as `from`/`to` do.

It is drawn as a **labelled block with its leads fanning out to their holes**, each lead named at
the block end. The block never covers the Pico, another part's block, or any hole in use - so a
sensor on long jumpers sits in the free margin, and a pot whose legs are in three adjacent holes
sits over them without hiding them. Two-leg `from`/`to` parts, LEDs, wires and the layout maths
are unchanged.

### Every lead in words

The legend and the tool's answer carry **one line per lead** - "OLED · SDA → GP4 (pin 6), column
55, row j" - which is what Cyclops reads out, one lead at a time. The answer also returns the
leads as structured data (part, lead name, hole, pin if any), not only prose.

### Resistor bands from the value

A resistor takes `ohms`, and its bands are computed from it instead of the fixed 330 Ω ones.
**5-band, 1 %** codes, because that is what his kit is (brown-orange-black-black-brown = 130 Ω,
read out in session `2026-09-27_11-04-12`). No value: drawn plain, no bands guessed.

### No silent drops

Asking for more parts than the picture can hold answers the model with an error ("too many for one
picture - split it") instead of cutting at six.

### Schema

`SHOW`'s `parts` items gain `leads` and `ohms`; the description says when to use `leads` (anything
with more than two legs or pins) and that it is still the only way to show Pico wiring. Keep what
it adds to the standing prompt small and say the number.

## Done when
- [x] A part with N named leads puts each lead in a real hole, and 022's refusals (taken, under
      the Pico, in a pin's strip without naming the pin) apply to every lead. — pytest
- [x] The tool's answer lists every lead with its name and hole. — pytest on the returned data
- [x] 130 Ω and 330 Ω give the right 5-band colours, and a resistor with no value still draws. —
      pytest
- [x] Asking for more parts than fit returns an error the model sees, and nothing is dropped. —
      pytest
- [x] No block covers the Pico, another block, or a hole in use. — pytest on the drawing geometry
- [ ] A picture with an OLED on I2C, a pot on an ADC pin and an LED reads at a glance: every lead
      goes to the hole its label says. — the render, shown to Marco before any tests are written
- [x] What the schema change adds to the prompt with a Pico plugged in is a stated number. —
      `tools/prompt_audit.py --usb`
- [x] "Wire an SSD1306 OLED to the Pico over I2C" (layout given in the same sentence) calls
      `pico_show` with one part carrying 4 leads, to 3V3, GND and an SDA/SCL pair on one I2C bus.
      — `talk_probe.py --script pico --runs 3`: one more line in the existing script, still 3
      sessions (~2 min; nothing more). The tally also says 022's "GP15 → pin 20" line again.
- [x] `uv run pytest` passes and is no slower than master. — `uv run pytest`, both numbers said
- [ ] A part of three or more legs from the bench, on the breadboard: the picture matches it and
      every lead goes in first time. — yours, on the Pi

## Built — 2026-09-27
A part can now carry `leads` (a list of `{name, to}`) instead of `from`/`to`. Every lead goes
through 022's resolution and refusals. A GND lead picks the ground nearest the part's other
leads. The part is drawn as a labelled block, each lead named where it leaves the block and
running to its hole. The block is placed as close to its holes as it can get without covering the
Pico, another block, an LED or resistor, or any hole in use. When the board has no room it moves
off the board into the dark margin. Blocks draw over wires, never under them. The legend and the
answer give one line per lead (`OLED · SDA → GP4 (pin 6), column 55, row j`), and the answer
also returns `wiring` as data: part, lead, hole, pin. Resistors take `ohms` and draw 5-band 1%
codes; with no value they are drawn plain, and a resistor given a value but no label is labelled
"330 ohm resistor". Once the legend has more than four lines it splits into two columns, and long
lines shrink to fit. "Too many" is an error in three cases: more than 6 parts (one per palette
colour), more legend lines than fit, or a picture that the blocks and legend would push below a
20 px pitch. A long standing board that 022 already drew small is still drawn.
Measured: pico_show adds +152 tokens with a Pico plugged in (745 → 897; standing prompt 11,639 →
11,791). Probe, 3 sessions: OLED as one part with 4 leads on 3V3, GND and GP4/GP5 (I2C0) in
3 of 3, and GP15 → pin 20 in 3 of 3. In run 3 the tool refused all three of the model's layouts on
the LED line, and Cyclops told you the board was "crowded". That is 022's refusals at work, but
the probe doesn't record why. Tests: 1054 in ~19.8 s against master's 1048 in ~19.9 s. Both were
already over the 10 s budget.
Not ticked: whether the render reads at a glance is yours (pictures on the page), and so is the
bench part on the Pi.
Hands-on: `/try 023`. Put a pot or an OLED on the breadboard, say how the Pico sits, and ask
Cyclops to show you the wiring. Expect a block with named leads, one line per lead read out, each
lead going in first time.
factory/html/023.html
