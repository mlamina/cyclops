---
state: done
opened: 2026-09-20
---

# Device extensions: a plug-in per USB device, carrying its own instructions and tools

a new concept of "device extentions". essentially a modular plug-in system that allows us to
write custom instructions + agent tools for specific USB-connected devices. each extension should
be a self-contained python module. when a device is conneted, on agent session start, its special
instructions + agent tools should dynamically be added to the agent's harness / context. build the
system with the currently connected endoscope cam as a very simple example that only carries
instructions (can get into tight spaces / take close-up images of small things), but no agent
tools. a future, second extension would be my behringer TD-3 synth, where the agent should be able
to read and write patches for me. antoher example would be an audio interface that automatically
adds incoming audio to the recorded video

## Plan

### What this sits on

Job 015 is merged (`24eedfc`). It gives this job everything it needs to know what is on the bus:
`devices.connected()` (`src/cyclops/devices.py:140`), a `Device(vid, pid, name, category)` record,
the four coarse categories, a `KNOWN` override table for devices the bus describes badly, and a
1 Hz watcher in the kiosk that already announces a plug mid-session
(`kiosk.py:_sync_bus` -> `ui.py:166` -> `agent.py:1766` -> `agent.py:1777`).

**Read `devices.py` before implementing.** The API above was read on 2026-09-20 and this repo
moves under long plans.

### The shape

`src/cyclops/extensions/` - one module per device, and a small registry in its `__init__.py`.
A module declares one thing:

```python
EXTENSION = Extension(
    name="Endoscope",
    usb_ids=("2ce3:3828", "0329:2022"),   # a device the bus describes badly
    categories=(),                         # or ("music",) for a class nobody can enumerate
    category="camera",                     # what it is, when the bus will not say
    instructions="...",                    # what goes in the prompt while it is plugged in
    tools=(),                              # zero or more
)
```

and a tool is three things in one place, because today they live in four:

```python
Tool(
    schema=READ_PATCH_TOOL,        # the realtime function-tool dict, exactly as agent.py writes them
    run=read_patch,                # (args: dict, device: Device) -> dict. Blocking; run in a thread
    caption="reading the patch…",  # what the panel says while it runs
)
```

Two matchers, because his three examples need both: `usb_ids` for the endoscope and the TD-3,
which are specific boxes; `categories` for "any audio interface", which is a class nobody can
enumerate. An extension matched by either is applied once however many devices matched it.

Discovery is `pkgutil.iter_modules` over the package, in module-name order, resolved once per
session. **A module that raises on import is logged and skipped** - a broken extension must not
cost a session, and that is a criterion below.

**Why one file is the whole point.** A tool name is spelled in four places today: the schema
constant, its `_xxx_tools(settings)` gate (`agent.py:1645`), the `_dispatch_tool` branch
(`agent.py:2299`), and the `_activity_line` branch (`agent.py:4176`). An extension collapses all
four into one module, and adds the one place nothing else has - the prompt block. Adding a device
should touch one new file and nothing else.

### The four seams in agent.py

- **Instructions.** `build_instructions` (`agent.py:1298`) gains one block after 015's
  `PLUGGED INTO YOU RIGHT NOW`, carrying each matched extension's text. Nothing matched, not a
  word - as with every other block there.
- **Tools.** `session_config` (`agent.py:1645`) gains `*extension_tools(...)` beside the nine
  existing gates. A tool for a device that is not plugged in is never offered, which is the rule
  `_project_tools` and the rest already follow: left out rather than offered and refused.
- **Dispatch.** `_dispatch_tool` (`agent.py:2299`) gains one branch before the unknown-tool
  fallback: a name belonging to a matched extension runs its handler in a thread
  (`asyncio.to_thread`, as `_run_project_tool` does), sends the returned dict as the function
  output, and requests a response. A handler that raises answers `{"ok": False, ...}` rather than
  leaving the model waiting - the thing every handler in that file already promises.
- **The panel.** `_activity_line` (`agent.py:4176`) uses the tool's caption instead of falling
  through to "working…".

### It re-arms live

Plugging a synth in and being told to wake the box again is a turn spent on Cyclops itself. So
when the set of matched extensions changes mid-session, `add_bus_change` (`agent.py:1777`)
re-sends `conn.session.update(...)` **before** it sends the synthetic user turn - so the model can
use a new tool in the very response that acknowledges the plug. An unplug takes its tools and its
block away the same way, and still asks for nothing said.

Two things make this cheap and safe, and both were checked: `_on_session_ready` is idempotent
(`agent.py:2003`), so a second `session.updated` re-greets nothing and re-sounds no lid; and the
re-sent instructions are **the ones the session opened with, with only the extension block
swapped**. `build_instructions` is not called again - it re-reads the card and rebuilds the clock,
and the clock belongs to the greeting and to nothing after it.

### The endoscope extension

`src/cyclops/extensions/endoscope.py`. IDs `2ce3:3828` and `0329:2022`, category camera, name
"Endoscope", **no tools**. Its instructions say, in his words, that this camera gets into tight
spaces and takes close-ups of small things - a few lines, not a paragraph.

### The extension owns its device

His answer: everything about a device lives in its own file. So `devices.KNOWN` is **built from
the loaded extensions** rather than hardcoded, and the endoscope's two rows move out of
`devices.py:46` into `endoscope.py`. `devices.py` keeps its bus-class fallback and its `IGNORED`
set untouched. Mind the import cycle: extensions declare their category as a plain string and
import nothing from `devices`; `devices` builds the override table through a cached lookup, inside
a function, so a per-device read does not pay for it.

**`webcam.USEEPLUS_IDS` stays where it is** (`webcam.py:37`). That tuple is how the camera is
*opened*, and putting the camera path behind the plug-in loader means a broken extension is a dead
camera. It keeps its comment pointing at the extension, and a test holds the two lists - and the
udev rules file, which is the third copy - to agreement so they cannot drift.

### Proving the tool half

The endoscope carries no tools, so the tool path would otherwise ship unexercised. A **fixture
extension** - a fake device with one tool - is used by pytest for the wiring and by
`tools/talk_probe.py` for the part a test cannot see: whether the model actually calls it. The
registry takes an optional extra package so the fixture is passed in, never shipped and never
behind a setting. (A wired tool is not a used tool: `talk_probe.py --usb` already fakes a bus by
patching `devices.connected`.)

### Not touched

Which camera `open_camera` picks and the switch-on-plug that landed with it; 015's panel rail and
its glyphs; the greeting rules; the settings screen; the udev rules.

And **the audio interface is not built**. "Automatically adds incoming audio to the recorded
video" is neither instructions nor a tool - it is a hook into `record.py`, a third kind of
declaration. This job leaves room for one and adds none.

## Done when

- [x] The endoscope's block is in the prompt when its ID is on a fake bus, and absent when it is
      not. — pytest on `build_instructions` with a faked bus
- [x] A category-matched extension applies to a device nobody wrote an ID for. — pytest, a fake
      audio interface on the bus
- [x] Two devices matching one extension add its block once. — pytest
- [x] An extension's tool is in the session's tool list while its device is plugged in, and not
      there when it is not. — pytest on `session_config`
- [x] A call reaches the extension's handler and its dict comes back as the function output; a
      handler that raises still answers the model. — pytest with the fake conn `tests/test_devices.py`
      already has
- [x] An extension that raises on import is skipped and the session still opens. — pytest
- [x] The panel line while an extension's tool runs is the extension's own caption, not
      "working…". — pytest on `_activity_line`
- [x] The endoscope is still `camera` named "Endoscope" with the table now built from the
      extensions, and the lav mic is still never listed. — pytest, 015's own tests unchanged
- [x] The endoscope's IDs agree everywhere they appear: the extension, `webcam.USEEPLUS_IDS` and
      `deploy/99-useeplus-camera.rules`. — pytest
- [x] Discovery and matching cost under 5 ms, so the greeting's margin behind the lid is
      unchanged. — pytest timing the call, the same bar 015 set for the bus read
- [x] Adding a device touches one new file: the endoscope's whole extension is one module under
      `src/cyclops/extensions/`. — the file list, in the outcome
- [x] A device arriving mid-session re-sends the config once, with its tools, and no second
      greeting or lid sound follows. — pytest with the fake conn
- [x] A device leaving mid-session takes its tools and its block away. — pytest
- [x] The re-sent instructions are the ones the session opened with bar the extension block - the
      clock is not rebuilt mid-session. — pytest
- [x] What the endoscope costs the standing prompt is a stated number, and an empty bus leaves it
      byte-identical to master's. — `tools/prompt_audit.py`, both numbers said (master today:
      4,353 tok of instructions, 5,848 of tools)
- [x] With the endoscope faked onto the bus, over 10 wakes it talks like something that knows it
      has a snake cam. — `talk_probe.py --usb camera:Endoscope --wakes 10`, transcripts in the outcome
- [x] With the fixture extension faked in and asked the question its tool is for, the model calls
      the tool. — `talk_probe.py`, the count said
- [x] `uv run pytest` passes and is no slower than master. — `uv run pytest`, both numbers said
- [ ] Endoscope plugged in, wake it, ask it to look somewhere awkward: it behaves like something
      that knows what it is holding. — yours, on the Pi
- [ ] Unplug it mid-session and it stops believing it has one. — yours, on the Pi
- [ ] Plug it in mid-session and use it in the next sentence, without waking the box again. —
      yours, on the Pi

## Built — 2026-09-20
Each USB device can now have one file under `src/cyclops/extensions/` that says what the model
should know about it and which tools it brings. While the device is on the bus, its text goes in a
block under 015's plugged-in list and its tools are offered. A module that crashes on load is
logged and skipped. The endoscope is the first extension: instructions only, 86 tokens, no tools.
`devices` now builds its table of badly-described devices from the extensions, so the endoscope's
name and category live in its own file. Its IDs are still in `webcam.USEEPLUS_IDS` and the udev
rule too, and a test keeps all three the same. A plug or unplug mid-session re-sends only the
instructions and tools, not the voice (which can't change once it has spoken), before the turn
that mentions the device. The instructions are the ones the session opened with, with only the
extension block swapped. The tool half is tested with a fake synth under `tests/fake_extensions/`.
`talk_probe.py` gained `--extensions`, `--script synth`, and turns `--usb camera:Endoscope` into
the endoscope's real ID. `prompt_audit.py` gained `--usb`. One thing I decided along the way:
"10 wakes" was run as `--situations 5 --repeat 2`, because `--wakes` takes a file path.
Results: 10/10 greetings say what the endoscope is for, but every one now opens on it (6/10 say
"tight spaces"). That's for you to judge, and the fix would go in the endoscope's own file. The
synth tool was called 3/3. The empty-bus prompt and tools are byte-identical to master's. pytest:
1007 passed in about 19.5 s; master had 994 in about 19.5 s.
Hands-on: /try it, then wake with the endoscope in and ask it to look somewhere awkward. Next,
pull it out mid-session: it should say nothing and stop treating it as the camera. Then plug it
back in mid-session and use it in your next sentence: expect one line about it, with no re-wake.
factory/html/016.html
