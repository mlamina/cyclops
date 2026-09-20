"""A fake step synth with one tool: how the tool half of an extension gets exercised at all.

The endoscope carries no tools, so without this the tool path would ship unused. Never shipped -
it is not under ``src/`` - and never behind a setting: whoever wants it hands this folder to
:func:`cyclops.extensions.load` as its extra. The suite does, and so does
``tools/talk_probe.py --extensions tests/fake_extensions``, which is where the model is asked the
question the tool is for.

Shaped like the TD-3 one will be: a box with a pattern on it that the model can read.
"""

from __future__ import annotations

from cyclops.extensions import Extension, Tool

READ_PATCH_TOOL = {
    "type": "function",
    "name": "read_patch",
    "description": (
        "Read the pattern loaded on the Bench Synth right now: its tempo and the note, accent "
        "and slide on each of its 16 steps. Use it whenever they ask what is on the synth, what "
        "a step is set to or what they are hearing - never answer that from memory."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

PATTERN = {
    "tempo": 124,
    "steps": ["C2", "C2", "rest", "D#2", "C2", "rest", "G2", "C3",
              "C2", "rest", "A#1", "C2", "rest", "D#2", "F2", "G2"],
    "accents": [1, 5, 8, 13],
    "slides": [7, 15],
}


def read_patch(args: dict, device) -> dict:
    return {"ok": True, "synth": device.name, **PATTERN}


EXTENSION = Extension(
    name="Bench Synth",
    usb_ids=("f00d:0303",),
    category="music",
    instructions=(
        "a 16-step bass synth. You can read the pattern loaded on it with read_patch, so when "
        "they ask about it, read it rather than guess."
    ),
    tools=(Tool(schema=READ_PATCH_TOOL, run=read_patch, caption="reading the pattern…"),),
)
