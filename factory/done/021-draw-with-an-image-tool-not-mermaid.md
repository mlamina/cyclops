---
state: done
opened: 2026-09-26
---

# Remove Mermaid from the scratchpad; drawing goes to a generic image generation tool

right now, cyclops draws diagrams on the scratchpad with mermaid. these diagrams never feel very
helpful. let's remove the mermaid functionality from the scratchpad entirely and adjust the
prompts. instead, cyclops should get a generic image generation tool. so whenever I ask to draw or
visualize something, i'll use that

## Plan

**Found.** Mermaid lives only in the `sketch` tool (Prefab scratchpad, `sketch.py`, gated by
`CYCLOPS_SKETCH`), whose description tells the model "draw_diagram is now the narrow exception"
and routes wiring/flows to Mermaid - contradicting the system prompt. The image tool already
exists (`draw_diagram` → `imagine.draw`, gpt-image-2.5-sunburst, ~26 s), but its description,
`DRAW_PROMPT` ("A technical diagram…", "no shading, no perspective…") and `DEFAULT_STYLE` pin it
to technical diagrams. Mermaid's JS is bundled inside Prefab's renderer (`prefab-ui`); nothing of
ours ships it, and it is inert once nothing emits a `Mermaid` node - leave the renderer alone.

**1. Mermaid out of the scratchpad.**
- `sketch._globals` stops exporting `Mermaid` (so `Mermaid(...)` fails to compile).
- Delete the label sanitiser: `_MERMAID_UNSAFE`, `_NODE_LABEL`, `_EDGE_LABEL`, `_quote_labels`,
  `_clean`, its call in `compile()`, and `import re`.
- Delete the three sanitiser tests in `tests/test_sketch.py`; reword the docstring that cites
  Mermaid.
- Reword comments that argue from Mermaid (`sketch.py` docstring, `panel.css` `.sketch`,
  `admin/views.py` `renderer()`) to argue from charts. CSS rules and `mode="light"` stay.

**2. No drawing on either scratchpad.** All drawing goes to the image tool, even a smiley.
- `SKETCH_TOOL`: remove the Mermaid pitch, the `Mermaid(text)` entry, the battery/fuse example,
  the "draw_diagram is now the narrow exception" passage, and `Svg(markup)` as a drawing
  primitive (drop it from `_globals` too).
- `SCRATCHPAD_TOOL` (`write_on_scratchpad`): remove "a simple drawing as inline SVG" and the
  "draw_diagram is the expensive exception" framing.
- Both scratchpads keep words, numbers, lists, tables, and charts of real readings.

**3. `draw_diagram` becomes `draw`, a generic image tool.**
- Rename the tool, its handler and dispatch, the caption, and the `test_caption` entry.
- Description: draw or visualize anything as a picture - a circuit, a flow, a concept, what
  something will look like, a scene. Reach for it whenever they say draw / visualize / show me a
  picture of. Keep the honest warnings: drawn not checked; not to scale, never measure off it;
  say a connection out loud when a wrong one would cost a part or a shock.
- `DRAW_PROMPT` loses the "A technical diagram" framing and the blanket ban on shading and
  perspective. It keeps the legibility rules: few large elements, horizontal labels, the exact
  values given and no others. `style` carries the look: a service-manual wiring diagram for a
  circuit, a clean illustration for a scene. `DEFAULT_STYLE` becomes a neutral, clean labelled
  illustration.
- Unchanged: session-log role `drawn`, `photos/*_drawn.jpg`, recall and edit_photo on a drawn
  picture, the half-minute announce flow.

**4. Prompts.** SHOWING THEM SOMETHING in `BASE_INSTRUCTIONS` becomes:
- the scratchpad for words, numbers, lists and charts;
- `draw` for anything drawn or visualized;
- `edit_photo` for their own photo changed.

Rename `draw_diagram` → `draw` wherever else the prompt and tools say it (the half-minute line,
`point_at`, `recall`, `edit_photo`), plus `docs/tools.md`.

**5. Checks.**
- `smoke.py` turn 4 calls `draw`.
- New `talk_probe.py --script draw` sends four utterances and tallies the tools called:
  - "Can you draw me a smiley face?"
  - "Visualize how the battery, the fuse and the motor connect."
  - "Draw what the shelf will look like when it's finished."
  - "Show me the boot sequence as a flowchart."

## Done when
- [x] A sketch that calls `Mermaid(...)` (or `Svg(...)`) compiles to nothing — new test in `tests/test_sketch.py`, `uv run pytest`
- [x] No offered tool is named `draw_diagram`, and `draw` is offered when diagrams are on — test over the offered tool schemas (tool names are machine-facing), `uv run pytest`
- [ ] Whole suite passes in under 10 s — `uv run pytest`, the time it prints
- [x] `draw` works end to end, a picture reaching the panel file — `uv run cyclops-smoke` (turn 4; one image)
- [x] All four draw utterances route to `draw`, and none to `sketch` or `write_on_scratchpad` — `uv run tools/talk_probe.py --script draw --runs 1` (4 live sessions, 4 images, ~3 min; nothing more)
- [ ] "Draw me …" on the Pi puts a generated picture on the panel, and nothing asked for draws a Mermaid chart — yours, on the Pi

## Built — 2026-09-26
The scratchpad can't draw any more. `Mermaid` and `Svg` are gone from what sketch code can reach, and the label sanitiser went with them. `draw_diagram` is now `draw`, a general picture tool: "draw", "visualize" or "show me a picture of" all go there, from a smiley to a wiring diagram. The image prompt no longer says "technical diagram" and no longer bans shading. The style note sets the look, and the default style is a plain labelled illustration. Both scratchpad descriptions and the system prompt now send every drawing to `draw`. The probe ran with `CYCLOPS_SKETCH=1`, since that's the scratchpad that drew Mermaid: 4 of 4 requests called `draw` and none touched the scratchpad. The pictures are on the page. Smoke turn 4 passed (109 KB in 22.7 s, offered to the panel). The Mac has no camera, so its photo turn was fed a saved capture. Its take_a_look turn failed for the same reason, which has nothing to do with this job.
Not met: **suite under 10 s**. It's 20 s, but master is 21.5 s on the same machine at the same load, so it was already over before this job. It's left unticked. I also added `_offered_tools()` in agent.py so the new test checks the real offered list.
Hands-on: /try, then say "draw me a smiley" and "show me how X wires up". Expect a generated picture after about 20 s and no Mermaid chart. It still describes a drawing aloud before it lands; see whether that bothers you.
factory/html/021.html
