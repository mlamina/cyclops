# Manuals

Cyclops can read a product manual and answer from it mid-conversation, without being asked to
look. Drop a PDF on the **Manuals** screen of the admin page and that is the whole of the work
you do: it works out what the manual is for by reading it, and a minute later the pages are
searchable and showable on the panel.

## Where they live

`manuals/` sits beside `sessions/`, `projects/` and `captures/` at the root of the card. One
folder per manual:

```
manuals/mo.unit blue manual/
  mo.unit blue manual.pdf    what you uploaded
  Manual.md                  what it is: part, maker, revision, the names it gets called
  pages.json                 per page - a heading, the transcription, what illustrations are on it
  Pages/0028.jpg             one render per page, 150 dpi
```

**Not inside a project, deliberately.** A manual documents a *part*, and the same m.unit manual
serves two bikes; filed under one of them it would be invisible to the other, and "have we got
the manual for this?" would start depending on which project the conversation happened to be
near. So it is its own thing with its own identity, resolvable by name from any conversation -
`Manual.md`'s `aliases` is what makes "m.unit", "mo unit" and "munit" all find it.

Only the PDF is irreplaceable. `pages.json` and `Pages/` are derived: delete them and the next
sweep rebuilds them, at the cost of one vision call per page. Delete the whole recall index and
it costs nothing at all.

## How a page gets read

`cyclops-index` notices the new folder through inotify, renders each page, and asks
`gpt-5.4-nano` what is on it. Measured on a 45-page manual with **no text layer at all**: 23.5 s,
about a penny, and a median 0.96 word recall against the original. Resumable to the page, so a
power cut costs the pages in flight.

**Every page goes through vision even when the PDF has real text in it**, which looks wasteful
and is not. On the m.unit manual 14 of 45 pages carry broken ligatures - the extracted text holds
`conﬁ guration`, so searching it for "configuration" finds nothing, while the transcription finds
every one. And the illustrations are drawn in vector, so `page.get_images()` returns **0** on the
wiring-diagram pages: nothing but looking at the page will tell you a diagram is there.

## Asking

Cyclops has one manual tool, `ask_manual`, and behind it a librarian: a small pydantic-ai agent
(`gpt-5.4-mini`, `src/cyclops/librarian.py`) with the manuals to itself. Cyclops says "Let me check
the manual.", hands over one self-contained question, and gets back a few speakable lines with the
exact values and the pages they came from - or `answered: false` and the exact reason: what was
searched, what was read, and whether the manuals do not cover it or the question was ambiguous.

The librarian can list the manuals, read a contents page, search by meaning, match exact words
("Menu 8", "AUX1"), read a page's transcription, and look at the page itself when the text is not
enough - wiring diagrams and wire colours, at full resolution. None of that reaches the realtime
conversation: it pays for every token on every turn, so it only ever gets the answer. A page goes
on the panel only when they ask to see it.

Manual pages are not in `recall` any more; that searches photos, sessions and project files.

Every lookup is in the session's `session.md`: the question, each step, the answer or the reason.
`tools/manual_eval.py` asks the questions that went wrong on the Pi and grades the answers.

## Turning it off

`CYCLOPS_MANUALS=0` in `.env` stops manuals being read and takes the list out of the session
prompt. `CYCLOPS_MANUALS_DIR` moves the folder. The screen is on the LAN page only - the panel
has no filesystem to drag a PDF out of.
