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

There is no manual tool. A page is an ordinary `recall` item, so the tool that already finds
photos finds pages, puts one on the panel, and hands the model the pixels.

That last part is the rule worth knowing: **the transcription only ever finds a page, and the
answer is read off the render.** Index text is written by a small model to make a page findable
and is not reliable about a number - two passes over one manual page once disagreed about the
torque printed on it. So when you ask what a figure actually is, Cyclops reads it off the page.

**Name the part in the question.** With two manuals on the card, "how much power does it need"
found the wrong one at a score barely over the floor; "pi 5 power and ports" found the right page
comfortably. The session prompt tells Cyclops to do this, and it is why `aliases` matters.

## Turning it off

`CYCLOPS_MANUALS=0` in `.env` stops manuals being read and takes the list out of the session
prompt. `CYCLOPS_MANUALS_DIR` moves the folder. The screen is on the LAN page only - the panel
has no filesystem to drag a PDF out of.
