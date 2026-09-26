---
state: done
opened: 2026-09-26
---

# I want to long-press a recorded session and see a delete button

I want to be able to long-press a recorded session in the list and see a delete button. when i
press the button, the session and all of its images/video data should be deleted from the system

## What was found first

Measured on 2026-09-26, before any of this was planned:

- **The Pi holds 114 sessions and 3.0 GB, and there is no way to remove one.** Not from the
  panel, not from the page, not from a CLI: `cyclops-sessions` has `--fix`, `--recover` and
  `--tidy` and no per-session delete verb, and `admin/urls.py` has no delete route of any kind.
- **The list is the admin page, not a panel of its own.** `admin/static/app.js:171-193` builds
  every row; the kiosk opens the warm browser straight onto `#/sessions` (`kiosk.py:1324`), so a
  finger on the glass is already on these rows. `body.kiosk` is what separates the panel design
  from the laptop one, and it is set server-side by `views._is_local`.
- **The finger arrives as a mouse.** `tests/render_check.mjs:293` says so and `app.js` agrees -
  no `touchstart` anywhere, `pointerdown` twice. So this is a pointer gesture, not a touch one.
- **A builder with no Pi can check this.** Proven, not assumed: Chromium is installed, and with
  `/api/sessions` stubbed the list renders at 800x480 in kiosk mode, rows 762x100, and an 800 ms
  hold on a row today navigates into the session - a hold is just a slow tap. Screenshot of the
  list as it stands in `~/Downloads/cyclops-sessions-today.png`.
- **`card.KNOWN` is missing `videos/`.** `card.VIDEOS` was added (card.py:72) without being added
  to the allow-list at card.py:106, so `card.surprises` calls a watched-video bookmark somebody
  else's file. **Any session holding one can never be removed today** - `--tidy` and the boot
  recovery both walk away from it. Fixing that belongs to this job.
- **Nothing outside the session folder needs cleaning up.** `recall.npz` is derived and
  self-pruning (`indexer.reconcile` drops items whose files are gone, no model call); the
  captioner, the clipper and the `after` child all already survive a folder vanishing under them;
  `library._cache` tolerates it; there is no index, manifest, registry or "latest session"
  pointer beside `sessions/`, and no symlinks. Checked.

## Plan

**The gesture, in the vocabulary the panel already has.** Hold a row for **0.7 s** -
`LONG_PRESS_S`, the same hold that opens the SYSTEM menu from the eye - and the row fills as you
hold it, the browser's version of `kiosk._holding()`. At 0.7 s the row arms: it becomes **DELETE**
(in `--red`) and **CANCEL**. That is the power menu's shape exactly - a hold to reach the
destructive thing, a way out beside it - and it is the reason there is no confirmation dialog:
the hold is the confirmation, and you cannot do it by accident.

Rows act on the **press, not the release** (`kiosk._choose`), so the press that armed the row
cannot pick DELETE. One row armed at a time. A tap elsewhere, a scroll, or 20 s disarms it. A
short tap still opens the session. A row that says RECORDING cannot be armed.

Two traps: `.row` is itself a `<button>`, so the delete cell cannot nest inside it (convert the
row to a `div role="button"`, or sit the cell outside it); and `dragScroll` swallows any click
after 6 px of travel (`app.js:1520`), which the hold must respect or scrolling the list will arm
rows under your thumb.

**Deleting is gone at once.** No undo, no grace period, no trash folder - a trash folder is a
second thing on the card that needs tending, and the card is what you were clearing.

**The panel only.** Loopback-gated with `views._is_local`, like the volume, the switches and
Close. The page has no login and `docs/admin-page.md` already names an unowned LAN as the first
thing to revisit; a one-tap irreversible delete from anything on the network is the wrong side of
that line. **The LAN page shows no delete affordance at all** - not a button that 403s. `body.kiosk`
is what splits them.

**`src/cyclops/session.py`** - `erase(folder)` beside `_remove`. `_remove` is the one destructive
function in the program but it is built for folders *nothing survived in*: it `rmdir`s `photos/`,
so a session with a photograph in it fails by design. `erase` keeps both of that function's safety
properties - `card.surprises` first, so a file we did not write keeps the session, and `rmdir`
never `rmtree`, so anything unexpected still stops it - but it takes `photos/` (jpgs,
`captions.json`, strays), `clips/`, `parts/`, `videos/` and the root files, because you asked it
to. Then `card.sync_dir`. Never a half-deleted folder: `_wanting` drives
`cyclops-recover.service`'s exit code, and a folder left with a log and no page retries every boot.

**`src/cyclops/card.py`** - `VIDEOS` joins `KNOWN`. The comment at `session.py:1445` already says
these two lists move together.

**`src/cyclops/admin/views.py` and `urls.py`** - `POST api/session/<name>/delete`, written beside
`find_clips` and copying it line for line: `_is_local` → 403, `library.resolve` → 404,
`card.locked` → 400 "that session is still recording", `OSError` → 400 with the reason in the
body, and it answers with the fresh list so the row can leave without a reload.

**`src/cyclops/admin/static/app.js`** - the hold, the armed row, and the delete. It must null
`held` (app.js:149) or the list repaints stale for up to 20 s.

**What deliberately stays behind:** the project's copies of the photos it filed, its `Log.md`
entry and its README counts. That is the existing design - `store.py:672` says the copies exist
*because* "the session folder may be deleted while the project outlives it" - and `Log.md` is
never rewritten. The project is the thing you keep coming back to; the session is the raw
recording.

Nothing here talks to the model, so there is no live-session budget to agree.

**For whoever builds it:** stub `/api/panel` in the harness as well, or a pending picture covers
the whole list and every row measures as invisible.

## Done when

- [x] A tap opens the session; a 0.7 s hold arms it instead, showing DELETE and CANCEL — new case in `tests/render_check.mjs`, plus the shot in `/tmp/cyclops-render/`
- [x] The row fills as you hold it, so you can see the hold landing — same harness, a frame mid-hold, look at it
- [x] A drag that scrolls the list never arms a row — same harness, 40 px of travel with the button down
- [x] A row that is RECORDING cannot be armed — same harness against the `live` fixture row
- [x] The laptop page has no delete affordance on it at all — same harness against a non-loopback origin, look at the shot
- [x] DELETE takes the folder and everything in it — `uv run pytest` over a fixture session holding `session.jsonl`, `session.md`, `summary.md`, `project.md`, `video.mp4`, `photos/` (jpgs + `captions.json`), `clips/`, `videos/` and a `.tmp` stray; the folder is gone
- [x] A file we did not write keeps the session, and the page is told why — pytest: drop `notes.txt` in, the folder survives and the response says what stopped it
- [x] A session still recording is refused — pytest with the flock held (`card.claim`), expect 400
- [x] A laptop on the LAN is refused — pytest with `REMOTE_ADDR=192.168.1.44`, expect 403; loopback passes
- [x] The row leaves the list without a reload, and the 20 s cache does not bring it back — `render_check.mjs`, delete a stubbed row, assert it is out of the DOM and `/api/sessions` was re-asked
- [x] `uv run pytest` green, and the suite no slower than the 20.8 s it takes today — the run, both numbers in the outcome
- [x] A session the clipper is working on is refused — pytest with `cut.CUT_LOCK` held, expect 400
- [x] A delete that cannot finish leaves the folder whole, not hollow — pytest: let something appear in `clips/` after it is emptied, then assert `session.jsonl`, `session.md`, `video.mp4` and `photos/` are all still there
- [ ] Long-press a real session on the panel with a finger; it is gone from the list and off the card — yours, on the Pi
- [ ] Delete a session seconds after it ends, while the clipper is still on it; it is either refused or removed whole — yours, on the Pi
- [x] `docs/admin-page.md` no longer claims the page is almost all read-only, and `docs/sessions.md` says a session can be deleted — read them

## Built — 2026-09-26
On the panel's SESSIONS list, holding a row fills it red over 0.7 s, and then it becomes DELETE
(red) and CANCEL. Both act on the press. DELETE posts to a new loopback-only route, which calls
`session.erase`: that one refuses a file Cyclops didn't write (and says which file), then takes
photos/, videos/, clips/, parts/ and the root files, all with rmdir and never rmtree. The row
leaves and the list is asked for again. A tap off an armed row, a scroll, or 20 s puts the row back.
That off-row tap does nothing else, so it can't open a different session by accident. RECORDING
rows can't be armed. The LAN page has none of this, and the route answers it with 403. `videos/`
went into `card.KNOWN`, as planned. I also added it to `_remove`'s generated-dirs list, because
the comment there says the two lists move together. Without it, a triaged-empty folder holding
videos/ could never be removed. One call made on the way: DELETE and CANCEL are spans inside the
row's existing `<button>`, not a restructured row. That avoids the nested-button trap and leaves
the laptop markup unchanged.
Suite: 1030 passed in 19.7 s. It was 20.5–21.2 s measured today before the change, and the job
file said 20.8 s. render_check.mjs gained a sessions section, which can be run on its own:
`ONLY=sessions BASE=http://127.0.0.1:<port> node tests/render_check.mjs`. It passes, and the
shots are in factory/html/020/.
Hands-on: /try, then hold a real session on the glass until it turns red. Press DELETE and it
should leave the list at once. Check the folder is gone from ~/cyclops/sessions. Also try a quick
tap (it should still open the session) and a scroll (no row should arm).
factory/html/020.html

## Feedback — 2026-09-26

refuse a session the clipper is working on, and never leave a half-deleted folder

What happened on the Pi, for whoever picks this up. Two sessions were deleted from the panel.
`2026-09-26_11-29-35_screen-flip-test` went cleanly. `2026-09-26_11-44-13_pico-led-buzzer-testing`
was deleted about two minutes after it ended, while `cyclops-index` was still rendering its
highlight into `clips/`. The journal has the collision:

```
11:49:31 · planned 2026-09-26_11-44-13_pico-led-buzzer-testing: 4m 47s of recording plays in 4m 20s
11:51:41 · could not clip 2026-09-26_11-44-13_pico-led-buzzer-testing (1): FileNotFoundError:
          'sessions/…/clips/.1.mp4.tmp' -> 'sessions/…/clips/1.mp4'
```

`erase` emptied the folder, then could not `rmdir` `clips/` because ffmpeg had written into it
again in between. It stopped there - after `session.jsonl`, `session.md`, `summary.md`,
`video.mp4` and `photos/` were already gone. What is left is the folder plus `clips/plan.json`,
and it **still shows on the panel** as a row titled "Pico Led Buzzer Testing", `verdict: "empty"`.

Two things to fix, and they are separate:
- `card.locked` refuses a session that is still **recording**. Nothing consults `cut.CUT_LOCK`,
  which is the lock the clipper holds. That is the guard that was missing.
- A delete that cannot finish must leave the folder **whole**, not hollow. Half-deleted is the one
  state worse than either end - the plan already said so and the build did not hold the line.
  The irreplaceable things (`photos/`, `video.mp4`) should be the last to go, not the first.

## Built — 2026-09-26 (round 2)
Both fixes from the feedback are in. A delete now holds the clipper's lock while it runs, so no
sweep can start on the folder mid-delete. If the clipper already holds the lock and is working on
*this* session, the delete is refused with "a clip is being made from it - try in a minute". Other
sessions can still be deleted during a render. `erase` now goes in order: clips/, parts/, videos/
and the summary go first. Then it checks that nothing new turned up. Only after that do photos/,
video.mp4, session.md and session.jsonl go, in that order. If a clip lands mid-delete, erase stops
before touching the recording. I tested that exact case, and the old code failed both new tests.
Suite: 1032 passed in 19.7–20.5 s. The hollow "Pico Led Buzzer Testing" folder on the Pi is still
there. Nothing here repairs it, but the new erase can delete it (I checked the code path, not the
glass).
Hands-on: /try, then hold the Pico row and DELETE it. End a short session and delete it within
a minute or two. You should see the refusal in the row, and the session should still be whole in
the list. Once the clip is made, delete it again and it should go.
factory/html/020.html
