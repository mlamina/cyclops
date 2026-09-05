/* What is on the panel: poll for it, paint it, and say when it is up.

   Everything Cyclops shows comes through here, and it is one of two things. A picture - a photo
   off the shutter, one recalled from the card, an edit, a diagram - is a JPEG data URL in the
   offer file, fetched by id and dropped into #stage as an <img>. A scratchpad is markup the model
   wrote itself, and goes into #stage as an <iframe> with a document of its own; see SCRATCHPAD_HEAD.

   This file used to be four times longer and carried a symbol library: what a "resistor" or a
   "header" looked like, laid out on the page by JointJS from a JSON scene graph. Diagrams are
   drawn as images now, so the library, the layout, the SVG serialiser and 527 KB of vendored
   bundles all came out on 2026-09-04. What is left is the half that was always about pictures.

   Loaded on every client, and that is companion mode: a phone or an iPad open beside the bench
   paints the same picture at the same moment, and a press on either screen puts it away on both.
   Three things here really are the panel's alone and are gated on body.kiosk - the paint
   handshake the kiosk waits on before it uncovers, the screen it steers its own browser to, and
   the faster poll. Everything else is the same code doing the same job on two screens. */

const stage = document.getElementById('stage');
// Which screen this is. The server decides it (views._is_local) and the template stamps it on
// <body>; status.js reads it the same way for the kiosk's input hardening.
const KIOSK = document.body.classList.contains('kiosk');

// Any press on a picture puts it away. On the press and not the click, because that is what
// the kiosk's own tab row does and a screen with no travel has nothing else to answer with;
// and on the stage rather than the document because the stage IS the panel while it is up -
// body.drawing collapses the grid to one row and gives it the whole 800x480.
// ...and so does any press on a scratchpad, which is why .scratchpad takes no pointer events.
stage.addEventListener('pointerdown', (event) => {
  if (!document.body.classList.contains('photo')) return;  // nothing is up
  event.preventDefault();  // no synthetic click behind it, and no double-tap zoom
  // On a companion, put our own copy away now and remember which one it was. The note we leave
  // below has to travel to the kiosk, be noticed within ADMIN_POLL_S and come back as a withdraw
  // before /api/panel stops naming this picture - the better part of a second - and a screen that
  // sat there holding a picture you had already dismissed is a screen that ignored you. Drawn
  // before the round trip, like the volume and the two switches in status.js.
  //
  // `showing` is deliberately left where it is. The next poll almost always still names this
  // picture, and forgetting it would have watch() decide it is new and paint it straight back -
  // a picture flashing back onto a screen you just dismissed it from. It is cleared for real when
  // the withdraw lands, or replaced when a genuinely different picture arrives.
  if (!KIOSK) drop();
  window.__leave();
});
// ---- a scratchpad's own document ----
//
// The other thing that can land on the stage: a piece of HTML the model wrote
// (cyclops/panel.py's offer_scratchpad). It goes in an <iframe srcdoc> rather than into
// #stage.innerHTML, and the reason is the cascade rather than security. A model asked to write a
// screen writes `body { ... }` and `h1 { ... }`, and in THIS document those rules would land on
// the dashboard behind it. A second document is the only cheap way to let it write a whole one.
//
// The scratchpad is white, always. Not the phosphor green everything else on this panel wears,
// and not white-only-for-drawings either, which is what it was for two commits: a screen that
// changed colour depending on whether a <svg> happened to be in the markup, with two sets of
// defaults behind it and a bug in each. It is a blank sheet. Anything written or drawn on it
// brings its own colour, which is what a sheet of paper is for.
//
// Two of these lines are load-bearing:
//
//   The CSP, because the iframe's `load` event waits on subresources. One <img src="https://...">
//   the model invented would stall the paint for as long as DNS and TCP take, on a panel whose
//   whole promise is that this appears while he is still talking. default-src 'none' turns that
//   into an instant block. It also means no script runs, belt to the sandbox attribute's braces.
//
//   vh and vw, which nothing else in this interface may use (see --u in base.css). They are exact
//   here and nowhere else: the scratchpad is always the whole panel, 800x480 at
//   --force-device-scale-factor=1, so 1vh is 4.8px and stays 4.8px.
//
// Everything else is one job: make `<h1>25 Nm</h1>` with no CSS at all a good screen, because
// every character of styling the model has to write is silence between the question and the
// answer. It sets nothing beyond that. SVG in particular is left entirely alone - its own
// defaults are right on white, and the last thing it needs is us inheriting a stroke onto the
// background rect it drew for itself.
const SCRATCHPAD_HEAD = `<!doctype html><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy"
      content="default-src 'none'; style-src 'unsafe-inline'; img-src data:">
<style>
  html, body { margin: 0; height: 100%; }
  * { box-sizing: border-box; }
  body {
    background: #fff; color: #111;
    font-family: ui-monospace, "DejaVu Sans Mono", "Liberation Mono", "Noto Color Emoji", monospace;
    font-size: 4vh; line-height: 1.35;
    /* A column rather than a centred grid: grid tracks size to their content and then overflow,
       which put a drawing under a heading off the bottom of the panel. Flex lets it give way. */
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    text-align: center; gap: 0.6em; padding: 3vh 4vw; overflow: hidden;
    -webkit-user-select: none; user-select: none; -webkit-touch-callout: none;
  }
  ::-webkit-scrollbar { width: 0; height: 0; }
  h1 { font-size: 16vh; line-height: 1; margin: 0; }
  h2 { font-size: 8vh; line-height: 1.1; margin: 0; }
  /* The one thing a centred body gets wrong. A numbered list of what to do next is read down the
     left edge, and ragged-centre numbering is unreadable at arm's length. */
  ol, ul { text-align: left; margin: 0; padding-left: 1.4em; }
  li { margin: 0.15em 0; }
  /* As big as what is left, and shrinkable so a heading above it is not pushed off the panel. */
  svg { flex: 1 1 auto; min-height: 0; width: 100%; height: 100%; }
</style>
`;
const SCRATCHPAD_PAINT_MS = 500;  // long enough for a document with no subresources; see scratchpad() below

// One read on the server either way. The panel must not wait on a poll - the kiosk is holding a
// window over this and uncovers on the strength of it - and a companion across the room can.
const PANEL_EVERY = KIOSK ? 400 : 800;
let showing = null;
// ...and which screen the panel last asked this page to be on. The warm browser is loaded once at
// boot and never navigated, so the two things on the panel that uncover it - his face, which
// promises the sessions, and the heat gauge, which promises the numbers behind itself - say which
// through the same poll. Remembered rather than compared against location.hash, so that a hash the
// *user* changed by tapping a tab is never yanked back to whatever the kiosk asked for last.
let asked = null;

// A picture on the stage, decoded before it goes there: the kiosk uncovers on the strength of
// what this function has done, and an <img> still fetching paints nowhere at all.
async function shot(url) {
  const img = new Image();
  img.className = 'shot';
  img.alt = '';
  img.src = url;
  try { await img.decode(); } catch (e) { /* show it anyway: a slow decode beats nothing */ }
  stage.textContent = '';
  stage.appendChild(img);
}

// A scratchpad on the stage, in a document of its own. Same promise as shot(): do not come back
// until there is something to uncover onto.
async function scratchpad(html) {
  // The outer page gives up its bezel and its gutter for this: see panel.css. Every scratchpad,
  // not just a drawing - a blank sheet with a phosphor frame round it is two ideas about what
  // the panel is, and the frame is the one that loses.
  document.body.classList.add('paper');
  const frame = document.createElement('iframe');
  frame.className = 'scratchpad';
  // Both set while it is still detached. An <iframe> inserted empty navigates to about:blank
  // first, and a load listener attached after that fires against the blank one - we would post
  // `painted` for a frame with nothing in it. Set here, insertion navigates exactly once.
  frame.setAttribute('sandbox', '');   // no script, no navigation, an origin of its own
  frame.srcdoc = SCRATCHPAD_HEAD + html;
  const loaded = new Promise((done) => {
    frame.addEventListener('load', done, { once: true });
    // A scratchpad that never fires load must not sit on the kiosk's PAINT_WAIT_S for eight
    // seconds. There is nothing to fetch in it - the CSP saw to that - so this is unreachable
    // in the ordinary case and a late frame rather than a hang in every other.
    setTimeout(done, SCRATCHPAD_PAINT_MS);
  });
  stage.textContent = '';
  stage.appendChild(frame);
  await loaded;
}

async function show(id) {
  const r = await fetch('/api/picture/' + encodeURIComponent(id), { cache: 'no-store' });
  if (!r.ok) {
    // The offer was withdrawn between the poll that named it and this fetch (/api/picture 404s on
    // an id that is no longer pending). Forget it, or this screen never paints anything again.
    // It cannot spin: both routes read the same file, so the next poll reports no picture at all.
    showing = null;
    return;
  }
  const found = await r.json();
  // Cleared before it is decided, so a drawing arriving after a photo gets its button back -
  // and so a photograph landing on top of a drawing gets the bezel back with it.
  document.body.classList.remove('photo', 'paper');
  // The class before the picture: a stage that is still display:none has no size, and an image
  // fitted to a box of zero by zero paints nowhere at all.
  window.__drawing(true);
  // Whatever it is, a press anywhere puts it away, so body.photo goes on before either branch.
  // There was a `drawn` flag here until 2026-09-05 that gave a diagram a small square in the
  // corner instead - a second way out for one gesture, and the one nobody finds on a screen they
  // are not standing in front of. Everything the panel shows is a picture now. What makes the
  // press work over a scratchpad is `pointer-events: none` on .scratchpad - see panel.css.
  document.body.classList.add('photo');
  if (found.scratchpad) {
    await scratchpad(found.scratchpad);
  } else if (found.image) {
    await shot(found.image);
  }
  // Tell the kiosk it may uncover the panel. Sent even when there was no picture in the payload:
  // the alternative is a panel that never uncovers and a user who is told nothing.
  //
  // The kiosk's own browser only. This is the handshake that drops a window on the Pi, and it is
  // the panel's paint it is waiting on - a companion's says nothing about whether there is
  // anything to uncover onto. views.picture_painted refuses anyone else anyway; this is the same
  // rule stated on the side that knows why.
  if (!KIOSK) return;
  try { await fetch('/panel/painted', { method: 'POST', body: '' }); } catch (e) {}
}

// Nothing on the stage, and the page uncovered again. Let go of the image while we are here: on
// the panel there is one browser and no tabs, so the next picture is the only thing that will
// ever want this memory back.
function drop() {
  window.__drawing(false);
  document.body.classList.remove('photo', 'paper');
  stage.textContent = '';
}

async function watch() {
  try {
    const r = await fetch('/api/panel', { cache: 'no-store' });
    const s = await r.json();
    // Which screen the kiosk wants *its own* browser on - his face promising the sessions, the
    // heat gauge promising the numbers. A companion is not that browser and has its own reader
    // pointing at things; yanking an iPad to another screen because somebody touched the panel is
    // the opposite of following along.
    if (KIOSK && s.screen && s.screen !== asked) {
      asked = s.screen;
      location.hash = '#' + s.screen;
    } else if (!s.screen) {
      asked = null;
    }
    if (s.picture && s.picture !== showing) {
      showing = s.picture;
      await show(s.picture);
    } else if (!s.picture && showing) {
      // The kiosk took the panel back and cleared the offer.
      showing = null;
      drop();
    }
  } catch (e) { /* the service will come back, or the kiosk will time the page out */ }
  setTimeout(watch, PANEL_EVERY);
}
watch();
