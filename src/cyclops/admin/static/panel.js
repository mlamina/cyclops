/* What is on the panel: poll for it, paint it, and say when it is up.

   Everything Cyclops shows comes through here, and it is one of two things. A picture - a photo
   off the shutter, one recalled from the card, an edit, a diagram - is a JPEG data URL in the
   offer file, fetched by id and dropped into #stage as an <img>. A scratchpad is markup the model
   wrote itself, and goes into #stage as an <iframe> with a document of its own; see SCRATCHPAD_HEAD.

   This file used to be four times longer and carried a symbol library: what a "resistor" or a
   "header" looked like, laid out on the page by JointJS from a JSON scene graph. Diagrams are
   drawn as images now, so the library, the layout, the SVG serialiser and 527 KB of vendored
   bundles all came out on 2026-09-04. What is left is the half that was always about pictures.

   Loaded only on the kiosk ({% if local %} in the template): a laptop on the LAN has no panel
   to paint. */

const stage = document.getElementById('stage');

// Any press on a picture puts it away. On the press and not the click, because that is what
// the kiosk's own tab row does and a screen with no travel has nothing else to answer with;
// and on the stage rather than the document because the stage IS the panel while it is up -
// body.drawing collapses the grid to one row and gives it the whole 800x480.
// ...and so does any press on a scratchpad, which is why .scratchpad takes no pointer events.
stage.addEventListener('pointerdown', (event) => {
  if (!document.body.classList.contains('photo')) return;  // a drawing keeps its corner square
  event.preventDefault();  // no synthetic click behind it, and no double-tap zoom
  window.__leave();
});
// ---- a scratchpad's own document ----
//
// The other thing that can land on the stage: a piece of HTML the model wrote (cyclops/panel.py's
// offer_scratchpad). It goes in an <iframe srcdoc> rather than into #stage.innerHTML, and the reason is
// the cascade rather than security. A model asked to write a screen writes `body { ... }` and
// `h1 { ... }`, and in THIS document those rules land on the dashboard behind it. A second
// document is the only cheap way to let it write a whole screen.
//
// Everything below is prepended to whatever it wrote. Two of these lines are load-bearing:
//
//   The CSP, because the iframe's `load` event waits on subresources. One <img src="https://...">
//   the model invented would stall the paint for as long as DNS and TCP take, on a panel whose
//   whole promise is that this appears while he is still talking. default-src 'none' turns that
//   into an instant block. It also means no script runs, belt to the sandbox attribute's braces.
//
//   vh and vw, which nothing else in this interface may use (see --u in base.css). They are exact
//   here and nowhere else: the card is always the whole panel, 800x480 at
//   --force-device-scale-factor=1, so 1vh is 4.8px and stays 4.8px.
//
// The rest is one job: make `<h1>25 Nm</h1>` with no CSS at all a good screen, because every
// character of styling the model has to write is silence between the question and the answer. It
// sets nothing it does not have to, so a scratchpad that DOES want to be pink can be.
const SCRATCHPAD_HEAD = `<!doctype html><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy"
      content="default-src 'none'; style-src 'unsafe-inline'; img-src data:">
<style>
  html, body { margin: 0; height: 100%; }
  * { box-sizing: border-box; }
  body {
    background: #050f0a; color: #56ff8c;
    font-family: ui-monospace, "DejaVu Sans Mono", "Liberation Mono", "Noto Color Emoji", monospace;
    font-size: 4vh; line-height: 1.35;
    display: grid; place-content: center; justify-items: center; text-align: center;
    gap: 0.6em; padding: 4vh 5vw; overflow: hidden;
    -webkit-user-select: none; user-select: none; -webkit-touch-callout: none;
  }
  ::-webkit-scrollbar { width: 0; height: 0; }
  h1 { font-size: 16vh; line-height: 1; margin: 0; }
  h2 { font-size: 8vh; line-height: 1.1; margin: 0; }
  /* The one thing a centred body gets wrong. A numbered list of what to do next is read down the
     left edge, and ragged-centre numbering is unreadable at arm's length. */
  ol, ul { text-align: left; margin: 0; padding-left: 1.4em; }
  li { margin: 0.15em 0; }
  /* An <svg> with no width attribute is 300x150 by default, and one with a viewBox and no bounds
     overflows the screen. Both are the model's likeliest mistake, and both stop here. */
  svg { max-width: 80vw; max-height: 60vh; fill: none; stroke: currentColor; }
  /* ...and this is the likeliest mistake of all, seen the first time one was asked for a sketch.
     A model drawing SVG writes stroke="black", because line art is a thing you do on paper. On a
     #050f0a screen that lands invisible - the drawing was there, correct, and unlit.

     So black is read as "I did not choose a colour" rather than as a colour, and becomes his own
     green. Any other colour it names is left exactly as asked, which is the whole point of the
     attribute selector: a red bar stays red. Presentation attributes lose to any CSS rule that
     matches the element itself, which is why these have to name the shapes and not just <svg>. */
  svg [stroke="black"], svg [stroke="#000"], svg [stroke="#000000"] { stroke: currentColor; }
  svg [fill="black"], svg [fill="#000"], svg [fill="#000000"] { fill: currentColor; }
  svg text { fill: currentColor; stroke: none; }
</style>
`;
const SCRATCHPAD_PAINT_MS = 500;  // long enough for a document with no subresources; see scratchpad() below

const PANEL_EVERY = 400;   // one stat() on the server; the picture must not wait on a poll
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
  if (!r.ok) return;
  const found = await r.json();
  // Cleared before it is decided, so a drawing arriving after a photo gets its button back.
  document.body.classList.remove('photo');
  // The class before the picture: a stage that is still display:none has no size, and an image
  // fitted to a box of zero by zero paints nowhere at all.
  window.__drawing(true);
  if (found.scratchpad) {
    // A scratchpad puts itself away on any press, like a photograph and unlike a diagram. Cyclops
    // put this up unasked while he was talking, over the eye and the way out of the session, so
    // the way back has to be the whole screen rather than a corner somebody has to find first.
    // What makes that work is `pointer-events: none` on .scratchpad - see panel.css.
    document.body.classList.add('photo');
    await scratchpad(found.scratchpad);
  } else if (found.image) {
    // A photograph puts itself away on any press; a diagram keeps the corner square. See
    // system.css, and `drawn` in cyclops/panel.py's offer_image.
    if (!found.drawn) document.body.classList.add('photo');
    await shot(found.image);
  }
  // Tell the kiosk it may uncover the panel. Sent even when there was no picture in the payload:
  // the alternative is a panel that never uncovers and a user who is told nothing.
  try { await fetch('/panel/painted', { method: 'POST', body: '' }); } catch (e) {}
}

async function watch() {
  try {
    const r = await fetch('/api/panel', { cache: 'no-store' });
    const s = await r.json();
    if (s.screen && s.screen !== asked) {
      asked = s.screen;
      location.hash = '#' + s.screen;
    } else if (!s.screen) {
      asked = null;
    }
    if (s.picture && s.picture !== showing) {
      showing = s.picture;
      await show(s.picture);
    } else if (!s.picture && showing) {
      // The kiosk took the panel back and cleared the offer. Go quiet, and let go of the image:
      // on a box with one browser and no tabs, the next picture is the only thing that will ever
      // want this memory.
      showing = null;
      window.__drawing(false);
      document.body.classList.remove('photo');
      stage.textContent = '';
    }
  } catch (e) { /* the service will come back, or the kiosk will time the page out */ }
  setTimeout(watch, PANEL_EVERY);
}
watch();
