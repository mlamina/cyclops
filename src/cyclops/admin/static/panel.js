/* A picture, on the panel: poll for one, paint it, and say when it is up.

   Everything Cyclops shows comes through here - a photo off the shutter, one recalled from the
   card, an edit, a diagram - and they are all the same thing: a JPEG data URL in the offer file,
   fetched by id and dropped into #stage.

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
stage.addEventListener('pointerdown', (event) => {
  if (!document.body.classList.contains('photo')) return;  // a drawing keeps its corner square
  event.preventDefault();  // no synthetic click behind it, and no double-tap zoom
  window.__leave();
});
const PANEL_EVERY = 400;   // one stat() on the server; the picture must not wait on a poll
let showing = null;
// ...and which screen the panel last asked this page to be on. The warm browser is loaded once at
// boot and never navigated, so the two things on the panel that uncover it - his face, which
// promises the sessions, and the heat gauge, which promises the numbers behind itself - say which
// through the same poll. Remembered rather than compared against location.hash, so that a hash the
// *user* changed by tapping a tab is never yanked back to whatever the kiosk asked for last.
let asked = null;

async function show(id) {
  const r = await fetch('/api/picture/' + encodeURIComponent(id), { cache: 'no-store' });
  if (!r.ok) return;
  const found = await r.json();
  // Cleared before it is decided, so a drawing arriving after a photo gets its button back.
  document.body.classList.remove('photo');
  // The class before the picture: a stage that is still display:none has no size, and an image
  // fitted to a box of zero by zero paints nowhere at all.
  window.__drawing(true);
  if (found.image) {
    // A photograph puts itself away on any press; a diagram keeps the corner square. See
    // system.css, and `drawn` in cyclops/panel.py's offer_image.
    if (!found.drawn) document.body.classList.add('photo');
    // Decoded before we say we have painted it, or the kiosk uncovers onto an empty stage.
    const img = new Image();
    img.className = 'shot';
    img.alt = '';
    img.src = found.image;
    try { await img.decode(); } catch (e) { /* show it anyway: a slow decode beats nothing */ }
    stage.textContent = '';
    stage.appendChild(img);
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
