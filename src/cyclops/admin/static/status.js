/* The four readings, the three settings, and the way back to the camera.

   This is the half of the page the server also renders: dashboard.html ships the first set of
   numbers inline and views._payload() answers /api/status with the same keys, so the poll below
   is a refresh and not a first paint. Everything it drives is [data-field] or [data-meter], and
   nothing here knows what any of them mean.

   The controls are all null-checked rather than gated: they only exist on the kiosk's own
   browser ({% if local %} in the template), and a laptop simply finds nothing to bind.

   Loaded before app.js, and it publishes window.__leave for diagram.js to reach. */

const fields = document.querySelectorAll('[data-field]');
const meters = document.querySelectorAll('[data-meter]');
const temp = document.getElementById('temp');
const body = document.body;
const EVERY = 5000, SLOW = 15000;   // nothing here moves fast; hidden tabs move slower still

async function poll() {
  try {
    const r = await fetch('/api/status', { cache: 'no-store' });
    const s = await r.json();
    for (const el of fields) {
      const v = s[el.dataset.field];
      if (v != null) el.textContent = v;
    }
    // The bars. A second loop and a second attribute because this writes a style property
    // rather than text - and because a bar has to be told when its reading has gone away.
    // Left where it was, a stale bar and a true one look identical, so a number that is no
    // longer there draws an empty track instead of the last one this page happened to know.
    for (const el of meters) {
      const v = s[el.dataset.meter];
      el.style.setProperty('--pct', v == null ? 0 : Math.max(0, Math.min(100, v)));
    }
    temp.dataset.band = s.temp_band;
    body.classList.remove('offline');
  } catch (e) {
    body.classList.add('offline');
  }
  // Self-scheduling rather than setInterval, so a slow Pi never stacks requests on itself.
  setTimeout(poll, document.visibilityState === 'hidden' ? SLOW : EVERY);
}
setTimeout(poll, EVERY);   // the server already rendered the first set of numbers

// The page cannot set the volume any more than it can close its own window: it leaves the
// level for the kiosk, which is the process that lives inside the audio session.
const vol = document.getElementById('vol');
if (vol) {
  const out = document.getElementById('volval');
  // WebKit cannot fill a range track on its own, so the track's gradient reads --v and this
  // keeps it in step. Set once up front, or the bar opens empty over a speaker that is not.
  const paint = () => vol.style.setProperty('--v', vol.value);
  paint();
  const send = (v) => fetch('/volume', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: 'level=' + v,
  }).catch(() => {});
  let sent = 0;
  vol.addEventListener('input', () => {
    out.textContent = vol.value + '%';
    paint();
    // Throttled while the finger is down, so a session in progress gets louder as you drag
    // without one request per pixel of travel.
    const now = Date.now();
    if (now - sent > 200) { sent = now; send(vol.value); }
  });
  vol.addEventListener('change', () => send(vol.value));   // the level on release always lands
}

// Whether Cyclops may be cut off mid-sentence. Same shape as the volume: the page leaves a
// note, and the process holding the microphone picks it up - within half a second, so the
// switch is worth reaching for in the middle of a sentence rather than only between two.
const barge = document.getElementById('barge');
if (barge) {
  const hint = document.getElementById('bargehint');
  barge.addEventListener('click', async () => {
    const on = barge.getAttribute('aria-checked') !== 'true';
    // Drawn before the round trip. This is a note on a filesystem two processes share, so
    // there is nothing to fail that waiting would reveal - and a switch that stays where it
    // was for a moment is a switch you press twice.
    barge.setAttribute('aria-checked', on ? 'true' : 'false');
    barge.textContent = on ? 'ON' : 'OFF';
    hint.textContent = on ? 'talk over Cyclops to cut in' : 'wait for Cyclops to finish';
    try {
      await fetch('/barge-in', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: 'on=' + (on ? '1' : '0'),
      });
    } catch (e) { /* the service is gone; the next render says what actually stuck */ }
  });
}

// What a session's video is of. The same note-and-pick-up as the two above, with one honest
// difference the hint has to carry: an encoder is opened once, at one frame size, so this lands
// on the next session rather than on any recording already running.
const rec = document.getElementById('rec');
if (rec) {
  const hint = document.getElementById('rechint');
  rec.addEventListener('click', async () => {
    const screen = rec.getAttribute('aria-checked') !== 'true';
    rec.setAttribute('aria-checked', screen ? 'true' : 'false');   // drawn before the round trip
    rec.textContent = screen ? 'SCREEN' : 'CAMERA';
    hint.textContent = screen ? 'this screen, chrome and all' : 'the camera alone, full size';
    try {
      await fetch('/record-source', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: 'source=' + (screen ? 'screen' : 'camera'),
      });
    } catch (e) { /* the service is gone; the next render says what actually stuck */ }
  });
}

const close = document.getElementById('close');
const closeword = document.getElementById('closeword');
// Named and hung on window, because the panel has a second way out: a picture is put away by
// pressing it anywhere at all (see the diagram script below, which is a script of its own and
// cannot see a const in this one). Both have to make the same journey - leave the kiosk its
// note, put the button back, reset the hash - so there is one of it rather than two.
async function leave() {
  if (!close || close.disabled) return;  // a second press while the first is in flight
  close.disabled = true;
  // The word only, never the whole button: over a diagram the word is hidden and the glyph is
  // the entire control, and replacing the button's text used to take the glyph with it.
  if (closeword) closeword.textContent = '\u00a0Closing…';
  // The kiosk owns the browser process; this only leaves it a note asking it to take its
  // panel back. It does not take this page down with it - the browser stays running behind
  // the panel so the next tap of the gear is instant - so the button has to put itself back.
  // Without that, the second opening finds it disabled and reading "Closing…" for ever.
  try { await fetch('/close', { method: 'POST' }); }
  catch (e) { /* the service is gone; the kiosk cannot be told, but the button still resets */ }
  finally {
    close.disabled = false;
    if (closeword) closeword.textContent = '\u00a0Close';
    // Leaving the page leaves the screen too: the next tap of HISTORY should land where the
    // kiosk points the browser at boot, not wherever this was left three days ago.
    location.hash = '#/sessions';
  }
}
window.__leave = leave;
if (close) close.addEventListener('click', leave);

// kiosk hardening: no context menu, no pinch/double-tap zoom, and nothing is draggable.
// A native drag is a scroll that did not happen - the finger picks a picture up off the grid
// instead of moving the grid under it - and there is nowhere in this interface to drop one.
document.addEventListener('dragstart', (e) => e.preventDefault());
document.addEventListener('contextmenu', (e) => e.preventDefault());
document.addEventListener('gesturestart', (e) => e.preventDefault());
document.addEventListener('dblclick', (e) => e.preventDefault());
