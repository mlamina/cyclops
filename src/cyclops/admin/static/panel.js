/* What is on the panel: poll for it, paint it, and say when it is up.

   Everything Cyclops shows comes through here, and it is one of two things. A picture - a photo
   off the shutter, one recalled from the card, an edit, a diagram - is a JPEG data URL in the
   offer file, fetched by id and dropped into #stage as an <img>. A scratchpad is markup the model
   wrote itself, and goes into #stage as an <iframe> with a document of its own; see SCRATCHPAD_HEAD.
   A sketch is neither: it is already on screen in a renderer that has been running since page
   load, and all that arrives here is permission to stop covering it. See sketch.js. A video is a
   <video> on a URL YouTube signed a moment ago, seeked before it is uncovered, and it is the only
   thing here that makes a sound - which is why it is also the only one that has to be stopped by
   hand on the way out.

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
  // ...except a page of a manual, where a press may be the start of a scroll. It goes on the tap
  // instead - see the click listener below.
  if (document.body.classList.contains('page')) return;
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
  // Pause before anything else, and pause rather than drop(). On the panel the withdraw has to
  // travel to the kiosk and come back, which is the better part of a second - silent for a
  // picture, but a video would keep talking behind the kiosk's own window after it was covered
  // again. drop() here instead would hit the flash-back the note above describes.
  // ...except on the seeker, which is the one thing on this screen that takes a press and does
  // not mean "put it away". Same exemption the recording player makes, and for the same reason:
  // everything else on the stage, letterboxing included, is a way out.
  if (event.target.closest('.scrub')) return;
  const playing = stage.querySelector('video');
  if (playing) playing.pause();
  if (!KIOSK) drop();
  window.__leave();
});
// A page of a manual is laid across the whole width and is taller than the panel, so a finger
// on it is as likely to be scrolling as leaving. It is put away by a tap: a press and a lift
// that stayed inside dragScroll's slop. A drag never gets here - dragScroll swallows the click
// that ends one and marks it handled - and on a companion's touchscreen the browser's own pan
// sends no click at all.
stage.addEventListener('click', (event) => {
  if (!document.body.classList.contains('page') || event.defaultPrevented) return;
  if (!KIOSK) drop();
  window.__leave();
});
// On the panel a finger arrives as a mouse, so the drag is turned into a scroll by hand, exactly
// as it is for the lists (app.js). Only while a page is up: every other picture keeps its press.
if (KIOSK) dragScroll(stage, () => document.body.classList.contains('page'));
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
// ...and how long its picture gets to be drawn before we stop waiting and let the kiosk uncover
// onto a recording that stays black. This comes out of the same budget as the line above: the
// kiosk allows PAINT_WAIT_S = 8 s for the poll, the paint and this together, so 0.7 s is a
// ceiling nothing should reach rather than a wait anybody will feel. See raster() below.
const SCRATCHPAD_STILL_MS = 700;
const SCRATCHPAD_STILL_Q = 0.85;  // a sheet of flat white and black text; artefacts show early

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
  stage.scrollTop = 0;  // a page opens at the top, whatever the last one was scrolled to
}

// How long to wait for a video to be ready before uncovering anyway. The kiosk gives the whole
// poll-and-paint round PAINT_WAIT_S = 8 s, and the poll has already eaten some of it.
const CLIP_READY_MS = 4000;

// A video on the stage, already seeked to the second worth watching. Same promise as shot():
// do not come back until there is something worth uncovering onto - here that means seeked and
// not merely loaded, because uncovering on a loaded-but-unseeked video shows second zero for a
// beat and then jumps, which reads as a bug rather than as a start.
// The seeker, built here rather than in the template because the video it belongs to is built
// here too and the two have to go up and come down together. Markup, classes and gesture are the
// recording player's (app.js, views.css .scrub): one press-and-drag on a 6 px bar under the
// picture, green fill, white while a finger is on it.
//
// Under the picture and never on it, which is the reason that player gives and it is the same
// reason here: a demonstration is filmed across a bench and the bottom of the frame is where
// somebody's hands are. It is also the only thing on this screen that takes a press without
// putting the video away, so it has to be somewhere a thumb can find without aiming.
function seeker(video) {
  const bar = document.createElement('div');
  bar.className = 'scrub';
  const fill = document.createElement('div');
  fill.className = 'scrub-fill';
  bar.appendChild(fill);

  const paint = () => {
    fill.style.width = (video.duration ? 100 * video.currentTime / video.duration : 0) + '%';
  };
  // object-fit: contain letterboxes inside the element and leaves nothing in the DOM to measure,
  // so the pillar is worked out and taken off each end - the bar stops where the picture does
  // rather than running out over the black. On a 16:9 video at 800x480 this comes to nothing,
  // which is the point: it costs nothing when it is not needed.
  const fit = () => {
    const box = video.getBoundingClientRect();
    if (!box.width || !box.height) return;
    const ratio = (video.videoWidth || 16) / (video.videoHeight || 9);
    const wide = box.width / box.height > ratio;
    const pad = wide ? (box.width - box.height * ratio) / 2 : 0;
    bar.style.marginLeft = pad + 'px';
    bar.style.marginRight = pad + 'px';
  };
  const seek = (e) => {
    const box = bar.getBoundingClientRect();
    if (!video.duration || !box.width) return;
    const along = Math.min(1, Math.max(0, (e.clientX - box.left) / box.width));
    video.currentTime = along * video.duration;
    paint();
  };
  bar.addEventListener('pointerdown', (e) => { bar.setPointerCapture(e.pointerId); seek(e); });
  bar.addEventListener('pointermove', (e) => { if (bar.hasPointerCapture(e.pointerId)) seek(e); });
  video.addEventListener('timeupdate', paint);
  video.addEventListener('loadedmetadata', () => { fit(); paint(); });
  window.addEventListener('resize', fit);
  return bar;
}

async function clip(url, start) {
  const video = document.createElement('video');
  video.className = 'clip';
  video.playsInline = true;
  video.preload = 'auto';
  // Unmuted on purpose, and it works because the kiosk's Chromium runs with
  // --autoplay-policy=no-user-gesture-required - the tap that asked for this landed on the
  // kiosk's own window and the page never saw it. Elsewhere (a phone on the LAN) the policy
  // still applies, so take the browser's refusal and play it muted rather than not at all.
  video.muted = false;
  video.src = url;
  stage.textContent = '';
  stage.appendChild(video);
  // The class before the bar: the stage has to be a two-row grid before the bar is measured, or
  // fit() reads a picture that is still the full height of the stage.
  document.body.classList.add('clipping');
  stage.appendChild(seeker(video));
  await new Promise((ready) => {
    let done = false;
    const settle = () => { if (!done) { done = true; ready(); } };
    // Seek from the metadata event: currentTime before the duration is known is discarded.
    video.addEventListener('loadedmetadata', () => {
      if (start > 0 && start < video.duration) video.currentTime = start;
      else settle();  // nothing to wait for; canplay is enough
    }, { once: true });
    video.addEventListener('seeked', settle, { once: true });
    video.addEventListener('canplay', () => { if (start <= 0) settle(); }, { once: true });
    video.addEventListener('error', settle, { once: true });
    setTimeout(settle, CLIP_READY_MS);
  });
  try {
    await video.play();
  } catch (e) {
    video.muted = true;
    try { await video.play(); } catch (e2) { /* a still frame beats a blank stage */ }
  }
  // It ran out on its own. Nobody is going to press a screen they have stopped watching, so the
  // panel takes itself back rather than sitting on a stopped last frame for fifteen minutes.
  video.addEventListener('ended', () => { if (!KIOSK) drop(); window.__leave(); }, { once: true });
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
  return frame;
}

// The same scratchpad again, as a JPEG, for the recording. A session recording the screen samples
// what the kiosk paints, and while this page has the glass the kiosk paints nothing - so
// everything else on the panel reaches the video out of the offer file it arrived in. A scratchpad
// arrives as markup and has no pixels there to reach it with, and the video went black for exactly
// the minutes somebody spent reading what they had asked for.
//
// Drawn from the markup and not from the frame above, because the frame cannot be read: `sandbox`
// with no tokens gives it an origin of its own, so even the page around it is another origin. What
// goes in here is the identical string, so the two renderings cannot drift apart on anything but a
// browser bug.
//
// Nothing external can load: SVG rendered as an image runs no script and fetches nothing, the
// document's own CSP says the same, and [src] pointing anywhere but data: is dropped below. That is
// what keeps the canvas untainted, and an untainted canvas is the whole reason toDataURL works.
async function raster(frame, html) {
  const box = frame.getBoundingClientRect();
  const width = Math.round(box.width) || 800, height = Math.round(box.height) || 480;
  // foreignObject is parsed as XML, and what the model wrote is HTML - `<br>`, an unclosed <li>,
  // a bare `&`. Parsing it as HTML and serialising it as XML is what makes those legal: the
  // forgiving parser fixes them and the strict serialiser writes them back closed and escaped.
  const page = new DOMParser().parseFromString(SCRATCHPAD_HEAD + html, 'text/html');
  page.querySelectorAll('script, [src]:not([src^="data:"])').forEach((n) => n.remove());
  // documentElement, not body: SCRATCHPAD_HEAD styles `html, body` and does all its centring,
  // sizing and padding there. Serialising the root keeps a real <body> for those rules to land on
  // and declares the XHTML namespace itself. Wrapping the body's children in a bare <div> instead
  // points every one of them at the SVG root, and the picture stops looking like the panel.
  const xml = new XMLSerializer().serializeToString(page.documentElement);
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}"` +
              ` viewBox="0 0 ${width} ${height}">` +
              `<foreignObject width="${width}" height="${height}">${xml}</foreignObject></svg>`;
  const img = new Image(width, height);
  // encodeURIComponent and not btoa: btoa throws on anything outside Latin-1, and a scratchpad is
  // full of degree signs, plus-minus and emoji.
  img.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg);
  await img.decode();
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const pen = canvas.getContext('2d');
  pen.fillStyle = '#fff';   // the sheet. A JPEG has no transparency, so unpainted would come out black
  pen.fillRect(0, 0, width, height);
  pen.drawImage(img, 0, 0, width, height);
  return canvas.toDataURL('image/jpeg', SCRATCHPAD_STILL_Q);
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
  document.body.classList.remove('photo', 'paper', 'sketching', 'clipping', 'page');
  // The class before the picture: a stage that is still display:none has no size, and an image
  // fitted to a box of zero by zero paints nowhere at all.
  window.__drawing(true);
  // Whatever it is, a press anywhere puts it away, so body.photo goes on before either branch.
  // There was a `drawn` flag here until 2026-09-05 that gave a diagram a small square in the
  // corner instead - a second way out for one gesture, and the one nobody finds on a screen they
  // are not standing in front of. Everything the panel shows is a picture now. What makes the
  // press work over a scratchpad is `pointer-events: none` on .scratchpad - see panel.css.
  document.body.classList.add('photo');
  let kept = '';
  if (found.scratchpad) {
    const frame = await scratchpad(found.scratchpad);
    // Only the kiosk's page draws this. A companion across the room is showing the same markup on
    // a phone, and a picture of a phone is not a picture of the panel.
    //
    // Raced rather than awaited outright: a scratchpad whose picture will not draw must cost a
    // black stretch of recording and never the reveal. Every way this can fail - markup the XML
    // parser still refuses, a canvas that came out tainted after all - ends in an empty body,
    // which is exactly what the panel did before any of this existed.
    if (KIOSK) {
      kept = await Promise.race([
        raster(frame, found.scratchpad).catch(() => ''),
        new Promise((giveUp) => setTimeout(() => giveUp(''), SCRATCHPAD_STILL_MS)),
      ]);
    }
  } else if (found.sketch) {
    // Nothing to paint. The renderer in #sketchframe was loaded by the sketch's first frame and has
    // been drawing frames off the companion stream for a few hundred milliseconds already - see
    // sketch.js. All this does is stop covering it up, which is why there is no await here and
    // why the kiosk is told it may uncover on the very next line.
    //
    // The recording gets nothing, and that is a known hole rather than an oversight: raster()
    // redraws a scratchpad from the markup it was handed, and there is no equivalent here - the
    // renderer paints into a shadow root from a stylesheet foreignObject cannot see. So a session
    // that sketches has a stretch of black video for as long as it is up, exactly as a scratchpad
    // did before raster() existed. A single grim capture from the kiosk once it settles is the
    // cheap fix if it turns out to matter.
    document.body.classList.add('sketching');
  } else if (found.video) {
    // Before found.image, and that order is load-bearing. A video offer carries a thumbnail
    // under `image` as well - cyclops.panel.offer_video explains why: still.of_panel reads
    // that key and nothing else, so without it the session recording is black for as long as
    // the video runs. Test the picture first and the panel paints the title card and never
    // starts anything. See tests/test_video.py, which pins this order.
    await clip(found.video, found.start || 0);
  } else if (found.image) {
    // A page of a manual goes edge to edge and scrolls (panel.css). The class before the picture,
    // for the reason the stage's own class goes first above.
    if (found.page) document.body.classList.add('page');
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
  // The picture rides in this request rather than one of its own, and that is what makes the
  // ordering safe: the kiosk reads the still once, the moment this flag appears, so anything
  // posted separately would be racing a reader that has already been. views.picture_painted
  // writes it down before it touches the flag.
  try { await fetch('/panel/painted', { method: 'POST', body: kept }); } catch (e) {}
}

// Nothing on the stage, and the page uncovered again. Let go of the image while we are here: on
// the panel there is one browser and no tabs, so the next picture is the only thing that will
// ever want this memory back.
function drop() {
  window.__drawing(false);
  document.body.classList.remove('photo', 'paper', 'sketching', 'clipping', 'page');
  // Paused before it is dropped. Detaching the element is enough in practice, but "in practice"
  // is doing a lot of work there and the failure would be a voice still talking from a screen
  // showing something else.
  const playing = stage.querySelector('video');
  if (playing) playing.pause();
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
