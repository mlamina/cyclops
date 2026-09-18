/* The card, in the browser: seven screens, one document, no navigation.

   The kiosk's browser must never navigate - it is warm, already pointed here, holding a 400 ms
   watch on the panel, and spawning it costs 1.4 s against the 2 ms this page costs (kiosk.py).
   So the hash picks a view and a body class shows it.

   Plain DOM throughout. There is no build step here and nothing vendored any more - the 527 KB
   of JointJS this used to mention came out with the drawing schema - and the panel has no
   internet to reach a CDN over.

   Publishes window.__drawing, which panel.js calls when a picture takes the panel.
 */

// ---------------------------------------------------------------- the card, in the browser
//
// Four screens, one document, no navigation - see the .view rules. Everything below is plain
// DOM: there is no build step here and nothing vendored to reach for, and the panel has no
// internet to reach a CDN over.

// The narrow layout does not *hide* the transcript, it never asks the server for it, which is
// the whole of "everything on the desktop, only the video on the panel".
//
// The stylesheet owns this query and this reads it, so there is one definition of it rather
// than two strings that have to be kept in step. It is not a layout question - lan.css answers
// those with auto-fit and wrapping - it is whether this is a device you sit down and read on,
// and the answer decides whether the transcript is fetched at all. See --wide-q in lan.css.
const WIDE = window.matchMedia(
  getComputedStyle(document.documentElement).getPropertyValue('--wide-q').trim().slice(1, -1));

const VIEWS = ['view-status', 'view-live', 'view-sessions', 'view-session', 'view-media',
               'view-highlights', 'view-projects', 'view-project', 'view-manuals'];
// The five things a project is, in the rail's order. The first is where a project opens, and an
// address naming none of them - or one this list has never heard of - lands there too.
const SECTIONS = ['overview', 'log', 'sheets', 'photos', 'youtube', 'files'];
const FRESH = 20000;   // a listing is worth re-using for as long as nothing new can have ended

const vSessions = document.getElementById('v-sessions');
const vMedia = document.getElementById('v-media');
const vHighlights = document.getElementById('v-highlights');
// The reel. Every identifier here is reel*-prefixed on purpose: status.js, stream.js and panel.js
// share this global scope, and a second const of a name one of them already declares is a
// SyntaxError that takes the whole page down.
// reelBox is the picture alone and reelStack is the picture plus the bar under it. The split
// matters twice: the tap gesture is measured against the picture (a finger on the seeker must not
// skip the clip), and an empty reel hides the stack (hiding the picture alone would leave the bar
// behind on its own).
const reelBox = document.getElementById('reel');
const reelStack = document.getElementById('reelstack');
const reelPair = [document.getElementById('reel0'), document.getElementById('reel1')];
const reelFill = document.getElementById('reelfill');
const reelFlashEl = document.getElementById('reelflash');
const reelTitle = document.getElementById('reeltitle');
const reelWhen = document.getElementById('reelwhen');
const vProjects = document.getElementById('v-projects');
// Null on the panel: the manuals screen is rendered for the LAN only, so every
// listener below is guarded on it the way the upload controls guard on putBar.
const vManuals = document.getElementById('v-manuals');
const crumbTrail = document.getElementById('crumbtrail');
const railName = document.getElementById('railname');
const rails = [...document.querySelectorAll('.railtab')];
// The one element every section of a project screen writes: a listing, a document or a grid. It
// is the pane's *body* and not the pane, because the crumb bar above it holds controls whose
// listeners are bound once at load and must outlive a section change.
const pbody = document.getElementById('pbody');
const video = document.getElementById('video');
const stitle = document.getElementById('stitle');
const smeta = document.getElementById('smeta');
const ssum = document.getElementById('ssum');
const talk = document.getElementById('talk');
// Companion mode's three, and they are only on the page away from the panel ({% if not local %}
// in the template), so everything below null-checks the way the controls in status.js do.
const liveTalk = document.getElementById('livetalk');
const liveTitle = document.getElementById('livetitle');
const liveBar = document.getElementById('livebar');
const liveView = document.getElementById('v-live');
// The panel navigates with four tabs and every other client with one menu (see dashboard.html).
// Only one of these is ever on the page, and the code below simply drives whichever it found.
const tabs = [...document.querySelectorAll('.tab')];
// Not `pick`: that id belongs to the Upload button down in the crumb bar, and an id claimed
// twice is silently won by whichever element is earlier in the document.
const menu = document.getElementById('menu');
const playing = document.querySelector('.playing');
const scrub = document.getElementById('scrub');
const scrubFill = document.getElementById('scrubfill');

const esc = (text) => String(text == null ? '' : text)
  .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
  .replace(/"/g, '&quot;').replace(/'/g, '&#39;');

// m:ss, the same clock session.md puts beside every line - and the same number the video is
// seeked to, because the transcript's t *is* the recording's timeline.
const clock = (t) => {
  const whole = Math.max(0, Math.floor(t || 0));
  return Math.floor(whole / 60) + ':' + String(whole % 60).padStart(2, '0');
};
const span = (seconds) => {
  const whole = Math.round(seconds || 0);
  if (whole >= 3600) return Math.floor(whole / 3600) + 'h ' + String(Math.floor((whole % 3600) / 60)).padStart(2, '0') + 'm';
  if (whole >= 60) return Math.floor(whole / 60) + 'm ' + (whole % 60) + 's';
  return whole + 's';
};
const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
const DAYS = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
const day = (iso) => {
  const at = new Date(iso);
  if (isNaN(at)) return '';
  return DAYS[at.getDay()] + ' ' + at.getDate() + ' ' + MONTHS[at.getMonth()];
};
const time = (iso) => {
  const at = new Date(iso);
  if (isNaN(at)) return '';
  return String(at.getHours()).padStart(2, '0') + ':' + String(at.getMinutes()).padStart(2, '0');
};
const megabytes = (n) => !n ? '' : (n / 1048576).toFixed(1) + ' MB';
// On a still it is a running time, not a duration in prose - written the way a player writes it.
const stamp = (seconds) => {
  const whole = Math.max(0, Math.round(seconds || 0));
  const mm = Math.floor(whole / 60) % 60, ss = String(whole % 60).padStart(2, '0');
  const hh = Math.floor(whole / 3600);
  return hh ? hh + ':' + String(mm).padStart(2, '0') + ':' + ss : mm + ':' + ss;
};
const many = (n, one) => n === 1 ? '1 ' + one : n + ' ' + one + 's';
// What a file weighs, in the unit it is actually in. megabytes() above is for recordings; a
// 692-byte README rendered through it reads "0.0 MB", which is worse than saying nothing.
const heft = (n) => !n ? '' : n < 1024 ? n + ' B'
  : n < 1048576 ? Math.round(n / 1024) + ' KB' : (n / 1048576).toFixed(1) + ' MB';
// A project's `updated` is a date with no time. new Date() reads a bare YYYY-MM-DD as UTC
// midnight, which west of Greenwich is the day before - so it is pinned to local time first.
const dated = (iso) => day(/^\d{4}-\d{2}-\d{2}$/.test(iso || '') ? iso + 'T00:00' : iso);

// What a scratchpad said, in a few words. The markup the model wrote is one screenful of
// headings and list items, so nothing here needs to be a parser - the answer only has to be
// recognisable. Same job as session._scratchpad_gist, and the same 70 characters.
const GIST = 70;
const gist = (html) => {
  const text = String(html || '').replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
  return text.length > GIST ? text.slice(0, GIST - 1).trimEnd() + '…' : text;
};

async function grab(url) {
  const r = await fetch(url, { cache: 'no-store' });
  if (!r.ok) throw new Error(r.status);
  return r.json();
}

// ---------------------------------------------------------------- the sessions

let held = null, heldAt = 0;
async function listing() {
  if (held && Date.now() - heldAt < FRESH) return held;
  held = (await grab('/api/sessions')).sessions;
  heldAt = Date.now();
  return held;
}

// The still is the recording's own frame at two seconds, asked for with a media fragment. It
// costs no poster file, no ffmpeg run and nothing on the card - and it only works at all
// because the server answers byte ranges now. Mounted as it scrolls into view, so a card with
// fifty sessions does not open fifty of them at once.
const watching = new IntersectionObserver((rows) => {
  for (const row of rows) {
    if (!row.isIntersecting) continue;
    const el = row.target;
    watching.unobserve(el);
    el.preload = 'metadata';
    el.src = el.dataset.src + '#t=2';
  }
}, { rootMargin: '200px' });

function sessionRow(one) {
  // What a row says, and where. The still carries how long it runs, the corner carries when it
  // was, and the rest is the title. The project it was filed under and the count of drawings
  // were both true, and neither earned a line of a list you are scanning.
  const still = one.video
    ? '<video class="thumb" muted playsinline preload="none" data-src="/media/' +
      encodeURIComponent(one.name) + '/video.mp4"></video>' +
      '<span class="len">' + stamp(one.seconds) + '</span>'
    : '<span class="thumb none">no video</span>';
  // One line under the date, and a warning outranks a count: a session still recording, or one
  // that never finished, is the thing about that row you need to know.
  let sub = '';
  if (one.verdict === 'live') sub = '<span class="rowsub live">recording</span>';
  else if (one.verdict !== 'finished') sub = '<span class="rowsub broke">' + one.verdict + '</span>';
  else if (one.photos) sub = '<span class="rowsub">' + many(one.photos, 'photo') + '</span>';
  return '<button class="row" type="button" data-open="' + esc(one.name) + '">' +
    '<span class="shot">' + still + '</span>' +
    '<span class="rowtext"><span class="rowtitle">' + esc(one.title) + '</span>' +
    '<span class="rowsum">' + esc(one.summary) + '</span></span>' +
    '<span class="rowwhen">' + day(one.started) +
    '<span class="rowtime">' + time(one.started) + '</span>' + sub + '</span>' +
    '</button>';
}

async function showSessions() {
  try {
    const found = await listing();
    vSessions.innerHTML = found.length
      ? found.map(sessionRow).join('')
      : '<div class="empty">nothing recorded yet</div>';
    for (const el of vSessions.querySelectorAll('video.thumb')) watching.observe(el);
  } catch (e) {
    vSessions.innerHTML = '<div class="empty">could not read the card</div>';
  }
}

// ---------------------------------------------------------------- the reel

// panel.js declares its own KIOSK on the same global scope - these are classic scripts and
// not modules, so a second const of that name is a SyntaxError that takes the page down.
const ON_PANEL = document.body.classList.contains('kiosk');

// The whole state of the screen: what there is, which one is playing, and which of the two
// elements is showing it. Sound is not in here, because it is not a setting: a clip of somebody
// talking with the talking taken out is a silent film of a bench, so the reel always wants sound.
// The only thing that ever takes it away is the browser refusing, which reelShow handles.
let reelList = [], reelAt = 0, reelOn = 0;

function reelSrcOf(i) {
  const one = reelList[((i % reelList.length) + reelList.length) % reelList.length];
  return one ? one.src : '';
}

function reelInto(el, i) {
  const src = reelSrcOf(i);
  if (!src || el.dataset.src === src) return;   // already holding it: do not fetch it twice
  el.dataset.src = src;
  el.src = src;
  el.load();
}

function reelPaint() {
  const one = reelList[reelAt] || {};
  reelTitle.textContent = one.title || '';
  reelWhen.textContent = (one.started ? day(one.started) + ' · ' : '') +
    (reelList.length > 1 ? (reelAt + 1) + ' / ' + reelList.length : '');
}

function reelShow(i) {
  if (!reelList.length) return;
  reelAt = ((i % reelList.length) + reelList.length) % reelList.length;
  const front = reelPair[reelOn], back = reelPair[1 - reelOn];
  reelInto(front, reelAt);
  back.pause();
  back.classList.remove('on');
  front.classList.add('on');
  front.muted = false;
  try { front.currentTime = 0; } catch (e) { /* not seekable yet; it starts at 0 anyway */ }
  // Unmuted autoplay is the browser's to refuse, and a refusal must not be swallowed. Nothing
  // recovers a rejected play(): `ended` never fires, so the reel would sit frozen on a black
  // rectangle and not even step on. So take the sound off and go again - one clip arriving quiet
  // beats a screen that looks broken.
  //
  // **Muting here is about this attempt and never about the reel.** There is no mute control on
  // this screen and no flag behind one - every clip asks for sound again, because the refusal is
  // temporary and nothing tells us when it lifts. Open the page straight on #/highlights, where
  // the browser has had no gesture yet, and the first clip is silent; step or let it run on and
  // the next one has sound, because by then you have touched the page. An earlier version latched
  // a flag here instead and every clip after the first was silent for the life of the page.
  //
  // The cost of asking is one refused play() per clip, which is a rejected promise and nothing a
  // viewer can see. On the panel it is never refused at all: kiosk.py's CHROME_FLAGS carries
  // --autoplay-policy, because the tap that wakes the screen lands on the kiosk and not in here.
  front.play().catch(() => {
    front.muted = true;
    front.play().catch(() => {});
  });
  reelPaint();
  // Back to empty, and here rather than in the timeupdate handler because this is the one
  // function every switch goes through. Nothing used to reset it: the bar just snapped back at
  // the new clip's first timeupdate, which nobody could see on a 3 px hairline over the picture
  // and everybody can see on a 6 px bar under it.
  reelFill.style.width = '0%';
  reelInto(back, reelAt + 1);   // the next one buffers behind the one you are watching
}

// The edge between two clips. Kept off reelShow on purpose: reelShow is the choke point but two
// of its callers are not a switch - the first paint on arriving at the screen, and picking the
// reel back up after a drawing has covered it (same clip, same position). Hung on next/prev it
// catches every real one: the `ended` advance, the skip over a clip that will not decode, a tap
// on either third, and the arrow keys.
let reelFlashAt = null;
// A white flash is exactly what this setting exists to refuse, so it is a check and not a
// softening. It has to be asked here rather than in CSS because the animation is driven from
// here; on the panel it never matches, so the panel always flashes.
const reelStill = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)');
function reelFlash() {
  if (!reelFlashEl || (reelStill && reelStill.matches)) return;
  // Mashing an arrow key must not stack animations on one element, each fading from wherever the
  // last had got to.
  if (reelFlashAt) reelFlashAt.cancel();
  reelFlashAt = reelFlashEl.animate(
    [{ opacity: 0 }, { opacity: .85, offset: .12 }, { opacity: 0 }],
    { duration: 220, easing: 'ease-out' });
}

function reelNext() { reelFlash(); reelOn = 1 - reelOn; reelShow(reelAt + 1); }
function reelPrev() { reelFlash(); reelOn = 1 - reelOn; reelShow(reelAt - 1); }

for (const el of reelPair) {
  // Only the one on screen advances: the other's `ended` is the previous clip finishing behind
  // the swap, and acting on it would skip one.
  el.addEventListener('ended', () => { if (el.classList.contains('on')) reelNext(); });
  // A clip that 404s or will not decode is stepped over rather than left frozen on a black
  // rectangle, which is the one failure that makes the whole screen look broken.
  el.addEventListener('error', () => { if (el.classList.contains('on')) reelNext(); });
  el.addEventListener('timeupdate', () => {
    if (!el.classList.contains('on') || !el.duration) return;
    reelFill.style.width = (100 * el.currentTime / el.duration) + '%';
  });
}

// Instagram's gesture, because everybody already has it: the sides step, the middle is the
// picture itself. The middle used to toggle sound and now it pauses, which is what a tap on a
// playing video means everywhere else - and the thing you actually want when somebody walks up to
// the bench mid-clip. Pausing silences it too, so nothing was lost by dropping the mute.
if (reelBox) {
  reelBox.addEventListener('click', (e) => {
    const where = (e.clientX - reelBox.getBoundingClientRect().left) / reelBox.clientWidth;
    if (where < 0.3) return reelPrev();
    if (where > 0.7) return reelNext();
    reelPause();
  });
}

// One name for it, because the tap and the spacebar must not drift apart.
function reelPause() {
  const front = reelPair[reelOn];
  if (front.paused) front.play().catch(() => {}); else front.pause();
}

document.addEventListener('keydown', (e) => {
  if (at !== '/highlights') return;
  if (e.key === 'ArrowRight') reelNext();
  else if (e.key === 'ArrowLeft') reelPrev();
  else if (e.key === ' ') {
    e.preventDefault();
    reelPause();
  }
});

// Whatever the caption block has to say when there is no clip to name. An empty reel is never a
// blank screen with no explanation - but the explanation is one sentence now. The counts that used
// to sit under the picture ("89 sessions looked at · 18 had something in them") are gone: they
// answered a question about the indexer on a screen for watching, and they were the last thing on
// it that was not the clip. /api/highlights still returns them for anything that wants them.
function reelSay(words) {
  reelTitle.textContent = words;
  reelWhen.textContent = '';
}

async function showHighlights() {
  try {
    const got = await grab('/api/highlights');
    const same = reelList.length === got.clips.length &&
                 reelList.every((c, i) => c.id === got.clips[i].id);
    reelList = got.clips;
    reelStack.hidden = !reelList.length;
    if (!reelList.length) {
      reelSay('nothing worth a clip yet');
      return;
    }
    if (same && reelPair[reelOn].dataset.src) return;   // a repoll must not restart the reel
    reelOn = 0;
    reelShow(0);
  } catch (e) {
    reelSay('could not read the card');
    reelStack.hidden = true;
  }
}

// Leaving the screen. Without the load() the decoder keeps the last file open behind whatever
// you opened instead, which on the panel is a whole core of a four-core board.
function stopReel() {
  for (const el of reelPair) {
    el.classList.remove('on');
    el.pause();
    el.removeAttribute('src');
    el.dataset.src = '';
    el.load();
  }
}

// ---------------------------------------------------------------- one session

// The same words session.md uses for the things that are not speech, because they are the same
// events and a reader should not have to learn them twice.
function aside(record) {
  const kind = record.type;
  if (kind === 'search') {
    const query = '"' + esc(record.query || '') + '"';
    if (record.error) return 'Searched the web — ' + query + ' → failed';
    return 'Searched the web — ' + query + ' → ' + (record.chars || 0) + ' characters back';
  }
  if (kind === 'recall') {
    // He went looking on the card for a picture somebody described out loud. Same words as
    // session.md uses for it, and the reason this branch exists at all: `recall` has always been
    // in library.SPOKEN, so without it every one of these read "Something went wrong — " with an
    // empty message. Invisible in a finished transcript nobody re-reads; not invisible at all on
    // a companion watching the line land.
    const asked = '"' + esc(record.query || '') + '"';
    if (!record.title) return 'Looked for — ' + asked + ' → nothing on the card matched';
    return 'Looked for — ' + asked + ' → <b>' + esc(record.title || '') + '</b>' +
           (record.shown ? ' and put it on the panel' : '');
  }
  if (kind === 'screen') {
    // The scratchpad: the one thing he writes while he is still talking. What it said, out of the
    // markup he wrote - the same gist session.py takes, taken here from the same field.
    return 'Wrote on the scratchpad — ' + esc(gist(record.html || ''));
  }
  if (kind === 'transcript_failed') {
    return 'You said something that could not be transcribed';
  }
  if (kind === 'tutorial') {
    // A walkthrough, in session.md's words. In SPOKEN, so without this branch every step would
    // read "Something went wrong — ", which is the trap `recall` fell into first.
    if (record.action === 'started') {
      const steps = record.steps || [];
      return 'Started walking you through ' + steps.length + ' steps — ' + esc(steps.join('; '));
    }
    if (record.action === 'advanced') {
      return 'Step ' + record.step + ' of ' + record.of + ' — ' + esc(record.label || '');
    }
    if (record.action === 'finished') return 'Finished the walkthrough';
    return 'Stopped the walkthrough at step ' + record.step + ' of ' + record.of;
  }
  if (kind === 'project') {
    const verb = record.action === 'tracked' ? 'Started keeping notes on' : 'Looked up its notes on';
    return verb + ' — <b>' + esc(record.name || '') + '</b>';
  }
  if (kind === 'data') {
    const project = '<b>' + esc(record.project || '') + '</b>';
    const keys = esc((record.keys || []).join(', '));
    if (record.action === 'saved') return 'Wrote down — ' + project + ' / ' + esc(record.tab || '') + ': ' + keys;
    if (record.action === 'forgot') return 'Deleted a value — ' + project + ' / ' + esc(record.tab || '') + ': ' + keys;
    return 'Looked up a value — "' + esc(record.query || '') + '" in ' + project +
           ' → ' + (record.hits ? record.hits + ' found' : 'nothing written down');
  }
  if (kind === 'photo') {
    // Two of these are pictures Cyclops made rather than ones anybody took. Same record type
    // because they are jpgs in photos/ like any other; `by` is what tells them apart.
    if (record.by === 'drawn') {
      const asked = '"' + esc(record.request || '') + '"';
      if (record.error) return 'Tried to draw — ' + asked + ' → failed';
      return 'Drew a diagram — ' + asked;
    }
    if (record.by === 'edit') {
      const what = '"' + esc(record.request || '') + '"';
      if (record.error) return 'Tried to imagine a change — ' + what + ' → failed';
      return 'Imagined a change — ' + what;
    }
    const seen = record.shown ? 'Cyclops looked at it.' : 'Cyclops never saw this one.';
    return 'You took a photo — the shutter button. ' + seen;
  }
  return 'Something went wrong — ' + esc(record.message || '');
}

function line(record) {
  const at = '<span class="at">' + clock(record.t) + '</span>';
  const kind = record.type;
  if (kind === 'you' || kind === 'cyclops') {
    const mark = record.interrupted ? ' (interrupted)' : '';
    return '<div class="line ' + kind + '" data-t="' + (record.t || 0) + '">' + at +
      '<span><span class="who">' + (kind === 'you' ? 'You' : 'Cyclops') + mark + '</span>' +
      '<span class="said">' + esc(record.text || '') + '</span></span></div>';
  }
  let shot = '';
  if (record.url) {
    shot = '<img class="inline" loading="lazy" draggable="false" src="' +
           esc(record.url) + '" alt="">';
  }
  return '<div class="line ' + (kind === 'error' ? 'bad' : '') + '" data-t="' + (record.t || 0) + '">' +
    at + '<span><span class="aside">' + aside(record) + '</span>' + shot + '</span></div>';
}

// ---------------------------------------------------------------- the player

// How wide the picture actually is. object-fit: contain letterboxes inside the element and
// leaves nothing in the DOM to hang anything off, so the pillar has to be measured - and the
// same arithmetic is right on a phone held upright, where it comes out as nothing to take off.
//
// The seeker is narrowed rather than offset: it is a grid row under the video now (views.css),
// so what it needs is the pillar taken off each end, and the row it sits in already puts it the
// reel's 6 px below the frame.
function fit() {
  if (!document.body.classList.contains('solo')) return;
  const box = video.getBoundingClientRect();
  if (!box.width || !box.height) return;
  const ratio = (video.videoWidth || 5) / (video.videoHeight || 3);  // 5:3 until it says
  const wide = box.width / box.height > ratio;
  const w = wide ? box.height * ratio : box.width;
  const h = wide ? box.height : box.width / ratio;
  const padX = (box.width - w) / 2;
  scrub.style.marginLeft = padX + 'px';
  scrub.style.marginRight = padX + 'px';
}

function solo(on) {
  document.body.classList.toggle('solo', on);
  video.controls = !on;   // the browser's own bar is desktop furniture at 800x480
  if (on) requestAnimationFrame(fit);
}

const paint = () => {
  scrubFill.style.width = (video.duration ? 100 * video.currentTime / video.duration : 0) + '%';
};
video.addEventListener('timeupdate', paint);
video.addEventListener('loadedmetadata', () => {
  fit();
  paint();
});
window.addEventListener('resize', fit);

// A press puts the recording away, which is what a press on a picture means everywhere else on
// this panel - the lightbox, and a drawing on the stage (system.css). Hung on .playing and not on
// the video so that the black beside a 4:3 frame, and the name written across the top of it,
// close it too: the target is the screen, and the seeker is the one thing cut out of it.
//
// It is the only way back now. The way out to the camera is not on this screen either, so a
// press that landed on nothing would leave the panel with a video and no exit.
if (playing) {
  playing.addEventListener('click', (e) => {
    if (!document.body.classList.contains('solo')) return;
    if (e.target.closest('.scrub')) return;
    location.hash = '#/sessions';
  });
}

const seek = (e) => {
  const box = scrub.getBoundingClientRect();
  if (!video.duration || !box.width) return;
  video.currentTime = Math.min(1, Math.max(0, (e.clientX - box.left) / box.width)) * video.duration;
  paint();
};
scrub.addEventListener('pointerdown', (e) => { scrub.setPointerCapture(e.pointerId); seek(e); });
scrub.addEventListener('pointermove', (e) => {
  if (scrub.hasPointerCapture(e.pointerId)) seek(e);
});

// ---------------------------------------------------------------- one session

let openName = null;

async function transcript(name) {
  try {
    const found = (await grab('/api/session/' + encodeURIComponent(name) + '/records')).records;
    if (openName !== name) return;
    talk.innerHTML = found.length ? found.map(line).join('')
                                  : '<div class="empty">nothing was said</div>';
  } catch (e) { /* the video is the point; a missing transcript is not worth a broken page */ }
}

// The panel and the laptop are not the same screen, and a window can become the other one.
WIDE.addEventListener('change', () => {
  if (!openName) return;
  solo(!WIDE.matches);
  if (WIDE.matches && !talk.childElementCount) transcript(openName);
  else if (!WIDE.matches) talk.innerHTML = '';
});

// One session, one file. The clips this session produced are on the reel and are not opened
// here - a session screen is the recording it was made from, which is the thing a transcript
// can be scrubbed against.
async function showSession(name) {
  openName = name;
  talk.innerHTML = '';
  solo(!WIDE.matches);
  try {
    const one = await grab('/api/session/' + encodeURIComponent(name));
    if (openName !== name) return;   // you tapped through to another one while this landed
    stitle.textContent = one.title;
    const bits = [day(one.started) + ' ' + time(one.started), span(one.seconds)];
    if (one.entrypoint) bits.push(one.entrypoint);
    if (one.video_bytes) bits.push(megabytes(one.video_bytes));
    if (one.filed) bits.push('filed under ' + one.filed);
    smeta.textContent = bits.join(' · ');
    ssum.textContent = one.summary;
    if (one.video) {
      video.src = '/media/' + encodeURIComponent(name) + '/video.mp4';
      video.hidden = false;
      // Every recording asks for sound again. The muting below is about one refused attempt and
      // never about the screen: the refusal is temporary and nothing tells us when it lifts, so
      // a latched flag here is every session after the first one silent for the life of the page.
      video.muted = false;
      // You tapped a session to watch it, so watch it. On the panel kiosk.py's --autoplay-policy
      // settles it; everywhere else the click that got you here is the gesture the policy wants,
      // and it may well have been spent by the time this landed. There is no play button to fall
      // back on any more - a press is the way *out* - so a refusal is asked again without the
      // sound rather than left as a still frame nothing can start. Same move as the reel, and the
      // same reason: a recording playing quiet beats a screen that looks broken.
      video.play().catch(() => {
        video.muted = true;
        video.play().catch(() => {});
      });
    } else {
      video.removeAttribute('src');
      video.load();
      video.hidden = true;
    }
  } catch (e) {
    stitle.textContent = 'that session is not on the card';
    smeta.textContent = '';
    ssum.textContent = '';
  }
  if (!WIDE.matches) return;   // the panel is the video and its name, and asks for no more
  await transcript(name);
}

function letGo() {
  openName = null;
  solo(false);
  video.pause();
  video.removeAttribute('src');
  video.load();   // without this the decoder keeps the last file open behind the panel
  talk.innerHTML = '';
}

talk.addEventListener('click', (e) => {
  const el = e.target.closest('.line[data-t]');
  if (!el || !video.src) return;
  video.currentTime = parseFloat(el.dataset.t) || 0;
  video.play().catch(() => {});
});

// ---------------------------------------------------------------- following along

/* Companion mode: the conversation arriving as it is spoken.

   Every line here is drawn by the same line() the Sessions screen uses, out of the same records,
   with the same CSS - it is the same conversation, and there is no second way of reading one.
   What is different is that this one is not finished: the log is being appended to on the card
   while we read it, so this asks for what it has not got yet rather than for the whole thing.

   A poll and not a socket. The admin service is four gunicorn slots in front of a session it
   shares no memory with (they are different processes; everything between them is a file), and
   two seconds of latency on a line somebody has just said out loud is not a thing anybody can
   feel. What would be felt is the Pi getting warmer. */

const LIVE_EVERY = 2000, LIVE_SLOW = 10000;   // ...and slower still when nobody is looking
let liveName = null;   // the session this screen is showing, as the server last named it
let liveHeld = 0;      // how many of its lines we already have - the `since` we ask with
let liveGen = 0;       // bumped on the way out, so a timer from a screen you left dies quietly

function stopLive() {
  liveGen++;
  liveReset();
}

// Newest first, which is the other way up from the recording this transcript was written for.
// There you are reading a conversation that finished, from the beginning. Here it has not
// finished: the line worth seeing is the one he just said, and a screen propped against a vice
// should be showing it without being touched. So the newest lands at the top and the older ones
// go down, the way anything you follow rather than read does.
//
// Which turns the scroll question over with it. At the top, we stay at the top and the new line
// simply appears. Anywhere else, the reader has scrolled back to something and inserting above
// them would slide it out from under their eyes - so the insert's own height is added back to
// keep what they are reading exactly where it was.
const PINNED = 40;   // u-independent: a line's worth of slack for a finger mid-scroll

function prepend(el, html) {
  const held = el.scrollTop > PINNED ? el.scrollHeight : 0;
  el.insertAdjacentHTML('afterbegin', html);
  el.scrollTop = held ? el.scrollTop + (el.scrollHeight - held) : 0;
}

// The lines of one poll, newest first. They arrive oldest first - it is a log - and the block as
// a whole goes above everything already on screen, so reversing the block is the whole of it.
const newestFirst = (records) => records.slice().reverse().map(line).join('');

// What the screen is between sessions: the boot mark and what to press, centred, instead of a
// named session with nothing under it. The markup is in the template and never changes - this
// only says which of the two the screen is. Idle is *nothing to show*, not merely nothing
// running: a conversation that ended a minute ago is still worth reading, so it keeps the screen.
function liveIdle(on) {
  if (!liveView) return;
  if (on) liveView.dataset.idle = '1'; else liveView.removeAttribute('data-idle');
}

function liveHead(title, on) {
  if (liveTitle) liveTitle.textContent = title;
  if (liveBar) {
    // The header's own bottom rule, lit and travelling. Nothing to write and nothing to read: it
    // is either moving or it is not there. See .livebar in lan.css.
    if (on) liveBar.dataset.on = '1'; else liveBar.removeAttribute('data-on');
  }
  if (liveTalk) liveIdle(!on && !liveTalk.childElementCount);
}

// The screen with nothing on it: no session, no lines, no name in the head, and no light on the
// header's rule. What a companion shows most of the day, and where all three ways out of a
// session land - opening this screen, leaving it, and one ending underneath it.
//
// The transcript is emptied before liveHead is asked, and that ordering is the whole of it:
// liveIdle's test is "is there anything to show", so the mark and the button come back only once
// there is not. The light is liveHead's doing too, and it matters that it is: #livebar rides the
// header rule rather than this view, so it is on screen from the Sessions list as readily as from
// here, and leaving it lit on the way out would leave it sweeping with nothing left polling to
// ever put it out.
function liveReset() {
  liveName = null;
  liveHeld = 0;
  if (liveTalk) liveTalk.innerHTML = '';
  liveHead('', false);
}

async function liveTick(mine) {
  if (mine !== liveGen) return;
  try {
    const s = await grab('/api/live?name=' + encodeURIComponent(liveName || '') +
                         '&since=' + liveHeld);
    if (mine !== liveGen) return;   // you left while this was in flight
    if (!s.name) {
      // He stopped talking, and the screen goes back to what it opens on: the eye, and what to
      // press. The folder's lock goes the moment the session closes - well before it is renamed
      // and summarised - so this lands within a couple of seconds of the last word.
      //
      // This used to freeze the transcript instead, on the argument that you might still be
      // reading it. That is the wrong trade for this screen. A companion is propped against a
      // vice and glanced at, and one still showing a conversation that ended twenty minutes ago
      // is answering the only question it exists to answer - is he up, and what is he saying -
      // with something that stopped being true. Nothing is lost by clearing it: seconds later
      // that same conversation is a row on the Sessions screen with its video and its summary,
      // which is where a finished one is read.
      //
      // `liveName` is the guard, and no second flag is needed for it: it holds a name only while
      // a session is on the screen, so a name means there is something to put away and a null one
      // means this has already been done. Every poll after the first one has nothing to do at
      // all, which is what this branch is for the rest of the day.
      if (liveName) liveReset();
    } else {
      if (s.name !== liveName) {
        // A different session, or the first one. Start again, and put its name up: /api/session
        // is the same route the Sessions screen opens with, asked once rather than polled.
        liveName = s.name;
        liveHeld = 0;
        liveTalk.innerHTML = '';
        liveHead('listening', true);
        grab('/api/session/' + encodeURIComponent(s.name))
          .then((one) => { if (liveName === s.name) liveHead(one.title, true); })
          .catch(() => {});
      } else if (s.n < liveHeld) {
        // The log got shorter, which only happens when something rewrote it under us
        // (`cyclops-sessions --fix`, which push.sh runs). Our index means nothing now.
        liveHeld = 0;
        liveTalk.innerHTML = '';
      }
      if (s.records.length) {
        // Inserted, never re-rendered. Replacing the lot every two seconds would throw away the
        // scroll position and make every <img class="inline"> already on screen decode again.
        prepend(liveTalk, newestFirst(s.records));
      }
      liveHeld = s.n;
      // He is up but has not been spoken to yet. Not the idle screen - that one says "press the
      // button", and the button has been pressed.
      if (!liveTalk.childElementCount) {
        liveTalk.innerHTML = '<div class="empty">he is listening; nothing said yet</div>';
      }
    }
  } catch (e) { /* the service will come back; the lines already on screen are still true */ }
  if (mine !== liveGen) return;
  // Self-scheduling, like every other poll on this page: a slow Pi never stacks requests on
  // itself. And nothing is asked for at all while a picture is covering this screen.
  const hidden = document.visibilityState === 'hidden' ||
                 document.body.classList.contains('drawing');
  setTimeout(() => liveTick(mine), hidden ? LIVE_SLOW : LIVE_EVERY);
}

function showLive() {
  if (!liveTalk) return;   // the panel has no such screen
  // Opened on the idle screen rather than on an empty transcript, because until the first poll
  // lands that is the honest answer and it is the commoner one - which is stopLive's doing now,
  // since opening this screen and leaving it arrive at exactly the same blank.
  stopLive();
  liveTick(liveGen);
}

// ---------------------------------------------------------------- the projects

// Where a project, one section of it and one file inside it live in the hash. The project and the
// path are encoded whole, slashes included - shelf.file_route builds the same string server-side -
// so a nested Photos/x.jpg stays one segment. The section between them is a bare word out of
// SECTIONS, which is what lets a split on "/" find all three however deep the file is.
const pRoute = (name, sec, path) => '#/p/' + encodeURIComponent(name) + '/' + (sec || 'overview') +
  (path ? '/' + encodeURIComponent(path) : '');
const fRoute = (name, path) => '#/f/' + encodeURIComponent(name) + '/' + encodeURIComponent(path || '');
// #/p/<name>/<sec>/<path> and #/f/<name>/<path>. A file route has no section, so `sec` comes back
// empty and the caller supplies the one it means.
const hashParts = (rest, hasSection) => {
  const found = rest.split('/');
  return [decodeURIComponent(found[0] || ''),
          hasSection ? (found[1] || '') : '',
          decodeURIComponent(found[hasSection ? 2 : 1] || '')];
};

let shelfHeld = null, shelfAt = 0;
async function shelf() {
  if (shelfHeld && Date.now() - shelfAt < FRESH) return shelfHeld;
  shelfHeld = (await grab('/api/projects')).projects;
  shelfAt = Date.now();
  return shelfHeld;
}

function projectRow(one) {
  // The first photo in the folder, which is the one the sweep picked as the hero shot. No
  // media fragment and no <video> here: a project's still is a file, not a frame of one.
  const shot = one.thumb
    ? '<img class="thumb" loading="lazy" alt="" src="' + esc(one.thumb) + '">'
    : '<span class="thumb none">no photo</span>';
  // A project that is paused or done says so; one that is neither says how much is in it.
  const sub = one.status !== 'active'
    ? '<span class="rowsub">' + esc(one.status) + '</span>'
    : one.sessions ? '<span class="rowsub">' + many(one.sessions, 'session') + '</span>' : '';
  return '<button class="row" type="button" data-hash="' + esc(pRoute(one.name, 'overview')) + '">' +
    '<span class="shot">' + shot + '</span>' +
    '<span class="rowtext"><span class="rowtitle">' + esc(one.title) + '</span>' +
    '<span class="rowsum">' + esc(one.tagline) + '</span></span>' +
    '<span class="rowwhen">' + dated(one.updated) + sub + '</span>' +
    '</button>';
}

// ---------------------------------------------------------------- manuals

// How far through a manual the indexer has got. It is the only thing on this screen that moves,
// and it moves in a different process - so the row says what the frontmatter said when it was
// fetched, and a poll below re-asks while anything is unfinished.
function manualRow(one) {
  const cover = one.cover
    ? '<img class="thumb" loading="lazy" alt="" src="' + esc(one.cover) + '">'
    : '<span class="thumb none">reading</span>';
  const sub = one.ready
    ? many(one.pages, 'page')
    : one.pages ? 'reading ' + one.read + ' of ' + one.pages : 'reading';
  // The part, not the document's own title: "Instruction Manual" is what half of these call
  // themselves, and it tells you nothing about which one you are looking at.
  const title = one.part || one.name || one.folder;
  const also = one.aliases && one.aliases.length
    ? '<span class="rowsum">also called: ' + esc(one.aliases.join(', ')) + '</span>' : '';
  return '<div class="row manual">' +
    '<span class="shot">' + cover + '</span>' +
    '<span class="rowtext"><span class="rowtitle">' + esc(title) + '</span>' + also + '</span>' +
    '<span class="rowwhen">' + esc(sub) + (one.revision ? '<span class="rowsub">' +
      esc(one.revision) + '</span>' : '') + '</span>' +
    '</div>';
}

let manualPoll = 0;
function paintManuals(found) {
  if (!vManuals) return;
  const rows = found.manuals.length
    ? found.manuals.map(manualRow).join('')
    : '<div class="empty">no manuals yet - drop a PDF here</div>';
  vManuals.innerHTML =
    '<div class="crumbs"><span class="crumb here">Manuals</span>' +
    '<span class="put"><span class="flightbar" id="mflight"></span>' +
    '<button class="crumb act" id="mpick" type="button">&#8593; Upload</button>' +
    '<input class="pickfile" id="mpickfile" type="file" accept="application/pdf" multiple>' +
    '</span></div><div class="pbody sheet" id="mbody">' + rows + '</div>';
  bindManuals();
  // Only while something is unfinished, and only on this screen. A poll for a screen nobody is
  // on is a warm Pi - the same rule showLive and showHighlights state when they stop themselves.
  clearTimeout(manualPoll);
  if (found.reading && at === '/manuals') manualPoll = setTimeout(showManuals, 4000);
}

async function showManuals() {
  if (!vManuals) return;
  try {
    paintManuals(await grab('/api/manuals'));
  } catch (e) {
    vManuals.innerHTML = '<div class="empty">could not read the card</div>';
  }
}

function sayManual(text, bad) {
  const flight = document.getElementById('mflight');
  if (!flight) return;
  flight.textContent = text || '';
  flight.classList.toggle('bad', !!bad);
}

// One request per file, in order, stopping at the first refusal. A manual that did not arrive
// must never be left looking like one that did.
async function sendManuals(list) {
  const rest = [...list].filter((f) => /\.pdf$/i.test(f.name));
  if (!rest.length) return sayManual('a manual has to be a PDF', true);
  for (let n = 0; n < rest.length; n++) {
    const one = rest[n];
    sayManual((rest.length > 1 ? (n + 1) + '/' + rest.length + ' ' : '') + one.name);
    try {
      // octet-stream for the reason the project uploader gives: it is not a CORS-simple type,
      // so every upload sits behind a preflight this server never answers.
      const r = await fetch('/api/manuals/upload?name=' + encodeURIComponent(one.name),
        { method: 'POST', body: one, headers: { 'Content-Type': 'application/octet-stream' } });
      if (!r.ok) throw new Error((await r.text()).slice(0, 60) || r.status);
      paintManuals(await r.json());
    } catch (e) {
      return sayManual(String(e.message || e).slice(0, 60), true);
    }
  }
  sayManual('reading it now - that takes about a minute');
}

function bindManuals() {
  const pick = document.getElementById('mpick');
  const file = document.getElementById('mpickfile');
  if (!pick || !file) return;
  pick.onclick = () => file.click();
  file.onchange = () => { if (file.files.length) sendManuals(file.files); file.value = ''; };
}

// Drag onto the whole screen, not onto the list. A drop landing anywhere this does not cover
// makes Chromium navigate to file:/// - fatal on a page whose premise is never navigating - so
// the handler goes on the view and dragover is prevented across all of it.
if (vManuals) {
  const lit = (on) => {
    const body = document.getElementById('mbody');
    if (body) body.classList.toggle('drop', on);
  };
  vManuals.addEventListener('dragover', (e) => { e.preventDefault(); lit(true); });
  vManuals.addEventListener('dragleave', (e) => {
    if (!vManuals.contains(e.relatedTarget)) lit(false);
  });
  vManuals.addEventListener('drop', (e) => {
    e.preventDefault();
    lit(false);
    if (e.dataTransfer.files.length) sendManuals(e.dataTransfer.files);
  });
}

async function showProjects() {
  try {
    const found = await shelf();
    vProjects.innerHTML = found.length
      ? found.map(projectRow).join('')
      : '<div class="empty">no projects yet</div>';
  } catch (e) {
    vProjects.innerHTML = '<div class="empty">could not read the card</div>';
  }
}

// ---------------------------------------------------------------- one project

// Where you are inside FILES, and every step of the way back. Up is one step, whatever that means
// from here: out of a file, or out of a folder. It stops at the section root - the way out of the
// project is the arrow at the top of the rail, and a second one saying the same thing in the same
// bar would be two answers to one question.
//
// The trail begins at "Files" rather than at the project, because the rail is already saying
// which project this is - and it says it by title, where a folder tree can only say it by folder
// name. One name for one thing.
function crumbs(name, path, leaf) {
  const walk = path ? path.split('/') : [];
  const up = leaf ? pRoute(name, 'files', path)
    : walk.length ? pRoute(name, 'files', walk.slice(0, -1).join('/'))
    : '';
  let out = up
    ? '<button class="crumb up" type="button" aria-label="Back" data-hash="' +
      esc(up) + '">&#8249;</button>'
    : '';
  out += '<button class="crumb" type="button" data-hash="' + esc(pRoute(name, 'files')) +
    '">Files</button>';
  let walked = '';
  for (const part of walk) {
    walked = walked ? walked + '/' + part : part;
    out += '<span class="sep">/</span><button class="crumb" type="button" data-hash="' +
      esc(pRoute(name, 'files', walked)) + '">' + esc(part) + '</button>';
  }
  if (leaf) out += '<span class="sep">/</span><span class="crumb here">' + esc(leaf) + '</span>';
  return out;
}

// What a file is, in one character. The four things a project folder actually holds, and a
// mark for everything else rather than a blank column that reads as a failure to load.
const MARKS = {
  '.md': '\u00b6', '.xlsx': '\u25a4', '.json': '{}', '.csv': '\u25a4', '.txt': '\u00b6',
  '.jpg': '\u25a3', '.jpeg': '\u25a3', '.png': '\u25a3', '.svg': '\u25a3', '.mp4': '\u25b6',
};

function fileRow(name, one) {
  const dir = one.kind === 'dir';
  const mark = dir ? '\u25b8' : (MARKS[one.suffix] || '\u00b7');
  const hash = dir ? pRoute(name, 'files', one.path) : fRoute(name, one.path);
  return '<button class="row" type="button" data-hash="' + esc(hash) + '">' +
    '<span class="mark' + (dir ? ' dir' : '') + '">' + mark + '</span>' +
    '<span class="rowtext"><span class="rowtitle">' + esc(one.name) + '</span></span>' +
    '<span class="rowwhen">' + (dir ? '' : heft(one.size)) + '</span>' +
    '</button>';
}

// Which folder the browser is pointed at. The controls beside the crumbs need it and the router
// does not tell them - it calls showBrowse and moves on - so this is where the two meet.
let here = { name: '', path: '' };
// Set below when the server sent the controls, so showBrowse can clear them without knowing
// whether there are any. Null on the panel, where there is nothing to reset.
let putReset = null;

// The listing, drawn. Split out of showBrowse because a folder is now repainted from two places:
// asking for it, and having just changed it. It is handed the folder it is drawing rather than
// reading `here`, so a listing that lands late cannot paint itself into whatever is on screen by
// then - the same care showSession takes with `openName`.
function redraw(at, found) {
  pbody.innerHTML = found.entries && found.entries.length
    ? found.entries.map((one) => fileRow(at.name, one)).join('')
    : '<div class="empty">this folder is empty</div>';
}

async function showBrowse(name, path) {
  const at = here = { name, path };
  if (putReset) putReset();   // a refusal is about the folder you were in, not the one you opened
  crumbTrail.innerHTML = crumbs(name, path, '');
  const url = '/api/project/' + encodeURIComponent(name) +
    '/files?path=' + encodeURIComponent(path);
  try {
    const found = await grab(url);
    if (here !== at) return;
    redraw(at, found);
  } catch (e) {
    pbody.innerHTML = '<div class="empty">could not read that folder</div>';
  }
}

// ---------------------------------------------------------------- putting something in

// Everything below exists only when the server sent the controls, which it does for everything
// except the panel (dashboard.html). One guard rather than one per listener: on the kiosk there
// is nothing here to bind to, and a page that spent its startup looking for buttons that were
// never rendered would be doing it on the slowest browser of the two.
const putBar = document.getElementById('put');
if (putBar) {
  const flight = document.getElementById('flight');
  const newdir = document.getElementById('newdir');
  const mkdirBtn = document.getElementById('mkdir');
  const pick = document.getElementById('pick');
  const pickfile = document.getElementById('pickfile');

  // Built from the folder a run started in, never from `here`. A batch of six photos takes a
  // moment, and clicking into another folder while it runs must not send the seventh somewhere
  // else - or paint folder A's listing over folder B.
  const pUrl = (at, leaf) => '/api/project/' + encodeURIComponent(at.name) + '/' + leaf +
    '?path=' + encodeURIComponent(at.path);

  // grab()'s sibling, for the two routes that write. It reads the body on a failure where grab
  // throws the status, because these are the only requests on this page whose refusal has
  // something to say - "that name has nothing in it a file can be called" is worth showing, and
  // "400" is not.
  async function put(url, body, headers) {
    const r = await fetch(url, { method: 'POST', body, headers });
    if (!r.ok) throw new Error((await r.text().catch(() => '')) || r.status);
    return r.json();
  }

  const saying = (text, bad) => {
    flight.textContent = text;
    flight.classList.toggle('bad', !!bad);
  };

  // The files still to go, drawn as rows that are already there and dimmed. Without this a photo
  // crossing the LAN looks like a folder doing nothing, and the second click is how you end up
  // sending it twice.
  function waiting(rest) {
    const gone = pbody.querySelector('.empty');
    if (gone) gone.remove();
    pbody.insertAdjacentHTML('beforeend', rest.map((file) =>
      '<button class="row flight" type="button" disabled>' +
      '<span class="mark">\u00b7</span>' +
      '<span class="rowtext"><span class="rowtitle">' + esc(file.name) + '</span></span>' +
      '<span class="rowwhen">' + heft(file.size) + '</span></button>').join(''));
  }

  // One request per file rather than one for all of them, which is what makes the count below
  // honest: each response is a folder that really does contain what it says. It also keeps any
  // single failure to the file it belongs to instead of losing a batch to one bad name.
  async function send(list) {
    const at = here;
    const rest = [...list];
    const all = rest.length;
    while (rest.length) {
      const file = rest[0];
      saying(all > 1 ? (all - rest.length + 1) + '/' + all + ' ' + file.name : file.name);
      waiting(rest);
      try {
        // application/octet-stream and not whatever the File claims to be. A .txt would go as
        // text/plain, which is one of the three types a cross-origin page may post without a
        // preflight - and a preflight this server never answers is the nearest thing to a CSRF
        // token a page with no login has. Naming the type keeps every upload behind one.
        const found = await put(pUrl(at, 'upload') + '&name=' + encodeURIComponent(file.name),
          file, { 'Content-Type': 'application/octet-stream' });
        if (here !== at) return;   // you moved on; that folder is not what is on screen now
        redraw(at, found);
      } catch (e) {
        // Stop, and say which one. A file that did not arrive must never be left looking like
        // one that did, and the dimmed rows below it are still on screen saying what is missing.
        if (here === at) saying(String(e.message || e).slice(0, 60), true);
        return;
      }
      rest.shift();
    }
    saying('');
  }

  // Naming a folder happens in the bar itself. A prompt() would freeze the page and look like
  // nothing else here; a field that takes the button's place is the smallest thing that works.
  function naming(on) {
    newdir.hidden = !on;
    mkdirBtn.hidden = on;
    if (on) { newdir.value = ''; newdir.focus(); }
  }

  async function made() {
    const at = here;
    const name = newdir.value.trim();
    naming(false);
    if (!name) return;
    try {
      const found = await put(pUrl(at, 'folder'), 'name=' + encodeURIComponent(name),
        { 'Content-Type': 'application/x-www-form-urlencoded' });
      if (here !== at) return;
      redraw(at, found);
      saying('');
    } catch (e) {
      if (here === at) saying(String(e.message || e).slice(0, 60), true);
    }
  }

  // What showBrowse calls on the way into a folder: a refusal and a half-typed name both belong
  // to the folder you were looking at when you caused them.
  putReset = () => { saying(''); naming(false); };

  mkdirBtn.addEventListener('click', () => naming(true));
  newdir.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') made();
    if (e.key === 'Escape') naming(false);
  });
  newdir.addEventListener('blur', () => naming(false));

  pick.addEventListener('click', () => pickfile.click());
  pickfile.addEventListener('change', () => {
    if (pickfile.files.length) send(pickfile.files);
    pickfile.value = '';   // so picking the same file twice in a row fires change twice
  });

  // And the gesture everyone already has for this. dragover must preventDefault or the browser
  // takes the drop itself and navigates away to the file - which on this page, whose whole
  // premise is that it never navigates, is the expensive kind of bug.
  //
  // Bound to the whole screen and not to the file list, for that same reason. The list stops
  // short of the crumb bar and of the empty space under a short folder, and a drop landing on
  // either of those is Chromium leaving for file:///. Catching the screen means every drop
  // anywhere in the browser is ours. The outline still goes on the list, because that is the
  // thing you are putting something into.
  const screen = document.getElementById('v-project');
  screen.addEventListener('dragover', (e) => {
    e.preventDefault();
    pbody.classList.add('drop');
  });
  screen.addEventListener('dragleave', (e) => {
    if (!screen.contains(e.relatedTarget)) pbody.classList.remove('drop');
  });
  screen.addEventListener('drop', (e) => {
    e.preventDefault();
    pbody.classList.remove('drop');
    if (e.dataTransfer.files.length) send(e.dataTransfer.files);
  });
}

// ---------------------------------------------------------------- one file

// The server decided what this file is worth showing as; this only lays it out. Markdown
// arrives as HTML because shelf.py is where the project folder is known - which is what lets
// ![cap](Photos/x.jpg) come back already pointed at a URL this page can fetch.
function asDoc(found) {
  if (found.kind === 'markdown') return found.html;
  if (found.kind === 'sheet') return workbook(found.tabs);
  if (found.kind === 'image') return '<img class="lone" alt="" src="' + esc(found.url) + '">';
  if (found.kind === 'video') {
    return '<video class="lone" controls playsinline preload="metadata" src="' +
      esc(found.url) + '"></video>';
  }
  if (found.kind === 'text') return '<pre class="raw">' + esc(found.text) + '</pre>';
  return '<div class="empty">' + (heft(found.size) || 'empty') + ' \u2014 nothing to show it with</div>';
}

// A workbook as what it is: pairs, under the tab they were written on.
function workbook(tabs) {
  if (!tabs.length) return '<div class="empty">nothing written down yet</div>';
  return tabs.map((tab) => '<h2>' + esc(tab.name) + '</h2>' + (tab.rows.length
    ? tab.rows.map((row) =>
        '<div class="pair"><span class="pkey">' + esc(row.key) + '</span>' +
        '<span class="pval">' + esc(row.value) + '</span>' +
        (row.note ? '<span class="pnote">' + esc(row.note) + '</span>' : '') +
        '</div>').join('')
    : '<div class="empty">empty</div>')).join('');
}

// One file, drawn where the listing was. Handed the folder it belongs to as well as the file, so
// a listing that lands late cannot paint itself over a document you have already opened - the
// same care showBrowse takes, through the same `here`.
async function showFile(name, path) {
  const at = here = { name, path };
  const cut = path.lastIndexOf('/');
  crumbTrail.innerHTML = crumbs(name, cut < 0 ? '' : path.slice(0, cut), path.slice(cut + 1));
  pbody.innerHTML = '';
  try {
    const found = await grab(fileUrl(name, path));
    if (here !== at) return;
    pbody.innerHTML = asDoc(found);
  } catch (e) {
    if (here === at) pbody.innerHTML = '<div class="empty">could not read that file</div>';
  }
}

// ---------------------------------------------------------------- the five sections

const fileUrl = (name, path) => '/api/project/' + encodeURIComponent(name) +
  '/file?path=' + encodeURIComponent(path);

// What the project's own root holds. The rail needs it to find the workbooks - "Spreadsheets" is
// whatever .xlsx is in there, not a filename this page has memorised - and it is the same request
// FILES makes for its first folder, so opening a project costs one listing and not five.
let rootHeld = null, rootOf = '', rootAt = 0;
async function projectRoot(name) {
  if (rootHeld && rootOf === name && Date.now() - rootAt < FRESH) return rootHeld;
  rootHeld = await grab('/api/project/' + encodeURIComponent(name) + '/files?path=');
  rootOf = name;
  rootAt = Date.now();
  return rootHeld;
}

// README.md and Log.md, which every project has because store.py makes both when it makes the
// project. Rendered by the same asDoc() a file opened out of FILES goes through, because it is
// the same file - all this section does is know its name so you do not have to.
async function showDoc(name, leaf) {
  const at = here = { name, path: leaf };
  pbody.innerHTML = '';
  try {
    const found = await grab(fileUrl(name, leaf));
    if (here !== at) return;
    pbody.innerHTML = asDoc(found);
  } catch (e) {
    if (here === at) pbody.innerHTML = '<div class="empty">nothing written down yet</div>';
  }
}

// Every workbook in the project's root, under its own name when there is more than one. In
// practice there is exactly one and it is called Project Data.xlsx - but the name is the model's
// to choose and a second book somebody dropped in is still this project's numbers.
async function showSheets(name) {
  const at = here = { name, path: '' };
  pbody.innerHTML = '';
  try {
    const root = await projectRoot(name);
    const books = root.entries.filter((one) => one.suffix === '.xlsx');
    if (!books.length) {
      if (here === at) pbody.innerHTML = '<div class="empty">no spreadsheets in this project</div>';
      return;
    }
    const found = await Promise.all(books.map((one) => grab(fileUrl(name, one.path))));
    if (here !== at) return;
    pbody.innerHTML = found.map((book, i) =>
      (books.length > 1 ? '<h1>' + esc(books[i].name) + '</h1>' : '') +
      workbook(book.tabs || [])).join('');
  } catch (e) {
    if (here === at) pbody.innerHTML = '<div class="empty">could not read that</div>';
  }
}

// The rail's five destinations, and what #pbody is wearing while it shows each. The class decides
// the padding and the scrolling, which is why it is set here and not inside the renderers: a
// section that failed to load still has to be the right shape to say so in.
const SHOW = {
  overview: { dress: 'doc', open: (name) => showDoc(name, 'README.md') },
  log:      { dress: 'doc', open: (name) => showDoc(name, 'Log.md') },
  sheets:   { dress: 'doc', open: showSheets },
  photos:   { dress: 'shots', open: showPhotos },
  youtube:  { dress: 'shots', open: showYouTube },
  files:    { dress: 'files', open: (name, path) => showBrowse(name, path) },
  // One file out of FILES. Not a sixth thing in the rail - it stays lit on FILES, because that is
  // where you were and where the way back goes - but the pane is a document rather than a list.
  file:     { dress: 'doc', open: showFile },
};

// One project, on whichever of its screens you asked for. `shown` is a key of SHOW and has been
// settled by the router already, so there is no fallback here to disagree with the one up there -
// and it is not always the section the rail is lit on, because a file lights FILES.
function showProject(name, shown, path) {
  if (putReset) putReset();   // a refusal belongs to the folder that caused it, not to the next screen
  pbody.className = 'pbody ' + SHOW[shown].dress;
  // The title and not the folder name: the folder is what a person renamed in Finder, and the
  // title is what the project calls itself. It is already on the card from the shelf listing, so
  // this is a paint and not a request - and it opens as the folder name for the one moment a
  // deep-linked address arrives before that listing does.
  railName.textContent = name;
  shelf().then((found) => {
    const one = found.find((project) => project.name === name);
    if (one && here.name === name) railName.textContent = one.title;
  }).catch(() => {});
  SHOW[shown].open(name, path);
}

// ---------------------------------------------------------------- the picture stream

// The pictures on show, whichever screen is showing them. One array and not two, because only one
// of the two galleries is ever on screen and light() has to be able to open whatever was clicked.
let shots = [];

// A run of pictures as a grid. Both galleries draw the same thing - the whole card on MEDIA, one
// project on its PHOTOS - and shelf.pictures deliberately answers in library.Item's shape so that
// stays true.
function gallery(items, empty) {
  shots = items;
  return items.length
    ? '<div class="grid">' + items.map((one, i) =>
        '<button class="cell" type="button" data-shot="' + i + '">' +
        '<img loading="lazy" draggable="false" src="' + esc(one.url) +
        '" alt="' + esc(one.title) + '">' +
        '</button>').join('') + '</div>'
    : '<div class="empty">' + empty + '</div>';
}

async function showMedia() {
  try {
    vMedia.innerHTML = gallery((await grab('/api/media')).items, 'no pictures yet');
  } catch (e) {
    vMedia.innerHTML = '<div class="empty">could not read the card</div>';
  }
}

// Every picture in one project, and not only the ones the sweep filed into Photos/ - a folder of
// drawings you made is as much this project as a photograph Cyclops took. See shelf.pictures.
async function showPhotos(name) {
  const at = here = { name, path: '' };
  pbody.innerHTML = '';
  try {
    const found = await grab('/api/project/' + encodeURIComponent(name) + '/pictures');
    if (here !== at) return;
    pbody.innerHTML = gallery(found.items, 'no pictures in this project yet');
  } catch (e) {
    if (here === at) pbody.innerHTML = '<div class="empty">could not read the card</div>';
  }
}

// One picture on a black screen. A press anywhere on it puts it away, which is the whole of
// the control surface - nothing to aim at. When it was taken sits in the corner, because the
// grid no longer says: it is pointer-events: none, so it is a caption and not a target.
// The videos filed onto this project, which are references and not pictures: each one is a
// title card, what it was called, and the second somebody was watching it at.
//
// Its own array and its own attribute, not `shots` and `data-shot`. gallery() overwrites `shots`
// globally so that one lightbox listener can serve both galleries, which is right while both are
// galleries of pictures - but a press here opens YouTube rather than a lightbox, and sharing the
// array would have the wrong one of the two answering.
let tubes = [];

async function showYouTube(name) {
  const at = here = { name, path: '' };
  pbody.innerHTML = '';
  try {
    const found = await grab('/api/project/' + encodeURIComponent(name) + '/youtube');
    if (here !== at) return;
    tubes = found.items;
    pbody.innerHTML = tubes.length
      ? '<div class="grid">' + tubes.map((one, i) =>
          '<button class="cell tube" type="button" data-tube="' + i + '">' +
          '<img loading="lazy" draggable="false" src="' + esc(one.url) +
          '" alt="' + esc(one.title) + '">' +
          '<span class="len">' + esc(one.clock) + '</span>' +
          '<span class="tubename">' + esc(one.title) + '</span>' +
          '</button>').join('') + '</div>'
      : '<div class="empty">no videos kept for this project yet</div>';
  } catch (e) {
    if (here === at) pbody.innerHTML = '<div class="empty">could not read the card</div>';
  }
}

// A new tab, and not the lightbox the pictures use: the thing worth having back is the video
// itself, playing where it lives, at the second it was found at. The panel has no way to open
// one of these - on the panel you ask for it out loud and recall plays it - so this is the
// laptop's and the phone's half of "show me that one again".
document.addEventListener('click', (e) => {
  const el = e.target.closest('[data-tube]');
  if (!el) return;
  const one = tubes[parseInt(el.dataset.tube, 10)];
  if (one && one.watch) window.open(one.watch, '_blank', 'noopener');
});

const lightbox = document.getElementById('lightbox');

function light(i) {
  const one = shots[i];
  if (!one) return;
  // What it is of, when something has actually looked. Project pictures carry a caption from the
  // index service and session ones do not, so this is a description on one screen and a date on
  // the other - and never a filename repeated under a picture of itself, which is what `title`
  // falls back to and why it is not the field read here.
  const said = one.caption ? esc(one.caption) + ' · ' : '';
  lightbox.innerHTML =
    '<img draggable="false" src="' + esc(one.url) + '" alt="' + esc(one.title) + '">' +
    '<div class="lightwhen">' + said + esc(day(one.when) + ' ' + time(one.when)) + '</div>';
  document.body.classList.add('lit');
}
const douse = () => { document.body.classList.remove('lit'); lightbox.textContent = ''; };

// One listener for both galleries, the way [data-hash] is one listener for every row and crumb -
// only one of them is ever on screen, and they draw the same markup out of the same array.
document.addEventListener('click', (e) => {
  const el = e.target.closest('[data-shot]');
  if (el) light(parseInt(el.dataset.shot, 10));
});
lightbox.addEventListener('click', douse);
// Escape is not a control on the picture - it is nowhere on the screen - so it can stay.
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && document.body.classList.contains('lit')) douse();
});

// ---------------------------------------------------------------- dragging a list about

// Insurance, not the main event. A touchscreen that the stack reports as a touchscreen pans by
// itself and this stays out of the way. The panel's ft5x06 is also exposed as `mouse0` by the
// kernel's legacy mousedev emulation, and anywhere that wins, a finger arrives as a *mouse*
// drag - which selects, or drags, or does nothing at all, but never scrolls. So for pointers
// that are not touch, the drag is turned into a scroll here.
//
// `when` is for an element that only scrolls some of the time: panel.js hands it the stage,
// which scrolls a page of a manual and must leave every other picture - and the seeker under a
// video, which has a drag of its own - exactly as it was.
function dragScroll(el, when = () => true) {
  const SLOP = 6;          // below this it was a tap with a shaky hand, not a drag
  let id = null, y0 = 0, top0 = 0, far = 0, last = 0, lastAt = 0, vel = 0, glide = 0;
  const halt = () => { cancelAnimationFrame(glide); glide = 0; };

  el.addEventListener('pointerdown', (e) => {
    halt(); id = null; far = 0;
    if (e.pointerType === 'touch' || e.button || !when()) return;
    id = e.pointerId; y0 = last = e.clientY; lastAt = e.timeStamp;
    top0 = el.scrollTop; vel = 0;
  });
  el.addEventListener('pointermove', (e) => {
    if (e.pointerId !== id) return;
    const dy = e.clientY - y0;
    far = Math.max(far, Math.abs(dy));
    if (far <= SLOP) return;
    el.setPointerCapture(id);   // the list keeps the drag even when the finger leaves a row
    el.scrollTop = top0 - dy;
    const gap = e.timeStamp - lastAt;
    if (gap > 0) vel = (e.clientY - last) / gap;   // px per ms, for the throw
    last = e.clientY; lastAt = e.timeStamp;
    e.preventDefault();
  });
  const release = () => {
    if (id === null) return;
    id = null;
    if (Math.abs(vel) < 0.06) return;
    // A list that stops dead under your finger feels broken even when it is not. Decay per
    // frame, and give up once the movement is under a pixel or so.
    let v = vel;
    const roll = () => {
      v *= 0.94;
      el.scrollTop -= v * 16;
      if (Math.abs(v) > 0.02) glide = requestAnimationFrame(roll);
    };
    glide = requestAnimationFrame(roll);
  };
  el.addEventListener('pointerup', release);
  el.addEventListener('pointercancel', release);
  // A drag that travelled is not a tap, whatever it happens to finish on top of.
  el.addEventListener('click', (e) => {
    if (far > SLOP) { e.stopPropagation(); e.preventDefault(); }
  }, true);
}
// The panel only. This is the mousedev workaround above, and on anything with a real pointer it
// is a bug: it swallows drag-to-select, so a laptop cannot highlight a paragraph of a README.
if (document.body.classList.contains('kiosk')) {
  dragScroll(vSessions);
  dragScroll(vMedia);
  dragScroll(vProjects);
  dragScroll(pbody);
}

// ---------------------------------------------------------------- the router

vSessions.addEventListener('click', (e) => {
  const el = e.target.closest('[data-open]');
  if (el) location.hash = '#/s/' + encodeURIComponent(el.dataset.open);
});
for (const tab of tabs) {
  tab.addEventListener('click', () => { location.hash = '#' + tab.dataset.go; });
}
if (menu) menu.addEventListener('change', () => { location.hash = '#' + menu.value; });
// The rail. data-sec and not data-hash like everything else on this page, because only half of a
// section's address is written in the markup - the other half is whichever project you are in,
// which `here` is holding.
for (const el of rails) {
  el.addEventListener('click', () => {
    if (here.name) location.hash = pRoute(here.name, el.dataset.sec);
  });
}
// Every project row, every file row and every crumb says where it goes, so one listener moves
// all of them. The markdown's own links need none: shelf.py writes them as hashes already, and
// a hash is the one href that changes this page without navigating away from it.
document.addEventListener('click', (e) => {
  const el = e.target.closest('[data-hash]');
  if (el) location.hash = el.dataset.hash;
});

let at = '';
function route() {
  const path = (location.hash || '#/').slice(1) || '/';
  if (path === at) return;
  if (at.startsWith('/s/')) letGo();
  if (at === '/live') stopLive();   // a poll for a screen nobody is on is a warm Pi
  if (at === '/highlights') stopReel();   // and two decoders for a screen nobody is on is a warm Pi
  if (at === '/manuals') clearTimeout(manualPoll);   // and so is polling one you have left
  if (document.body.classList.contains('lit')) douse();
  at = path;

  let view = 'view-status', tab = '/', sec = '';
  if (path === '/live') { view = 'view-live'; tab = '/live'; showLive(); }
  else if (path === '/sessions') { view = 'view-sessions'; tab = '/sessions'; showSessions(); }
  else if (path === '/media') { view = 'view-media'; tab = '/media'; showMedia(); }
  else if (path === '/highlights') { view = 'view-highlights'; tab = '/highlights'; showHighlights(); }
  else if (path.startsWith('/s/')) {
    view = 'view-session'; tab = '/sessions';
    showSession(decodeURIComponent(path.slice(3)));
  }
  else if (path === '/projects') { view = 'view-projects'; tab = '/projects'; showProjects(); }
  else if (path === '/manuals') { view = 'view-manuals'; tab = '/manuals'; showManuals(); }
  else if (path.startsWith('/p/')) {
    view = 'view-project'; tab = '/projects';
    const [name, asked, where] = hashParts(path.slice(3), true);
    // An address naming no section, or one this page has never heard of, opens the project rather
    // than nothing - which is also what every #/p/<name>/ bookmark from before the rail existed
    // now does.
    sec = SECTIONS.includes(asked) ? asked : 'overview';
    showProject(name, sec, where);
  }
  // One file out of FILES, and the only route the server writes: shelf.file_route puts this hash
  // in every relative link of a rendered README. So the two files the rail already has a section
  // for are sent to it - the README's own footer links Log.md, and landing on LOG is what that
  // link means - and everything else opens where its folder is, with FILES still lit.
  else if (path.startsWith('/f/')) {
    const [name, , where] = hashParts(path.slice(3), false);
    const asSection = { 'README.md': 'overview', 'Log.md': 'log' }[where];
    if (asSection) { location.hash = pRoute(name, asSection); return; }
    view = 'view-project'; tab = '/projects'; sec = 'files';
    showProject(name, 'file', where);
  }
  document.body.classList.remove(...VIEWS);
  document.body.classList.add(view);
  // Which of the five the rail is lit on, set the same way and for the same reason the header's
  // tabs are: the markup says what the sections are, and the router says which one you are on.
  document.body.classList.remove(...SECTIONS.map((one) => 'sec-' + one));
  if (sec) document.body.classList.add('sec-' + sec);
  for (const el of rails) el.setAttribute('aria-current', el.dataset.sec === sec ? 'true' : 'false');
  for (const el of tabs) el.setAttribute('aria-current', el.dataset.go === tab ? 'true' : 'false');
  // The parent route, so a session or a file leaves the menu reading the list it came from.
  if (menu) menu.value = tab;
  // Views scroll on their own, and a new one always starts at the top. The project screen keeps
  // its rail and crumb bar still and scrolls the pane under them, so that one is named here
  // rather than caught by the .view sweep.
  for (const el of document.querySelectorAll('.view')) el.scrollTop = 0;
  pbody.scrollTop = 0;
}
window.addEventListener('hashchange', route);
route();

// Opening the page while he is talking should be the same gesture as looking up: you get the
// conversation. Only on a fresh open with nothing asked for - a hash you typed, followed or were
// left on is never overruled - and only away from the panel, whose browser is pointed at a screen
// by the kiosk and boots with a hash already on it.
//
// One request, and it is the same one the screen would make anyway a moment later.
if (!document.body.classList.contains('kiosk') && !location.hash) {
  grab('/api/live')
    .then((s) => { if (s.name && at === '/') location.hash = '#/live'; })
    .catch(() => {});
}

// A drawing arriving outranks whatever you were reading - it is Cyclops answering out loud, and
// the panel is where the answer goes. The view class is left alone throughout, so when the
// kiosk clears the drawing the screen you were on is simply uncovered again.
window.__drawing = (on) => {
  document.body.classList.toggle('drawing', on);
  // body.drawing .view hides a <video> with display:none, which does not stop it decoding - so
  // both players are stopped by hand, and the reel is picked back up where it was afterwards.
  if (on) { video.pause(); for (const el of reelPair) el.pause(); }
  else if (at === '/highlights') reelShow(reelAt);
  // A stream into a covered <img> is a stream still arriving, and the picture is the one thing
  // on this screen that costs the Pi something to send. His voice is deliberately not cut: a
  // drawing is what he is talking about. See stream.js.
  if (window.__cam) window.__cam(!on);
};
