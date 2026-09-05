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
               'view-projects', 'view-browse', 'view-file'];
const FRESH = 20000;   // a listing is worth re-using for as long as nothing new can have ended

const vSessions = document.getElementById('v-sessions');
const vMedia = document.getElementById('v-media');
const vProjects = document.getElementById('v-projects');
const crumbTrail = document.getElementById('crumbtrail');
const files = document.getElementById('files');
const fileCrumbs = document.getElementById('fcrumbs');
const doc = document.getElementById('doc');
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
// The panel navigates with four tabs and every other client with one menu (see dashboard.html).
// Only one of these is ever on the page, and the code below simply drives whichever it found.
const tabs = [...document.querySelectorAll('.tab')];
// Not `pick`: that id belongs to the Upload button down in the crumb bar, and an id claimed
// twice is silently won by whichever element is earlier in the document.
const menu = document.getElementById('menu');
const rig = document.getElementById('rig');
const play = document.getElementById('play');
const vnow = document.getElementById('vnow');
const vall = document.getElementById('vall');
const vtitle = document.getElementById('vtitle');
const scrub = document.getElementById('scrub');
const scrubTrack = scrub.querySelector('.scrub-track');
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

// Where the picture actually is. object-fit: contain letterboxes inside the element and leaves
// nothing in the DOM to hang anything off, so the bars have to be measured. Everything that
// sits over or beside the picture is placed from this - which is also what makes it correct on
// a phone held upright, where the same arithmetic puts the bars top and bottom instead.
function fit() {
  if (!document.body.classList.contains('solo')) return;
  const box = video.getBoundingClientRect();
  if (!box.width || !box.height) return;
  const ratio = (video.videoWidth || 5) / (video.videoHeight || 3);  // 5:3 until it says
  const wide = box.width / box.height > ratio;
  const w = wide ? box.height * ratio : box.width;
  const h = wide ? box.height : box.width / ratio;
  const padX = (box.width - w) / 2, padY = (box.height - h) / 2;
  for (const el of [vtitle, scrub]) { el.style.left = padX + 'px'; el.style.right = padX + 'px'; }
  vtitle.style.top = padY + 'px';
  scrub.style.bottom = padY + 'px';
  // Centred in the gutter to the left of the picture, and never off the edge of a screen too
  // narrow to have one - there it sits over the picture, as it does in every player.
  const gutter = box.left + padX;
  rig.style.left = Math.max(6, (gutter - rig.offsetWidth) / 2) + 'px';
  rig.style.top = Math.round(box.top + box.height / 2 - rig.offsetHeight / 2) + 'px';
}

function solo(on) {
  document.body.classList.toggle('solo', on);
  video.controls = !on;   // the browser's own bar is desktop furniture at 800x480
  if (on) requestAnimationFrame(fit);
}

const paint = () => {
  vnow.textContent = clock(video.currentTime);
  scrubFill.style.width = (video.duration ? 100 * video.currentTime / video.duration : 0) + '%';
};
video.addEventListener('timeupdate', paint);
video.addEventListener('loadedmetadata', () => {
  vall.textContent = isFinite(video.duration) ? clock(video.duration) : '';
  fit();
  paint();
});
const mark = () => { if (video.paused) play.removeAttribute('data-on'); else play.dataset.on = '1'; };
video.addEventListener('play', mark);
video.addEventListener('pause', mark);
const toggle = () => { if (video.paused) video.play().catch(() => {}); else video.pause(); };
play.addEventListener('click', toggle);
// The picture is the biggest target on the screen; on a panel it should do the commonest thing.
video.addEventListener('click', () => { if (document.body.classList.contains('solo')) toggle(); });
window.addEventListener('resize', fit);

const seek = (e) => {
  const box = scrubTrack.getBoundingClientRect();
  if (!video.duration || !box.width) return;
  video.currentTime = Math.min(1, Math.max(0, (e.clientX - box.left) / box.width)) * video.duration;
  paint();
};
scrub.addEventListener('pointerdown', (e) => { scrub.setPointerCapture(e.pointerId); seek(e); });
scrub.addEventListener('pointermove', (e) => {
  if (scrub.hasPointerCapture(e.pointerId)) seek(e);
});

document.getElementById('back').addEventListener('click', () => { location.hash = '#/sessions'; });

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
    vtitle.textContent = one.title;
    if (one.video) {
      video.src = '/media/' + encodeURIComponent(name) + '/video.mp4';
      video.hidden = false;
      // You tapped a session to watch it, so watch it. The click that got you here is the
      // gesture the autoplay policy wants; if a browser refuses anyway it stays paused, and
      // the play button is now 76 u of thumb rather than something to aim at.
      video.play().catch(() => {});
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
let liveOver = false;  // ...and whether we have already said so; see the `!s.name` branch below
let liveGen = 0;       // bumped on the way out, so a timer from a screen you left dies quietly

function stopLive() {
  liveGen++;
  liveName = null;
  liveHeld = 0;
  liveOver = false;
  if (liveTalk) liveTalk.innerHTML = '';
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

function liveHead(title, on) {
  if (liveTitle) liveTitle.textContent = title;
  if (!liveBar) return;
  // The header's own bottom rule, lit and travelling. Nothing to write and nothing to read: it
  // is either moving or it is not there. See .livebar in lan.css.
  if (on) liveBar.dataset.on = '1'; else liveBar.removeAttribute('data-on');
}

async function liveTick(mine) {
  if (mine !== liveGen) return;
  try {
    const s = await grab('/api/live?name=' + encodeURIComponent(liveName || '') +
                         '&since=' + liveHeld);
    if (mine !== liveGen) return;   // you left while this was in flight
    if (!s.name) {
      // He stopped talking. The folder's lock goes the moment the session closes and well before
      // it is renamed and summarised, so this arrives seconds after the last word - and clearing
      // the screen here would wipe the conversation you are still reading. Freeze it instead.
      //
      // Written once and not on every poll after it. `liveName` is deliberately kept: it is what
      // says a conversation ended rather than never started, and the next session will differ
      // from it and reset the screen through the ordinary branch below.
      if (!liveOver) {
        liveOver = true;
        liveHead(liveName ? 'that session has ended' : 'nothing running', false);
      }
    } else {
      liveOver = false;
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
  stopLive();
  liveHead('nothing running', false);
  liveTick(liveGen);
}

// ---------------------------------------------------------------- the projects

// Where a project's folder and one file inside it live in the hash. Both halves are encoded
// whole, slashes included - shelf.file_route builds the same string server-side - so a nested
// Photos/x.jpg stays one segment and the router's split cannot land in the middle of it.
const pRoute = (name, path) => '#/p/' + encodeURIComponent(name) + '/' + encodeURIComponent(path || '');
const fRoute = (name, path) => '#/f/' + encodeURIComponent(name) + '/' + encodeURIComponent(path || '');
const twoParts = (rest) => {
  const cut = rest.indexOf('/');
  return cut < 0 ? [decodeURIComponent(rest), '']
    : [decodeURIComponent(rest.slice(0, cut)), decodeURIComponent(rest.slice(cut + 1))];
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
  return '<button class="row" type="button" data-hash="' + esc(pRoute(one.name, '')) + '">' +
    '<span class="shot">' + shot + '</span>' +
    '<span class="rowtext"><span class="rowtitle">' + esc(one.title) + '</span>' +
    '<span class="rowsum">' + esc(one.tagline) + '</span></span>' +
    '<span class="rowwhen">' + dated(one.updated) + sub + '</span>' +
    '</button>';
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

// Where you are, and every step of the way back. Up is one step, whatever that means from
// here: out of a file, out of a folder, or off the project entirely and back to the shelf.
function crumbs(name, path, leaf) {
  const parts = path ? path.split('/') : [];
  const up = leaf ? pRoute(name, path)
    : parts.length ? pRoute(name, parts.slice(0, -1).join('/'))
    : '#/projects';
  let out = '<button class="crumb up" type="button" aria-label="Back" data-hash="' +
    esc(up) + '">&#8249;</button>' +
    '<button class="crumb" type="button" data-hash="' + esc(pRoute(name, '')) + '">' +
    esc(name) + '</button>';
  let walked = '';
  for (const part of parts) {
    walked = walked ? walked + '/' + part : part;
    out += '<span class="sep">/</span><button class="crumb" type="button" data-hash="' +
      esc(pRoute(name, walked)) + '">' + esc(part) + '</button>';
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
  const hash = dir ? pRoute(name, one.path) : fRoute(name, one.path);
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
  files.innerHTML = found.entries && found.entries.length
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
    files.innerHTML = '<div class="empty">could not read that folder</div>';
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
    const gone = files.querySelector('.empty');
    if (gone) gone.remove();
    files.insertAdjacentHTML('beforeend', rest.map((file) =>
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
  const screen = document.getElementById('v-browse');
  screen.addEventListener('dragover', (e) => {
    e.preventDefault();
    files.classList.add('drop');
  });
  screen.addEventListener('dragleave', (e) => {
    if (!screen.contains(e.relatedTarget)) files.classList.remove('drop');
  });
  screen.addEventListener('drop', (e) => {
    e.preventDefault();
    files.classList.remove('drop');
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

async function showFile(name, path) {
  const cut = path.lastIndexOf('/');
  fileCrumbs.innerHTML = crumbs(name, cut < 0 ? '' : path.slice(0, cut), path.slice(cut + 1));
  doc.innerHTML = '';
  const url = '/api/project/' + encodeURIComponent(name) +
    '/file?path=' + encodeURIComponent(path);
  try {
    doc.innerHTML = asDoc(await grab(url));
  } catch (e) {
    doc.innerHTML = '<div class="empty">could not read that file</div>';
  }
}

// ---------------------------------------------------------------- the picture stream

let shots = [];
async function showMedia() {
  try {
    shots = (await grab('/api/media')).items;
    vMedia.innerHTML = shots.length
      ? '<div class="grid">' + shots.map((one, i) =>
          '<button class="cell" type="button" data-shot="' + i + '">' +
          '<img loading="lazy" draggable="false" src="' + esc(one.url) +
          '" alt="' + esc(one.title) + '">' +
          '<div class="cellcap">' + esc(day(one.when) + ' ' + time(one.when)) +
          '</div></button>').join('') + '</div>'
      : '<div class="empty">no pictures yet</div>';
  } catch (e) {
    vMedia.innerHTML = '<div class="empty">could not read the card</div>';
  }
}

// Flipping through, which is what the stream is for. The side columns are 96u wide - a thumb,
// not a glyph - because on the panel the gesture is the control.
const lightbox = document.getElementById('lightbox');
const litshot = document.getElementById('litshot');
const litcap = document.getElementById('litcap');
let lit = -1;

function light(i) {
  if (!shots.length) return;
  lit = (i + shots.length) % shots.length;
  const one = shots[lit];
  litshot.innerHTML =
    '<img draggable="false" src="' + esc(one.url) + '" alt="' + esc(one.title) + '">';
  litcap.textContent = one.title + ' · ' + day(one.when) + ' ' + time(one.when) +
    ' · ' + (lit + 1) + ' of ' + shots.length;
  document.body.classList.add('lit');
}
const douse = () => { document.body.classList.remove('lit'); litshot.textContent = ''; lit = -1; };

vMedia.addEventListener('click', (e) => {
  const el = e.target.closest('[data-shot]');
  if (el) light(parseInt(el.dataset.shot, 10));
});
document.getElementById('litprev').addEventListener('click', () => light(lit - 1));
document.getElementById('litnext').addEventListener('click', () => light(lit + 1));
document.getElementById('litclose').addEventListener('click', douse);
document.addEventListener('keydown', (e) => {
  if (!document.body.classList.contains('lit')) return;
  if (e.key === 'Escape') douse();
  else if (e.key === 'ArrowLeft') light(lit - 1);
  else if (e.key === 'ArrowRight') light(lit + 1);
});

// ---------------------------------------------------------------- dragging a list about

// Insurance, not the main event. A touchscreen that the stack reports as a touchscreen pans by
// itself and this stays out of the way. The panel's ft5x06 is also exposed as `mouse0` by the
// kernel's legacy mousedev emulation, and anywhere that wins, a finger arrives as a *mouse*
// drag - which selects, or drags, or does nothing at all, but never scrolls. So for pointers
// that are not touch, the drag is turned into a scroll here.
function dragScroll(el) {
  const SLOP = 6;          // below this it was a tap with a shaky hand, not a drag
  let id = null, y0 = 0, top0 = 0, far = 0, last = 0, lastAt = 0, vel = 0, glide = 0;
  const halt = () => { cancelAnimationFrame(glide); glide = 0; };

  el.addEventListener('pointerdown', (e) => {
    halt(); id = null; far = 0;
    if (e.pointerType === 'touch' || e.button) return;
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
  dragScroll(files);
  dragScroll(doc);
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
  if (document.body.classList.contains('lit')) douse();
  at = path;

  let view = 'view-status', tab = '/';
  if (path === '/live') { view = 'view-live'; tab = '/live'; showLive(); }
  else if (path === '/sessions') { view = 'view-sessions'; tab = '/sessions'; showSessions(); }
  else if (path === '/media') { view = 'view-media'; tab = '/media'; showMedia(); }
  else if (path.startsWith('/s/')) {
    view = 'view-session'; tab = '/sessions';
    showSession(decodeURIComponent(path.slice(3)));
  }
  else if (path === '/projects') { view = 'view-projects'; tab = '/projects'; showProjects(); }
  else if (path.startsWith('/p/')) {
    view = 'view-browse'; tab = '/projects';
    showBrowse(...twoParts(path.slice(3)));
  }
  else if (path.startsWith('/f/')) {
    view = 'view-file'; tab = '/projects';
    showFile(...twoParts(path.slice(3)));
  }
  document.body.classList.remove(...VIEWS);
  document.body.classList.add(view);
  for (const el of tabs) el.setAttribute('aria-current', el.dataset.go === tab ? 'true' : 'false');
  // The parent route, so a session or a file leaves the menu reading the list it came from.
  if (menu) menu.value = tab;
  // Views scroll on their own, and a new one always starts at the top. The browser and the
  // reader keep their crumb bar still and scroll the half under it, so those two are named
  // here rather than caught by the .view sweep.
  for (const el of document.querySelectorAll('.view')) el.scrollTop = 0;
  files.scrollTop = doc.scrollTop = 0;
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
  if (on) video.pause();
};
