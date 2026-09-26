/*
 * Does the panel actually paint? The half of the picture path that Python cannot see.
 *
 * This used to check JointJS: port-label selectors that rendered as nothing, and drawings whose
 * outboard labels were clipped off the paper with the SVG still perfectly valid. Diagrams are
 * images now and that whole layer is gone - but what replaced it is the *entire* panel path, and
 * it fails just as silently. A payload the page cannot decode leaves an empty stage, no error,
 * and a kiosk that uncovers onto nothing. Nobody sees that from a laptop.
 *
 * So this checks what is left. A picture reaches #stage as an <img> with real pixels in it; a
 * scratchpad the model wrote reaches it as an <iframe> with a document of its own, laid out by that
 * document's own stylesheet and letting a press through to the stage behind it; the page posts
 * back so the kiosk may uncover; every one of them puts itself away on a press anywhere; and what
 * the model wrote cannot run a script or reach the network. A page of a manual is the one
 * exception to the press: it fills the width, scrolls under a drag, and goes away on a tap.
 *
 * Not run by pytest, deliberately: pyproject says the suite must run anywhere in a second, and
 * this needs Chromium and a server. Run it when you touch the template, diagram.js or the CSS.
 *
 *   uv run cyclops-admin --port=8099 &
 *   node tests/render_check.mjs                     # or: BASE=http://127.0.0.1:80 node ...
 *
 * Restart the service after every edit under admin/ - the template, the CSS or the JS. DEBUG is
 * off, so Django caches the compiled template for the life of the process, and the ?v= digest on
 * the stylesheets and scripts is likewise read once at import (views._asset_version). An edited
 * file is simply not what you are testing - this check passed twice against a bug that was
 * sitting in the working tree the whole time.
 *
 * Screenshots land in /tmp/cyclops-render/ so a failure can be looked at rather than guessed at.
 */

import { execSync } from 'node:child_process';
import { mkdirSync, writeFileSync, unlinkSync, existsSync, statSync } from 'node:fs';
import { homedir, networkInterfaces } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

// There is no package.json here and nothing is installed locally - this repo is Python. So take
// playwright from wherever it is: a local node_modules if someone made one, otherwise the global
// install that `playwright install` puts on the PATH. ESM will not search NODE_PATH for either.
async function loadPlaywright() {
  try {
    return await import('playwright');
  } catch {
    const root = execSync('npm root -g', { encoding: 'utf8' }).trim();
    return await import(pathToFileURL(join(root, 'playwright', 'index.mjs')).href);
  }
}
const { chromium } = await loadPlaywright();

const BASE = process.env.BASE || 'http://127.0.0.1:8099';
const PENDING = join(homedir(), '.cache', 'cyclops', 'panel.json');
// The two notes the companion case below turns on: whether anybody can see the picture, and the
// note a press leaves for the kiosk. See cyclops/config.py.
const PICTURE_UP = join(homedir(), '.cache', 'cyclops', 'picture-up');
const CLOSE_FLAG = join(homedir(), '.cache', 'cyclops', 'browser-close');
const SHOTS = '/tmp/cyclops-render';

// A real JPEG, inline rather than read off the card so the check needs no fixture: a plain
// 200x120 green rectangle. Its only job is to have pixels, so that naturalWidth below can tell a
// picture that decoded from an <img> the browser silently gave up on.
const JPEG_B64 =
  '/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAoHBwgHBgoICAgLCgoLDhgQDg0NDh0VFhEYIx8lJCIfIiEmKzcvJik0KSEiMEEx' +
  'NDk7Pj4+JS5ESUM8SDc9Pjv/2wBDAQoLCw4NDhwQEBw7KCIoOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7' +
  'Ozs7Ozs7Ozs7Ozs7Ozv/wAARCAB4AMgDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAA' +
  'AgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6' +
  'Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXG' +
  'x8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREA' +
  'AgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5' +
  'OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPE' +
  'xcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDWooorzz5UKKKKACiiigAooooAKKKKACii' +
  'igAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACii' +
  'igAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACii' +
  'igAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACii' +
  'igAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACii' +
  'igAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACii' +
  'igAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAP/Z';

const CASES = [
  {
    name: 'drawing',
    kind: 'image',
    payload: { title: 'Relay driven from GPIO 17', image: 'data:image/jpeg;base64,' + JPEG_B64 },
    // A diagram is a picture like any other. It carried a `drawn: true` here until 2026-09-05,
    // which kept it a corner button instead - one gesture with two answers, and the corner is
    // the answer nobody finds on a companion screen across the room.
    wantPhotoClass: true,
  },
  {
    name: 'photo',
    kind: 'image',
    payload: { title: '14-32-40_you', image: 'data:image/jpeg;base64,' + JPEG_B64 },
    wantPhotoClass: true,
  },
  {
    name: 'scratchpad',
    kind: 'scratchpad',
    // Not a scrap of styling, which is the case that matters. The model has to be able to write
    // `<h1>25 Nm</h1>` and get a good screen, because every character of CSS it writes instead is
    // silence between the question and the answer - see SCRATCHPAD_HEAD in panel.js.
    payload: { scratchpad: '<h1>25 Nm</h1><ol><li>Loosen the pinch bolt</li>' +
                     '<li>Torque the crank</li></ol>' },
    // He put this up unasked, over his eye and over the way out of the session, so it goes away
    // on a press anywhere rather than at a corner square somebody has to find first.
    wantPhotoClass: true,
    wantTall: { sel: 'h1', px: 48 },
  },
  {
    name: 'sketch',
    kind: 'scratchpad',
    // His own bench sketch from 2026-09-05, verbatim: every shape stroke="black". It was
    // invisible when a drawing was drawn straight onto the #050f0a screen. It is correct now
    // because a drawing gets white paper, which is the whole of the fix - so what this pins is
    // the paper, not the ink.
    payload: { scratchpad:
      '<h2>Bench</h2><svg viewBox="0 0 200 120">' +
      '<rect x="10" y="80" width="180" height="25" fill="none" stroke="black"/>' +
      '<circle cx="57.5" cy="15" r="7" fill="none" stroke="black"/>' +
      '<rect x="125" y="35" width="45" height="45" fill="none" stroke="#c33"/></svg>' },
    wantPhotoClass: true,
    // Edge to edge and nothing framing it. Three separate things each took a slice off this
    // before it was measured rather than looked at: the bezel, the 17 px gutter that exists to
    // sit inside the bezel, and an inset white sheet of my own - four frames around one diagram
    // on a panel 800 px wide, which is what Marco saw and called ugly.
    wantFills: true,
    // A colour he chose is his. stroke="black" is not rescued into anything - on white it is
    // simply the right answer, and the rescue that used to be here was only ever a symptom of
    // the ground being dark.
    wantInk: [
      { sel: 'rect[stroke="black"]', is: 'rgb(0, 0, 0)' },
      { sel: 'rect[stroke="#c33"]', is: 'rgb(204, 51, 51)' },
    ],
  },
  {
    name: 'hostile',
    kind: 'scratchpad',
    // What the model will write by accident sooner or later, and what has to happen to it. The
    // image is the one that matters and it is not about safety: the iframe's load event waits on
    // subresources, so an external src would stall the paint for as long as DNS and TCP take, on
    // a panel whose whole promise is that this lands while he is still talking.
    payload: { scratchpad: '<img src="https://example.com/nope.png">' +
                     '<script>document.title = "ran"</script><h1>still painted</h1>' },
    wantPhotoClass: true,
    // Their presence IS the assertion - see the splice below.
    allow: /Content Security Policy|sandboxed/,
  },
];

mkdirSync(SHOTS, { recursive: true });
mkdirSync(join(homedir(), '.cache', 'cyclops'), { recursive: true });

const browser = await chromium.launch();
// 800x480 at scale 1 is the panel exactly - the page's --u unit resolves to 1px only there, so
// checking any other size checks a layout nobody will ever see.
const page = await browser.newPage({ viewport: { width: 800, height: 480 }, deviceScaleFactor: 2 });
const errors = [];
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
page.on('pageerror', (e) => errors.push('PAGEERROR: ' + e.message));

let failed = 0;

// ---- deleting a session ----
//
// The list is the admin page's own, so this drives the page against a stubbed card: /api/sessions
// answers the fixture rows below and /api/panel answers "nothing up" - otherwise a pending picture
// covers the list and every row measures as invisible. Nothing is deleted from anywhere; the
// delete route is stubbed too, and only the page's half of it is under test. The server's half is
// tests/test_delete_session.py.
//
// ONLY=sessions runs this section and stops, so it can be run without touching the panel offer
// the picture cases below write into ~/.cache/cyclops.
const ROWS = ['a', 'b', 'c', 'd', 'e', 'f'].map((k, i) => ({
  name: `2026-09-2${i}_10-00-00_row-${k}`, title: `Row ${k.toUpperCase()}`,
  summary: 'Swapped the fork seals and bled the brakes.', started: `2026-09-2${i}T10:00:00`,
  seconds: 600, video: false, photos: 2, verdict: i === 1 ? 'live' : 'finished',
}));
async function stubCard(target) {
  const card = { rows: [...ROWS], asked: 0, deleted: [] };
  await target.route('**/api/panel', (r) => r.fulfill({ json: { picture: null, screen: null } }));
  await target.route('**/api/sessions', (r) => { card.asked++; r.fulfill({ json: { sessions: card.rows } }); });
  await target.route('**/api/session/*/delete', (r) => {
    const name = decodeURIComponent(r.request().url().split('/').slice(-2)[0]);
    card.deleted.push(name);
    card.rows = card.rows.filter((one) => one.name !== name);
    r.fulfill({ json: { sessions: card.rows } });
  });
  return card;
}
const rowOf = (target, i) => target.locator('#v-sessions .row').nth(i);
const middle = async (loc) => {
  const b = await loc.boundingBox();
  return [b.x + b.width / 2, b.y + b.height / 2];
};
const rowState = (target, i) => rowOf(target, i).evaluate((el) => ({
  holding: el.classList.contains('holding'), armed: el.classList.contains('armed'),
  cells: [...el.querySelectorAll('[data-act]')].map((c) => c.textContent),
}));

{
  const panel = await browser.newPage({ viewport: { width: 800, height: 480 }, deviceScaleFactor: 2 });
  panel.on('pageerror', (e) => errors.push('SESSIONS PAGEERROR: ' + e.message));
  const card = await stubCard(panel);
  const bad = [], notes = [];
  const toList = async () => {
    await panel.evaluate(() => { location.hash = '#/status'; });
    await panel.evaluate(() => { location.hash = '#/sessions'; });
    await panel.waitForSelector('#v-sessions .row', { timeout: 10000 });
    await panel.waitForTimeout(200);
  };
  await panel.goto(BASE + '/#/sessions', { waitUntil: 'domcontentloaded' });
  await panel.waitForSelector('#v-sessions .row', { timeout: 10000 });
  if (!(await panel.evaluate(() => document.body.classList.contains('kiosk')))) {
    bad.push('the panel page came back without body.kiosk - is BASE loopback?');
  }
  await panel.waitForTimeout(300);
  await panel.screenshot({ path: join(SHOTS, 'sessions-before.png') });

  // A tap still opens the session.
  let [x, y] = await middle(rowOf(panel, 0));
  await panel.mouse.click(x, y);
  await panel.waitForTimeout(200);
  if (!(await panel.evaluate(() => location.hash)).startsWith('#/s/')) bad.push('a tap no longer opens a session');
  await toList();

  // A hold fills the row, then arms it; the release that follows opens nothing.
  [x, y] = await middle(rowOf(panel, 0));
  await panel.mouse.move(x, y);
  await panel.mouse.down();
  await panel.waitForTimeout(350);
  const mid = await rowState(panel, 0);
  const fill = await rowOf(panel, 0).evaluate((el) => getComputedStyle(el, '::after').transform);
  await panel.screenshot({ path: join(SHOTS, 'sessions-holding.png') });
  if (!mid.holding || mid.armed) bad.push(`at 350 ms the row is ${JSON.stringify(mid)}, wanted filling`);
  notes.push(`fill mid-hold ${fill}`);
  await panel.waitForTimeout(450);
  const held = await rowState(panel, 0);
  await panel.mouse.up();
  await panel.waitForTimeout(250);
  await panel.screenshot({ path: join(SHOTS, 'sessions-armed.png') });
  if (!held.armed || held.cells.join() !== 'DELETE,CANCEL') {
    bad.push(`a 0.8 s hold left the row ${JSON.stringify(held)}, wanted DELETE and CANCEL`);
  }
  if ((await panel.evaluate(() => location.hash)) !== '#/sessions') bad.push('the release after arming opened the session');

  // CANCEL puts it back, on the press.
  [x, y] = await middle(panel.locator('.armno'));
  await panel.mouse.move(x, y);
  await panel.mouse.down();
  const cancelled = await rowState(panel, 0);
  await panel.mouse.up();
  await panel.waitForTimeout(200);
  if (cancelled.armed) bad.push('CANCEL did not disarm the row on the press');
  if ((await panel.evaluate(() => location.hash)) !== '#/sessions') bad.push('CANCEL opened the session');

  // 40 px of travel with the button down is a scroll, however long the finger stays down.
  [x, y] = await middle(rowOf(panel, 2));
  const top0 = await panel.evaluate(() => document.getElementById('v-sessions').scrollTop);
  await panel.mouse.move(x, y);
  await panel.mouse.down();
  await panel.mouse.move(x, y - 40, { steps: 8 });
  await panel.waitForTimeout(900);
  const dragged = await panel.evaluate(() => ({
    armed: document.querySelectorAll('#v-sessions .row.armed').length,
    top: document.getElementById('v-sessions').scrollTop,
  }));
  await panel.mouse.up();
  await panel.waitForTimeout(600);
  if (dragged.armed) bad.push('a 40 px drag armed a row');
  if (dragged.top === top0) notes.push('the drag did not scroll (list may fit)');
  else notes.push(`drag scrolled ${Math.round(dragged.top - top0)} px, armed nothing`);
  await panel.evaluate(() => { document.getElementById('v-sessions').scrollTop = 0; });
  await panel.waitForTimeout(200);

  // The RECORDING row cannot be armed.
  [x, y] = await middle(rowOf(panel, 1));
  await panel.mouse.move(x, y);
  await panel.mouse.down();
  await panel.waitForTimeout(900);
  const live = await rowState(panel, 1);
  await panel.screenshot({ path: join(SHOTS, 'sessions-live-held.png') });
  await panel.mouse.up();
  if (live.armed || live.holding) bad.push('the RECORDING row armed or filled under a hold');
  await toList();

  // DELETE takes the row out without a reload, and the list is asked for again.
  const doomed = ROWS[0].name;
  [x, y] = await middle(rowOf(panel, 0));
  await panel.mouse.move(x, y);
  await panel.mouse.down();
  await panel.waitForTimeout(800);
  await panel.mouse.up();
  const askedBefore = card.asked;
  [x, y] = await middle(panel.locator('.armdel'));
  await panel.mouse.move(x, y);
  await panel.mouse.down();
  await panel.mouse.up();
  await panel.waitForTimeout(600);
  await panel.screenshot({ path: join(SHOTS, 'sessions-deleted.png') });
  const left = await panel.evaluate((n) => ({
    there: !!document.querySelector(`#v-sessions .row[data-open="${n}"]`),
    rows: document.querySelectorAll('#v-sessions .row').length,
    hash: location.hash,
  }), doomed);
  if (card.deleted.join() !== doomed) bad.push(`deleted ${JSON.stringify(card.deleted)}, wanted ${doomed}`);
  if (left.there) bad.push('the deleted row is still in the list');
  if (card.asked <= askedBefore) bad.push('/api/sessions was not asked again after the delete');
  if (left.hash !== '#/sessions') bad.push(`the DELETE press navigated to ${left.hash}`);
  // ...and leaving and coming back inside the 20 s window does not paint it back.
  await toList();
  if (await panel.evaluate((n) => !!document.querySelector(`#v-sessions .row[data-open="${n}"]`), doomed)) {
    bad.push('the deleted row came back from the cached listing');
  }
  notes.push(`${left.rows} rows left, /api/sessions asked ${card.asked - askedBefore}x after`);
  await panel.close();

  if (bad.length) { failed++; console.log('FAIL sessions:', bad.join('; ')); }
  else console.log('ok   sessions: tap opens, hold fills and arms, drag and RECORDING never arm, ' +
                   'DELETE removes the row -', notes.join('; '));
}

// The laptop: the same list from a LAN address, held for a second - and nothing to press appears.
{
  const lan = (() => {
    for (const rows of Object.values(networkInterfaces())) {
      for (const row of rows || []) {
        if (row.family === 'IPv4' && !row.internal) return `http://${row.address}:${new URL(BASE).port}`;
      }
    }
    return null;
  })();
  if (!lan) console.log('skip sessions-lan: no non-loopback address on this machine');
  else {
    const laptop = await browser.newPage({ viewport: { width: 1280, height: 800 } });
    await stubCard(laptop);
    await laptop.goto(lan + '/#/sessions', { waitUntil: 'domcontentloaded' });
    await laptop.waitForSelector('#v-sessions .row', { timeout: 10000 });
    const [x, y] = await middle(rowOf(laptop, 0));
    await laptop.mouse.move(x, y);
    await laptop.mouse.down();
    await laptop.waitForTimeout(1000);
    await laptop.screenshot({ path: join(SHOTS, 'sessions-lan-held.png') });
    const got = await laptop.evaluate(() => ({
      kiosk: document.body.classList.contains('kiosk'),
      arm: document.querySelectorAll('.arm, [data-act], .row.holding, .row.armed').length,
    }));
    await laptop.mouse.up();
    await laptop.close();
    const bad = [];
    if (got.kiosk) bad.push('the LAN page wore body.kiosk');
    if (got.arm) bad.push(`${got.arm} delete affordance(s) on the laptop page`);
    if (bad.length) { failed++; console.log('FAIL sessions-lan:', bad.join('; ')); }
    else console.log('ok   sessions-lan: a one-second hold on a laptop shows nothing to delete with');
  }
}

if (process.env.ONLY === 'sessions') {
  await browser.close();
  if (errors.length) { failed++; console.log('FAIL console errors:', errors); }
  console.log(failed ? `\n${failed} failed - see ${SHOTS}` : '\nall good');
  process.exit(failed ? 1 : 0);
}

for (const item of CASES) {
  writeFileSync(PENDING, JSON.stringify({ id: item.name + '-' + Date.now(), ...item.payload }));
  let posted = false, kept = '';
  // The body of that request is empty. It used to carry the page's own picture of a scratchpad for
  // the recording; the recording is taken off the screen now (cyclops/screen.py), so anything in
  // it is a leftover the kiosk no longer reads.
  const watch = (r) => {
    if (!r.url().endsWith('/panel/painted')) return;
    posted = true;
    kept = r.postData() || '';
  };
  page.on('request', watch);
  const seen = errors.length;
  try {
    await page.goto(BASE, { waitUntil: 'domcontentloaded' });
    await page.waitForSelector('body.drawing', { timeout: 20000 });
    await page.waitForTimeout(1200);
    await page.screenshot({ path: join(SHOTS, item.name + '.png') });
    const got = await page.evaluate(() => {
      const img = document.querySelector('#stage img.shot');
      const frame = document.querySelector('#stage iframe.scratchpad');
      return {
        // naturalWidth is the whole point: an <img> whose src the browser could not decode is
        // still in the DOM, still has a class, and is zero by zero.
        width: img ? img.naturalWidth : 0,
        height: img ? img.naturalHeight : 0,
        // A scratchpad is a document of its own, so there is little to measure from out here beyond
        // its arrival and whether it lets a press reach the stage - which is the only way one is
        // ever dismissed. What is inside it is read through the frame tree below.
        frame: !!frame,
        clickThrough: frame ? getComputedStyle(frame).pointerEvents === 'none' : null,
        photo: document.body.classList.contains('photo'),
        frameBox: frame ? `${Math.round(frame.getBoundingClientRect().width)}x` +
                          `${Math.round(frame.getBoundingClientRect().height)}` : null,
        // Nothing may hang off the panel: the stage is the whole 800x480 while it is up.
        overflow: document.documentElement.scrollWidth > 800,
      };
    });
    const bad = [];
    if (item.kind === 'scratchpad') {
      if (!got.frame) bad.push('the scratchpad never reached the stage as an iframe');
      if (got.clickThrough === false) {
        bad.push('the frame takes the press itself, so nothing can put the scratchpad away');
      }
    }
    if (item.kind === 'scratchpad') {
      const inside = page.frameLocator('#stage iframe.scratchpad');
      // White, always, whatever is on it. One ground and one set of defaults: the two-scheme
      // version - green for writing, white for drawing - had a bug in each.
      const ground = await inside.locator('body').first()
        .evaluate((b) => getComputedStyle(b).backgroundColor).catch(() => null);
      if (ground !== 'rgb(255, 255, 255)') {
        bad.push(`the scratchpad's ground is ${ground}, wanted white`);
      }
      if (item.wantFills) {
        // Measured on the frame and not on the drawing inside it, because the drawing is
        // legitimately smaller when there is a heading over it. What must never come back is
        // anything taking a slice off the panel before the scratchpad even starts.
        if (got.frameBox !== '800x480') {
          bad.push(`the scratchpad is ${got.frameBox} of the panel's 800x480` +
                   ' - the bezel or its gutter is framing it again');
        }
      }
      for (const want of item.wantInk || []) {
        const stroke = await inside.locator(want.sel).first()
          .evaluate((el) => getComputedStyle(el).stroke).catch(() => null);
        if (stroke !== want.is) bad.push(`${want.sel} strokes ${stroke}, wanted ${want.is}`);
      }
      if (item.wantTall) {
        // Read through the frame tree and not page.evaluate, because the frame is sandboxed. This
        // is the assertion that matters: it proves SCRATCHPAD_HEAD's own stylesheet reached the
        // document, which is the failure that otherwise ships silently as 16 px text on a panel
        // being read at arm's length across a bench.
        const box = await page.frameLocator('#stage iframe.scratchpad')
          .locator(item.wantTall.sel).first().boundingBox().catch(() => null);
        if (!box) bad.push(`nothing matched ${item.wantTall.sel} inside the frame`);
        else if (box.height < item.wantTall.px) {
          bad.push(`${item.wantTall.sel} is ${Math.round(box.height)} px tall, wanted ` +
                   `${item.wantTall.px}+ - the scratchpad's own stylesheet did not apply`);
        }
      }
    } else if (!got.width || !got.height) {
      bad.push('the picture never decoded onto the stage');
    }
    if (item.allow) {
      // The complaints this case is here to provoke, taken off the list so the end-of-run check
      // stays a check. Silence would mean the CSP and the sandbox had stopped holding, and that
      // is the failure - not the noise.
      const raised = errors.splice(seen);
      const kept = raised.filter((line) => !item.allow.test(line));
      errors.push(...kept);
      if (raised.length === kept.length) bad.push('nothing was blocked; the guards did not fire');
    }
    if (got.photo !== item.wantPhotoClass) {
      bad.push(`body.photo is ${got.photo}, wanted ${item.wantPhotoClass}` +
               ' - press-to-dismiss keys off it, and it is the only way out of a picture');
    }
    if (got.overflow) bad.push('the page scrolls sideways behind the picture');
    if (!posted) bad.push('the page never told the kiosk it had painted; it would uncover late');
    if (kept) bad.push(`the paint posted ${kept.length} chars; it should post nothing`);
    if (bad.length) { failed++; console.log(`FAIL ${item.name}: ${bad.join('; ')}`); }
    else {
      const what = item.kind === 'scratchpad'
        ? 'painted in a frame of its own'
        : `${got.width}x${got.height} decoded`;
      console.log(`ok   ${item.name}: ${what}, posted back`);
    }
  } finally {
    page.off('request', watch);
  }
}

// And that taking the offer away puts the panel back to the dashboard, which is what the kiosk
// relies on every time it uncovers onto something else.
try { unlinkSync(PENDING); } catch { /* already gone */ }
await page.waitForTimeout(1200);
const cleared = await page.evaluate(() => ({
  drawing: document.body.classList.contains('drawing'),
  stage: document.getElementById('stage').childElementCount,
}));
if (cleared.drawing || cleared.stage) {
  failed++;
  console.log(`FAIL withdraw: body.drawing=${cleared.drawing}, ${cleared.stage} left on the stage`);
} else {
  console.log('ok   withdraw: the stage went back to the dashboard');
}

// ---- a page of a manual ----
//
// The one picture that is not contained. A portrait page inside the bezel was a 315 px strip of
// small print between two black bars, so a page (`page: true` in the offer, see
// cyclops/panel.py) goes the full width, opens at the top, scrolls under a finger - which on the
// panel arrives as a mouse, hence real mouse drags here - and is put away by a tap rather than
// a press. Every other picture keeps the press, and the last case below holds it to that.
//
// The page is drawn here on a canvas, A4-shaped at what imagine.for_page sends (800x1132), so the
// check still needs no fixture. PAGE_JPEG=<a real render> swaps in a real one for screenshots.
const PAGE_URL = process.env.PAGE_JPEG
  ? 'data:image/jpeg;base64,' + (await import('node:fs')).readFileSync(process.env.PAGE_JPEG)
      .toString('base64')
  : await page.evaluate(() => {
    const sheet = document.createElement('canvas');
    sheet.width = 800;
    sheet.height = 1132;
    const pen = sheet.getContext('2d');
    pen.fillStyle = '#fff';
    pen.fillRect(0, 0, 800, 1132);
    pen.fillStyle = '#111';
    pen.font = 'bold 40px sans-serif';
    pen.fillText('TOP OF THE PAGE', 60, 80);
    pen.font = '16px sans-serif';
    for (let y = 130; y < 1060; y += 26) {
      pen.fillText(`Line ${(y - 104) / 26}: tighten the M6 bolts to 10 Nm in a cross pattern.`, 60, y);
    }
    pen.font = 'bold 40px sans-serif';
    pen.fillText('BOTTOM OF THE PAGE', 60, 1110);
    return sheet.toDataURL('image/jpeg', 0.85);
  });

const closes = [];
const onClose = (r) => { if (r.url().endsWith('/close') && r.method() === 'POST') closes.push(r); };
page.on('request', onClose);
const stageNow = () => page.evaluate(() => {
  const st = document.getElementById('stage');
  const img = st.querySelector('img.shot');
  const box = img ? img.getBoundingClientRect() : null;
  return {
    page: document.body.classList.contains('page'),
    drawing: document.body.classList.contains('drawing'),
    width: box ? Math.round(box.width) : 0,
    left: box ? Math.round(box.left) : null,
    top: box ? Math.round(box.top) : null,
    scrollTop: Math.round(st.scrollTop),
    room: st.scrollHeight - st.clientHeight,
  };
});
const drag = async (from, to) => {
  await page.mouse.move(400, from);
  await page.mouse.down();
  await page.mouse.move(400, to, { steps: 12 });
  await page.mouse.up();
};

{
  writeFileSync(PENDING, JSON.stringify({ id: 'page-' + Date.now(), title: 'p0042',
    image: PAGE_URL, page: true }));
  const bad = [];
  await page.waitForSelector('body.drawing', { timeout: 20000 });
  await page.waitForTimeout(800);
  await page.screenshot({ path: join(SHOTS, 'page.png') });
  const opened = await stageNow();
  if (!opened.page) bad.push('no body.page, so the page is fitted like a photograph');
  if (opened.width !== 800 || opened.left !== 0) {
    bad.push(`the page is ${opened.width} px wide at x=${opened.left}, wanted 800 at 0`);
  }
  if (opened.top !== 0 || opened.scrollTop !== 0) {
    bad.push(`the page opens at top=${opened.top}, scrollTop=${opened.scrollTop}, wanted 0 and 0`);
  }
  if (opened.room <= 0) bad.push('the page does not scroll; a portrait page is taller than 480');

  // One 200 px drag, read the moment the finger lifts - before any glide - and then again once it
  // has settled, which is what a flick does to a page.
  await drag(400, 200);
  const lifted = (await stageNow()).scrollTop;
  await page.waitForTimeout(1500);
  const settled = (await stageNow()).scrollTop;
  if (lifted <= 0) bad.push('a 200 px drag did not move the page');
  // ...and on until there is no more page. Slow drags as well as quick ones would be nice, but a
  // page that can be reached the bottom of at all is the claim.
  let bottom = await stageNow();
  for (let i = 0; i < 12 && bottom.scrollTop < bottom.room - 1; i++) {
    await drag(400, 200);
    await page.waitForTimeout(400);
    bottom = await stageNow();
  }
  await page.screenshot({ path: join(SHOTS, 'page-bottom.png') });
  if (bottom.scrollTop < bottom.room - 1) {
    bad.push(`drags stopped at ${bottom.scrollTop} of ${bottom.room}; the bottom is out of reach`);
  }
  if (!bottom.drawing || !bottom.page) bad.push('dragging put the page away');
  if (closes.length) bad.push(`dragging POSTed /close ${closes.length} time(s)`);

  // A tap - press and lift in one place - is the way out.
  await page.mouse.click(400, 240);
  await page.waitForTimeout(500);
  if (closes.length !== 1) bad.push(`a tap POSTed /close ${closes.length} time(s), wanted 1`);

  if (bad.length) { failed++; console.log('FAIL page:', bad.join('; ')); }
  else {
    console.log(`ok   page: 800 px edge to edge from the top; a 200 px drag moved it ${lifted} px ` +
                `(${settled} after the glide), ${bottom.room} px reached, a tap closed it`);
  }
}

// ...and a photograph behind it is exactly what it was: contained, and away on the press - the
// POST goes before the finger lifts.
{
  closes.length = 0;
  writeFileSync(PENDING, JSON.stringify({ id: 'photo-after-page-' + Date.now(),
    title: '14-32-40_you', image: 'data:image/jpeg;base64,' + JPEG_B64 }));
  await page.waitForFunction(() => document.body.classList.contains('drawing') &&
    !document.body.classList.contains('page') &&
    document.querySelector('#stage img.shot')?.naturalWidth === 200, null, { timeout: 20000 });
  const bad = [];
  const got = await stageNow();
  const fit = await page.evaluate(() =>
    getComputedStyle(document.querySelector('#stage img.shot')).objectFit);
  if (fit !== 'contain') bad.push(`the photo is object-fit ${fit}, wanted contain`);
  if (got.room > 0) bad.push('the stage scrolls under a photograph');
  await page.mouse.move(400, 240);
  await page.mouse.down();
  await page.waitForTimeout(400);
  if (closes.length !== 1) bad.push(`a press POSTed /close ${closes.length} time(s), wanted 1`);
  await page.mouse.up();
  if (bad.length) { failed++; console.log('FAIL photo press:', bad.join('; ')); }
  else console.log('ok   photo press: contained, and put away on the press as before');
}
page.off('request', onClose);
try { unlinkSync(PENDING); } catch { /* already gone */ }
try { unlinkSync(CLOSE_FLAG); } catch { /* already gone */ }
await page.waitForTimeout(1200);

// ---- companion mode ----
//
// The same picture on a second screen, and the half no Python test can see. A phone or an iPad on
// the LAN gets the identical document with `local` false, so it has no #close button, no volume
// slider and no paint handshake - and it must still paint the picture and still put it away.
//
// Driven from a real LAN address rather than 127.0.0.1, because that address IS the feature:
// views._is_local reads REMOTE_ADDR, and loopback would silently test the kiosk path again.
const LAN = process.env.COMPANION || (() => {
  for (const rows of Object.values(networkInterfaces())) {
    for (const row of rows || []) {
      if (row.family === 'IPv4' && !row.internal) return `http://${row.address}:${new URL(BASE).port}`;
    }
  }
  return null;
})();

if (!LAN) {
  console.log('skip companion: no non-loopback address on this machine');
} else {
  // A phone, not a panel: portrait, no fixed scale. If it works here it works on the iPad, and
  // the narrow case is the one the wide-only transcript rules could have broken.
  const phone = await browser.newPage({ viewport: { width: 390, height: 844 } });
  phone.on('console', (m) => { if (m.type() === 'error') errors.push('COMPANION: ' + m.text()); });
  phone.on('pageerror', (e) => errors.push('COMPANION PAGEERROR: ' + e.message));
  let painted = false;
  phone.on('request', (r) => { if (r.url().endsWith('/panel/painted')) painted = true; });

  // A picture is on the glass. Both notes, because they answer different questions and the
  // dismissal below needs the second one: PENDING is what to paint, PICTURE_UP is whether
  // anybody can see it - the gate views.close_browser holds the LAN to.
  writeFileSync(PENDING, JSON.stringify({ id: 'companion-' + Date.now(),
    title: 'Relay driven from GPIO 17', image: 'data:image/jpeg;base64,' + JPEG_B64 }));
  writeFileSync(PICTURE_UP, '');
  try { unlinkSync(CLOSE_FLAG); } catch { /* already gone */ }

  await phone.goto(LAN, { waitUntil: 'domcontentloaded' });
  await phone.waitForSelector('body.drawing', { timeout: 20000 });
  await phone.waitForTimeout(1400);
  await phone.screenshot({ path: join(SHOTS, 'companion.png') });

  const got = await phone.evaluate(() => {
    const img = document.querySelector('#stage img.shot');
    return {
      kiosk: document.body.classList.contains('kiosk'),
      width: img ? img.naturalWidth : 0,
      photo: document.body.classList.contains('photo'),
      overflow: document.documentElement.scrollWidth > window.innerWidth,
    };
  });
  const bad = [];
  if (got.kiosk) bad.push('the LAN page came back wearing body.kiosk - REMOTE_ADDR was loopback');
  if (!got.width) bad.push('the picture never decoded on the companion');
  if (!got.photo) bad.push('no body.photo, so nothing on this screen can put the picture away');
  if (got.overflow) bad.push('the picture hangs off the side of a 390px screen');
  if (painted) bad.push('the companion posted /panel/painted - only the kiosk answers for a reveal');

  // ...and the gesture. A press anywhere puts it away here and on the panel: the stage clears at
  // once, and the note the kiosk is waiting on is left behind.
  await phone.locator('#stage').dispatchEvent('pointerdown');
  await phone.waitForTimeout(300);
  const after = await phone.evaluate(() => ({
    drawing: document.body.classList.contains('drawing'),
    stage: document.getElementById('stage').childElementCount,
  }));
  if (after.drawing || after.stage) bad.push('the press left the picture on the companion');
  if (!existsSync(CLOSE_FLAG)) bad.push('the press left the kiosk no note, so the panel keeps it');

  // The picture is still offered - the kiosk has not withdrawn it yet - and this screen must not
  // paint it back on the next poll. That regression is invisible in a screenshot taken too early.
  await phone.waitForTimeout(1400);
  const later = await phone.evaluate(() => document.body.classList.contains('drawing'));
  if (later) bad.push('the picture flashed back on the next poll after being dismissed');

  if (bad.length) { failed++; console.log('FAIL companion:', bad.join('; ')); }
  else console.log('ok   companion: painted on the LAN, dismissed on a press, left the note');

  try { unlinkSync(PENDING); } catch { /* already gone */ }
  try { unlinkSync(PICTURE_UP); } catch { /* already gone */ }
  try { unlinkSync(CLOSE_FLAG); } catch { /* already gone */ }
  await phone.close();
}

await browser.close();

if (errors.length) { failed++; console.log('FAIL console errors:', errors); }
console.log(failed ? `\n${failed} failed - see ${SHOTS}` : '\nall good');
process.exit(failed ? 1 : 0);
