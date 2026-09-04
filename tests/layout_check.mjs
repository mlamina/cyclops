/*
 * Does the page still fit the screen it is on? The half of the interface no unit test can see.
 *
 * Two questions, one harness, because they are the same measurement asked twice:
 *
 *   1. THE PANEL MUST NOT MOVE. Every layout change for a phone is a chance to shift something
 *      on the 800x480 the interface was drawn for, and the panel is the one client that has no
 *      scrollbar, no zoom and nobody watching it boot. So the kiosk page is measured element by
 *      element and diffed against a saved baseline: any change at all is a failure until
 *      somebody re-baselines it on purpose.
 *
 *   2. NOTHING MAY BE DRAWN OFF THE EDGE. The bug this file was written for was silent: <body>
 *      is a grid whose single implicit column is sized by min-content, the header's min-content
 *      is ~626 px, and `overflow: hidden` then clipped the difference. Every descendant laid
 *      itself out at 626 px inside a 390 px phone - so session titles were sliced mid-word, the
 *      four readings showed a bar and no number, and two of the four tabs could not be tapped.
 *      Nothing errored. Nothing logged. It just was not there.
 *
 * Geometry and computed style, not pixels. A screenshot of this page carries a live CPU
 * percentage, a temperature that wanders and a video frame that decodes when it feels like it;
 * hashing that gives you a red run a week and teaches everyone to ignore it. A rectangle does
 * not wander. Screenshots are still written, because a number tells you *that* something moved
 * and only a picture tells you what it looks like now.
 *
 * The API is stubbed from fixtures below rather than read off the box: the session list has to
 * hold the same rows on a laptop with no sessions/ directory as on a Pi with forty, or the two
 * runs cannot be compared - and the long-title row is a case no real card is guaranteed to have.
 *
 * Not run by pytest, deliberately - same argument as render_check.mjs: this needs Chromium and
 * a server, and the suite must run anywhere in a second.
 *
 *   uv run cyclops-admin --host=0.0.0.0 --port=8099 &
 *   node tests/layout_check.mjs --save     # write the baseline (do this before you change CSS)
 *   node tests/layout_check.mjs            # compare against it
 *
 * Editing a fixture below moves the baseline too, because the geometry is the geometry of these
 * rows. Re-save in the same commit, and read the diff first: a fixture change should move the
 * things it added and nothing else.
 *
 * KIOSK= is the loopback origin, which is what makes views._is_local() true and puts
 * class="kiosk" on the body. LAN= must NOT be loopback, or you measure the panel twice and
 * conclude the phone is fine. Screenshots land in /tmp/cyclops-layout/.
 */

import { execSync } from 'node:child_process';
import { mkdirSync, writeFileSync, readFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

// Same resolution trick as render_check.mjs: this repo is Python, there is no node_modules.
async function loadPlaywright() {
  try { return await import('playwright'); } catch {
    const root = execSync('npm root -g', { encoding: 'utf8' }).trim();
    return await import(pathToFileURL(join(root, 'playwright', 'index.mjs')).href);
  }
}
const { chromium } = await loadPlaywright();

const HERE = dirname(fileURLToPath(import.meta.url));
const KIOSK = process.env.KIOSK || 'http://127.0.0.1:8099';
const LAN = process.env.LAN || 'http://192.168.1.31:8099';
const SHOTS = '/tmp/cyclops-layout';
const BASELINE = join(HERE, 'layout_baseline.json');
const SAVE = process.argv.includes('--save');

// ---------------------------------------------------------------- what the server would say
//
// Enough shape to exercise every branch the layout has: a row with a video and a row without,
// a title long enough to clamp, a summary long enough to clamp, a verdict that is a warning
// rather than a count, and one unbroken 44-character word - which is the input that finds a
// missing `min-width: 0` and nothing else does.

const STATUS = {
  temp_c: 55.1, temp_band: '', cpu_percent: 19, temp_percent: 41,
  mem_percent: 15, disk_percent: 31, mem_used: 1288490188, mem_total: 8455716864,
  disk_used: 40000000000, disk_free: 87616000000, disk_total: 128000000000,
  cpu_text: '19%', temp_text: '55.1°', mem_text: '1.2 / 7.9 GB', disk_text: '81.6 GB free',
  sessions: 7, sessions_in_progress: 0, uptime_s: 91234, load1: 0.42,
};
const SESSIONS = {
  sessions: [
    { name: '2026-09-01_18-13-08', title: 'Planned to swap a USB webcam and speaker for a Raspberry Pi camera module',
      summary: 'Worked through the case cut-out, the ribbon length and whether the official module or the Arducam clone is the one to buy.',
      started: '2026-09-01T18:13:08', seconds: 88, video: true, verdict: 'finished', photos: 3 },
    { name: '2026-09-01_17-02-55', title: 'ThermalThrottlingInvestigationOnThePiFiveUnderLoad',
      summary: 'A single unbroken word, which is the only input that finds a missing min-width.',
      started: '2026-09-01T17:02:55', seconds: 19, video: true, verdict: 'finished', photos: 0 },
    { name: '2026-08-31_09-40-00', title: 'Introduced yourself and blocked out what to build next',
      summary: 'Short one.', started: '2026-08-31T09:40:00', seconds: 26, video: false, verdict: 'live', photos: 0 },
    { name: '2026-08-30_21-15-42', title: 'Removed background, added a presentation and brightened the result',
      summary: 'Three photographs and a drawing came out of this one.',
      started: '2026-08-30T21:15:42', seconds: 262, video: true, verdict: 'unfinished', photos: 12 },
  ],
};
const SESSION = {
  name: '2026-09-01_18-13-08',
  title: 'Planned to swap a USB webcam and speaker for a Raspberry Pi camera module',
  summary: 'Worked through the case cut-out, the ribbon length and whether the official module or the Arducam clone is the one to buy. Settled on the official Camera Module 3 Wide.',
  started: '2026-09-01T18:13:08', seconds: 88, video: true, verdict: 'finished',
  photos: ['photos/18-13-40_bracket.jpg', 'photos/18-14-02_ribbon.jpg'], diagrams: [],
};
// The shape library.records() actually returns - `type` and `t` and `text`, not a prose
// invention. A fixture that does not match renders as a column of "Something went wrong", which
// is the transcript layout going untested while the check reports green.
const RECORDS = {
  records: [
    { type: 'you', t: 2.0, text: 'What would it take to put a proper camera on this thing?' },
    { type: 'cyclops', t: 6.5, text: 'The Camera Module 3 Wide is the one that fits the case without a new cut-out, and it is the official part, so libcamera already knows it. The Arducam clone wants a libcamera fork on the Pi, which is a maintenance bill rather than a purchase.' },
    { type: 'you', t: 19.2, text: 'Fine. Order it.', interrupted: true },
    { type: 'photo', t: 21.0, shown: true, file: 'photos/18-13-40_bracket.jpg' },
    { type: 'diagram', t: 44.0, title: 'Ribbon routing', found: false },
    { type: 'error', t: 61.0, message: 'the camera stopped answering for nine seconds' },
  ],
};
// library.Item, under the key media_stream() actually uses. A `drawn: false` diagram is in
// here because it is the one tile that renders text instead of a picture.
const shot = (kind, title, when, drawn) => ({
  kind, session: '2026-09-01_18-13-08', session_title: 'Planned to swap a USB webcam',
  title, when, url: `/media/2026-09-01_18-13-08/photos/${when.slice(11, 13)}.jpg`, drawn });
const MEDIA = {
  items: [
    shot('photo', 'a bracket', '2026-09-01T18:13:40', true),
    shot('photo', 'the ribbon, seated', '2026-09-01T18:14:02', true),
    shot('diagram', 'Ribbon routing from the module to the header', '2026-08-30T21:16:00', true),
    shot('diagram', 'A drawing the panel never sent back', '2026-08-30T21:18:00', false),
    shot('photo', 'the case, cut', '2026-08-30T21:20:00', true),
  ],
};
// shelf.projects(): name, title, tagline, status, updated, sessions, photos, thumb.
const PROJECTS = {
  projects: [
    { name: 'cyclops-camera-swap', title: 'Camera swap', tagline: 'Official Module 3 Wide, case-mounted, no new cut-out.',
      status: 'active', updated: '2026-09-01T18:20:00', sessions: 4, photos: 11, thumb: '' },
    { name: 'bench-power-supply', title: 'Bench power supply', tagline: 'Two rails and a current limit.',
      status: 'paused', updated: '2026-08-28T11:00:00', sessions: 2, photos: 3, thumb: '' },
  ],
};

const STUBS = [
  ['**/api/status', STATUS], ['**/api/sessions', SESSIONS], ['**/api/media', MEDIA],
  ['**/api/projects', PROJECTS], ['**/api/session/*/records', RECORDS], ['**/api/session/*', SESSION],
];

async function stub(ctx) {
  for (const [pattern, body] of STUBS) {
    await ctx.route(pattern, (route) =>
      route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) }));
  }
  // A picture is served, and a recording is not. A broken <img> is not a small picture: it is a
  // box sized by its own alt text, which on the media grid made a 4:3 tile 580 px tall and would
  // have had this file reporting a layout that does not exist. So stills get a 4:3 SVG, and the
  // aspect ratio the grid is measured at is the real one. Video is still aborted - .thumb is
  // `position: absolute; inset: 0` so it decides no size, and a decoder that finishes when it
  // feels like it is exactly the flake this file exists to avoid.
  const STILL = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 4 3">' +
                '<rect width="4" height="3" fill="#0d2418"/></svg>';
  await ctx.route('**/*.mp4', (route) => route.abort());
  for (const pattern of ['**/media/**', '**/project-media/**'])
    await ctx.route(pattern, (route) =>
      route.fulfill({ status: 200, contentType: 'image/svg+xml', body: STILL }));
}

// ---------------------------------------------------------------- what gets measured
//
// Every element matching one of these, in document order, with the box it actually occupies and
// the handful of properties that decide it. Not all of getComputedStyle: that is 340 properties
// per element of which 330 never move, and a diff nobody reads is a check nobody runs.

const WATCH = ['body', 'header', '.brand', '.nav', '.tab', '.link', '.system', '.controls', '.ctl',
  '.meters', '.meter', '.mtrack', '.mrow', '.mlabel', '.mvalue', '.vslider', '.sw', '.close',
  '.view', '.row', '.shot', '.rowtext', '.rowtitle', '.rowsum', '.rowwhen', '.grid', '.cell',
  '.watch', '.playing', '.video', '.talk', '.line', '.caption', '.stitle', '.smeta', '.ssum',
  '.lightbox', '.crumbs', '.files', '.doc', '.projects'];
const PROPS = ['display', 'grid-template-columns', 'grid-template-rows', 'flex-wrap',
  'font-size', 'padding', 'gap', 'align-content', 'overflow-x', 'overflow-y', 'position'];

async function probe(page) {
  return page.evaluate(({ WATCH, PROPS }) => {
    const round = (n) => Math.round(n * 100) / 100;
    const out = { u: getComputedStyle(document.body).getPropertyValue('--u').trim(),
                  cls: document.body.className, vw: innerWidth, vh: innerHeight, el: [] };
    for (const sel of WATCH) {
      document.querySelectorAll(sel).forEach((el, i) => {
        const r = el.getBoundingClientRect();
        if (!r.width && !r.height) return;      // hidden views are not geometry
        const cs = getComputedStyle(el);
        const style = {};
        for (const p of PROPS) style[p] = cs.getPropertyValue(p);
        out.el.push({ sel, i, x: round(r.x), y: round(r.y), w: round(r.width), h: round(r.height),
                      sw: el.scrollWidth, cw: el.clientWidth, style });
      });
    }
    return out;
  }, { WATCH, PROPS });
}

// The two assertions that would have caught the bug in the first place.
async function overflows(page) {
  return page.evaluate(() => {
    const bad = [];
    const doc = document.documentElement;
    if (doc.scrollWidth > innerWidth + 1)
      bad.push(`document is ${doc.scrollWidth}px wide in a ${innerWidth}px viewport`);
    // Anything drawn past the right edge of the window, however it got there.
    for (const el of document.querySelectorAll('body *')) {
      const r = el.getBoundingClientRect();
      if (!r.width || !r.height) continue;
      if (r.right > innerWidth + 1)
        bad.push(`${el.tagName.toLowerCase()}.${el.className || '(none)'} ends at ` +
                 `${Math.round(r.right)}px, past the ${innerWidth}px edge`);
      if (bad.length > 12) break;
    }
    return bad;
  });
}

async function tabsReachable(page) {
  return page.evaluate(() => {
    const bad = [];
    // A recording gets the whole screen and the header goes with it - that is body.solo doing
    // what it is for, not a tab that has gone missing. See the `.solo` rules in the stylesheet.
    if (document.body.classList.contains('solo')) return bad;
    for (const tab of document.querySelectorAll('.tab')) {
      const r = tab.getBoundingClientRect();
      if (r.width < 1 || r.height < 1) { bad.push(`${tab.textContent} has no box`); continue; }
      if (r.right > innerWidth + 1 || r.bottom > innerHeight + 1 || r.left < -1 || r.top < -1)
        bad.push(`${tab.textContent} is outside the viewport ` +
                 `(${Math.round(r.left)},${Math.round(r.top)} to ${Math.round(r.right)},${Math.round(r.bottom)})`);
      if (r.height < 40) bad.push(`${tab.textContent} is only ${Math.round(r.height)}px tall`);
    }
    return bad;
  });
}

// ---------------------------------------------------------------- the runs

const SCREENS = [
  ['system', '#/'],
  ['sessions', '#/sessions'],
  ['session', '#/s/2026-09-01_18-13-08'],
  ['media', '#/media'],
  ['projects', '#/projects'],
];
// Five shapes, chosen to be the ones in the house rather than a list of famous handsets: a
// phone both ways up, a tablet upright, the smallest laptop worth naming, and a 15".
const SIZES = [
  ['phone', 390, 844], ['phone-wide', 844, 390], ['tablet', 820, 1180],
  ['small-laptop', 1024, 768], ['laptop', 1512, 945],
];

async function visit(ctx, base, hash, shot) {
  const page = await ctx.newPage();
  const errors = [];
  // ERR_FAILED is this file aborting its own thumbnail requests; anything else is the page's.
  page.on('console', (m) => {
    if (m.type() === 'error' && !/ERR_FAILED|ERR_ABORTED/.test(m.text())) errors.push(m.text());
  });
  page.on('pageerror', (e) => errors.push(String(e)));
  await stub(ctx);
  await page.goto(base + '/' + hash, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(900);          // the router fetches, then paints
  await page.screenshot({ path: join(SHOTS, shot + '.png') });
  return { page, errors };
}

mkdirSync(SHOTS, { recursive: true });
const browser = await chromium.launch();
const found = {};
const failures = [];

// --- the panel, at the size it actually is ---------------------------------------------------
for (const [name, hash] of SCREENS) {
  const ctx = await browser.newContext({ viewport: { width: 800, height: 480 }, deviceScaleFactor: 1 });
  const { page, errors } = await visit(ctx, KIOSK, hash, `panel-${name}`);
  if (!(await page.evaluate(() => document.body.classList.contains('kiosk'))))
    failures.push(`panel/${name}: KIOSK=${KIOSK} is not loopback - this is the LAN page, not the panel`);
  found[`panel/${name}`] = await probe(page);
  for (const e of errors) failures.push(`panel/${name}: console: ${e}`);
  await ctx.close();
}

// --- every device on the LAN -------------------------------------------------------------------
for (const [size, width, height] of SIZES) {
  for (const [name, hash] of SCREENS) {
    const ctx = await browser.newContext({ viewport: { width, height },
      deviceScaleFactor: 2, isMobile: width < 900, hasTouch: width < 1200 });
    const { page, errors } = await visit(ctx, LAN, hash, `${size}-${name}`);
    if (await page.evaluate(() => document.body.classList.contains('kiosk')))
      failures.push(`${size}/${name}: LAN=${LAN} resolved to loopback - measuring the panel twice`);
    found[`${size}/${name}`] = await probe(page);
    for (const line of await overflows(page)) failures.push(`${size}/${name}: ${line}`);
    for (const line of await tabsReachable(page)) failures.push(`${size}/${name}: tab ${line}`);
    for (const e of errors) failures.push(`${size}/${name}: console: ${e}`);
    await ctx.close();
  }
}
await browser.close();

// --- the panel baseline ------------------------------------------------------------------------
const panelOnly = Object.fromEntries(Object.entries(found).filter(([k]) => k.startsWith('panel/')));
if (SAVE) {
  writeFileSync(BASELINE, JSON.stringify(panelOnly, null, 1));
  console.log(`· baseline written: ${BASELINE}`);
} else if (!existsSync(BASELINE)) {
  console.log('· no baseline yet; run with --save before changing anything');
} else {
  const was = JSON.parse(readFileSync(BASELINE, 'utf8'));
  for (const key of Object.keys(panelOnly)) {
    const a = was[key], b = panelOnly[key];
    if (!a) { failures.push(`${key}: not in the baseline`); continue; }
    if (a.u !== b.u) failures.push(`${key}: --u was ${a.u}, is ${b.u}`);
    const index = (p) => Object.fromEntries(p.el.map((e) => [`${e.sel}[${e.i}]`, e]));
    const [ia, ib] = [index(a), index(b)];
    for (const id of new Set([...Object.keys(ia), ...Object.keys(ib)])) {
      const x = ia[id], y = ib[id];
      if (!x) { failures.push(`${key}: ${id} appeared`); continue; }
      if (!y) { failures.push(`${key}: ${id} vanished`); continue; }
      for (const k of ['x', 'y', 'w', 'h'])
        if (x[k] !== y[k]) failures.push(`${key}: ${id} ${k} was ${x[k]}, is ${y[k]}`);
      for (const p of PROPS)
        if (x.style[p] !== y.style[p])
          failures.push(`${key}: ${id} ${p} was "${x.style[p]}", is "${y.style[p]}"`);
    }
  }
}

writeFileSync(join(SHOTS, 'probe.json'), JSON.stringify(found, null, 1));
console.log(`· screenshots and probe.json in ${SHOTS}`);
if (failures.length) {
  console.error(`\n✗ ${failures.length} problem(s):`);
  for (const f of failures.slice(0, 60)) console.error('  ' + f);
  if (failures.length > 60) console.error(`  ... and ${failures.length - 60} more`);
  process.exit(1);
}
console.log('· the panel has not moved, and nothing is drawn off the edge');
