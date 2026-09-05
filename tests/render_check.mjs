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
 * snippet the model wrote reaches it as an <iframe> with a document of its own, laid out by that
 * document's own stylesheet and letting a press through to the stage behind it; the page posts
 * back so the kiosk may uncover; a drawing keeps its corner button while a photograph and a
 * snippet do not; and what the model wrote cannot run a script or reach the network.
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
import { mkdirSync, writeFileSync, unlinkSync } from 'node:fs';
import { homedir } from 'node:os';
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
    payload: { title: 'Relay driven from GPIO 17', image: 'data:image/jpeg;base64,' + JPEG_B64,
               drawn: true },
    // A drawing is a thing you read and point at while you talk about it, so it keeps the corner
    // button rather than putting itself away under the finger you are pointing with.
    wantPhotoClass: false,
  },
  {
    name: 'photo',
    kind: 'image',
    payload: { title: '14-32-40_you', image: 'data:image/jpeg;base64,' + JPEG_B64, drawn: false },
    wantPhotoClass: true,
  },
  {
    name: 'card',
    kind: 'card',
    // Not a scrap of styling, which is the case that matters. The model has to be able to write
    // `<h1>25 Nm</h1>` and get a good screen, because every character of CSS it writes instead is
    // silence between the question and the answer - see CARD_HEAD in panel.js.
    payload: { html: '<h1>25 Nm</h1><ol><li>Loosen the pinch bolt</li>' +
                     '<li>Torque the crank</li></ol>' },
    // He put this up unasked, over his eye and over the way out of the session, so it goes away
    // on a press anywhere rather than at a corner square somebody has to find first.
    wantPhotoClass: true,
    wantTall: { sel: 'h1', px: 48 },
  },
  {
    name: 'hostile',
    kind: 'card',
    // What the model will write by accident sooner or later, and what has to happen to it. The
    // image is the one that matters and it is not about safety: the iframe's load event waits on
    // subresources, so an external src would stall the paint for as long as DNS and TCP take, on
    // a panel whose whole promise is that this lands while he is still talking.
    payload: { html: '<img src="https://example.com/nope.png">' +
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
for (const item of CASES) {
  writeFileSync(PENDING, JSON.stringify({ id: item.name + '-' + Date.now(), ...item.payload }));
  let posted = false;
  const watch = (r) => { if (r.url().endsWith('/panel/painted')) posted = true; };
  page.on('request', watch);
  const seen = errors.length;
  try {
    await page.goto(BASE, { waitUntil: 'domcontentloaded' });
    await page.waitForSelector('body.drawing', { timeout: 20000 });
    await page.waitForTimeout(1200);
    await page.screenshot({ path: join(SHOTS, item.name + '.png') });
    const got = await page.evaluate(() => {
      const img = document.querySelector('#stage img.shot');
      const frame = document.querySelector('#stage iframe.card');
      return {
        // naturalWidth is the whole point: an <img> whose src the browser could not decode is
        // still in the DOM, still has a class, and is zero by zero.
        width: img ? img.naturalWidth : 0,
        height: img ? img.naturalHeight : 0,
        // A snippet is a document of its own, so there is little to measure from out here beyond
        // its arrival and whether it lets a press reach the stage - which is the only way one is
        // ever dismissed. What is inside it is read through the frame tree below.
        frame: !!frame,
        clickThrough: frame ? getComputedStyle(frame).pointerEvents === 'none' : null,
        photo: document.body.classList.contains('photo'),
        // Nothing may hang off the panel: the stage is the whole 800x480 while it is up.
        overflow: document.documentElement.scrollWidth > 800,
      };
    });
    const bad = [];
    if (item.kind === 'card') {
      if (!got.frame) bad.push('the snippet never reached the stage as an iframe');
      if (got.clickThrough === false) {
        bad.push('the frame takes the press itself, so nothing can put the snippet away');
      }
      if (item.wantTall) {
        // Read through the frame tree and not page.evaluate, because the frame is sandboxed. This
        // is the assertion that matters: it proves CARD_HEAD's own stylesheet reached the
        // document, which is the failure that otherwise ships silently as 16 px text on a panel
        // being read at arm's length across a bench.
        const box = await page.frameLocator('#stage iframe.card')
          .locator(item.wantTall.sel).first().boundingBox().catch(() => null);
        if (!box) bad.push(`nothing matched ${item.wantTall.sel} inside the frame`);
        else if (box.height < item.wantTall.px) {
          bad.push(`${item.wantTall.sel} is ${Math.round(box.height)} px tall, wanted ` +
                   `${item.wantTall.px}+ - the snippet's own stylesheet did not apply`);
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
               ' - the corner button and press-to-dismiss both key off it');
    }
    if (got.overflow) bad.push('the page scrolls sideways behind the picture');
    if (!posted) bad.push('the page never told the kiosk it had painted; it would uncover late');
    if (bad.length) { failed++; console.log(`FAIL ${item.name}: ${bad.join('; ')}`); }
    else {
      const what = item.kind === 'card'
        ? 'painted in a frame of its own' : `${got.width}x${got.height} decoded`;
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

await browser.close();

if (errors.length) { failed++; console.log('FAIL console errors:', errors); }
console.log(failed ? `\n${failed} failed - see ${SHOTS}` : '\nall good');
process.exit(failed ? 1 : 0);
