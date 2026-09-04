/*
 * Does the panel actually draw? The half of tests/test_diagram.py that Python cannot see.
 *
 * Both failures that matter here are silent. A port label whose selector is wrong renders as
 * nothing at all - no error, no warning, just a pinout of anonymous dots. A drawing fitted with
 * `useModelGeometry: true` has its outboard labels clipped off the paper edge, and the SVG is
 * still perfectly valid. Neither shows up in a unit test, a lint, or a 200 from the server. So
 * this loads the real page from the real service and counts what came out.
 *
 * Not run by pytest, deliberately: pyproject says the suite must run anywhere in a second, and
 * this needs Chromium and a server. Run it when you touch the template, the symbol library, or
 * the vendored bundles.
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
const PENDING = join(homedir(), '.cache', 'cyclops', 'diagram.json');
const SHOTS = '/tmp/cyclops-render';

// One drawing of each kind, because each exercises a different thing: ports and the manhattan
// router, forty labels outside the body, and a layout nobody supplied coordinates for.
const box = (id, label, sublabel) => ({
  id, type: 'box', label, sublabel,
  ports: [{ id: 'in', side: 'left' }, { id: 'out', side: 'right' }, { id: 'ref', side: 'left' }],
});
const PINS = ['3V3', '5V', 'GPIO2', '5V', 'GPIO3', 'GND', 'GPIO4', 'GPIO14', 'GND', 'GPIO15',
  'GPIO17', 'GPIO18', 'GPIO27', 'GND', 'GPIO22', 'GPIO23', '3V3', 'GPIO24', 'GPIO10', 'GND',
  'GPIO9', 'GPIO25', 'GPIO11', 'GPIO8', 'GND', 'GPIO7', 'GPIO0', 'GPIO1', 'GPIO5', 'GND',
  'GPIO6', 'GPIO12', 'GPIO13', 'GND', 'GPIO19', 'GPIO16', 'GPIO26', 'GPIO20', 'GND', 'GPIO21'];

const CASES = [
  {
    name: 'wiring',
    // A resistor and an LED (real symbols), five labelled ports on the relay, and wires that
    // must cross - which is the only case that exercises the jumpover connector.
    want: { cells: 15, links: 9, minLabels: 22 },
    spec: {
      title: 'Relay driven from GPIO 17', caption: '', kind: 'wiring', layout: 'manual',
      nodes: [
        { id: 'pi', type: 'box', label: 'Raspberry Pi 5', sublabel: 'GPIO header',
          at: { x: 10, y: 100 }, size: { w: 150, h: 112 }, ports: [
            { id: 'GPIO17', side: 'right', label: 'GPIO17', kind: 'signal' },
            { id: '5V', side: 'right', label: '5V', kind: 'power' },
            { id: 'GND', side: 'right', label: 'GND', kind: 'ground' }] },
        { id: 'r1', type: 'resistor', label: '1k', at: { x: 266, y: 96 },
          ports: [{ id: 'a', side: 'left' }, { id: 'b', side: 'right' }] },
        { id: 'relay', type: 'box', label: 'Relay module', sublabel: 'SRD-05VDC',
          at: { x: 432, y: 84 }, size: { w: 158, h: 150 }, ports: [
            { id: 'IN', side: 'left', label: 'IN', kind: 'signal' },
            { id: 'VCC', side: 'left', label: 'VCC', kind: 'power' },
            { id: 'GND', side: 'left', label: 'GND', kind: 'ground' },
            { id: 'COM', side: 'right', label: 'COM', kind: 'hot' },
            { id: 'NO', side: 'right', label: 'NO', kind: 'hot' }] },
        { id: 'led', type: 'led', label: 'STATUS', at: { x: 274, y: 240 },
          ports: [{ id: 'a', side: 'left' }, { id: 'k', side: 'right' }] },
        { id: 'psu', type: 'box', label: '12 V', sublabel: 'supply', at: { x: 690, y: 40 },
          size: { w: 100, h: 62 }, ports: [{ id: '+', side: 'left', kind: 'hot' },
                                           { id: '-', side: 'bottom', kind: 'ground' }] },
        { id: 'lamp', type: 'box', label: 'Lamp', sublabel: '12 V 10 W', at: { x: 690, y: 236 },
          size: { w: 100, h: 62 }, ports: [{ id: 'L', side: 'top', kind: 'hot' },
                                           { id: 'N', side: 'left', kind: 'ground' }] }],
      wires: [
        { from: 'pi:GPIO17', to: 'r1:a', kind: 'signal', label: 'GPIO 17' },
        { from: 'r1:b', to: 'relay:IN', kind: 'signal' },
        { from: 'pi:5V', to: 'relay:VCC', kind: 'power', label: '5 V' },
        { from: 'pi:GND', to: 'relay:GND', kind: 'ground', label: 'GND' },
        { from: 'pi:GND', to: 'led:k', kind: 'ground' },
        { from: 'r1:b', to: 'led:a', kind: 'signal' },
        { from: 'psu:+', to: 'relay:COM', kind: 'hot', label: '+12 V' },
        { from: 'relay:NO', to: 'lamp:L', kind: 'hot' },
        { from: 'psu:-', to: 'lamp:N', kind: 'ground' }],
    },
  },
  {
    name: 'pinout',
    // 40 pin labels plus the header's own. This is the case that catches the labelText selector:
    // get it wrong and every one of these is silently missing.
    want: { cells: 1, links: 0, minLabels: 41 },
    spec: {
      title: 'Raspberry Pi 5 40-pin header', caption: '', kind: 'pinout', layout: 'manual',
      nodes: [{ id: 'hdr', type: 'header', label: 'J8 · 40-PIN GPIO', at: { x: 0, y: 0 },
        size: { w: 120, h: 300 },
        ports: PINS.map((n, i) => ({
          id: 'p' + (i + 1),
          side: i % 2 === 0 ? 'left-out' : 'right-out',
          label: i % 2 === 0 ? `${n}  ${i + 1}` : `${i + 1}  ${n}`,
          kind: n === 'GND' ? 'ground' : (n === '5V' || n === '3V3') ? 'power' : 'signal',
        })) }],
      wires: [],
    },
  },
  {
    name: 'block',
    want: { cells: 12, links: 6, minLabels: 14 },
    spec: {
      title: 'Audio path, mic to speaker', caption: '', kind: 'block', layout: 'dagre',
      nodes: [box('mic', 'Mic', '24 kHz'), box('guard', 'EchoGuard', 'half-duplex'),
              box('rt', 'Realtime', 'gpt-realtime-2.1'), box('spk', 'Speaker', 'PipeWire'),
              box('rec', 'Recorder', 'video.mp4'), box('log', 'SessionLog', 'session.jsonl')],
      wires: [
        { from: 'mic:out', to: 'guard:in', kind: 'signal' },
        { from: 'guard:out', to: 'rt:in', kind: 'signal' },
        { from: 'rt:out', to: 'spk:in', kind: 'signal' },
        { from: 'spk:out', to: 'guard:ref', kind: 'ground', label: 'reference' },
        { from: 'spk:out', to: 'rec:in', kind: 'power' },
        { from: 'rt:out', to: 'log:in', kind: 'power', label: 'events' }],
    },
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
  writeFileSync(PENDING, JSON.stringify({
    id: item.name + '-' + Date.now(), title: item.spec.title, spec: item.spec, svg: null }));
  let posted = null;
  const watch = (r) => { if (r.url().endsWith('/diagram/shown')) posted = (r.postData() || '').length; };
  page.on('request', watch);
  try {
    await page.goto(BASE, { waitUntil: 'domcontentloaded' });
    await page.waitForSelector('body.drawing', { timeout: 20000 });
    await page.waitForTimeout(1500);
    await page.screenshot({ path: join(SHOTS, item.name + '.png') });
    const got = await page.evaluate(() => ({
      cells: document.querySelectorAll('#stage .joint-cell').length,
      links: document.querySelectorAll('#stage .joint-link').length,
      labels: [...document.querySelectorAll('#stage text')].map((t) => t.textContent)
        .filter((t) => t && t.trim()).length,
      title: document.getElementById('dtitle').textContent,
      // The whole drawing must be inside the paper. A label hanging off the left edge is the
      // useModelGeometry bug, and it is invisible in every other check.
      clipped: (() => {
        const svg = document.querySelector('#stage svg');
        if (!svg) return 'no svg';
        const paper = svg.getBoundingClientRect();
        for (const t of svg.querySelectorAll('text')) {
          const b = t.getBoundingClientRect();
          if (!b.width) continue;
          if (b.left < paper.left - 0.5 || b.right > paper.right + 0.5 ||
              b.top < paper.top - 0.5 || b.bottom > paper.bottom + 0.5) return t.textContent;
        }
        return null;
      })(),
    }));
    const bad = [];
    if (got.cells !== item.want.cells) bad.push(`cells ${got.cells} != ${item.want.cells}`);
    if (got.links !== item.want.links) bad.push(`links ${got.links} != ${item.want.links}`);
    if (got.labels < item.want.minLabels) bad.push(`only ${got.labels} labels drawn (want >= ${item.want.minLabels}) - port labels missing?`);
    if (got.clipped) bad.push(`"${got.clipped}" is clipped off the paper`);
    if (!posted) bad.push('the page never sent the picture back');
    if (bad.length) { failed++; console.log(`FAIL ${item.name}: ${bad.join('; ')}`); }
    else console.log(`ok   ${item.name}: ${got.cells} cells, ${got.links} links, ${got.labels} labels, ${posted} B svg`);
  } finally {
    page.off('request', watch);
  }
}

try { unlinkSync(PENDING); } catch { /* already gone */ }
await browser.close();

if (errors.length) { failed++; console.log('FAIL console errors:', errors); }
console.log(failed ? `\n${failed} failed - screenshots in ${SHOTS}` : `\nall ${CASES.length} drew - screenshots in ${SHOTS}`);
process.exit(failed ? 1 : 0);
