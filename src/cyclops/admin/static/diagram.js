/* A drawing, on the panel: the symbol library, and the JointJS graph it builds.

   The library lives here, once, so that no model ever draws a resistor - the model says
   "resistor" and this is what that means. cyclops/diagram.py's SYMBOLS is the other half of
   that contract, and tests/test_diagram.py holds the two together by reading this file as text.

   Loaded only on the kiosk ({% if local %} in the template): a laptop on the LAN has no panel
   to draw on, and 527 KB of JointJS is a rude thing to hand it for a page of four numbers. It
   must come after the three vendored bundles and the graphlib bridge beside them. */

const GREEN = '#56ff8c', MID = '#2eb064', DIM = '#207646',
      AMBER = '#ffb83c', RED = '#ff5646', SCREEN = '#050f0a';
const FONT = 'ui-monospace, "DejaVu Sans Mono", Menlo, monospace';
const KIND = { signal: GREEN, power: AMBER, ground: DIM, hot: RED, '': MID };

// The symbol library. It lives here, once, so that no model ever draws a resistor: the model
// says "resistor" and this is what that means. cyclops/diagram.py's SYMBOLS is the other half
// of this contract - a shape added there and not here renders as a plain box.
//
// The port label's markup selector and the attrs key that carries its text must be the same
// word. They are `labelText` here because that is what JointJS's own label layout callbacks
// write textAnchor and y into, while the default markup calls it `text` - so writing the text
// to `portLabel`, or to `text` against this markup, lands it on nothing. There is no error and
// no warning: a 40-pin header renders as forty anonymous dots. tests/render_check.mjs counts
// the labels for exactly this reason, and fails at 1 of 41.
const portGroup = (side, labelSide) => ({
  position: { name: side },
  label: { position: { name: labelSide },
           markup: [{ tagName: 'text', selector: 'labelText' }] },
  markup: [{ tagName: 'circle', selector: 'portBody' }],
  attrs: {
    portBody: { r: 3.5, fill: SCREEN, stroke: MID, strokeWidth: 1.3, magnet: false },
    labelText: { fill: MID, fontSize: 9.5, fontFamily: FONT, letterSpacing: '.05em' }
  }
});
// "left"/"right" put the label inside the body, which is what a board or a module wants;
// the "-out" pair put it outside, which is the only way a pin header reads.
const PORTS = { groups: {
  'left': portGroup('left', 'right'),
  'right': portGroup('right', 'left'),
  'left-out': portGroup('left', 'left'),
  'right-out': portGroup('right', 'right'),
  'top': portGroup('top', 'top'),
  'bottom': portGroup('bottom', 'bottom')
} };

const Box = joint.dia.Element.define('cyclops.Box', {
  size: { width: 130, height: 66 },
  attrs: {
    body: { width: 'calc(w)', height: 'calc(h)', fill: SCREEN, stroke: MID, strokeWidth: 1.4, rx: 2 },
    label: { text: '', fill: GREEN, fontSize: 12, fontFamily: FONT, letterSpacing: '.08em',
             textAnchor: 'middle', textVerticalAnchor: 'middle',
             x: 'calc(w/2)', y: 'calc(h/2 - 7)' },
    sublabel: { text: '', fill: DIM, fontSize: 9.5, fontFamily: FONT, letterSpacing: '.1em',
                textAnchor: 'middle', textVerticalAnchor: 'middle',
                x: 'calc(w/2)', y: 'calc(h/2 + 9)' }
  }, ports: PORTS
}, { markup: [{ tagName: 'rect', selector: 'body' }, { tagName: 'text', selector: 'label' },
              { tagName: 'text', selector: 'sublabel' }] });

const Resistor = joint.dia.Element.define('cyclops.Resistor', {
  size: { width: 84, height: 34 },
  attrs: {
    body: { d: 'M 0 17 L 14 17 L 20 7 L 30 27 L 40 7 L 50 27 L 60 7 L 66 17 L 84 17',
            fill: 'none', stroke: GREEN, strokeWidth: 1.6, strokeLinejoin: 'round' },
    label: { text: '', fill: GREEN, fontSize: 11, fontFamily: FONT, letterSpacing: '.06em',
             textAnchor: 'middle', x: 42, y: -5 }
  }, ports: PORTS
}, { markup: [{ tagName: 'path', selector: 'body' }, { tagName: 'text', selector: 'label' }] });

const Led = joint.dia.Element.define('cyclops.Led', {
  size: { width: 64, height: 40 },
  attrs: {
    lead: { d: 'M 0 26 L 18 26 M 46 26 L 64 26', stroke: GREEN, strokeWidth: 1.6, fill: 'none' },
    body: { d: 'M 18 14 L 18 38 L 42 26 Z', fill: 'none', stroke: GREEN, strokeWidth: 1.6,
            strokeLinejoin: 'round' },
    bar: { d: 'M 44 14 L 44 38', stroke: GREEN, strokeWidth: 1.8 },
    rays: { d: 'M 26 12 L 34 2 M 33 12 L 41 2 M 32.2 3.4 L 34 2 L 33.2 4.8 M 39.2 3.4 L 41 2 L 40.2 4.8',
            stroke: AMBER, strokeWidth: 1.2, fill: 'none', strokeLinejoin: 'round' },
    label: { text: '', fill: DIM, fontSize: 9.5, fontFamily: FONT, letterSpacing: '.08em',
             textAnchor: 'middle', x: 32, y: 54 }
  }, ports: PORTS
}, { markup: [{ tagName: 'path', selector: 'lead' }, { tagName: 'path', selector: 'body' },
              { tagName: 'path', selector: 'bar' }, { tagName: 'path', selector: 'rays' },
              { tagName: 'text', selector: 'label' }] });

const Header = joint.dia.Element.define('cyclops.Header', {
  size: { width: 96, height: 300 },
  attrs: {
    body: { width: 'calc(w)', height: 'calc(h)', fill: SCREEN, stroke: MID, strokeWidth: 1.4, rx: 3 },
    label: { text: '', fill: DIM, fontSize: 10, fontFamily: FONT, letterSpacing: '.22em',
             textAnchor: 'middle', x: 'calc(w/2)', y: -10 }
  }, ports: PORTS
}, { markup: [{ tagName: 'rect', selector: 'body' }, { tagName: 'text', selector: 'label' }] });

joint.shapes.cyclops = { Box: Box, Resistor: Resistor, Led: Led, Header: Header };
const TYPES = { box: Box, resistor: Resistor, led: Led, header: Header };

// The spec -> a drawing. Everything it reads has already been validated server-side, so this
// never defends against a missing field; what it does defend against is a spec that is valid
// and still ugly, which is what the fitting at the end is for.
function build(spec, stageEl) {
  stageEl.textContent = '';
  const graph = new joint.dia.Graph({}, { cellNamespace: joint.shapes });
  const paper = new joint.dia.Paper({
    el: stageEl, model: graph, cellViewNamespace: joint.shapes,
    width: '100%', height: '100%', background: { color: 'transparent' },
    interactive: false, defaultConnectionPoint: { name: 'boundary' }
  });

  for (const n of spec.nodes) {
    const El = TYPES[n.type] || Box;
    const el = new El({ id: n.id });
    if (n.size) el.resize(n.size.w, n.size.h);
    if (n.at) el.position(n.at.x, n.at.y);
    if (n.label) el.attr('label/text', n.label);
    if (n.sublabel) el.attr('sublabel/text', n.sublabel);
    const labelled = (n.ports || []).some((p) => p.label);
    // A title centred in the body collides with port labels drawn inside it, and lifting it to
    // the top is not enough because the first port sits at a quarter height. Above the box is
    // what a schematic block does anyway.
    if (labelled) {
      el.attr('label/y', -25);
      el.attr('label/fontSize', 12.5);
      el.attr('sublabel/y', -11);
    }
    for (const p of (n.ports || [])) {
      const colour = KIND[p.kind] || MID;
      const port = { id: p.id, group: p.side || 'left', attrs: { portBody: { stroke: colour } } };
      // A port with nothing to say gets no label element at all. Handing JointJS an empty
      // string does not skip it - it draws a "-" at every unlabelled port, which is eighteen
      // stray glyphs on a six-box block diagram and a fistful of dashes hiding under the pins
      // of anything denser.
      if (p.label) port.attrs.labelText = { text: p.label, fill: colour };
      else port.label = { markup: [] };
      el.addPort(port);
    }
    graph.addCell(el);
  }

  for (const w of (spec.wires || [])) {
    const from = w.from.split(':'), to = w.to.split(':');
    const colour = KIND[w.kind] || MID;
    const link = new joint.shapes.standard.Link({
      source: { id: from[0], port: from[1] }, target: { id: to[0], port: to[1] },
      // The whole argument for JointJS over anything that draws its own lines: wires route
      // around the boxes, and where they cross they hop rather than merge into a junction.
      router: { name: 'manhattan', args: { step: 10, padding: 18 } },
      connector: { name: 'jumpover', args: { size: 5 } },
      attrs: { line: { stroke: colour, strokeWidth: 1.5,
               targetMarker: { type: 'path', d: 'M 6 -3 0 0 6 3 z', fill: colour } } }
    });
    if (w.label) link.labels([{ position: { distance: .5 },
      attrs: { text: { text: w.label, fill: colour, fontSize: 9.5, fontFamily: FONT },
               rect: { fill: SCREEN, stroke: colour, strokeWidth: .6, rx: 2 } } }]);
    graph.addCell(link);
  }

  if (spec.layout === 'dagre') {
    joint.layout.DirectedGraph.layout(graph, {
      rankDir: 'LR', nodeSep: 36, edgeSep: 22, rankSep: 70,
      marginX: 20, marginY: 16, setLinkVertices: false });
  }
  // useModelGeometry false, not true: true fits the element boxes and knows nothing about the
  // port labels hanging outside them, which silently clips every outboard label off the edge.
  // Asymmetric: the only chrome over the drawing is the title and the X, and both live in a
  // band across the top. Padding the sides to match would give away the width this layout
  // exists to win back.
  paper.transformToFitContent({
    padding: { top: 58, right: 16, bottom: 16, left: 16 },
    maxScale: 1.3, useModelGeometry: false,
    verticalAlign: 'middle', horizontalAlign: 'middle' });
  return paper;
}

// What gets kept. The panel draws green on near-black, so an SVG saved without a ground behind
// it opens as invisible green-on-white in anything that views one - a Finder preview, a browser
// tab, a README. The rect is the difference between a file and a file worth having.
function serialize(stageEl) {
  const svg = stageEl.querySelector('svg');
  if (!svg) return null;
  const copy = svg.cloneNode(true);
  const box = svg.getBoundingClientRect();
  copy.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  copy.setAttribute('width', Math.round(box.width));
  copy.setAttribute('height', Math.round(box.height));
  const ground = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
  ground.setAttribute('width', '100%');
  ground.setAttribute('height', '100%');
  ground.setAttribute('fill', SCREEN);
  copy.insertBefore(ground, copy.firstChild);
  return new XMLSerializer().serializeToString(copy);
}

const stage = document.getElementById('stage');
const dtitle = document.getElementById('dtitle');

// Any press on a picture puts it away. On the press and not the click, because that is what
// the kiosk's own tab row does and a screen with no travel has nothing else to answer with;
// and on the stage rather than the document because the stage IS the panel while it is up -
// body.drawing collapses the grid to one row and gives it the whole 800x480.
stage.addEventListener('pointerdown', (event) => {
  if (!document.body.classList.contains('photo')) return;  // a drawing keeps its corner square
  event.preventDefault();  // no synthetic click behind it, and no double-tap zoom
  window.__leave();
});
const PANEL_EVERY = 400;   // one stat() on the server; the drawing must not wait on a poll
let showing = null;

async function show(id) {
  const r = await fetch('/api/diagram/' + encodeURIComponent(id), { cache: 'no-store' });
  if (!r.ok) return;
  const found = await r.json();
  dtitle.textContent = found.title || '';
  // Cleared before it is decided, so a drawing arriving after a picture gets its button back.
  document.body.classList.remove('photo');
  // The class first: a stage that is still display:none has no size, and fitting a drawing to
  // a box of zero by zero puts it nowhere at all.
  window.__drawing(true);
  stage.innerHTML = '<div class="laying-out">drawing…</div>';
  if (found.image) {
    document.body.classList.add('photo');  // no corner button; the whole stage is the way out
    // A picture rather than a spec, so there is nothing to lay out - but it still has to be
    // decoded before we say we have painted it, or the kiosk uncovers onto an empty stage.
    const img = new Image();
    img.className = 'shot';
    img.alt = '';
    img.src = found.image;
    try { await img.decode(); } catch (e) { /* show it anyway: a slow decode beats nothing */ }
    stage.textContent = '';
    stage.appendChild(img);
    try { await fetch('/diagram/shown', { method: 'POST', body: '' }); } catch (e) {}
    return;
  }
  let svg = null;
  try {
    build(found.spec, stage);
    svg = serialize(stage);
  } catch (e) {
    console.error('[cyclops] could not draw', e);
    stage.innerHTML = '<div class="laying-out">this one would not draw</div>';
  }
  // Tell the kiosk it may uncover the panel, and hand back the picture in the same request.
  // Sent even when the drawing failed: the alternative is a panel that never uncovers and a
  // user who is told nothing.
  try {
    await fetch('/diagram/shown', {
      method: 'POST', headers: { 'Content-Type': 'image/svg+xml' }, body: svg || '' });
  } catch (e) { /* the kiosk falls back to its own timeout */ }
}

async function watch() {
  try {
    const r = await fetch('/api/panel', { cache: 'no-store' });
    const s = await r.json();
    if (s.diagram && s.diagram !== showing) {
      showing = s.diagram;
      await show(s.diagram);
    } else if (!s.diagram && showing) {
      // The kiosk took the panel back and cleared the drawing. Go quiet, and let go of the
      // graph: on a box with one browser and no tabs, the next diagram is the only thing that
      // will ever want this memory.
      showing = null;
      window.__drawing(false);
      document.body.classList.remove('photo');
      stage.textContent = '';
      dtitle.textContent = '';
    }
  } catch (e) { /* the service will come back, or the kiosk will time the page out */ }
  setTimeout(watch, PANEL_EVERY);
}
watch();
