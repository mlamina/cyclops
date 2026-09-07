/* What a companion pulls off the kiosk's own port: the picture, and the voice.
 *
 * Not app.js's job. That file owns the card in a browser - sessions, projects, media, the
 * transcript - and every byte of it arrives as JSON from this service. These two arrive from a
 * different process on a different port and never stop arriving, which is a different kind of
 * thing to own: what matters here is not what the bytes say but when to stop asking for them.
 *
 * The whole file sits inside one test for #cam, which only companion mode is sent. That is the
 * panel guard and it is also scope: these are classic scripts sharing one global lexical scope,
 * so a bare `const shot` up here would be a name app.js could collide with on its next edit.
 *
 * It must never: POST anything, or make the box do something. The switch below changes what
 * this phone does with a stream it is being sent. The box going quiet while it is on is the
 * kiosk noticing that something has taken his voice - see cyclops/kiosk.py:_sync_handover - and
 * not this page reaching across the network to mute it. The four routes that really do set the
 * box still answer to loopback and nothing else.
 */

const cam = document.getElementById('cam');

if (cam) {
  const FROM = 'http://' + location.hostname + ':' + cam.dataset.port;
  const view = document.getElementById('v-live');
  const shot = document.getElementById('camshot');
  const spk = document.getElementById('spk');
  const hint = document.getElementById('spkhint');
  const KEY = 'cyclops.speaker';

  // ---- the picture ----------------------------------------------------------------------

  // A 1x1 transparent GIF. Setting src is what actually cancels an in-flight load in every
  // engine - removeAttribute leaves the connection open in some and a broken-image glyph in
  // others - and this leaves the frame empty rather than broken.
  const BLANK = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';
  const WATCH_MS = 6000;   // a connection that opens and never paints fires no event at all
  const RETRY_MS = [2000, 4000, 8000, 15000];
  let tries = 0;
  let watchdog = null;
  let again = null;

  const drawing = () => document.body.classList.contains('drawing');
  const onLive = () => (location.hash || '#/').slice(1) === '/live' &&
                       document.visibilityState === 'visible';
  const wanted = () => onLive() && !drawing();

  const sign = (state) => {
    if (view) view.dataset.cam = state;
  };

  const stopWatching = () => {
    clearTimeout(watchdog); clearTimeout(again);
    watchdog = again = null;
  };

  const hide = () => {
    stopWatching();
    if (shot.dataset.on !== '1') return;
    shot.dataset.on = '';
    shot.src = BLANK;
  };

  const retry = () => {
    // Every attempt re-reads wanted(): a timer that fires for a screen you left must do nothing,
    // and reading the one source of truth beats keeping a second copy of it (app.js's liveGen is
    // for in-flight JSON that would paint a screen you have gone from - a stream is cancelled by
    // assigning src, which is immediate, so there is nothing left in flight to discard).
    again = setTimeout(() => { if (wanted()) watch(); }, RETRY_MS[Math.min(tries++, 3)]);
  };

  const lost = () => {
    sign('off');
    hide();
    if (wanted()) retry();
  };

  const watch = () => {
    if (!wanted() || shot.dataset.on === '1') return;
    shot.dataset.on = '1';
    sign('on');   // optimistic: it is connecting, and a black frame under the word "no picture"
    // A stream is not a file, and Safari will otherwise hand back the connection it just gave up
    // on rather than making a new one.
    shot.src = FROM + '/camera.mjpg?t=' + Date.now();
    clearTimeout(watchdog);
    // naturalWidth rather than the load event, because multipart never settles: some engines
    // fire load per part, some fire it once at the end, and one that has stalled fires nothing.
    // A decoded frame has a width whatever the events did.
    watchdog = setTimeout(() => { if (!shot.naturalWidth) lost(); }, WATCH_MS);
  };

  shot.addEventListener('error', lost);
  shot.addEventListener('load', () => { tries = 0; sign('on'); });

  // ---- the voice ------------------------------------------------------------------------

  const RATE = 24000;    // what the service sends, and the only rate it sends
  const BLOCK = 4096;    // frames per callback: ~85 ms at 48 kHz, which a phone can make in time
  const HOLD = RATE / 5; // 200 ms in hand before the first sample is let out
  const DEEP = RATE;     // ...and never more than a second, or this drifts behind the room
  const ring = new Float32Array(RATE * 2);
  let wrote = 0;         // samples ever written
  let read = 0;          // ...and ever played. A float: see the step below
  let filling = true;
  let ctx = null;
  let node = null;
  let stop = null;

  const fill = (samples) => {
    for (let i = 0; i < samples.length; i++) {
      ring[wrote % ring.length] = samples[i] / 32768;
      wrote++;
    }
    // The service streams silence between turns, so bytes never stop arriving, and the writer's
    // clock and this phone's DAC differ by tens of parts per million. Left alone the queue only
    // grows. Drop the stale middle and keep the newest: in a conversation happening in the room,
    // old audio is the wrong audio.
    if (wrote - read > DEEP) read = wrote - HOLD;
  };

  const pump = (event) => {
    const out = event.outputBuffer.getChannelData(0);
    if (filling) {
      if (wrote - read < HOLD) { out.fill(0); return; }
      filling = false;   // and re-armed below, so a late packet does not restart us mid-word
    }
    // Nothing assumes the rate we asked for was granted: Safari has a long history of handing
    // back the hardware's instead, and a different one again after a media element has played.
    // When it is honoured the step is exactly 1 and this degenerates to a copy.
    const step = RATE / ctx.sampleRate;
    for (let i = 0; i < out.length; i++) {
      if (read + step >= wrote) { out.fill(0, i); filling = true; return; }
      const at = Math.floor(read);
      const f = read - at;
      const a = ring[at % ring.length];
      const b = ring[(at + 1) % ring.length];
      out[i] = a + (b - a) * f;
      read += step;
    }
  };

  const drink = async (signal) => {
    const res = await fetch(FROM + '/voice.pcm', { signal, cache: 'no-store' });
    const reader = res.body.getReader();
    let odd = null;   // a chunk can end mid-sample, and half an int16 is a buzz
    for (;;) {
      const { value, done } = await reader.read();
      if (done) return;
      let bytes = value;
      if (odd) {
        const join = new Uint8Array(odd.length + bytes.length);
        join.set(odd); join.set(bytes, odd.length);
        bytes = join; odd = null;
      }
      const even = bytes.length - (bytes.length % 2);
      if (even < bytes.length) odd = bytes.slice(even);
      // slice() rather than a view: a joined or offset buffer is not guaranteed to be aligned
      // for Int16Array, and the throw would be one dropped packet a minute on a bad connection.
      const cut = bytes.slice(0, even);
      fill(new Int16Array(cut.buffer));
    }
  };

  const hush = () => {
    if (stop) { stop.abort(); stop = null; }
    wrote = read = 0; filling = true;
    if (ctx && ctx.suspend) ctx.suspend();
  };

  const paint = (on, why) => {
    spk.setAttribute('aria-checked', on ? 'true' : 'false');
    spk.textContent = on ? 'ON' : 'OFF';
    // What the switch says off is what it will do; what it says on is what is true. Both are
    // the words it was asked for rather than a description of the machinery behind them.
    hint.textContent = why || (on ? 'this device is the speaker'
                                  : 'use this device as speaker');
  };

  const ears = () => {
    if (!ctx) {
      const Make = window.AudioContext || window.webkitAudioContext;
      if (!Make) { paint(false, 'this device cannot play sound'); return; }
      ctx = new Make({ sampleRate: RATE });
      // ScriptProcessorNode, deprecated and knowingly: AudioWorklet is secure-context only and
      // this page is plain http on a LAN name, so ctx.audioWorklet does not exist here at all.
      // The same trade app.js makes for the clipboard - the modern API is the upgrade, not the
      // path. Held in a variable because a disconnected processor is collected mid-sentence.
      node = ctx.createScriptProcessor(BLOCK, 1, 1);
      node.onaudioprocess = pump;
      node.connect(ctx.destination);
    }
    // Must be started from inside the click, or iOS never unlocks the context.
    ctx.resume().then(() => {
      if (ctx.state !== 'running') { paint(false, 'this device would not play sound'); return; }
      stop = new AbortController();
      drink(stop.signal).catch(() => {
        if (spk.getAttribute('aria-checked') === 'true') setTimeout(ears, 2000);
      });
    }).catch(() => paint(false, 'this device would not play sound'));
  };

  spk.addEventListener('click', () => {
    const on = spk.getAttribute('aria-checked') !== 'true';
    paint(on);   // drawn before anything is attempted, exactly as the panel's own switches are
    try { localStorage.setItem(KEY, on ? '1' : ''); } catch (e) { /* private mode: this visit */ }
    if (on) ears(); else hush();
  });

  // ---- when either of them runs ----------------------------------------------------------

  const follow = () => {
    if (wanted()) watch(); else hide();
    // The voice deliberately does not follow the view. Somebody who armed the speaker and then
    // opened a photo, or put the phone in their pocket to keep both hands free, is exactly the
    // person who still wants to hear the answer.
  };

  window.addEventListener('hashchange', follow);
  document.addEventListener('visibilitychange', follow);
  window.__cam = follow;   // app.js calls this when a drawing takes the stage

  // Remembered across visits, never assumed: a context that will not start without a fresh
  // gesture paints itself back off, which is what is true.
  try {
    if (localStorage.getItem(KEY)) { paint(true); ears(); }
  } catch (e) { /* nothing remembered, which is the safe way round */ }
  follow();
}
