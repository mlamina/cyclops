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
  const BLOCK = 4096;          // frames per callback, and the slack the main thread gets
  // The cushion has to cover a whole callback, because a callback takes its 4096 frames in one
  // go rather than a sample at a time: at 24 kHz that is 170 ms leaving the buffer at once,
  // against chunks of 50 ms arriving. Measured with 200 ms in hand, the buffer sat between 30
  // and 130 ms - under one block - and ran dry about twice a second, which is exactly what chop
  // is. Two blocks plus a couple of chunks, and it stops touching the floor.
  const HOLD = Math.round(RATE * 0.45);
  // ...and never more. This used to be 1.5 s, on the grounds that anything the listener does
  // not notice is free. It was not: the kiosk holds its mic shut for however long this phone
  // might still be talking (audio.COMPANION_LAG_S), so every sample of slack here is time you
  // wait before Cyclops can hear you answer. Trimming to HOLD costs a skip of DEEP-HOLD, and
  // the service streams silence between turns, so the queue grows and is cut while nobody is
  // saying anything - which is why this can be tightened almost for free and the cushion
  // underneath it cannot.
  const DEEP = Math.round(RATE * 0.7);
  const ring = new Float32Array(RATE * 3);
  let wrote = 0;         // samples ever written
  let read = 0;          // ...and ever played. A float: see the step below
  let filling = true;
  let ctx = null;
  let node = null;
  let stop = null;
  let beat = null;
  let want = false;      // the preference, remembered across visits - not whether it is working
  let starved = 0;   // callbacks that ran out of audio mid-block; see window.__ears
  const BEAT_MS = 3000;
  const START_MS = 250;  // how long the context gets to actually start before the switch is read   // companion.LISTEN_BEAT_S; the kiosk gives the claim 8 s to be renewed

  // Saying, every few seconds, that this device is still the one being used as the speaker. The
  // panel takes his voice back the moment these stop - which is what makes a locked phone, a
  // closed laptop and a browser that died all end the same way, instead of leaving a box that
  // makes no sound and nothing on screen to explain why. An open socket cannot say this: a dead
  // peer holds one for as long as TCP takes to notice, which is a minute or never.
  const thump = () => {
    fetch(FROM + '/listening', { cache: 'no-store' }).catch(() => {});
    beat = setTimeout(thump, BEAT_MS);
  };

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
      if (read + step >= wrote) { out.fill(0, i); filling = true; starved++; return; }
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
    clearTimeout(beat); beat = null;
    if (stop) { stop.abort(); stop = null; }
    wrote = read = 0; filling = true;
    if (ctx && ctx.suspend) ctx.suspend();
    settled();
  };

  // What the switch says is what is *happening*, never what was asked for. Those two came apart
  // on a reload: the preference said on, so the switch was painted on, and the audio context sat
  // suspended behind it because a page that has not been touched yet may not make a sound. It
  // read as broken - the only cure was to turn it off and on again - and the switch was the
  // thing lying about it. So `want` is the preference and this is the truth, and the truth is
  // what is drawn: a context that is running, with a stream actually open on it.
  const settled = () => {
    const live = !!(ctx && ctx.state === 'running' && stop);
    // The switch's own word is the whole of the state now: the label beside it is a fixed
    // sentence saying what it does. Wanted-but-not-running therefore shows OFF, which is exactly
    // what is true - a browser will not make a sound on a page nobody has touched, and the tap
    // that turns it on is the tap it was waiting for.
    spk.setAttribute('aria-checked', live ? 'true' : 'false');
    spk.textContent = live ? 'ON' : 'OFF';
  };

  const ears = () => {
    if (!ctx) {
      const Make = window.AudioContext || window.webkitAudioContext;
      if (!Make) { want = false; settled(); return; }
      ctx = new Make({ sampleRate: RATE });
      // ScriptProcessorNode, deprecated and knowingly: AudioWorklet is secure-context only and
      // this page is plain http on a LAN name, so ctx.audioWorklet does not exist here at all.
      // The same trade app.js makes for the clipboard - the modern API is the upgrade, not the
      // path. Held in a variable because a disconnected processor is collected mid-sentence.
      node = ctx.createScriptProcessor(BLOCK, 1, 1);
      node.onaudioprocess = pump;
      node.connect(ctx.destination);
      ctx.onstatechange = settled;   // it can be taken away as well as granted
    }
    // resume() must be called from inside the click or iOS never unlocks the context - but its
    // promise is not the answer to "did it start". On a page nobody has touched, Chrome leaves
    // it pending indefinitely rather than rejecting, which is precisely how a switch came to sit
    // on `ON` with nothing behind it. Ask the state a moment later instead.
    ctx.resume().catch(() => {});
    setTimeout(() => {
      if (!ctx || ctx.state !== 'running') { settled(); return; }
      if (!stop) {
        stop = new AbortController();
        clearTimeout(beat);
        thump();
        drink(stop.signal)
          .catch(() => {})
          .then(() => {   // ended or failed; either way this device is no longer the speaker
            stop = null;
            settled();
            if (want) setTimeout(ears, 2000);
          });
      }
      settled();
    }, START_MS);
  };

  // What the audio path is actually doing, for when it sounds wrong. Read it from the console
  // or from a driver: `__ears()`. Cheap enough to leave in - it is five numbers.
  window.__ears = () => ({
    rate: ctx && ctx.sampleRate,          // what we got, which is not always what we asked for
    block: BLOCK,                          // the slack the main thread has between callbacks
    held: ((wrote - read) / RATE).toFixed(3),   // seconds of audio in hand
    starved,                               // blocks that ran dry: the sound of chop
    want,                                  // what was asked for...
    live: !!(ctx && ctx.state === 'running' && stop),   // ...and what is actually happening
  });

  spk.addEventListener('click', () => {
    want = !want;
    try { localStorage.setItem(KEY, want ? '1' : ''); } catch (e) { /* private mode: this visit */ }
    if (want) ears(); else hush();
    settled();
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

  // Remembered across visits, and never assumed to have worked. A browser will not let a page
  // it has not been touched on make a sound, so on a reload this attempt usually fails and the
  // switch correctly reads OFF with `tap to use this device as speaker` under it - one tap, and
  // the tap is the thing the browser was waiting for anyway.
  try {
    want = !!localStorage.getItem(KEY);
  } catch (e) { /* nothing remembered, which is the safe way round */ }
  settled();
  if (want) ears();
  follow();
}
