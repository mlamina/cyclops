/* The host half of the MCP Apps handshake, and the stream that feeds it.

   Cyclops is not an MCP host and has no MCP server. What it borrows from that protocol
   (SEP-1865) is the one part worth borrowing: a renderer that is already mounted and listening,
   so what a model writes can be pushed into it while it is still being written, instead of
   arriving whole a second after the sentence ended.

   The renderer in #sketchframe is Prefab's, unmodified, and it speaks JSON-RPC over postMessage
   because that is what it expects a host to speak. Answering that handshake costs the forty lines
   below and buys its whole client - the component tree, the React reconciliation that makes one
   frame turn into the next without a remount or a flicker, its error and waiting states. Writing
   our own channel into that frame would have been about the same amount of code and none of it
   would have been tested by anybody else.

   Three messages matter and the rest are refused politely:

     ui/initialize                    - the frame asks who is hosting it. We answer once.
     ui/notifications/initialized     - it is ready. Anything we held back goes now.
     ui/notifications/tool-result     - every frame after that, with Prefab's wire JSON in
                                        params.structuredContent.

   The frames themselves do not come through the page. They arrive on an EventSource from the
   kiosk's own port, which is the process the model's code was compiled in - see
   cyclops/companion.py. The admin service this page came from never sees one. */

const frame = document.getElementById('sketchframe');

if (frame) {
  const PORT = frame.dataset.port || '8081';
  const FROM = 'http://' + location.hostname + ':' + PORT + '/sketch.sse';
  // Version of the MCP Apps protocol we answer as. The renderer warns on a mismatch and carries
  // on; it does not refuse, which is the right behaviour for a panel and the reason this is a
  // constant here rather than a negotiation.
  const PROTOCOL = '1.0';
  // The app's origin. It is served from this same host, so this is exact rather than '*' - the
  // frame is a renderer that runs code we did not write, and the frames we post into it name a
  // machine somebody could be sitting in front of.
  const ORIGIN = location.origin;

  let loaded = false;
  let ready = false;
  let pending = null;   // the last frame that arrived before the renderer said it was listening

  // The renderer runs only while there is a sketch. Left loaded with nothing to draw it
  // re-renders without end and holds a whole core of the Pi - see dashboard.html.
  function load() {
    if (loaded) return;
    loaded = true;
    frame.src = frame.dataset.src;
  }

  function unload() {
    loaded = ready = false;
    pending = null;
    frame.src = 'about:blank';
  }

  function post(message) {
    if (frame.contentWindow) frame.contentWindow.postMessage(message, ORIGIN);
  }

  function deliver(wire) {
    post({ jsonrpc: '2.0', method: 'ui/notifications/tool-result', params: { structuredContent: wire } });
  }

  window.addEventListener('message', (event) => {
    if (event.source !== frame.contentWindow) return;
    const msg = event.data;
    if (!msg || msg.jsonrpc !== '2.0' || !msg.method) return;

    if (msg.method === 'ui/initialize') {
      post({
        jsonrpc: '2.0',
        id: msg.id,
        result: {
          protocolVersion: PROTOCOL,
          hostInfo: { name: 'cyclops', version: '1' },
          // Every one of these is a thing the app may ask us to do, and we can do none of them:
          // there is no chat to send a message to, no model whose context to update, no second
          // window to open a link in. Declaring an empty set is how the SDK is told that, and it
          // is honest - a panel on a bench is not a chat client.
          hostCapabilities: {},
          hostContext: {
            theme: 'dark',
            displayMode: 'fullscreen',
            // The panel, exactly: 800x480 at --force-device-scale-factor=1. Prefab puts these in
            // its own state as $host, so a program can lay itself out against the real screen
            // instead of guessing at one.
            containerDimensions: { width: 800, height: 480 },
          },
        },
      });
      return;
    }

    if (msg.method === 'ui/notifications/initialized') {
      ready = true;
      if (pending !== undefined && pending !== null) deliver(pending);
      pending = null;
      return;
    }

    // Everything else the app can send is a request for something this host does not have. A
    // request with an id must be answered or the SDK waits on it for ever; a notification is
    // dropped. Method-not-found is the truthful answer and the SDK handles it.
    if (msg.id !== undefined && msg.id !== null) {
      post({ jsonrpc: '2.0', id: msg.id, error: { code: -32601, message: 'no such host method' } });
    }
  });

  // One connection, opened at page load and never closed. EventSource reconnects on its own, so
  // a kiosk restart or a lost socket costs the default three seconds and no code here. It is idle
  // for hours at a time, which is what it is for: the alternative is a poll, and the panel already
  // runs two.
  const feed = new EventSource(FROM);
  feed.onmessage = (event) => {
    let wire = null;
    try { wire = JSON.parse(event.data).wire; } catch (e) { return; }
    if (!wire) { if (loaded) unload(); return; }  // the sketch was put away
    load();
    if (!ready) { pending = wire; return; }
    deliver(wire);
  };
  // Deliberately silent. The stream is served by the kiosk process, which is not running when
  // somebody is looking at this page from a laptop with the panel switched off, and a console
  // full of connection errors is how a real one gets missed.
  feed.onerror = () => {};
}
