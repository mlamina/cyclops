# Third-party code served by the admin page

These three files are vendored rather than fetched from a CDN. The panel's UI layer has no
internet assumption — the Pi reaches OpenAI and nothing else — and a diagram that will not draw
because a CDN is unreachable is worse than one that costs 527 KB on the card.

They are redistributed unmodified. Update them by re-fetching the same paths at a new version and
re-running the render check in `tests/` — nothing here is patched, so there is no diff to carry.

| File | Package | Version | Licence |
|---|---|---|---|
| `joint.min.js` | [`@joint/core`](https://www.npmjs.com/package/@joint/core) | 4.3.2 | MPL-2.0 |
| `directed-graph.min.js` | [`@joint/layout-directed-graph`](https://www.npmjs.com/package/@joint/layout-directed-graph) | 4.3.0 | MPL-2.0 |
| `dagre.min.js` | [`@dagrejs/dagre`](https://www.npmjs.com/package/@dagrejs/dagre) | 3.1.1 | MIT |

`joint.min.js` and `directed-graph.min.js` carry the MPL-2.0 notice inline at the top of the file,
which is what that licence asks for. A copy of the licence is at https://mozilla.org/MPL/2.0/.

`dagre.min.js` is minified without a header, so its notice is reproduced here:

> Copyright (c) 2012-2014 Chris Pettitt
>
> Permission is hereby granted, free of charge, to any person obtaining a copy of this software
> and associated documentation files (the "Software"), to deal in the Software without
> restriction, including without limitation the rights to use, copy, modify, merge, publish,
> distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
> Software is furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all copies or
> substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
> BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
> NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
> DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
