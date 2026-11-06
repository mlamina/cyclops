#!/usr/bin/env python3
"""hq#609: does anything native in the second check touch the network? Run under strace on the Pi.

A Python-level socket patch can't see this: onnxruntime is a C++ library and makes its own calls. So
this only loads the real check, feeds it silence and tone, waits, and exits; hq_609_native_net.sh
reads the system calls. Imports cyclops.ear for the check alone: no session, no mic is opened, no
model call. HOME is a scratch folder, so anything the library writes there can be listed afterwards.

Usage: strace ... hq_609_native_net.py <seconds to stay alive>
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

SCRATCH = tempfile.mkdtemp(prefix="qa-609-net-")
os.environ["HOME"] = SCRATCH
os.environ["XDG_CACHE_HOME"] = os.path.join(SCRATCH, ".cache")

import numpy as np  # noqa: E402

from cyclops import ear  # noqa: E402

assert "cyclops.session" not in sys.modules, "session code was loaded"

if os.environ.get(
    "QA_DISABLE_TELEMETRY"
):  # is the library's own switch enough? Not this PR's code.
    import onnxruntime

    onnxruntime.disable_telemetry_events()
    print("onnxruntime.disable_telemetry_events() called before the model loaded", flush=True)

check = ear.SpeechCheck()
tone = (np.sin(np.arange(ear.BLOCK_FRAMES) / 8.0) * 8000).astype(np.int16).tobytes()
for _ in range(200):
    check.feed(tone)
print(f"scratch home: {SCRATCH}", flush=True)
time.sleep(float(sys.argv[1]))
for root, _dirs, files in os.walk(SCRATCH):
    for name in files:
        print("wrote:", os.path.relpath(os.path.join(root, name), SCRATCH), flush=True)
print("exiting cleanly", flush=True)
